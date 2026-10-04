"""Single-process synchronous rollouts under the standard TRL train lifecycle.

This versioned backend bridge uses the public rollout hook and Trainer callbacks.
Sampling/publication implementations belong to the caller. It does not implement
an optimizer loop, persistent recovery ledger, or distributed serving runtime.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

from lmflow.agentic.contracts import TaskSpec, build_task_batch
from lmflow.agentic.trl_grpo_trainer import (
    _build_behavior_logprob_trainer_class,
    _load_trl,
    _SealedRolloutBridge,
    _validate_training_args,
)
from lmflow.utils.protocol import DataProto


class _SynchronousRolloutBridge:
    def __init__(self, tasks, args, rollout_fn, publish_policy, reward_fn, policy_prefix):
        self.tasks = list(tasks)
        self.args = args
        self.rollout_fn_impl = rollout_fn
        self.publish_policy = publish_policy
        self.reward_fn_impl = reward_fn
        self.policy_prefix = policy_prefix
        self.handles = [f"lmflow-sync-task-{i}" for i in range(len(tasks))]
        self.active = None
        self.trainer = None
        self.started = False
        self.completed_steps = 0
        self.rollout_ids = set()
        self.history = []
        self.final_publication = None

    def _publish(self, trainer, step):
        version = f"{self.policy_prefix}@{step}"
        was_training = trainer.model.training
        try:
            receipt = self.publish_policy(trainer, version)
        finally:
            trainer.model.train(was_training)
        if not isinstance(receipt, Mapping):
            raise TypeError("publish_policy must return a policy publication mapping")
        receipt = copy.deepcopy(dict(receipt))
        if receipt.get("policy_version") != version or receipt.get("global_step") != step:
            raise ValueError("publication does not identify the current trainer policy/step")
        if not receipt.get("weight_digest") or not receipt.get("source"):
            raise ValueError("publication requires weight_digest and sampling source")
        return receipt

    def rollout_func(self, prompts, trainer):
        step = trainer.state.global_step
        if step != self.completed_steps or self.active is not None:
            raise RuntimeError("fresh rollout requested before the previous update completed")
        if step >= self.args.max_steps:
            raise RuntimeError("rollout requested beyond the configured update budget")
        expected = Counter({handle: self.args.num_generations for handle in self.handles})
        if Counter(prompts) != expected:
            raise ValueError("TRL sampler must request each complete task group once per update")
        receipt = self._publish(trainer, step)
        expanded = [task for task in self.tasks for _ in range(self.args.num_generations)]
        requests = build_task_batch(expanded)
        requests.non_tensor_batch["group_ids"] = np.repeat(
            np.arange(len(self.tasks), dtype=np.int64) + step * len(self.tasks), self.args.num_generations
        )
        requests.non_tensor_batch["rollout_ids"] = np.arange(len(expanded), dtype=np.int64) + step * len(expanded)
        requests.meta_info.update(policy_version=receipt["policy_version"], policy_publication=copy.deepcopy(receipt))
        expected_identities = {
            key: requests.non_tensor_batch[key].copy() for key in ("task_ids", "group_ids", "rollout_ids")
        }
        # A sampler may enter eval/no_grad, but must not leave the training policy
        # in eval mode when TRL resumes its own scoring and training lifecycle.
        was_training = trainer.model.training
        try:
            data = self.rollout_fn_impl(requests, trainer)
        finally:
            trainer.model.train(was_training)
        if not isinstance(data, DataProto):
            raise TypeError("rollout_fn must return a sealed DataProto")
        if data.meta_info.get("policy_version") != receipt["policy_version"]:
            raise ValueError("stale rollout policy version")
        if data.meta_info.get("policy_publication") != receipt:
            raise ValueError("rollout sampling publication does not match the published policy")
        for key in ("task_ids", "group_ids", "rollout_ids"):
            if not np.array_equal(data.non_tensor_batch.get(key), expected_identities[key]):
                raise ValueError(f"rollout {key} do not match the dispatched requests")
        if self.reward_fn_impl is not None:
            data.batch["rewards"] = self.reward_fn_impl(data)
        bridge = _SealedRolloutBridge(data)
        _validate_training_args(self.args, bridge, max_steps=self.args.max_steps)
        ids = set(data.non_tensor_batch["rollout_ids"])
        if ids & self.rollout_ids:
            raise ValueError("rollout identity was already consumed")
        self.rollout_ids.update(ids)
        self.active = bridge
        # Each task has one complete group, in the exact dispatched order.
        translated = [bridge._group_handles[self.handles.index(handle)] for handle in prompts]
        result = bridge.rollout_func(translated, trainer)
        trainer.lmflow_logprob_provenance = copy.deepcopy(bridge.logprob_provenance)
        self.history.append(
            dict(
                global_step=step,
                publication=receipt,
                rollout_ids=data.non_tensor_batch["rollout_ids"].tolist(),
                logprob_provenance=copy.deepcopy(bridge.logprob_provenance),
                status="rollout_sealed",
            )
        )
        return result

    def reward_func(self, **kwargs):
        if self.active is None:
            raise RuntimeError("reward requested without a sealed rollout batch")
        return self.active.reward_func(**kwargs)

    def callback(self):
        from transformers import TrainerCallback

        bridge = self

        class Lifecycle(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if bridge.started or state.global_step != 0:
                    raise RuntimeError(
                        "synchronous bridge supports one fresh train() call; resume requires a recovery protocol"
                    )
                bridge.started = True

            def on_step_end(self, args, state, control, **kwargs):
                if bridge.trainer.accelerator.optimizer_step_was_skipped:
                    raise RuntimeError("optimizer step was skipped; policy publication is forbidden")
                if bridge.active is None or not bridge.active._reward_consumed:
                    raise RuntimeError("optimizer update completed without a consumed sealed batch")
                if state.global_step != bridge.completed_steps + 1:
                    raise RuntimeError("non-contiguous synchronous optimizer update")
                bridge.history[-1]["status"] = "updated"
                bridge.completed_steps = state.global_step
                bridge.active = None

            def on_train_end(self, args, state, control, **kwargs):
                if state.global_step != args.max_steps or bridge.active is not None:
                    raise RuntimeError("synchronous training ended before all requested updates completed")
                bridge.final_publication = bridge._publish(bridge.trainer, state.global_step)

        return Lifecycle()


def build_synchronous_trl_grpo_trainer(
    model: Any,
    processing_class: Any,
    args: Any,
    tasks: Sequence[TaskSpec],
    *,
    rollout_fn: Callable,
    publish_policy: Callable,
    policy_prefix: str,
    old_logprobs_source: str,
    reward_fn: Callable | None = None,
    peft_config: Any = None,
    callbacks: list[Any] | None = None,
):
    """Build one native Trainer for fresh synchronous rollout/update cycles.

    ``rollout_fn(requests, trainer)`` returns actual sampled tokens/probabilities
    in the existing sealed DataProto contract. ``publish_policy(trainer, version)``
    publishes current weights, returning policy_version/global_step/weight_digest/
    source. The sampler must confirm that receipt in ``policy_publication``.
    An in-process sampler can use the live model; an external sampler must load
    the publication before returning its receipt. This is backend-internal and
    version-locked, with the same objective support matrix as the one-step bridge.
    """
    dataset_class, config_class, trainer_base = _load_trl()
    if not isinstance(args, config_class):
        raise TypeError("args must be a TRL 1.9.2 GRPOConfig")
    if old_logprobs_source != "behavior":
        raise ValueError("old_logprobs_source must explicitly select 'behavior'")
    if not policy_prefix or not isinstance(policy_prefix, str):
        raise ValueError("policy_prefix must be a non-empty string")
    tasks = list(tasks)
    if not tasks or any(not isinstance(task, TaskSpec) for task in tasks):
        raise ValueError("tasks must contain TaskSpec values")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")
    if not isinstance(args.max_steps, int) or args.max_steps < 1:
        raise ValueError("max_steps must be a positive update count")
    shape = SimpleNamespace(
        num_generations=args.num_generations,
        batch_size=len(tasks) * args.num_generations,
        max_prompt_length=0,
        max_completion_length=1,
    )
    _validate_training_args(args, shape, max_steps=args.max_steps)
    bridge = _SynchronousRolloutBridge(tasks, args, rollout_fn, publish_policy, reward_fn, policy_prefix)
    trainer_class = _build_behavior_logprob_trainer_class(trainer_base)
    trainer = trainer_class(
        model=model,
        processing_class=processing_class,
        args=args,
        train_dataset=dataset_class.from_dict({"prompt": bridge.handles}),
        rollout_func=bridge.rollout_func,
        reward_funcs=bridge.reward_func,
        peft_config=peft_config,
        callbacks=[bridge.callback(), *(callbacks or [])],
    )
    bridge.trainer = trainer
    trainer.lmflow_sync_bridge = bridge
    return trainer
