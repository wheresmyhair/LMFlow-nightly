"""TRL GRPO lifecycle bridge for sealed token-native rollout groups."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch

from lmflow.utils.protocol import DataProto

_SUPPORTED_TRL_VERSION = "1.9.2"


def _load_trl():
    try:
        installed_version = version("trl")
    except PackageNotFoundError as exc:
        raise ImportError(
            f"the sealed GRPO bridge requires trl=={_SUPPORTED_TRL_VERSION}; install the Agentic environment"
        ) from exc
    if installed_version != _SUPPORTED_TRL_VERSION:
        raise RuntimeError(f"the sealed GRPO bridge supports trl=={_SUPPORTED_TRL_VERSION}, found {installed_version}")

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    return Dataset, GRPOConfig, GRPOTrainer


def _require_tensor(data: DataProto, key: str) -> torch.Tensor:
    if data.batch is None or key not in data.batch:
        raise KeyError(f"sealed GRPO rollouts require DataProto.batch[{key!r}]")
    return data.batch[key]


def _require_non_tensor(data: DataProto, key: str) -> np.ndarray:
    try:
        value = data.non_tensor_batch[key]
    except KeyError as exc:
        raise KeyError(f"sealed GRPO rollouts require DataProto.non_tensor_batch[{key!r}]") from exc
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.shape[0] != len(data):
        raise ValueError(f"{key} must be a one-dimensional NumPy array with {len(data)} rows")
    return value


def _validate_binary_tensor(value: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must contain only 0 or 1")


class _SealedRolloutBridge:
    """Single-use TRL hooks backed by one immutable, complete rollout batch."""

    def __init__(self, data: DataProto) -> None:
        if not isinstance(data, DataProto):
            raise TypeError("sealed rollouts must be a DataProto")
        if len(data) == 0:
            raise ValueError("sealed rollout batch must not be empty")

        input_ids = _require_tensor(data, "input_ids")
        attention_mask = _require_tensor(data, "attention_mask")
        loss_mask = _require_tensor(data, "loss_mask")
        old_log_probs = _require_tensor(data, "old_log_probs")
        rewards = _require_tensor(data, "rewards")
        prompt_lengths = _require_tensor(data, "prompt_lengths")
        batch_size = len(data)

        if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
            raise ValueError(f"input_ids must have shape ({batch_size}, sequence), got {tuple(input_ids.shape)}")
        for name, value in {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "old_log_probs": old_log_probs,
        }.items():
            if value.shape != input_ids.shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match input_ids {tuple(input_ids.shape)}")
        if rewards.shape != (batch_size,):
            raise ValueError(f"rewards must have shape ({batch_size},), got {tuple(rewards.shape)}")
        if prompt_lengths.shape != (batch_size,):
            raise ValueError(f"prompt_lengths must have shape ({batch_size},), got {tuple(prompt_lengths.shape)}")
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must use torch.long, got {input_ids.dtype}")
        if prompt_lengths.dtype != torch.long:
            raise TypeError(f"prompt_lengths must use torch.long, got {prompt_lengths.dtype}")
        _validate_binary_tensor(attention_mask, name="attention_mask")
        _validate_binary_tensor(loss_mask, name="loss_mask")
        if not torch.isfinite(old_log_probs).all():
            raise ValueError("old_log_probs must contain only finite values")
        if not torch.isfinite(rewards).all():
            raise ValueError("rewards must contain only finite values")
        if torch.any(input_ids < 0):
            raise ValueError("input_ids must be non-negative")
        if torch.any((loss_mask != 0) & (attention_mask == 0)):
            raise ValueError("loss_mask must exclude padding tokens")

        task_ids = _require_non_tensor(data, "task_ids")
        group_ids = _require_non_tensor(data, "group_ids")
        rollout_ids = _require_non_tensor(data, "rollout_ids")
        try:
            policy_version = data.meta_info["policy_version"]
        except KeyError as exc:
            raise KeyError("sealed GRPO rollouts require DataProto.meta_info['policy_version']") from exc
        if isinstance(policy_version, bool) or not isinstance(policy_version, str | int):
            raise TypeError("policy_version must be a string or integer")
        try:
            behavior_provenance = data.meta_info["logprob_provenance"]["behavior"]
        except (KeyError, TypeError) as exc:
            raise KeyError("sealed GRPO rollouts require meta_info['logprob_provenance']['behavior']") from exc
        if not isinstance(behavior_provenance, Mapping):
            raise TypeError("behavior log-prob provenance must be a mapping")
        if behavior_provenance.get("policy_version") != policy_version:
            raise ValueError("behavior log-prob provenance policy_version does not match the sealed batch")
        if not isinstance(behavior_provenance.get("source"), str) or not behavior_provenance["source"]:
            raise ValueError("behavior log-prob provenance source must be a non-empty string")

        groups: dict[Hashable, list[int]] = defaultdict(list)
        seen_rollout_ids: set[Hashable] = set()
        for index, (task_id, group_id, rollout_id) in enumerate(zip(task_ids, group_ids, rollout_ids, strict=True)):
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"task_ids[{index}] must be a non-empty string")
            if not isinstance(group_id, Hashable):
                raise TypeError(f"group_ids[{index}] must be hashable")
            if not isinstance(rollout_id, Hashable):
                raise TypeError(f"rollout_ids[{index}] must be hashable")
            if rollout_id in seen_rollout_ids:
                raise ValueError(f"rollout_ids must be unique; duplicate {rollout_id!r}")
            seen_rollout_ids.add(rollout_id)
            groups[group_id].append(index)

        group_sizes = {len(indices) for indices in groups.values()}
        if len(group_sizes) != 1:
            raise ValueError(f"all sealed rollout groups must have the same size, got {sorted(group_sizes)}")
        self.num_generations = group_sizes.pop()
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two rollouts per group")

        rows: list[dict[str, Any]] = []
        self.max_prompt_length = 0
        self.max_completion_length = 0
        for index in range(batch_size):
            active_length = int(attention_mask[index].sum().item())
            if active_length < 2:
                raise ValueError(f"row {index} must contain at least two active tokens")
            expected_attention = torch.zeros_like(attention_mask[index])
            expected_attention[:active_length] = 1
            if not torch.equal(attention_mask[index], expected_attention):
                raise ValueError(f"row {index} attention_mask must use contiguous right padding")
            prompt_length = int(prompt_lengths[index].item())
            if not 0 < prompt_length < active_length:
                raise ValueError(f"prompt_lengths[{index}] must be between 1 and active length {active_length - 1}")
            if torch.any(loss_mask[index, :prompt_length] != 0):
                raise ValueError(f"row {index} loss_mask must exclude every prompt token")
            completion_mask = loss_mask[index, prompt_length:active_length]
            if not torch.any(completion_mask > 0):
                raise ValueError(f"row {index} must contain at least one policy completion token")
            self.max_prompt_length = max(self.max_prompt_length, prompt_length)
            self.max_completion_length = max(self.max_completion_length, active_length - prompt_length)
            rows.append(
                {
                    "prompt_ids": input_ids[index, :prompt_length].detach().cpu().tolist(),
                    "completion_ids": input_ids[index, prompt_length:active_length].detach().cpu().tolist(),
                    "logprobs": old_log_probs[index, prompt_length:active_length].detach().cpu().tolist(),
                    "env_mask": completion_mask.detach().cpu().tolist(),
                    "audited_reward": float(rewards[index].item()),
                    "task_id": task_ids[index],
                    "group_id": group_ids[index],
                    "rollout_id": rollout_ids[index],
                    "policy_version": policy_version,
                }
            )

        self._group_rows: dict[str, list[dict[str, Any]]] = {}
        self._group_handles: list[str] = []
        for group_index, indices in enumerate(groups.values()):
            group_rows = [rows[index] for index in indices]
            if len({row["task_id"] for row in group_rows}) != 1:
                raise ValueError("every sealed rollout group must contain exactly one task_id")
            handle = f"lmflow-sealed-group-{group_index}"
            self._group_handles.append(handle)
            self._group_rows[handle] = group_rows
        self._rollout_consumed = False
        self._reward_consumed = False
        self._last_rollout_rows: list[dict[str, Any]] | None = None
        self.logprob_provenance = {
            "behavior": copy.deepcopy(dict(behavior_provenance)),
            "trainer_old": {
                "source": "behavior",
                "input_field": "DataProto.batch['old_log_probs']",
                "trl_field": "old_per_token_logps",
                "compatibility_contract": "trl==1.9.2:post-generate-score-injection",
            },
            "reference": {
                "enabled": False,
                "source": None,
                "reason": "beta=0",
            },
        }

    @property
    def batch_size(self) -> int:
        return len(self._group_handles) * self.num_generations

    def dataset_dict(self) -> dict[str, list[str]]:
        return {"prompt": self._group_handles.copy()}

    def rollout_func(self, prompts: list[str], trainer: Any) -> dict[str, Any]:
        del trainer
        if self._rollout_consumed:
            raise RuntimeError("sealed rollout batch has already been consumed")
        if not isinstance(prompts, list) or any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("TRL rollout prompts must be a list of sealed group handles")
        expected = Counter({handle: self.num_generations for handle in self._group_handles})
        if Counter(prompts) != expected:
            raise ValueError("TRL sampler did not request every sealed rollout group exactly num_generations times")

        positions: defaultdict[str, int] = defaultdict(int)
        ordered_rows = []
        for handle in prompts:
            position = positions[handle]
            positions[handle] += 1
            ordered_rows.append(self._group_rows[handle][position])
        self._last_rollout_rows = ordered_rows
        self._rollout_consumed = True
        return {
            "prompt_ids": [row["prompt_ids"] for row in ordered_rows],
            "completion_ids": [row["completion_ids"] for row in ordered_rows],
            "logprobs": [row["logprobs"] for row in ordered_rows],
            "env_mask": [row["env_mask"] for row in ordered_rows],
            "audited_reward": [row["audited_reward"] for row in ordered_rows],
            "task_id": [row["task_id"] for row in ordered_rows],
            "group_id": [row["group_id"] for row in ordered_rows],
            "rollout_id": [row["rollout_id"] for row in ordered_rows],
            "policy_version": [row["policy_version"] for row in ordered_rows],
        }

    def reward_func(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Sequence[Sequence[int]],
        audited_reward: Sequence[float],
        task_id: Sequence[str],
        group_id: Sequence[Hashable],
        rollout_id: Sequence[Hashable],
        policy_version: Sequence[str | int],
        **kwargs,
    ) -> list[float]:
        del prompts, completions, kwargs
        if self._reward_consumed:
            raise RuntimeError("sealed rollout rewards have already been consumed")
        if self._last_rollout_rows is None:
            raise RuntimeError("sealed rewards cannot be consumed before rollout data")
        rows = self._last_rollout_rows
        fields = {
            "completion_ids": completion_ids,
            "audited_reward": audited_reward,
            "task_id": task_id,
            "group_id": group_id,
            "rollout_id": rollout_id,
            "policy_version": policy_version,
        }
        if any(len(values) != len(rows) for values in fields.values()):
            raise ValueError("TRL reward inputs do not align with the sealed rollout batch")
        for index, row in enumerate(rows):
            expected = {
                "completion_ids": row["completion_ids"],
                "audited_reward": row["audited_reward"],
                "task_id": row["task_id"],
                "group_id": row["group_id"],
                "rollout_id": row["rollout_id"],
                "policy_version": row["policy_version"],
            }
            for name, values in fields.items():
                actual = list(values[index]) if name == "completion_ids" else values[index]
                if actual != expected[name]:
                    raise ValueError(f"sealed rollout {name} mismatch at row {index}: {actual!r} != {expected[name]!r}")
        self._reward_consumed = True
        return [float(value) for value in audited_reward]


def _build_behavior_logprob_trainer_class(base_class):
    class _BehaviorLogprobGRPOTrainer(base_class):
        """TRL 1.9.2 compatibility shim that selects sampled log-probs as PPO old log-probs."""

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            sampling = output.get("sampling_per_token_logps")
            if sampling is None:
                raise RuntimeError("TRL rollout_func did not preserve sampled token log-probs")
            if "old_per_token_logps" in output:
                raise RuntimeError("TRL unexpectedly produced old_per_token_logps before behavior-logprob injection")
            completion_ids = output.get("completion_ids")
            if not isinstance(completion_ids, torch.Tensor) or sampling.shape != completion_ids.shape:
                raise RuntimeError("sampled log-probs do not align with TRL completion IDs")
            if not torch.isfinite(sampling).all():
                raise RuntimeError("sampled log-probs must contain only finite values")
            output["old_per_token_logps"] = sampling.detach().clone()
            self.lmflow_old_logprobs_source = "behavior"
            return output

    _BehaviorLogprobGRPOTrainer.__name__ = "BehaviorLogprobGRPOTrainer"
    return _BehaviorLogprobGRPOTrainer


def _validate_training_args(args: Any, bridge: _SealedRolloutBridge) -> None:
    expected: Mapping[str, Any] = {
        "beta": 0.0,
        "sync_ref_model": False,
        "use_bias_correction_kl": False,
        "importance_sampling_level": "token",
        "loss_type": "grpo",
        "multi_objective_aggregation": "sum_then_normalize",
        "num_iterations": 1,
        "scale_rewards": "group",
        "use_liger_kernel": False,
        "use_vllm": False,
        "vllm_importance_sampling_correction": False,
        "mask_truncated_completions": False,
        "shuffle_dataset": False,
        "temperature": 1.0,
        "top_entropy_quantile": 1.0,
        "gradient_checkpointing": True,
        "use_cache": False,
    }
    mismatches = [
        f"{name}={getattr(args, name, None)!r} (expected {required!r})"
        for name, required in expected.items()
        if getattr(args, name, None) != required
    ]
    if getattr(args, "max_steps", None) != 1:
        mismatches.append(f"max_steps={getattr(args, 'max_steps', None)!r} (expected 1)")
    if getattr(args, "world_size", None) != 1:
        mismatches.append(f"world_size={getattr(args, 'world_size', None)!r} (expected 1)")
    if getattr(args, "per_device_train_batch_size", None) != 1:
        mismatches.append(
            f"per_device_train_batch_size={getattr(args, 'per_device_train_batch_size', None)!r} (expected 1)"
        )
    if getattr(args, "num_generations", None) != bridge.num_generations:
        mismatches.append(
            f"num_generations={getattr(args, 'num_generations', None)!r} "
            f"(expected sealed group size {bridge.num_generations})"
        )
    if getattr(args, "generation_batch_size", None) != bridge.batch_size:
        mismatches.append(
            f"generation_batch_size={getattr(args, 'generation_batch_size', None)!r} "
            f"(expected sealed batch size {bridge.batch_size})"
        )
    if getattr(args, "steps_per_generation", None) != getattr(args, "gradient_accumulation_steps", None):
        mismatches.append(
            f"steps_per_generation={getattr(args, 'steps_per_generation', None)!r} must equal "
            f"gradient_accumulation_steps={getattr(args, 'gradient_accumulation_steps', None)!r}"
        )
    if getattr(args, "gradient_accumulation_steps", None) != bridge.batch_size:
        mismatches.append(
            f"gradient_accumulation_steps={getattr(args, 'gradient_accumulation_steps', None)!r} "
            f"(expected sealed batch size {bridge.batch_size})"
        )
    max_prompt_length = getattr(args, "max_prompt_length", None)
    if max_prompt_length is not None and max_prompt_length < bridge.max_prompt_length:
        mismatches.append(
            f"max_prompt_length={max_prompt_length!r} is smaller than sealed maximum {bridge.max_prompt_length}"
        )
    max_completion_length = getattr(args, "max_completion_length", None)
    if max_completion_length is None or max_completion_length < bridge.max_completion_length:
        mismatches.append(
            f"max_completion_length={max_completion_length!r} is smaller than sealed maximum "
            f"{bridge.max_completion_length}"
        )
    if getattr(args, "reward_weights", None) is not None:
        mismatches.append("reward_weights must be None for one audited reward")
    if getattr(args, "off_policy_mask_threshold", None) is not None:
        mismatches.append("off_policy_mask_threshold must be None")
    if getattr(args, "entropy_coef", 0.0) != 0.0 or getattr(args, "use_adaptive_entropy", False):
        mismatches.append("entropy control must be disabled")
    if getattr(args, "delta", None) is not None:
        mismatches.append("delta must be None")
    if getattr(args, "epsilon", None) != 0.2:
        mismatches.append(f"epsilon={getattr(args, 'epsilon', None)!r} (expected 0.2)")
    epsilon_high = getattr(args, "epsilon_high", None)
    if epsilon_high is not None and epsilon_high != getattr(args, "epsilon", None):
        mismatches.append("epsilon_high must be None or equal epsilon")
    if mismatches:
        raise ValueError("unsupported sealed TRL GRPO configuration: " + "; ".join(mismatches))


def build_one_step_trl_grpo_trainer(
    model: torch.nn.Module,
    processing_class: Any,
    args: Any,
    sealed_rollouts: DataProto,
    *,
    old_logprobs_source: str,
    peft_config: Any = None,
    callbacks: list[Any] | None = None,
):
    """Build a TRL 1.9.2 trainer that consumes one complete sealed rollout batch.

    The returned object is a normal ``GRPOTrainer`` and must be run through
    ``trainer.train()``. The only compatibility override promotes the sampled
    log-probs returned by TRL's public ``rollout_func`` hook to the old policy
    log-probs consumed by the GRPO objective.
    """

    if old_logprobs_source != "behavior":
        raise ValueError("old_logprobs_source must explicitly select 'behavior'")
    dataset_class, config_class, trainer_base = _load_trl()
    if not isinstance(args, config_class):
        raise TypeError(f"args must be a TRL {_SUPPORTED_TRL_VERSION} GRPOConfig")
    bridge = _SealedRolloutBridge(sealed_rollouts)
    _validate_training_args(args, bridge)
    trainer_class = _build_behavior_logprob_trainer_class(trainer_base)
    trainer = trainer_class(
        model=model,
        reward_funcs=bridge.reward_func,
        args=args,
        train_dataset=dataset_class.from_dict(bridge.dataset_dict()),
        processing_class=processing_class,
        callbacks=callbacks,
        peft_config=peft_config,
        rollout_func=bridge.rollout_func,
    )
    trainer.lmflow_sealed_rollout_bridge = bridge
    trainer.lmflow_logprob_provenance = copy.deepcopy(bridge.logprob_provenance)
    return trainer


__all__ = ["build_one_step_trl_grpo_trainer"]
