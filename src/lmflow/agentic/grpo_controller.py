"""Minimal same-process controller for one synchronous GRPO update."""

import time
from collections.abc import Callable, Hashable, Iterable
from typing import Protocol

import numpy as np
import torch

from lmflow.agentic.contracts import TaskSpec, build_task_batch
from lmflow.agentic.grpo_recipe import run_grpo_step
from lmflow.agentic.rollout_groups import RolloutGroupAssembler
from lmflow.utils.protocol import DataProto


class _RolloutFunction(Protocol):
    def __call__(self, requests: DataProto) -> DataProto: ...


class _RewardFunction(Protocol):
    def __call__(self, rollouts: DataProto) -> torch.Tensor: ...


class _PolicyTrainer(Protocol):
    def train_step(self, data: DataProto) -> dict[str, float]: ...


def _build_rollout_requests(
    tasks: list[TaskSpec],
    *,
    group_size: int,
    policy_version: Hashable,
) -> DataProto:
    expanded_tasks = [task for task in tasks for _ in range(group_size)]
    requests = build_task_batch(expanded_tasks)
    requests.non_tensor_batch.update(
        {
            "group_ids": np.repeat(np.arange(len(tasks), dtype=np.int64), group_size),
            "rollout_ids": np.arange(len(expanded_tasks), dtype=np.int64),
        }
    )
    requests.meta_info["policy_version"] = policy_version
    return requests


def _validate_task_identities(requests: DataProto, rollouts: DataProto) -> None:
    try:
        rollout_ids = rollouts.non_tensor_batch["rollout_ids"]
        task_ids = rollouts.non_tensor_batch["task_ids"]
    except KeyError as exc:
        raise KeyError("rollout results require non_tensor_batch['rollout_ids'] and ['task_ids']") from exc

    if rollout_ids.ndim != 1 or task_ids.ndim != 1:
        raise ValueError("rollout_ids and task_ids must both have shape (batch,)")

    expected_by_rollout_id = dict(
        zip(
            requests.non_tensor_batch["rollout_ids"],
            requests.non_tensor_batch["task_ids"],
        )
    )
    for index, (rollout_id, task_id) in enumerate(zip(rollout_ids, task_ids)):
        if not isinstance(rollout_id, Hashable):
            raise TypeError(f"rollout_ids[{index}] must be hashable, got {type(rollout_id).__name__}")
        if rollout_id not in expected_by_rollout_id:
            raise KeyError(f"rollout_id {rollout_id!r} was not dispatched by this step")

        expected_task_id = expected_by_rollout_id[rollout_id]
        if task_id != expected_task_id:
            raise ValueError(
                f"task_id mismatch for rollout_id {rollout_id!r}: expected {expected_task_id!r}, got {task_id!r}"
            )


def _attach_rewards(data: DataProto, reward_fn: _RewardFunction) -> None:
    if data.batch is None:
        raise ValueError("rollout_fn must return token tensors before rewards can be attached")

    rewards = reward_fn(data)
    if not isinstance(rewards, torch.Tensor):
        raise TypeError(f"reward_fn must return a torch.Tensor, got {type(rewards).__name__}")
    if rewards.ndim != 1 or rewards.shape[0] != len(data):
        raise ValueError(f"rewards must have shape ({len(data)},), got {tuple(rewards.shape)}")
    if torch.is_complex(rewards):
        raise TypeError("rewards must be real-valued")

    rewards = rewards.detach()
    if not torch.is_floating_point(rewards):
        rewards = rewards.float()
    if not torch.all(torch.isfinite(rewards)):
        raise ValueError("rewards must contain only finite values")
    data.batch["rewards"] = rewards


def run_synchronous_grpo_step(
    trainer: _PolicyTrainer,
    tasks: Iterable[TaskSpec],
    *,
    rollout_fn: _RolloutFunction,
    reward_fn: _RewardFunction,
    group_size: int,
    policy_version: Hashable,
    group_timeout_seconds: float = 300.0,
    scale_rewards: bool = True,
    epsilon: float = 1e-4,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, float]:
    """Run one task-to-rollout-to-reward-to-policy-update reference step."""
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)):
        raise TypeError(f"group_size must be an integer, got {type(group_size).__name__}")
    group_size = int(group_size)
    if group_size < 2:
        raise ValueError(f"group_size must be at least 2 for GRPO, got {group_size}")

    task_list = list(tasks)
    if not task_list:
        raise ValueError("tasks must contain at least one TaskSpec")

    requests = _build_rollout_requests(
        task_list,
        group_size=group_size,
        policy_version=policy_version,
    )
    assembler = RolloutGroupAssembler(
        policy_version=policy_version,
        timeout_seconds=group_timeout_seconds,
        clock=clock,
    )
    started_at = float(clock())
    for group_id in range(len(task_list)):
        start = group_id * group_size
        assembler.register_group(
            group_id,
            range(start, start + group_size),
            started_at=started_at,
        )

    rollouts = rollout_fn(requests)
    if not isinstance(rollouts, DataProto):
        raise TypeError(f"rollout_fn must return a DataProto, got {type(rollouts).__name__}")

    received_at = float(clock())
    expired_group_ids = assembler.expire(now=received_at)
    if expired_group_ids:
        raise TimeoutError(
            f"rollout collection exceeded {group_timeout_seconds} seconds for groups {expired_group_ids}"
        )

    _validate_task_identities(requests, rollouts)
    completed_group_ids = assembler.add(rollouts, received_at=received_at)
    if len(completed_group_ids) != len(task_list):
        raise ValueError(
            f"synchronous rollout returned incomplete groups: completed {len(completed_group_ids)} of {len(task_list)}"
        )

    ready_batch = assembler.pop_ready()
    if ready_batch is None:
        raise RuntimeError("all rollout groups completed but no ready batch was produced")

    _attach_rewards(ready_batch, reward_fn)
    metrics = run_grpo_step(
        trainer,
        ready_batch,
        scale_rewards=scale_rewards,
        epsilon=epsilon,
    )
    metrics.update(assembler.metrics())
    return metrics


__all__ = ["run_synchronous_grpo_step"]
