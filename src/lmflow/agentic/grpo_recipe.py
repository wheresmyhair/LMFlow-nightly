"""Minimal orchestration for a rewarded GRPO batch."""

from typing import Protocol

import torch

from lmflow.agentic.algorithms import compute_group_advantages
from lmflow.utils.protocol import DataProto


class _PolicyTrainer(Protocol):
    def train_step(self, data: DataProto) -> dict[str, float]: ...


def _population_stats(values: torch.Tensor) -> tuple[float, float]:
    values = values.detach().float()
    return values.mean().item(), values.std(unbiased=False).item()


def run_grpo_step(
    trainer: _PolicyTrainer,
    data: DataProto,
    *,
    scale_rewards: bool = True,
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """Compute group advantages, apply one policy update, and name step metrics."""
    advantages = compute_group_advantages(
        data,
        scale_rewards=scale_rewards,
        epsilon=epsilon,
    )
    data.batch["advantages"] = advantages

    trainer_metrics = trainer.train_step(data)
    metrics = {(name if "/" in name else f"train/{name}"): float(value) for name, value in trainer_metrics.items()}
    if "train/selected_tokens" in metrics:
        metrics["train/tokens"] = metrics.pop("train/selected_tokens")

    rewards = data.batch["rewards"]
    reward_mean, reward_std = _population_stats(rewards)
    advantage_mean, advantage_std = _population_stats(advantages)
    metrics.update(
        {
            "train/advantage_mean": advantage_mean,
            "train/advantage_std": advantage_std,
            "reward/total_mean": reward_mean,
            "reward/total_std": reward_std,
            "rollout/trajectories": float(rewards.shape[0]),
            "rollout/groups": float(len(set(data.non_tensor_batch["group_ids"]))),
        }
    )
    return metrics


__all__ = ["run_grpo_step"]
