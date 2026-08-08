"""Minimal GRPO advantage and policy-objective components."""

from collections import defaultdict
from collections.abc import Hashable

import torch

from lmflow.utils.protocol import DataProto


def _require_tensor(data: DataProto, key: str) -> torch.Tensor:
    if data.batch is None or key not in data.batch:
        raise KeyError(f"GRPO requires DataProto.batch[{key!r}]")
    return data.batch[key]


@torch.no_grad()
def compute_group_advantages(
    data: DataProto,
    *,
    scale_rewards: bool = True,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Center rewards by rollout group and optionally divide by group std."""
    rewards = _require_tensor(data, "rewards")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must have shape (batch,), got {tuple(rewards.shape)}")

    try:
        group_ids = data.non_tensor_batch["group_ids"]
    except KeyError as exc:
        raise KeyError("GRPO requires DataProto.non_tensor_batch['group_ids']") from exc

    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must have shape (batch,), got {group_ids.shape}")
    if scale_rewards and epsilon <= 0:
        raise ValueError(f"epsilon must be positive when scale_rewards is enabled, got {epsilon}")

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        if not isinstance(group_id, Hashable):
            raise TypeError(f"group_ids[{index}] must be hashable, got {type(group_id).__name__}")
        groups[group_id].append(index)

    working_dtype = torch.float64 if rewards.dtype == torch.float64 else torch.float32
    working_rewards = rewards.to(dtype=working_dtype)
    advantages = torch.zeros_like(working_rewards)

    for indices in groups.values():
        index_tensor = torch.tensor(indices, device=rewards.device)
        group_rewards = working_rewards.index_select(0, index_tensor)
        centered = group_rewards - group_rewards.mean()
        if scale_rewards:
            std = group_rewards.std() if len(indices) > 1 else group_rewards.new_zeros(())
            centered = centered / (std + epsilon)
        advantages.index_copy_(0, index_tensor, centered)

    return advantages


def grpo_policy_loss(
    data: DataProto,
    current_log_probs: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Compute the token-level clipped GRPO objective with per-sequence normalization."""
    loss_mask = _require_tensor(data, "loss_mask")
    advantages = _require_tensor(data, "advantages")

    if current_log_probs.ndim != 2:
        raise ValueError(f"current_log_probs must have shape (batch, sequence), got {tuple(current_log_probs.shape)}")
    if loss_mask.shape != current_log_probs.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} does not match current_log_probs "
            f"shape {tuple(current_log_probs.shape)}"
        )
    if torch.any(loss_mask < 0):
        raise ValueError("loss_mask values must be non-negative")
    if clip_epsilon < 0:
        raise ValueError(f"clip_epsilon must be non-negative, got {clip_epsilon}")

    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    elif advantages.shape != current_log_probs.shape:
        raise ValueError(
            f"advantages must have shape (batch,) or match current_log_probs; got {tuple(advantages.shape)}"
        )

    old_log_probs = data.batch.get("old_log_probs")
    if old_log_probs is None:
        old_log_probs = current_log_probs.detach()
    elif old_log_probs.shape != current_log_probs.shape:
        raise ValueError(
            f"old_log_probs shape {tuple(old_log_probs.shape)} does not match current_log_probs "
            f"shape {tuple(current_log_probs.shape)}"
        )

    mask = loss_mask.to(device=current_log_probs.device, dtype=current_log_probs.dtype)
    token_counts = mask.sum(dim=-1)
    if torch.any(token_counts <= 0):
        raise ValueError("each sequence must select at least one token in loss_mask")

    advantages = advantages.to(device=current_log_probs.device, dtype=current_log_probs.dtype).detach()
    old_log_probs = old_log_probs.to(device=current_log_probs.device, dtype=current_log_probs.dtype).detach()
    probability_ratio = torch.exp(current_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(probability_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    per_token_objective = torch.minimum(probability_ratio * advantages, clipped_ratio * advantages)
    per_sequence_loss = -(per_token_objective * mask).sum(dim=-1) / token_counts
    return per_sequence_loss.mean()
