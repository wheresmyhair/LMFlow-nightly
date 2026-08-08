"""Minimal causal-LM policy update helpers."""

import torch
import torch.nn.functional as F

from lmflow.agentic.algorithms import grpo_policy_loss
from lmflow.utils.protocol import DataProto


def _require_tensor(data: DataProto, key: str) -> torch.Tensor:
    if data.batch is None or key not in data.batch:
        raise KeyError(f"policy update requires DataProto.batch[{key!r}]")
    return data.batch[key]


def causal_token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Gather causal-LM log-probs aligned with their target token positions."""
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (batch, sequence, vocabulary), got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, sequence), got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"logits batch/sequence shape {tuple(logits.shape[:2])} does not match input_ids "
            f"shape {tuple(input_ids.shape)}"
        )
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")

    next_token_log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    target_ids = input_ids[:, 1:].unsqueeze(-1)
    selected_log_probs = next_token_log_probs.gather(dim=-1, index=target_ids).squeeze(-1)
    first_token_log_probs = selected_log_probs.new_zeros((input_ids.shape[0], 1))
    return torch.cat((first_token_log_probs, selected_log_probs), dim=1)


def grpo_loss_from_model(
    model: torch.nn.Module,
    data: DataProto,
    *,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Run a causal LM and connect its aligned token log-probs to GRPO."""
    input_ids = _require_tensor(data, "input_ids")
    attention_mask = _require_tensor(data, "attention_mask")
    loss_mask = _require_tensor(data, "loss_mask")

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape (batch, sequence), got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )
    if loss_mask.shape != input_ids.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} does not match input_ids shape {tuple(input_ids.shape)}"
        )
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")
    if torch.any(loss_mask[:, 0] != 0):
        raise ValueError("loss_mask must exclude the first token because it has no causal log-prob")
    if torch.any((loss_mask > 0) & (attention_mask == 0)):
        raise ValueError("loss_mask must exclude tokens hidden by attention_mask")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if not hasattr(outputs, "logits"):
        raise TypeError("causal LM output must expose a logits attribute")
    current_log_probs = causal_token_log_probs(outputs.logits, input_ids)
    return grpo_policy_loss(data, current_log_probs, clip_epsilon=clip_epsilon)


__all__ = ["causal_token_log_probs", "grpo_loss_from_model"]
