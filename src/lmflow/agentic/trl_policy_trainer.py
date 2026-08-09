"""TRL-backed policy updates for precomputed Agentic batches."""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from lmflow.utils.protocol import DataProto

_SUPPORTED_TRL_VERSION = "1.9.2"


def _load_trl():
    try:
        installed_version = version("trl")
    except PackageNotFoundError as exc:
        raise ImportError(
            f"TRLPolicyTrainer requires trl=={_SUPPORTED_TRL_VERSION}; install the Agentic environment"
        ) from exc
    if installed_version != _SUPPORTED_TRL_VERSION:
        raise RuntimeError(f"TRLPolicyTrainer supports trl=={_SUPPORTED_TRL_VERSION}, found {installed_version}")

    from datasets import Dataset
    from trl import GRPOTrainer

    return Dataset, GRPOTrainer


def _require_tensor(data: DataProto, key: str) -> torch.Tensor:
    if data.batch is None or key not in data.batch:
        raise KeyError(f"TRL policy update requires DataProto.batch[{key!r}]")
    return data.batch[key]


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, non_blocking=True)


def _build_trl_inputs(data: DataProto, device: torch.device) -> dict[str, torch.Tensor]:
    """Map full-sequence LMFlow tensors to TRL's prompt/completion loss inputs."""
    input_ids = _require_tensor(data, "input_ids")
    attention_mask = _require_tensor(data, "attention_mask")
    loss_mask = _require_tensor(data, "loss_mask")
    advantages = _require_tensor(data, "advantages")

    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError(f"input_ids must have shape (batch, sequence>=2), got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}"
        )
    if loss_mask.shape != input_ids.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} does not match input_ids shape {tuple(input_ids.shape)}"
        )
    if advantages.ndim not in (1, 2):
        raise ValueError(f"advantages must have shape (batch,) or (batch, sequence), got {tuple(advantages.shape)}")
    if advantages.shape[0] != input_ids.shape[0]:
        raise ValueError(
            f"advantages batch size {advantages.shape[0]} does not match input_ids batch size {input_ids.shape[0]}"
        )
    if advantages.ndim == 2 and advantages.shape != input_ids.shape:
        raise ValueError(
            f"token-level advantages shape {tuple(advantages.shape)} does not match input_ids shape "
            f"{tuple(input_ids.shape)}"
        )
    if torch.any(loss_mask < 0):
        raise ValueError("loss_mask values must be non-negative")
    if torch.any(loss_mask[:, 0] != 0):
        raise ValueError("loss_mask must exclude the first token because it has no causal log-prob")
    if torch.any((loss_mask > 0) & (attention_mask == 0)):
        raise ValueError("loss_mask must exclude tokens hidden by attention_mask")
    if torch.any(loss_mask.sum(dim=-1) <= 0):
        raise ValueError("each sequence must select at least one token in loss_mask")

    trl_inputs = {
        "prompt_ids": _to_device(input_ids[:, :1], device),
        "prompt_mask": _to_device(attention_mask[:, :1], device),
        "completion_ids": _to_device(input_ids[:, 1:], device),
        "completion_mask": _to_device(attention_mask[:, 1:], device),
        # TRL multiplies completion_mask by tool_mask before applying its objective.
        # Keeping attention and optimization masks separate preserves LMFlow's
        # full-sequence context while selecting only policy-authored tokens.
        "tool_mask": _to_device(loss_mask[:, 1:], device),
        "advantages": _to_device(advantages[:, 1:] if advantages.ndim == 2 else advantages, device),
    }

    old_log_probs = data.batch.get("old_log_probs")
    if old_log_probs is not None:
        if old_log_probs.shape != input_ids.shape:
            raise ValueError(
                f"old_log_probs shape {tuple(old_log_probs.shape)} does not match input_ids shape "
                f"{tuple(input_ids.shape)}"
            )
        trl_inputs["old_per_token_logps"] = _to_device(old_log_probs[:, 1:], device)

    return trl_inputs


def _validate_grpo_config(args: Any) -> None:
    expected: Mapping[str, Any] = {
        "beta": 0.0,
        "gradient_accumulation_steps": 1,
        "importance_sampling_level": "token",
        "loss_type": "grpo",
        "temperature": 1.0,
        "top_entropy_quantile": 1.0,
        "use_adaptive_entropy": False,
        "use_vllm": False,
    }
    mismatches = [
        f"{name}={getattr(args, name, None)!r} (expected {required!r})"
        for name, required in expected.items()
        if getattr(args, name, None) != required
    ]
    if getattr(args, "entropy_coef", 0.0) != 0.0:
        mismatches.append(f"entropy_coef={args.entropy_coef!r} (expected 0.0)")
    if getattr(args, "off_policy_mask_threshold", None) is not None:
        mismatches.append(f"off_policy_mask_threshold={args.off_policy_mask_threshold!r} (expected None)")
    if getattr(args, "delta", None) is not None:
        mismatches.append(f"delta={args.delta!r} (expected None)")
    epsilon_high = getattr(args, "epsilon_high", None)
    if epsilon_high is not None and epsilon_high != getattr(args, "epsilon", None):
        mismatches.append(f"epsilon_high={epsilon_high!r} must equal epsilon={getattr(args, 'epsilon', None)!r}")
    if mismatches:
        raise ValueError("unsupported TRLPolicyTrainer v1 configuration: " + "; ".join(mismatches))


def _unused_reward(completions, **kwargs):
    """Satisfy GRPOTrainer initialization; precomputed updates never call this."""
    del kwargs
    return [0.0] * len(completions)


class TRLPolicyTrainer:
    """Run one-step GRPO updates from LMFlow ``DataProto`` batches using TRL."""

    def __init__(
        self,
        model: torch.nn.Module,
        processing_class: Any,
        args: Any,
        *,
        peft_config: Any = None,
    ) -> None:
        dataset_class, trainer_class = _load_trl()
        _validate_grpo_config(args)

        # TRL requires a dataset and reward source at construction time even when
        # generation is bypassed. These sentinels are never read by compute_loss().
        initialization_dataset = dataset_class.from_dict({"prompt": [""] * max(1, args.num_generations or 1)})
        trainer = trainer_class(
            model=model,
            reward_funcs=_unused_reward,
            args=args,
            train_dataset=initialization_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
        )
        trainer.current_gradient_accumulation_steps = 1
        trainer.create_optimizer()
        trainer.model, trainer.optimizer = trainer.accelerator.prepare(trainer.model, trainer.optimizer)
        trainer.model_wrapped = trainer.model

        self._trainer = trainer

    @property
    def model(self) -> torch.nn.Module:
        return self._trainer.model

    def unwrap_model(self) -> torch.nn.Module:
        """Return the underlying model without an Accelerate wrapper."""
        return self._trainer.accelerator.unwrap_model(self.model)

    def compute_loss(self, data: DataProto) -> torch.Tensor:
        """Compute TRL's locked GRPO loss for a precomputed LMFlow batch."""
        inputs = _build_trl_inputs(data, self._trainer.accelerator.device)
        with self._trainer.compute_loss_context_manager():
            return self._trainer.compute_loss(self.model, inputs)

    def train_step(self, data: DataProto) -> dict[str, float]:
        """Apply one optimizer update and return process-local step metrics."""
        self.model.train()
        self._trainer.optimizer.zero_grad()
        loss = self.compute_loss(data)
        self._trainer.accelerator.backward(loss)

        max_grad_norm = self._trainer.args.max_grad_norm
        if max_grad_norm is not None and max_grad_norm > 0:
            self._trainer.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self._trainer.optimizer.step()
        self._trainer.optimizer.zero_grad()

        selected_tokens = (_require_tensor(data, "loss_mask") > 0).sum().detach().item()
        return {
            "loss": loss.detach().float().item(),
            "selected_tokens": float(selected_tokens),
        }


__all__ = ["TRLPolicyTrainer"]
