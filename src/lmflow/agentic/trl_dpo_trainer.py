"""TRL-backed offline DPO for LMFlow paired conversations."""

import copy
import math
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Any, Optional

from lmflow.datasets.dataset import Dataset

_SUPPORTED_TRL_VERSION = "1.9.2"
_MESSAGE_ROLES = {"user", "assistant", "tool"}


def _load_trl():
    try:
        installed_version = version("trl")
    except PackageNotFoundError as exc:
        raise ImportError(
            f"TRLDPOTrainer requires trl=={_SUPPORTED_TRL_VERSION}; install the Agentic environment"
        ) from exc
    if installed_version != _SUPPORTED_TRL_VERSION:
        raise RuntimeError(f"TRLDPOTrainer supports trl=={_SUPPORTED_TRL_VERSION}, found {installed_version}")

    from trl import DPOConfig, DPOTrainer

    return DPOConfig, DPOTrainer


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _normalize_message(message: Any, path: str) -> tuple[dict[str, Any], Optional[bool]]:
    normalized = copy.deepcopy(dict(_as_mapping(message, path)))
    role = normalized.get("role")
    if role not in _MESSAGE_ROLES:
        raise ValueError(f"{path}.role must be one of {sorted(_MESSAGE_ROLES)}, got {role!r}")

    loss = normalized.pop("loss", None)
    if loss is not None and not isinstance(loss, bool):
        raise ValueError(f"{path}.loss must be a boolean or null")
    if role != "assistant" and loss is not None:
        raise ValueError(f"{path}.loss is only valid on assistant messages")
    return normalized, loss


def _normalize_conversation(conversation: Any, path: str) -> dict[str, Any]:
    conversation = _as_mapping(conversation, path)
    system = conversation.get("system")
    if system is not None and not isinstance(system, str):
        raise ValueError(f"{path}.system must be a string or null")

    tools = conversation.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise ValueError(f"{path}.tools must be a list or null")

    messages = conversation.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{path}.messages must be a list")
    if len(messages) < 2:
        raise ValueError(f"{path}.messages must contain context followed by one assistant completion")

    normalized_messages = [
        _normalize_message(message, f"{path}.messages[{index}]") for index, message in enumerate(messages)
    ]
    if normalized_messages[0][0]["role"] != "user":
        raise ValueError(f"{path}.messages must start with a user message")
    if normalized_messages[-1][0]["role"] != "assistant":
        raise ValueError(f"{path}.messages must end with an assistant completion")
    if normalized_messages[-2][0]["role"] == "assistant":
        raise ValueError(f"{path}.messages must provide user or tool context before the final assistant completion")
    if normalized_messages[-1][1] is False:
        raise ValueError(f"{path} final assistant completion cannot set loss=false for DPO")

    return {
        "system": system,
        "tools": copy.deepcopy(tools),
        "messages": [message for message, _ in normalized_messages],
    }


def _paired_conversation_to_preference(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one LMFlow pair into TRL's explicit prompt/completion format.

    The pair must share an identical rendered context and differ only in the
    final assistant message. This keeps environment and user messages out of
    the DPO completion mask instead of relying on longest-prefix inference.
    """
    instance = _as_mapping(instance, "paired_conversation instance")
    chosen = _normalize_conversation(instance.get("chosen"), "chosen")
    rejected = _normalize_conversation(instance.get("rejected"), "rejected")

    if chosen["system"] != rejected["system"]:
        raise ValueError("chosen and rejected must use the same system prompt")
    if chosen["tools"] != rejected["tools"]:
        raise ValueError("chosen and rejected must use the same tool definitions")

    chosen_prompt = chosen["messages"][:-1]
    rejected_prompt = rejected["messages"][:-1]
    if chosen_prompt != rejected_prompt:
        raise ValueError("chosen and rejected must share every message before the final assistant completion")

    chosen_completion = chosen["messages"][-1]
    rejected_completion = rejected["messages"][-1]
    if chosen_completion == rejected_completion:
        raise ValueError("chosen and rejected final assistant completions must differ")

    prompt = copy.deepcopy(chosen_prompt)
    if chosen["system"] is not None:
        prompt.insert(0, {"role": "system", "content": chosen["system"]})

    tools = chosen["tools"]
    return {
        "prompt": prompt,
        "chosen": [copy.deepcopy(chosen_completion)],
        "rejected": [copy.deepcopy(rejected_completion)],
        # Preserve the source structure and insertion order so SFT and DPO use
        # identical chat-template bytes for the same prompt.
        "tools": copy.deepcopy(tools) if tools else None,
    }


def _prepare_paired_conversation_dataset(dataset: Dataset, split_name: str):
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{split_name}_dataset must be an lmflow.datasets.Dataset")
    if dataset.get_backend() != "huggingface":
        raise ValueError(f"{split_name}_dataset must use the huggingface backend")
    if dataset.get_type() != "paired_conversation":
        raise ValueError(f"{split_name}_dataset must have type 'paired_conversation', got {dataset.get_type()!r}")

    backend_dataset = dataset.get_backend_dataset()
    if len(backend_dataset) == 0:
        raise ValueError(f"{split_name}_dataset must contain at least one preference pair")

    map_kwargs = {
        "remove_columns": backend_dataset.column_names,
        "desc": f"Preparing {split_name} paired conversations for DPO",
    }
    num_proc = getattr(dataset.get_data_args(), "preprocessing_num_workers", None)
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return backend_dataset.map(_paired_conversation_to_preference, **map_kwargs)


def _validate_dpo_config(args: Any, config_class: type) -> None:
    if not isinstance(args, config_class):
        raise TypeError(f"args must be a TRL {_SUPPORTED_TRL_VERSION} DPOConfig")
    loss_types = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
    if loss_types != ["sigmoid"]:
        raise ValueError(f"TRLDPOTrainer v1 supports standard sigmoid DPO only, got loss_type={loss_types!r}")
    if not isinstance(args.beta, Real) or not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError(f"DPO beta must be finite and positive, got {args.beta!r}")


class TRLDPOTrainer:
    """Train standard DPO on LMFlow ``paired_conversation`` datasets."""

    def __init__(
        self,
        model: Any,
        processing_class: Any,
        args: Any,
        train_dataset: Dataset,
        *,
        eval_dataset: Optional[Dataset] = None,
        ref_model: Any = None,
        peft_config: Any = None,
    ) -> None:
        config_class, trainer_class = _load_trl()
        _validate_dpo_config(args, config_class)

        prepared_train_dataset = _prepare_paired_conversation_dataset(train_dataset, "train")
        prepared_eval_dataset = (
            _prepare_paired_conversation_dataset(eval_dataset, "eval") if eval_dataset is not None else None
        )
        self._trainer = trainer_class(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=prepared_train_dataset,
            eval_dataset=prepared_eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
        )

    @property
    def model(self) -> Any:
        return self._trainer.model

    def unwrap_model(self) -> Any:
        """Return the policy model without an Accelerate wrapper."""
        return self._trainer.accelerator.unwrap_model(self.model)

    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict[str, float]:
        """Run TRL training and return version-neutral scalar metrics."""
        result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = {key: float(value) for key, value in result.metrics.items() if isinstance(value, Real)}
        metrics["global_step"] = float(self._trainer.state.global_step)
        return metrics

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save the policy and processing artifacts through the locked trainer."""
        self._trainer.save_model(output_dir)


__all__ = ["TRLDPOTrainer"]
