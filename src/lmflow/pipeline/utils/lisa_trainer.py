import logging
from collections.abc import MutableMapping
from operator import attrgetter
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, TrainerCallback

logger = logging.getLogger(__name__)


class DynamicLayerActivationCallback(TrainerCallback):
    """Switch LISA layers and manage the optimizer state tied to them.

    PyTorch optimizers initialize per-parameter state lazily, but setting
    ``requires_grad=False`` does not release state that was already created.
    State for all sampled intermediate layers is removed at every LISA
    boundary. This bounds optimizer-state device memory and restarts Adam's
    local time at zero for every sampling interval, including when the same
    layer is sampled consecutively.
    """

    def __init__(
        self,
        n_layers: int,
        interval_steps: int,
        model: PreTrainedModel,
        lisa_layers_attribute: Optional[str] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.interval_steps = interval_steps
        self.model = model

        class_to_layers_map = {
            "LlamaForCausalLM": "model.model.layers",
            "Qwen2ForCausalLM": "model.model.layers",
            "MistralForCausalLM": "model.model.layers",
            "MixtralForCausalLM": "model.model.layers",
            "GemmaForCausalLM": "model.model.layers",
            "GPT2LMHeadModel": "model.transformer.h",
            "HymbaForCausalLM": "model.model.layers",
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            if lisa_layers_attribute is None:
                raise ValueError("Please provide the attribute used to access the model layers.")
            self.layers_attribute = lisa_layers_attribute

        self.total_layers = len(self._get_layers())
        if self.interval_steps <= 0:
            raise ValueError("lisa_interval_steps must be greater than zero.")
        if not 0 < self.n_layers <= self.total_layers:
            raise ValueError(
                f"lisa_activated_layers must be between 1 and {self.total_layers}, got {self.n_layers}."
            )

        self.active_layers_indices = []
        self._last_switch_step = None

    def _get_layers(self):
        return attrgetter(self.layers_attribute)(self)

    @staticmethod
    def _optimizer_state(optimizer) -> Optional[MutableMapping]:
        """Find the mutable state mapping through common optimizer wrappers."""
        seen = set()
        current = optimizer
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            state = getattr(current, "state", None)
            if isinstance(state, MutableMapping):
                return state
            current = next(
                (
                    wrapped
                    for attribute in ("optimizer", "base_optimizer", "optim")
                    if (wrapped := getattr(current, attribute, None)) is not None
                ),
                None,
            )
        return None

    @staticmethod
    def _state_tensor_bytes(value) -> int:
        if isinstance(value, torch.Tensor):
            local_tensor = getattr(value, "_local_tensor", None)
            if isinstance(local_tensor, torch.Tensor):
                return local_tensor.numel() * local_tensor.element_size()
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(DynamicLayerActivationCallback._state_tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(DynamicLayerActivationCallback._state_tensor_bytes(item) for item in value)
        return 0

    @staticmethod
    def _distributed_device() -> torch.device:
        backend = str(torch.distributed.get_backend()).lower()
        if "nccl" in backend:
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _sample_active_layers(self):
        """Sample once and broadcast so every distributed rank trains the same layers."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return np.random.choice(range(self.total_layers), self.n_layers, replace=False).tolist()

        device = self._distributed_device()
        if torch.distributed.get_rank() == 0:
            sampled = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
            indices = torch.tensor(sampled, dtype=torch.long, device=device)
        else:
            indices = torch.empty(self.n_layers, dtype=torch.long, device=device)
        torch.distributed.broadcast(indices, src=0)
        return indices.cpu().tolist()

    def freeze_all_layers(self):
        for layer in self._get_layers():
            for param in layer.parameters():
                param.requires_grad = False

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        """Start a fresh LISA interval after any optimizer checkpoint is loaded."""
        self.switch_active_layers(optimizer=optimizer)
        self._last_switch_step = state.global_step

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        if self._last_switch_step is None or state.global_step - self._last_switch_step >= self.interval_steps:
            self.switch_active_layers(optimizer=optimizer)
            self._last_switch_step = state.global_step

    def _drop_lisa_optimizer_state(self, optimizer) -> tuple[int, int]:
        optimizer_state = self._optimizer_state(optimizer)
        if optimizer_state is None:
            logger.warning("LISA could not access optimizer.state; inactive optimizer tensors were not released.")
            return 0, 0

        lisa_params = {param for layer in self._get_layers() for param in layer.parameters()}

        dropped_parameters = 0
        dropped_bytes = 0
        for param in lisa_params:
            if param not in optimizer_state:
                continue
            dropped_bytes += self._state_tensor_bytes(optimizer_state[param])
            del optimizer_state[param]
            dropped_parameters += 1

        if dropped_parameters and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dropped_parameters, dropped_bytes

    def switch_active_layers(self, optimizer=None):
        self.freeze_all_layers()
        self.active_layers_indices = self._sample_active_layers()

        if optimizer is not None:
            dropped_parameters, dropped_bytes = self._drop_lisa_optimizer_state(optimizer)
            if dropped_parameters and (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                logger.info(
                    "LISA released optimizer state for %d parameters (%.2f MiB of tensor storage on this rank).",
                    dropped_parameters,
                    dropped_bytes / (1024**2),
                )

        logger.info("Activating layers at indices %s for the next steps.", self.active_layers_indices)
        layers = self._get_layers()
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True
