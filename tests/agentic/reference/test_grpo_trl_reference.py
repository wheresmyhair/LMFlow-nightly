"""Differential checks against the locked TRL GRPO implementation."""

from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
import torch

from lmflow.agentic.algorithms import compute_group_advantages, grpo_policy_loss
from lmflow.utils.protocol import DataProto

pytestmark = pytest.mark.optional_backend

_TRL_VERSION = "1.9.2"


def _load_trl_reference():
    try:
        installed_version = version("trl")
    except PackageNotFoundError:
        pytest.skip(f"requires trl=={_TRL_VERSION}")
    if installed_version != _TRL_VERSION:
        pytest.skip(f"requires trl=={_TRL_VERSION}, found {installed_version}")

    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.utils import nanstd

    return GRPOTrainer, nanstd


class _SingleProcessAccelerator:
    num_processes = 1
    sync_gradients = True

    @staticmethod
    def gather(tensor):
        return tensor

    @staticmethod
    def gather_for_metrics(tensor):
        return tensor

    @staticmethod
    def reduce(tensor, reduction="sum"):
        assert reduction == "sum"
        return tensor


def _trl_grpo_loss(current_log_probs, old_log_probs, advantages, loss_mask):
    grpo_trainer, _ = _load_trl_reference()
    trainer = object.__new__(grpo_trainer)
    model = SimpleNamespace(training=True)
    trainer.model = model
    trainer.accelerator = _SingleProcessAccelerator()
    trainer.args = SimpleNamespace(delta=None, use_bias_correction_kl=False)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (
        current_log_probs,
        torch.zeros_like(current_log_probs),
        None,
    )
    trainer.aux_loss_enabled = False
    trainer.beta = 0.0
    trainer.current_gradient_accumulation_steps = 1
    trainer.epsilon_high = 0.2
    trainer.epsilon_low = 0.2
    trainer._entropy_bonus_enabled = False
    trainer.importance_sampling_level = "token"
    trainer.loss_type = "grpo"
    trainer.off_policy_mask_threshold = None
    trainer.top_entropy_quantile = 1.0
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False

    batch_size, sequence_length = current_log_probs.shape
    inputs = {
        "prompt_ids": torch.zeros((batch_size, 1), dtype=torch.long),
        "prompt_mask": torch.ones((batch_size, 1), dtype=loss_mask.dtype),
        "completion_ids": torch.zeros((batch_size, sequence_length), dtype=torch.long),
        "completion_mask": loss_mask,
        "advantages": advantages,
        "old_per_token_logps": old_log_probs,
    }
    return grpo_trainer._compute_loss(trainer, model, inputs)


def test_group_advantages_match_trl_reference():
    _, trl_nanstd = _load_trl_reference()
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    data = DataProto.from_dict(
        tensors={"rewards": rewards},
        non_tensors={"group_ids": ["a", "a", "b", "b"]},
    )

    actual = compute_group_advantages(data)

    grouped_rewards = rewards.view(2, 2)
    means = grouped_rewards.mean(dim=1).repeat_interleave(2)
    standard_deviations = trl_nanstd(grouped_rewards, dim=1).repeat_interleave(2)
    expected = (rewards - means) / (standard_deviations + 1e-4)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_policy_loss_gradient_and_optimizer_delta_match_trl():
    initial_log_probs = torch.tensor(
        [[0.3, -0.4, 0.0], [-0.3, 0.4, 0.1]],
        dtype=torch.float64,
    )
    old_log_probs = torch.zeros_like(initial_log_probs)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
    loss_mask = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    lmflow_log_probs = torch.nn.Parameter(initial_log_probs.clone())
    trl_log_probs = torch.nn.Parameter(initial_log_probs.clone())
    lmflow_optimizer = torch.optim.SGD([lmflow_log_probs], lr=0.1)
    trl_optimizer = torch.optim.SGD([trl_log_probs], lr=0.1)
    data = DataProto.from_dict(
        tensors={
            "loss_mask": loss_mask,
            "advantages": advantages,
            "old_log_probs": old_log_probs,
        }
    )

    lmflow_loss = grpo_policy_loss(data, lmflow_log_probs)
    trl_loss = _trl_grpo_loss(trl_log_probs, old_log_probs, advantages, loss_mask)
    lmflow_loss.backward()
    trl_loss.backward()

    torch.testing.assert_close(lmflow_loss.detach(), trl_loss.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lmflow_log_probs.grad, trl_log_probs.grad, rtol=1e-6, atol=1e-6)

    lmflow_optimizer.step()
    trl_optimizer.step()
    torch.testing.assert_close(lmflow_log_probs, trl_log_probs, rtol=1e-6, atol=1e-6)
