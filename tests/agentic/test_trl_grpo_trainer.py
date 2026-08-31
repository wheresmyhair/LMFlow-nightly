from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lmflow.agentic.trl_grpo_trainer as trl_grpo_trainer
from lmflow.agentic.trl_grpo_trainer import (
    _build_behavior_logprob_trainer_class,
    _SealedRolloutBridge,
    _validate_training_args,
)
from lmflow.utils.protocol import DataProto


def _sealed_rollouts() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor(
                [
                    [1, 10, 20, 30, 0, 0],
                    [1, 10, 21, 31, 0, 0],
                    [1, 11, 22, 32, 40, 0],
                    [1, 11, 23, 33, 41, 0],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                ]
            ),
            "loss_mask": torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                ]
            ),
            "old_log_probs": torch.tensor(
                [
                    [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.2, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.3, 0.0, -0.4, 0.0],
                    [0.0, 0.0, -0.5, 0.0, -0.6, 0.0],
                ]
            ),
            "prompt_lengths": torch.tensor([2, 2, 2, 2]),
            "rewards": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        },
        non_tensors={
            "task_ids": np.asarray(["task-0", "task-0", "task-1", "task-1"]),
            "group_ids": np.asarray([10, 10, 11, 11]),
            "rollout_ids": np.asarray([100, 101, 102, 103]),
        },
        meta_info={
            "policy_version": "policy-7",
            "logprob_provenance": {
                "behavior": {
                    "source": "test.sampled-token-logprobs",
                    "policy_version": "policy-7",
                }
            },
        },
    )


def _training_args(**overrides):
    values = {
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
        "max_steps": 1,
        "world_size": 1,
        "per_device_train_batch_size": 1,
        "num_generations": 2,
        "generation_batch_size": 4,
        "steps_per_generation": 4,
        "gradient_accumulation_steps": 4,
        "max_prompt_length": 2,
        "max_completion_length": 4,
        "reward_weights": None,
        "off_policy_mask_threshold": None,
        "entropy_coef": 0.0,
        "use_adaptive_entropy": False,
        "delta": None,
        "epsilon": 0.2,
        "epsilon_high": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_projects_sealed_groups_to_official_rollout_and_reward_hooks():
    bridge = _SealedRolloutBridge(_sealed_rollouts())

    assert bridge.dataset_dict() == {"prompt": ["lmflow-sealed-group-0", "lmflow-sealed-group-1"]}
    rollout = bridge.rollout_func(
        [
            "lmflow-sealed-group-0",
            "lmflow-sealed-group-0",
            "lmflow-sealed-group-1",
            "lmflow-sealed-group-1",
        ],
        trainer=None,
    )

    assert rollout["prompt_ids"] == [[1, 10], [1, 10], [1, 11], [1, 11]]
    assert rollout["completion_ids"] == [[20, 30], [21, 31], [22, 32, 40], [23, 33, 41]]
    assert rollout["env_mask"] == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    for actual, expected in zip(
        rollout["logprobs"],
        [[-0.1, 0.0], [-0.2, 0.0], [-0.3, 0.0, -0.4], [-0.5, 0.0, -0.6]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert rollout["audited_reward"] == [1.0, 0.0, 0.0, 1.0]
    assert rollout["policy_version"] == ["policy-7"] * 4
    assert bridge.logprob_provenance == {
        "behavior": {
            "source": "test.sampled-token-logprobs",
            "policy_version": "policy-7",
        },
        "trainer_old": {
            "source": "behavior",
            "input_field": "DataProto.batch['old_log_probs']",
            "trl_field": "old_per_token_logps",
            "compatibility_contract": "trl==1.9.2:post-generate-score-injection",
        },
        "reference": {"enabled": False, "source": None, "reason": "beta=0"},
    }

    rewards = bridge.reward_func(
        prompts=["internal"] * 4,
        completions=["decoded"] * 4,
        completion_ids=rollout["completion_ids"],
        audited_reward=rollout["audited_reward"],
        task_id=rollout["task_id"],
        group_id=rollout["group_id"],
        rollout_id=rollout["rollout_id"],
        policy_version=rollout["policy_version"],
    )
    assert rewards == [1.0, 0.0, 0.0, 1.0]

    with pytest.raises(RuntimeError, match="already been consumed"):
        bridge.rollout_func(["lmflow-sealed-group-0"] * 2, trainer=None)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data.batch.pop("prompt_lengths"), "prompt_lengths"),
        (
            lambda data: data.meta_info["logprob_provenance"]["behavior"].update({"policy_version": "stale-policy"}),
            "policy_version",
        ),
        (lambda data: data.batch["loss_mask"].__setitem__((0, 2), 0.5), "only 0 or 1"),
        (lambda data: data.batch["attention_mask"].__setitem__((0, 1), 0), "contiguous right padding"),
        (lambda data: data.non_tensor_batch["group_ids"].__setitem__(3, 12), "same size"),
        (lambda data: data.non_tensor_batch["rollout_ids"].__setitem__(3, 102), "unique"),
        (lambda data: data.non_tensor_batch["task_ids"].__setitem__(1, "other-task"), "one task_id"),
    ],
)
def test_sealed_bridge_fails_closed_on_contract_drift(mutation, match):
    data = _sealed_rollouts()
    mutation(data)

    with pytest.raises((KeyError, TypeError, ValueError), match=match):
        _SealedRolloutBridge(data)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"use_vllm": True}, "use_vllm"),
        ({"vllm_importance_sampling_correction": True}, "importance_sampling_correction"),
        ({"beta": 0.1}, "beta"),
        ({"sync_ref_model": True}, "sync_ref_model"),
        ({"use_bias_correction_kl": True}, "use_bias_correction_kl"),
        ({"num_iterations": 2}, "num_iterations"),
        ({"max_steps": 2}, "max_steps"),
        ({"loss_type": "dapo"}, "loss_type"),
        ({"epsilon": 0.1}, "epsilon"),
        ({"steps_per_generation": 2}, "must equal"),
        ({"per_device_train_batch_size": 2}, "per_device_train_batch_size"),
        ({"gradient_accumulation_steps": 2, "steps_per_generation": 2}, "sealed batch size"),
        ({"max_completion_length": 2}, "sealed maximum"),
    ],
)
def test_training_support_matrix_rejects_unverified_combinations(overrides, match):
    bridge = _SealedRolloutBridge(_sealed_rollouts())

    with pytest.raises(ValueError, match=match):
        _validate_training_args(_training_args(**overrides), bridge)


def test_training_support_matrix_accepts_frozen_single_step_recipe():
    bridge = _SealedRolloutBridge(_sealed_rollouts())

    _validate_training_args(_training_args(), bridge)


def test_sealed_group_allows_per_rollout_multi_turn_conditioning_prefixes():
    data = _sealed_rollouts()
    data.batch["input_ids"][1, 1] = 12

    rollout = _SealedRolloutBridge(data).rollout_func(
        [
            "lmflow-sealed-group-0",
            "lmflow-sealed-group-0",
            "lmflow-sealed-group-1",
            "lmflow-sealed-group-1",
        ],
        trainer=None,
    )

    assert rollout["prompt_ids"][:2] == [[1, 10], [1, 12]]


def test_behavior_logprob_bridge_injects_a_clone_without_rewriting_sampling_values():
    sampling = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])

    class BaseTrainer:
        def _generate_and_score_completions(self, inputs):
            assert inputs == ["sealed"]
            return {
                "completion_ids": torch.tensor([[10, 11], [12, 13]]),
                "sampling_per_token_logps": sampling,
            }

    trainer = _build_behavior_logprob_trainer_class(BaseTrainer)()
    output = trainer._generate_and_score_completions(["sealed"])

    assert output["sampling_per_token_logps"] is sampling
    assert output["old_per_token_logps"] is not sampling
    torch.testing.assert_close(output["old_per_token_logps"], sampling)
    assert trainer.lmflow_old_logprobs_source == "behavior"


@pytest.mark.parametrize(
    "output,match",
    [
        ({"completion_ids": torch.tensor([[1]])}, "did not preserve"),
        (
            {
                "completion_ids": torch.tensor([[1]]),
                "sampling_per_token_logps": torch.tensor([[-0.1]]),
                "old_per_token_logps": torch.tensor([[-0.1]]),
            },
            "unexpectedly produced",
        ),
        (
            {
                "completion_ids": torch.tensor([[1, 2]]),
                "sampling_per_token_logps": torch.tensor([[-0.1]]),
            },
            "do not align",
        ),
        (
            {
                "completion_ids": torch.tensor([[1]]),
                "sampling_per_token_logps": torch.tensor([[float("nan")]]),
            },
            "finite",
        ),
    ],
)
def test_behavior_logprob_bridge_fails_closed_on_private_output_drift(output, match):
    class BaseTrainer:
        def _generate_and_score_completions(self, inputs):
            del inputs
            return output

    trainer = _build_behavior_logprob_trainer_class(BaseTrainer)()

    with pytest.raises(RuntimeError, match=match):
        trainer._generate_and_score_completions([])


def test_backend_fails_closed_on_trl_version_drift(monkeypatch):
    monkeypatch.setattr(trl_grpo_trainer, "version", lambda package: "1.9.3")

    with pytest.raises(RuntimeError, match="supports trl==1.9.2, found 1.9.3"):
        trl_grpo_trainer._load_trl()
