import unittest
from types import SimpleNamespace

import torch

from lmflow.agentic.trl_policy_trainer import _build_trl_inputs, _validate_grpo_config
from lmflow.utils.protocol import DataProto


def _supported_config(**overrides):
    values = {
        "beta": 0.0,
        "delta": None,
        "entropy_coef": 0.0,
        "epsilon": 0.2,
        "epsilon_high": None,
        "gradient_accumulation_steps": 1,
        "importance_sampling_level": "token",
        "loss_type": "grpo",
        "off_policy_mask_threshold": None,
        "temperature": 1.0,
        "top_entropy_quantile": 1.0,
        "use_adaptive_entropy": False,
        "use_vllm": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TRLInputMappingTest(unittest.TestCase):
    def test_maps_full_sequence_fields_without_changing_context_mask(self):
        data = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                "loss_mask": torch.tensor([[0.0, 0.0, 0.5, 1.0], [0.0, 1.0, 1.0, 0.0]]),
                "advantages": torch.tensor([1.0, -1.0]),
                "old_log_probs": torch.tensor([[0.0, -0.1, -0.2, -0.3], [0.0, -0.4, -0.5, 0.0]]),
            }
        )

        actual = _build_trl_inputs(data, torch.device("cpu"))

        torch.testing.assert_close(actual["prompt_ids"], data.batch["input_ids"][:, :1])
        torch.testing.assert_close(actual["completion_ids"], data.batch["input_ids"][:, 1:])
        torch.testing.assert_close(actual["completion_mask"], data.batch["attention_mask"][:, 1:])
        torch.testing.assert_close(actual["tool_mask"], data.batch["loss_mask"][:, 1:])
        torch.testing.assert_close(actual["advantages"], data.batch["advantages"])
        torch.testing.assert_close(actual["old_per_token_logps"], data.batch["old_log_probs"][:, 1:])

    def test_strips_unpredicted_column_from_token_advantages(self):
        advantages = torch.tensor([[0.0, 1.0, 2.0]])
        data = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
                "advantages": advantages,
            }
        )

        actual = _build_trl_inputs(data, torch.device("cpu"))

        torch.testing.assert_close(actual["advantages"], advantages[:, 1:])

    def test_rejects_selected_first_or_hidden_tokens(self):
        tensors = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "loss_mask": torch.tensor([[1.0, 0.0, 1.0]]),
            "advantages": torch.tensor([1.0]),
        }
        with self.assertRaisesRegex(ValueError, "first token"):
            _build_trl_inputs(DataProto.from_dict(tensors=tensors), torch.device("cpu"))

        tensors["loss_mask"] = torch.tensor([[0.0, 0.0, 1.0]])
        tensors["attention_mask"] = torch.tensor([[1, 1, 0]])
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            _build_trl_inputs(DataProto.from_dict(tensors=tensors), torch.device("cpu"))


class TRLConfigValidationTest(unittest.TestCase):
    def test_accepts_locked_grpo_subset(self):
        _validate_grpo_config(_supported_config())

    def test_rejects_semantics_not_implemented_by_v1(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            _validate_grpo_config(_supported_config(temperature=0.7))
        with self.assertRaisesRegex(ValueError, "entropy_coef"):
            _validate_grpo_config(_supported_config(entropy_coef=0.01))


if __name__ == "__main__":
    unittest.main()
