import copy
import unittest
from types import SimpleNamespace

import torch

from lmflow.agentic.policy import causal_token_log_probs, grpo_loss_from_model
from lmflow.utils.protocol import DataProto


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocabulary_size=7, hidden_size=5, dtype=torch.float64):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size, dtype=dtype)
        self.lm_head = torch.nn.Linear(hidden_size, vocabulary_size, bias=False, dtype=dtype)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class CausalTokenLogProbTest(unittest.TestCase):
    def test_aligns_log_probs_with_target_token_positions(self):
        logits = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0],
                    [0.0, 3.0, 1.0],
                    [-1.0, 0.0, 2.0],
                ]
            ],
            requires_grad=True,
        )
        input_ids = torch.tensor([[0, 2, 1]])

        actual = causal_token_log_probs(logits, input_ids)

        expected = torch.stack(
            [
                logits.new_zeros(()),
                logits[0, 0].log_softmax(dim=-1)[2],
                logits[0, 1].log_softmax(dim=-1)[1],
            ]
        ).unsqueeze(0)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1]))

    def test_rejects_misaligned_shapes(self):
        with self.assertRaisesRegex(ValueError, "does not match input_ids"):
            causal_token_log_probs(torch.zeros((2, 3, 5)), torch.zeros((2, 2), dtype=torch.long))


class PolicyUpdateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = _TinyCausalLM()
        self.input_ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 2]])
        self.attention_mask = torch.ones_like(self.input_ids)
        self.loss_mask = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        with torch.no_grad():
            logits = self.model(self.input_ids, self.attention_mask).logits
            old_log_probs = causal_token_log_probs(logits, self.input_ids)
        self.data = DataProto.from_dict(
            tensors={
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "loss_mask": self.loss_mask,
                "advantages": torch.tensor([1.0, -1.0], dtype=torch.float64),
                "old_log_probs": old_log_probs,
            }
        )

    def test_completes_a_tiny_causal_lm_update(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        parameters_before = [parameter.detach().clone() for parameter in self.model.parameters()]

        loss = grpo_loss_from_model(self.model, self.data)
        loss.backward()
        optimizer.step()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(all(parameter.grad is not None for parameter in self.model.parameters()))
        parameters_after = list(self.model.parameters())
        self.assertEqual(len(parameters_after), len(parameters_before))
        self.assertTrue(
            any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(parameters_after, parameters_before)
            )
        )

    def test_rejects_first_or_padded_policy_tokens(self):
        first_token_data = copy.deepcopy(self.data)
        first_token_data.batch["loss_mask"][0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "first token"):
            grpo_loss_from_model(self.model, first_token_data)

        padded_token_data = copy.deepcopy(self.data)
        padded_token_data.batch["attention_mask"][0, -1] = 0
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            grpo_loss_from_model(self.model, padded_token_data)


if __name__ == "__main__":
    unittest.main()
