import unittest

import torch

from lmflow.agentic.algorithms import compute_group_advantages, grpo_policy_loss
from lmflow.utils.protocol import DataProto


class GroupAdvantageTest(unittest.TestCase):
    def test_normalizes_noncontiguous_groups(self):
        data = DataProto.from_dict(
            tensors={"rewards": torch.tensor([1.0, 2.0, 3.0, 4.0])},
            non_tensors={"group_ids": ["a", "b", "a", "b"]},
        )

        advantages = compute_group_advantages(data)

        expected = torch.tensor([-0.7070568, -0.7070568, 0.7070568, 0.7070568])
        torch.testing.assert_close(advantages, expected)

    def test_zeroes_constant_and_singleton_groups(self):
        data = DataProto.from_dict(
            tensors={"rewards": torch.tensor([2.0, 2.0, 5.0])},
            non_tensors={"group_ids": ["constant", "constant", "singleton"]},
        )

        advantages = compute_group_advantages(data)

        torch.testing.assert_close(advantages, torch.zeros(3))

    def test_can_center_rewards_without_scaling(self):
        data = DataProto.from_dict(
            tensors={"rewards": torch.tensor([1.0, 3.0])},
            non_tensors={"group_ids": ["group", "group"]},
        )

        advantages = compute_group_advantages(data, scale_rewards=False)

        torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0]))


class GRPOPolicyLossTest(unittest.TestCase):
    def test_applies_clipping_and_weighted_loss_mask(self):
        current_log_probs = torch.tensor(
            [[0.3, -0.4, 0.0], [-0.3, 0.4, 0.0]],
            requires_grad=True,
        )
        data = DataProto.from_dict(
            tensors={
                "loss_mask": torch.tensor([[1.0, 0.5, 0.0], [1.0, 1.0, 0.0]]),
                "advantages": torch.tensor([1.0, -1.0]),
                "old_log_probs": torch.zeros((2, 3)),
            }
        )

        loss = grpo_policy_loss(data, current_log_probs, clip_epsilon=0.2)

        ratio = current_log_probs.detach().exp()
        clipped_ratio = ratio.clamp(0.8, 1.2)
        advantages = data.batch["advantages"].unsqueeze(1)
        token_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        expected = -torch.stack(
            [
                (token_objective[0, 0] + 0.5 * token_objective[0, 1]) / 1.5,
                (token_objective[1, 0] + token_objective[1, 1]) / 2.0,
            ]
        ).mean()
        torch.testing.assert_close(loss.detach(), expected)

    def test_completes_one_optimizer_update(self):
        data = DataProto.from_dict(
            tensors={
                "rewards": torch.tensor([1.0, 0.0]),
                "loss_mask": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            },
            non_tensors={"group_ids": ["group", "group"]},
        )
        data.batch["advantages"] = compute_group_advantages(data)
        policy_log_probs = torch.nn.Parameter(torch.zeros((2, 2)))
        optimizer = torch.optim.SGD([policy_log_probs], lr=0.1)

        before = policy_log_probs.detach().clone()
        loss = grpo_policy_loss(data, policy_log_probs)
        loss.backward()
        optimizer.step()

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(policy_log_probs.grad.shape, policy_log_probs.shape)
        torch.testing.assert_close(policy_log_probs[:, 0], before[:, 0])
        self.assertFalse(torch.equal(policy_log_probs[:, 1], before[:, 1]))

    def test_gradient_matches_finite_difference(self):
        data = DataProto.from_dict(
            tensors={
                "loss_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
                "advantages": torch.tensor([0.75, -0.25], dtype=torch.float64),
                "old_log_probs": torch.zeros((2, 2), dtype=torch.float64),
            }
        )
        current_log_probs = torch.tensor(
            [[0.05, -0.05], [0.02, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: grpo_policy_loss(data, value),
                (current_log_probs,),
            )
        )

    def test_rejects_sequences_without_policy_tokens(self):
        data = DataProto.from_dict(
            tensors={
                "loss_mask": torch.tensor([[0.0, 0.0]]),
                "advantages": torch.tensor([1.0]),
            }
        )

        with self.assertRaisesRegex(ValueError, "at least one token"):
            grpo_policy_loss(data, torch.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
