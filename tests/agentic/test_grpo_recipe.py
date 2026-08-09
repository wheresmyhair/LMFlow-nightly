import unittest

import torch

from lmflow.agentic.grpo_recipe import run_grpo_step
from lmflow.utils.protocol import DataProto


class _RecordingTrainer:
    def __init__(self):
        self.data = None

    def train_step(self, data):
        self.data = data
        return {"loss": 0.25, "selected_tokens": 7.0, "grad_norm": 1.5}


class GRPORecipeTest(unittest.TestCase):
    def setUp(self):
        self.data = DataProto.from_dict(
            tensors={"rewards": torch.tensor([1.0, 3.0, 2.0, 6.0])},
            non_tensors={"group_ids": ["a", "a", "b", "b"]},
        )

    def test_connects_rewarded_batch_to_trainer_and_names_metrics(self):
        trainer = _RecordingTrainer()

        metrics = run_grpo_step(trainer, self.data)

        self.assertIs(trainer.data, self.data)
        expected_advantages = torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071])
        torch.testing.assert_close(
            self.data.batch["advantages"],
            expected_advantages,
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertEqual(metrics["train/loss"], 0.25)
        self.assertEqual(metrics["train/tokens"], 7.0)
        self.assertEqual(metrics["train/grad_norm"], 1.5)
        self.assertEqual(metrics["rollout/trajectories"], 4.0)
        self.assertEqual(metrics["rollout/groups"], 2.0)
        self.assertAlmostEqual(metrics["reward/total_mean"], 3.0)
        self.assertAlmostEqual(metrics["reward/total_std"], 3.5**0.5, places=6)
        self.assertAlmostEqual(metrics["train/advantage_mean"], 0.0, places=6)

    def test_can_center_rewards_without_std_scaling(self):
        trainer = _RecordingTrainer()

        run_grpo_step(trainer, self.data, scale_rewards=False)

        torch.testing.assert_close(
            self.data.batch["advantages"],
            torch.tensor([-1.0, 1.0, -2.0, 2.0]),
        )


if __name__ == "__main__":
    unittest.main()
