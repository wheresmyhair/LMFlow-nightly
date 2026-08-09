import unittest

import torch

from lmflow.agentic import RolloutGroupAssembler
from lmflow.agentic.grpo_recipe import run_grpo_step
from lmflow.utils.protocol import DataProto


def _results(*rows, policy_version=7):
    return DataProto.from_dict(
        tensors={"rewards": torch.tensor([row[2] for row in rows], dtype=torch.float32)},
        non_tensors={
            "group_ids": [row[0] for row in rows],
            "rollout_ids": [row[1] for row in rows],
        },
        meta_info={"policy_version": policy_version},
    )


class _RecordingTrainer:
    def __init__(self):
        self.data = None

    def train_step(self, data):
        self.data = data
        return {"loss": 0.5, "selected_tokens": 4.0}


class RolloutGroupAssemblerTest(unittest.TestCase):
    def setUp(self):
        self.assembler = RolloutGroupAssembler(policy_version=7, timeout_seconds=5.0)

    def test_complete_group_does_not_wait_for_unrelated_slow_group(self):
        self.assembler.register_group("slow", ["slow-0", "slow-1"], started_at=0.0)
        self.assembler.register_group("fast", ["fast-0", "fast-1"], started_at=0.0)

        self.assertEqual(self.assembler.add(_results(("slow", "slow-0", 1.0)), received_at=1.0), ())
        self.assertEqual(self.assembler.add(_results(("fast", "fast-1", 4.0)), received_at=2.0), ())
        self.assertEqual(
            self.assembler.add(_results(("fast", "fast-0", 2.0)), received_at=3.0),
            ("fast",),
        )

        ready = self.assembler.pop_ready()
        self.assertEqual(ready.non_tensor_batch["rollout_ids"].tolist(), ["fast-0", "fast-1"])
        torch.testing.assert_close(ready.batch["rewards"], torch.tensor([2.0, 4.0]))
        self.assertIsNone(self.assembler.pop_ready())

        self.assertEqual(
            self.assembler.add(_results(("slow", "slow-1", 3.0)), received_at=5.0),
            ("slow",),
        )
        slow = self.assembler.pop_ready()
        self.assertEqual(slow.non_tensor_batch["rollout_ids"].tolist(), ["slow-0", "slow-1"])

        metrics = self.assembler.metrics()
        self.assertEqual(metrics["rollout/groups_completed"], 2.0)
        self.assertAlmostEqual(metrics["rollout/group_completion_latency_mean"], 4.0)
        self.assertAlmostEqual(metrics["rollout/group_straggler_wait_mean"], 2.5)

    def test_pop_ready_preserves_whole_group_boundaries(self):
        self.assembler.register_group("a", ["a-0", "a-1"], started_at=0.0)
        self.assembler.register_group("b", ["b-0", "b-1"], started_at=0.0)
        completed = self.assembler.add(
            _results(
                ("b", "b-1", 4.0),
                ("a", "a-1", 2.0),
                ("b", "b-0", 3.0),
                ("a", "a-0", 1.0),
            ),
            received_at=1.0,
        )

        self.assertEqual(completed, ("a", "b"))
        first = self.assembler.pop_ready(max_groups=1)
        second = self.assembler.pop_ready(max_groups=1)
        self.assertEqual(first.non_tensor_batch["rollout_ids"].tolist(), ["a-0", "a-1"])
        self.assertEqual(second.non_tensor_batch["rollout_ids"].tolist(), ["b-0", "b-1"])

    def test_duplicate_result_is_rejected_without_partial_mutation(self):
        self.assembler.register_group("group", ["r0", "r1"], started_at=0.0)
        result = _results(("group", "r0", 1.0))
        self.assembler.add(result, received_at=1.0)

        with self.assertRaisesRegex(ValueError, "duplicate rollout result"):
            self.assembler.add(
                _results(("group", "r1", 2.0), ("group", "r0", 1.0)),
                received_at=2.0,
            )

        self.assertIsNone(self.assembler.pop_ready())
        self.assertEqual(self.assembler.metrics()["rollout/trajectories_pending"], 1.0)
        self.assertEqual(self.assembler.metrics()["rollout/duplicate_results_rejected"], 1.0)
        self.assertEqual(self.assembler.add(_results(("group", "r1", 2.0)), received_at=3.0), ("group",))

    def test_policy_version_mismatch_fails_closed(self):
        self.assembler.register_group("group", ["r0"], started_at=0.0)

        with self.assertRaisesRegex(ValueError, "policy_version mismatch"):
            self.assembler.add(_results(("group", "r0", 1.0), policy_version=8), received_at=1.0)

        self.assertIsNone(self.assembler.pop_ready())
        self.assertEqual(self.assembler.metrics()["rollout/trajectories_pending"], 0.0)
        self.assertEqual(self.assembler.add(_results(("group", "r0", 1.0)), received_at=2.0), ("group",))

    def test_timeout_closes_even_a_group_with_no_results(self):
        self.assembler.register_group("group", ["r0", "r1"], started_at=10.0)

        self.assertEqual(self.assembler.expire(now=14.9), ())
        self.assertEqual(self.assembler.expire(now=15.0), ("group",))
        with self.assertRaisesRegex(ValueError, "already closed"):
            self.assembler.add(_results(("group", "r0", 1.0)), received_at=16.0)

        metrics = self.assembler.metrics()
        self.assertEqual(metrics["rollout/groups_timed_out"], 1.0)
        self.assertEqual(metrics["rollout/groups_pending"], 0.0)

    def test_cancelled_attempt_cannot_contaminate_whole_group_retry(self):
        self.assembler.register_group("attempt-1", ["old-0", "old-1"], started_at=0.0)
        self.assembler.add(_results(("attempt-1", "old-0", 1.0)), received_at=1.0)
        self.assembler.cancel_group("attempt-1")

        self.assembler.register_group("attempt-2", ["new-0", "new-1"], started_at=2.0)
        with self.assertRaisesRegex(ValueError, "already closed"):
            self.assembler.add(_results(("attempt-1", "old-1", 9.0)), received_at=3.0)

        self.assembler.add(
            _results(("attempt-2", "new-0", 2.0), ("attempt-2", "new-1", 4.0)),
            received_at=4.0,
        )
        ready = self.assembler.pop_ready()
        self.assertEqual(ready.non_tensor_batch["rollout_ids"].tolist(), ["new-0", "new-1"])
        self.assertEqual(self.assembler.metrics()["rollout/groups_cancelled"], 1.0)

    def test_ready_groups_feed_the_existing_grpo_recipe(self):
        self.assembler.register_group("a", ["a-0", "a-1"], started_at=0.0)
        self.assembler.register_group("b", ["b-0", "b-1"], started_at=0.0)
        self.assembler.add(
            _results(
                ("a", "a-0", 1.0),
                ("a", "a-1", 3.0),
                ("b", "b-0", 2.0),
                ("b", "b-1", 6.0),
            ),
            received_at=1.0,
        )
        batch = self.assembler.pop_ready()
        trainer = _RecordingTrainer()

        metrics = run_grpo_step(trainer, batch)

        self.assertIs(trainer.data, batch)
        self.assertEqual(metrics["rollout/groups"], 2.0)
        self.assertEqual(metrics["rollout/trajectories"], 4.0)
        torch.testing.assert_close(
            batch.batch["advantages"],
            torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071]),
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
