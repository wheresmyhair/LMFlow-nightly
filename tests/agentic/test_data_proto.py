import unittest

import torch

from lmflow.agentic import TaskSpec, build_task_batch
from lmflow.utils.protocol import DataProto


class AgenticDataProtoTest(unittest.TestCase):
    def test_carries_core_and_algorithm_specific_fields(self):
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[10, 11, 0], [10, 21, 22]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                "loss_mask": torch.tensor([[0, 1, 0], [0, 1, 1]]),
                "rewards": torch.tensor([1.0, 0.0]),
                "advantages": torch.zeros((2, 3)),
            },
            non_tensors={
                "task_ids": ["task-a", "task-b"],
                "group_ids": ["group", "group"],
            },
            meta_info={"algorithm": "grpo", "policy_version": 3},
        )

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch.batch["advantages"].shape, (2, 3))
        self.assertEqual(batch.non_tensor_batch["group_ids"].tolist(), ["group", "group"])
        self.assertEqual(batch.meta_info["policy_version"], 3)

    def test_toy_task_rollout_reward_flow(self):
        first = TaskSpec(task_id="task-a", messages=[{"role": "user", "content": "A"}])
        second = TaskSpec(task_id="task-b", messages=[{"role": "user", "content": "B"}])
        task_batch = build_task_batch([first, first, second, second])

        rollout_batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor(
                    [
                        [10, 11, 12, 0],
                        [10, 11, 13, 14],
                        [20, 21, 0, 0],
                        [20, 22, 23, 0],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 0],
                        [1, 1, 1, 1],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                    ]
                ),
                "loss_mask": torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.5, 1.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                    ]
                ),
                "rewards": torch.tensor([1.0, 0.0, -1.0, 0.5]),
            },
            non_tensors={
                "task_ids": task_batch.non_tensor_batch["task_ids"],
                "group_ids": ["group-a", "group-a", "group-b", "group-b"],
            },
            meta_info={"algorithm": "grpo", "policy_version": 1},
        )

        self.assertEqual(len(rollout_batch), 4)
        self.assertEqual(rollout_batch.batch["input_ids"].shape, (4, 4))
        self.assertEqual(rollout_batch.batch["loss_mask"][1, 2].item(), 0.5)
        self.assertEqual(
            rollout_batch.non_tensor_batch["task_ids"].tolist(),
            ["task-a", "task-a", "task-b", "task-b"],
        )
        self.assertEqual(
            rollout_batch.non_tensor_batch["group_ids"].tolist(),
            ["group-a", "group-a", "group-b", "group-b"],
        )
        self.assertEqual(rollout_batch.meta_info["policy_version"], 1)


if __name__ == "__main__":
    unittest.main()
