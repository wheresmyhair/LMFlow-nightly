import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lmflow.agentic import TaskSpec, run_synchronous_grpo_step
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


class _TinyPolicyTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.data = None

    def train_step(self, data):
        self.data = data
        self.optimizer.zero_grad(set_to_none=True)
        loss = grpo_loss_from_model(self.model, data)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.detach().item(),
            "selected_tokens": (data.batch["loss_mask"] > 0).sum().item(),
        }


class _DeterministicRollout:
    def __init__(self, model):
        self.model = model
        self.requests = None
        self.input_ids = None
        self.attention_mask = None
        self.loss_mask = None
        self.old_log_probs = None

    def __call__(self, requests):
        self.requests = requests
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 5],
                [1, 2, 4, 5],
                [1, 2, 4, 6],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ]
        )
        loss_mask = torch.tensor(
            [
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        with torch.no_grad():
            old_log_probs = causal_token_log_probs(
                self.model(input_ids, attention_mask).logits,
                input_ids,
            )

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.loss_mask = loss_mask
        self.old_log_probs = old_log_probs

        order = np.array([3, 1, 2, 0])
        order_tensor = torch.from_numpy(order)
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids[order_tensor],
                "attention_mask": attention_mask[order_tensor],
                "loss_mask": loss_mask[order_tensor],
                "old_log_probs": old_log_probs[order_tensor],
            },
            non_tensors={key: values[order] for key, values in requests.non_tensor_batch.items()},
            meta_info=dict(requests.meta_info),
        )


class SynchronousGRPOControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = _TinyCausalLM()
        self.trainer = _TinyPolicyTrainer(self.model)
        self.tasks = [
            TaskSpec(task_id="task-a", messages=[{"role": "user", "content": "A"}]),
            TaskSpec(task_id="task-b", messages=[{"role": "user", "content": "B"}]),
        ]

    def test_runs_task_to_rollout_to_reward_to_real_optimizer_update(self):
        rollout = _DeterministicRollout(self.model)
        parameters_before = [parameter.detach().clone() for parameter in self.model.parameters()]

        def reward_fn(data):
            reward_by_rollout_id = torch.tensor([1.0, 3.0, 2.0, 6.0], dtype=torch.float64)
            rollout_ids = torch.from_numpy(data.non_tensor_batch["rollout_ids"].astype(np.int64))
            return reward_by_rollout_id[rollout_ids]

        metrics = run_synchronous_grpo_step(
            self.trainer,
            self.tasks,
            rollout_fn=rollout,
            reward_fn=reward_fn,
            group_size=2,
            policy_version=11,
        )

        self.assertEqual(
            rollout.requests.non_tensor_batch["task_ids"].tolist(),
            ["task-a", "task-a", "task-b", "task-b"],
        )
        self.assertEqual(rollout.requests.non_tensor_batch["group_ids"].tolist(), [0, 0, 1, 1])
        self.assertEqual(rollout.requests.non_tensor_batch["rollout_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(rollout.requests.meta_info["policy_version"], 11)

        trained = self.trainer.data
        self.assertEqual(trained.non_tensor_batch["rollout_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(trained.non_tensor_batch["group_ids"].tolist(), [0, 0, 1, 1])
        self.assertEqual(trained.non_tensor_batch["task_ids"].tolist(), ["task-a", "task-a", "task-b", "task-b"])
        torch.testing.assert_close(trained.batch["input_ids"], rollout.input_ids)
        torch.testing.assert_close(trained.batch["attention_mask"], rollout.attention_mask)
        torch.testing.assert_close(trained.batch["loss_mask"], rollout.loss_mask)
        torch.testing.assert_close(trained.batch["old_log_probs"], rollout.old_log_probs)
        torch.testing.assert_close(
            trained.batch["advantages"],
            torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertEqual(metrics["train/tokens"], 7.0)
        self.assertEqual(metrics["rollout/groups"], 2.0)
        self.assertEqual(metrics["rollout/groups_completed"], 2.0)
        self.assertTrue(
            any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(self.model.parameters(), parameters_before)
            )
        )

    def test_rejects_singleton_groups_before_rollout(self):
        with self.assertRaisesRegex(ValueError, "at least 2 for GRPO"):
            run_synchronous_grpo_step(
                self.trainer,
                self.tasks,
                rollout_fn=lambda requests: self.fail("rollout must not be called"),
                reward_fn=lambda data: self.fail("reward must not be called"),
                group_size=1,
                policy_version=11,
            )

        self.assertIsNone(self.trainer.data)

    def test_missing_rollout_task_ids_fail_before_reward_or_update(self):
        rollout = _DeterministicRollout(self.model)

        def missing_task_ids(requests):
            data = rollout(requests)
            del data.non_tensor_batch["task_ids"]
            return data

        with self.assertRaisesRegex(KeyError, "task_ids"):
            run_synchronous_grpo_step(
                self.trainer,
                self.tasks,
                rollout_fn=missing_task_ids,
                reward_fn=lambda data: self.fail("reward must not be called"),
                group_size=2,
                policy_version=11,
            )

        self.assertIsNone(self.trainer.data)

    def test_mismatched_rollout_task_ids_fail_before_reward_or_update(self):
        rollout = _DeterministicRollout(self.model)

        def mismatched_task_ids(requests):
            data = rollout(requests)
            data.non_tensor_batch["task_ids"] = data.non_tensor_batch["task_ids"].copy()
            data.non_tensor_batch["task_ids"][0] = "wrong-task"
            return data

        with self.assertRaisesRegex(ValueError, "task_id mismatch for rollout_id"):
            run_synchronous_grpo_step(
                self.trainer,
                self.tasks,
                rollout_fn=mismatched_task_ids,
                reward_fn=lambda data: self.fail("reward must not be called"),
                group_size=2,
                policy_version=11,
            )

        self.assertIsNone(self.trainer.data)

    def test_incomplete_preselected_group_fails_before_reward_or_update(self):
        reward_called = False

        def partial_rollout(requests):
            return requests[:-1]

        def reward_fn(data):
            nonlocal reward_called
            reward_called = True
            return torch.zeros(len(data))

        with self.assertRaisesRegex(ValueError, "incomplete groups"):
            run_synchronous_grpo_step(
                self.trainer,
                self.tasks,
                rollout_fn=partial_rollout,
                reward_fn=reward_fn,
                group_size=2,
                policy_version=11,
            )

        self.assertFalse(reward_called)
        self.assertIsNone(self.trainer.data)

    def test_expired_collection_fails_before_reward_or_update(self):
        timestamps = iter([10.0, 16.0])

        with self.assertRaisesRegex(TimeoutError, "exceeded 5.0 seconds"):
            run_synchronous_grpo_step(
                self.trainer,
                self.tasks,
                rollout_fn=lambda requests: requests,
                reward_fn=lambda data: torch.zeros(len(data)),
                group_size=2,
                policy_version=11,
                group_timeout_seconds=5.0,
                clock=lambda: next(timestamps),
            )

        self.assertIsNone(self.trainer.data)

    def test_nonfinite_rewards_fail_before_update(self):
        rollout = _DeterministicRollout(self.model)

        with self.assertRaisesRegex(ValueError, "finite"):
            run_synchronous_grpo_step(
                self.trainer,
                self.tasks,
                rollout_fn=rollout,
                reward_fn=lambda data: torch.tensor([1.0, float("nan"), 2.0, 3.0]),
                group_size=2,
                policy_version=11,
            )

        self.assertIsNone(self.trainer.data)


if __name__ == "__main__":
    unittest.main()
