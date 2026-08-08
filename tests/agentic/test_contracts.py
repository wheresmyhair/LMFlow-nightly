import unittest

from lmflow.agentic import TaskSpec, build_task_batch


class TaskSpecTest(unittest.TestCase):
    def test_exposes_normalized_task_fields(self):
        task = TaskSpec(
            task_id="django__django-123",
            messages=[{"role": "user", "content": "Fix the failing test."}],
            tools=[{"name": "bash"}],
            environment={"repository": "django/django"},
            metadata={"split": "dev"},
        )

        self.assertEqual(task.task_id, "django__django-123")
        self.assertEqual(task.messages[0]["role"], "user")
        self.assertEqual(task.tools, [{"name": "bash"}])
        self.assertEqual(task.environment["repository"], "django/django")
        self.assertEqual(task.metadata["split"], "dev")

    def test_uses_independent_default_containers(self):
        first = TaskSpec(task_id="first", messages=[])
        second = TaskSpec(task_id="second", messages=[])

        first.metadata["source"] = "test"

        self.assertEqual(second.metadata, {})

    def test_batches_variable_task_payloads_without_expanding_them(self):
        first = TaskSpec(
            task_id="first",
            messages=[{"role": "user", "content": "First task"}],
        )
        second = TaskSpec(
            task_id="second",
            messages=[
                {"role": "system", "content": "Use bash"},
                {"role": "user", "content": "Second task"},
            ],
            tools=[{"name": "bash"}],
        )

        batch = build_task_batch(task for task in (first, second))

        self.assertIsNone(batch.batch)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch.non_tensor_batch["tasks"].shape, (2,))
        self.assertIs(batch.non_tensor_batch["tasks"][0], first)
        self.assertIs(batch.non_tensor_batch["tasks"][1], second)
        self.assertEqual(batch.non_tensor_batch["task_ids"].tolist(), ["first", "second"])

    def test_builds_an_empty_task_batch(self):
        batch = build_task_batch([])

        self.assertEqual(len(batch), 0)
        self.assertEqual(batch.non_tensor_batch["tasks"].shape, (0,))
        self.assertEqual(batch.non_tensor_batch["task_ids"].shape, (0,))


if __name__ == "__main__":
    unittest.main()
