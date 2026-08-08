"""Minimal task input contract for Agentic workflows."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmflow.utils.protocol import DataProto


@dataclass
class TaskSpec:
    """A normalized task passed from a dataset adapter to an agent."""

    task_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_task_batch(tasks: Iterable[TaskSpec]) -> "DataProto":
    """Batch normalized tasks for the dataset-adapter-to-agent handoff."""
    import numpy as np

    from lmflow.utils.protocol import DataProto

    task_list = list(tasks)
    task_objects = np.empty(len(task_list), dtype=object)
    task_objects[:] = task_list

    return DataProto.from_dict(
        non_tensors={
            "tasks": task_objects,
            "task_ids": [task.task_id for task in task_list],
        }
    )
