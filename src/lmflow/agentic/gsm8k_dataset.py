"""Batch data generation for GSM8K reward-tool trajectories."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lmflow.agentic.atif import atif_trajectory_to_conversation
from lmflow.agentic.completion import CompletionBackend
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.gsm8k_episode import run_gsm8k_tool_episode

PathLike = str | os.PathLike[str]

_DATASET_FILENAME = "data.json"
_RAW_TRAJECTORIES_FILENAME = "trajectories.jsonl"
_REPORT_FILENAME = "report.json"


def _resolve_new_artifact_dir(artifact_dir: PathLike) -> Path:
    path = Path(artifact_dir)
    if not path.name:
        raise ValueError("artifact_dir must name a new directory")
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("artifact_dir parent must be an existing directory") from error
    if not parent.is_dir():
        raise ValueError("artifact_dir parent must be an existing directory")
    target = parent / path.name
    if os.path.lexists(target):
        raise FileExistsError(f"artifact_dir already exists: {target}")
    return target


def _validate_tasks(tasks: Iterable[TaskSpec]) -> list[TaskSpec]:
    task_list = list(tasks)
    if not task_list:
        raise ValueError("tasks must contain at least one TaskSpec")

    seen_task_ids = set()
    for index, task in enumerate(task_list):
        if not isinstance(task, TaskSpec):
            raise TypeError(f"tasks[{index}] must be a TaskSpec")
        if not isinstance(task.task_id, str) or not task.task_id.strip():
            raise ValueError(f"tasks[{index}].task_id must be a non-empty string")
        if task.task_id in seen_task_ids:
            raise ValueError(f"tasks contains duplicate task_id {task.task_id!r}")
        seen_task_ids.add(task.task_id)
    return task_list


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            value,
            output_file,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def generate_gsm8k_tool_dataset(
    backend: CompletionBackend,
    tasks: Iterable[TaskSpec],
    *,
    artifact_dir: PathLike,
    model_name: str,
    session_id: str,
    model_kwargs: Mapping[str, Any] | None = None,
    rollouts_per_task: int = 1,
    max_steps: int = 4,
) -> dict[str, Any]:
    """Generate GSM8K-tool trajectories and an SFT conversation dataset.

    Episodes run synchronously in task order and then rollout-index order. Raw
    ATIF trajectories retain every completed episode, while only trajectories
    with binary reward ``1.0`` enter ``dataset/data.json``. The returned report
    is also stored as ``report.json`` next to ``trajectories.jsonl``.

    The complete artifact directory is published atomically after every episode
    and ATIF conversion succeeds. Any exception leaves no partial artifact. The
    caller must pass the nested ``dataset`` directory, rather than the artifact
    root, to LMFlow's Dataset loader.
    """

    task_list = _validate_tasks(tasks)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping")
    if isinstance(rollouts_per_task, bool) or not isinstance(rollouts_per_task, int):
        raise TypeError("rollouts_per_task must be an integer")
    if rollouts_per_task < 1:
        raise ValueError("rollouts_per_task must be positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    target_dir = _resolve_new_artifact_dir(artifact_dir)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.",
            suffix=".tmp",
            dir=target_dir.parent,
        )
    )
    published = False
    trajectory_count = 0
    successful_count = 0
    completion_cost = 0.0
    model_steps = 0
    reward_tool_calls = 0

    try:
        dataset_dir = staging_dir / "dataset"
        dataset_dir.mkdir()
        raw_path = staging_dir / _RAW_TRAJECTORIES_FILENAME
        dataset_path = dataset_dir / _DATASET_FILENAME

        with (
            raw_path.open("x", encoding="utf-8", newline="\n") as raw_file,
            dataset_path.open("x", encoding="utf-8", newline="\n") as dataset_file,
        ):
            dataset_file.write('{"type":"conversation","instances":[')
            for task in task_list:
                for rollout_index in range(rollouts_per_task):
                    trajectory_id = f"{session_id}:{task.task_id}:rollout-{rollout_index}"
                    try:
                        trajectory = run_gsm8k_tool_episode(
                            backend,
                            task,
                            model_name=model_name,
                            trajectory_id=trajectory_id,
                            model_kwargs=copy.deepcopy(dict(model_kwargs)),
                            session_id=session_id,
                            max_steps=max_steps,
                        )
                        conversation = atif_trajectory_to_conversation(trajectory)
                    except Exception as error:
                        error.add_note(f"GSM8K task_id={task.task_id!r}, rollout_index={rollout_index}")
                        raise

                    json.dump(
                        trajectory,
                        raw_file,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    raw_file.write("\n")

                    final_metrics = trajectory["final_metrics"]
                    reward = final_metrics["reward"]
                    if reward == 1.0:
                        if successful_count:
                            dataset_file.write(",")
                        json.dump(
                            conversation,
                            dataset_file,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        successful_count += 1

                    trajectory_count += 1
                    completion_cost += final_metrics["completion_cost"]
                    model_steps += final_metrics["model_steps"]
                    reward_tool_calls += final_metrics["reward_tool_calls"]

            dataset_file.write("]}\n")
            raw_file.flush()
            dataset_file.flush()
            os.fsync(raw_file.fileno())
            os.fsync(dataset_file.fileno())

        report = {
            "session_id": session_id,
            "model_name": model_name,
            "task_count": len(task_list),
            "rollouts_per_task": rollouts_per_task,
            "trajectory_count": trajectory_count,
            "successful_trajectory_count": successful_count,
            "conversation_count": successful_count,
            "success_rate": successful_count / trajectory_count,
            "completion_cost": completion_cost,
            "model_steps": model_steps,
            "reward_tool_calls": reward_tool_calls,
        }
        _write_json(staging_dir / _REPORT_FILENAME, report)

        try:
            staging_dir.rename(target_dir)
        except FileExistsError as error:
            raise FileExistsError(f"artifact_dir already exists: {target_dir}") from error
        published = True
        return copy.deepcopy(report)
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)


__all__ = ["generate_gsm8k_tool_dataset"]
