"""Strict SWE-bench task, artifact, and verification adapters."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.sandbox import ProcessLimits, ProcessResult, ProcessSandbox
from lmflow.agentic.scaffolds.mini_swe_agent._vendor import Model
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.agent import AgentConfig
from lmflow.agentic.scaffolds.mini_swe_agent.runner import run_mini_swe_agent_episode
from lmflow.agentic.workspace import EpisodeWorkspace

PathLike = str | os.PathLike[str]

_ENVIRONMENT_KIND = "swe_bench"
_FULL_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_PATCH_FILENAME = "model.patch"
_TRAJECTORY_FILENAME = "trajectory.json"
_TRAJECTORY_FORMAT = "mini-swe-agent-1.1"


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"SWE-bench field {key!r} must be a string")
    if not value.strip() or "\0" in value:
        raise ValueError(f"SWE-bench field {key!r} must be non-empty and must not contain NUL bytes")
    return value


def _resolve_directory(path: PathLike, *, name: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError(f"{name} must be an existing directory") from error
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return resolved


def prepare_swe_bench_task(instance: Mapping[str, Any], *, source_repo: PathLike) -> TaskSpec:
    """Project one official SWE-bench instance into the existing TaskSpec.

    Only model-visible task input and repository identity are copied. Golden
    patches, test patches, hints, and F2P/P2P grading fields remain outside the
    task passed to the scaffold.
    """
    if not isinstance(instance, Mapping):
        raise TypeError("instance must be a mapping")
    instance_id = _required_string(instance, "instance_id")
    repo = _required_string(instance, "repo")
    base_commit = _required_string(instance, "base_commit")
    if _FULL_COMMIT_PATTERN.fullmatch(base_commit) is None:
        raise ValueError("SWE-bench field 'base_commit' must be a full 40-character Git commit")
    problem_statement = _required_string(instance, "problem_statement")
    source = _resolve_directory(source_repo, name="source_repo")

    return TaskSpec(
        task_id=instance_id,
        messages=[{"role": "user", "content": problem_statement}],
        environment={
            "kind": _ENVIRONMENT_KIND,
            "repo": repo,
            "source_repo": str(source),
            "base_revision": base_commit.lower(),
        },
        metadata={"benchmark": "swe-bench", "repo": repo},
    )


def _task_identity(task: TaskSpec) -> tuple[str, str]:
    if not isinstance(task, TaskSpec):
        raise TypeError("task must be a TaskSpec")
    if not isinstance(task.task_id, str) or not task.task_id or "\0" in task.task_id:
        raise ValueError("task.task_id must be a non-empty string without NUL bytes")
    if not isinstance(task.environment, Mapping) or task.environment.get("kind") != _ENVIRONMENT_KIND:
        raise ValueError("task.environment must describe a prepared SWE-bench task")
    base_revision = task.environment.get("base_revision")
    if not isinstance(base_revision, str) or _FULL_COMMIT_PATTERN.fullmatch(base_revision) is None:
        raise ValueError("task.environment.base_revision must be a full 40-character Git commit")
    if len(task.messages) != 1 or not isinstance(task.messages[0], Mapping):
        raise ValueError("prepared SWE-bench task must contain exactly one user message")
    message = task.messages[0]
    if message.get("role") != "user":
        raise ValueError("prepared SWE-bench task message role must be 'user'")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or "\0" in content:
        raise ValueError("prepared SWE-bench task message content must be a non-empty string")
    return content, base_revision


def _task_runtime(task: TaskSpec) -> tuple[str, Path, str]:
    content, base_revision = _task_identity(task)
    source_repo = _resolve_directory(task.environment.get("source_repo"), name="task source_repo")
    return content, source_repo, base_revision


def run_swe_bench_episode(
    *,
    task: TaskSpec,
    model: Model,
    agent_config: AgentConfig,
    rollout_id: str,
    workspace_root: PathLike,
    artifact_dir: PathLike,
    sandbox_timeout_seconds: float = 60.0,
    sandbox_max_output_bytes: int = 1_000_000,
    sandbox_limits: ProcessLimits | None = None,
    required_sandbox_capabilities: Iterable[str] = (),
    git_timeout_seconds: float = 120.0,
) -> Path:
    """Run one prepared SWE-bench task through the existing episode runner."""
    task_text, source_repo, base_revision = _task_runtime(task)
    return run_mini_swe_agent_episode(
        model=model,
        agent_config=agent_config,
        task=task_text,
        task_id=task.task_id,
        rollout_id=rollout_id,
        source_repo=source_repo,
        revision=base_revision,
        workspace_root=workspace_root,
        artifact_dir=artifact_dir,
        sandbox_timeout_seconds=sandbox_timeout_seconds,
        sandbox_max_output_bytes=sandbox_max_output_bytes,
        sandbox_limits=sandbox_limits,
        required_sandbox_capabilities=required_sandbox_capabilities,
        git_timeout_seconds=git_timeout_seconds,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _load_episode_artifact(task: TaskSpec, artifact_dir: PathLike) -> tuple[bytes, str]:
    task_text, base_revision = _task_identity(task)
    artifact = _resolve_directory(artifact_dir, name="artifact_dir")
    patch_path = artifact / _PATCH_FILENAME
    trajectory_path = artifact / _TRAJECTORY_FILENAME
    if patch_path.is_symlink() or trajectory_path.is_symlink():
        raise ValueError("episode artifact files must not be symlinks")
    if not patch_path.is_file() or not trajectory_path.is_file():
        raise ValueError("artifact_dir must contain model.patch and trajectory.json files")
    try:
        trajectory = json.loads(
            trajectory_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("trajectory.json must contain strict UTF-8 JSON") from error
    if not isinstance(trajectory, Mapping) or trajectory.get("trajectory_format") != _TRAJECTORY_FORMAT:
        raise ValueError(f"trajectory.json must use {_TRAJECTORY_FORMAT!r}")
    try:
        lmflow_info = trajectory["info"]["lmflow"]
    except (KeyError, TypeError) as error:
        raise ValueError("trajectory.json is missing LMFlow episode identity") from error
    if not isinstance(lmflow_info, Mapping):
        raise ValueError("trajectory.json LMFlow episode identity must be a mapping")
    expected_identity = {
        "task_id": task.task_id,
        "task": task_text,
        "base_revision": base_revision,
    }
    for key, expected in expected_identity.items():
        if lmflow_info.get(key) != expected:
            raise ValueError(f"episode artifact {key} does not match the prepared task")
    rollout_id = lmflow_info.get("rollout_id")
    if not isinstance(rollout_id, str) or not rollout_id or "\0" in rollout_id:
        raise ValueError("episode artifact rollout_id must be a non-empty string")
    try:
        patch = patch_path.read_bytes()
    except OSError as error:
        raise ValueError("failed to read episode model.patch") from error
    return patch, rollout_id


def swe_bench_prediction_from_artifact(
    *,
    task: TaskSpec,
    artifact_dir: PathLike,
    model_name_or_path: str,
) -> dict[str, str]:
    """Build the three-field prediction consumed by the official harness."""
    if not isinstance(model_name_or_path, str) or not model_name_or_path.strip() or "\0" in model_name_or_path:
        raise ValueError("model_name_or_path must be a non-empty string without NUL bytes")
    patch, _ = _load_episode_artifact(task, artifact_dir)
    try:
        model_patch = patch.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("official SWE-bench prediction JSON requires a UTF-8 model patch") from error
    return {
        "instance_id": task.task_id,
        "model_name_or_path": model_name_or_path,
        "model_patch": model_patch,
    }


def verify_swe_bench_artifact(
    *,
    task: TaskSpec,
    artifact_dir: PathLike,
    verifier_command: Sequence[str],
    workspace_root: PathLike,
    verifier_env: Mapping[str, str] | None = None,
    sandbox_timeout_seconds: float = 900.0,
    sandbox_max_output_bytes: int = 4_000_000,
    sandbox_limits: ProcessLimits | None = None,
    required_sandbox_capabilities: Iterable[str] = (),
    git_timeout_seconds: float = 120.0,
) -> ProcessResult:
    """Replay an episode patch in a fresh checkout and run one verifier argv.

    A zero return code is a successful verifier command. Benchmark-specific
    F2P/P2P parsing and final resolved status remain the responsibility of the
    official SWE-bench harness or another explicit verifier adapter.
    """
    _, source_repo, base_revision = _task_runtime(task)
    patch, rollout_id = _load_episode_artifact(task, artifact_dir)
    with EpisodeWorkspace.create(
        workspace_root,
        source_repo=source_repo,
        revision=base_revision,
        task_id=task.task_id,
        rollout_id=f"{rollout_id}:verification",
        git_timeout_seconds=git_timeout_seconds,
    ) as workspace:
        workspace.apply_patch_bytes(patch)
        sandbox = ProcessSandbox(
            workspace.path,
            timeout_seconds=sandbox_timeout_seconds,
            max_output_bytes=sandbox_max_output_bytes,
            limits=sandbox_limits,
            required_capabilities=required_sandbox_capabilities,
        )
        return sandbox.run(verifier_command, env=verifier_env)


__all__ = [
    "prepare_swe_bench_task",
    "run_swe_bench_episode",
    "swe_bench_prediction_from_artifact",
    "verify_swe_bench_artifact",
]
