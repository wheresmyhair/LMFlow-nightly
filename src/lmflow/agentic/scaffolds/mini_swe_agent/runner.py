"""One-episode orchestration for the vendored mini-swe-agent scaffold."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from lmflow.agentic.sandbox import ProcessLimits, ProcessSandbox
from lmflow.agentic.scaffolds.mini_swe_agent._vendor import UPSTREAM_COMMIT, Model
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.agent import AgentConfig, DefaultAgent
from lmflow.agentic.scaffolds.mini_swe_agent.adapters import ProcessSandboxEnvironment
from lmflow.agentic.workspace import EpisodeWorkspace

PathLike = str | os.PathLike[str]

_PATCH_FILENAME = "model.patch"
_TRAJECTORY_FILENAME = "trajectory.json"
_TRAJECTORY_FORMAT = "mini-swe-agent-1.1"


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


def _resolve_existing_artifact_dir(artifact_dir: PathLike) -> Path:
    try:
        path = Path(artifact_dir).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("artifact_dir must be an existing directory") from error
    if not path.is_dir():
        raise ValueError("artifact_dir must be an existing directory")
    return path


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def load_mini_swe_agent_artifact(artifact_dir: PathLike) -> tuple[dict[str, Any], bytes]:
    """Load one exact raw trajectory/patch pair published by the runner."""
    artifact = _resolve_existing_artifact_dir(artifact_dir)
    patch_path = artifact / _PATCH_FILENAME
    trajectory_path = artifact / _TRAJECTORY_FILENAME
    if patch_path.is_symlink() or trajectory_path.is_symlink():
        raise ValueError("episode artifact files must not be symlinks")
    if not patch_path.is_file() or not trajectory_path.is_file():
        raise ValueError("artifact_dir must contain model.patch and trajectory.json files")
    try:
        trajectory = json.loads(
            trajectory_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("trajectory.json must contain strict UTF-8 JSON") from error
    if not isinstance(trajectory, Mapping) or trajectory.get("trajectory_format") != _TRAJECTORY_FORMAT:
        raise ValueError(f"trajectory.json must use {_TRAJECTORY_FORMAT!r}")
    try:
        patch = patch_path.read_bytes()
    except OSError as error:
        raise ValueError("failed to read episode model.patch") from error
    return dict(trajectory), patch


def _write_artifact(path: Path, content: bytes) -> None:
    with path.open("xb") as artifact_file:
        artifact_file.write(content)
        artifact_file.flush()
        os.fsync(artifact_file.fileno())


def _publish_episode_artifacts(
    *,
    agent: DefaultAgent,
    workspace: EpisodeWorkspace,
    staging_dir: Path,
    artifact_dir: Path,
    task: str,
) -> None:
    trajectory = agent.serialize(
        {
            "info": {
                "lmflow": {
                    "task_id": workspace.task_id,
                    "rollout_id": workspace.rollout_id,
                    "task": task,
                    "base_revision": workspace.base_revision,
                    "scaffold_commit": UPSTREAM_COMMIT,
                }
            }
        }
    )
    trajectory_bytes = json.dumps(trajectory, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n"
    patch_bytes = workspace.export_patch_bytes()
    _write_artifact(staging_dir / _TRAJECTORY_FILENAME, trajectory_bytes)
    _write_artifact(staging_dir / _PATCH_FILENAME, patch_bytes)
    try:
        staging_dir.rename(artifact_dir)
    except FileExistsError as error:
        raise FileExistsError(f"artifact_dir already exists: {artifact_dir}") from error


def run_mini_swe_agent_episode(
    *,
    model: Model,
    agent_config: AgentConfig,
    task: str,
    task_id: str,
    rollout_id: str,
    source_repo: PathLike,
    revision: str,
    workspace_root: PathLike,
    artifact_dir: PathLike,
    sandbox_timeout_seconds: float = 60.0,
    sandbox_max_output_bytes: int = 1_000_000,
    sandbox_limits: ProcessLimits | None = None,
    required_sandbox_capabilities: Iterable[str] = (),
    git_timeout_seconds: float = 120.0,
) -> Path:
    """Run one mini-swe-agent episode and publish its raw artifacts.

    The caller owns task ingestion, repository preparation, model/provider
    construction, and later verification. This function owns only the
    per-attempt workspace, process environment, scaffold execution, raw
    trajectory, patch export, and cleanup.

    ``artifact_dir`` must not already exist. Successful episodes and ordinary
    Python exceptions from the scaffold both publish ``trajectory.json`` and
    ``model.patch`` atomically as one directory. Exceptions are re-raised after
    publication so infrastructure failures cannot be mistaken for completed
    rollouts.
    """

    if not isinstance(agent_config, AgentConfig):
        raise TypeError("agent_config must be an AgentConfig instance")
    if agent_config.output_path is not None:
        raise ValueError("agent_config.output_path must be None because the episode runner owns artifact output")
    if not isinstance(task, str):
        raise TypeError("task must be a string")

    target_dir = _resolve_new_artifact_dir(artifact_dir)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.",
            suffix=".tmp",
            dir=target_dir.parent,
        )
    )
    published = False
    try:
        with EpisodeWorkspace.create(
            workspace_root,
            source_repo=source_repo,
            revision=revision,
            task_id=task_id,
            rollout_id=rollout_id,
            git_timeout_seconds=git_timeout_seconds,
        ) as workspace:
            sandbox = ProcessSandbox(
                workspace.path,
                timeout_seconds=sandbox_timeout_seconds,
                max_output_bytes=sandbox_max_output_bytes,
                limits=sandbox_limits,
                required_capabilities=required_sandbox_capabilities,
            )
            environment = ProcessSandboxEnvironment(sandbox, cwd=".")
            agent = DefaultAgent(
                model=model,
                env=environment,
                **agent_config.model_dump(),
            )
            try:
                agent.run(task)
            except Exception:
                _publish_episode_artifacts(
                    agent=agent,
                    workspace=workspace,
                    staging_dir=staging_dir,
                    artifact_dir=target_dir,
                    task=task,
                )
                published = True
                raise
            else:
                _publish_episode_artifacts(
                    agent=agent,
                    workspace=workspace,
                    staging_dir=staging_dir,
                    artifact_dir=target_dir,
                    task=task,
                )
                published = True
        return target_dir
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)


__all__ = ["load_mini_swe_agent_artifact", "run_mini_swe_agent_episode"]
