import copy
import json
import os
import subprocess

import pytest

from lmflow.agentic.scaffolds.mini_swe_agent import (
    AgentConfig,
    LMFlowMiniSWEAgentModel,
    run_mini_swe_agent_episode,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="ProcessSandbox currently requires POSIX")


class RecordingBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, *, messages, tools, model_name, model_kwargs):
        self.requests.append(copy.deepcopy(messages))
        return copy.deepcopy(self.responses.pop(0))


class FailingBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        raise RuntimeError("provider unavailable")


def _response(command, *, call_id):
    message = {
        "role": "assistant",
        "content": "Working on the task.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }
    return {
        "message": message,
        "finish_reason": "tool_calls",
        "cost": 0.0,
        "raw_response": {"id": f"response-{call_id}", "message": message},
    }


def _run_git(repo, *arguments):
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _create_source_repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _run_git(repo, "init", "--quiet")
    _run_git(repo, "config", "user.name", "LMFlow Test")
    _run_git(repo, "config", "user.email", "lmflow@example.invalid")
    (repo / "README.md").write_text("original\n")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "--quiet", "-m", "initial")
    return repo, _run_git(repo, "rev-parse", "HEAD")


def _agent_config(**overrides):
    config = {
        "system_template": "You are a software engineer with a Bash tool.",
        "instance_template": "Solve this task: {{task}}",
        "step_limit": 4,
        "cost_limit": 0.0,
    }
    config.update(overrides)
    return AgentConfig(**config)


def _run_episode(tmp_path, backend, *, artifact_name="episode-output", agent_config=None):
    source_repo, base_revision = _create_source_repo(tmp_path)
    workspace_root = tmp_path / "workspaces"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    artifact_root.mkdir()
    artifact_dir = artifact_root / artifact_name
    model = LMFlowMiniSWEAgentModel(backend, model_name="fixture-model", model_kwargs={"temperature": 0.0})
    result = run_mini_swe_agent_episode(
        model=model,
        agent_config=agent_config or _agent_config(),
        task="Update README.md",
        task_id="task-1",
        rollout_id="rollout-0",
        source_repo=source_repo,
        revision=base_revision,
        workspace_root=workspace_root,
        artifact_dir=artifact_dir,
    )
    return result, source_repo, base_revision, workspace_root, artifact_root


def test_episode_runner_publishes_raw_trajectory_and_patch_then_cleans_workspace(tmp_path):
    backend = RecordingBackend(
        [
            _response(
                "printf 'changed\\n' > README.md; printf '\\000\\377\\001' > blob.bin",
                call_id="call-edit",
            ),
            _response(
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'",
                call_id="call-submit",
            ),
        ]
    )

    artifact_dir, source_repo, base_revision, workspace_root, artifact_root = _run_episode(tmp_path, backend)

    assert artifact_dir == (artifact_root / "episode-output").resolve()
    assert sorted(path.name for path in artifact_dir.iterdir()) == ["model.patch", "trajectory.json"]
    assert list(workspace_root.iterdir()) == []
    assert [path.name for path in artifact_root.iterdir()] == ["episode-output"]
    assert (source_repo / "README.md").read_text() == "original\n"

    trajectory = json.loads((artifact_dir / "trajectory.json").read_text())
    assert trajectory["trajectory_format"] == "mini-swe-agent-1.1"
    assert trajectory["info"]["exit_status"] == "Submitted"
    assert trajectory["info"]["submission"] == "done\n"
    assert trajectory["info"]["lmflow"] == {
        "task_id": "task-1",
        "rollout_id": "rollout-0",
        "task": "Update README.md",
        "base_revision": base_revision,
        "scaffold_commit": "a83fcae82d2a08f0ee0c688f9d137b3566c097f8",
    }
    assert trajectory["info"]["config"]["sandbox"]["timeout_seconds"] == 60.0
    assert trajectory["info"]["config"]["sandbox"]["capabilities"]["filesystem_isolation"] is False
    first_assistant = next(message for message in trajectory["messages"] if message["role"] == "assistant")
    assert first_assistant["extra"]["response"]["id"] == "response-call-edit"

    patch = (artifact_dir / "model.patch").read_text()
    assert "diff --git a/README.md b/README.md" in patch
    assert "+changed" in patch
    assert "diff --git a/blob.bin b/blob.bin" in patch
    assert "GIT binary patch" in patch


def test_episode_runner_publishes_failure_artifacts_and_reraises(tmp_path):
    backend = FailingBackend()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        _run_episode(tmp_path, backend)

    artifact_dir = tmp_path / "artifacts" / "episode-output"
    trajectory = json.loads((artifact_dir / "trajectory.json").read_text())
    assert trajectory["info"]["exit_status"] == "RuntimeError"
    assert trajectory["messages"][-1]["role"] == "exit"
    assert trajectory["messages"][-1]["extra"]["exception_str"] == "provider unavailable"
    assert (artifact_dir / "model.patch").read_bytes() == b""
    assert list((tmp_path / "workspaces").iterdir()) == []
    assert backend.calls == 1


def test_episode_runner_rejects_existing_artifact_directory_before_starting_workspace(tmp_path):
    source_repo, base_revision = _create_source_repo(tmp_path)
    workspace_root = tmp_path / "workspaces"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    artifact_root.mkdir()
    artifact_dir = artifact_root / "episode-output"
    artifact_dir.mkdir()
    backend = FailingBackend()
    model = LMFlowMiniSWEAgentModel(backend, model_name="fixture-model")

    with pytest.raises(FileExistsError, match="artifact_dir already exists"):
        run_mini_swe_agent_episode(
            model=model,
            agent_config=_agent_config(),
            task="Update README.md",
            task_id="task-1",
            rollout_id="rollout-0",
            source_repo=source_repo,
            revision=base_revision,
            workspace_root=workspace_root,
            artifact_dir=artifact_dir,
        )

    assert list(workspace_root.iterdir()) == []
    assert backend.calls == 0
