import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from lmflow.agentic import (
    EpisodeWorkspace,
    EpisodeWorkspaceError,
    prepare_swe_bench_task,
    run_swe_bench_episode,
    swe_bench_prediction_from_artifact,
    verify_swe_bench_artifact,
)
from lmflow.agentic.scaffolds.mini_swe_agent import AgentConfig, LMFlowMiniSWEAgentModel

pytestmark = pytest.mark.skipif(os.name != "posix", reason="SWE-bench process verification requires POSIX")


class RecordingBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, *, messages, tools, model_name, model_kwargs):
        self.requests.append(copy.deepcopy(messages))
        return copy.deepcopy(self.responses.pop(0))


def _run_git(repo, *arguments, input_bytes=None):
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        input=input_bytes,
        check=True,
        capture_output=True,
    ).stdout


def _create_source_repo(path):
    path.mkdir()
    _run_git(path, "init", "--quiet")
    _run_git(path, "config", "user.name", "LMFlow Test")
    _run_git(path, "config", "user.email", "lmflow@example.invalid")
    (path / "value.txt").write_text("original\n", encoding="utf-8")
    _run_git(path, "add", "value.txt")
    _run_git(path, "commit", "--quiet", "-m", "Initial fixture")
    return _run_git(path, "rev-parse", "HEAD").decode("ascii").strip()


def _instance(revision):
    return {
        "repo": "example/project",
        "instance_id": "example__project-1",
        "base_commit": revision,
        "patch": "GOLD PATCH MUST NOT LEAK",
        "test_patch": "HIDDEN TEST PATCH MUST NOT LEAK",
        "problem_statement": "Change value.txt from original to fixed.",
        "hints_text": "PRIVATE HINT MUST NOT LEAK",
        "created_at": "2026-01-01T00:00:00Z",
        "version": "1.0",
        "FAIL_TO_PASS": '["test_hidden"]',
        "PASS_TO_PASS": '["test_existing"]',
        "environment_setup_commit": revision,
    }


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
        "raw_response": {"id": f"response-{call_id}", "choices": [{"message": message}]},
    }


def _agent_config():
    return AgentConfig(
        system_template="You are a software engineer with a Bash tool.",
        instance_template="Solve this task: {{task}}",
        step_limit=4,
        cost_limit=0.0,
    )


def _run_prepared_episode(tmp_path):
    source_repo = tmp_path / "source"
    revision = _create_source_repo(source_repo)
    task = prepare_swe_bench_task(_instance(revision), source_repo=source_repo)
    backend = RecordingBackend(
        [
            _response("printf 'fixed\\n' > value.txt", call_id="call-edit"),
            _response(
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'",
                call_id="call-submit",
            ),
        ]
    )
    model = LMFlowMiniSWEAgentModel(backend, model_name="fixture-model", model_kwargs={"temperature": 0.0})
    workspace_root = tmp_path / "episode-workspaces"
    verification_root = tmp_path / "verification-workspaces"
    artifact_root = tmp_path / "artifacts"
    workspace_root.mkdir()
    verification_root.mkdir()
    artifact_root.mkdir()
    artifact_dir = artifact_root / "rollout-0"
    run_swe_bench_episode(
        task=task,
        model=model,
        agent_config=_agent_config(),
        rollout_id="rollout-0",
        workspace_root=workspace_root,
        artifact_dir=artifact_dir,
    )
    return task, source_repo, artifact_dir, workspace_root, verification_root


def test_prepare_task_excludes_gold_and_grading_fields(tmp_path):
    source_repo = tmp_path / "source"
    revision = _create_source_repo(source_repo)

    task = prepare_swe_bench_task(_instance(revision.upper()), source_repo=source_repo)

    assert task.task_id == "example__project-1"
    assert task.messages == [{"role": "user", "content": "Change value.txt from original to fixed."}]
    assert task.tools == []
    assert task.environment == {
        "kind": "swe_bench",
        "repo": "example/project",
        "source_repo": str(source_repo.resolve()),
        "base_revision": revision,
    }
    assert task.metadata == {"benchmark": "swe-bench", "repo": "example/project"}
    serialized = json.dumps(asdict(task), sort_keys=True)
    for hidden_value in (
        "GOLD PATCH MUST NOT LEAK",
        "HIDDEN TEST PATCH MUST NOT LEAK",
        "PRIVATE HINT MUST NOT LEAK",
        "test_hidden",
        "test_existing",
    ):
        assert hidden_value not in serialized


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("instance_id", "", "instance_id"),
        ("repo", None, "repo"),
        ("base_commit", "abc123", "40-character"),
        ("problem_statement", "\x00", "problem_statement"),
    ],
)
def test_prepare_task_rejects_invalid_required_fields(tmp_path, field, value, error):
    source_repo = tmp_path / "source"
    revision = _create_source_repo(source_repo)
    instance = _instance(revision)
    instance[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        prepare_swe_bench_task(instance, source_repo=source_repo)


def test_episode_prediction_and_fresh_workspace_verification(tmp_path):
    task, source_repo, artifact_dir, workspace_root, verification_root = _run_prepared_episode(tmp_path)

    assert list(workspace_root.iterdir()) == []
    assert (source_repo / "value.txt").read_text(encoding="utf-8") == "original\n"
    prediction = swe_bench_prediction_from_artifact(
        task=task,
        artifact_dir=artifact_dir,
        model_name_or_path="Qwen/Qwen3-8B",
    )
    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}
    assert prediction["instance_id"] == task.task_id
    assert prediction["model_name_or_path"] == "Qwen/Qwen3-8B"
    assert "+fixed" in prediction["model_patch"]

    result = verify_swe_bench_artifact(
        task=task,
        artifact_dir=artifact_dir,
        verifier_command=(
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('value.txt').read_text() == 'fixed\\n'; print('verified')",
        ),
        workspace_root=verification_root,
    )

    assert result.returncode == 0
    assert result.timed_out is False
    assert result.stdout == "verified\n"
    assert list(verification_root.iterdir()) == []
    assert (source_repo / "value.txt").read_text(encoding="utf-8") == "original\n"


def test_prediction_export_does_not_require_the_source_repository(tmp_path):
    task, source_repo, artifact_dir, _, _ = _run_prepared_episode(tmp_path)
    detached_repo = tmp_path / "detached-source"
    source_repo.rename(detached_repo)

    prediction = swe_bench_prediction_from_artifact(
        task=task,
        artifact_dir=artifact_dir,
        model_name_or_path="Qwen/Qwen3-8B",
    )

    assert prediction["instance_id"] == task.task_id
    assert "+fixed" in prediction["model_patch"]


def test_verifier_returns_nonzero_process_result_and_cleans_workspace(tmp_path):
    task, _, artifact_dir, _, verification_root = _run_prepared_episode(tmp_path)

    result = verify_swe_bench_artifact(
        task=task,
        artifact_dir=artifact_dir,
        verifier_command=(sys.executable, "-c", "import sys; print('failed'); sys.exit(3)"),
        workspace_root=verification_root,
    )

    assert result.returncode == 3
    assert result.stdout == "failed\n"
    assert list(verification_root.iterdir()) == []


def test_artifact_identity_mismatch_is_rejected_before_verification(tmp_path):
    task, _, artifact_dir, _, verification_root = _run_prepared_episode(tmp_path)
    trajectory_path = artifact_dir / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["info"]["lmflow"]["task_id"] = "another-task"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

    with pytest.raises(ValueError, match="task_id does not match"):
        verify_swe_bench_artifact(
            task=task,
            artifact_dir=artifact_dir,
            verifier_command=(sys.executable, "-c", "raise AssertionError('must not run')"),
            workspace_root=verification_root,
        )

    assert list(verification_root.iterdir()) == []


def test_unknown_artifact_format_is_rejected_before_verification(tmp_path):
    task, _, artifact_dir, _, verification_root = _run_prepared_episode(tmp_path)
    trajectory_path = artifact_dir / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["trajectory_format"] = "future-format"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

    with pytest.raises(ValueError, match="mini-swe-agent-1.1"):
        verify_swe_bench_artifact(
            task=task,
            artifact_dir=artifact_dir,
            verifier_command=(sys.executable, "-c", "raise AssertionError('must not run')"),
            workspace_root=verification_root,
        )

    assert list(verification_root.iterdir()) == []


def test_patch_for_another_base_fails_closed_before_verifier_command(tmp_path):
    task, _, artifact_dir, _, verification_root = _run_prepared_episode(tmp_path)
    marker = tmp_path / "verifier-ran"
    (artifact_dir / "model.patch").write_text(
        "diff --git a/missing.txt b/missing.txt\n--- a/missing.txt\n+++ b/missing.txt\n@@ -1 +1 @@\n-old\n+new\n",
        encoding="utf-8",
    )

    with pytest.raises(EpisodeWorkspaceError, match="git apply failed"):
        verify_swe_bench_artifact(
            task=task,
            artifact_dir=artifact_dir,
            verifier_command=(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ),
            workspace_root=verification_root,
        )

    assert not marker.exists()
    assert list(verification_root.iterdir()) == []


def test_empty_patch_is_a_valid_noop_verification(tmp_path):
    task, _, artifact_dir, _, verification_root = _run_prepared_episode(tmp_path)
    (artifact_dir / "model.patch").write_bytes(b"")

    result = verify_swe_bench_artifact(
        task=task,
        artifact_dir=artifact_dir,
        verifier_command=(
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('value.txt').read_text() == 'original\\n'",
        ),
        workspace_root=verification_root,
    )

    assert result.returncode == 0
    assert list(verification_root.iterdir()) == []


def test_workspace_apply_patch_requires_bytes(tmp_path):
    source_repo = tmp_path / "source"
    revision = _create_source_repo(source_repo)
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()

    with pytest.raises(TypeError, match="patch must be bytes"):
        with EpisodeWorkspace.create(
            workspace_root,
            source_repo=source_repo,
            revision=revision,
            task_id="task",
            rollout_id="rollout",
        ) as workspace:
            workspace.apply_patch_bytes("not bytes")
