import shutil
import subprocess
import sys

import pytest

from lmflow.agentic import EpisodeWorkspace, EpisodeWorkspaceError, ProcessSandbox

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for episode workspace tests")


def _git(repo, *arguments, input_bytes=None):
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        input=input_bytes,
        capture_output=True,
        check=True,
    ).stdout


def _create_source_repo(path):
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "LMFlow Test")
    _git(path, "config", "user.email", "lmflow-test@example.com")
    (path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (path / "deleted.txt").write_text("keep me\n", encoding="utf-8")
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "Initial fixture")
    return _git(path, "rev-parse", "HEAD").decode("ascii").strip()


def test_create_uses_committed_source_and_safe_unique_identity(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)
    (source / "tracked.txt").write_text("uncommitted source change\n", encoding="utf-8")

    first = EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="../../unsafe task",
        rollout_id="attempt/one",
    )
    second = EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="../../unsafe task",
        rollout_id="attempt/one",
    )
    try:
        assert first.path != second.path
        assert first.episode_dir.parent == storage
        assert second.episode_dir.parent == storage
        assert "unsafe" not in first.episode_dir.name
        assert "attempt" not in first.episode_dir.name
        assert (first.path / "tracked.txt").read_text(encoding="utf-8") == "baseline\n"
        assert _git(first.path, "rev-parse", "HEAD").decode("ascii").strip() == revision
        assert first.base_revision == revision
        assert first.active is True
    finally:
        first.cleanup()
        second.cleanup()

    assert (source / "tracked.txt").read_text(encoding="utf-8") == "uncommitted source change\n"
    assert list(storage.iterdir()) == []


def test_reset_restores_baseline_and_removes_untracked_and_ignored_files(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)

    with EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-1",
        rollout_id="rollout-1",
    ) as workspace:
        (workspace.path / "tracked.txt").write_text("changed\n", encoding="utf-8")
        (workspace.path / "deleted.txt").unlink()
        (workspace.path / "untracked.txt").write_text("temporary\n", encoding="utf-8")
        (workspace.path / "ignored.txt").write_text("cached output\n", encoding="utf-8")
        shutil.rmtree(workspace.path / ".git")

        workspace.reset()

        assert (workspace.path / "tracked.txt").read_text(encoding="utf-8") == "baseline\n"
        assert (workspace.path / "deleted.txt").read_text(encoding="utf-8") == "keep me\n"
        assert not (workspace.path / "untracked.txt").exists()
        assert not (workspace.path / "ignored.txt").exists()
        assert _git(workspace.path, "status", "--porcelain") == b""

    assert list(storage.iterdir()) == []


def test_export_patch_is_repeatable_and_preserves_workspace_files(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)

    with EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-2",
        rollout_id="rollout-1",
    ) as workspace:
        (workspace.path / "tracked.txt").write_text("updated\n", encoding="utf-8")
        (workspace.path / "deleted.txt").unlink()
        (workspace.path / "new.txt").write_text("new file\n", encoding="utf-8")
        (workspace.path / "binary.dat").write_bytes(b"\x00\x01\xff\x00")

        first_patch = workspace.export_patch()
        second_patch = workspace.export_patch()

        assert first_patch == second_patch
        assert "diff --git a/tracked.txt b/tracked.txt" in first_patch
        assert "deleted file mode" in first_patch
        assert "new file mode" in first_patch
        assert "GIT binary patch" in first_patch
        assert (workspace.path / "tracked.txt").read_text(encoding="utf-8") == "updated\n"
        assert (workspace.path / "new.txt").read_text(encoding="utf-8") == "new file\n"
        assert not (workspace.path / "deleted.txt").exists()
        assert _git(workspace.path, "diff", "--cached", "--quiet") == b""

        _git(workspace.path, "reset", "--hard", revision)
        _git(workspace.path, "clean", "-ffdx")
        _git(workspace.path, "apply", "--check", "-", input_bytes=first_patch.encode("utf-8"))
        _git(workspace.path, "apply", "-", input_bytes=first_patch.encode("utf-8"))
        assert (workspace.path / "tracked.txt").read_text(encoding="utf-8") == "updated\n"
        assert (workspace.path / "binary.dat").read_bytes() == b"\x00\x01\xff\x00"


def test_process_sandbox_changes_can_be_exported_as_episode_patch(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)

    with EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-3",
        rollout_id="rollout-2",
    ) as workspace:
        result = ProcessSandbox(workspace.path).run(
            [sys.executable, "-c", "from pathlib import Path; Path('tracked.txt').write_text('agent edit\\n')"]
        )

        assert result.returncode == 0
        assert "+agent edit" in workspace.export_patch()


def test_patch_bytes_preserve_non_utf8_text(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)

    with EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-encoding",
        rollout_id="rollout-1",
    ) as workspace:
        (workspace.path / "legacy.txt").write_bytes(b"\x80legacy text\n")

        patch = workspace.export_patch_bytes()

        assert b"\x80legacy text" in patch
        with pytest.raises(EpisodeWorkspaceError, match="contains non-UTF-8 text"):
            workspace.export_patch()


def test_failed_reset_keeps_the_existing_workspace(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)

    with EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-reset",
        rollout_id="rollout-1",
    ) as workspace:
        (workspace.path / "tracked.txt").write_text("keep after failed reset\n", encoding="utf-8")
        source.rename(tmp_path / "source-moved")

        with pytest.raises(EpisodeWorkspaceError, match="git clone failed"):
            workspace.reset()

        assert workspace.path.is_dir()
        assert (workspace.path / "tracked.txt").read_text(encoding="utf-8") == "keep after failed reset\n"
        assert sorted(path.name for path in workspace.episode_dir.iterdir()) == ["git-home", "workspace"]


def test_cleanup_is_idempotent_and_closed_workspace_rejects_actions(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)
    workspace = EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-4",
        rollout_id="rollout-1",
    )

    workspace.cleanup()
    workspace.cleanup()

    assert workspace.active is False
    assert not workspace.episode_dir.exists()
    with pytest.raises(EpisodeWorkspaceError, match="already been cleaned up"):
        workspace.reset()
    with pytest.raises(EpisodeWorkspaceError, match="already been cleaned up"):
        workspace.export_patch()


def test_cleanup_refuses_a_directory_not_owned_by_the_workspace(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)
    workspace = EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id="task-ownership",
        rollout_id="rollout-1",
    )
    owned_episode_dir = workspace.episode_dir
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    workspace.episode_dir = unrelated

    with pytest.raises(EpisodeWorkspaceError, match="refusing to remove unowned"):
        workspace.cleanup()

    assert unrelated.is_dir()
    workspace.episode_dir = owned_episode_dir
    workspace.cleanup()


@pytest.mark.parametrize("revision", ["missing-ref", "--help"])
def test_failed_creation_removes_partial_episode_directory(tmp_path, revision):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    _create_source_repo(source)

    with pytest.raises(EpisodeWorkspaceError, match="git rev-parse failed"):
        EpisodeWorkspace.create(
            storage,
            source_repo=source,
            revision=revision,
            task_id="task-5",
            rollout_id="rollout-1",
        )

    assert list(storage.iterdir()) == []


def test_rejects_non_repository_source_and_invalid_arguments(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    storage = tmp_path / "episodes"
    storage.mkdir()

    with pytest.raises(EpisodeWorkspaceError, match="git rev-parse failed"):
        EpisodeWorkspace.create(
            storage,
            source_repo=source,
            revision="HEAD",
            task_id="task-6",
            rollout_id="rollout-1",
        )
    assert list(storage.iterdir()) == []

    with pytest.raises(TypeError, match="task_id must be a string"):
        EpisodeWorkspace.create(
            storage,
            source_repo=source,
            revision="HEAD",
            task_id=1,
            rollout_id="rollout-1",
        )
    with pytest.raises(ValueError, match="rollout_id must be non-empty"):
        EpisodeWorkspace.create(
            storage,
            source_repo=source,
            revision="HEAD",
            task_id="task-6",
            rollout_id="",
        )
    with pytest.raises(ValueError, match="git_timeout_seconds must be finite and positive"):
        EpisodeWorkspace.create(
            storage,
            source_repo=source,
            revision="HEAD",
            task_id="task-6",
            rollout_id="rollout-1",
            git_timeout_seconds=0,
        )


def test_context_manager_cleans_up_after_exception(tmp_path):
    source = tmp_path / "source"
    storage = tmp_path / "episodes"
    storage.mkdir()
    revision = _create_source_repo(source)

    with pytest.raises(RuntimeError, match="rollout failed"):
        with EpisodeWorkspace.create(
            storage,
            source_repo=source,
            revision=revision,
            task_id="task-7",
            rollout_id="rollout-1",
        ):
            raise RuntimeError("rollout failed")

    assert list(storage.iterdir()) == []
