"""Per-rollout repository workspace lifecycle for Agentic tasks."""

import hashlib
import math
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Union


class EpisodeWorkspaceError(RuntimeError):
    """Raised when an episode workspace cannot complete a lifecycle action."""


class EpisodeWorkspace:
    """Own an isolated repository checkout for one rollout attempt.

    The checkout has independent Git metadata and a unique directory, but it
    may share immutable objects with ``source_repo``. This class provides
    workspace lifecycle isolation; it does not isolate filesystem or network
    access for commands executed inside the checkout.
    """

    def __init__(
        self,
        *,
        storage_root: Path,
        episode_dir: Path,
        path: Path,
        git_home: Path,
        source_repo: Path,
        base_revision: str,
        task_id: str,
        rollout_id: str,
        git_timeout_seconds: float,
    ) -> None:
        self.storage_root = storage_root
        self.episode_dir = episode_dir
        self.path = path
        self.source_repo = source_repo
        self.base_revision = base_revision
        self.task_id = task_id
        self.rollout_id = rollout_id
        self.git_timeout_seconds = git_timeout_seconds
        self._git_home = git_home
        self._active = True

    @classmethod
    def create(
        cls,
        storage_root: Union[str, os.PathLike],
        *,
        source_repo: Union[str, os.PathLike],
        revision: str,
        task_id: str,
        rollout_id: str,
        git_timeout_seconds: float = 120.0,
    ) -> "EpisodeWorkspace":
        """Create a fresh checkout for one task rollout.

        ``source_repo`` must be a local Git repository containing ``revision``.
        The source working tree and its uncommitted changes are never copied.
        """
        root = cls._resolve_directory(storage_root, name="storage_root")
        source = cls._resolve_directory(source_repo, name="source_repo")
        cls._validate_identity(task_id, name="task_id")
        cls._validate_identity(rollout_id, name="rollout_id")
        cls._validate_revision(revision)
        timeout = cls._validate_timeout(git_timeout_seconds)

        identity = hashlib.sha256(f"{task_id}\0{rollout_id}".encode()).hexdigest()[:12]
        episode_dir = root / f"episode-{identity}-{uuid.uuid4().hex}"
        workspace_path = episode_dir / "workspace"
        git_home = episode_dir / "git-home"
        episode_dir.mkdir(mode=0o700)
        git_home.mkdir(mode=0o700)

        try:
            cls._run_git(
                ("rev-parse", "--git-dir"),
                cwd=source,
                git_home=git_home,
                timeout_seconds=timeout,
            )
            base_revision = (
                cls._run_git(
                    ("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"),
                    cwd=source,
                    git_home=git_home,
                    timeout_seconds=timeout,
                )
                .decode("ascii")
                .strip()
            )
            cls._populate_checkout(
                source_repo=source,
                destination=workspace_path,
                base_revision=base_revision,
                git_home=git_home,
                timeout_seconds=timeout,
            )
        except BaseException:
            cls._remove_episode_dir(root, episode_dir)
            raise

        return cls(
            storage_root=root,
            episode_dir=episode_dir,
            path=workspace_path,
            git_home=git_home,
            source_repo=source,
            base_revision=base_revision,
            task_id=task_id,
            rollout_id=rollout_id,
            git_timeout_seconds=timeout,
        )

    @property
    def active(self) -> bool:
        """Whether this object still owns a live episode directory."""
        return self._active

    def reset(self) -> None:
        """Restore tracked, untracked, and ignored files to the base commit."""
        self._ensure_active()
        suffix = uuid.uuid4().hex
        replacement = self.episode_dir / f"workspace-reset-{suffix}"
        previous = self.episode_dir / f"workspace-previous-{suffix}"
        try:
            self._populate_checkout(
                source_repo=self.source_repo,
                destination=replacement,
                base_revision=self.base_revision,
                git_home=self._git_home,
                timeout_seconds=self.git_timeout_seconds,
            )
            os.replace(self.path, previous)
            try:
                os.replace(replacement, self.path)
            except BaseException:
                os.replace(previous, self.path)
                raise
            self._remove_episode_child(previous)
        except BaseException:
            self._remove_episode_child(replacement)
            raise

    def export_patch(self) -> str:
        """Return a UTF-8 Git patch, including Git-encoded binary changes."""
        patch = self.export_patch_bytes()
        try:
            return patch.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EpisodeWorkspaceError(
                "git patch contains non-UTF-8 text; use export_patch_bytes() to preserve the original bytes"
            ) from exc

    def export_patch_bytes(self) -> bytes:
        """Return the exact Git patch bytes without changing workspace files."""
        self._ensure_active()
        try:
            self._git(("add", "--all", "--", "."))
            patch = self._git(
                (
                    "diff",
                    "--cached",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    self.base_revision,
                    "--",
                )
            )
        finally:
            self._git(("reset", "--mixed", "HEAD"))
        return patch

    def cleanup(self) -> None:
        """Remove the owned episode directory; repeated calls are safe."""
        if not self._active:
            return
        self._remove_episode_dir(self.storage_root, self.episode_dir)
        self._active = False

    def __enter__(self) -> "EpisodeWorkspace":
        self._ensure_active()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    def _git(self, arguments: Sequence[str]) -> bytes:
        return self._run_git(
            arguments,
            cwd=self.path,
            git_home=self._git_home,
            timeout_seconds=self.git_timeout_seconds,
        )

    def _ensure_active(self) -> None:
        if not self._active:
            raise EpisodeWorkspaceError("episode workspace has already been cleaned up")
        if not self.path.is_dir():
            raise EpisodeWorkspaceError(f"episode workspace is missing: {self.path}")

    @staticmethod
    def _resolve_directory(path: Union[str, os.PathLike], *, name: str) -> Path:
        try:
            resolved = Path(path).resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as exc:
            raise EpisodeWorkspaceError(f"{name} must be an existing directory") from exc
        if not resolved.is_dir():
            raise EpisodeWorkspaceError(f"{name} must be an existing directory")
        return resolved

    @staticmethod
    def _validate_identity(value: str, *, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value or "\0" in value:
            raise ValueError(f"{name} must be non-empty and must not contain NUL bytes")

    @staticmethod
    def _validate_revision(revision: str) -> None:
        if not isinstance(revision, str):
            raise TypeError("revision must be a string")
        if not revision or "\0" in revision:
            raise ValueError("revision must be non-empty and must not contain NUL bytes")

    @staticmethod
    def _validate_timeout(value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("git_timeout_seconds must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("git_timeout_seconds must be finite and positive")
        return float(value)

    @staticmethod
    def _git_environment(git_home: Path) -> dict[str, str]:
        return {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(git_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ.get("PATH", os.defpath),
            "XDG_CONFIG_HOME": str(git_home / "xdg"),
        }

    @classmethod
    def _populate_checkout(
        cls,
        *,
        source_repo: Path,
        destination: Path,
        base_revision: str,
        git_home: Path,
        timeout_seconds: float,
    ) -> None:
        cls._run_git(
            ("clone", "--shared", "--no-checkout", "--", str(source_repo), str(destination)),
            cwd=destination.parent,
            git_home=git_home,
            timeout_seconds=timeout_seconds,
        )
        cls._run_git(
            ("checkout", "--detach", "--force", base_revision),
            cwd=destination,
            git_home=git_home,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def _run_git(
        cls,
        arguments: Sequence[str],
        *,
        cwd: Path,
        git_home: Path,
        timeout_seconds: float,
    ) -> bytes:
        command = ("git", *arguments)
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=cls._git_environment(git_home),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise EpisodeWorkspaceError(f"git command timed out after {timeout_seconds:g} seconds") from exc
        except OSError as exc:
            raise EpisodeWorkspaceError(f"failed to execute git: {exc}") from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            detail = stderr or f"exit status {result.returncode}"
            raise EpisodeWorkspaceError(f"git {arguments[0]} failed: {detail}")
        return result.stdout

    def _remove_episode_child(self, path: Path) -> None:
        if path.parent != self.episode_dir or not path.name.startswith(("workspace-reset-", "workspace-previous-")):
            raise EpisodeWorkspaceError(f"refusing to remove unowned episode child: {path}")
        self._remove_path(path)

    @staticmethod
    def _remove_episode_dir(storage_root: Path, episode_dir: Path) -> None:
        if (
            episode_dir.parent != storage_root
            or re.fullmatch(r"episode-[0-9a-f]{12}-[0-9a-f]{32}", episode_dir.name) is None
        ):
            raise EpisodeWorkspaceError(f"refusing to remove unowned episode directory: {episode_dir}")
        EpisodeWorkspace._remove_path(episode_dir)

    @staticmethod
    def _remove_path(path: Path) -> None:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise EpisodeWorkspaceError(f"failed to remove workspace path {path}: {exc}") from exc


__all__ = ["EpisodeWorkspace", "EpisodeWorkspaceError"]
