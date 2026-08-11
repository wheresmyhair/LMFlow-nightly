"""Atomic node-local repository snapshots for Agentic episode workspaces."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit

PathLike = str | os.PathLike[str]

_CACHE_FORMAT = "lmflow-prepared-repository-v1"
_FULL_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_NETWORK_ENVIRONMENT_KEYS = (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)


class PreparedRepositoryCacheError(RuntimeError):
    """Raised when a repository snapshot cannot be prepared or validated."""


class PreparedRepositoryCache:
    """Prepare immutable, commit-addressed Git repositories below one cache root.

    Each published entry contains the exact commit needed by an
    ``EpisodeWorkspace``. A per-key POSIX advisory lock serializes builders, and
    a completed staging directory is renamed into place only after Git and the
    READY marker have been validated. Published repositories are consumer-owned
    read-only inputs by contract; this class does not mutate or evict them.

    The initial transport boundary accepts public HTTPS URLs, ``file://`` URLs,
    and local filesystem paths. URL credentials and query strings are rejected;
    configure proxies through the standard proxy environment variables.
    """

    def __init__(
        self,
        cache_root: PathLike,
        *,
        git_timeout_seconds: float = 1_200.0,
        lock_timeout_seconds: float = 1_200.0,
    ) -> None:
        self.cache_root = self._resolve_directory(cache_root, name="cache_root")
        self.git_timeout_seconds = self._validate_timeout(git_timeout_seconds, name="git_timeout_seconds")
        self.lock_timeout_seconds = self._validate_timeout(lock_timeout_seconds, name="lock_timeout_seconds")
        self._repositories_root = self._owned_directory("repositories")
        self._locks_root = self._owned_directory("locks")
        self._staging_root = self._owned_directory("staging")
        self._git_home = self._owned_directory("git-home")

    def prepare(self, repository: str, revision: str) -> Path:
        """Return a local bare repository containing one exact commit."""

        repository = self._normalize_repository(repository)
        revision = self._normalize_revision(revision)
        key = hashlib.sha256(f"{_CACHE_FORMAT}\0{repository}\0{revision}".encode()).hexdigest()
        target = self._repositories_root / f"repository-{key}"
        lock_path = self._locks_root / f"repository-{key}.lock"

        with self._lock(lock_path):
            if os.path.lexists(target):
                return self._validate_entry(target, repository=repository, revision=revision)

            staging = Path(tempfile.mkdtemp(prefix=f"repository-{key}-", dir=self._staging_root))
            try:
                repository_path = staging / "repository.git"
                self._run_git(("init", "--bare", "--", str(repository_path)), cwd=staging)
                self._run_git(
                    ("fetch", "--no-tags", "--depth=1", "--", repository, revision),
                    cwd=repository_path,
                )
                resolved_revision = (
                    self._run_git(
                        ("rev-parse", "--verify", "--end-of-options", "FETCH_HEAD^{commit}"),
                        cwd=repository_path,
                    )
                    .decode("ascii")
                    .strip()
                )
                if resolved_revision.lower() != revision:
                    raise PreparedRepositoryCacheError(
                        "fetched repository revision does not match the requested commit"
                    )
                self._run_git(("update-ref", "refs/lmflow/base", revision), cwd=repository_path)
                self._run_git(("symbolic-ref", "HEAD", "refs/lmflow/base"), cwd=repository_path)
                try:
                    (repository_path / "FETCH_HEAD").unlink()
                except FileNotFoundError:
                    pass
                self._write_ready_marker(staging / "READY", repository=repository, revision=revision)
                staging.rename(target)
            except BaseException:
                self._remove_staging(staging, key=key)
                raise

            return self._validate_entry(target, repository=repository, revision=revision)

    @staticmethod
    def _resolve_directory(path: PathLike, *, name: str) -> Path:
        try:
            resolved = Path(path).resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as error:
            raise PreparedRepositoryCacheError(f"{name} must be an existing directory") from error
        if not resolved.is_dir():
            raise PreparedRepositoryCacheError(f"{name} must be an existing directory")
        return resolved

    @staticmethod
    def _validate_timeout(value: float, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return float(value)

    def _owned_directory(self, name: str) -> Path:
        path = self.cache_root / name
        try:
            path.mkdir(mode=0o700, exist_ok=True)
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise PreparedRepositoryCacheError(f"failed to initialize cache directory {name!r}") from error
        if path.is_symlink() or resolved.parent != self.cache_root or not resolved.is_dir():
            raise PreparedRepositoryCacheError(f"cache directory {name!r} must be owned by cache_root")
        return resolved

    @staticmethod
    def _normalize_repository(repository: str) -> str:
        if not isinstance(repository, str):
            raise TypeError("repository must be a string")
        if not repository or "\0" in repository:
            raise ValueError("repository must be non-empty and must not contain NUL bytes")
        if re.fullmatch(r"[^/\\]+@[^/:]+:.+", repository):
            raise ValueError("SCP-style repository URLs are not supported; use an HTTPS or file URL")

        parsed = urlsplit(repository)
        if parsed.scheme:
            if parsed.scheme not in {"file", "https"}:
                raise ValueError("repository URL scheme must be 'https' or 'file'")
            if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
                raise ValueError("repository URLs must not contain credentials, query strings, or fragments")
            if parsed.scheme == "https" and not parsed.netloc:
                raise ValueError("HTTPS repository URLs must include a host")
            return repository

        try:
            return str(Path(repository).resolve(strict=False))
        except (OSError, RuntimeError) as error:
            raise ValueError("local repository path cannot be resolved") from error

    @staticmethod
    def _normalize_revision(revision: str) -> str:
        if not isinstance(revision, str):
            raise TypeError("revision must be a string")
        if _FULL_COMMIT_PATTERN.fullmatch(revision) is None:
            raise ValueError("revision must be a full 40-character Git commit")
        return revision.lower()

    @contextmanager
    def _lock(self, lock_path: Path) -> Iterator[None]:
        try:
            import fcntl
        except ImportError as error:
            raise PreparedRepositoryCacheError("repository cache preparation requires POSIX advisory locks") from error

        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise PreparedRepositoryCacheError("failed to open repository cache lock") from error

        lock_file: BinaryIO = os.fdopen(descriptor, "a+b")
        deadline = time.monotonic() + self.lock_timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PreparedRepositoryCacheError(
                            f"timed out waiting for repository cache lock after {self.lock_timeout_seconds:g} seconds"
                        ) from error
                    time.sleep(min(0.05, remaining))
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()

    def _validate_entry(self, entry: Path, *, repository: str, revision: str) -> Path:
        marker_path = entry / "READY"
        repository_path = entry / "repository.git"
        if (
            entry.is_symlink()
            or not entry.is_dir()
            or marker_path.is_symlink()
            or not marker_path.is_file()
            or repository_path.is_symlink()
            or not repository_path.is_dir()
        ):
            raise PreparedRepositoryCacheError(f"repository cache entry is incomplete or unsafe: {entry}")

        expected_marker = {
            "format": _CACHE_FORMAT,
            "repository_sha256": hashlib.sha256(repository.encode()).hexdigest(),
            "revision": revision,
        }
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PreparedRepositoryCacheError(
                f"repository cache entry has an invalid READY marker: {entry}"
            ) from error
        if marker != expected_marker:
            raise PreparedRepositoryCacheError(f"repository cache entry identity does not match its key: {entry}")

        resolved_revision = (
            self._run_git(
                ("rev-parse", "--verify", "--end-of-options", "refs/lmflow/base^{commit}"),
                cwd=repository_path,
            )
            .decode("ascii")
            .strip()
        )
        if resolved_revision.lower() != revision:
            raise PreparedRepositoryCacheError(
                f"repository cache entry commit does not match its READY marker: {entry}"
            )
        return repository_path.resolve(strict=True)

    @staticmethod
    def _write_ready_marker(path: Path, *, repository: str, revision: str) -> None:
        marker = {
            "format": _CACHE_FORMAT,
            "repository_sha256": hashlib.sha256(repository.encode()).hexdigest(),
            "revision": revision,
        }
        try:
            with path.open("x", encoding="utf-8") as marker_file:
                json.dump(marker, marker_file, sort_keys=True, separators=(",", ":"))
                marker_file.write("\n")
                marker_file.flush()
                os.fsync(marker_file.fileno())
        except OSError as error:
            raise PreparedRepositoryCacheError("failed to write repository cache READY marker") from error

    def _run_git(self, arguments: Sequence[str], *, cwd: Path) -> bytes:
        command = ("git", *arguments)
        environment = {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(self._git_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ.get("PATH", os.defpath),
            "XDG_CONFIG_HOME": str(self._git_home / "xdg"),
        }
        environment.update({key: os.environ[key] for key in _NETWORK_ENVIRONMENT_KEYS if key in os.environ})
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=self.git_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise PreparedRepositoryCacheError(
                f"git command timed out after {self.git_timeout_seconds:g} seconds"
            ) from error
        except OSError as error:
            raise PreparedRepositoryCacheError(f"failed to execute git: {error}") from error
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            detail = stderr or f"exit status {result.returncode}"
            raise PreparedRepositoryCacheError(f"git {arguments[0]} failed: {detail}")
        return result.stdout

    def _remove_staging(self, staging: Path, *, key: str) -> None:
        if staging.parent != self._staging_root or not staging.name.startswith(f"repository-{key}-"):
            raise PreparedRepositoryCacheError(f"refusing to remove unowned cache staging directory: {staging}")
        try:
            shutil.rmtree(staging)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise PreparedRepositoryCacheError(f"failed to remove cache staging directory: {staging}") from error


__all__ = ["PreparedRepositoryCache", "PreparedRepositoryCacheError"]
