"""Zero-dependency process supervision for Agentic tool execution."""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

_CAPABILITIES = {
    "bounded_output": True,
    "clean_environment": True,
    "filesystem_isolation": False,
    "network_isolation": False,
    "process_group_cleanup": os.name == "posix",
    "resource_limits": os.name == "posix",
    "snapshot": False,
    "wall_time_limit": True,
    "working_directory_containment": True,
}


class SandboxCapabilityError(RuntimeError):
    """Raised when a task requires isolation that this backend cannot provide."""


class _BoundedOutputCollector:
    def __init__(self, limit: int):
        self._limit = limit
        self._data = bytearray()
        self._truncated = False
        self._lock = threading.Lock()

    def drain(self, stream: BinaryIO) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                with self._lock:
                    remaining = self._limit - len(self._data)
                    if remaining > 0:
                        self._data.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        self._truncated = True
        except (OSError, ValueError):
            with self._lock:
                self._truncated = True
        finally:
            stream.close()

    def result(self, *, drain_completed: bool) -> tuple[str, bool]:
        with self._lock:
            data = bytes(self._data)
            truncated = self._truncated or not drain_completed
        return data.decode("utf-8", errors="replace"), truncated


@dataclass(frozen=True)
class ProcessLimits:
    """Optional POSIX limits applied before the command is executed."""

    cpu_seconds: Optional[int] = None
    memory_bytes: Optional[int] = None
    file_size_bytes: Optional[int] = None
    open_files: Optional[int] = None
    processes: Optional[int] = None

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def as_dict(self) -> dict[str, int]:
        return {
            name: value
            for name, value in (
                ("cpu_seconds", self.cpu_seconds),
                ("memory_bytes", self.memory_bytes),
                ("file_size_bytes", self.file_size_bytes),
                ("open_files", self.open_files),
                ("processes", self.processes),
            )
            if value is not None
        }


@dataclass(frozen=True)
class ProcessResult:
    """Bounded output and termination metadata from one command."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    stdout_truncated: bool
    stderr_truncated: bool


class ProcessSandbox:
    """Run commands under a workspace root with process and resource guards.

    This backend limits accidental damage and resource loss. It does not isolate
    the filesystem or network and must not be used as a security boundary for
    actively malicious code.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        *,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 1_000_000,
        limits: Optional[ProcessLimits] = None,
        required_capabilities: Iterable[str] = (),
    ) -> None:
        root_path = Path(root).resolve(strict=True)
        if not root_path.is_dir():
            raise NotADirectoryError(str(root_path))
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int):
            raise TypeError("max_output_bytes must be an integer")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if limits is not None and not isinstance(limits, ProcessLimits):
            raise TypeError("limits must be a ProcessLimits instance or None")

        self.root = root_path
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = max_output_bytes
        self.limits = limits or ProcessLimits()
        self.require_capabilities(required_capabilities)

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return the isolation and supervision features actually provided."""
        return dict(_CAPABILITIES)

    def require_capabilities(self, required_capabilities: Iterable[str]) -> None:
        if isinstance(required_capabilities, (str, bytes)):
            raise TypeError("required_capabilities must be an iterable of capability names")
        required = set(required_capabilities)
        if any(not isinstance(name, str) for name in required):
            raise TypeError("required_capabilities must contain only strings")
        unknown = required.difference(_CAPABILITIES)
        if unknown:
            raise SandboxCapabilityError(f"unknown sandbox capabilities: {sorted(unknown)}")
        missing = {name for name in required if not _CAPABILITIES[name]}
        if missing:
            raise SandboxCapabilityError(f"ProcessSandbox cannot provide required capabilities: {sorted(missing)}")

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Union[str, os.PathLike] = ".",
        env: Optional[Mapping[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        limits: Optional[ProcessLimits] = None,
        required_capabilities: Iterable[str] = (),
    ) -> ProcessResult:
        """Execute one argv-only command and clean its process group."""
        if os.name != "posix":
            raise SandboxCapabilityError("ProcessSandbox currently requires a POSIX host")
        command = self._normalize_command(argv)
        workdir = self._resolve_cwd(cwd)
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise TypeError("timeout_seconds must be a number")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        run_limits = self.limits if limits is None else limits
        if not isinstance(run_limits, ProcessLimits):
            raise TypeError("limits must be a ProcessLimits instance or None")
        self.require_capabilities(required_capabilities)

        with tempfile.TemporaryDirectory(prefix=".lmflow-sandbox-", dir=self.root) as runtime_dir:
            runtime_path = Path(runtime_dir)
            home_path = runtime_path / "home"
            temp_path = runtime_path / "tmp"
            home_path.mkdir()
            temp_path.mkdir()
            child_env = self._build_environment(home_path, temp_path, env)
            wrapped_command = self._wrap_command(command, run_limits)
            started_at = time.monotonic()
            process = subprocess.Popen(
                wrapped_command,
                cwd=workdir,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
            )
            stdout_collector = _BoundedOutputCollector(self.max_output_bytes)
            stderr_collector = _BoundedOutputCollector(self.max_output_bytes)
            stdout_thread = threading.Thread(target=stdout_collector.drain, args=(process.stdout,), daemon=True)
            stderr_thread = threading.Thread(target=stderr_collector.drain, args=(process.stderr,), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            timed_out = False
            try:
                process.wait(timeout=float(timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                self._kill_process_group(process.pid)
                if process.poll() is None:
                    process.wait()
            duration_seconds = time.monotonic() - started_at
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            stdout, stdout_truncated = stdout_collector.result(drain_completed=not stdout_thread.is_alive())
            stderr, stderr_truncated = stderr_collector.result(drain_completed=not stderr_thread.is_alive())

        return ProcessResult(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=duration_seconds,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    @staticmethod
    def _normalize_command(argv: Sequence[str]) -> tuple[str, ...]:
        if isinstance(argv, (str, bytes)):
            raise TypeError("argv must be a sequence of strings; shell command strings are not accepted")
        command = tuple(argv)
        if not command:
            raise ValueError("argv must not be empty")
        for argument in command:
            if not isinstance(argument, str):
                raise TypeError("every argv item must be a string")
            if "\0" in argument:
                raise ValueError("argv items must not contain NUL bytes")
        return command

    def _resolve_cwd(self, cwd: Union[str, os.PathLike]) -> str:
        candidate = Path(cwd)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"cwd must resolve inside sandbox root {self.root}") from exc
        return str(resolved)

    @staticmethod
    def _build_environment(home_path: Path, temp_path: Path, extra_env: Optional[Mapping[str, str]]) -> dict[str, str]:
        child_env = {
            "HOME": str(home_path),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONIOENCODING": "utf-8",
            "TMPDIR": str(temp_path),
        }
        if extra_env is None:
            return child_env
        if not isinstance(extra_env, Mapping):
            raise TypeError("env must be a mapping of strings")
        for name, value in extra_env.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError("env keys and values must be strings")
            if not name or "=" in name or "\0" in name or "\0" in value:
                raise ValueError("env contains an invalid key or value")
            child_env[name] = value
        return child_env

    @staticmethod
    def _wrap_command(command: tuple[str, ...], limits: ProcessLimits) -> tuple[str, ...]:
        helper = Path(__file__).with_name("_sandbox_exec.py")
        return (sys.executable, str(helper), "--limits", json.dumps(limits.as_dict()), "--", *command)

    @staticmethod
    def _kill_process_group(process_id: int) -> None:
        try:
            os.killpg(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


__all__ = ["ProcessLimits", "ProcessResult", "ProcessSandbox", "SandboxCapabilityError"]
