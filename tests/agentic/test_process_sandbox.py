import json
import os
import sys
import time
from pathlib import Path

import pytest

from lmflow.agentic import ProcessLimits, ProcessSandbox, SandboxCapabilityError

pytestmark = pytest.mark.skipif(os.name != "posix", reason="ProcessSandbox currently requires POSIX")


def _pid_is_running(process_id):
    stat_path = Path(f"/proc/{process_id}/stat")
    try:
        stat = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    state = stat.rsplit(")", 1)[1].strip().split(maxsplit=1)[0]
    return state != "Z"


def _wait_until_stopped(process_id, timeout_seconds=2.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_is_running(process_id):
            return
        time.sleep(0.02)
    pytest.fail(f"child process {process_id} was still running after sandbox cleanup")


def test_runs_with_clean_environment_and_workspace_scoped_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LMFLOW_TEST_SECRET", "must-not-leak")
    sandbox = ProcessSandbox(tmp_path)
    code = (
        "import json, os; "
        "print(json.dumps({'cwd': os.getcwd(), 'home': os.environ['HOME'], "
        "'tmpdir': os.environ['TMPDIR'], 'secret': os.environ.get('LMFLOW_TEST_SECRET'), "
        "'visible': os.environ.get('VISIBLE')}))"
    )

    result = sandbox.run([sys.executable, "-c", code], env={"VISIBLE": "yes"})

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.args == (sys.executable, "-c", code)
    assert payload["cwd"] == str(tmp_path)
    assert Path(payload["home"]).is_relative_to(tmp_path)
    assert Path(payload["tmpdir"]).is_relative_to(tmp_path)
    assert not Path(payload["home"]).exists()
    assert not Path(payload["tmpdir"]).exists()
    assert payload["secret"] is None
    assert payload["visible"] == "yes"


def test_rejects_shell_strings_and_cwd_escape(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    sandbox = ProcessSandbox(root)

    with pytest.raises(TypeError, match="shell command strings"):
        sandbox.run("echo unsafe")
    with pytest.raises(ValueError, match="cwd must resolve inside"):
        sandbox.run([sys.executable, "-c", "pass"], cwd="escape")


def test_fails_closed_for_unavailable_capabilities(tmp_path):
    with pytest.raises(SandboxCapabilityError, match="filesystem_isolation"):
        ProcessSandbox(tmp_path, required_capabilities={"filesystem_isolation"})

    sandbox = ProcessSandbox(tmp_path)
    assert sandbox.capabilities["clean_environment"] is True
    assert sandbox.capabilities["process_group_cleanup"] is True
    assert sandbox.capabilities["resource_limits"] is True
    assert sandbox.capabilities["working_directory_containment"] is True
    assert sandbox.capabilities["filesystem_isolation"] is False
    assert sandbox.capabilities["network_isolation"] is False

    with pytest.raises(SandboxCapabilityError, match="unknown sandbox capabilities"):
        sandbox.require_capabilities({"unknown"})
    with pytest.raises(TypeError, match="iterable of capability names"):
        sandbox.require_capabilities("bounded_output")


def test_bounds_stdout_and_stderr_without_pipe_backpressure(tmp_path):
    sandbox = ProcessSandbox(tmp_path, max_output_bytes=32)
    code = "import sys; sys.stdout.write('o' * 2000000); sys.stderr.write('e' * 2000000)"

    result = sandbox.run([sys.executable, "-c", code])

    assert result.returncode == 0
    assert result.stdout == "o" * 32
    assert result.stderr == "e" * 32
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_can_merge_stderr_for_ordered_agent_observations(tmp_path):
    sandbox = ProcessSandbox(tmp_path)
    code = "import os; os.write(1, b'first\\n'); os.write(2, b'second\\n'); os.write(1, b'third\\n')"

    result = sandbox.run([sys.executable, "-c", code], merge_stderr=True)

    assert result.returncode == 0
    assert result.stdout == "first\nsecond\nthird\n"
    assert result.stderr == ""
    assert result.stderr_truncated is False


def test_timeout_kills_the_command_process_group(tmp_path):
    sandbox = ProcessSandbox(tmp_path, timeout_seconds=0.2)
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path('child.pid').write_text(str(child.pid), encoding='utf-8'); "
        "time.sleep(60)"
    )

    result = sandbox.run([sys.executable, "-c", code])

    child_pid = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
    assert result.timed_out is True
    assert result.returncode < 0
    _wait_until_stopped(child_pid)


def test_normal_parent_exit_still_cleans_descendants(tmp_path):
    sandbox = ProcessSandbox(tmp_path)
    code = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path('child.pid').write_text(str(child.pid), encoding='utf-8')"
    )

    result = sandbox.run([sys.executable, "-c", code])

    child_pid = int((tmp_path / "child.pid").read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert result.timed_out is False
    _wait_until_stopped(child_pid)


def test_applies_file_size_limit_before_exec(tmp_path):
    sandbox = ProcessSandbox(tmp_path, limits=ProcessLimits(file_size_bytes=1024))
    code = "from pathlib import Path; Path('large.bin').write_bytes(b'x' * 4096)"

    result = sandbox.run([sys.executable, "-c", code])

    assert result.returncode != 0
    assert (tmp_path / "large.bin").stat().st_size <= 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu_seconds": 0},
        {"memory_bytes": -1},
        {"file_size_bytes": True},
        {"open_files": 1.5},
        {"processes": "2"},
    ],
)
def test_rejects_invalid_resource_limits(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        ProcessLimits(**kwargs)
