import copy
import os
import shutil
import subprocess

import pytest

from lmflow.agentic.atif import atif_trajectory_to_conversation
from lmflow.agentic.bash_agent import (
    BashAgentTurn,
    BashToolCall,
    MinimalBashAgent,
    MinimalBashAgentError,
    bash_agent_turn_from_openai_message,
    bash_tool_definition,
)
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.sandbox import ProcessSandbox
from lmflow.agentic.workspace import EpisodeWorkspace

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("git") is None or not os.path.exists("/bin/bash"),
    reason="MinimalBashAgent tests require POSIX, Git, and /bin/bash",
)


class ScriptedModel:
    def __init__(self, turns, *, mutate_requests=False):
        self.turns = list(turns)
        self.mutate_requests = mutate_requests
        self.requests = []

    def __call__(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if self.mutate_requests:
            messages.append({"role": "user", "content": "injected by model adapter"})
            tools.clear()
        return self.turns.pop(0)


def _git(repo, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout


def _create_source_repo(path):
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "LMFlow Test")
    _git(path, "config", "user.email", "lmflow-test@example.com")
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "Initial fixture")
    return _git(path, "rev-parse", "HEAD").decode("ascii").strip()


def _task(*, tools=None):
    return TaskSpec(
        task_id="task-1",
        messages=[
            {"role": "system", "content": "Work carefully in the repository."},
            {"role": "user", "content": "Update tracked.txt and submit the result."},
        ],
        tools=[] if tools is None else tools,
    )


def _create_workspace(tmp_path, *, task_id="task-1", rollout_id="rollout-1"):
    source = tmp_path / f"source-{rollout_id}"
    storage = tmp_path / f"episodes-{rollout_id}"
    storage.mkdir()
    revision = _create_source_repo(source)
    workspace = EpisodeWorkspace.create(
        storage,
        source_repo=source,
        revision=revision,
        task_id=task_id,
        rollout_id=rollout_id,
    )
    return workspace, ProcessSandbox(workspace.path, timeout_seconds=5)


def test_run_submits_patch_and_emits_converter_compatible_atif(tmp_path):
    model = ScriptedModel(
        [
            BashAgentTurn(
                reasoning_content="Apply the requested edit.",
                tool_calls=(
                    BashToolCall(
                        tool_call_id="call-edit",
                        command="printf 'fixed\\n' > tracked.txt",
                    ),
                ),
            ),
            BashAgentTurn(
                reasoning_content="The change is complete.",
                tool_calls=(
                    BashToolCall(
                        tool_call_id="call-submit",
                        command="printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nPatch ready.\\n'",
                    ),
                ),
            ),
        ],
        mutate_requests=True,
    )
    agent = MinimalBashAgent(model, model_name="test-model", max_steps=4)
    workspace, sandbox = _create_workspace(tmp_path)

    with workspace:
        result = agent.run(
            _task(),
            sandbox=sandbox,
            trajectory_id="trajectory-1",
        )
        model_patch = workspace.export_patch_bytes()

        assert result.status == "submitted"
        assert result.submission == "Patch ready.\n"
        assert b"+fixed" in model_patch
        assert len(result.process_results) == 2
        assert all(process.returncode == 0 for process in result.process_results)
        assert result.trajectory["extra"]["lmflow"] == {
            "task_id": "task-1",
            "exit_status": "submitted",
            "submission": "Patch ready.\n",
        }

        conversation = atif_trajectory_to_conversation(result.trajectory)
        assert conversation["conversation_id"] == "trajectory-1"
        assert conversation["system"] == "Work carefully in the repository."
        assert [message["role"] for message in conversation["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert conversation["messages"][1]["reasoning_content"] == "Apply the requested edit."
        assert conversation["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
            '{"command":"printf \'fixed\\\\n\' > tracked.txt"}'
        )
        assert "<returncode>0</returncode>" in conversation["messages"][2]["content"]
        assert "<output>" in conversation["messages"][2]["content"]

    assert len(model.requests) == 2
    assert [message["role"] for message in model.requests[1][0]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert all(message["content"] != "injected by model adapter" for message in model.requests[1][0])
    assert model.requests[0][1] == [bash_tool_definition()]


def test_failed_submission_command_reaches_step_limit_and_preserves_stderr(tmp_path):
    model = ScriptedModel(
        [
            BashAgentTurn(
                content="Run a failing command.",
                tool_calls=(
                    BashToolCall(
                        tool_call_id="call-fail",
                        command=(
                            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; printf 'failure detail\\n' >&2; exit 3"
                        ),
                    ),
                ),
            )
        ]
    )
    agent = MinimalBashAgent(model, model_name="test-model", max_steps=1)
    workspace, sandbox = _create_workspace(tmp_path)

    with workspace:
        result = agent.run(
            _task(),
            sandbox=sandbox,
            trajectory_id="trajectory-failed-command",
        )

        assert result.status == "step_limit"
        assert result.submission == ""
        assert result.process_results[0].returncode == 3
        assert result.process_results[0].stderr == "failure detail\n"
        conversation = atif_trajectory_to_conversation(result.trajectory)
        observation = conversation["messages"][-1]["content"]
        assert "<returncode>3</returncode>" in observation
        assert "failure detail" in observation


def test_submission_marker_must_be_the_first_stdout_line(tmp_path):
    model = ScriptedModel(
        [
            BashAgentTurn(
                tool_calls=(
                    BashToolCall(
                        tool_call_id="call-submit",
                        command="printf '\\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nToo late.\\n'",
                    ),
                ),
            )
        ]
    )
    agent = MinimalBashAgent(model, model_name="test-model", max_steps=1)
    workspace, sandbox = _create_workspace(tmp_path)

    with workspace:
        result = agent.run(
            _task(),
            sandbox=sandbox,
            trajectory_id="trajectory-late-marker",
        )

        assert result.status == "step_limit"
        assert result.submission == ""


@pytest.mark.parametrize(
    "tool_calls",
    [
        (),
        (
            BashToolCall(tool_call_id="call-1", command="true"),
            BashToolCall(tool_call_id="call-2", command="true"),
        ),
    ],
)
def test_requires_exactly_one_tool_call_per_model_turn(tmp_path, tool_calls):
    model = ScriptedModel([BashAgentTurn(tool_calls=tool_calls)])
    agent = MinimalBashAgent(model, model_name="test-model")
    workspace, sandbox = _create_workspace(tmp_path)

    with workspace, pytest.raises(MinimalBashAgentError, match="exactly one Bash tool call"):
        agent.run(
            _task(),
            sandbox=sandbox,
            trajectory_id="trajectory-invalid-calls",
        )

    assert model.requests


def test_rejects_duplicate_tool_call_ids_before_second_command(tmp_path):
    model = ScriptedModel(
        [
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-1", command="true"),)),
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-1", command="touch should-not-exist"),)),
        ]
    )
    agent = MinimalBashAgent(model, model_name="test-model", max_steps=2)
    workspace, sandbox = _create_workspace(tmp_path)

    with workspace:
        with pytest.raises(MinimalBashAgentError, match="duplicate tool_call_id"):
            agent.run(
                _task(),
                sandbox=sandbox,
                trajectory_id="trajectory-duplicate-call",
            )
        assert not (workspace.path / "should-not-exist").exists()


def test_rejects_custom_tools_before_calling_model(tmp_path):
    turn = BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-1", command="true"),))
    model = ScriptedModel([turn])
    agent = MinimalBashAgent(model, model_name="test-model", max_steps=1)
    workspace, sandbox = _create_workspace(tmp_path)

    with workspace:
        with pytest.raises(MinimalBashAgentError, match="built-in Bash tool"):
            agent.run(
                _task(tools=[{"type": "function", "function": {"name": "python"}}]),
                sandbox=sandbox,
                trajectory_id="trajectory-custom-tool",
            )

    assert model.requests == []


def test_bash_tool_definition_returns_independent_json_objects():
    first = bash_tool_definition()
    second = bash_tool_definition()

    first["function"]["name"] = "changed"

    assert second["function"]["name"] == "bash"


def test_openai_message_adapter_normalizes_tool_calls():
    turn = bash_agent_turn_from_openai_message(
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Inspect the repository.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
        }
    )

    assert turn == BashAgentTurn(
        content="",
        reasoning_content="Inspect the repository.",
        tool_calls=(BashToolCall(tool_call_id="call-1", command="pwd"),),
    )


@pytest.mark.parametrize(
    ("message", "match"),
    [
        ({"role": "user", "tool_calls": []}, "role must be 'assistant'"),
        ({"role": "assistant", "tool_calls": None}, "tool_calls must be an array"),
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "python", "arguments": "{}"},
                    }
                ],
            },
            "function.name must be 'bash'",
        ),
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "not-json"},
                    }
                ],
            },
            "must contain valid JSON",
        ),
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"pwd","extra":true}',
                        },
                    }
                ],
            },
            "must contain only 'command'",
        ),
    ],
)
def test_openai_message_adapter_rejects_unsupported_shapes(message, match):
    with pytest.raises(MinimalBashAgentError, match=match):
        bash_agent_turn_from_openai_message(message)


def test_constructor_and_turn_validation_fail_closed():
    with pytest.raises(ValueError, match="max_steps must be positive"):
        MinimalBashAgent(lambda messages, tools: None, model_name="test-model", max_steps=0)
    with pytest.raises(ValueError, match="shell must be an absolute path"):
        MinimalBashAgent(lambda messages, tools: None, model_name="test-model", shell="bash")
    with pytest.raises(TypeError, match="tool_calls must be a tuple"):
        BashAgentTurn(tool_calls=[])
    with pytest.raises(ValueError, match="command must be non-empty"):
        BashToolCall(tool_call_id="call-1", command="")
