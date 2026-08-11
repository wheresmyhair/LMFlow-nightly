import copy
import os
import shutil
import subprocess

import pytest

from lmflow.agentic.atif import atif_trajectory_to_conversation
from lmflow.agentic.bash_agent import (
    BashAgentFormatError,
    BashAgentTurn,
    BashToolCall,
    MinimalBashAgent,
    MinimalBashAgentConfig,
    MinimalBashAgentError,
    ProcessSandboxBashEnvironment,
    bash_agent_turn_from_openai_message,
    bash_tool_definition,
)
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.sandbox import ProcessResult, ProcessSandbox
from lmflow.agentic.workspace import EpisodeWorkspace

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("git") is None or not os.path.exists("/bin/bash"),
    reason="MinimalBashAgent tests require POSIX, Git, and /bin/bash",
)


class ScriptedModel:
    def __init__(self, outputs, *, mutate_requests=False):
        self.outputs = list(outputs)
        self.mutate_requests = mutate_requests
        self.requests = []

    def __call__(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if self.mutate_requests:
            messages.append({"role": "user", "content": "injected by model adapter"})
            tools.clear()
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class RecordingEnvironment:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        return self.results.pop(0)


def _process_result(command, *, stdout="", stderr="", returncode=0, timed_out=False):
    return ProcessResult(
        args=("/bin/bash", "-lc", command),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=0.01,
        stdout_truncated=False,
        stderr_truncated=False,
    )


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
    sandbox = ProcessSandbox(workspace.path, timeout_seconds=5)
    return workspace, ProcessSandboxBashEnvironment(sandbox)


def _submit_turn(*, cost=0.0, command=None):
    command = command or "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nPatch ready.\\n'"
    return BashAgentTurn(
        reasoning_content="The change is complete.",
        tool_calls=(BashToolCall(tool_call_id="call-submit", command=command),),
        cost=cost,
        raw_response={"id": "response-submit"},
    )


def test_run_submits_patch_and_emits_raw_and_atif_trajectories(tmp_path):
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
                cost=0.25,
                finish_reason="tool_calls",
                raw_response={"id": "response-edit"},
            ),
            _submit_turn(cost=0.5),
        ],
        mutate_requests=True,
    )
    agent = MinimalBashAgent(
        model,
        model_name="test-model",
        config=MinimalBashAgentConfig(step_limit=4),
    )
    workspace, environment = _create_workspace(tmp_path)

    with workspace:
        result = agent.run(
            _task(),
            environment=environment,
            trajectory_id="trajectory-1",
        )
        model_patch = workspace.export_patch_bytes()

        assert result.exit_status == "Submitted"
        assert result.submission == "Patch ready.\n"
        assert result.n_model_calls == 2
        assert result.model_cost == pytest.approx(0.75)
        assert result.format_error_count == 0
        assert b"+fixed" in model_patch
        assert len(result.process_results) == 2
        assert all(process.returncode == 0 for process in result.process_results)
        assert result.raw_trajectory["trajectory_format"] == "lmflow-mini-swe-agent-raw-v1"
        assert result.raw_trajectory["info"]["upstream"]["commit"].startswith("a83fcae")
        assert result.raw_trajectory["info"]["model_stats"] == {
            "instance_cost": 0.75,
            "api_calls": 2,
        }
        assert result.raw_trajectory["messages"][-1]["role"] == "exit"

        assert result.atif_trajectory is not None
        conversation = atif_trajectory_to_conversation(result.atif_trajectory)
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

    assert len(model.requests) == 2
    assert [message["role"] for message in model.requests[1][0]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert all("extra" not in message for message in model.requests[1][0])
    assert all(message["content"] != "injected by model adapter" for message in model.requests[1][0])
    assert model.requests[0][1] == [bash_tool_definition()]


def test_executes_multiple_tool_calls_and_returns_observations_together():
    model = ScriptedModel(
        [
            BashAgentTurn(
                content="Inspect both values.",
                tool_calls=(
                    BashToolCall(tool_call_id="call-1", command="first"),
                    BashToolCall(tool_call_id="call-2", command="second"),
                ),
            ),
            _submit_turn(),
        ]
    )
    environment = RecordingEnvironment(
        [
            _process_result("first", stdout="one\n"),
            _process_result("second", stdout="two\n"),
            _process_result(
                "submit",
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nPatch ready.\n",
            ),
        ]
    )

    result = MinimalBashAgent(model, model_name="test-model").run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-parallel-calls",
    )

    assert [call.command for call in environment.calls] == ["first", "second", _submit_turn().tool_calls[0].command]
    assert [message["role"] for message in model.requests[1][0]][-3:] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert result.atif_trajectory is not None
    first_agent_step = result.atif_trajectory["steps"][2]
    assert [call["tool_call_id"] for call in first_agent_step["tool_calls"]] == ["call-1", "call-2"]
    assert [item["source_call_id"] for item in first_agent_step["observation"]["results"]] == [
        "call-1",
        "call-2",
    ]


def test_recovers_from_format_error_but_withholds_loss_ready_atif():
    model = ScriptedModel(
        [
            BashAgentFormatError(
                "Use the bash tool.",
                cost=0.25,
                raw_response={"id": "malformed-response"},
            ),
            _submit_turn(cost=0.5),
        ]
    )
    environment = RecordingEnvironment(
        [
            _process_result(
                "submit",
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nPatch ready.\n",
            )
        ]
    )

    result = MinimalBashAgent(model, model_name="test-model").run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-format-recovery",
    )

    assert result.exit_status == "Submitted"
    assert result.n_model_calls == 2
    assert result.model_cost == pytest.approx(0.75)
    assert result.format_error_count == 1
    assert result.atif_trajectory is None
    feedback = result.raw_trajectory["messages"][2]
    assert feedback["role"] == "user"
    assert feedback["extra"]["response"] == {"id": "malformed-response"}
    assert model.requests[1][0][-1] == {"role": "user", "content": "Use the bash tool."}


def test_repeated_format_errors_stop_without_executing_commands():
    model = ScriptedModel(
        [
            BashAgentFormatError("Try again.", cost=0.2),
            BashAgentFormatError("Try again.", cost=0.3),
        ]
    )
    environment = RecordingEnvironment([])
    config = MinimalBashAgentConfig(max_consecutive_format_errors=2)

    result = MinimalBashAgent(model, model_name="test-model", config=config).run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-format-limit",
    )

    assert result.exit_status == "RepeatedFormatError"
    assert result.n_model_calls == 2
    assert result.model_cost == pytest.approx(0.5)
    assert result.format_error_count == 2
    assert result.atif_trajectory is None
    assert environment.calls == []


def test_clean_turn_resets_consecutive_format_error_counter():
    model = ScriptedModel(
        [
            BashAgentFormatError("Try again."),
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-ok", command="true"),)),
            BashAgentFormatError("Try again."),
            _submit_turn(),
        ]
    )
    environment = RecordingEnvironment(
        [
            _process_result("true"),
            _process_result(
                "submit",
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nPatch ready.\n",
            ),
        ]
    )
    config = MinimalBashAgentConfig(max_consecutive_format_errors=2)

    result = MinimalBashAgent(model, model_name="test-model", config=config).run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-format-reset",
    )

    assert result.exit_status == "Submitted"
    assert result.n_model_calls == 4
    assert result.format_error_count == 2
    assert result.atif_trajectory is None


@pytest.mark.parametrize(
    ("config", "turn", "expected_calls"),
    [
        (
            MinimalBashAgentConfig(step_limit=1, cost_limit=0),
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-step", command="true"),)),
            1,
        ),
        (
            MinimalBashAgentConfig(step_limit=0, cost_limit=0.5),
            BashAgentTurn(
                tool_calls=(BashToolCall(tool_call_id="call-cost", command="true"),),
                cost=1.0,
            ),
            1,
        ),
    ],
)
def test_enforces_step_and_cost_limits_before_the_next_model_call(config, turn, expected_calls):
    model = ScriptedModel([turn])
    environment = RecordingEnvironment([_process_result("true")])

    result = MinimalBashAgent(model, model_name="test-model", config=config).run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-budget",
    )

    assert result.exit_status == "LimitsExceeded"
    assert result.n_model_calls == expected_calls
    assert len(model.requests) == expected_calls


def test_wall_time_limit_is_checked_between_model_calls():
    model = ScriptedModel([BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-sleep", command="sleep 1.1"),))])
    environment = RecordingEnvironment([_process_result("sleep 1.1")])
    config = MinimalBashAgentConfig(
        step_limit=0,
        cost_limit=0,
        wall_time_limit_seconds=1,
    )

    original_execute = environment.execute

    def delayed_execute(call):
        import time

        time.sleep(1.05)
        return original_execute(call)

    environment.execute = delayed_execute
    result = MinimalBashAgent(model, model_name="test-model", config=config).run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-wall-time",
    )

    assert result.exit_status == "TimeExceeded"
    assert result.n_model_calls == 1


def test_submission_preserves_pinned_leading_whitespace_behavior():
    model = ScriptedModel(
        [_submit_turn(command="printf '\\n  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT  \\nAccepted.\\n'")]
    )
    environment = RecordingEnvironment(
        [
            _process_result(
                "submit",
                stdout="\n  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT  \nAccepted.\n",
            )
        ]
    )

    result = MinimalBashAgent(model, model_name="test-model").run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-leading-whitespace",
    )

    assert result.exit_status == "Submitted"
    assert result.submission == "Accepted.\n"


def test_submission_stops_remaining_parallel_calls_and_withholds_atif(tmp_path):
    model = ScriptedModel(
        [
            BashAgentTurn(
                tool_calls=(
                    BashToolCall(
                        tool_call_id="call-submit",
                        command="printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nDone.\\n'",
                    ),
                    BashToolCall(tool_call_id="call-late", command="touch should-not-exist"),
                )
            )
        ]
    )
    agent = MinimalBashAgent(model, model_name="test-model")
    workspace, environment = _create_workspace(tmp_path)

    with workspace:
        result = agent.run(
            _task(),
            environment=environment,
            trajectory_id="trajectory-early-submit",
        )
        assert not (workspace.path / "should-not-exist").exists()

    assert result.exit_status == "Submitted"
    assert len(result.process_results) == 1
    assert result.atif_trajectory is None


def test_rejects_duplicate_tool_call_ids_before_second_command(tmp_path):
    model = ScriptedModel(
        [
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-1", command="true"),)),
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-1", command="touch should-not-exist"),)),
        ]
    )
    agent = MinimalBashAgent(model, model_name="test-model")
    workspace, environment = _create_workspace(tmp_path)

    with workspace:
        with pytest.raises(MinimalBashAgentError, match="duplicate tool_call_id"):
            agent.run(
                _task(),
                environment=environment,
                trajectory_id="trajectory-duplicate-call",
            )
        assert not (workspace.path / "should-not-exist").exists()


def test_rejects_custom_tools_before_calling_model():
    model = ScriptedModel([_submit_turn()])
    environment = RecordingEnvironment([])

    with pytest.raises(MinimalBashAgentError, match="pinned Bash tool"):
        MinimalBashAgent(model, model_name="test-model").run(
            _task(tools=[{"type": "function", "function": {"name": "python"}}]),
            environment=environment,
            trajectory_id="trajectory-custom-tool",
        )

    assert model.requests == []


def test_process_sandbox_environment_preserves_merged_output_order(tmp_path):
    environment = ProcessSandboxBashEnvironment(ProcessSandbox(tmp_path), env={"CUSTOM_VALUE": "visible"})

    result = environment.execute(
        BashToolCall(
            tool_call_id="call-output",
            command="printf '%s' \"$PAGER/$CUSTOM_VALUE\"; printf ' second' >&2; printf ' third'",
        )
    )

    assert result.stdout == "cat/visible second third"
    assert result.stderr == ""


def test_long_observation_is_compacted_for_the_model_but_raw_output_is_retained():
    model = ScriptedModel(
        [
            BashAgentTurn(tool_calls=(BashToolCall(tool_call_id="call-long", command="long"),)),
            _submit_turn(),
        ]
    )
    long_output = "0123456789" * 3
    environment = RecordingEnvironment(
        [
            _process_result("long", stdout=long_output),
            _process_result(
                "submit",
                stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nPatch ready.\n",
            ),
        ]
    )
    config = MinimalBashAgentConfig(observation_max_chars=20)

    result = MinimalBashAgent(model, model_name="test-model", config=config).run(
        _task(),
        environment=environment,
        trajectory_id="trajectory-long-output",
    )

    observation = model.requests[1][0][-1]["content"]
    assert "<output_head>" in observation
    assert "10 characters elided" in observation
    assert "<output_tail>" in observation
    assert result.process_results[0].stdout == long_output


def test_bash_tool_definition_returns_independent_json_objects():
    first = bash_tool_definition()
    second = bash_tool_definition()

    first["function"]["name"] = "changed"

    assert second["function"]["name"] == "bash"


def test_openai_message_adapter_accepts_multiple_calls_and_extra_arguments():
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
                        "arguments": '{"command":"pwd","ignored":true}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"git status"}',
                    },
                },
            ],
        },
        cost=0.4,
        finish_reason="tool_calls",
        raw_response={"id": "response-1"},
    )

    assert turn == BashAgentTurn(
        content="",
        reasoning_content="Inspect the repository.",
        tool_calls=(
            BashToolCall(tool_call_id="call-1", command="pwd"),
            BashToolCall(tool_call_id="call-2", command="git status"),
        ),
        cost=0.4,
        finish_reason="tool_calls",
        raw_response={"id": "response-1"},
    )


@pytest.mark.parametrize(
    ("message", "match"),
    [
        ({"role": "assistant", "tool_calls": []}, "No tool calls found"),
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
            "Unknown tool",
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
            "Error parsing tool call arguments",
        ),
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            "Missing 'command'",
        ),
    ],
)
def test_openai_message_adapter_raises_recoverable_format_errors(message, match):
    with pytest.raises(BashAgentFormatError, match=match):
        bash_agent_turn_from_openai_message(message, raw_response={"id": "response-invalid"})


def test_truncated_openai_response_gets_specific_retry_feedback():
    with pytest.raises(BashAgentFormatError, match="output token limit") as error:
        bash_agent_turn_from_openai_message(
            {"role": "assistant", "content": "unfinished", "tool_calls": []},
            cost=0.5,
            finish_reason="length",
            raw_response={"id": "response-truncated"},
        )

    assert error.value.cost == 0.5
    assert error.value.raw_response == {"id": "response-truncated"}


def test_constructor_config_and_turn_validation_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="step_limit must be non-negative"):
        MinimalBashAgentConfig(step_limit=-1)
    with pytest.raises(ValueError, match="positive even integer"):
        MinimalBashAgentConfig(observation_max_chars=9)
    with pytest.raises(TypeError, match="config must be"):
        MinimalBashAgent(lambda messages, tools: None, model_name="test-model", config={})
    with pytest.raises(TypeError, match="tool_calls must be a tuple"):
        BashAgentTurn(tool_calls=[])
    with pytest.raises(ValueError, match="command must be non-empty"):
        BashToolCall(tool_call_id="call-1", command="")
    with pytest.raises(ValueError, match="shell must be an absolute path"):
        ProcessSandboxBashEnvironment(ProcessSandbox(tmp_path), shell="bash")
