import copy
import json
import os
from pathlib import Path

import pytest

from lmflow.agentic import ProcessSandbox
from lmflow.agentic.scaffolds.mini_swe_agent import (
    BASH_TOOL,
    UPSTREAM_COMMIT,
    UPSTREAM_VERSION,
    DefaultAgent,
    FormatError,
    LMFlowMiniSWEAgentModel,
    ProcessSandboxEnvironment,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="ProcessSandbox currently requires POSIX")

SYSTEM_TEMPLATE = "You are a software engineer with a Bash tool."
INSTANCE_TEMPLATE = "Solve this task: {{task}}"


class RecordingBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, *, messages, tools, model_name, model_kwargs):
        self.requests.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "model_name": model_name,
                "model_kwargs": copy.deepcopy(model_kwargs),
            }
        )
        return copy.deepcopy(self.responses.pop(0))


def _response(command, *, call_id="call-1", content="Working on it.", cost=0.0):
    message = {
        "role": "assistant",
        "content": content,
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
        "cost": cost,
        "raw_response": {"id": f"response-{call_id}", "message": message},
    }


def _make_agent(tmp_path, responses, **agent_overrides):
    backend = RecordingBackend(responses)
    model = LMFlowMiniSWEAgentModel(
        backend,
        model_name="fixture-model",
        model_kwargs={"temperature": 0.0},
        format_error_template="{{ error }}",
    )
    environment = ProcessSandboxEnvironment(ProcessSandbox(tmp_path), cwd=".")
    config = {
        "system_template": SYSTEM_TEMPLATE,
        "instance_template": INSTANCE_TEMPLATE,
        "step_limit": 8,
        "cost_limit": 0.0,
        "max_consecutive_format_errors": 2,
    }
    config.update(agent_overrides)
    return DefaultAgent(model=model, env=environment, **config), backend


def test_vendored_source_identity_is_pinned():
    assert UPSTREAM_VERSION == "2.4.6"
    assert UPSTREAM_COMMIT == "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
    license_path = Path(__file__).parents[2] / "LICENSES" / "mini-swe-agent-MIT.txt"
    assert "Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez" in license_path.read_text()


def test_toolcall_model_adapter_preserves_raw_response_and_strips_internal_extra():
    backend = RecordingBackend([_response("printf hello")])
    model = LMFlowMiniSWEAgentModel(backend, model_name="fixture-model")
    input_messages = [{"role": "user", "content": "task", "extra": {"private": "metadata"}}]

    message = model.query(input_messages)

    assert backend.requests == [
        {
            "messages": [{"role": "user", "content": "task"}],
            "tools": [BASH_TOOL],
            "model_name": "fixture-model",
            "model_kwargs": {},
        }
    ]
    assert message["extra"]["actions"] == [{"command": "printf hello", "tool_call_id": "call-1"}]
    assert message["extra"]["response"]["id"] == "response-call-1"


def test_invalid_tool_call_uses_vendored_format_error_contract():
    response = _response("printf hello")
    response["message"]["tool_calls"][0]["function"]["name"] = "python"
    backend = RecordingBackend([response])
    model = LMFlowMiniSWEAgentModel(backend, model_name="fixture-model", format_error_template="{{ error }}")

    with pytest.raises(FormatError) as error:
        model.query([])

    assert "Unknown tool 'python'" in error.value.messages[0]["content"]
    assert error.value.messages[0]["extra"]["response"]["id"] == "response-call-1"


def test_model_adapter_rejects_non_serializable_raw_response():
    response = _response("printf hello")
    response["raw_response"] = {"nested": object()}
    model = LMFlowMiniSWEAgentModel(RecordingBackend([response]), model_name="fixture-model")

    with pytest.raises(TypeError, match="raw_response must be JSON-compatible"):
        model.query([])


def test_process_environment_merges_output_and_preserves_submit_semantics(tmp_path):
    environment = ProcessSandboxEnvironment(ProcessSandbox(tmp_path))

    result = environment.execute({"command": "printf 'out\\n'; printf 'err\\n' >&2"})

    assert result["returncode"] == 0
    assert result["output"] == "out\nerr\n"
    assert result["extra"]["timed_out"] is False


def test_process_environment_translates_timeout_to_agent_observation(tmp_path):
    environment = ProcessSandboxEnvironment(ProcessSandbox(tmp_path, timeout_seconds=0.05))

    result = environment.execute({"command": "sleep 10"})

    assert result["returncode"] == -1
    assert result["output"] == ""
    assert result["extra"]["timed_out"] is True
    assert result["extra"]["exception_type"] == "TimeoutExpired"
    assert "timed out after 0.05 seconds" in result["exception_info"]


def test_golden_trajectory_executes_two_steps_and_submits(tmp_path):
    agent, backend = _make_agent(
        tmp_path,
        [
            _response("printf 'artifact' > result.txt", call_id="call-edit"),
            _response(
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'",
                call_id="call-submit",
                content="Submitting.",
            ),
        ],
    )

    result = agent.run("Create result.txt")

    assert result == {"exit_status": "Submitted", "submission": "done\n"}
    assert (tmp_path / "result.txt").read_text() == "artifact"
    assert agent.n_calls == 2
    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "exit",
    ]
    assert backend.requests[1]["messages"][-1]["content"] == "<returncode>0</returncode>\n<output>\n</output>"
    serialized = agent.serialize()
    assert serialized["trajectory_format"] == "mini-swe-agent-1.1"
    assert serialized["info"]["mini_version"] == "2.4.6"


def test_golden_trajectory_recovers_from_format_error(tmp_path):
    malformed = {
        "message": {"role": "assistant", "content": "No tool call", "tool_calls": []},
        "finish_reason": "stop",
        "cost": 0.25,
        "raw_response": {"id": "malformed"},
    }
    agent, backend = _make_agent(
        tmp_path,
        [
            malformed,
            _response("printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nrecovered\\n'", call_id="call-submit"),
        ],
    )

    result = agent.run("Recover and submit")

    assert result["exit_status"] == "Submitted"
    assert result["submission"] == "recovered\n"
    assert agent.n_calls == 2
    assert agent.cost == 0.25
    assert backend.requests[1]["messages"][-1] == {
        "role": "user",
        "content": "No tool calls found in the response. Every response MUST include at least one tool call.",
    }


def test_golden_trajectory_enforces_step_budget(tmp_path):
    agent, _ = _make_agent(tmp_path, [_response("printf first")], step_limit=1)

    result = agent.run("Keep working")

    assert result == {"exit_status": "LimitsExceeded", "submission": ""}
    assert agent.n_calls == 1
