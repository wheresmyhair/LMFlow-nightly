import copy
import json

import pytest

from lmflow.agentic import (
    mini_swe_agent_artifact_to_atif,
    mini_swe_agent_artifact_to_conversation,
    mini_swe_agent_trajectory_to_atif,
)
from lmflow.agentic.scaffolds.mini_swe_agent import load_mini_swe_agent_artifact


def _assistant_message(command, call_id, *, content="Working.", reasoning_content=None, response_marker=None):
    arguments = json.dumps({"command": command})
    message = {
        "role": "assistant",
        "content": content,
        "refusal": None,
        "annotations": [],
        "audio": None,
        "function_call": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": arguments},
            }
        ],
        "extra": {
            "actions": [{"command": command, "tool_call_id": call_id}],
            "response": {"audit_marker": response_marker or f"raw-{call_id}"},
            "cost": 0.0,
            "timestamp": 1_786_440_000.0,
        },
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return message


def _raw_trajectory():
    return {
        "info": {
            "model_stats": {"instance_cost": 0.0, "api_calls": 2},
            "config": {
                "agent": {
                    "system_template": "You are a software engineer with a Bash tool.",
                    "instance_template": "Solve this task: {{task}}",
                    "step_limit": 4,
                    "cost_limit": 0.0,
                    "wall_time_limit_seconds": 0,
                    "max_consecutive_format_errors": 3,
                    "output_path": None,
                },
                "agent_type": "lmflow.agentic.scaffolds.mini_swe_agent._vendor.agent.DefaultAgent",
                "model": {
                    "model_name": "fixture-model",
                    "model_kwargs": {"temperature": 0.0},
                    "format_error_template": "{{ error }}",
                    "observation_template": "<returncode>{{output.returncode}}</returncode>",
                    "multimodal_regex": "",
                },
                "model_type": "lmflow.agentic.scaffolds.mini_swe_agent.adapters.LMFlowMiniSWEAgentModel",
                "environment": {"cwd": ".", "env": {}, "timeout_seconds": None},
                "environment_type": ("lmflow.agentic.scaffolds.mini_swe_agent.adapters.ProcessSandboxEnvironment"),
                "sandbox": {
                    "sandbox_type": "lmflow.agentic.sandbox.ProcessSandbox",
                    "timeout_seconds": 60.0,
                    "max_output_bytes": 1_000_000,
                    "limits": {},
                    "capabilities": {},
                },
            },
            "mini_version": "2.4.6",
            "exit_status": "Submitted",
            "submission": "done\n",
            "lmflow": {
                "task_id": "org/repo__1",
                "rollout_id": "rollout:0",
                "task": "Update README.md",
                "base_revision": "0123456789abcdef0123456789abcdef01234567",
                "scaffold_commit": "a83fcae82d2a08f0ee0c688f9d137b3566c097f8",
            },
        },
        "messages": [
            {"role": "system", "content": "SYSTEM_TOKEN"},
            {"role": "user", "content": "USER_TOKEN"},
            _assistant_message(
                "printf 'changed\\n' > README.md",
                "call-edit",
                content=None,
                reasoning_content="REASON_TOKEN",
            ),
            {
                "role": "tool",
                "tool_call_id": "call-edit",
                "content": "<returncode>0</returncode>\n<output>OBS_TOKEN</output>",
                "extra": {"raw_output": "OBS_TOKEN", "returncode": 0},
            },
            _assistant_message(
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'",
                "call-submit",
                content="Submitting.",
            ),
            {
                "role": "exit",
                "content": "done\n",
                "extra": {"exit_status": "Submitted", "submission": "done\n"},
            },
        ],
        "trajectory_format": "mini-swe-agent-1.1",
    }


def _write_artifact(tmp_path, trajectory=None):
    artifact_dir = tmp_path / "episode"
    artifact_dir.mkdir()
    (artifact_dir / "trajectory.json").write_text(
        json.dumps(trajectory or _raw_trajectory(), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (artifact_dir / "model.patch").write_bytes(b"diff --git a/README.md b/README.md\n")
    return artifact_dir


def test_exports_model_visible_history_through_atif_without_mutating_raw_trajectory():
    trajectory = _raw_trajectory()
    original = copy.deepcopy(trajectory)

    atif = mini_swe_agent_trajectory_to_atif(trajectory)

    assert trajectory == original
    assert atif["schema_version"] == "ATIF-v1.7"
    assert atif["trajectory_id"] == "lmflow-mini-swe:org%2Frepo__1:rollout%3A0"
    assert atif["session_id"] == atif["trajectory_id"]
    assert atif["agent"]["name"] == "mini-swe-agent"
    assert atif["agent"]["model_name"] == "fixture-model"
    assert [step["source"] for step in atif["steps"]] == ["system", "user", "agent", "agent"]
    first_agent_step = atif["steps"][2]
    assert first_agent_step["message"] == ""
    assert first_agent_step["reasoning_content"] == "REASON_TOKEN"
    assert first_agent_step["tool_calls"][0]["arguments"] == {"command": "printf 'changed\\n' > README.md"}
    assert first_agent_step["observation"]["results"] == [
        {
            "source_call_id": "call-edit",
            "content": "<returncode>0</returncode>\n<output>OBS_TOKEN</output>",
        }
    ]
    assert "observation" not in atif["steps"][-1]
    assert atif["extra"]["lmflow"]["exit_status"] == "Submitted"
    assert "audit_marker" not in json.dumps(atif)


def test_artifact_composes_loader_atif_and_conversation_boundaries(tmp_path):
    artifact_dir = _write_artifact(tmp_path)

    loaded_trajectory, patch = load_mini_swe_agent_artifact(artifact_dir)
    atif = mini_swe_agent_artifact_to_atif(artifact_dir)
    conversation = mini_swe_agent_artifact_to_conversation(artifact_dir)

    assert loaded_trajectory == _raw_trajectory()
    assert patch == b"diff --git a/README.md b/README.md\n"
    assert atif == mini_swe_agent_trajectory_to_atif(loaded_trajectory)
    assert conversation["conversation_id"] == atif["trajectory_id"]
    assert conversation["system"] == "SYSTEM_TOKEN"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert conversation["messages"][1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command":"printf \'changed\\\\n\' > README.md"}'
    )
    assert conversation["messages"][-1]["tool_calls"][0]["id"] == "call-submit"


def test_preserves_visible_format_correction_but_excludes_rejected_provider_payload():
    trajectory = _raw_trajectory()
    correction = {
        "role": "user",
        "content": "Every response must include a Bash tool call.",
        "extra": {
            "interrupt_type": "FormatError",
            "response": {"rejected_payload": "DO_NOT_TRAIN_RAW_PROVIDER_OBJECT"},
        },
    }
    trajectory["messages"].insert(4, correction)

    atif = mini_swe_agent_trajectory_to_atif(trajectory)

    assert [step["source"] for step in atif["steps"]] == ["system", "user", "agent", "user", "agent"]
    assert atif["steps"][3]["message"] == correction["content"]
    assert "DO_NOT_TRAIN_RAW_PROVIDER_OBJECT" not in json.dumps(atif)


def test_ignores_additive_audit_metadata_outside_model_visible_messages():
    trajectory = _raw_trajectory()
    trajectory["audit_extension"] = {"collector": "fixture"}
    trajectory["info"]["diagnostics"] = {"latency_ms": 12}
    trajectory["messages"][2]["extra"]["backend_diagnostics"] = {"queue_ms": 3}

    atif = mini_swe_agent_trajectory_to_atif(trajectory)

    assert [step["source"] for step in atif["steps"]] == ["system", "user", "agent", "agent"]
    assert "diagnostics" not in json.dumps(atif)


def test_rejects_missing_nonterminal_tool_observation():
    trajectory = _raw_trajectory()
    trajectory["messages"].pop(3)

    with pytest.raises(ValueError, match="missing tool observations before a later model-visible message"):
        mini_swe_agent_trajectory_to_atif(trajectory)


def test_rejects_actions_that_do_not_match_accepted_tool_calls():
    trajectory = _raw_trajectory()
    trajectory["messages"][2]["extra"]["actions"][0]["command"] = "different command"

    with pytest.raises(ValueError, match="actions must match"):
        mini_swe_agent_trajectory_to_atif(trajectory)


def test_preserves_additional_json_tool_arguments_accepted_by_the_scaffold():
    trajectory = _raw_trajectory()
    trajectory["messages"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"command": "printf 'changed\\n' > README.md", "trace_label": "edit"}
    )

    atif = mini_swe_agent_trajectory_to_atif(trajectory)

    assert atif["steps"][2]["tool_calls"][0]["arguments"] == {
        "command": "printf 'changed\\n' > README.md",
        "trace_label": "edit",
    }


def test_preserves_empty_bash_command_accepted_by_the_scaffold():
    trajectory = _raw_trajectory()
    trajectory["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"command":""}'
    trajectory["messages"][2]["extra"]["actions"][0]["command"] = ""

    atif = mini_swe_agent_trajectory_to_atif(trajectory)

    assert atif["steps"][2]["tool_calls"][0]["arguments"] == {"command": ""}


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda trajectory: trajectory["info"]["config"]["model"].__setitem__("multimodal_regex", "image-pattern"),
            "multimodal.*not supported",
        ),
        (
            lambda trajectory: trajectory["messages"][2].__setitem__("refusal", "Cannot comply"),
            "refusal is not supported",
        ),
        (
            lambda trajectory: trajectory["messages"][2].__setitem__("provider_extension", True),
            "unsupported fields",
        ),
    ],
)
def test_fails_closed_for_unsupported_raw_semantics(mutate, error):
    trajectory = _raw_trajectory()
    mutate(trajectory)

    with pytest.raises(ValueError, match=error):
        mini_swe_agent_trajectory_to_atif(trajectory)


@pytest.mark.parametrize(
    "trajectory_json",
    [
        '{"trajectory_format":"mini-swe-agent-1.1","messages":[],"info":{},"info":{}}',
        '{"trajectory_format":"mini-swe-agent-1.1","messages":[],"info":{"value":NaN}}',
    ],
)
def test_artifact_loader_rejects_non_strict_json(tmp_path, trajectory_json):
    artifact_dir = tmp_path / "episode"
    artifact_dir.mkdir()
    (artifact_dir / "trajectory.json").write_text(trajectory_json, encoding="utf-8")
    (artifact_dir / "model.patch").write_bytes(b"")

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_mini_swe_agent_artifact(artifact_dir)
