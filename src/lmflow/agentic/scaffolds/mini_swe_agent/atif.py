"""Export pinned mini-swe-agent artifacts through the ATIF SFT boundary."""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from lmflow.agentic.atif import atif_trajectory_to_conversation
from lmflow.agentic.scaffolds.mini_swe_agent._vendor import UPSTREAM_COMMIT, UPSTREAM_VERSION
from lmflow.agentic.scaffolds.mini_swe_agent._vendor.toolcalls import BASH_TOOL
from lmflow.agentic.scaffolds.mini_swe_agent.runner import load_mini_swe_agent_artifact

PathLike = str | os.PathLike[str]

_TRAJECTORY_FORMAT = "mini-swe-agent-1.1"
_ATIF_SCHEMA_VERSION = "ATIF-v1.7"
_AGENT_TYPE = "lmflow.agentic.scaffolds.mini_swe_agent._vendor.agent.DefaultAgent"
_MODEL_TYPE = "lmflow.agentic.scaffolds.mini_swe_agent.adapters.LMFlowMiniSWEAgentModel"
_ENVIRONMENT_TYPE = "lmflow.agentic.scaffolds.mini_swe_agent.adapters.ProcessSandboxEnvironment"
_FULL_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")

_TEXT_MESSAGE_FIELDS = frozenset({"content", "extra", "role"})
_ASSISTANT_MESSAGE_FIELDS = frozenset(
    {
        "annotations",
        "audio",
        "content",
        "extra",
        "function_call",
        "reasoning_content",
        "refusal",
        "role",
        "tool_calls",
    }
)
_TOOL_MESSAGE_FIELDS = frozenset({"content", "extra", "role", "tool_call_id"})
_EXIT_MESSAGE_FIELDS = frozenset({"content", "extra", "role"})
_ACTION_FIELDS = frozenset({"command", "tool_call_id"})
_TOOL_CALL_FIELDS = frozenset({"function", "id", "type"})
_FUNCTION_FIELDS = frozenset({"arguments", "name"})


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _as_text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _as_nonempty_text(value: Any, path: str) -> str:
    value = _as_text(value, path)
    if not value or "\0" in value:
        raise ValueError(f"{path} must be a non-empty string without NUL characters")
    return value


def _reject_unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        fields = ", ".join(sorted(repr(field) for field in unknown))
        raise ValueError(f"{path} contains unsupported fields: {fields}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _parse_arguments(value: Any, path: str) -> dict[str, Any]:
    arguments_text = _as_text(value, path)
    try:
        arguments = json.loads(
            arguments_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path} must contain a strict JSON object") from error
    if not isinstance(arguments, dict):
        raise ValueError(f"{path} must contain a strict JSON object")
    return arguments


def _validate_info(trajectory: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if trajectory.get("trajectory_format") != _TRAJECTORY_FORMAT:
        raise ValueError(f"trajectory.trajectory_format must be {_TRAJECTORY_FORMAT!r}")

    info = _as_mapping(trajectory.get("info"), "trajectory.info")
    if info.get("mini_version") != UPSTREAM_VERSION:
        raise ValueError(f"trajectory.info.mini_version must be {UPSTREAM_VERSION!r}")

    config = _as_mapping(info.get("config"), "trajectory.info.config")
    expected_types = {
        "agent_type": _AGENT_TYPE,
        "model_type": _MODEL_TYPE,
        "environment_type": _ENVIRONMENT_TYPE,
    }
    for field, expected in expected_types.items():
        if config.get(field) != expected:
            raise ValueError(f"trajectory.info.config.{field} must be {expected!r}")

    model_config = _as_mapping(config.get("model"), "trajectory.info.config.model")
    if model_config.get("multimodal_regex") != "":
        raise ValueError("multimodal mini-swe-agent trajectories are not supported for SFT export")
    _as_nonempty_text(model_config.get("model_name"), "trajectory.info.config.model.model_name")

    lmflow = _as_mapping(info.get("lmflow"), "trajectory.info.lmflow")
    for field in ("task_id", "rollout_id"):
        _as_nonempty_text(lmflow.get(field), f"trajectory.info.lmflow.{field}")
    _as_text(lmflow.get("task"), "trajectory.info.lmflow.task")
    base_revision = _as_nonempty_text(lmflow.get("base_revision"), "trajectory.info.lmflow.base_revision")
    if _FULL_COMMIT_PATTERN.fullmatch(base_revision) is None:
        raise ValueError("trajectory.info.lmflow.base_revision must be a full Git commit")
    if lmflow.get("scaffold_commit") != UPSTREAM_COMMIT:
        raise ValueError(f"trajectory.info.lmflow.scaffold_commit must be {UPSTREAM_COMMIT!r}")

    _as_nonempty_text(info.get("exit_status"), "trajectory.info.exit_status")
    _as_text(info.get("submission"), "trajectory.info.submission")
    return dict(info), dict(config), dict(lmflow)


def _validate_exit(raw_exit: Any, info: Mapping[str, Any], path: str) -> None:
    exit_message = _as_mapping(raw_exit, path)
    _reject_unknown_fields(exit_message, _EXIT_MESSAGE_FIELDS, path)
    if exit_message.get("role") != "exit":
        raise ValueError(f"{path}.role must be 'exit'")
    content = _as_text(exit_message.get("content"), f"{path}.content")
    extra = _as_mapping(exit_message.get("extra"), f"{path}.extra")
    if extra.get("exit_status") != info["exit_status"] or extra.get("submission") != info["submission"]:
        raise ValueError(f"{path}.extra must match trajectory.info exit metadata")
    if info["exit_status"] == "Submitted" and content != info["submission"]:
        raise ValueError(f"{path}.content must match the submitted output")


def _convert_text_step(raw_message: Any, path: str, step_id: int) -> dict[str, Any]:
    message = _as_mapping(raw_message, path)
    _reject_unknown_fields(message, _TEXT_MESSAGE_FIELDS, path)
    role = message.get("role")
    if role not in {"system", "user"}:
        raise ValueError(f"{path}.role must be 'system' or 'user'")
    content = _as_text(message.get("content"), f"{path}.content")
    if message.get("extra") is not None:
        _as_mapping(message.get("extra"), f"{path}.extra")
    return {
        "step_id": step_id,
        "source": role,
        "message": content,
    }


def _placeholder_is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _convert_assistant_step(
    raw_message: Any,
    path: str,
    step_id: int,
    seen_call_ids: set[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    message = _as_mapping(raw_message, path)
    _reject_unknown_fields(message, _ASSISTANT_MESSAGE_FIELDS, path)
    if message.get("role") != "assistant":
        raise ValueError(f"{path}.role must be 'assistant'")
    for field in ("annotations", "audio", "function_call", "refusal"):
        if not _placeholder_is_empty(message.get(field)):
            raise ValueError(f"{path}.{field} is not supported")

    raw_tool_calls = _as_list(message.get("tool_calls"), f"{path}.tool_calls")
    if not raw_tool_calls:
        raise ValueError(f"{path}.tool_calls must not be empty")
    tool_calls = []
    call_names = {}
    commands = []
    for call_index, raw_call in enumerate(raw_tool_calls):
        call_path = f"{path}.tool_calls[{call_index}]"
        call = _as_mapping(raw_call, call_path)
        _reject_unknown_fields(call, _TOOL_CALL_FIELDS, call_path)
        call_id = _as_nonempty_text(call.get("id"), f"{call_path}.id")
        if call_id in seen_call_ids:
            raise ValueError(f"{call_path}.id duplicates tool call {call_id!r}")
        if call.get("type") != "function":
            raise ValueError(f"{call_path}.type must be 'function'")
        function = _as_mapping(call.get("function"), f"{call_path}.function")
        _reject_unknown_fields(function, _FUNCTION_FIELDS, f"{call_path}.function")
        if function.get("name") != "bash":
            raise ValueError(f"{call_path}.function.name must be 'bash'")
        arguments = _parse_arguments(function.get("arguments"), f"{call_path}.function.arguments")
        command = _as_text(arguments.get("command"), f"{call_path}.function.arguments.command")
        tool_calls.append(
            {
                "tool_call_id": call_id,
                "function_name": "bash",
                "arguments": arguments,
            }
        )
        call_names[call_id] = "bash"
        commands.append({"command": command, "tool_call_id": call_id})
        seen_call_ids.add(call_id)

    extra = _as_mapping(message.get("extra"), f"{path}.extra")
    raw_actions = _as_list(extra.get("actions"), f"{path}.extra.actions")
    actions = []
    for action_index, raw_action in enumerate(raw_actions):
        action_path = f"{path}.extra.actions[{action_index}]"
        action = _as_mapping(raw_action, action_path)
        _reject_unknown_fields(action, _ACTION_FIELDS, action_path)
        actions.append(
            {
                "command": _as_text(action.get("command"), f"{action_path}.command"),
                "tool_call_id": _as_nonempty_text(action.get("tool_call_id"), f"{action_path}.tool_call_id"),
            }
        )
    if actions != commands:
        raise ValueError(f"{path}.extra.actions must match the accepted assistant tool calls")

    if "content" not in message:
        raise ValueError(f"{path}.content must be present")
    content_value = message["content"]
    if content_value is None:
        content = ""
    else:
        content = _as_text(content_value, f"{path}.content")
    step: dict[str, Any] = {
        "step_id": step_id,
        "source": "agent",
        "message": content,
        "llm_call_count": 1,
        "tool_calls": tool_calls,
    }
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None:
        step["reasoning_content"] = _as_text(reasoning_content, f"{path}.reasoning_content")
    return step, call_names


def _append_observations(
    step: dict[str, Any],
    raw_messages: list[Any],
    start_index: int,
    call_names: Mapping[str, str],
) -> int:
    observations = {}
    index = start_index
    while index < len(raw_messages) - 1:
        raw_message = _as_mapping(raw_messages[index], f"trajectory.messages[{index}]")
        if raw_message.get("role") != "tool":
            break
        path = f"trajectory.messages[{index}]"
        _reject_unknown_fields(raw_message, _TOOL_MESSAGE_FIELDS, path)
        call_id = _as_nonempty_text(raw_message.get("tool_call_id"), f"{path}.tool_call_id")
        if call_id not in call_names:
            raise ValueError(f"{path}.tool_call_id references unknown tool call {call_id!r}")
        if call_id in observations:
            raise ValueError(f"{path}.tool_call_id duplicates observation for {call_id!r}")
        content = _as_text(raw_message.get("content"), f"{path}.content")
        if raw_message.get("extra") is not None:
            _as_mapping(raw_message.get("extra"), f"{path}.extra")
        observations[call_id] = content
        index += 1

    if observations:
        missing_call_ids = set(call_names) - set(observations)
        if missing_call_ids:
            missing = ", ".join(sorted(missing_call_ids))
            raise ValueError(f"trajectory.messages[{start_index}] is missing tool observations: {missing}")
        step["observation"] = {
            "results": [
                {
                    "source_call_id": call_id,
                    "content": observations[call_id],
                }
                for call_id in call_names
            ]
        }
    return index


def _trajectory_id(task_id: str, rollout_id: str) -> str:
    return f"lmflow-mini-swe:{quote(task_id, safe='')}:{quote(rollout_id, safe='')}"


def mini_swe_agent_trajectory_to_atif(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one exact LMFlow mini-swe-agent raw trajectory to ATIF v1.7.

    The raw artifact remains the audit source. This exporter retains only the
    model-visible text/tool history accepted by the pinned scaffold. Provider
    payloads, environment diagnostics, and the terminal control message remain
    in the raw trajectory and are summarized only as stable provenance.
    """

    trajectory = _as_mapping(trajectory, "trajectory")
    info, config, lmflow = _validate_info(trajectory)
    raw_messages = _as_list(trajectory.get("messages"), "trajectory.messages")
    if len(raw_messages) < 3:
        raise ValueError("trajectory.messages must contain system, user, and exit messages")
    _validate_exit(raw_messages[-1], info, f"trajectory.messages[{len(raw_messages) - 1}]")

    first_role = _as_mapping(raw_messages[0], "trajectory.messages[0]").get("role")
    second_role = _as_mapping(raw_messages[1], "trajectory.messages[1]").get("role")
    if (first_role, second_role) != ("system", "user"):
        raise ValueError("trajectory.messages must start with system and user messages")

    steps = []
    seen_call_ids: set[str] = set()
    message_index = 0
    while message_index < len(raw_messages) - 1:
        path = f"trajectory.messages[{message_index}]"
        raw_message = _as_mapping(raw_messages[message_index], path)
        role = raw_message.get("role")
        step_id = len(steps) + 1
        if role in {"system", "user"}:
            if role == "system" and message_index != 0:
                raise ValueError(f"{path}.role introduces an unsupported later system message")
            steps.append(_convert_text_step(raw_message, path, step_id))
            message_index += 1
            continue
        if role != "assistant":
            raise ValueError(f"{path}.role is not valid at this point in the trajectory")

        step, call_names = _convert_assistant_step(raw_message, path, step_id, seen_call_ids)
        next_index = _append_observations(step, raw_messages, message_index + 1, call_names)
        is_terminal_agent_step = next_index == len(raw_messages) - 1
        if "observation" not in step and not is_terminal_agent_step:
            raise ValueError(f"{path} is missing tool observations before a later model-visible message")
        steps.append(step)
        message_index = next_index

    model_config = _as_mapping(config["model"], "trajectory.info.config.model")
    trajectory_id = _trajectory_id(lmflow["task_id"], lmflow["rollout_id"])
    result = {
        "schema_version": _ATIF_SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "session_id": trajectory_id,
        "agent": {
            "name": "mini-swe-agent",
            "version": UPSTREAM_VERSION,
            "model_name": model_config["model_name"],
            "tool_definitions": [copy.deepcopy(BASH_TOOL)],
            "extra": {"scaffold_commit": UPSTREAM_COMMIT},
        },
        "steps": steps,
        "extra": {
            "lmflow": {
                "source_format": _TRAJECTORY_FORMAT,
                "task_id": lmflow["task_id"],
                "rollout_id": lmflow["rollout_id"],
                "task": lmflow["task"],
                "base_revision": lmflow["base_revision"],
                "scaffold_commit": lmflow["scaffold_commit"],
                "exit_status": info["exit_status"],
                "submission": info["submission"],
            }
        },
    }
    return result


def mini_swe_agent_artifact_to_atif(artifact_dir: PathLike) -> dict[str, Any]:
    """Load one raw artifact directory and export its trajectory to ATIF."""

    trajectory, _patch = load_mini_swe_agent_artifact(artifact_dir)
    return mini_swe_agent_trajectory_to_atif(trajectory)


def mini_swe_agent_artifact_to_conversation(artifact_dir: PathLike) -> dict[str, Any]:
    """Load one raw artifact and produce an LMFlow conversation SFT example."""

    return atif_trajectory_to_conversation(mini_swe_agent_artifact_to_atif(artifact_dir))


__all__ = [
    "mini_swe_agent_artifact_to_atif",
    "mini_swe_agent_artifact_to_conversation",
    "mini_swe_agent_trajectory_to_atif",
]
