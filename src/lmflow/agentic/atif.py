"""Convert the model-visible ATIF v1.7 SFT subset to LMFlow conversation data."""

import json
from collections.abc import Collection, Mapping
from typing import Any, Literal, Optional

_SUPPORTED_SCHEMA_VERSION = "ATIF-v1.7"
_DETERMINISTIC_STEP_VISIBILITIES = frozenset({"model_context", "orchestration_only"})
_ROOT_FIELDS = frozenset(
    {
        "agent",
        "continued_trajectory_ref",
        "extra",
        "final_metrics",
        "notes",
        "schema_version",
        "session_id",
        "steps",
        "subagent_trajectories",
        "trajectory_id",
    }
)
_AGENT_FIELDS = frozenset({"extra", "model_name", "name", "tool_definitions", "version"})
_STEP_FIELDS = frozenset(
    {
        "extra",
        "is_copied_context",
        "llm_call_count",
        "message",
        "metrics",
        "model_name",
        "observation",
        "reasoning_content",
        "reasoning_effort",
        "source",
        "step_id",
        "timestamp",
        "tool_calls",
    }
)
_TOOL_CALL_FIELDS = frozenset({"arguments", "extra", "function_name", "tool_call_id"})
_OBSERVATION_FIELDS = frozenset({"results"})
_OBSERVATION_RESULT_FIELDS = frozenset({"content", "extra", "source_call_id", "subagent_trajectory_ref"})


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
    if not value:
        raise ValueError(f"{path} must not be empty")
    return value


def _optional_text(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _as_text(value, path)


def _optional_nonnegative_int(value: Any, path: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _normalize_deterministic_step_visibility(
    value: Optional[Mapping[int, Literal["model_context", "orchestration_only"]]],
) -> dict[int, Literal["model_context", "orchestration_only"]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("deterministic_step_visibility must be a mapping")

    normalized = {}
    for step_id, visibility in value.items():
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 1:
            raise ValueError("deterministic_step_visibility step IDs must be positive integers")
        if visibility not in _DETERMINISTIC_STEP_VISIBILITIES:
            allowed = ", ".join(sorted(_DETERMINISTIC_STEP_VISIBILITIES))
            raise ValueError(f"deterministic_step_visibility[{step_id}] must be one of: {allowed}")
        normalized[step_id] = visibility
    return normalized


def _select_model_visible_tools(
    tools: list[dict[str, Any]],
    defined_tool_names: set[str],
    value: Optional[Collection[str]],
) -> tuple[list[dict[str, Any]], set[str]]:
    if value is None:
        return tools, defined_tool_names
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Collection):
        raise ValueError("model_visible_tool_names must be a collection of tool names")

    visible_tool_names = set()
    for index, raw_name in enumerate(value):
        name = _as_nonempty_text(raw_name, f"model_visible_tool_names[{index}]")
        if name in visible_tool_names:
            raise ValueError(f"model_visible_tool_names duplicates {name!r}")
        visible_tool_names.add(name)

    unknown_tool_names = visible_tool_names - defined_tool_names
    if unknown_tool_names:
        unknown = ", ".join(sorted(unknown_tool_names))
        raise ValueError(f"model_visible_tool_names references undefined tools: {unknown}")

    visible_tools = [tool for tool in tools if tool["function"]["name"] in visible_tool_names]
    return visible_tools, visible_tool_names


def _reject_unknown_fields(value: Mapping[str, Any], allowed_fields: frozenset[str], path: str) -> None:
    unknown_fields = set(value) - allowed_fields
    if unknown_fields:
        unknown = ", ".join(sorted(repr(field) for field in unknown_fields))
        raise ValueError(f"{path} contains unsupported fields: {unknown}")


def _json_copy(value: Any, path: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain JSON-compatible values") from error


def _convert_tool_definitions(agent: Mapping[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    raw_tools = agent.get("tool_definitions")
    if raw_tools is None:
        return [], set()

    tools = _as_list(raw_tools, "agent.tool_definitions")
    tool_names = set()
    for index, tool in enumerate(tools):
        path = f"agent.tool_definitions[{index}]"
        tool = _as_mapping(tool, path)
        if tool.get("type") != "function":
            raise ValueError(f"{path}.type must be 'function'")
        function = _as_mapping(tool.get("function"), f"{path}.function")
        function_name = _as_nonempty_text(function.get("name"), f"{path}.function.name")
        if function_name in tool_names:
            raise ValueError(f"{path}.function.name duplicates {function_name!r}")
        tool_names.add(function_name)

    return _json_copy(tools, "agent.tool_definitions"), tool_names


def _convert_tool_calls(
    raw_calls: Any,
    step_path: str,
    seen_call_ids: set[str],
    defined_tool_names: set[str],
    *,
    require_defined_tools: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if raw_calls is None:
        return [], {}

    calls = _as_list(raw_calls, f"{step_path}.tool_calls")
    converted = []
    call_names = {}
    for index, call in enumerate(calls):
        path = f"{step_path}.tool_calls[{index}]"
        call = _as_mapping(call, path)
        _reject_unknown_fields(call, _TOOL_CALL_FIELDS, path)
        call_id = _as_nonempty_text(call.get("tool_call_id"), f"{path}.tool_call_id")
        if call_id in seen_call_ids:
            raise ValueError(f"{path}.tool_call_id duplicates {call_id!r}")

        function_name = _as_nonempty_text(call.get("function_name"), f"{path}.function_name")
        if require_defined_tools and function_name not in defined_tool_names:
            raise ValueError(f"{path}.function_name references undefined tool {function_name!r}")
        arguments = _as_mapping(call.get("arguments"), f"{path}.arguments")
        try:
            arguments_json = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}.arguments must contain JSON-compatible values") from error

        converted.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments_json,
                },
            }
        )
        call_names[call_id] = function_name
        seen_call_ids.add(call_id)

    return converted, call_names


def _append_tool_observations(
    messages: list[dict[str, Any]],
    raw_observation: Any,
    call_names: Mapping[str, str],
    step_path: str,
) -> None:
    if raw_observation is None:
        if call_names:
            raise ValueError(f"{step_path}.observation is required for tool calls")
        return

    if not call_names:
        raise ValueError(f"{step_path}.observation without tool calls is not supported")

    observation = _as_mapping(raw_observation, f"{step_path}.observation")
    _reject_unknown_fields(observation, _OBSERVATION_FIELDS, f"{step_path}.observation")
    results = _as_list(observation.get("results"), f"{step_path}.observation.results")
    observed_content = {}
    for index, result in enumerate(results):
        path = f"{step_path}.observation.results[{index}]"
        result = _as_mapping(result, path)
        _reject_unknown_fields(result, _OBSERVATION_RESULT_FIELDS, path)
        if result.get("subagent_trajectory_ref") not in (None, []):
            raise ValueError(f"{path}.subagent_trajectory_ref is not supported")

        call_id = _as_nonempty_text(result.get("source_call_id"), f"{path}.source_call_id")
        if call_id not in call_names:
            raise ValueError(f"{path}.source_call_id references unknown tool call {call_id!r}")
        if call_id in observed_content:
            raise ValueError(f"{path}.source_call_id duplicates observation for {call_id!r}")

        observed_content[call_id] = _as_text(result.get("content"), f"{path}.content")

    missing_call_ids = set(call_names) - set(observed_content)
    if missing_call_ids:
        missing = ", ".join(sorted(missing_call_ids))
        raise ValueError(f"{step_path}.observation is missing results for tool calls: {missing}")

    for call_id, function_name in call_names.items():
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": function_name,
                "content": observed_content[call_id],
            }
        )


def atif_trajectory_to_conversation(
    trajectory: Mapping[str, Any],
    *,
    deterministic_step_visibility: Optional[Mapping[int, Literal["model_context", "orchestration_only"]]] = None,
    model_visible_tool_names: Optional[Collection[str]] = None,
) -> dict[str, Any]:
    """Convert one linear, text-only ATIF v1.7 trajectory for conversation SFT.

    The adapter intentionally supports only model-visible, linear trajectories with
    at most one LLM inference per agent step. Copied assistant steps remain in
    context with ``loss=false``. ATIF does not record whether an ordinary
    deterministic agent step was included in a later model prompt, so every
    ``llm_call_count=0`` step must be classified by step ID as ``model_context``
    or ``orchestration_only``. The former is retained with ``loss=false``; the
    latter is validated and omitted. A copied deterministic step is already
    asserted to be context and does not need a separate classification. A dispatch
    hidden from the model whose observation was injected separately cannot be
    reconstructed from ATIF alone and remains unsupported.

    Deterministic steps are supported only when they contain structured tool calls
    with complete text observations and no reasoning or model metrics. Context
    replacement, aggregated multi-call steps, subagents, continuations, multimodal
    content, and non-tool observations fail closed. Every agent step must explicitly
    set ``llm_call_count`` to 0 or 1 so its SFT attribution is known. Whenever a
    deterministic step is present, callers must also provide the exact top-level
    tools shown to the model through ``model_visible_tool_names``. Tool call IDs
    must be unique across the trajectory.

    Keeping copied steps as input while excluding their assistant labels is LMFlow's
    context-preserving SFT interpretation; it is intentionally different from
    deleting those steps from the constructed conversation.

    The returned object is one LMFlow ``conversation`` instance. Callers retain the
    source ATIF document or a sidecar manifest for provenance beyond
    ``conversation_id``.
    """

    visibility_by_step = _normalize_deterministic_step_visibility(deterministic_step_visibility)
    trajectory = _as_mapping(trajectory, "trajectory")
    _reject_unknown_fields(trajectory, _ROOT_FIELDS, "trajectory")
    schema_version = _as_text(trajectory.get("schema_version"), "schema_version")
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SUPPORTED_SCHEMA_VERSION!r}; got {schema_version!r}")

    if trajectory.get("continued_trajectory_ref") is not None:
        raise ValueError("continued_trajectory_ref is not supported")
    if trajectory.get("subagent_trajectories") not in (None, []):
        raise ValueError("subagent_trajectories are not supported")

    trajectory_id = _optional_text(trajectory.get("trajectory_id"), "trajectory_id")
    session_id = _optional_text(trajectory.get("session_id"), "session_id")
    conversation_id = trajectory_id or session_id

    agent = _as_mapping(trajectory.get("agent"), "agent")
    _reject_unknown_fields(agent, _AGENT_FIELDS, "agent")
    _as_nonempty_text(agent.get("name"), "agent.name")
    _as_nonempty_text(agent.get("version"), "agent.version")
    tools, defined_tool_names = _convert_tool_definitions(agent)
    tools, visible_tool_names = _select_model_visible_tools(
        tools,
        defined_tool_names,
        model_visible_tool_names,
    )

    steps = _as_list(trajectory.get("steps"), "steps")
    if not steps:
        raise ValueError("steps must not be empty")

    system = None
    messages: list[dict[str, Any]] = []
    has_trainable_assistant_step = False
    seen_step_ids: set[int] = set()
    seen_call_ids: set[str] = set()
    for index, raw_step in enumerate(steps):
        step_path = f"steps[{index}]"
        step = _as_mapping(raw_step, step_path)
        _reject_unknown_fields(step, _STEP_FIELDS, step_path)

        step_id = step.get("step_id")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id != index + 1:
            raise ValueError(f"{step_path}.step_id must equal {index + 1}")
        seen_step_ids.add(step_id)

        is_copied_context = step.get("is_copied_context")
        if is_copied_context is not None and not isinstance(is_copied_context, bool):
            raise ValueError(f"{step_path}.is_copied_context must be a boolean")

        extra = step.get("extra")
        if extra is not None:
            extra = _as_mapping(extra, f"{step_path}.extra")
            if extra.get("context_management") is not None:
                raise ValueError(f"{step_path}.extra.context_management is not supported")

        source = _as_text(step.get("source"), f"{step_path}.source")
        content = _as_text(step.get("message"), f"{step_path}.message")
        llm_call_count = _optional_nonnegative_int(
            step.get("llm_call_count"),
            f"{step_path}.llm_call_count",
        )
        declared_visibility = visibility_by_step.get(step_id)
        is_deterministic_dispatch = source == "agent" and llm_call_count == 0
        if declared_visibility is not None and not is_deterministic_dispatch:
            raise ValueError(
                f"deterministic_step_visibility[{step_id}] is only valid for source='agent' steps with llm_call_count=0"
            )
        if source == "system":
            if step.get("tool_calls") not in (None, []):
                raise ValueError(f"{step_path}.tool_calls are only supported on agent steps")
            if step.get("reasoning_content") is not None:
                raise ValueError(f"{step_path}.reasoning_content is only supported on agent steps")
            if step.get("observation") is not None:
                raise ValueError(f"{step_path}.observation on system steps is not supported")
            if system is None and not messages:
                system = content
            else:
                messages.append({"role": "system", "content": content})
            continue

        if source == "user":
            if step.get("tool_calls") not in (None, []):
                raise ValueError(f"{step_path}.tool_calls are only supported on agent steps")
            if step.get("reasoning_content") is not None:
                raise ValueError(f"{step_path}.reasoning_content is only supported on agent steps")
            if step.get("observation") is not None:
                raise ValueError(f"{step_path}.observation on user steps is not supported")
            messages.append({"role": "user", "content": content})
            continue

        if source != "agent":
            raise ValueError(f"{step_path}.source must be 'system', 'user', or 'agent'")

        if llm_call_count is None:
            raise ValueError(
                f"{step_path}.llm_call_count must be 0 or 1 for SFT attribution; "
                "null or omitted means the producer did not track it"
            )
        if llm_call_count is not None and llm_call_count > 1:
            raise ValueError(f"{step_path}.llm_call_count greater than 1 is not supported")
        if is_deterministic_dispatch:
            if step.get("reasoning_content") is not None:
                raise ValueError(f"{step_path}.reasoning_content must be absent when llm_call_count is 0")
            if step.get("metrics") is not None:
                raise ValueError(f"{step_path}.metrics must be absent when llm_call_count is 0")
            if model_visible_tool_names is None:
                raise ValueError(
                    f"{step_path}.llm_call_count=0 requires model_visible_tool_names "
                    "because ATIF tool definitions do not prove prompt visibility"
                )
            if is_copied_context is not True and declared_visibility is None:
                raise ValueError(
                    f"{step_path}.llm_call_count=0 requires an explicit "
                    "deterministic_step_visibility classification because ATIF "
                    "does not record model-context visibility"
                )

        is_orchestration_only = is_deterministic_dispatch and declared_visibility == "orchestration_only"
        tool_calls, call_names = _convert_tool_calls(
            step.get("tool_calls"),
            step_path,
            seen_call_ids,
            defined_tool_names,
            require_defined_tools=not is_orchestration_only,
        )
        if is_deterministic_dispatch and not tool_calls:
            raise ValueError(f"{step_path}.tool_calls must not be empty when llm_call_count is 0")

        if is_orchestration_only:
            if is_copied_context is True:
                raise ValueError(f"deterministic_step_visibility[{step_id}] conflicts with is_copied_context=true")
            _append_tool_observations([], step.get("observation"), call_names, step_path)
            continue

        hidden_call_tools = set(call_names.values()) - visible_tool_names
        if hidden_call_tools:
            hidden = ", ".join(sorted(hidden_call_tools))
            raise ValueError(f"{step_path}.tool_calls use tools not visible to the model: {hidden}")

        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        reasoning_content = step.get("reasoning_content")
        if reasoning_content is not None:
            reasoning_content = _as_text(reasoning_content, f"{step_path}.reasoning_content")
            assistant_message["reasoning_content"] = reasoning_content
        if not content and not reasoning_content and not tool_calls:
            raise ValueError(f"{step_path} contains no assistant content")
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        excluded_from_loss = is_copied_context is True or is_deterministic_dispatch
        if excluded_from_loss:
            assistant_message["loss"] = False
        messages.append(assistant_message)
        if not excluded_from_loss:
            has_trainable_assistant_step = True

        _append_tool_observations(messages, step.get("observation"), call_names, step_path)

    unknown_visibility_steps = set(visibility_by_step) - seen_step_ids
    if unknown_visibility_steps:
        unknown = ", ".join(str(step_id) for step_id in sorted(unknown_visibility_steps))
        raise ValueError("deterministic_step_visibility references missing step IDs: " + unknown)

    if not has_trainable_assistant_step:
        raise ValueError("trajectory contains no trainable agent steps")

    return {
        "conversation_id": conversation_id,
        "system": system,
        "tools": tools,
        "messages": messages,
    }
