"""Mini-swe-agent-compatible Bash scaffold for repository tasks.

The control-flow and tool-call behavior in this module is adapted from
mini-swe-agent at commit a83fcae82d2a08f0ee0c688f9d137b3566c097f8.
See ``THIRD_PARTY_NOTICES.md`` and ``LICENSES/mini-swe-agent-MIT.txt``.
"""

import copy
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.sandbox import ProcessResult, ProcessSandbox

_MINI_SWE_AGENT_COMMIT = "a83fcae82d2a08f0ee0c688f9d137b3566c097f8"
_SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_DEFAULT_COMMAND_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}
_BASH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


class MinimalBashAgentError(RuntimeError):
    """Raised when a task or adapter violates the Bash-agent contract."""


class BashAgentFormatError(MinimalBashAgentError):
    """Recoverable model-output error that is fed back into the conversation."""

    def __init__(
        self,
        feedback: str,
        *,
        cost: float = 0.0,
        raw_response: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(feedback)
        self.feedback = _require_nonempty_text(feedback, name="feedback")
        self.cost = _validate_cost(cost)
        self.raw_response = None if raw_response is None else _json_copy(raw_response, name="raw_response")


@dataclass(frozen=True)
class BashToolCall:
    """One normalized Bash tool call returned by a model adapter."""

    tool_call_id: str
    command: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.tool_call_id, name="tool_call_id")
        _require_nonempty_text(self.command, name="command")


@dataclass(frozen=True)
class BashAgentTurn:
    """Normalized model output and accounting data for one agent turn."""

    content: str = ""
    tool_calls: tuple[BashToolCall, ...] = ()
    reasoning_content: Optional[str] = None
    cost: float = 0.0
    finish_reason: Optional[str] = None
    raw_response: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple")
        if any(not isinstance(call, BashToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain only BashToolCall instances")
        if self.reasoning_content is not None and not isinstance(self.reasoning_content, str):
            raise TypeError("reasoning_content must be a string or None")
        _validate_cost(self.cost)
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise TypeError("finish_reason must be a string or None")
        if self.raw_response is not None:
            object.__setattr__(self, "raw_response", _json_copy(self.raw_response, name="raw_response"))


class BashAgentModel(Protocol):
    """Callable boundary implemented by an OpenAI, vLLM, or test adapter."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> BashAgentTurn: ...


class BashAgentEnvironment(Protocol):
    """Environment boundary shared by local, container, and remote backends."""

    def execute(self, call: BashToolCall) -> ProcessResult: ...


class ProcessSandboxBashEnvironment:
    """Expose ``ProcessSandbox`` through the model-facing Bash environment."""

    def __init__(
        self,
        sandbox: ProcessSandbox,
        *,
        shell: str = "/bin/bash",
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(sandbox, ProcessSandbox):
            raise TypeError("sandbox must be a ProcessSandbox")
        _require_nonempty_text(shell, name="shell")
        if not Path(shell).is_absolute():
            raise ValueError("shell must be an absolute path")
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("env must be a mapping of strings or None")
        command_env = dict(_DEFAULT_COMMAND_ENV)
        if env is not None:
            for name, value in env.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise TypeError("env keys and values must be strings")
                command_env[name] = value
        self.sandbox = sandbox
        self.shell = shell
        self.env = command_env

    def execute(self, call: BashToolCall) -> ProcessResult:
        if not isinstance(call, BashToolCall):
            raise TypeError("call must be a BashToolCall")
        return self.sandbox.run(
            (self.shell, "-lc", call.command),
            env=self.env,
            merge_stderr=True,
        )


@dataclass(frozen=True)
class MinimalBashAgentConfig:
    """Budget and formatting settings for the pinned mini-compatible profile."""

    step_limit: int = 250
    cost_limit: float = 3.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    observation_max_chars: int = 10_000

    def __post_init__(self) -> None:
        _validate_nonnegative_int(self.step_limit, name="step_limit")
        _validate_cost(self.cost_limit)
        _validate_nonnegative_int(self.wall_time_limit_seconds, name="wall_time_limit_seconds")
        _validate_nonnegative_int(
            self.max_consecutive_format_errors,
            name="max_consecutive_format_errors",
        )
        if isinstance(self.observation_max_chars, bool) or not isinstance(self.observation_max_chars, int):
            raise TypeError("observation_max_chars must be an integer")
        if self.observation_max_chars <= 0 or self.observation_max_chars % 2:
            raise ValueError("observation_max_chars must be a positive even integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_limit": self.step_limit,
            "cost_limit": self.cost_limit,
            "wall_time_limit_seconds": self.wall_time_limit_seconds,
            "max_consecutive_format_errors": self.max_consecutive_format_errors,
            "observation_max_chars": self.observation_max_chars,
        }


@dataclass(frozen=True)
class MinimalBashAgentResult:
    """Raw rollout facts plus an optional loss-ready ATIF projection."""

    exit_status: Literal[
        "Submitted",
        "LimitsExceeded",
        "TimeExceeded",
        "RepeatedFormatError",
    ]
    submission: str
    raw_trajectory: dict[str, Any]
    atif_trajectory: Optional[dict[str, Any]]
    process_results: tuple[ProcessResult, ...]
    n_model_calls: int
    model_cost: float
    format_error_count: int


def bash_tool_definition() -> dict[str, Any]:
    """Return an independent copy of the pinned mini-swe-agent Bash tool."""

    return copy.deepcopy(_BASH_TOOL_DEFINITION)


def bash_agent_turn_from_openai_message(
    message: Mapping[str, Any],
    *,
    cost: float = 0.0,
    finish_reason: Optional[str] = None,
    raw_response: Optional[Mapping[str, Any]] = None,
) -> BashAgentTurn:
    """Normalize an OpenAI assistant message using mini-swe-agent semantics.

    Missing or malformed tool calls raise ``BashAgentFormatError`` so the agent
    can append a corrective user message and retry within its configured budget.
    Provider adapters should pass the billed cost and complete raw response.
    """

    if not isinstance(message, Mapping):
        raise MinimalBashAgentError("assistant message must be an object")
    role = message.get("role")
    if role not in (None, "assistant"):
        raise MinimalBashAgentError("assistant message role must be 'assistant'")
    validated_cost = _validate_cost(cost)
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TypeError("finish_reason must be a string or None")
    persisted_response = raw_response if raw_response is not None else message

    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise MinimalBashAgentError("assistant message content must be a string or null")
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise MinimalBashAgentError("assistant message reasoning_content must be a string or null")

    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise BashAgentFormatError(
            _format_error_feedback(
                "No tool calls found in the response. Every response MUST include at least one tool call.",
                finish_reason=finish_reason,
                has_tool_calls=False,
            ),
            cost=validated_cost,
            raw_response=persisted_response,
        )

    calls = []
    for index, raw_call in enumerate(raw_calls):
        path = f"assistant message tool_calls[{index}]"
        error_parts = []
        call_id = None
        command = None
        if not isinstance(raw_call, Mapping):
            error_parts.append(f"{path} must be an object.")
        else:
            if raw_call.get("type", "function") != "function":
                error_parts.append(f"{path}.type must be 'function'.")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                error_parts.append(f"{path}.function must be an object.")
            else:
                if function.get("name") != "bash":
                    error_parts.append(f"Unknown tool {function.get('name')!r}.")
                raw_arguments = function.get("arguments")
                if not isinstance(raw_arguments, str):
                    error_parts.append("Tool call arguments must be a JSON string.")
                else:
                    try:
                        arguments = json.loads(raw_arguments)
                    except (TypeError, ValueError) as error:
                        error_parts.append(f"Error parsing tool call arguments: {error}.")
                    else:
                        if not isinstance(arguments, dict) or "command" not in arguments:
                            error_parts.append("Missing 'command' argument in bash tool call.")
                        else:
                            command = arguments["command"]
        if not error_parts:
            try:
                calls.append(BashToolCall(tool_call_id=call_id, command=command))
            except (TypeError, ValueError) as error:
                error_parts.append(str(error))
        if error_parts:
            raise BashAgentFormatError(
                _format_error_feedback(
                    " ".join(error_parts),
                    finish_reason=finish_reason,
                    has_tool_calls=True,
                ),
                cost=validated_cost,
                raw_response=persisted_response,
            )

    return BashAgentTurn(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tuple(calls),
        cost=validated_cost,
        finish_reason=finish_reason,
        raw_response=persisted_response,
    )


class MinimalBashAgent:
    """Run the pinned mini-swe-agent control flow over an LMFlow environment.

    ``TaskSpec`` contains already-rendered system/user messages. Repository
    preparation, patch export, verification, and cleanup remain outer-pipeline
    responsibilities. Raw model messages are retained as the factual rollout;
    ATIF is emitted only when every model-visible transition has a faithful
    representation in the currently supported ATIF subset.
    """

    def __init__(
        self,
        model: BashAgentModel,
        *,
        model_name: str,
        config: Optional[MinimalBashAgentConfig] = None,
        agent_name: str = "lmflow-mini-swe-agent",
        agent_version: str = "1",
    ) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        _require_nonempty_text(model_name, name="model_name")
        if config is not None and not isinstance(config, MinimalBashAgentConfig):
            raise TypeError("config must be a MinimalBashAgentConfig or None")
        _require_nonempty_text(agent_name, name="agent_name")
        _require_nonempty_text(agent_version, name="agent_version")

        self._model = model
        self.model_name = model_name
        self.config = config or MinimalBashAgentConfig()
        self.agent_name = agent_name
        self.agent_version = agent_version

    def run(
        self,
        task: TaskSpec,
        *,
        environment: BashAgentEnvironment,
        trajectory_id: str,
    ) -> MinimalBashAgentResult:
        """Run one rollout and return raw facts plus a safe ATIF projection."""

        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if not callable(getattr(environment, "execute", None)):
            raise TypeError("environment must provide a callable execute() method")
        _require_nonempty_text(trajectory_id, name="trajectory_id")

        messages, atif_steps = _prepare_task(task)
        tools = [bash_tool_definition()]
        if task.tools and _json_copy(task.tools, name="task.tools") != tools:
            raise MinimalBashAgentError("MinimalBashAgent supports only the pinned Bash tool")

        started_at = time.monotonic()
        process_results: list[ProcessResult] = []
        seen_call_ids: set[str] = set()
        n_model_calls = 0
        model_cost = 0.0
        format_error_count = 0
        consecutive_format_errors = 0
        atif_exportable = True
        submission = ""
        exit_status: Literal[
            "Submitted",
            "LimitsExceeded",
            "TimeExceeded",
            "RepeatedFormatError",
        ]

        while True:
            if (0 < self.config.step_limit <= n_model_calls) or (0 < self.config.cost_limit <= model_cost):
                exit_status = "LimitsExceeded"
                messages.append(_exit_message(exit_status, ""))
                break
            elapsed_seconds = int(time.monotonic() - started_at)
            if 0 < self.config.wall_time_limit_seconds <= elapsed_seconds:
                exit_status = "TimeExceeded"
                messages.append(_exit_message(exit_status, ""))
                break

            n_model_calls += 1
            try:
                turn = self._model(_messages_for_model(messages), copy.deepcopy(tools))
            except BashAgentFormatError as error:
                model_cost += error.cost
                format_error_count += 1
                consecutive_format_errors += 1
                atif_exportable = False
                messages.append(
                    {
                        "role": "user",
                        "content": error.feedback,
                        "extra": {
                            "interrupt_type": "FormatError",
                            "cost": error.cost,
                            "response": error.raw_response,
                        },
                    }
                )
                if 0 < self.config.max_consecutive_format_errors <= consecutive_format_errors:
                    exit_status = "RepeatedFormatError"
                    messages.append(_exit_message(exit_status, ""))
                    break
                continue

            if not isinstance(turn, BashAgentTurn):
                raise MinimalBashAgentError("model must return a BashAgentTurn")
            model_cost += turn.cost
            consecutive_format_errors = 0
            if not turn.tool_calls:
                raise MinimalBashAgentError("model adapters must convert missing tool calls into BashAgentFormatError")
            call_ids = [call.tool_call_id for call in turn.tool_calls]
            duplicate_ids = seen_call_ids.intersection(call_ids)
            if len(call_ids) != len(set(call_ids)) or duplicate_ids:
                duplicate_id = next(
                    call_id
                    for index, call_id in enumerate(call_ids)
                    if call_id in seen_call_ids or call_id in call_ids[:index]
                )
                raise MinimalBashAgentError(f"duplicate tool_call_id: {duplicate_id!r}")
            seen_call_ids.update(call_ids)

            messages.append(_assistant_message(turn))
            executed_calls = []
            observations = []
            submitted = False
            for call in turn.tool_calls:
                process_result = environment.execute(call)
                if not isinstance(process_result, ProcessResult):
                    raise MinimalBashAgentError("environment.execute() must return a ProcessResult")
                process_results.append(process_result)
                executed_calls.append(call)
                observation = _observation_from_result(
                    process_result,
                    max_chars=self.config.observation_max_chars,
                )
                observations.append(observation)

                submitted_output = _extract_submission(process_result)
                if submitted_output is not None:
                    submission = submitted_output
                    exit_status = "Submitted"
                    messages.append(_exit_message(exit_status, submission))
                    submitted = True
                    if len(executed_calls) != len(turn.tool_calls):
                        atif_exportable = False
                    break

            atif_steps.append(
                _agent_step(
                    step_id=len(atif_steps) + 1,
                    turn=turn,
                    calls=executed_calls,
                    observations=observations,
                    model_name=self.model_name,
                )
            )
            if submitted:
                break
            for call, observation in zip(executed_calls, observations):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.tool_call_id,
                        "name": "bash",
                        "content": observation,
                    }
                )

        raw_trajectory = {
            "trajectory_format": "lmflow-mini-swe-agent-raw-v1",
            "info": {
                "model_stats": {
                    "instance_cost": model_cost,
                    "api_calls": n_model_calls,
                },
                "config": {
                    "agent": self.config.as_dict(),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    "model_name": self.model_name,
                },
                "upstream": {
                    "project": "SWE-agent/mini-swe-agent",
                    "commit": _MINI_SWE_AGENT_COMMIT,
                },
                "exit_status": exit_status,
                "submission": submission,
                "format_error_count": format_error_count,
            },
            "messages": _json_copy(messages, name="messages"),
        }
        atif_trajectory = None
        if atif_exportable:
            atif_trajectory = {
                "schema_version": "ATIF-v1.7",
                "trajectory_id": trajectory_id,
                "agent": {
                    "name": self.agent_name,
                    "version": self.agent_version,
                    "model_name": self.model_name,
                    "tool_definitions": tools,
                },
                "steps": atif_steps,
                "extra": {
                    "lmflow": {
                        "task_id": task.task_id,
                        "exit_status": exit_status,
                        "submission": submission,
                        "raw_trajectory_format": raw_trajectory["trajectory_format"],
                    }
                },
            }
        return MinimalBashAgentResult(
            exit_status=exit_status,
            submission=submission,
            raw_trajectory=raw_trajectory,
            atif_trajectory=atif_trajectory,
            process_results=tuple(process_results),
            n_model_calls=n_model_calls,
            model_cost=model_cost,
            format_error_count=format_error_count,
        )


def _prepare_task(task: TaskSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(task.task_id, str) or not task.task_id:
        raise MinimalBashAgentError("task.task_id must be a non-empty string")
    if not isinstance(task.messages, list) or not task.messages:
        raise MinimalBashAgentError("task.messages must be a non-empty list")
    if not isinstance(task.tools, list):
        raise MinimalBashAgentError("task.tools must be a list")

    messages = []
    steps = []
    has_user_message = False
    for index, message in enumerate(task.messages):
        if not isinstance(message, dict):
            raise MinimalBashAgentError(f"task.messages[{index}] must be an object")
        unknown_fields = set(message) - {"role", "content"}
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise MinimalBashAgentError(f"task.messages[{index}] contains unsupported fields: {unknown}")
        role = message.get("role")
        if role not in {"system", "user"}:
            raise MinimalBashAgentError(f"task.messages[{index}].role must be 'system' or 'user'")
        content = message.get("content")
        if not isinstance(content, str):
            raise MinimalBashAgentError(f"task.messages[{index}].content must be a string")
        has_user_message = has_user_message or role == "user"
        messages.append({"role": role, "content": content})
        steps.append({"step_id": index + 1, "source": role, "message": content})

    if not has_user_message:
        raise MinimalBashAgentError("task.messages must contain at least one user message")
    return messages, steps


def _messages_for_model(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy({key: value for key, value in message.items() if key != "extra"})
        for message in messages
        if message.get("role") != "exit"
    ]


def _assistant_message(turn: BashAgentTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": call.tool_call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps(
                        {"command": call.command},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            }
            for call in turn.tool_calls
        ],
        "extra": {
            "actions": [{"command": call.command, "tool_call_id": call.tool_call_id} for call in turn.tool_calls],
            "cost": turn.cost,
            "finish_reason": turn.finish_reason,
            "response": turn.raw_response,
            "timestamp": time.time(),
        },
    }
    if turn.reasoning_content is not None:
        message["reasoning_content"] = turn.reasoning_content
    return message


def _agent_step(
    *,
    step_id: int,
    turn: BashAgentTurn,
    calls: list[BashToolCall],
    observations: list[str],
    model_name: str,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "source": "agent",
        "message": turn.content,
        "model_name": model_name,
        "llm_call_count": 1,
        "tool_calls": [
            {
                "tool_call_id": call.tool_call_id,
                "function_name": "bash",
                "arguments": {"command": call.command},
            }
            for call in calls
        ],
        "observation": {
            "results": [
                {
                    "source_call_id": call.tool_call_id,
                    "content": observation,
                }
                for call, observation in zip(calls, observations)
            ]
        },
    }
    if turn.reasoning_content is not None:
        step["reasoning_content"] = turn.reasoning_content
    return step


def _observation_from_result(result: ProcessResult, *, max_chars: int) -> str:
    exception_parts = []
    if result.timed_out:
        exception_parts.append("command timed out")
    if result.stdout_truncated or result.stderr_truncated:
        exception_parts.append("command output was truncated by the execution backend")
    exception_info = "; ".join(exception_parts)
    returncode = -1 if result.timed_out else result.returncode
    output = result.stdout
    if result.stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += result.stderr

    prefix = f"<exception>{exception_info}</exception>\n" if exception_info else ""
    if len(output) < max_chars:
        return f"{prefix}<returncode>{returncode}</returncode>\n<output>\n{output}</output>"

    half = max_chars // 2
    elided_chars = len(output) - max_chars
    return (
        f"{prefix}<returncode>{returncode}</returncode>\n"
        "<warning>\n"
        "The output of your last command was too long.\n"
        "Please try a different command that produces less output.\n"
        "</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{elided_chars} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


def _extract_submission(result: ProcessResult) -> Optional[str]:
    if result.returncode != 0 or result.timed_out:
        return None
    lines = result.stdout.lstrip().splitlines(keepends=True)
    if not lines or lines[0].strip() != _SUBMISSION_MARKER:
        return None
    return "".join(lines[1:])


def _exit_message(exit_status: str, submission: str) -> dict[str, Any]:
    content = submission if exit_status == "Submitted" else exit_status
    return {
        "role": "exit",
        "content": content,
        "extra": {
            "exit_status": exit_status,
            "submission": submission,
        },
    }


def _format_error_feedback(error: str, *, finish_reason: Optional[str], has_tool_calls: bool) -> str:
    if finish_reason == "length" or (finish_reason == "tool_calls" and not has_tool_calls):
        return (
            f"Your previous response reached the output token limit (finish_reason={finish_reason}) "
            "before you produced a tool call, so it was cut off. Respond more concisely and finish "
            "with exactly one bash tool call."
        )
    return (
        "Tool call error:\n\n"
        f"<error>\n{error}\n</error>\n\n"
        "Every response needs to use the 'bash' tool at least once. Call the bash tool with "
        'arguments in the form {"command": "your_command_here"}.'
    )


def _validate_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cost must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and non-negative")
    return float(value)


def _require_nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or "\0" in value:
        raise ValueError(f"{name} must be non-empty and must not contain NUL bytes")
    return value


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise MinimalBashAgentError(f"{name} must contain JSON-compatible values") from error


__all__ = [
    "BashAgentEnvironment",
    "BashAgentFormatError",
    "BashAgentModel",
    "BashAgentTurn",
    "BashToolCall",
    "MinimalBashAgent",
    "MinimalBashAgentConfig",
    "MinimalBashAgentError",
    "MinimalBashAgentResult",
    "ProcessSandboxBashEnvironment",
    "bash_agent_turn_from_openai_message",
    "bash_tool_definition",
]
