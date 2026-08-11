"""Minimal, provider-agnostic Bash agent for repository tasks."""

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.sandbox import ProcessResult

_SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
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
    """Raised when the model or task violates the minimal Bash-agent contract."""


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
    """Normalized model output for one minimal-agent turn."""

    content: str = ""
    tool_calls: tuple[BashToolCall, ...] = ()
    reasoning_content: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple")
        if any(not isinstance(call, BashToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain only BashToolCall instances")
        if self.reasoning_content is not None and not isinstance(self.reasoning_content, str):
            raise TypeError("reasoning_content must be a string or None")


class BashAgentModel(Protocol):
    """Callable boundary implemented by an OpenAI, vLLM, or test adapter."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> BashAgentTurn: ...


class BashCommandExecutor(Protocol):
    """Narrow command boundary implemented by ``ProcessSandbox`` or an adapter."""

    def run(self, argv: Sequence[str]) -> ProcessResult: ...


@dataclass(frozen=True)
class MinimalBashAgentResult:
    """ATIF projection and raw command results from one rollout."""

    status: Literal["submitted", "step_limit"]
    submission: str
    trajectory: dict[str, Any]
    process_results: tuple[ProcessResult, ...]


def bash_tool_definition() -> dict[str, Any]:
    """Return an independent copy of the mini-swe-agent-compatible Bash tool."""

    return copy.deepcopy(_BASH_TOOL_DEFINITION)


def bash_agent_turn_from_openai_message(message: Mapping[str, Any]) -> BashAgentTurn:
    """Normalize one OpenAI-compatible assistant message for the Bash loop."""

    if not isinstance(message, Mapping):
        raise MinimalBashAgentError("assistant message must be an object")
    role = message.get("role")
    if role not in (None, "assistant"):
        raise MinimalBashAgentError("assistant message role must be 'assistant'")

    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise MinimalBashAgentError("assistant message content must be a string or null")
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise MinimalBashAgentError("assistant message reasoning_content must be a string or null")

    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise MinimalBashAgentError("assistant message tool_calls must be an array")
    calls = []
    for index, raw_call in enumerate(raw_calls):
        path = f"assistant message tool_calls[{index}]"
        if not isinstance(raw_call, Mapping):
            raise MinimalBashAgentError(f"{path} must be an object")
        if raw_call.get("type", "function") != "function":
            raise MinimalBashAgentError(f"{path}.type must be 'function'")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            raise MinimalBashAgentError(f"{path}.function must be an object")
        if function.get("name") != "bash":
            raise MinimalBashAgentError(f"{path}.function.name must be 'bash'")
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise MinimalBashAgentError(f"{path}.function.arguments must be a JSON string")
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError) as error:
            raise MinimalBashAgentError(f"{path}.function.arguments must contain valid JSON") from error
        if not isinstance(arguments, dict) or set(arguments) != {"command"}:
            raise MinimalBashAgentError(f"{path}.function.arguments must contain only 'command'")
        try:
            calls.append(BashToolCall(tool_call_id=call_id, command=arguments["command"]))
        except (TypeError, ValueError) as error:
            raise MinimalBashAgentError(f"{path} is invalid: {error}") from error

    return BashAgentTurn(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tuple(calls),
    )


class MinimalBashAgent:
    """Run a linear model/Bash loop through one process executor.

    The model adapter receives OpenAI-style messages plus exactly one Bash tool
    definition and must return exactly one Bash call per turn. A successful
    command whose first stdout line is ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
    ends the rollout. Repository state and command supervision remain owned by
    the outer pipeline and ``ProcessSandbox`` respectively. The scaffold does
    not own repository setup, reset, patch export, or cleanup.
    """

    def __init__(
        self,
        model: BashAgentModel,
        *,
        model_name: str,
        max_steps: int = 20,
        shell: str = "/bin/bash",
        agent_name: str = "lmflow-minimal-bash-agent",
        agent_version: str = "1",
    ) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        _require_nonempty_text(model_name, name="model_name")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        _require_nonempty_text(shell, name="shell")
        if not Path(shell).is_absolute():
            raise ValueError("shell must be an absolute path")
        _require_nonempty_text(agent_name, name="agent_name")
        _require_nonempty_text(agent_version, name="agent_version")

        self._model = model
        self.model_name = model_name
        self.max_steps = max_steps
        self.shell = shell
        self.agent_name = agent_name
        self.agent_version = agent_version

    def run(
        self,
        task: TaskSpec,
        *,
        sandbox: BashCommandExecutor,
        trajectory_id: str,
    ) -> MinimalBashAgentResult:
        """Run one rollout and return its ATIF v1.7 projection."""

        if not isinstance(task, TaskSpec):
            raise TypeError("task must be a TaskSpec")
        if not callable(getattr(sandbox, "run", None)):
            raise TypeError("sandbox must provide a callable run() method")
        _require_nonempty_text(trajectory_id, name="trajectory_id")

        messages, steps = _prepare_task(task)
        tools = [bash_tool_definition()]
        if task.tools and _json_copy(task.tools, name="task.tools") != tools:
            raise MinimalBashAgentError("MinimalBashAgent supports only the built-in Bash tool")

        process_results: list[ProcessResult] = []
        seen_call_ids: set[str] = set()
        submission = ""
        status: Literal["submitted", "step_limit"] = "step_limit"

        for _ in range(self.max_steps):
            turn = self._model(copy.deepcopy(messages), copy.deepcopy(tools))
            if not isinstance(turn, BashAgentTurn):
                raise MinimalBashAgentError("model must return a BashAgentTurn")
            if len(turn.tool_calls) != 1:
                raise MinimalBashAgentError("every model turn must contain exactly one Bash tool call")

            call = turn.tool_calls[0]
            if call.tool_call_id in seen_call_ids:
                raise MinimalBashAgentError(f"duplicate tool_call_id: {call.tool_call_id!r}")
            seen_call_ids.add(call.tool_call_id)

            messages.append(_assistant_message(turn, call))
            process_result = sandbox.run((self.shell, "-lc", call.command))
            if not isinstance(process_result, ProcessResult):
                raise MinimalBashAgentError("sandbox.run() must return a ProcessResult")
            process_results.append(process_result)
            observation = _format_observation(process_result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.tool_call_id,
                    "name": "bash",
                    "content": observation,
                }
            )
            steps.append(
                _agent_step(
                    step_id=len(steps) + 1,
                    turn=turn,
                    call=call,
                    observation=observation,
                    model_name=self.model_name,
                )
            )

            submitted_output = _extract_submission(process_result)
            if submitted_output is not None:
                submission = submitted_output
                status = "submitted"
                break

        trajectory = {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": trajectory_id,
            "agent": {
                "name": self.agent_name,
                "version": self.agent_version,
                "model_name": self.model_name,
                "tool_definitions": tools,
            },
            "steps": steps,
            "extra": {
                "lmflow": {
                    "task_id": task.task_id,
                    "exit_status": status,
                    "submission": submission,
                }
            },
        }
        return MinimalBashAgentResult(
            status=status,
            submission=submission,
            trajectory=trajectory,
            process_results=tuple(process_results),
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


def _assistant_message(turn: BashAgentTurn, call: BashToolCall) -> dict[str, Any]:
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
        ],
    }
    if turn.reasoning_content is not None:
        message["reasoning_content"] = turn.reasoning_content
    return message


def _agent_step(
    *,
    step_id: int,
    turn: BashAgentTurn,
    call: BashToolCall,
    observation: str,
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
        ],
        "observation": {
            "results": [
                {
                    "source_call_id": call.tool_call_id,
                    "content": observation,
                }
            ]
        },
    }
    if turn.reasoning_content is not None:
        step["reasoning_content"] = turn.reasoning_content
    return step


def _format_observation(result: ProcessResult) -> str:
    exceptions = []
    if result.timed_out:
        exceptions.append("command timed out")
    if result.stdout_truncated or result.stderr_truncated:
        exceptions.append("command output was truncated")
    prefix = "".join(f"<exception>{message}</exception>\n" for message in exceptions)

    output = result.stdout
    if result.stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += result.stderr
    return f"{prefix}<returncode>{result.returncode}</returncode>\n<output>\n{output}</output>"


def _extract_submission(result: ProcessResult) -> Optional[str]:
    if result.returncode != 0 or result.timed_out:
        return None
    lines = result.stdout.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != _SUBMISSION_MARKER:
        return None
    return "".join(lines[1:])


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
    "BashAgentModel",
    "BashAgentTurn",
    "BashCommandExecutor",
    "BashToolCall",
    "MinimalBashAgent",
    "MinimalBashAgentError",
    "MinimalBashAgentResult",
    "bash_agent_turn_from_openai_message",
    "bash_tool_definition",
]
