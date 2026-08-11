"""Synchronous multi-turn episode runner for the GSM8K reward tool."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any

from lmflow.agentic.completion import CompletionBackend
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.gsm8k import (
    GSM8K_AGENT_R1_REFERENCE_COMMIT,
    GSM8K_REWARD_TOOL_NAME,
    run_gsm8k_reward_tool,
    score_gsm8k_answer,
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def _parse_tool_arguments(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a JSON object string")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} must be strict JSON") from error
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must decode to an object")
    return parsed


def _ground_truth_from_task(task: TaskSpec) -> str:
    try:
        tools_kwargs = task.environment["tools_kwargs"]
        tool_kwargs = tools_kwargs[GSM8K_REWARD_TOOL_NAME]
        ground_truth = tool_kwargs["ground_truth"]
    except (KeyError, TypeError) as error:
        raise ValueError("task environment must contain GSM8K reward-tool ground truth") from error
    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError("GSM8K reward-tool ground truth must be a non-empty string")
    return ground_truth


def _validate_task(task: TaskSpec) -> str:
    if not isinstance(task, TaskSpec):
        raise TypeError("task must be a TaskSpec")
    if not task.messages:
        raise ValueError("task.messages must not be empty")
    for index, message in enumerate(task.messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"task.messages[{index}] must be a mapping")
        if message.get("role") not in {"system", "user"}:
            raise ValueError(f"task.messages[{index}].role must be 'system' or 'user'")
        if not isinstance(message.get("content"), str):
            raise TypeError(f"task.messages[{index}].content must be a string")

    tool_names = []
    for index, tool in enumerate(task.tools):
        try:
            tool_names.append(tool["function"]["name"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"task.tools[{index}] must contain a function name") from error
    if tool_names != [GSM8K_REWARD_TOOL_NAME]:
        raise ValueError(f"task.tools must contain only {GSM8K_REWARD_TOOL_NAME!r}")
    return _ground_truth_from_task(task)


def _normalize_completion(response: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise TypeError("completion backend must return a mapping")
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise TypeError("completion response must contain a message mapping")
    if message.get("role") != "assistant":
        raise ValueError("completion message role must be 'assistant'")

    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("completion message content must be a string or null")
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TypeError("completion message reasoning_content must be a string or null")
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise TypeError("completion message tool_calls must be a list")
    if not content and not reasoning_content and not tool_calls:
        raise ValueError("completion message must contain content, reasoning_content, or tool calls")

    cost = response.get("cost", 0.0)
    if isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("completion response cost must be a finite non-negative number")

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": copy.deepcopy(tool_calls),
        "finish_reason": _json_copy(response.get("finish_reason"), name="finish_reason"),
        "cost": float(cost),
        "raw_response": _json_copy(response.get("raw_response", response), name="raw_response"),
    }


def run_gsm8k_tool_episode(
    backend: CompletionBackend,
    task: TaskSpec,
    *,
    model_name: str,
    trajectory_id: str,
    model_kwargs: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    max_steps: int = 4,
) -> dict[str, Any]:
    """Run one control-plane episode and return an ATIF v1.7 trajectory.

    The returned trajectory is suitable for the existing ATIF-to-conversation
    SFT path. This HTTP-style control-plane runner does not provide sampled
    token IDs or log-probabilities and therefore is not an online-RL rollout
    implementation.
    """

    ground_truth = _validate_task(task)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        raise ValueError("trajectory_id must be a non-empty string")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise ValueError("session_id must be a non-empty string when provided")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if model_kwargs is None:
        model_kwargs = {}
    if not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping")

    history = copy.deepcopy(task.messages)
    steps = [
        {
            "step_id": index,
            "source": message["role"],
            "message": message["content"],
        }
        for index, message in enumerate(task.messages, start=1)
    ]
    seen_call_ids: set[str] = set()
    reward_tool_calls = 0
    completion_cost = 0.0

    for model_step in range(1, max_steps + 1):
        response = backend.complete(
            messages=copy.deepcopy(history),
            tools=copy.deepcopy(task.tools),
            model_name=model_name,
            model_kwargs=copy.deepcopy(dict(model_kwargs)),
        )
        completion = _normalize_completion(response)
        completion_cost += completion["cost"]
        raw_tool_calls = completion["tool_calls"]
        atif_tool_calls = []
        history_tool_calls = []
        observation_results = []

        for call_index, raw_call in enumerate(raw_tool_calls):
            path = f"completion tool_calls[{call_index}]"
            if not isinstance(raw_call, Mapping):
                raise TypeError(f"{path} must be a mapping")
            if raw_call.get("type") != "function":
                raise ValueError(f"{path}.type must be 'function'")
            call_id = raw_call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"{path}.id must be a non-empty string")
            if call_id in seen_call_ids:
                raise ValueError(f"{path}.id duplicates {call_id!r}")
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                raise TypeError(f"{path}.function must be a mapping")
            function_name = function.get("name")
            if function_name != GSM8K_REWARD_TOOL_NAME:
                raise ValueError(f"{path}.function.name must be {GSM8K_REWARD_TOOL_NAME!r}")
            arguments_text = function.get("arguments")
            arguments = _parse_tool_arguments(arguments_text, path=f"{path}.function.arguments")
            observation, details = run_gsm8k_reward_tool(arguments, ground_truth=ground_truth)

            seen_call_ids.add(call_id)
            reward_tool_calls += 1
            atif_tool_calls.append(
                {
                    "tool_call_id": call_id,
                    "function_name": function_name,
                    "arguments": arguments,
                }
            )
            history_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments_text,
                    },
                }
            )
            observation_results.append(
                {
                    "source_call_id": call_id,
                    "content": observation,
                    "extra": {
                        "answer": details["answer"],
                        "reward": details["reward"],
                    },
                }
            )

        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": completion["content"],
            "llm_call_count": 1,
            "extra": {
                "provider": {
                    "finish_reason": completion["finish_reason"],
                    "cost": completion["cost"],
                    "raw_response": completion["raw_response"],
                }
            },
        }
        if completion["reasoning_content"] is not None:
            step["reasoning_content"] = completion["reasoning_content"]
        if atif_tool_calls:
            step["tool_calls"] = atif_tool_calls
            step["observation"] = {"results": observation_results}
        steps.append(step)

        history_message: dict[str, Any] = {
            "role": "assistant",
            "content": completion["content"],
        }
        if completion["reasoning_content"] is not None:
            history_message["reasoning_content"] = completion["reasoning_content"]
        if history_tool_calls:
            history_message["tool_calls"] = history_tool_calls
        history.append(history_message)

        if history_tool_calls:
            for call, result in zip(history_tool_calls, observation_results, strict=True):
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "content": result["content"],
                    }
                )
            continue

        reward = score_gsm8k_answer(completion["content"], ground_truth, method="flexible")
        step["metrics"] = {"reward": reward}
        trajectory = {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": trajectory_id,
            "agent": {
                "name": "gsm8k-tool",
                "version": GSM8K_AGENT_R1_REFERENCE_COMMIT,
                "model_name": model_name,
                "tool_definitions": copy.deepcopy(task.tools),
            },
            "steps": steps,
            "final_metrics": {
                "reward": reward,
                "reward_tool_calls": reward_tool_calls,
                "model_steps": model_step,
                "completion_cost": completion_cost,
            },
            "extra": {
                "task_id": task.task_id,
                "task_metadata": _json_copy(task.metadata, name="task.metadata"),
            },
        }
        if session_id is not None:
            trajectory["session_id"] = session_id
        return trajectory

    raise RuntimeError(f"GSM8K-tool episode reached max_steps={max_steps} without a final answer")


__all__ = ["run_gsm8k_tool_episode"]
