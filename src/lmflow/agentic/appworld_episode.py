"""Synchronous AppWorld episode runner using the pinned reference scaffold."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lmflow.agentic.appworld_protocol import (
    APPWORLD_CODE_VERSION,
    APPWORLD_DATA_VERSION,
    APPWORLD_REVISION,
    APPWORLD_SOURCE_SPLIT,
    canonical_appworld_sliced_instance_id,
    canonical_json_sha256,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.agentic.completion import CompletionBackend, normalize_completion_response
from lmflow.agentic.scaffolds.appworld_react_code.scaffold import (
    APPWORLD_REACT_CODE_SCAFFOLD,
    extract_first_python_code,
    load_reference_prompt,
    qwen3_reference_model_kwargs,
    render_reference_messages,
    scaffold_identity,
)

APPWORLD_EPISODE_FORMAT_VERSION = "lmflow.appworld-episode/v1"
APPWORLD_REPLAY_FORMAT_VERSION = "lmflow.appworld-replay/v1"
APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION = "lmflow.appworld-conversation-projection/v2"
APPWORLD_REACT_TRAINING_PROJECTION_FORMAT_VERSION = "lmflow.appworld-react-conversation-projection/v1"
APPWORLD_REACT_MESSAGE_PROJECTION_ID = "appworld.simplified-react-code/v1"
APPWORLD_PYTHON_TOOL_NAME = "appworld_python_repl"
_EXECUTION_FAILURE_PREFIX = "Execution failed. Traceback:"
_SENSITIVE_KEY_PARTS = ("access_token", "authorization", "password", "secret", "token")
_FREEZEGUN_IGNORE_PREFIXES = ("vllm",)


# AppWorld freezes task time and freezegun also rewrites imported aliases of
# ``time.perf_counter``/``time.monotonic``. ``clock_gettime`` is left untouched,
# so use the operating system's monotonic clock directly for latency evidence.
def _monotonic_clock() -> float:
    if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_MONOTONIC"):
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    return time.perf_counter()


@dataclass
class AppWorldEpisodeResult:
    """One benchmark-specific episode plus its official in-memory evaluator."""

    artifact: dict[str, Any]
    training_projection: dict[str, Any]
    raw_output_directory: Path
    official_tracker: Any | None


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_digest(path: Path) -> str:
    entries = []
    if path.is_dir():
        for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            entries.append(
                {
                    "path": file_path.relative_to(path).as_posix(),
                    "size": file_path.stat().st_size,
                    "sha256": _sha256_file(file_path),
                }
            )
    return canonical_json_sha256(entries)


def _redact(value: Any, *, key: str | None = None) -> Any:
    if key is not None and any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return _json_copy(value, name="API call value")


def _request_log(world: Any) -> list[dict[str, Any]]:
    try:
        requests = world.requester.request_tracker.requests
    except AttributeError:
        return []
    if not isinstance(requests, list):
        raise TypeError("AppWorld request tracker must expose a list")
    return copy.deepcopy(requests)


def _usage_from_completion(completion: Mapping[str, Any]) -> dict[str, int | None]:
    raw_response = completion.get("raw_response")
    raw_usage = raw_response.get("usage") if isinstance(raw_response, Mapping) else None
    if not isinstance(raw_usage, Mapping):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
    output_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
    total_tokens = raw_usage.get("total_tokens")
    return {
        "input_tokens": input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else None,
        "output_tokens": (
            output_tokens if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) else None
        ),
        "total_tokens": total_tokens if isinstance(total_tokens, int) and not isinstance(total_tokens, bool) else None,
    }


def _sum_usage(steps: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"model_calls": len(steps), "reported_for_all_calls": True}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [step["usage"][field] for step in steps]
        if values and all(isinstance(value, int) for value in values):
            result[field] = sum(values)
        else:
            result[field] = None
            result["reported_for_all_calls"] = False
    return result


def _observation_message(output: str) -> str:
    maybe_new_line = "\n" if not output.endswith("\n") else ""
    return "Output:\n```\n" + output + maybe_new_line + "```\n\n"


def _parsed_python_tool_call(call_id: str, code: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": APPWORLD_PYTHON_TOOL_NAME,
            "arguments": json.dumps(
                {"code": code},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
        "extra": {
            "origin": "parsed_assistant_content",
            "native_provider_tool_call": False,
        },
    }


def _canonical_reference_messages(
    messages: list[dict[str, Any]],
    *,
    trajectory_id: str,
) -> list[dict[str, Any]]:
    """Give prompt demonstrations semantic tool roles without changing their text."""

    canonical_messages: list[dict[str, Any]] = []
    pending_call_id: str | None = None
    action_index = 0
    for raw_message in messages:
        message = copy.deepcopy(raw_message)
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("AppWorld reference messages must contain text content")
        if role == "assistant":
            message["loss"] = False
            code, _ = extract_first_python_code(content)
            if code.strip():
                action_index += 1
                pending_call_id = f"{trajectory_id}:prompt-action-{action_index:03d}"
                message["tool_calls"] = [_parsed_python_tool_call(pending_call_id, code)]
            else:
                pending_call_id = None
        elif role == "user" and pending_call_id is not None and content.startswith("Output:\n```\n"):
            message = {
                "role": "tool",
                "name": APPWORLD_PYTHON_TOOL_NAME,
                "tool_call_id": pending_call_id,
                "content": content,
            }
            pending_call_id = None
        else:
            pending_call_id = None
        canonical_messages.append(message)
    return canonical_messages


def project_appworld_messages_for_react_scaffold(
    messages: list[dict[str, Any]],
    *,
    preserve_loss: bool = False,
) -> list[dict[str, Any]]:
    """Project semantic AppWorld messages to the official model-visible roles.

    AppWorld's pinned ReAct-code scaffold sends Python execution observations as
    ``user`` messages containing ``Output:`` blocks. LMFlow keeps them as semantic
    ``tool`` messages in Dataset views, then applies this projection both for
    online model requests and before SFT tokenization.
    """

    if not isinstance(messages, list):
        raise TypeError("AppWorld messages must be a list")
    known_calls: dict[str, str] = {}
    observed_calls: set[str] = set()
    projected_messages: list[dict[str, Any]] = []
    for index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            raise TypeError(f"AppWorld message {index} must be a mapping")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if not isinstance(content, str):
            raise TypeError(f"AppWorld message {index} must contain text content")
        if role in {"system", "user"}:
            projected_messages.append({"role": role, "content": content})
            continue
        if role == "assistant":
            projected = {"role": "assistant", "content": content}
            if preserve_loss and "loss" in raw_message:
                loss = raw_message.get("loss")
                if loss is not None and not isinstance(loss, bool):
                    raise TypeError(f"AppWorld assistant message {index} loss must be a boolean or null")
                projected["loss"] = loss
            reasoning_content = raw_message.get("reasoning_content")
            if reasoning_content is not None:
                if not isinstance(reasoning_content, str):
                    raise TypeError(f"AppWorld assistant message {index} reasoning_content must be text")
                projected["reasoning_content"] = reasoning_content
            raw_calls = raw_message.get("tool_calls", [])
            if not isinstance(raw_calls, list):
                raise TypeError(f"AppWorld assistant message {index} tool_calls must be a list")
            for call_index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, Mapping):
                    raise TypeError(f"AppWorld assistant message {index} tool call {call_index} must be a mapping")
                call_id = raw_call.get("id")
                function = raw_call.get("function")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(f"AppWorld assistant message {index} tool call {call_index} has no id")
                if call_id in known_calls:
                    raise ValueError(f"AppWorld tool call id is duplicated: {call_id!r}")
                if not isinstance(function, Mapping) or function.get("name") != APPWORLD_PYTHON_TOOL_NAME:
                    raise ValueError("AppWorld semantic conversations may only contain parsed Python actions")
                known_calls[call_id] = APPWORLD_PYTHON_TOOL_NAME
            projected_messages.append(projected)
            continue
        if role == "tool":
            call_id = raw_message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in known_calls:
                raise ValueError(f"AppWorld tool message {index} references an unknown action")
            if call_id in observed_calls:
                raise ValueError(f"AppWorld tool call has duplicate observations: {call_id!r}")
            if raw_message.get("name") != known_calls[call_id]:
                raise ValueError(f"AppWorld tool message {index} has the wrong tool name")
            observed_calls.add(call_id)
            projected_messages.append({"role": "user", "content": content})
            continue
        raise ValueError(f"Unsupported AppWorld semantic role at message {index}: {role!r}")
    return projected_messages


def project_appworld_conversation_for_react_scaffold(conversation: Mapping[str, Any]) -> dict[str, Any]:
    """Create a training-ready, scaffold-exact conversation Dataset view."""

    if not isinstance(conversation, Mapping):
        raise TypeError("AppWorld conversation must be a mapping")
    if conversation.get("type") != "conversation":
        raise ValueError("AppWorld conversation must use the LMFlow conversation Dataset type")
    instances = conversation.get("instances")
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], Mapping):
        raise ValueError("AppWorld conversation must contain exactly one instance")
    result = copy.deepcopy(dict(conversation))
    instance = result["instances"][0]
    metadata = instance.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("AppWorld conversation metadata is missing")
    if metadata.get("format_version") != APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION:
        raise ValueError("AppWorld conversation has an unsupported semantic format")
    messages = instance.get("messages")
    instance["messages"] = project_appworld_messages_for_react_scaffold(messages, preserve_loss=True)
    projected_metadata = copy.deepcopy(dict(metadata))
    projected_metadata.update(
        {
            "format_version": APPWORLD_REACT_TRAINING_PROJECTION_FORMAT_VERSION,
            "source_semantic_format_version": APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION,
            "projection_kind": "scaffold_exact",
            "scaffold_message_projection": APPWORLD_REACT_MESSAGE_PROJECTION_ID,
            "requires_scaffold_projection": False,
        }
    )
    instance["metadata"] = projected_metadata
    return result


def appworld_artifact_to_semantic_conversation(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the semantic conversation view from an immutable episode artifact."""

    if not isinstance(artifact, Mapping):
        raise TypeError("AppWorld artifact must be a mapping")
    verify_manifest_digest(artifact)
    if artifact.get("format_version") != APPWORLD_EPISODE_FORMAT_VERSION:
        raise ValueError("artifact is not an AppWorld episode")
    trajectory_id = artifact.get("trajectory_id")
    initial_messages = artifact.get("initial_messages")
    model_steps = artifact.get("model_steps")
    action_steps = artifact.get("action_steps")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("AppWorld artifact has no trajectory id")
    if not isinstance(initial_messages, list):
        raise ValueError("AppWorld artifact initial messages are missing")
    if not isinstance(model_steps, list) or not isinstance(action_steps, list):
        raise ValueError("AppWorld artifact model/action steps are missing")
    task = artifact.get("task")
    metrics = artifact.get("metrics")
    if not isinstance(task, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("AppWorld artifact task or metrics are missing")

    messages = _canonical_reference_messages(initial_messages, trajectory_id=trajectory_id)
    actions_by_step = {step.get("step_number"): step for step in action_steps if isinstance(step, Mapping)}
    if len(actions_by_step) != len(action_steps):
        raise ValueError("AppWorld artifact action step numbers must be unique")
    model_step_numbers = []
    for index, model_step in enumerate(model_steps):
        if not isinstance(model_step, Mapping):
            raise TypeError(f"AppWorld model step {index} must be a mapping")
        step_number = model_step.get("step_number")
        content = model_step.get("content")
        if not isinstance(step_number, int) or isinstance(step_number, bool):
            raise ValueError(f"AppWorld model step {index} has no integer step number")
        if not isinstance(content, str):
            raise TypeError(f"AppWorld model step {index} content must be text")
        model_step_numbers.append(step_number)
        action = actions_by_step.get(step_number)
        if action is None:
            code, _ = extract_first_python_code(content)
            loss = False
        else:
            code = action.get("code")
            if not isinstance(code, str):
                raise TypeError(f"AppWorld action step {step_number} code must be text")
            loss = action.get("valid") is True
        call_id = f"{trajectory_id}:action-{step_number:03d}"
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content + "\n\n",
            "loss": loss,
            "tool_calls": [_parsed_python_tool_call(call_id, code)],
        }
        reasoning_content = model_step.get("reasoning_content")
        if reasoning_content is not None:
            if not isinstance(reasoning_content, str):
                raise TypeError(f"AppWorld model step {index} reasoning_content must be text")
            assistant_message["reasoning_content"] = reasoning_content
        messages.append(assistant_message)

        has_later_model_step = index + 1 < len(model_steps)
        observation_was_visible = has_later_model_step or (
            index == len(model_steps) - 1 and metrics.get("termination_reason") == "model_backend_error"
        )
        if observation_was_visible:
            if action is None:
                raise ValueError(f"AppWorld model step {step_number} has a later turn but no executed action")
            output = action.get("output")
            if not isinstance(output, str):
                raise TypeError(f"AppWorld action step {step_number} output must be text")
            messages.append(
                {
                    "role": "tool",
                    "name": APPWORLD_PYTHON_TOOL_NAME,
                    "tool_call_id": call_id,
                    "content": _observation_message(output),
                }
            )
    if len(set(model_step_numbers)) != len(model_step_numbers):
        raise ValueError("AppWorld artifact model step numbers must be unique")

    return {
        "type": "conversation",
        "instances": [
            {
                "conversation_id": trajectory_id,
                "messages": messages,
                "metadata": {
                    "format_version": APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION,
                    "source_artifact_sha256": artifact["manifest_sha256"],
                    "task_id": task.get("task_id"),
                    "official_success": metrics.get("success") is True,
                    "semantic_roles": True,
                    "observation_role": "tool",
                    "parsed_action_tool": APPWORLD_PYTHON_TOOL_NAME,
                    "parsed_action_native_provider_tool_call": False,
                    "requires_scaffold_projection": True,
                    "scaffold_message_projection": APPWORLD_REACT_MESSAGE_PROJECTION_ID,
                    "replay_required": True,
                    "eligible_for_success_only_sft": False,
                    "eligible_for_success_plus_recovery_sft": False,
                },
            }
        ],
    }


def _action_is_valid(code: str, output: str) -> bool:
    return (
        bool(code.strip())
        and not output.startswith(_EXECUTION_FAILURE_PREFIX)
        and output != "No code available to execute."
    )


def _failure_type(*, official_success: bool, termination_reason: str, invalid_actions: int) -> str | None:
    if official_success:
        return None
    if termination_reason == "model_backend_error":
        return "model_backend_error"
    if termination_reason == "environment_error":
        return "environment_error"
    if termination_reason == "official_evaluator_error":
        return "official_evaluator_error"
    if termination_reason == "task_completed":
        return "official_verifier_failure"
    if termination_reason == "max_steps":
        return "max_steps_with_invalid_action" if invalid_actions else "max_steps_incomplete"
    return "incomplete"


def _official_evaluation_summary(evaluation: Any) -> dict[str, int | bool | None]:
    if not isinstance(evaluation, Mapping):
        return {"success": False, "num_tests": None, "pass_count": None, "failure_count": None}
    num_tests = evaluation.get("num_tests")
    passes = evaluation.get("passes")
    failures = evaluation.get("failures")
    return {
        "success": evaluation.get("success") is True,
        "num_tests": num_tests if isinstance(num_tests, int) and not isinstance(num_tests, bool) else None,
        "pass_count": len(passes) if isinstance(passes, list) else None,
        "failure_count": len(failures) if isinstance(failures, list) else None,
    }


def _all_official_tests_passed(summary: Mapping[str, Any]) -> bool:
    num_tests = summary.get("num_tests")
    return (
        summary.get("success") is True
        and isinstance(num_tests, int)
        and not isinstance(num_tests, bool)
        and num_tests > 0
        and summary.get("pass_count") == num_tests
        and summary.get("failure_count") == 0
    )


def configure_appworld_freezegun() -> None:
    """Keep AppWorld's task clock from traversing vLLM's lazy modules."""

    try:
        from freezegun import configure
    except ImportError as error:
        raise ImportError("freezegun is unavailable; synchronize the pinned AppWorld runtime dependencies") from error
    configure(extend_ignore_list=list(_FREEZEGUN_IGNORE_PREFIXES))


def _default_world_factory(**kwargs: Any) -> Any:
    configure_appworld_freezegun()
    try:
        from appworld import AppWorld
    except ImportError as error:
        raise ImportError(
            "AppWorld is unavailable; synchronize the agentic lock and run scripts/agentic/bootstrap_appworld.sh"
        ) from error
    return AppWorld(**kwargs)


def run_appworld_episode(
    backend: CompletionBackend,
    *,
    task_id: str,
    model_name: str,
    model_revision: str,
    trajectory_id: str,
    appworld_root: str | os.PathLike[str],
    appworld_source: str | os.PathLike[str],
    experiment_name: str,
    model_kwargs: Mapping[str, Any] | None = None,
    max_steps: int = 50,
    source_split: str = APPWORLD_SOURCE_SPLIT,
    world_factory: Callable[..., Any] | None = None,
) -> AppWorldEpisodeResult:
    """Run one local AppWorld task and evaluate it with AppWorld's verifier."""

    instance_id = canonical_appworld_sliced_instance_id(task_id, source_split=source_split)
    for name, value in (
        ("model_name", model_name),
        ("model_revision", model_revision),
        ("trajectory_id", trajectory_id),
        ("experiment_name", experiment_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be an integer")
    if not 1 <= max_steps <= APPWORLD_REACT_CODE_SCAFFOLD["configuration"]["max_steps"]:
        raise ValueError("max_steps must be between 1 and the reference scaffold's 50-step limit")
    if model_kwargs is None:
        model_kwargs = qwen3_reference_model_kwargs()
    if not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping")

    root = Path(appworld_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AppWorld root does not exist: {root}")
    os.environ["APPWORLD_ROOT"] = str(root)
    prompt = load_reference_prompt(appworld_source)
    scaffold = scaffold_identity(appworld_source)
    factory = world_factory or _default_world_factory

    started_at = _monotonic_clock()
    init_started_at = _monotonic_clock()
    official_tracker = None
    evaluator_error: dict[str, str] | None = None
    runner_error: dict[str, str] | None = None
    raw_output_directory = Path()
    model_steps: list[dict[str, Any]] = []
    action_steps: list[dict[str, Any]] = []
    termination_reason = "max_steps"
    recovery_count = 0
    pending_failed_action = False
    task_completed_signal = False

    with factory(
        task_id=task_id,
        experiment_name=experiment_name,
        load_ground_truth=True,
        random_seed=APPWORLD_REACT_CODE_SCAFFOLD["configuration"]["random_seed"],
        raise_on_extra_parameters=True,
    ) as world:
        initialization_seconds = _monotonic_clock() - init_started_at
        raw_output_directory = Path(world.output_directory)
        initial_state_sha256 = _directory_digest(Path(world.output_db_home_path_on_disk))
        initial_messages = render_reference_messages(prompt, world.task)
        training_messages = _canonical_reference_messages(initial_messages, trajectory_id=trajectory_id)
        last_execution_output: str | None = None
        last_action_call_id: str | None = None

        for step_number in range(1, max_steps + 1):
            if last_execution_output is not None:
                if last_action_call_id is None:
                    raise RuntimeError("AppWorld observation is missing its source action")
                observation = _observation_message(last_execution_output)
                training_messages.append(
                    {
                        "role": "tool",
                        "name": APPWORLD_PYTHON_TOOL_NAME,
                        "tool_call_id": last_action_call_id,
                        "content": observation,
                    }
                )

            model_started_at = _monotonic_clock()
            try:
                response = backend.complete(
                    messages=project_appworld_messages_for_react_scaffold(training_messages),
                    tools=[],
                    model_name=model_name,
                    model_kwargs=copy.deepcopy(dict(model_kwargs)),
                )
                completion = normalize_completion_response(response)
            except Exception as error:
                runner_error = {"type": type(error).__name__, "message": str(error)}
                termination_reason = "model_backend_error"
                break
            model_latency_seconds = _monotonic_clock() - model_started_at
            code, fixed_content = extract_first_python_code(completion["content"])
            assistant_message: dict[str, Any] = {"role": "assistant", "content": fixed_content + "\n\n"}
            if completion["reasoning_content"] is not None:
                assistant_message["reasoning_content"] = completion["reasoning_content"]
            last_action_call_id = f"{trajectory_id}:action-{step_number:03d}"
            assistant_message["tool_calls"] = [_parsed_python_tool_call(last_action_call_id, code)]
            training_message = copy.deepcopy(assistant_message)
            training_message["loss"] = True
            training_messages.append(training_message)

            usage = _usage_from_completion(completion)
            model_steps.append(
                {
                    "step_number": step_number,
                    "content": fixed_content,
                    "reasoning_content": completion["reasoning_content"],
                    "finish_reason": completion["finish_reason"],
                    "usage": usage,
                    "latency_seconds": model_latency_seconds,
                    "raw_response": completion["raw_response"],
                }
            )

            api_calls_before = _request_log(world)
            state_before_sha256 = _directory_digest(Path(world.output_db_home_path_on_disk))
            execution_started_at = _monotonic_clock()
            try:
                execution_output = world.execute(code)
            except Exception as error:
                training_message["loss"] = False
                runner_error = {"type": type(error).__name__, "message": str(error)}
                termination_reason = "environment_error"
                break
            execution_latency_seconds = _monotonic_clock() - execution_started_at
            api_calls_after = _request_log(world)
            api_calls = api_calls_after[len(api_calls_before) :]
            state_after_sha256 = _directory_digest(Path(world.output_db_home_path_on_disk))
            valid_action = _action_is_valid(code, execution_output)
            training_message["loss"] = valid_action
            recovered = pending_failed_action and valid_action
            if recovered:
                recovery_count += 1
            pending_failed_action = not valid_action
            action_steps.append(
                {
                    "step_number": step_number,
                    "code": code,
                    "output": execution_output,
                    "valid": valid_action,
                    "api_call_count": len(api_calls),
                    "api_calls_redacted": _redact(api_calls),
                    "state_before_sha256": state_before_sha256,
                    "state_after_sha256": state_after_sha256,
                    "state_changed": state_before_sha256 != state_after_sha256,
                    "recovered_prior_failure": recovered,
                    "latency_seconds": execution_latency_seconds,
                }
            )
            last_execution_output = execution_output
            task_completed_signal = bool(world.task_completed())
            if task_completed_signal:
                termination_reason = "task_completed"
                break

        evaluation_started_at = _monotonic_clock()
        try:
            official_tracker = world.evaluate(suppress_errors=True)
            official_evaluation = official_tracker.to_dict(stats_only=False)
        except Exception as error:
            official_evaluation = None
            evaluator_error = {"type": type(error).__name__, "message": str(error)}
            termination_reason = "official_evaluator_error"
        evaluation_seconds = _monotonic_clock() - evaluation_started_at
        final_state_sha256 = _directory_digest(Path(world.output_db_home_path_on_disk))

    total_seconds = _monotonic_clock() - started_at
    valid_actions = sum(bool(step["valid"]) for step in action_steps)
    invalid_actions = len(action_steps) - valid_actions
    official_success = bool(official_evaluation and official_evaluation.get("success"))
    if official_success:
        task_status = "success"
    elif evaluator_error:
        task_status = "evaluator_error"
    elif runner_error:
        task_status = "runner_error"
    elif task_completed_signal:
        task_status = "completed_verifier_failed"
    else:
        task_status = "incomplete"
    failure_type = _failure_type(
        official_success=official_success,
        termination_reason=termination_reason,
        invalid_actions=invalid_actions,
    )
    usage = _sum_usage(model_steps)
    metrics = {
        "success": official_success,
        "task_status": task_status,
        "failure_type": failure_type,
        "steps": len(action_steps),
        "model_calls": len(model_steps),
        "tool_calls": len(action_steps),
        "valid_tool_calls": valid_actions,
        "invalid_tool_calls": invalid_actions,
        "api_call_attempts": sum(step["api_call_count"] for step in action_steps),
        "state_change_steps": sum(bool(step["state_changed"]) for step in action_steps),
        "recovery_count": recovery_count,
        "task_completed_signal": task_completed_signal,
        "termination_reason": termination_reason,
        "usage": usage,
        "latency_seconds": {
            "initialization": initialization_seconds,
            "model": sum(step["latency_seconds"] for step in model_steps),
            "environment": sum(step["latency_seconds"] for step in action_steps),
            "evaluation": evaluation_seconds,
            "total": total_seconds,
        },
    }
    artifact = with_manifest_digest(
        {
            "format_version": APPWORLD_EPISODE_FORMAT_VERSION,
            "trajectory_id": trajectory_id,
            "task": {
                "task_id": task_id,
                "instance_id": instance_id,
                "source_split": source_split,
                "ground_truth_visibility": "official_evaluator_only",
            },
            "agent": scaffold,
            "model": {"name": model_name, "revision": model_revision},
            "sampling": _json_copy(model_kwargs, name="model_kwargs"),
            "execution_environment": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "appworld_code_version": APPWORLD_CODE_VERSION,
                "appworld_revision": APPWORLD_REVISION,
                "appworld_data_version": APPWORLD_DATA_VERSION,
                "mode": "local-in-process",
                "random_seed": APPWORLD_REACT_CODE_SCAFFOLD["configuration"]["random_seed"],
                "safety_guards": "official-defaults-enabled",
                "max_steps": max_steps,
            },
            "initial_messages": initial_messages,
            "initial_state_sha256": initial_state_sha256,
            "model_steps": model_steps,
            "action_steps": action_steps,
            "final_state_sha256": final_state_sha256,
            "official_evaluation": official_evaluation,
            "runner_error": runner_error,
            "evaluator_error": evaluator_error,
            "metrics": metrics,
        }
    )
    training_projection = appworld_artifact_to_semantic_conversation(artifact)
    if training_projection["instances"][0]["messages"] != training_messages:
        raise RuntimeError("AppWorld online messages and artifact-derived semantic projection diverged")
    return AppWorldEpisodeResult(
        artifact=artifact,
        training_projection=training_projection,
        raw_output_directory=raw_output_directory,
        official_tracker=official_tracker,
    )


def replay_appworld_episode(
    artifact: Mapping[str, Any],
    *,
    appworld_root: str | os.PathLike[str],
    experiment_name: str,
    world_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Replay recorded actions from a fresh reset and return sealed gate evidence."""

    if not isinstance(artifact, Mapping):
        raise TypeError("artifact must be a mapping")
    verify_manifest_digest(artifact)
    if artifact.get("format_version") != APPWORLD_EPISODE_FORMAT_VERSION:
        raise ValueError("artifact is not an AppWorld episode")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must be a non-empty string")
    task = artifact.get("task")
    if not isinstance(task, Mapping) or not isinstance(task.get("task_id"), str):
        raise ValueError("artifact task identity is missing")
    task_id = task["task_id"]
    source_split = task.get("source_split", APPWORLD_SOURCE_SPLIT)
    if not isinstance(source_split, str):
        raise ValueError("artifact source split is invalid")
    canonical_appworld_sliced_instance_id(task_id, source_split=source_split)
    action_steps = artifact.get("action_steps")
    if not isinstance(action_steps, list):
        raise ValueError("artifact action_steps must be a list")
    for action_step in action_steps:
        if not isinstance(action_step, Mapping) or not isinstance(action_step.get("code"), str):
            raise ValueError("every recorded action step must contain code")

    root = Path(appworld_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AppWorld root does not exist: {root}")
    os.environ["APPWORLD_ROOT"] = str(root)
    factory = world_factory or _default_world_factory
    replay_error: dict[str, Any] | None = None
    initial_state_sha256 = None
    final_state_sha256 = None
    official_evaluation = None
    replay_steps: list[dict[str, Any]] = []

    try:
        with factory(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            random_seed=APPWORLD_REACT_CODE_SCAFFOLD["configuration"]["random_seed"],
            raise_on_extra_parameters=True,
        ) as world:
            state_directory = Path(world.output_db_home_path_on_disk)
            initial_state_sha256 = _directory_digest(state_directory)
            for action_step in action_steps:
                state_before_sha256 = _directory_digest(state_directory)
                requests_before = _request_log(world)
                try:
                    output = world.execute(action_step["code"])
                except Exception as error:
                    replay_error = {
                        "stage": "execute",
                        "step_number": action_step.get("step_number"),
                        "type": type(error).__name__,
                    }
                    break
                requests_after = _request_log(world)
                state_after_sha256 = _directory_digest(state_directory)
                replay_steps.append(
                    {
                        "step_number": action_step.get("step_number"),
                        "output_sha256_match": hashlib.sha256(output.encode()).hexdigest()
                        == hashlib.sha256(str(action_step.get("output", "")).encode()).hexdigest(),
                        "validity_match": _action_is_valid(action_step["code"], output)
                        is bool(action_step.get("valid")),
                        "api_call_count_match": len(requests_after) - len(requests_before)
                        == action_step.get("api_call_count"),
                        "state_before_match": state_before_sha256 == action_step.get("state_before_sha256"),
                        "state_after_match": state_after_sha256 == action_step.get("state_after_sha256"),
                    }
                )
            final_state_sha256 = _directory_digest(state_directory)
            if replay_error is None:
                try:
                    official_evaluation = world.evaluate(suppress_errors=True).to_dict(stats_only=False)
                except Exception as error:
                    replay_error = {"stage": "official_evaluator", "type": type(error).__name__}
    except Exception as error:
        replay_error = {"stage": "environment_initialization", "type": type(error).__name__}

    original_summary = _official_evaluation_summary(artifact.get("official_evaluation"))
    replay_summary = _official_evaluation_summary(official_evaluation)
    action_count_match = len(replay_steps) == len(action_steps)
    step_evidence_match = action_count_match and all(
        all(value is True for key, value in replay_step.items() if key != "step_number") for replay_step in replay_steps
    )
    initial_state_match = initial_state_sha256 == artifact.get("initial_state_sha256")
    final_state_match = final_state_sha256 == artifact.get("final_state_sha256")
    official_summary_match = original_summary == replay_summary
    replay_match = (
        replay_error is None
        and initial_state_match
        and step_evidence_match
        and final_state_match
        and official_summary_match
    )
    collateral_invariant_passed = (
        replay_match and _all_official_tests_passed(original_summary) and _all_official_tests_passed(replay_summary)
    )
    return with_manifest_digest(
        {
            "format_version": APPWORLD_REPLAY_FORMAT_VERSION,
            "source_trajectory_id": artifact.get("trajectory_id"),
            "source_artifact_sha256": artifact["manifest_sha256"],
            "task_id": task_id,
            "source_split": source_split,
            "appworld_revision": APPWORLD_REVISION,
            "reset_seed": APPWORLD_REACT_CODE_SCAFFOLD["configuration"]["random_seed"],
            "action_count": len(action_steps),
            "replayed_action_count": len(replay_steps),
            "initial_state_match": initial_state_match,
            "step_evidence": replay_steps,
            "final_state_match": final_state_match,
            "official_summary_match": official_summary_match,
            "original_official_success": original_summary["success"],
            "replay_official_success": replay_summary["success"],
            "original_all_official_tests_passed": _all_official_tests_passed(original_summary),
            "replay_all_official_tests_passed": _all_official_tests_passed(replay_summary),
            "sealed_partial_signal": (
                original_summary["success"] is False
                and isinstance(original_summary["pass_count"], int)
                and original_summary["pass_count"] > 0
            ),
            "replay_error": replay_error,
            "replay_match": replay_match,
            "collateral_invariant_passed": collateral_invariant_passed,
            "hidden_verifier_material_included": False,
        }
    )


__all__ = [
    "APPWORLD_EPISODE_FORMAT_VERSION",
    "APPWORLD_REPLAY_FORMAT_VERSION",
    "APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION",
    "AppWorldEpisodeResult",
    "configure_appworld_freezegun",
    "replay_appworld_episode",
    "run_appworld_episode",
]
