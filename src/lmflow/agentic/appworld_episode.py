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
    canonical_appworld_instance_id,
    canonical_json_sha256,
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
APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION = "lmflow.appworld-conversation-projection/v1"
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
    world_factory: Callable[..., Any] | None = None,
) -> AppWorldEpisodeResult:
    """Run one local AppWorld task and evaluate it with AppWorld's verifier."""

    canonical_appworld_instance_id(task_id)
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
        history = copy.deepcopy(initial_messages)
        training_messages = []
        for message in initial_messages:
            projected = copy.deepcopy(message)
            if projected["role"] == "assistant":
                projected["loss"] = False
            training_messages.append(projected)
        last_execution_output: str | None = None

        for step_number in range(1, max_steps + 1):
            if last_execution_output is not None:
                observation = _observation_message(last_execution_output)
                history.append({"role": "user", "content": observation})
                training_messages.append({"role": "user", "content": observation})

            model_started_at = _monotonic_clock()
            try:
                response = backend.complete(
                    messages=copy.deepcopy(history),
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
            history.append(copy.deepcopy(assistant_message))
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
                runner_error = {"type": type(error).__name__, "message": str(error)}
                termination_reason = "environment_error"
                break
            execution_latency_seconds = _monotonic_clock() - execution_started_at
            api_calls_after = _request_log(world)
            api_calls = api_calls_after[len(api_calls_before) :]
            state_after_sha256 = _directory_digest(Path(world.output_db_home_path_on_disk))
            valid_action = _action_is_valid(code, execution_output)
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
                "instance_id": canonical_appworld_instance_id(task_id),
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
    projection_instance = {
        "conversation_id": trajectory_id,
        "messages": training_messages,
        "metadata": {
            "format_version": APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION,
            "source_artifact_sha256": artifact["manifest_sha256"],
            "task_id": task_id,
            "official_success": official_success,
            "eligible_for_success_only_sft": official_success,
        },
    }
    training_projection = {
        "type": "conversation",
        "instances": [projection_instance],
    }
    return AppWorldEpisodeResult(
        artifact=artifact,
        training_projection=training_projection,
        raw_output_directory=raw_output_directory,
        official_tracker=official_tracker,
    )


__all__ = [
    "APPWORLD_EPISODE_FORMAT_VERSION",
    "APPWORLD_TRAINING_PROJECTION_FORMAT_VERSION",
    "AppWorldEpisodeResult",
    "configure_appworld_freezegun",
    "run_appworld_episode",
]
