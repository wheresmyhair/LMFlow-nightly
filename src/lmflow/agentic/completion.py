"""Provider-neutral completion boundary for Agentic control-plane calls."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

_RESERVED_REQUEST_KEYS = frozenset({"messages", "model", "tools"})
_CLIENT_ONLY_REQUEST_KEYS = frozenset({"extra_headers"})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


class CompletionBackend(Protocol):
    """Synchronous chat-completion boundary shared by Agentic scaffolds."""

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        model_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _json_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        model_dump = getattr(value, "model_dump", None)
        if not callable(model_dump):
            raise TypeError(f"{name} must be a mapping or expose model_dump()")
        payload = model_dump(mode="json")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must serialize to a mapping")
    payload = copy.deepcopy(dict(payload))
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error
    return payload


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def parse_function_arguments(value: Any, *, path: str = "function.arguments") -> dict[str, Any]:
    """Parse one strict JSON object used as function-call arguments."""

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


def normalize_completion_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the shared single-choice completion backend response."""

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
    provider_reasoning = message.get("reasoning")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TypeError("completion message reasoning_content must be a string or null")
    if provider_reasoning is not None and not isinstance(provider_reasoning, str):
        raise TypeError("completion message reasoning must be a string or null")
    if reasoning_content is not None and provider_reasoning is not None and reasoning_content != provider_reasoning:
        raise ValueError("completion message reasoning and reasoning_content must match when both are present")
    if reasoning_content is None:
        reasoning_content = provider_reasoning
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


class OpenAICompatibleCompletionBackend:
    """Call a synchronous OpenAI-compatible Chat Completions endpoint.

    This backend implements the message/tool control plane used by Agentic
    scaffolds. It intentionally does not claim the token-native rollout
    contract required by online RL.

    When ``client`` is omitted, the backend creates and owns a synchronous
    ``openai.OpenAI`` client. ``base_url`` is required in that mode so an
    accidental missing local vLLM URL cannot send requests elsewhere. A
    caller-supplied client remains owned by the caller.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int | float):
            raise TypeError("timeout_seconds must be a number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self._owns_client = client is None
        self._closed = False
        if client is not None:
            if base_url is not None or api_key is not None:
                raise ValueError("base_url and api_key must be configured on a caller-supplied client")
            self._client = client
            return

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string when client is not supplied")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ValueError("api_key must be a non-empty string when provided")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError(
                "OpenAI-compatible completion requires the 'openai' package from the Agentic environment"
            ) from error

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=float(timeout_seconds),
            max_retries=max_retries,
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        model_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return one normalized assistant message and its raw provider response."""

        if self._closed:
            raise RuntimeError("completion backend is closed")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(model_kwargs, Mapping):
            raise TypeError("model_kwargs must be a mapping")
        if any(not isinstance(key, str) for key in model_kwargs):
            raise TypeError("model_kwargs keys must be strings")

        request_options = copy.deepcopy(dict(model_kwargs))
        reserved = sorted(_RESERVED_REQUEST_KEYS.intersection(request_options))
        if reserved:
            raise ValueError(f"model_kwargs cannot override request fields: {reserved}")
        client_only = sorted(_CLIENT_ONLY_REQUEST_KEYS.intersection(request_options))
        if client_only:
            raise ValueError(f"configure client-owned request fields on the OpenAI client: {client_only}")
        stream = request_options.get("stream", False)
        if stream is not False and stream is not None:
            raise ValueError("streaming completions are not supported")
        if "stream_options" in request_options:
            raise ValueError("stream_options require streaming and are not supported")
        n = request_options.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise ValueError("model_kwargs.n must be exactly 1")

        request = {
            "model": model_name,
            "messages": copy.deepcopy(messages),
            **request_options,
        }
        if tools:
            request["tools"] = copy.deepcopy(tools)
        response = self._client.chat.completions.create(
            **request,
        )
        choices = _field(response, "choices")
        if isinstance(choices, str | bytes) or not isinstance(choices, Sequence):
            raise TypeError("chat completion response choices must be a sequence")
        if len(choices) != 1:
            raise ValueError(f"chat completion must return exactly one choice, got {len(choices)}")
        choice = choices[0]
        message = _json_mapping(_field(choice, "message"), name="chat completion message")
        raw_response = _json_mapping(response, name="chat completion response")
        return {
            "message": message,
            "finish_reason": _field(choice, "finish_reason"),
            "cost": 0.0,
            "raw_response": raw_response,
        }

    def close(self) -> None:
        """Close an internally created HTTP client."""

        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleCompletionBackend:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "CompletionBackend",
    "OpenAICompatibleCompletionBackend",
    "normalize_completion_response",
    "parse_function_arguments",
]
