"""OpenAI-compatible Chat Completions backend for mini-swe-agent."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

_RESERVED_REQUEST_KEYS = frozenset({"messages", "model", "tools"})
_CLIENT_ONLY_REQUEST_KEYS = frozenset({"extra_headers"})


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


class OpenAICompatibleCompletionBackend:
    """Call a synchronous OpenAI-compatible Chat Completions endpoint.

    This backend implements the message/tool control plane used by the vendored
    mini-swe-agent scaffold. It intentionally does not claim the token-native
    rollout contract required by online RL.

    When ``client`` is omitted, the backend creates and owns a synchronous
    ``openai.OpenAI`` client. ``base_url`` is required in that mode so an
    accidental missing local vLLM URL cannot send requests to another endpoint.
    A caller-supplied client remains owned by the caller.
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

        response = self._client.chat.completions.create(
            model=model_name,
            messages=copy.deepcopy(messages),
            tools=copy.deepcopy(tools),
            **request_options,
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


__all__ = ["OpenAICompatibleCompletionBackend"]
