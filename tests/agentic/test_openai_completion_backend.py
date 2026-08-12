import copy
import json
from types import SimpleNamespace

import pytest

from lmflow.agentic.scaffolds.mini_swe_agent import (
    LMFlowMiniSWEAgentModel,
    OpenAICompatibleCompletionBackend,
)


class SerializableResponse:
    def __init__(self, choices, payload):
        self.choices = choices
        self.payload = payload

    def model_dump(self, *, mode):
        assert mode == "json"
        return copy.deepcopy(self.payload)


class RecordingCompletions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.response


class RecordingClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=RecordingCompletions(response))
        self.closed = False

    def close(self):
        self.closed = True


def _response(*, choices=1):
    message = {
        "role": "assistant",
        "content": "Inspecting the repository.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "git status --short"})},
            }
        ],
    }
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    raw = {
        "id": "chatcmpl-fixture",
        "model": "served-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    return SerializableResponse([choice] * choices, raw)


def test_backend_forwards_one_non_streaming_request_and_preserves_raw_response():
    client = RecordingClient(_response())
    backend = OpenAICompatibleCompletionBackend(client=client)
    messages = [{"role": "user", "content": "Fix the failing test."}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    model_kwargs = {
        "temperature": 0.2,
        "seed": 17,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    result = backend.complete(
        messages=messages,
        tools=tools,
        model_name="served-model",
        model_kwargs=model_kwargs,
    )
    messages[0]["content"] = "mutated"
    tools[0]["function"]["name"] = "mutated"
    model_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True

    assert client.chat.completions.requests == [
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "Fix the failing test."}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "temperature": 0.2,
            "seed": 17,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ]
    assert result["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert result["finish_reason"] == "tool_calls"
    assert result["cost"] == 0.0
    assert result["raw_response"]["usage"]["total_tokens"] == 20


def test_backend_connects_to_the_mini_swe_agent_model_adapter():
    client = RecordingClient(_response())
    backend = OpenAICompatibleCompletionBackend(client=client)
    model = LMFlowMiniSWEAgentModel(backend, model_name="served-model", model_kwargs={"temperature": 0.0})

    message = model.query([{"role": "user", "content": "Inspect the checkout."}])

    assert message["extra"]["actions"] == [{"command": "git status --short", "tool_call_id": "call-1"}]
    assert message["extra"]["response"]["id"] == "chatcmpl-fixture"


@pytest.mark.parametrize(
    ("model_kwargs", "error"),
    [
        ({"stream": True}, "streaming completions are not supported"),
        ({"stream": 0}, "streaming completions are not supported"),
        ({"stream_options": {"include_usage": True}}, "stream_options require streaming"),
        ({"n": 2}, "model_kwargs.n must be exactly 1"),
        ({"model": "other-model"}, "cannot override request fields"),
        ({"extra_headers": {"Authorization": "secret"}}, "configure client-owned request fields"),
    ],
)
def test_backend_rejects_request_shapes_that_break_one_episode_completion(model_kwargs, error):
    client = RecordingClient(_response())
    backend = OpenAICompatibleCompletionBackend(client=client)

    with pytest.raises(ValueError, match=error):
        backend.complete(messages=[], tools=[], model_name="served-model", model_kwargs=model_kwargs)

    assert client.chat.completions.requests == []


def test_backend_rejects_multiple_provider_choices():
    backend = OpenAICompatibleCompletionBackend(client=RecordingClient(_response(choices=2)))

    with pytest.raises(ValueError, match="exactly one choice, got 2"):
        backend.complete(messages=[], tools=[], model_name="served-model", model_kwargs={})


def test_backend_omits_an_empty_tools_array_for_provider_compatibility():
    client = RecordingClient(_response())
    backend = OpenAICompatibleCompletionBackend(client=client)

    backend.complete(
        messages=[{"role": "user", "content": "Answer directly."}],
        tools=[],
        model_name="served-model",
        model_kwargs={"temperature": 0.0},
    )

    assert client.chat.completions.requests == [
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "Answer directly."}],
            "temperature": 0.0,
        }
    ]


def test_backend_does_not_close_a_caller_supplied_client():
    client = RecordingClient(_response())
    backend = OpenAICompatibleCompletionBackend(client=client)

    backend.close()
    backend.close()

    assert client.closed is False
    with pytest.raises(RuntimeError, match="completion backend is closed"):
        backend.complete(messages=[], tools=[], model_name="served-model", model_kwargs={})


def test_backend_requires_explicit_connection_ownership():
    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        OpenAICompatibleCompletionBackend()

    with pytest.raises(ValueError, match="must be configured on a caller-supplied client"):
        OpenAICompatibleCompletionBackend(client=RecordingClient(_response()), base_url="http://provider.invalid/v1")


@pytest.mark.optional_backend
def test_backend_builds_and_closes_an_owned_openai_client(monkeypatch):
    openai = pytest.importorskip("openai")
    client = RecordingClient(_response())
    constructor_kwargs = []

    def build_client(**kwargs):
        constructor_kwargs.append(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", build_client)
    with OpenAICompatibleCompletionBackend(
        base_url="http://provider.invalid/v1",
        timeout_seconds=45,
        max_retries=3,
    ):
        pass

    assert constructor_kwargs == [
        {
            "base_url": "http://provider.invalid/v1",
            "api_key": "EMPTY",
            "timeout": 45.0,
            "max_retries": 3,
        }
    ]
    assert client.closed is True


@pytest.mark.optional_backend
def test_backend_round_trips_the_pinned_openai_sdk_without_network_access():
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    requests = []

    def handle_request(request):
        requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": json.loads(request.content),
            }
        )
        message = _response().payload["choices"][0]["message"]
        message["reasoning_content"] = "Need to inspect the working tree first."
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    client = openai.OpenAI(base_url="http://provider.invalid/v1", api_key="test-key", http_client=http_client)
    backend = OpenAICompatibleCompletionBackend(client=client)
    try:
        result = backend.complete(
            messages=[{"role": "user", "content": "Inspect the checkout."}],
            tools=[{"type": "function", "function": {"name": "bash"}}],
            model_name="served-model",
            model_kwargs={"temperature": 0.0},
        )
    finally:
        client.close()

    assert requests == [
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "Inspect the checkout."}],
                "model": "served-model",
                "temperature": 0.0,
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            },
        }
    ]
    assert result["message"]["tool_calls"][0]["function"]["arguments"] == '{"command": "git status --short"}'
    assert result["message"]["reasoning_content"] == "Need to inspect the working tree first."
    assert result["raw_response"]["usage"]["total_tokens"] == 20
