import json

import pytest
import torch

from lmflow.agentic.completion import OpenAICompatibleCompletionBackend, normalize_completion_response
from lmflow.agentic.vllm_token_native import (
    VLLMChatTokenData,
    assemble_vllm_chat_token_data,
    extract_vllm_chat_token_data,
    vllm_token_native_model_kwargs,
)


def _completion(*, response_id="chatcmpl-request-1", prompt=(1, 2), output=(3, 4), logprobs=(-0.1, -0.2)):
    return {
        "finish_reason": "stop",
        "raw_response": {
            "id": response_id,
            "prompt_token_ids": list(prompt),
            "choices": [
                {
                    "finish_reason": "stop",
                    "token_ids": list(output),
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{token_id}", "logprob": logprob}
                            for token_id, logprob in zip(output, logprobs, strict=True)
                        ]
                    },
                }
            ],
        },
    }


def test_adds_vllm_token_native_request_fields_without_overwriting_provider_options():
    actual = vllm_token_native_model_kwargs(
        {"extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}},
        request_id="rollout-7-call-0",
    )

    assert actual == {
        "logprobs": True,
        "top_logprobs": 0,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True},
            "request_id": "rollout-7-call-0",
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
    }


@pytest.mark.parametrize(
    "model_kwargs,match",
    [
        ({"logprobs": False}, "controls model_kwargs"),
        ({"extra_body": {"return_token_ids": False}}, "controls extra_body"),
    ],
)
def test_rejects_callers_overriding_token_native_request_fields(model_kwargs, match):
    with pytest.raises(ValueError, match=match):
        vllm_token_native_model_kwargs(model_kwargs, request_id="request-1")


def test_extracts_exact_sampled_token_ids_and_log_probs():
    actual = extract_vllm_chat_token_data(_completion(), expected_request_id="request-1")

    assert actual == VLLMChatTokenData(
        request_id="request-1",
        response_id="chatcmpl-request-1",
        prompt_token_ids=(1, 2),
        output_token_ids=(3, 4),
        output_log_probs=(-0.1, -0.2),
        finish_reason="stop",
    )


@pytest.mark.parametrize("response_id", ["request-1", "chatcmpl-wrong", "cmpl-request-1"])
def test_rejects_response_ids_that_do_not_match_vllm_chat_completion_mapping(response_id):
    with pytest.raises(ValueError, match="vLLM chat response ID mismatch"):
        extract_vllm_chat_token_data(
            _completion(response_id=response_id),
            expected_request_id="request-1",
        )


def test_rejects_decoded_token_text_as_sampled_token_identity():
    completion = _completion()
    completion["raw_response"]["choices"][0]["logprobs"]["content"][0]["token"] = "hello"

    with pytest.raises(ValueError, match="cannot prove sampled-token identity"):
        extract_vllm_chat_token_data(completion, expected_request_id="request-1")


def test_assembles_policy_and_environment_tokens_without_retokenizing():
    calls = [
        VLLMChatTokenData("first", "chatcmpl-first", (1, 2), (3, 4), (-0.1, -0.2), "tool_calls"),
        VLLMChatTokenData("second", "chatcmpl-second", (1, 2, 3, 4, 5, 6), (7,), (-0.3,), "stop"),
    ]

    actual = assemble_vllm_chat_token_data(calls)

    torch.testing.assert_close(actual.input_ids, torch.tensor([1, 2, 3, 4, 5, 6, 7]))
    torch.testing.assert_close(actual.attention_mask, torch.ones(7, dtype=torch.long))
    torch.testing.assert_close(actual.loss_mask, torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(
        actual.old_log_probs,
        torch.tensor([0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3]),
    )
    assert actual.call_spans == (
        {
            "request_id": "first",
            "response_id": "chatcmpl-first",
            "prompt_tokens": 2,
            "output_start": 2,
            "output_end": 4,
            "finish_reason": "tool_calls",
            "optimized": True,
        },
        {
            "request_id": "second",
            "response_id": "chatcmpl-second",
            "prompt_tokens": 6,
            "output_start": 6,
            "output_end": 7,
            "finish_reason": "stop",
            "optimized": True,
        },
    )


def test_can_exclude_framework_forced_generation_from_policy_loss():
    calls = [
        VLLMChatTokenData("forced", "chatcmpl-forced", (1, 2), (3, 4), (-0.1, -0.2), "tool_calls"),
        VLLMChatTokenData("policy", "chatcmpl-policy", (1, 2, 3, 4, 5), (6,), (-0.3,), "stop"),
    ]

    actual = assemble_vllm_chat_token_data(calls, optimize_calls=[False, True])

    torch.testing.assert_close(actual.loss_mask, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert actual.call_spans[0]["optimized"] is False
    assert actual.call_spans[1]["optimized"] is True


def test_fails_closed_when_next_turn_prompt_does_not_preserve_sampled_prefix():
    calls = [
        VLLMChatTokenData("first", "chatcmpl-first", (1, 2), (3, 4), (-0.1, -0.2), "tool_calls"),
        VLLMChatTokenData("second", "chatcmpl-second", (1, 2, 3, 99, 5), (6,), (-0.3,), "stop"),
    ]

    with pytest.raises(ValueError, match="prefix drift at position 3"):
        assemble_vllm_chat_token_data(calls)


@pytest.mark.optional_backend
def test_pinned_openai_sdk_preserves_vllm_token_metadata_without_network_access():
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    requests = []

    def handle_request(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-request-1",
                "object": "chat.completion",
                "created": 1,
                "model": "served-model",
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                        "token_ids": [3, 4],
                        "logprobs": {
                            "content": [
                                {"token": "token_id:3", "logprob": -0.1, "top_logprobs": []},
                                {"token": "token_id:4", "logprob": -0.2, "top_logprobs": []},
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    client = openai.OpenAI(
        base_url="http://provider.invalid/v1",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
    )
    backend = OpenAICompatibleCompletionBackend(client=client)
    try:
        response = backend.complete(
            messages=[{"role": "user", "content": "question"}],
            tools=[],
            model_name="served-model",
            model_kwargs=vllm_token_native_model_kwargs({}, request_id="request-1"),
        )
    finally:
        client.close()

    token_data = extract_vllm_chat_token_data(
        normalize_completion_response(response),
        expected_request_id="request-1",
    )
    assert token_data.output_token_ids == (3, 4)
    assert token_data.request_id == "request-1"
    assert token_data.response_id == "chatcmpl-request-1"
    assert requests[0]["request_id"] == "request-1"
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["return_tokens_as_token_ids"] is True
    assert requests[0]["logprobs"] is True
    assert requests[0]["top_logprobs"] == 0
