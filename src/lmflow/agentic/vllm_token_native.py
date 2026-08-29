"""Strict vLLM Chat Completions token metadata handling for online RL."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

_TOKEN_ID_PREFIX = "token_id:"
_CHAT_COMPLETION_ID_PREFIX = "chatcmpl-"
_CONTROLLED_EXTRA_BODY_FIELDS = frozenset(
    {
        "request_id",
        "return_token_ids",
        "return_tokens_as_token_ids",
    }
)
_CONTROLLED_MODEL_FIELDS = frozenset({"logprobs", "top_logprobs"})


@dataclass(frozen=True)
class VLLMChatTokenData:
    """One vLLM chat call's actual prompt and sampled token data."""

    request_id: str
    response_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_log_probs: tuple[float, ...]
    finish_reason: str | None


@dataclass(frozen=True)
class AssembledTokenSequence:
    """One multi-turn sequence with policy-token spans and old log-probs."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    old_log_probs: torch.Tensor
    call_spans: tuple[dict[str, Any], ...]


def vllm_token_native_model_kwargs(
    model_kwargs: Mapping[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    """Add the exact vLLM response fields required by an online-RL rollout."""

    if not isinstance(model_kwargs, Mapping) or any(not isinstance(key, str) for key in model_kwargs):
        raise TypeError("model_kwargs must be a string-keyed mapping")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")

    controlled = sorted(_CONTROLLED_MODEL_FIELDS.intersection(model_kwargs))
    if controlled:
        raise ValueError(f"token-native rollout controls model_kwargs fields: {controlled}")

    result = copy.deepcopy(dict(model_kwargs))
    extra_body = result.get("extra_body", {})
    if not isinstance(extra_body, Mapping) or any(not isinstance(key, str) for key in extra_body):
        raise TypeError("model_kwargs.extra_body must be a string-keyed mapping")
    controlled_extra = sorted(_CONTROLLED_EXTRA_BODY_FIELDS.intersection(extra_body))
    if controlled_extra:
        raise ValueError(f"token-native rollout controls extra_body fields: {controlled_extra}")

    result.update(
        {
            "logprobs": True,
            "top_logprobs": 0,
            "extra_body": {
                **copy.deepcopy(dict(extra_body)),
                "request_id": request_id,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
        }
    )
    return result


def _token_ids(value: Any, *, path: str, allow_empty: bool) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of token IDs")
    token_ids = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{path}[{index}] must be a non-negative integer")
        token_ids.append(token_id)
    if not allow_empty and not token_ids:
        raise ValueError(f"{path} must not be empty")
    return tuple(token_ids)


def extract_vllm_chat_token_data(
    completion: Mapping[str, Any],
    *,
    expected_request_id: str,
) -> VLLMChatTokenData:
    """Extract exact sampled IDs/log-probs from one normalized vLLM response."""

    if not isinstance(completion, Mapping):
        raise TypeError("completion must be a mapping")
    if not isinstance(expected_request_id, str) or not expected_request_id.strip():
        raise ValueError("expected_request_id must be a non-empty string")
    raw_response = completion.get("raw_response")
    if not isinstance(raw_response, Mapping):
        raise TypeError("completion.raw_response must be a mapping")
    response_id = raw_response.get("id")
    expected_response_id = f"{_CHAT_COMPLETION_ID_PREFIX}{expected_request_id}"
    if response_id != expected_response_id:
        raise ValueError(f"vLLM chat response ID mismatch: expected {expected_response_id!r}, got {response_id!r}")

    prompt_token_ids = _token_ids(
        raw_response.get("prompt_token_ids"),
        path="raw_response.prompt_token_ids",
        allow_empty=False,
    )
    choices = raw_response.get("choices")
    if isinstance(choices, str | bytes) or not isinstance(choices, Sequence):
        raise TypeError("raw_response.choices must be a sequence")
    if len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("raw_response.choices must contain exactly one mapping")
    choice = choices[0]
    output_token_ids = _token_ids(
        choice.get("token_ids"),
        path="raw_response.choices[0].token_ids",
        allow_empty=False,
    )

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        raise TypeError("raw_response.choices[0].logprobs must be a mapping")
    logprob_content = logprobs.get("content")
    if isinstance(logprob_content, str | bytes) or not isinstance(logprob_content, Sequence):
        raise TypeError("raw_response.choices[0].logprobs.content must be a sequence")
    if len(logprob_content) != len(output_token_ids):
        raise ValueError(
            "sampled token/log-prob length mismatch: "
            f"{len(output_token_ids)} token IDs and {len(logprob_content)} log-probs"
        )

    output_log_probs = []
    for index, (token_id, item) in enumerate(zip(output_token_ids, logprob_content, strict=True)):
        path = f"raw_response.choices[0].logprobs.content[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{path} must be a mapping")
        token = item.get("token")
        expected_token = f"{_TOKEN_ID_PREFIX}{token_id}"
        if token != expected_token:
            raise ValueError(f"{path}.token must be {expected_token!r}; token text cannot prove sampled-token identity")
        logprob = item.get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, int | float) or not math.isfinite(logprob):
            raise ValueError(f"{path}.logprob must be a finite number")
        output_log_probs.append(float(logprob))

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TypeError("raw_response.choices[0].finish_reason must be a string or null")
    normalized_finish_reason = completion.get("finish_reason")
    if normalized_finish_reason != finish_reason:
        raise ValueError(f"normalized and raw finish reasons differ: {normalized_finish_reason!r} != {finish_reason!r}")

    return VLLMChatTokenData(
        request_id=expected_request_id,
        response_id=response_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        output_log_probs=tuple(output_log_probs),
        finish_reason=finish_reason,
    )


def assemble_vllm_chat_token_data(
    calls: Sequence[VLLMChatTokenData],
    *,
    optimize_calls: Sequence[bool] | None = None,
) -> AssembledTokenSequence:
    """Assemble exact multi-turn tokens and fail on any re-rendered prefix drift."""

    if isinstance(calls, str | bytes) or not isinstance(calls, Sequence):
        raise TypeError("calls must be a sequence")
    if not calls:
        raise ValueError("calls must contain at least one token-native completion")
    if any(not isinstance(call, VLLMChatTokenData) for call in calls):
        raise TypeError("calls must contain only VLLMChatTokenData")
    if optimize_calls is None:
        optimize_calls = [True] * len(calls)
    if isinstance(optimize_calls, str | bytes) or not isinstance(optimize_calls, Sequence):
        raise TypeError("optimize_calls must be a sequence of booleans")
    if len(optimize_calls) != len(calls) or any(not isinstance(value, bool) for value in optimize_calls):
        raise ValueError("optimize_calls must contain one boolean per token-native completion")

    sequence: list[int] = []
    loss_mask: list[float] = []
    old_log_probs: list[float] = []
    spans = []

    for index, (call, optimize) in enumerate(zip(calls, optimize_calls, strict=True)):
        prompt = list(call.prompt_token_ids)
        if index == 0:
            sequence.extend(prompt)
            loss_mask.extend([0.0] * len(prompt))
            old_log_probs.extend([0.0] * len(prompt))
        else:
            if len(prompt) < len(sequence):
                raise ValueError(f"vLLM call {index} prompt is shorter than the previously assembled token sequence")
            if prompt[: len(sequence)] != sequence:
                mismatch = next(
                    position
                    for position, (actual, expected) in enumerate(zip(prompt, sequence, strict=False))
                    if actual != expected
                )
                raise ValueError(
                    f"vLLM call {index} prompt token prefix drift at position {mismatch}; "
                    "text re-rendering cannot be used as sampled training tokens"
                )
            environment_tokens = prompt[len(sequence) :]
            sequence.extend(environment_tokens)
            loss_mask.extend([0.0] * len(environment_tokens))
            old_log_probs.extend([0.0] * len(environment_tokens))

        output_start = len(sequence)
        sequence.extend(call.output_token_ids)
        loss_mask.extend([float(optimize)] * len(call.output_token_ids))
        old_log_probs.extend(call.output_log_probs)
        output_end = len(sequence)
        spans.append(
            {
                "request_id": call.request_id,
                "response_id": call.response_id,
                "prompt_tokens": len(call.prompt_token_ids),
                "output_start": output_start,
                "output_end": output_end,
                "finish_reason": call.finish_reason,
                "optimized": optimize,
            }
        )

    if not sequence or loss_mask[0] != 0.0:
        raise ValueError("assembled chat sequence must begin with at least one non-policy prompt token")

    input_ids = torch.tensor(sequence, dtype=torch.long)
    return AssembledTokenSequence(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
        old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32),
        call_spans=tuple(spans),
    )


__all__ = [
    "AssembledTokenSequence",
    "VLLMChatTokenData",
    "assemble_vllm_chat_token_data",
    "extract_vllm_chat_token_data",
    "vllm_token_native_model_kwargs",
]
