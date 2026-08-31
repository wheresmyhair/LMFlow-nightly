"""Benchmark-local token-native evidence for AppWorld policy calls."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lmflow.agentic.appworld_protocol import canonical_json_sha256, with_manifest_digest
from lmflow.agentic.completion import CompletionBackend, normalize_completion_response
from lmflow.agentic.vllm_token_native import (
    AssembledTokenSequence,
    VLLMChatTokenData,
    assemble_vllm_chat_token_data,
    extract_vllm_chat_token_data,
    vllm_token_native_model_kwargs,
)

APPWORLD_TOKEN_NATIVE_AUDIT_FORMAT_VERSION = "lmflow.appworld-token-native-audit/v1"
APPWORLD_QWEN3_REASONING_REPLAY_POLICY_ID = "lmflow.appworld-qwen3-reasoning-replay/v1"

_QWEN3_LAST_QUERY_CONDITION = (
    "and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>'))"
)
_APPWORLD_LAST_QUERY_CONDITION = _QWEN3_LAST_QUERY_CONDITION + (" and not(message.content.startswith('Output:\\n'))")

PromptTokenIdsRenderer = Callable[[list[dict[str, Any]], Mapping[str, Any]], Sequence[int]]
EvidenceSink = Callable[[str, int, Mapping[str, Any]], None]


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def _token_ids(value: Any, *, name: str) -> tuple[int, ...]:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{name} mapping must contain input_ids")
        value = value["input_ids"]
        if hasattr(value, "tolist") and callable(value.tolist):
            value = value.tolist()
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], str | bytes)
    ):
        value = value[0]
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of token IDs")
    result = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
        result.append(token_id)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _first_difference(expected: Sequence[int], actual: Sequence[int]) -> dict[str, Any] | None:
    for position, (expected_token, actual_token) in enumerate(zip(expected, actual, strict=False)):
        if expected_token != actual_token:
            return {
                "position": position,
                "expected_token_id": expected_token,
                "actual_token_id": actual_token,
                "kind": "token_mismatch",
            }
    if len(expected) != len(actual):
        position = min(len(expected), len(actual))
        return {
            "position": position,
            "expected_token_id": expected[position] if position < len(expected) else None,
            "actual_token_id": actual[position] if position < len(actual) else None,
            "kind": "length_mismatch",
        }
    return None


def _prefix_difference(expected_prefix: Sequence[int], actual: Sequence[int]) -> dict[str, Any] | None:
    comparable = actual[: len(expected_prefix)]
    difference = _first_difference(expected_prefix, comparable)
    if difference is not None:
        return difference
    if len(actual) < len(expected_prefix):
        position = len(actual)
        return {
            "position": position,
            "expected_token_id": expected_prefix[position],
            "actual_token_id": None,
            "kind": "actual_prompt_shorter",
        }
    return None


def qwen3_appworld_replay_chat_template(tokenizer: Any) -> str:
    """Keep sampled Qwen3 reasoning across AppWorld ``Output:`` observations.

    Qwen3's stock template treats every plain ``user`` message as a new query and
    therefore drops earlier ``reasoning_content``. The pinned AppWorld ReAct-code
    scaffold represents execution observations as ``user`` messages whose content
    begins with ``Output:``. This benchmark-local derivation keeps those messages
    model-visible exactly as the official scaffold specifies while excluding them
    from Qwen3's last-query boundary.
    """

    source_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(source_template, str) or not source_template:
        raise ValueError("Qwen3 tokenizer must contain a non-empty chat template")
    if source_template.count(_QWEN3_LAST_QUERY_CONDITION) != 1:
        raise ValueError("Qwen3 chat template last-query boundary is unsupported")
    return source_template.replace(_QWEN3_LAST_QUERY_CONDITION, _APPWORLD_LAST_QUERY_CONDITION)


def qwen3_appworld_replay_chat_template_identity(tokenizer: Any) -> dict[str, str]:
    """Return the fail-closed source and derived template identities."""

    source_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(source_template, str) or not source_template:
        raise ValueError("Qwen3 tokenizer must contain a non-empty chat template")
    replay_template = qwen3_appworld_replay_chat_template(tokenizer)
    return {
        "policy_id": APPWORLD_QWEN3_REASONING_REPLAY_POLICY_ID,
        "source_chat_template_sha256": hashlib.sha256(source_template.encode("utf-8")).hexdigest(),
        "replay_chat_template_sha256": hashlib.sha256(replay_template.encode("utf-8")).hexdigest(),
    }


def qwen3_appworld_prompt_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    model_kwargs: Mapping[str, Any],
) -> tuple[int, ...]:
    """Render the exact AppWorld chat prompt with Qwen3 chat-template options."""

    if not isinstance(messages, list) or any(not isinstance(message, Mapping) for message in messages):
        raise TypeError("messages must be a list of mappings")
    if not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping")
    extra_body = model_kwargs.get("extra_body", {})
    if not isinstance(extra_body, Mapping):
        raise TypeError("model_kwargs.extra_body must be a mapping")
    chat_template_kwargs = extra_body.get("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, Mapping):
        raise TypeError("model_kwargs.extra_body.chat_template_kwargs must be a mapping")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("tokenizer must expose apply_chat_template()")
    rendered = apply_chat_template(
        copy.deepcopy(messages),
        tokenize=True,
        add_generation_prompt=True,
        chat_template=qwen3_appworld_replay_chat_template(tokenizer),
        **copy.deepcopy(dict(chat_template_kwargs)),
    )
    return _token_ids(rendered, name="tokenizer.apply_chat_template result")


def build_appworld_token_native_audit(
    calls: Sequence[VLLMChatTokenData],
    *,
    expected_prompt_token_ids: Sequence[Sequence[int]],
    policy_version: str,
) -> dict[str, Any]:
    """Audit canonical prompts and sampled-token anchors without inventing tokens."""

    if isinstance(calls, str | bytes) or not isinstance(calls, Sequence):
        raise TypeError("calls must be a sequence")
    if not calls or any(not isinstance(call, VLLMChatTokenData) for call in calls):
        raise ValueError("calls must contain at least one VLLMChatTokenData")
    if isinstance(expected_prompt_token_ids, str | bytes) or not isinstance(expected_prompt_token_ids, Sequence):
        raise TypeError("expected_prompt_token_ids must be a sequence")
    if len(expected_prompt_token_ids) != len(calls):
        raise ValueError("expected_prompt_token_ids must contain one prompt per call")
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("policy_version must be a non-empty string")

    expected_prompts = [
        _token_ids(prompt, name=f"expected_prompt_token_ids[{index}]")
        for index, prompt in enumerate(expected_prompt_token_ids)
    ]
    call_evidence = []
    canonical_prompts_match = True
    for index, (call, expected_prompt) in enumerate(zip(calls, expected_prompts, strict=True)):
        actual_prompt = tuple(call.prompt_token_ids)
        difference = _first_difference(expected_prompt, actual_prompt)
        canonical_match = difference is None
        canonical_prompts_match = canonical_prompts_match and canonical_match
        call_evidence.append(
            {
                "call_index": index,
                "request_id": call.request_id,
                "response_id": call.response_id,
                "expected_prompt_token_ids": list(expected_prompt),
                "actual_prompt_token_ids": list(actual_prompt),
                "sampled_output_token_ids": list(call.output_token_ids),
                "sampled_output_log_probs": list(call.output_log_probs),
                "finish_reason": call.finish_reason,
                "canonical_prompt_match": canonical_match,
                "canonical_first_difference": difference,
            }
        )

    transition_evidence = []
    sampled_anchors_match = True
    for call_index in range(1, len(calls)):
        previous = calls[call_index - 1]
        current = calls[call_index]
        expected_prefix = tuple(previous.prompt_token_ids) + tuple(previous.output_token_ids)
        difference = _prefix_difference(expected_prefix, current.prompt_token_ids)
        prefix_match = difference is None
        sampled_anchors_match = sampled_anchors_match and prefix_match
        environment_token_ids = list(current.prompt_token_ids[len(expected_prefix) :]) if prefix_match else None
        transition_evidence.append(
            {
                "from_call_index": call_index - 1,
                "to_call_index": call_index,
                "expected_sampled_prefix_tokens": len(expected_prefix),
                "actual_next_prompt_tokens": len(current.prompt_token_ids),
                "sampled_anchor_match": prefix_match,
                "sampled_anchor_first_difference": difference,
                "environment_token_ids": environment_token_ids,
                "environment_tokens": len(environment_token_ids) if environment_token_ids is not None else None,
            }
        )

    flattened_ready = canonical_prompts_match and sampled_anchors_match
    flattened_sequence = None
    if flattened_ready:
        assembled = assemble_vllm_chat_token_data(calls)
        flattened_sequence = {
            "input_ids": assembled.input_ids.tolist(),
            "attention_mask": assembled.attention_mask.tolist(),
            "policy_origin_mask": assembled.loss_mask.tolist(),
            "sampled_old_log_probs": assembled.old_log_probs.tolist(),
            "call_spans": [dict(span) for span in assembled.call_spans],
        }

    return with_manifest_digest(
        {
            "format_version": APPWORLD_TOKEN_NATIVE_AUDIT_FORMAT_VERSION,
            "policy_version": policy_version,
            "call_count": len(calls),
            "canonical_prompts_match": canonical_prompts_match,
            "sampled_anchors_match": sampled_anchors_match,
            "single_flattened_rollout_ready": flattened_ready,
            "calls": call_evidence,
            "transitions": transition_evidence,
            "flattened_sequence": flattened_sequence,
            "mask_semantics": {
                "policy_origin_mask_one": "actual sampled assistant completion token",
                "policy_origin_mask_zero": (
                    "prompt, scaffold, history, tool observation, or deterministic transport token"
                ),
                "training_loss_mask_frozen": False,
            },
            "retokenized_sampled_tokens_used": False,
            "hidden_verifier_material_included": False,
        }
    )


def assemble_verified_appworld_token_sequence(
    calls: Sequence[VLLMChatTokenData],
    *,
    expected_prompt_token_ids: Sequence[Sequence[int]],
    policy_version: str,
) -> AssembledTokenSequence:
    """Build a flattened sequence only after the AppWorld continuity audit passes."""

    audit = build_appworld_token_native_audit(
        calls,
        expected_prompt_token_ids=expected_prompt_token_ids,
        policy_version=policy_version,
    )
    if not audit["canonical_prompts_match"]:
        difference = next(
            item["canonical_first_difference"] for item in audit["calls"] if not item["canonical_prompt_match"]
        )
        raise ValueError(f"AppWorld canonical prompt mismatch: {difference}")
    if not audit["sampled_anchors_match"]:
        difference = next(
            item["sampled_anchor_first_difference"] for item in audit["transitions"] if not item["sampled_anchor_match"]
        )
        raise ValueError(f"AppWorld sampled-token anchor mismatch: {difference}")
    return assemble_vllm_chat_token_data(calls)


class AppWorldTokenNativeCompletionRecorder:
    """Add vLLM token metadata to AppWorld calls and retain auditable evidence."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        request_id_prefix: str,
        prompt_token_ids_renderer: PromptTokenIdsRenderer,
        evidence_sink: EvidenceSink | None = None,
    ) -> None:
        if not isinstance(request_id_prefix, str) or not request_id_prefix.strip():
            raise ValueError("request_id_prefix must be a non-empty string")
        if not callable(prompt_token_ids_renderer):
            raise TypeError("prompt_token_ids_renderer must be callable")
        if evidence_sink is not None and not callable(evidence_sink):
            raise TypeError("evidence_sink must be callable")
        self._backend = backend
        self._request_id_prefix = request_id_prefix
        self._prompt_token_ids_renderer = prompt_token_ids_renderer
        self._evidence_sink = evidence_sink
        self._attempt_count = 0
        self._calls: list[VLLMChatTokenData] = []
        self._expected_prompts: list[tuple[int, ...]] = []

    @property
    def calls(self) -> tuple[VLLMChatTokenData, ...]:
        return tuple(self._calls)

    @property
    def expected_prompt_token_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._expected_prompts)

    def _emit(self, stage: str, call_index: int, evidence: Mapping[str, Any]) -> None:
        if self._evidence_sink is not None:
            self._evidence_sink(stage, call_index, _json_copy(evidence, name=f"{stage} evidence"))

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        model_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        call_index = self._attempt_count
        self._attempt_count += 1
        request_id = f"{self._request_id_prefix}-call-{call_index:03d}"
        expected_prompt = _token_ids(
            self._prompt_token_ids_renderer(copy.deepcopy(messages), copy.deepcopy(dict(model_kwargs))),
            name=f"expected prompt for AppWorld call {call_index}",
        )
        request_kwargs = vllm_token_native_model_kwargs(model_kwargs, request_id=request_id)
        self._emit(
            "request_intent",
            call_index,
            {
                "request_id": request_id,
                "model_name": model_name,
                "messages": messages,
                "messages_sha256": canonical_json_sha256(messages),
                "model_kwargs": request_kwargs,
                "model_kwargs_sha256": canonical_json_sha256(request_kwargs),
                "expected_prompt_tokens": len(expected_prompt),
            },
        )
        response = self._backend.complete(
            messages=messages,
            tools=tools,
            model_name=model_name,
            model_kwargs=request_kwargs,
        )
        self._emit("raw_response", call_index, response)
        completion = normalize_completion_response(response)
        self._emit(
            "normalized_response",
            call_index,
            {
                "content": completion["content"],
                "reasoning_content": completion["reasoning_content"],
                "tool_calls": completion["tool_calls"],
                "finish_reason": completion["finish_reason"],
                "cost": completion["cost"],
            },
        )
        try:
            token_call = extract_vllm_chat_token_data(completion, expected_request_id=request_id)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            self._emit(
                "token_evidence_error",
                call_index,
                {
                    "request_id": request_id,
                    "type": type(error).__name__,
                    "message": str(error),
                    "status": "invalid_token_identity",
                },
            )
            raise
        difference = _first_difference(expected_prompt, token_call.prompt_token_ids)
        self._emit(
            "token_evidence",
            call_index,
            {
                "request_id": request_id,
                "response_id": token_call.response_id,
                "expected_prompt_token_ids": list(expected_prompt),
                "actual_prompt_token_ids": list(token_call.prompt_token_ids),
                "sampled_output_token_ids": list(token_call.output_token_ids),
                "sampled_output_log_probs": list(token_call.output_log_probs),
                "finish_reason": token_call.finish_reason,
                "canonical_prompt_match": difference is None,
                "canonical_first_difference": difference,
            },
        )
        self._calls.append(token_call)
        self._expected_prompts.append(expected_prompt)
        self._emit(
            "request_termination",
            call_index,
            {
                "request_id": request_id,
                "response_id": token_call.response_id,
                "finish_reason": token_call.finish_reason,
                "status": "sealed",
            },
        )
        return response

    def build_audit(self, *, policy_version: str) -> dict[str, Any]:
        return build_appworld_token_native_audit(
            self.calls,
            expected_prompt_token_ids=self.expected_prompt_token_ids,
            policy_version=policy_version,
        )


__all__ = [
    "APPWORLD_QWEN3_REASONING_REPLAY_POLICY_ID",
    "APPWORLD_TOKEN_NATIVE_AUDIT_FORMAT_VERSION",
    "AppWorldTokenNativeCompletionRecorder",
    "assemble_verified_appworld_token_sequence",
    "build_appworld_token_native_audit",
    "qwen3_appworld_prompt_token_ids",
    "qwen3_appworld_replay_chat_template",
    "qwen3_appworld_replay_chat_template_identity",
]
