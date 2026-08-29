"""GSM8K calculator rollouts for the synchronous token-native GRPO path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import torch

from lmflow.agentic.completion import CompletionBackend, normalize_completion_response, parse_function_arguments
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.gsm8k import extract_gsm8k_answer
from lmflow.agentic.gsm8k_evaluation import (
    GSM8K_CALCULATOR_SYSTEM_PROMPT,
    GSM8K_CALCULATOR_TOOL,
    GSM8K_CALCULATOR_TOOL_NAME,
    GSM8K_USER_PROMPT,
    CalculatorArithmeticError,
    CalculatorExpressionError,
    evaluate_arithmetic_expression,
)
from lmflow.agentic.vllm_token_native import (
    AssembledTokenSequence,
    VLLMChatTokenData,
    assemble_vllm_chat_token_data,
    extract_vllm_chat_token_data,
    vllm_token_native_model_kwargs,
)
from lmflow.utils.protocol import DataProto

GSM8K_GRPO_ROLLOUT_FORMAT = "lmflow.gsm8k-grpo-rollout/v1"
_CONTROLLED_MODEL_KWARGS = frozenset(
    {
        "logprobs",
        "max_tokens",
        "seed",
        "temperature",
        "tool_choice",
        "top_logprobs",
        "top_p",
    }
)
_UNSUPPORTED_PROVIDER_SAMPLING_FIELDS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "top_k",
    }
)
_MAX_VLLM_SEED = 2**63 - 1


def gsm8k_grpo_task_from_row(row: Mapping[str, Any]) -> tuple[TaskSpec, str]:
    """Build a calculator task while returning hidden verifier material separately."""

    if not isinstance(row, Mapping):
        raise TypeError("GSM8K row must be a mapping")
    question = row.get("question", row.get("input"))
    solution = row.get("answer", row.get("output"))
    instance_id = row.get("instance_id")
    source_index = row.get("source_index")
    row_digest = row.get("row_digest")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K row question must be a non-empty string")
    if not isinstance(solution, str):
        raise TypeError("GSM8K row answer must be a string")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("GSM8K GRPO rows require a canonical instance_id")
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError("GSM8K GRPO rows require a non-negative source_index")
    if not isinstance(row_digest, str) or len(row_digest) != 64:
        raise ValueError("GSM8K GRPO rows require a SHA-256 row_digest")
    try:
        int(row_digest, 16)
    except ValueError as error:
        raise ValueError("GSM8K GRPO row_digest must be hexadecimal") from error
    gold_answer = extract_gsm8k_answer(solution, method="strict")
    if gold_answer is None:
        raise ValueError("GSM8K row answer must contain a strict '#### <answer>' target")

    task = TaskSpec(
        task_id=instance_id,
        messages=[
            {"role": "system", "content": GSM8K_CALCULATOR_SYSTEM_PROMPT},
            {"role": "user", "content": GSM8K_USER_PROMPT.format(question=question)},
        ],
        tools=[copy.deepcopy(GSM8K_CALCULATOR_TOOL)],
        environment={"evaluation_mode": "calculator"},
        metadata={
            "source_index": source_index,
            "row_digest": row_digest,
            "identity_scope": "canonical_source",
        },
    )
    return task, gold_answer


def _validate_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_task(task: TaskSpec) -> None:
    if not isinstance(task, TaskSpec):
        raise TypeError("GSM8K rollout tasks must be TaskSpec instances")
    if not isinstance(task.task_id, str) or not task.task_id.strip():
        raise ValueError("GSM8K rollout task_id must be a non-empty string")
    if task.tools != [GSM8K_CALCULATOR_TOOL]:
        raise ValueError("GSM8K token-native rollout requires the fixed calculate(expression) tool")
    if task.environment.get("evaluation_mode") != "calculator":
        raise ValueError("GSM8K token-native rollout requires calculator evaluation_mode")
    if not task.messages:
        raise ValueError("GSM8K rollout task messages must not be empty")
    for index, message in enumerate(task.messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"task.messages[{index}] must be a mapping")
        if message.get("role") not in {"system", "user"}:
            raise ValueError(f"task.messages[{index}].role must be system or user")
        if not isinstance(message.get("content"), str):
            raise TypeError(f"task.messages[{index}].content must be a string")


def _request_id(
    *,
    policy_version: Hashable,
    task_id: str,
    group_id: int,
    rollout_id: int,
    call_index: int,
) -> str:
    payload = json.dumps(
        {
            "policy_version": str(policy_version),
            "task_id": task_id,
            "group_id": group_id,
            "rollout_id": rollout_id,
            "call_index": call_index,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"gsm8k-grpo-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def _object_array(values: list[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class GSM8KTokenNativeRollout:
    """Produce forced-calculator vLLM rollouts in the existing ``DataProto``."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        model_name: str,
        pad_token_id: int,
        model_revision: str,
        tokenizer_revision: str,
        policy_checkpoint_sha256: str,
        model_kwargs: Mapping[str, Any] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        base_seed: int = 0,
        max_tokens_per_call: int = 1024,
        max_model_calls: int = 4,
        max_tool_calls: int = 4,
        max_concurrency: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide complete()")
        for name, value in {
            "model_name": model_name,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(policy_checkpoint_sha256, str) or len(policy_checkpoint_sha256) != 64:
            raise ValueError("policy_checkpoint_sha256 must be a SHA-256")
        try:
            int(policy_checkpoint_sha256, 16)
        except ValueError as error:
            raise ValueError("policy_checkpoint_sha256 must be hexadecimal") from error
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        if model_kwargs is None:
            model_kwargs = {}
        if not isinstance(model_kwargs, Mapping) or any(not isinstance(key, str) for key in model_kwargs):
            raise TypeError("model_kwargs must be a string-keyed mapping")
        controlled = sorted(_CONTROLLED_MODEL_KWARGS.intersection(model_kwargs))
        if controlled:
            raise ValueError(f"GSM8K token-native rollout controls model_kwargs fields: {controlled}")
        extra_body = model_kwargs.get("extra_body", {})
        if not isinstance(extra_body, Mapping):
            raise TypeError("model_kwargs.extra_body must be a mapping")
        unsupported_sampling = sorted(_UNSUPPORTED_PROVIDER_SAMPLING_FIELDS.intersection(extra_body))
        if unsupported_sampling:
            raise ValueError(
                "current synchronous GRPO reference does not reproduce provider sampling transforms: "
                f"{unsupported_sampling}"
            )
        try:
            json.dumps(model_kwargs, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("model_kwargs must be JSON-compatible") from error
        if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= _MAX_VLLM_SEED:
            raise ValueError(f"base_seed must be an integer in [0, {_MAX_VLLM_SEED}]")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.backend = backend
        self.model_name = model_name
        self.pad_token_id = pad_token_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.policy_checkpoint_sha256 = policy_checkpoint_sha256
        self.model_kwargs = copy.deepcopy(dict(model_kwargs))
        self.temperature = _validate_number(temperature, name="temperature", minimum=0.0)
        self.top_p = _validate_number(top_p, name="top_p", minimum=0.0)
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise ValueError("current synchronous GRPO reference requires temperature=1.0 and top_p=1.0")
        self.base_seed = base_seed
        self.max_tokens_per_call = _validate_positive_integer(max_tokens_per_call, name="max_tokens_per_call")
        self.max_model_calls = _validate_positive_integer(max_model_calls, name="max_model_calls")
        self.max_tool_calls = _validate_positive_integer(max_tool_calls, name="max_tool_calls")
        self.max_concurrency = _validate_positive_integer(max_concurrency, name="max_concurrency")
        self.clock = clock

    def _sampling_seed(self, rollout_id: int, call_index: int) -> int:
        seed = self.base_seed + rollout_id * self.max_model_calls + call_index
        if seed > _MAX_VLLM_SEED:
            raise ValueError("derived vLLM sampling seed exceeds the signed 64-bit range")
        return seed

    def _run_one(
        self,
        *,
        task: TaskSpec,
        task_id: str,
        group_id: int,
        rollout_id: int,
        policy_version: Hashable,
    ) -> dict[str, Any]:
        _validate_task(task)
        if task.task_id != task_id:
            raise ValueError(f"request task_id {task_id!r} does not match TaskSpec {task.task_id!r}")

        history = copy.deepcopy(task.messages)
        token_calls: list[VLLMChatTokenData] = []
        call_metadata = []
        seen_call_ids: set[str] = set()
        tool_call_count = 0
        valid_tool_call_count = 0
        successful_tool_call_count = 0
        tool_error_count = 0
        final_response = ""
        termination_reason = "model_budget_exhausted"

        for call_index in range(self.max_model_calls):
            request_id = _request_id(
                policy_version=policy_version,
                task_id=task_id,
                group_id=group_id,
                rollout_id=rollout_id,
                call_index=call_index,
            )
            sampling_seed = self._sampling_seed(rollout_id, call_index)
            request_kwargs = vllm_token_native_model_kwargs(
                self.model_kwargs,
                request_id=request_id,
            )
            request_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "seed": sampling_seed,
                    "max_tokens": self.max_tokens_per_call,
                }
            )
            if call_index == 0:
                request_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": GSM8K_CALCULATOR_TOOL_NAME},
                }

            call_started_at = float(self.clock())
            response = self.backend.complete(
                messages=copy.deepcopy(history),
                tools=copy.deepcopy(task.tools),
                model_name=self.model_name,
                model_kwargs=request_kwargs,
            )
            call_finished_at = float(self.clock())
            call_latency_seconds = call_finished_at - call_started_at
            if not math.isfinite(call_latency_seconds) or call_latency_seconds < 0:
                raise ValueError("clock must produce finite, monotonically non-decreasing timestamps")
            completion = normalize_completion_response(response)
            token_call = extract_vllm_chat_token_data(
                completion,
                expected_request_id=request_id,
            )
            token_calls.append(token_call)
            call_metadata.append(
                {
                    "request_id": request_id,
                    "sampling_seed": sampling_seed,
                    "finish_reason": token_call.finish_reason,
                    "input_tokens": len(token_call.prompt_token_ids),
                    "output_tokens": len(token_call.output_token_ids),
                    "latency_seconds": call_latency_seconds,
                    "cost": completion["cost"],
                }
            )
            final_response = completion["content"]
            raw_tool_calls = completion["tool_calls"]

            if not raw_tool_calls:
                termination_reason = "completed" if tool_call_count > 0 else "missing_required_tool"
                break
            if tool_call_count + len(raw_tool_calls) > self.max_tool_calls:
                termination_reason = "tool_budget_exhausted"
                break

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": completion["content"],
                "tool_calls": copy.deepcopy(raw_tool_calls),
            }
            if completion["reasoning_content"] is not None:
                assistant_message["reasoning_content"] = completion["reasoning_content"]
            history.append(assistant_message)

            invalid_tool_call = False
            for tool_index, raw_call in enumerate(raw_tool_calls):
                path = f"completion tool_calls[{tool_index}]"
                tool_call_count += 1
                try:
                    if not isinstance(raw_call, Mapping) or raw_call.get("type") != "function":
                        raise ValueError(f"{path}.type must be 'function'")
                    call_id = raw_call.get("id")
                    function = raw_call.get("function")
                    if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                        raise ValueError(f"{path}.id must be a unique non-empty string")
                    if not isinstance(function, Mapping):
                        raise TypeError(f"{path}.function must be a mapping")
                    if function.get("name") != GSM8K_CALCULATOR_TOOL_NAME:
                        raise ValueError(f"{path}.function.name must be {GSM8K_CALCULATOR_TOOL_NAME!r}")
                    arguments = parse_function_arguments(
                        function.get("arguments"),
                        path=f"{path}.function.arguments",
                    )
                    if set(arguments) != {"expression"} or not isinstance(arguments["expression"], str):
                        raise ValueError(f"{path}.function.arguments must contain only string field 'expression'")
                    valid_tool_call_count += 1
                    try:
                        result = evaluate_arithmetic_expression(arguments["expression"])
                        observation = f"Calculator result: {result}"
                        successful_tool_call_count += 1
                    except CalculatorArithmeticError as error:
                        tool_error_count += 1
                        observation = f"Calculator error: {error}"
                except (CalculatorExpressionError, TypeError, ValueError):
                    invalid_tool_call = True
                    termination_reason = "invalid_tool_call"
                    break

                seen_call_ids.add(call_id)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": GSM8K_CALCULATOR_TOOL_NAME,
                        "content": observation,
                    }
                )
            if invalid_tool_call:
                break
        else:
            termination_reason = "model_budget_exhausted"

        sequence = assemble_vllm_chat_token_data(
            token_calls,
            optimize_calls=[False] + [True] * (len(token_calls) - 1),
        )
        return {
            "sequence": sequence,
            "final_response": final_response,
            "termination_reason": termination_reason,
            "tool_call_count": tool_call_count,
            "valid_tool_call_count": valid_tool_call_count,
            "successful_tool_call_count": successful_tool_call_count,
            "tool_error_count": tool_error_count,
            "model_call_count": len(token_calls),
            "model_input_token_count": sum(len(call.prompt_token_ids) for call in token_calls),
            "model_output_token_count": sum(len(call.output_token_ids) for call in token_calls),
            "model_latency_seconds": sum(call["latency_seconds"] for call in call_metadata),
            "model_cost": sum(call["cost"] for call in call_metadata),
            "metadata": {
                "format_version": GSM8K_GRPO_ROLLOUT_FORMAT,
                "task_id": task_id,
                "group_id": group_id,
                "rollout_id": rollout_id,
                "policy_version": policy_version,
                "calls": call_metadata,
                "call_spans": [copy.deepcopy(span) for span in sequence.call_spans],
                "termination_reason": termination_reason,
                "tool_call_count": tool_call_count,
                "valid_tool_call_count": valid_tool_call_count,
                "successful_tool_call_count": successful_tool_call_count,
                "tool_error_count": tool_error_count,
            },
        }

    def __call__(self, requests: DataProto) -> DataProto:
        if not isinstance(requests, DataProto):
            raise TypeError("GSM8K token-native rollout requires a DataProto request batch")
        try:
            tasks = requests.non_tensor_batch["tasks"]
            task_ids = requests.non_tensor_batch["task_ids"]
            group_ids = requests.non_tensor_batch["group_ids"]
            rollout_ids = requests.non_tensor_batch["rollout_ids"]
            policy_version = requests.meta_info["policy_version"]
        except KeyError as error:
            raise KeyError(
                "rollout requests require tasks, task_ids, group_ids, rollout_ids, and policy_version"
            ) from error
        if isinstance(policy_version, bool) or not isinstance(policy_version, str | int):
            raise TypeError("GSM8K policy_version must be a string or integer")
        if len(requests) == 0:
            raise ValueError("GSM8K rollout request batch must not be empty")
        arrays = {"tasks": tasks, "task_ids": task_ids, "group_ids": group_ids, "rollout_ids": rollout_ids}
        for name, values in arrays.items():
            if getattr(values, "ndim", None) != 1 or len(values) != len(requests):
                raise ValueError(f"{name} must have shape ({len(requests)},)")

        jobs = []
        for index, (task, task_id, group_id, rollout_id) in enumerate(
            zip(tasks, task_ids, group_ids, rollout_ids, strict=True)
        ):
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"task_ids[{index}] must be a non-empty string")
            if isinstance(group_id, bool) or not isinstance(group_id, int | np.integer):
                raise TypeError(f"group_ids[{index}] must be an integer")
            if isinstance(rollout_id, bool) or not isinstance(rollout_id, int | np.integer):
                raise TypeError(f"rollout_ids[{index}] must be an integer")
            if int(group_id) < 0 or int(rollout_id) < 0:
                raise ValueError("group_ids and rollout_ids must be non-negative")
            jobs.append(
                {
                    "task": task,
                    "task_id": task_id,
                    "group_id": int(group_id),
                    "rollout_id": int(rollout_id),
                    "policy_version": policy_version,
                }
            )

        if self.max_concurrency == 1:
            rows = [self._run_one(**job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                rows = list(executor.map(lambda job: self._run_one(**job), jobs))

        sequences: list[AssembledTokenSequence] = [row["sequence"] for row in rows]
        maximum_length = max(sequence.input_ids.shape[0] for sequence in sequences)
        input_ids = torch.full((len(rows), maximum_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), maximum_length), dtype=torch.long)
        loss_mask = torch.zeros((len(rows), maximum_length), dtype=torch.float32)
        old_log_probs = torch.zeros((len(rows), maximum_length), dtype=torch.float32)
        for index, sequence in enumerate(sequences):
            length = sequence.input_ids.shape[0]
            input_ids[index, :length] = sequence.input_ids
            attention_mask[index, :length] = sequence.attention_mask
            loss_mask[index, :length] = sequence.loss_mask
            old_log_probs[index, :length] = sequence.old_log_probs

        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "old_log_probs": old_log_probs,
            },
            non_tensors={
                "task_ids": np.asarray(task_ids).copy(),
                "group_ids": np.asarray(group_ids).copy(),
                "rollout_ids": np.asarray(rollout_ids).copy(),
                "final_responses": _object_array([row["final_response"] for row in rows]),
                "termination_reasons": _object_array([row["termination_reason"] for row in rows]),
                "tool_call_counts": np.asarray([row["tool_call_count"] for row in rows], dtype=np.int64),
                "valid_tool_call_counts": np.asarray([row["valid_tool_call_count"] for row in rows], dtype=np.int64),
                "successful_tool_call_counts": np.asarray(
                    [row["successful_tool_call_count"] for row in rows], dtype=np.int64
                ),
                "tool_error_counts": np.asarray([row["tool_error_count"] for row in rows], dtype=np.int64),
                "model_call_counts": np.asarray([row["model_call_count"] for row in rows], dtype=np.int64),
                "model_input_token_counts": np.asarray(
                    [row["model_input_token_count"] for row in rows], dtype=np.int64
                ),
                "model_output_token_counts": np.asarray(
                    [row["model_output_token_count"] for row in rows], dtype=np.int64
                ),
                "model_latency_seconds": np.asarray([row["model_latency_seconds"] for row in rows], dtype=np.float64),
                "model_costs": np.asarray([row["model_cost"] for row in rows], dtype=np.float64),
                "selected_token_counts": np.asarray(
                    [int(row["sequence"].loss_mask.sum().item()) for row in rows], dtype=np.int64
                ),
                "rollout_metadata": _object_array([row["metadata"] for row in rows]),
            },
            meta_info={
                "policy_version": policy_version,
                "rollout_format": GSM8K_GRPO_ROLLOUT_FORMAT,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
                "scaffold": "lmflow.gsm8k.chat-completions@v1",
                "capability": "calculate(expression)@v1:forced-first-call",
                "sampling": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "base_seed": self.base_seed,
                    "max_tokens_per_call": self.max_tokens_per_call,
                    "provider_kwargs": copy.deepcopy(self.model_kwargs),
                },
            },
        )


def _numeric_equivalence(solution: str, gold_answer: str) -> float:
    candidate = extract_gsm8k_answer(solution, method="flexible")
    if candidate is None:
        return 0.0
    try:
        return float(Decimal(candidate.replace(",", "")) == Decimal(gold_answer.replace(",", "")))
    except InvalidOperation:
        return 0.0


def gsm8k_correctness_rewards(
    rollouts: DataProto,
    gold_answers: Mapping[str, str],
) -> torch.Tensor:
    """Return correctness-only hidden-verifier rewards for GSM8K rollouts."""

    if not isinstance(gold_answers, Mapping):
        raise TypeError("gold_answers must be a mapping kept outside model-visible rollout data")
    try:
        task_ids = rollouts.non_tensor_batch["task_ids"]
        final_responses = rollouts.non_tensor_batch["final_responses"]
    except KeyError as error:
        raise KeyError("GSM8K rewards require task_ids and final_responses") from error
    if len(task_ids) != len(rollouts) or len(final_responses) != len(rollouts):
        raise ValueError("GSM8K reward inputs must align with the rollout batch")

    rewards = []
    for index, (task_id, final_response) in enumerate(zip(task_ids, final_responses, strict=True)):
        if task_id not in gold_answers:
            raise KeyError(f"missing hidden gold answer for task_id {task_id!r}")
        gold_answer = gold_answers[task_id]
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError(f"gold_answers[{task_id!r}] must be a non-empty string")
        if not isinstance(final_response, str):
            raise TypeError(f"final_responses[{index}] must be a string")
        rewards.append(_numeric_equivalence(final_response, gold_answer))
    return torch.tensor(rewards, dtype=torch.float32)


def gsm8k_protocol_rewards(
    rollouts: DataProto,
    gold_answers: Mapping[str, str],
) -> torch.Tensor:
    """Reward only correct completions with at least one successful calculator call."""

    correctness = gsm8k_correctness_rewards(rollouts, gold_answers)
    try:
        successful_calls = rollouts.non_tensor_batch["successful_tool_call_counts"]
        termination_reasons = rollouts.non_tensor_batch["termination_reasons"]
    except KeyError as error:
        raise KeyError("GSM8K protocol rewards require successful tool counts and termination reasons") from error
    if len(successful_calls) != len(rollouts) or len(termination_reasons) != len(rollouts):
        raise ValueError("GSM8K protocol reward inputs must align with the rollout batch")
    compliance = torch.tensor(
        [
            float(int(tool_calls) > 0 and termination_reason == "completed")
            for tool_calls, termination_reason in zip(successful_calls, termination_reasons, strict=True)
        ],
        dtype=torch.float32,
    )
    return correctness * compliance


def summarize_gsm8k_group_variance(
    rollouts: DataProto,
    rewards: torch.Tensor,
) -> dict[str, float]:
    """Summarize binary reward variance without changing the training batch."""

    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1 or rewards.shape[0] != len(rollouts):
        raise ValueError(f"rewards must have shape ({len(rollouts)},)")
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("GSM8K readiness rewards must be binary")
    try:
        group_ids = rollouts.non_tensor_batch["group_ids"]
    except KeyError as error:
        raise KeyError("GSM8K variance summary requires group_ids") from error
    if rollouts.batch is None or "loss_mask" not in rollouts.batch:
        raise KeyError("GSM8K variance summary requires loss_mask")
    selected_tokens = (rollouts.batch["loss_mask"] > 0).sum(dim=-1).detach().cpu().tolist()
    groups: dict[Hashable, list[tuple[float, int]]] = defaultdict(list)
    for group_id, reward, selected in zip(
        group_ids,
        rewards.detach().cpu().tolist(),
        selected_tokens,
        strict=True,
    ):
        if not isinstance(group_id, Hashable):
            raise TypeError("group_ids must be hashable")
        groups[group_id].append((float(reward), int(selected)))
    mixed = sum(1 for values in groups.values() if 0.0 < sum(reward for reward, _ in values) < len(values))
    update_ready_groups = sum(1 for values in groups.values() if all(selected > 0 for _, selected in values))
    update_ready_mixed_groups = sum(
        1
        for values in groups.values()
        if all(selected > 0 for _, selected in values) and 0.0 < sum(reward for reward, _ in values) < len(values)
    )
    return {
        "rollout/groups": float(len(groups)),
        "rollout/mixed_groups": float(mixed),
        "rollout/mixed_group_rate": mixed / len(groups) if groups else 0.0,
        "rollout/update_ready_trajectories": float(sum(selected > 0 for selected in selected_tokens)),
        "rollout/update_ready_groups": float(update_ready_groups),
        "rollout/update_ready_mixed_groups": float(update_ready_mixed_groups),
        "reward/total_mean": rewards.detach().float().mean().item() if len(rewards) else 0.0,
    }


__all__ = [
    "GSM8K_GRPO_ROLLOUT_FORMAT",
    "GSM8KTokenNativeRollout",
    "gsm8k_correctness_rewards",
    "gsm8k_grpo_task_from_row",
    "gsm8k_protocol_rewards",
    "summarize_gsm8k_group_variance",
]
