"""GSM8K direct-answer and arithmetic-calculator evaluation recipes."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from lmflow.agentic.completion import (
    CompletionBackend,
    normalize_completion_response,
    parse_function_arguments,
)
from lmflow.agentic.contracts import TaskSpec
from lmflow.agentic.gsm8k import GSM8K_DATA_SOURCE, extract_gsm8k_answer, score_gsm8k_answer
from lmflow.pipeline.evaluation.recipe import (
    CapabilityProfile,
    EvaluationBudget,
    EvaluationRecipe,
    EvaluationTask,
    ModelRunOutput,
    SamplingConfig,
    VerificationOutcome,
)
from lmflow.pipeline.evaluation.result import (
    EvaluationFailureType,
    EvaluationSampleError,
    EvaluationUsage,
)

GSM8K_CALCULATOR_CAPABILITY = "calculator"
GSM8K_CALCULATOR_TOOL_NAME = "calculate"
GSM8K_DIRECT_RECIPE_NAME = "gsm8k-direct-answer-v1"
GSM8K_CALCULATOR_RECIPE_NAME = "gsm8k-calculator-v1"
GSM8K_REFERENCE_SCAFFOLD = {
    "role": "reference",
    "id": "lmflow.gsm8k.chat-completions",
    "revision": "v1",
}

GSM8K_DIRECT_SYSTEM_PROMPT = (
    "Solve the math problem carefully. Return the final numeric answer in the format `#### <answer>`."
)
GSM8K_CALCULATOR_SYSTEM_PROMPT = (
    "Solve the math problem carefully. Use the arithmetic calculator when it is useful, then return the final "
    "numeric answer in the format `#### <answer>`."
)
GSM8K_USER_PROMPT = "{question}\n\nShow your reasoning and finish with `#### <answer>`."

GSM8K_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": GSM8K_CALCULATOR_TOOL_NAME,
        "description": "Evaluate one arithmetic expression. This tool has no access to the expected answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression using numbers, parentheses, and +, -, *, /, //, %, or **.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}
_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}
_MAX_EXPRESSION_CHARS = 256
_MAX_EXPRESSION_NODES = 64
_MAX_ABSOLUTE_VALUE = 1e100
_MAX_ABSOLUTE_EXPONENT = 12


class CalculatorExpressionError(ValueError):
    """The model supplied an expression outside the calculator contract."""


class CalculatorArithmeticError(ArithmeticError):
    """A valid calculator expression failed during arithmetic execution."""


def _validate_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalculatorExpressionError("calculator constants must be integers or finite decimal numbers")
    if not math.isfinite(value) or abs(value) > _MAX_ABSOLUTE_VALUE:
        raise CalculatorExpressionError("calculator values must be finite and bounded")
    return value


def _evaluate_expression_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_expression_node(node.body)
    if isinstance(node, ast.Constant):
        return _validate_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
        value = _evaluate_expression_node(node.operand)
        return _validate_number(_ALLOWED_UNARY_OPERATORS[type(node.op)](value))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
        left = _evaluate_expression_node(node.left)
        right = _evaluate_expression_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_ABSOLUTE_EXPONENT:
            raise CalculatorExpressionError("calculator exponent is outside the supported bound")
        try:
            result = _ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        except (ArithmeticError, OverflowError) as error:
            raise CalculatorArithmeticError(str(error) or type(error).__name__) from error
        return _validate_number(result)
    raise CalculatorExpressionError(f"calculator expression contains unsupported syntax {type(node).__name__}")


def evaluate_arithmetic_expression(expression: str) -> str:
    """Safely evaluate the arithmetic-only calculator capability."""

    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorExpressionError("calculator expression must be a non-empty string")
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise CalculatorExpressionError(f"calculator expression exceeds {_MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculatorExpressionError("calculator expression must use valid arithmetic syntax") from error
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        raise CalculatorExpressionError("calculator expression is too complex")
    result = _evaluate_expression_node(tree)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def _gold_answer(solution: str) -> str:
    answer = extract_gsm8k_answer(solution, method="strict")
    if answer is None:
        raise ValueError("GSM8K gold solution must contain a final answer in '#### <answer>' format")
    return answer


def _validate_recipe_identity(*, split: str, data_source: str) -> None:
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a non-empty string")


@dataclass(frozen=True)
class _GSM8KTaskAdapter:
    mode: Literal["direct", "calculator"]
    split: str
    data_source: str

    def __call__(self, dataset: Any) -> Iterable[EvaluationTask]:
        if not callable(getattr(dataset, "to_list", None)):
            raise TypeError("GSM8K evaluation requires a Dataset-compatible object with to_list()")
        for index, row in enumerate(dataset.to_list()):
            if not isinstance(row, Mapping):
                raise TypeError(f"GSM8K dataset row {index} must be a mapping")
            question = row.get("question", row.get("input"))
            solution = row.get("answer", row.get("output"))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"GSM8K dataset row {index} question must be a non-empty string")
            if not isinstance(solution, str):
                raise TypeError(f"GSM8K dataset row {index} solution must be a string")
            gold_answer = _gold_answer(solution)
            system_prompt = GSM8K_DIRECT_SYSTEM_PROMPT if self.mode == "direct" else GSM8K_CALCULATOR_SYSTEM_PROMPT
            tools = [] if self.mode == "direct" else [copy.deepcopy(GSM8K_CALCULATOR_TOOL)]
            task = TaskSpec(
                task_id=f"{self.data_source}:{self.split}:{index}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": GSM8K_USER_PROMPT.format(question=question)},
                ],
                tools=tools,
                environment={"evaluation_mode": self.mode},
                metadata={
                    "data_source": self.data_source,
                    "split": self.split,
                    "index": index,
                },
            )
            yield EvaluationTask(
                task=task,
                verifier_material={"gold_answer": gold_answer},
            )


class GSM8KVerifier:
    """Hidden GSM8K verifier shared by the direct and calculator recipes."""

    def verify(self, task: TaskSpec, output: ModelRunOutput, *, verifier_material: Any) -> VerificationOutcome:
        if not isinstance(verifier_material, Mapping):
            raise TypeError("GSM8K verifier material must be a mapping")
        gold_answer = verifier_material.get("gold_answer")
        if not isinstance(gold_answer, str) or not gold_answer:
            raise ValueError("GSM8K verifier material must contain a non-empty gold_answer")
        if not isinstance(output.value, Mapping):
            raise TypeError("GSM8K model output must be a mapping")
        final_response = output.value.get("final_response")
        mode = output.value.get("mode")
        attempts = output.value.get("attempts")
        if not isinstance(final_response, str):
            raise TypeError("GSM8K model output final_response must be a string")
        if mode not in {"direct", "calculator"}:
            raise ValueError("GSM8K model output mode must be direct or calculator")
        if not isinstance(attempts, list):
            raise TypeError("GSM8K model output attempts must be a list")

        final_correct = score_gsm8k_answer(final_response, gold_answer, method="flexible")
        strict_correct = score_gsm8k_answer(final_response, gold_answer, method="strict")
        attempt_scores = []
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping) or not isinstance(attempt.get("answer"), str):
                raise TypeError(f"GSM8K model output attempts[{index}] must contain a string answer")
            attempt_scores.append(score_gsm8k_answer(attempt["answer"], gold_answer, method="flexible"))
        first_attempt_success = attempt_scores[0] if attempt_scores else final_correct
        tool_calls = output.usage.tool_calls
        tool_compliance = float(tool_calls == 0 if mode == "direct" else tool_calls > 0)
        direct_answer_fallback = float(mode == "calculator" and tool_calls == 0)
        recovery = float(final_correct == 1.0 and first_attempt_success == 0.0)

        return VerificationOutcome(
            metrics={
                "final_correctness": final_correct,
                "strict_correctness": strict_correct,
                "tool_compliance": tool_compliance,
                "first_attempt_success": first_attempt_success,
                "recovery": recovery,
                "direct_answer_fallback": direct_answer_fallback,
                "tool_error": float(output.value.get("tool_errors", 0) > 0),
            },
            passed=final_correct == 1.0,
            metadata={
                "final_answer": extract_gsm8k_answer(final_response, method="flexible"),
                "strict_answer": extract_gsm8k_answer(final_response, method="strict"),
                "attempt_count": len(attempt_scores),
            },
        )


def _default_direct_budget() -> EvaluationBudget:
    return EvaluationBudget(
        max_model_calls=1,
        max_tool_calls=0,
        max_steps=1,
        max_input_tokens=4096,
        max_output_tokens=1024,
        wall_time_seconds=120,
    )


def _default_calculator_budget() -> EvaluationBudget:
    return EvaluationBudget(
        max_model_calls=4,
        max_tool_calls=4,
        max_steps=4,
        max_input_tokens=16384,
        max_output_tokens=4096,
        wall_time_seconds=240,
    )


def create_gsm8k_direct_recipe(
    *,
    split: str = "test",
    data_source: str = GSM8K_DATA_SOURCE,
    sampling: SamplingConfig | None = None,
    budget: EvaluationBudget | None = None,
) -> EvaluationRecipe:
    """Create the held-out direct-answer recipe."""

    _validate_recipe_identity(split=split, data_source=data_source)
    sampling = sampling or SamplingConfig(temperature=0.0, top_p=1.0, seed=0)
    budget = budget or _default_direct_budget()
    return EvaluationRecipe(
        name=GSM8K_DIRECT_RECIPE_NAME,
        task_adapter=_GSM8KTaskAdapter(mode="direct", split=split, data_source=data_source),
        capability_profile=CapabilityProfile(name="direct-answer"),
        sampling=sampling,
        budget=budget,
        verifier=GSM8KVerifier(),
        metadata={
            "benchmark": "GSM8K",
            "protocol": "direct-answer",
            "split": split,
            "data_source": data_source,
            "gold_visibility": "hidden_verifier_only",
        },
    )


def create_gsm8k_calculator_recipe(
    *,
    split: str = "test",
    data_source: str = GSM8K_DATA_SOURCE,
    sampling: SamplingConfig | None = None,
    budget: EvaluationBudget | None = None,
    require_tool_use: bool = False,
) -> EvaluationRecipe:
    """Create the held-out arithmetic-calculator recipe."""

    _validate_recipe_identity(split=split, data_source=data_source)
    if not isinstance(require_tool_use, bool):
        raise TypeError("require_tool_use must be a boolean")
    sampling = sampling or SamplingConfig(temperature=0.0, top_p=1.0, seed=0)
    budget = budget or _default_calculator_budget()
    return EvaluationRecipe(
        name=GSM8K_CALCULATOR_RECIPE_NAME,
        task_adapter=_GSM8KTaskAdapter(mode="calculator", split=split, data_source=data_source),
        capability_profile=CapabilityProfile(
            name="gsm8k-calculator",
            affordances=(GSM8K_CALCULATOR_CAPABILITY,),
            config={
                "tool_name": GSM8K_CALCULATOR_TOOL_NAME,
                "scope": "arithmetic_only",
                "gold_access": False,
                "tool_use": "required" if require_tool_use else "optional",
            },
        ),
        sampling=sampling,
        budget=budget,
        verifier=GSM8KVerifier(),
        metadata={
            "benchmark": "GSM8K",
            "protocol": "calculator-tool",
            "split": split,
            "data_source": data_source,
            "gold_visibility": "hidden_verifier_only",
        },
    )


class GSM8KCompletionRunner:
    """Run GSM8K recipes through the shared synchronous completion boundary."""

    def __init__(
        self,
        *,
        backend: CompletionBackend | None = None,
        model_name: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        artifact_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if backend is not None and not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide complete()")
        if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
            raise ValueError("model_name must be a non-empty string when provided")
        if model_kwargs is None:
            model_kwargs = {}
        if not isinstance(model_kwargs, Mapping) or any(not isinstance(key, str) for key in model_kwargs):
            raise TypeError("model_kwargs must be a string-keyed mapping")
        conflicting = {"temperature", "top_p", "seed", "max_tokens", "tool_choice"}.intersection(model_kwargs)
        if conflicting:
            raise ValueError(f"sampling fields are controlled by the recipe: {sorted(conflicting)}")
        self.backend = backend
        self.model_name = model_name
        self.model_kwargs = copy.deepcopy(dict(model_kwargs))
        self.artifact_dir = None if artifact_dir is None else Path(artifact_dir)

    def scaffold_provenance(self) -> dict[str, str]:
        """Return the fixed model-visible scaffold used by this runner."""

        return copy.deepcopy(GSM8K_REFERENCE_SCAFFOLD)

    def _resolve_backend(self, model: Any) -> CompletionBackend:
        backend = self.backend if self.backend is not None else model
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("GSM8K evaluation requires a CompletionBackend as runner.backend or model")
        return backend

    def _resolve_model_name(self, model: Any) -> str:
        candidates = [
            self.model_name,
            getattr(model, "model_name", None),
            getattr(model, "model_name_or_path", None),
            getattr(getattr(model, "model_args", None), "model_name_or_path", None),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        raise ValueError("model_name must be configured on GSM8KCompletionRunner or the model")

    @staticmethod
    def _mode(task: TaskSpec, capability_profile: CapabilityProfile) -> Literal["direct", "calculator"]:
        mode = task.environment.get("evaluation_mode")
        affordances = set(capability_profile.affordances)
        if mode == "direct" and not task.tools and not affordances:
            return "direct"
        if (
            mode == "calculator"
            and task.tools == [GSM8K_CALCULATOR_TOOL]
            and affordances == {GSM8K_CALCULATOR_CAPABILITY}
        ):
            return "calculator"
        raise ValueError("GSM8K task and capability profile do not match a supported recipe")

    @staticmethod
    def _calculator_requires_tool(capability_profile: CapabilityProfile) -> bool:
        if GSM8K_CALCULATOR_CAPABILITY not in capability_profile.affordances:
            return False
        tool_use = capability_profile.config.get("tool_use")
        if tool_use not in {"optional", "required"}:
            raise ValueError("calculator capability tool_use must be optional or required")
        return tool_use == "required"

    @staticmethod
    def _usage(
        calls: list[dict[str, Any]],
        *,
        started_at: float,
        tool_calls: int,
        model_calls: int | None = None,
    ) -> EvaluationUsage:
        prompt_tokens = []
        completion_tokens = []
        for call in calls:
            raw_response = call["completion"]["raw_response"]
            raw_usage = raw_response.get("usage") if isinstance(raw_response, Mapping) else None
            if isinstance(raw_usage, Mapping):
                prompt_tokens.append(raw_usage.get("prompt_tokens"))
                completion_tokens.append(raw_usage.get("completion_tokens"))
            else:
                prompt_tokens.append(None)
                completion_tokens.append(None)
        input_tokens = (
            sum(prompt_tokens) if prompt_tokens and all(isinstance(value, int) for value in prompt_tokens) else None
        )
        output_tokens = (
            sum(completion_tokens)
            if completion_tokens and all(isinstance(value, int) for value in completion_tokens)
            else None
        )
        return EvaluationUsage(
            model_calls=len(calls) if model_calls is None else model_calls,
            tool_calls=tool_calls,
            steps=len(calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_time_seconds=time.monotonic() - started_at,
            cost=sum(call["completion"]["cost"] for call in calls),
        )

    @staticmethod
    def _reported_budget_error(usage: EvaluationUsage, budget: EvaluationBudget) -> str | None:
        checks = (
            ("model calls", usage.model_calls, budget.max_model_calls),
            ("tool calls", usage.tool_calls, budget.max_tool_calls),
            ("steps", usage.steps, budget.max_steps),
            ("input tokens", usage.input_tokens, budget.max_input_tokens),
            ("output tokens", usage.output_tokens, budget.max_output_tokens),
        )
        for label, actual, maximum in checks:
            if actual is not None and actual > maximum:
                return f"reported {label} {actual} exceed budget {maximum}"
        return None

    def _publish_artifact(self, task: TaskSpec, payload: Mapping[str, Any], *, suffix: str = "") -> str | None:
        if self.artifact_dir is None:
            return None
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:24]
        target = self.artifact_dir / f"{digest}{suffix}.json"
        if target.exists():
            raise FileExistsError(f"GSM8K evaluation artifact already exists: {target}")
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=self.artifact_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output_file:
                json.dump(payload, output_file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                output_file.write("\n")
                output_file.flush()
                os.fsync(output_file.fileno())
            os.link(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        return str(target)

    def _failure_artifact(
        self,
        task: TaskSpec,
        *,
        mode: str,
        capability_profile: CapabilityProfile,
        sampling: SamplingConfig,
        budget: EvaluationBudget,
        calls: list[dict[str, Any]],
        usage: EvaluationUsage,
        failure_type: EvaluationFailureType,
        message: str,
    ) -> str | None:
        return self._publish_artifact(
            task,
            {
                "task_id": task.task_id,
                "task": {
                    "messages": task.messages,
                    "tools": task.tools,
                    "environment": task.environment,
                    "metadata": task.metadata,
                },
                "mode": mode,
                "capability_profile": asdict(capability_profile),
                "scaffold": self.scaffold_provenance(),
                "sampling": asdict(sampling),
                "budget": asdict(budget),
                "calls": calls,
                "failure": {
                    "failure_type": failure_type.value,
                    "message": message,
                },
                "usage": asdict(usage),
            },
            suffix=".failed",
        )

    @staticmethod
    def _add_attempt(attempts: list[dict[str, str]], *, source: str, answer: str) -> None:
        if extract_gsm8k_answer(answer, method="flexible") is not None:
            attempts.append({"source": source, "answer": answer})

    def run(
        self,
        model: Any,
        task: TaskSpec,
        *,
        capability_profile: CapabilityProfile,
        sampling: SamplingConfig,
        budget: EvaluationBudget,
    ) -> ModelRunOutput:
        backend = self._resolve_backend(model)
        model_name = self._resolve_model_name(model)
        mode = self._mode(task, capability_profile)
        require_tool_use = mode == "calculator" and self._calculator_requires_tool(capability_profile)
        history = copy.deepcopy(task.messages)
        calls: list[dict[str, Any]] = []
        attempts: list[dict[str, str]] = []
        tool_call_count = 0
        tool_error_count = 0
        seen_call_ids: set[str] = set()
        model_call_count = 0
        started_at = time.monotonic()
        try:
            for _ in range(budget.max_model_calls):
                if len(calls) >= budget.max_steps:
                    raise EvaluationSampleError(
                        EvaluationFailureType.BUDGET_EXHAUSTED,
                        "GSM8K episode exhausted the step budget before a final answer",
                        usage=self._usage(
                            calls,
                            started_at=started_at,
                            tool_calls=tool_call_count,
                            model_calls=model_call_count,
                        ),
                    )
                if time.monotonic() - started_at > budget.wall_time_seconds:
                    raise EvaluationSampleError(
                        EvaluationFailureType.TIMEOUT,
                        "GSM8K episode exceeded the wall-time budget before a model call",
                        retryable=True,
                        usage=self._usage(
                            calls,
                            started_at=started_at,
                            tool_calls=tool_call_count,
                            model_calls=model_call_count,
                        ),
                    )
                prior_usage = self._usage(
                    calls,
                    started_at=started_at,
                    tool_calls=tool_call_count,
                    model_calls=model_call_count,
                )
                remaining_output_tokens = (
                    budget.max_output_tokens
                    if prior_usage.output_tokens is None
                    else budget.max_output_tokens - prior_usage.output_tokens
                )
                if remaining_output_tokens <= 0:
                    raise EvaluationSampleError(
                        EvaluationFailureType.BUDGET_EXHAUSTED,
                        "GSM8K episode exhausted the output-token budget before a final answer",
                        usage=prior_usage,
                    )
                request_kwargs = {
                    **copy.deepcopy(self.model_kwargs),
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "seed": sampling.seed,
                    "max_tokens": min(
                        remaining_output_tokens,
                        math.ceil(budget.max_output_tokens / budget.max_model_calls),
                    ),
                }
                if require_tool_use and tool_call_count == 0:
                    request_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": GSM8K_CALCULATOR_TOOL_NAME},
                    }
                model_call_count += 1
                response = backend.complete(
                    messages=copy.deepcopy(history),
                    tools=copy.deepcopy(task.tools),
                    model_name=model_name,
                    model_kwargs=copy.deepcopy(request_kwargs),
                )
                completion = normalize_completion_response(response)
                calls.append({"completion": completion})
                self._add_attempt(attempts, source="assistant", answer=completion["content"])
                raw_tool_calls = completion["tool_calls"]
                current_usage = self._usage(
                    calls,
                    started_at=started_at,
                    tool_calls=tool_call_count,
                    model_calls=model_call_count,
                )
                if current_usage.wall_time_seconds > budget.wall_time_seconds:
                    raise EvaluationSampleError(
                        EvaluationFailureType.TIMEOUT,
                        "GSM8K episode exceeded the wall-time budget during a model call",
                        retryable=True,
                        usage=current_usage,
                    )
                budget_error = self._reported_budget_error(current_usage, budget)
                if budget_error is not None:
                    raise EvaluationSampleError(
                        EvaluationFailureType.BUDGET_EXHAUSTED,
                        budget_error,
                        usage=current_usage,
                    )

                if mode == "direct" and raw_tool_calls:
                    tool_call_count += len(raw_tool_calls)
                    raise EvaluationSampleError(
                        EvaluationFailureType.INVALID_TOOL_CALL,
                        "direct-answer recipe received an unexpected tool call",
                        usage=self._usage(
                            calls,
                            started_at=started_at,
                            tool_calls=tool_call_count,
                            model_calls=model_call_count,
                        ),
                    )
                if not raw_tool_calls:
                    usage = self._usage(
                        calls,
                        started_at=started_at,
                        tool_calls=tool_call_count,
                        model_calls=model_call_count,
                    )
                    value = {
                        "mode": mode,
                        "final_response": completion["content"],
                        "attempts": attempts,
                        "tool_errors": tool_error_count,
                    }
                    artifact_ref = self._publish_artifact(
                        task,
                        {
                            "task_id": task.task_id,
                            "task": {
                                "messages": task.messages,
                                "tools": task.tools,
                                "environment": task.environment,
                                "metadata": task.metadata,
                            },
                            "mode": mode,
                            "capability_profile": asdict(capability_profile),
                            "scaffold": self.scaffold_provenance(),
                            "sampling": asdict(sampling),
                            "budget": asdict(budget),
                            "calls": calls,
                            "output": value,
                            "usage": asdict(usage),
                        },
                    )
                    return ModelRunOutput(
                        value=value,
                        usage=usage,
                        artifact_ref=artifact_ref,
                        metadata={
                            "mode": mode,
                            "finish_reason": completion["finish_reason"],
                            "tool_errors": tool_error_count,
                        },
                    )

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": completion["content"],
                    "tool_calls": copy.deepcopy(raw_tool_calls),
                }
                if completion["reasoning_content"] is not None:
                    assistant_message["reasoning_content"] = completion["reasoning_content"]
                history.append(assistant_message)

                for call_index, raw_call in enumerate(raw_tool_calls):
                    path = f"completion tool_calls[{call_index}]"
                    if tool_call_count >= budget.max_tool_calls:
                        raise EvaluationSampleError(
                            EvaluationFailureType.BUDGET_EXHAUSTED,
                            "GSM8K episode exceeded the tool-call budget",
                            usage=self._usage(
                                calls,
                                started_at=started_at,
                                tool_calls=tool_call_count,
                                model_calls=model_call_count,
                            ),
                        )
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
                            function.get("arguments"), path=f"{path}.function.arguments"
                        )
                        if set(arguments) != {"expression"} or not isinstance(arguments["expression"], str):
                            raise ValueError(f"{path}.function.arguments must contain only string field 'expression'")
                        try:
                            result = evaluate_arithmetic_expression(arguments["expression"])
                            observation = f"Calculator result: {result}"
                        except CalculatorArithmeticError as error:
                            tool_error_count += 1
                            observation = f"Calculator error: {error}"
                    except (CalculatorExpressionError, TypeError, ValueError) as error:
                        raise EvaluationSampleError(
                            EvaluationFailureType.INVALID_TOOL_CALL,
                            str(error),
                            usage=self._usage(
                                calls,
                                started_at=started_at,
                                tool_calls=tool_call_count,
                                model_calls=model_call_count,
                            ),
                        ) from error
                    seen_call_ids.add(call_id)
                    calls[-1].setdefault("tool_results", []).append(
                        {
                            "tool_call_id": call_id,
                            "expression": arguments["expression"],
                            "observation": observation,
                        }
                    )
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": GSM8K_CALCULATOR_TOOL_NAME,
                            "content": observation,
                        }
                    )
        except EvaluationSampleError as error:
            usage = error.usage or self._usage(
                calls,
                started_at=started_at,
                tool_calls=tool_call_count,
                model_calls=model_call_count,
            )
            artifact_ref = error.artifact_ref or self._failure_artifact(
                task,
                mode=mode,
                capability_profile=capability_profile,
                sampling=sampling,
                budget=budget,
                calls=calls,
                usage=usage,
                failure_type=error.failure_type,
                message=str(error),
            )
            raise EvaluationSampleError(
                error.failure_type,
                str(error),
                retryable=error.retryable,
                artifact_ref=artifact_ref,
                usage=usage,
            ) from error
        except TimeoutError as error:
            usage = self._usage(
                calls,
                started_at=started_at,
                tool_calls=tool_call_count,
                model_calls=model_call_count,
            )
            message = str(error) or "GSM8K completion timed out"
            artifact_ref = self._failure_artifact(
                task,
                mode=mode,
                capability_profile=capability_profile,
                sampling=sampling,
                budget=budget,
                calls=calls,
                usage=usage,
                failure_type=EvaluationFailureType.TIMEOUT,
                message=message,
            )
            raise EvaluationSampleError(
                EvaluationFailureType.TIMEOUT,
                message,
                retryable=True,
                artifact_ref=artifact_ref,
                usage=usage,
            ) from error
        except Exception as error:
            usage = self._usage(
                calls,
                started_at=started_at,
                tool_calls=tool_call_count,
                model_calls=model_call_count,
            )
            message = str(error) or type(error).__name__
            artifact_ref = self._failure_artifact(
                task,
                mode=mode,
                capability_profile=capability_profile,
                sampling=sampling,
                budget=budget,
                calls=calls,
                usage=usage,
                failure_type=EvaluationFailureType.BACKEND_FAILURE,
                message=message,
            )
            raise EvaluationSampleError(
                EvaluationFailureType.BACKEND_FAILURE,
                message,
                retryable=True,
                artifact_ref=artifact_ref,
                usage=usage,
            ) from error

        usage = self._usage(
            calls,
            started_at=started_at,
            tool_calls=tool_call_count,
            model_calls=model_call_count,
        )
        message = f"GSM8K episode reached max_model_calls={budget.max_model_calls} without a final answer"
        artifact_ref = self._failure_artifact(
            task,
            mode=mode,
            capability_profile=capability_profile,
            sampling=sampling,
            budget=budget,
            calls=calls,
            usage=usage,
            failure_type=EvaluationFailureType.BUDGET_EXHAUSTED,
            message=message,
        )
        raise EvaluationSampleError(
            EvaluationFailureType.BUDGET_EXHAUSTED,
            message,
            artifact_ref=artifact_ref,
            usage=usage,
        )


__all__ = [
    "CalculatorArithmeticError",
    "CalculatorExpressionError",
    "GSM8K_CALCULATOR_CAPABILITY",
    "GSM8K_CALCULATOR_RECIPE_NAME",
    "GSM8K_CALCULATOR_TOOL",
    "GSM8K_CALCULATOR_TOOL_NAME",
    "GSM8K_DIRECT_RECIPE_NAME",
    "GSM8K_REFERENCE_SCAFFOLD",
    "GSM8KCompletionRunner",
    "GSM8KVerifier",
    "create_gsm8k_calculator_recipe",
    "create_gsm8k_direct_recipe",
    "evaluate_arithmetic_expression",
]
