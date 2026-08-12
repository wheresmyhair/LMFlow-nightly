"""Core contracts for recipe-driven evaluation.

The contracts in this module deliberately stop at evaluation orchestration.
Model- and environment-specific episode semantics belong to a ``ModelRunner``
implementation, which can delegate to a shared Agentic episode executor.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from lmflow.agentic.contracts import TaskSpec


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def _implementation_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        target = value
    return f"{target.__module__}.{target.__qualname__}"


def _json_provenance(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    return _json_mapping(
        {key: field_value for key, field_value in value.items() if field_value is not None},
        name=name,
    )


def _validate_non_negative_int(value: int | None, *, name: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class AgentCapability:
    """One external affordance made available to the evaluated policy."""

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("capability name must be a non-empty string")
        object.__setattr__(self, "config", _json_mapping(self.config, name="capability config"))


@dataclass(frozen=True)
class SamplingConfig:
    """Provider-neutral sampling parameters that affect model behavior."""

    temperature: float
    top_p: float
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, int | float):
            raise TypeError("temperature must be a number")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, int | float):
            raise TypeError("top_p must be a number")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class EvaluationBudget:
    """Per-sample execution limits plus the run-level concurrency bound."""

    max_model_calls: int
    max_tool_calls: int
    max_steps: int
    max_input_tokens: int
    max_output_tokens: int
    wall_time_seconds: float
    max_concurrency: int

    def __post_init__(self) -> None:
        for name in (
            "max_model_calls",
            "max_tool_calls",
            "max_steps",
            "max_input_tokens",
            "max_output_tokens",
            "max_concurrency",
        ):
            value = getattr(self, name)
            _validate_non_negative_int(value, name=name)
            if name != "max_tool_calls" and value == 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.wall_time_seconds, bool) or not isinstance(self.wall_time_seconds, int | float):
            raise TypeError("wall_time_seconds must be a number")
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be finite and positive")


@dataclass(frozen=True)
class EvaluationTask:
    """A model-visible task paired with verifier-only material.

    Only ``task`` is passed to the model runner. ``verifier_material`` is sent
    directly to the hidden verifier after the episode completes.
    """

    task: TaskSpec
    verifier_material: Any = None

    def __post_init__(self) -> None:
        from lmflow.agentic.contracts import TaskSpec

        if not isinstance(self.task, TaskSpec):
            raise TypeError("evaluation task.task must be a TaskSpec")
        if not isinstance(self.task.task_id, str) or not self.task.task_id.strip():
            raise ValueError("evaluation task id must be a non-empty string")


@dataclass(frozen=True)
class EvaluationUsage:
    """Measured episode usage; token and cost fields may be unavailable."""

    model_calls: int
    tool_calls: int
    steps: int
    wall_time_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None

    def __post_init__(self) -> None:
        for name in ("model_calls", "tool_calls", "steps"):
            _validate_non_negative_int(getattr(self, name), name=name)
        for name in ("input_tokens", "output_tokens"):
            _validate_non_negative_int(getattr(self, name), name=name, allow_none=True)
        if isinstance(self.wall_time_seconds, bool) or not isinstance(self.wall_time_seconds, int | float):
            raise TypeError("wall_time_seconds must be a number")
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be finite and non-negative")
        if self.cost is not None:
            if isinstance(self.cost, bool) or not isinstance(self.cost, int | float):
                raise TypeError("cost must be a number when provided")
            if not math.isfinite(self.cost) or self.cost < 0:
                raise ValueError("cost must be finite and non-negative")


@dataclass(frozen=True)
class ModelRunOutput:
    """Narrow handoff from an episode runner to a hidden verifier."""

    value: Any
    usage: EvaluationUsage
    artifact_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.usage, EvaluationUsage):
            raise TypeError("model run usage must be EvaluationUsage")
        if self.artifact_ref is not None and (not isinstance(self.artifact_ref, str) or not self.artifact_ref.strip()):
            raise ValueError("artifact_ref must be a non-empty string when provided")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, name="model run metadata"))


@dataclass(frozen=True)
class VerificationOutcome:
    """Metrics produced by a hidden verifier for one completed episode."""

    metrics: Mapping[str, int | float]
    passed: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("verification metrics must be a non-empty mapping")
        normalized_metrics: dict[str, int | float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("verification metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"verification metric {name!r} must be a finite number")
            normalized_metrics[name] = value
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean when provided")
        object.__setattr__(self, "metrics", normalized_metrics)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, name="verification metadata"))


class EvaluationFailureType(str, Enum):
    """Stable top-level failure categories for evaluation records."""

    TIMEOUT = "timeout"
    INVALID_TOOL_CALL = "invalid_tool_call"
    BACKEND_FAILURE = "backend_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    VERIFIER_FAILURE = "verifier_failure"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class EvaluationFailure:
    failure_type: EvaluationFailureType
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.failure_type, EvaluationFailureType):
            raise TypeError("failure_type must be an EvaluationFailureType")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("failure message must be a non-empty string")
        if not isinstance(self.retryable, bool):
            raise TypeError("failure retryable must be a boolean")


class EvaluationSampleError(RuntimeError):
    """Typed per-sample failure raised by a model runner."""

    def __init__(
        self,
        failure_type: EvaluationFailureType,
        message: str,
        *,
        retryable: bool = False,
        artifact_ref: str | None = None,
        usage: EvaluationUsage | None = None,
    ) -> None:
        super().__init__(message)
        if not isinstance(failure_type, EvaluationFailureType):
            raise TypeError("failure_type must be an EvaluationFailureType")
        self.failure_type = failure_type
        self.retryable = retryable
        self.artifact_ref = artifact_ref
        self.usage = usage


@dataclass(frozen=True)
class EvaluationRecord:
    task_id: str
    status: str
    passed: bool | None = None
    metrics: Mapping[str, int | float] = field(default_factory=dict)
    usage: EvaluationUsage | None = None
    artifact_ref: str | None = None
    failure: EvaluationFailure | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("record task_id must be a non-empty string")
        if self.status not in {"completed", "failed"}:
            raise ValueError("record status must be 'completed' or 'failed'")
        if (self.status == "failed") != (self.failure is not None):
            raise ValueError("failed records must contain a failure and completed records must not")
        if self.status == "failed" and self.passed is not None:
            raise ValueError("failed records cannot contain a verifier pass result")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise TypeError("record passed must be a boolean when provided")
        if self.usage is not None and not isinstance(self.usage, EvaluationUsage):
            raise TypeError("record usage must be EvaluationUsage when provided")
        if self.artifact_ref is not None and (
            not isinstance(self.artifact_ref, str) or not self.artifact_ref.strip()
        ):
            raise ValueError("record artifact_ref must be a non-empty string when provided")
        normalized_metrics: dict[str, int | float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("record metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"record metric {name!r} must be a finite number")
            normalized_metrics[name] = value
        object.__setattr__(self, "metrics", normalized_metrics)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, name="evaluation record metadata"))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.failure is not None:
            value["failure"]["failure_type"] = self.failure.failure_type.value
        return value


@dataclass(frozen=True)
class EvaluationProvenance:
    recipe: Mapping[str, Any]
    capabilities: tuple[Mapping[str, Any], ...]
    model: Mapping[str, Any]
    dataset: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe", _json_provenance(self.recipe, name="recipe provenance"))
        object.__setattr__(
            self,
            "capabilities",
            tuple(_json_provenance(value, name="capability provenance") for value in self.capabilities),
        )
        object.__setattr__(self, "model", _json_provenance(self.model, name="model provenance"))
        object.__setattr__(self, "dataset", _json_provenance(self.dataset, name="dataset provenance"))


@dataclass(frozen=True)
class EvaluationResult:
    """Structured summary, per-sample records, and reproducibility data."""

    summary: Mapping[str, Any]
    records: tuple[EvaluationRecord, ...]
    provenance: EvaluationProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _json_mapping(self.summary, name="evaluation summary"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": copy.deepcopy(dict(self.summary)),
            "records": [record.to_dict() for record in self.records],
            "provenance": asdict(self.provenance),
        }


class ModelRunner(Protocol):
    """Run one model-visible task using existing episode execution semantics."""

    def run(
        self,
        model: Any,
        task: TaskSpec,
        *,
        capabilities: tuple[AgentCapability, ...],
        sampling: SamplingConfig,
        budget: EvaluationBudget,
    ) -> ModelRunOutput: ...


class Verifier(Protocol):
    """Score model output with material that is hidden from the runner."""

    def verify(
        self,
        task: TaskSpec,
        output: ModelRunOutput,
        *,
        verifier_material: Any,
    ) -> VerificationOutcome: ...


@dataclass(frozen=True)
class EvaluationRecipe:
    """Dataset projection, affordances, execution limits, and verifier."""

    name: str
    task_adapter: Callable[[Any], Iterable[EvaluationTask]]
    capabilities: tuple[AgentCapability, ...]
    sampling: SamplingConfig
    budget: EvaluationBudget
    verifier: Verifier
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("recipe name must be a non-empty string")
        if not callable(self.task_adapter):
            raise TypeError("task_adapter must be callable")
        capabilities = tuple(self.capabilities)
        if any(not isinstance(capability, AgentCapability) for capability in capabilities):
            raise TypeError("recipe capabilities must contain AgentCapability values")
        names = [capability.name for capability in capabilities]
        if len(names) != len(set(names)):
            raise ValueError("recipe capability names must be unique")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("recipe sampling must be SamplingConfig")
        if not isinstance(self.budget, EvaluationBudget):
            raise TypeError("recipe budget must be EvaluationBudget")
        if not callable(getattr(self.verifier, "verify", None)):
            raise TypeError("recipe verifier must provide verify()")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, name="recipe metadata"))

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_adapter": _implementation_name(self.task_adapter),
            "verifier": _implementation_name(self.verifier),
            "sampling": asdict(self.sampling),
            "budget": asdict(self.budget),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class LegacyEvaluationRecipe:
    """Compatibility descriptor for the existing scalar metric path."""

    metric: str

    def __post_init__(self) -> None:
        if self.metric not in {
            "acc",
            "accuracy",
            "ppl",
            "perplexity",
            "nll",
            "neg_log_likelihood",
        }:
            raise NotImplementedError(f"metric {self.metric} is not supported")


def _budget_failure(usage: EvaluationUsage, budget: EvaluationBudget, elapsed: float) -> str | None:
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
    observed_wall_time = max(float(usage.wall_time_seconds), elapsed)
    if observed_wall_time > budget.wall_time_seconds:
        return f"observed wall time {observed_wall_time:.6f}s exceeds budget {budget.wall_time_seconds:.6f}s"
    return None


def _failure_record(
    task_id: str,
    failure_type: EvaluationFailureType,
    error: BaseException,
    *,
    retryable: bool = False,
    artifact_ref: str | None = None,
    usage: EvaluationUsage | None = None,
) -> EvaluationRecord:
    message = str(error) or type(error).__name__
    return EvaluationRecord(
        task_id=task_id,
        status="failed",
        usage=usage,
        artifact_ref=artifact_ref,
        failure=EvaluationFailure(
            failure_type=failure_type,
            message=message,
            retryable=retryable,
        ),
    )


def _evaluate_task(recipe: EvaluationRecipe, runner: ModelRunner, model: Any, item: EvaluationTask) -> EvaluationRecord:
    started_at = time.monotonic()
    try:
        output = runner.run(
            model,
            item.task,
            capabilities=recipe.capabilities,
            sampling=recipe.sampling,
            budget=recipe.budget,
        )
    except EvaluationSampleError as error:
        return _failure_record(
            item.task.task_id,
            error.failure_type,
            error,
            retryable=error.retryable,
            artifact_ref=error.artifact_ref,
            usage=error.usage,
        )
    except TimeoutError as error:
        return _failure_record(item.task.task_id, EvaluationFailureType.TIMEOUT, error, retryable=True)
    except Exception as error:
        return _failure_record(item.task.task_id, EvaluationFailureType.BACKEND_FAILURE, error, retryable=True)

    if not isinstance(output, ModelRunOutput):
        error = TypeError("model runner must return ModelRunOutput")
        return _failure_record(item.task.task_id, EvaluationFailureType.INTERNAL_ERROR, error)

    elapsed = time.monotonic() - started_at
    budget_failure = _budget_failure(output.usage, recipe.budget, elapsed)
    if budget_failure is not None:
        return _failure_record(
            item.task.task_id,
            EvaluationFailureType.BUDGET_EXHAUSTED,
            RuntimeError(budget_failure),
            artifact_ref=output.artifact_ref,
            usage=output.usage,
        )

    try:
        verified = recipe.verifier.verify(
            item.task,
            output,
            verifier_material=item.verifier_material,
        )
    except Exception as error:
        return _failure_record(
            item.task.task_id,
            EvaluationFailureType.VERIFIER_FAILURE,
            error,
            artifact_ref=output.artifact_ref,
            usage=output.usage,
        )
    if not isinstance(verified, VerificationOutcome):
        error = TypeError("verifier must return VerificationOutcome")
        return _failure_record(
            item.task.task_id,
            EvaluationFailureType.VERIFIER_FAILURE,
            error,
            artifact_ref=output.artifact_ref,
            usage=output.usage,
        )

    return EvaluationRecord(
        task_id=item.task.task_id,
        status="completed",
        passed=verified.passed,
        metrics=verified.metrics,
        usage=output.usage,
        artifact_ref=output.artifact_ref,
        metadata={
            "runner": output.metadata,
            "verifier": verified.metadata,
        },
    )


def _summarize(records: tuple[EvaluationRecord, ...]) -> dict[str, Any]:
    completed = [record for record in records if record.status == "completed"]
    failures = [record for record in records if record.failure is not None]
    metric_values: dict[str, list[int | float]] = {}
    for record in completed:
        for name, value in record.metrics.items():
            metric_values.setdefault(name, []).append(value)

    metric_means = {name: sum(values) / len(values) for name, values in sorted(metric_values.items())}
    metric_counts = {name: len(values) for name, values in sorted(metric_values.items())}
    failure_counts = Counter(record.failure.failure_type.value for record in failures if record.failure is not None)
    passed = sum(record.passed is True for record in completed)
    scored = sum(record.passed is not None for record in completed)
    usage_records = [record.usage for record in records if record.usage is not None]
    usage_totals: dict[str, int | float] = {
        "model_calls": sum(usage.model_calls for usage in usage_records),
        "tool_calls": sum(usage.tool_calls for usage in usage_records),
        "steps": sum(usage.steps for usage in usage_records),
        "wall_time_seconds": sum(usage.wall_time_seconds for usage in usage_records),
    }
    for name in ("input_tokens", "output_tokens", "cost"):
        values = [getattr(usage, name) for usage in usage_records if getattr(usage, name) is not None]
        if values:
            usage_totals[name] = sum(values)

    return {
        "total_samples": len(records),
        "completed_samples": len(completed),
        "failed_samples": len(failures),
        "failure_rate": len(failures) / len(records) if records else 0.0,
        "passed_samples": passed,
        "scored_samples": scored,
        "pass_rate": passed / len(records) if records else 0.0,
        "completed_pass_rate": passed / scored if scored else None,
        "metrics": metric_means,
        "metric_counts": metric_counts,
        "failures": dict(sorted(failure_counts.items())),
        "usage": usage_totals,
    }


def evaluate_recipe(
    recipe: EvaluationRecipe,
    runner: ModelRunner,
    model: Any,
    dataset: Any,
    *,
    model_provenance: Mapping[str, Any],
    dataset_provenance: Mapping[str, Any],
) -> EvaluationResult:
    """Evaluate a dataset with bounded concurrency and deterministic records."""

    if not callable(getattr(runner, "run", None)):
        raise TypeError("runner must provide run()")
    items = list(recipe.task_adapter(dataset))
    seen_task_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, EvaluationTask):
            raise TypeError(f"task_adapter item {index} must be EvaluationTask")
        task_id = item.task.task_id
        if task_id in seen_task_ids:
            raise ValueError(f"task_adapter returned duplicate task_id {task_id!r}")
        seen_task_ids.add(task_id)

    if recipe.budget.max_concurrency == 1:
        records = tuple(_evaluate_task(recipe, runner, model, item) for item in items)
    else:
        with ThreadPoolExecutor(
            max_workers=recipe.budget.max_concurrency,
            thread_name_prefix="lmflow-evaluator",
        ) as executor:
            futures = [executor.submit(_evaluate_task, recipe, runner, model, item) for item in items]
            records = tuple(future.result() for future in futures)

    provenance = EvaluationProvenance(
        recipe=recipe.provenance(),
        capabilities=tuple(asdict(capability) for capability in recipe.capabilities),
        model={**dict(model_provenance), "runner": _implementation_name(runner)},
        dataset=dataset_provenance,
    )
    return EvaluationResult(
        summary=_summarize(records),
        records=records,
        provenance=provenance,
    )


__all__ = [
    "AgentCapability",
    "EvaluationBudget",
    "EvaluationFailure",
    "EvaluationFailureType",
    "LegacyEvaluationRecipe",
    "EvaluationProvenance",
    "EvaluationRecipe",
    "EvaluationRecord",
    "EvaluationResult",
    "EvaluationSampleError",
    "EvaluationTask",
    "EvaluationUsage",
    "ModelRunOutput",
    "ModelRunner",
    "SamplingConfig",
    "VerificationOutcome",
    "Verifier",
    "evaluate_recipe",
]
