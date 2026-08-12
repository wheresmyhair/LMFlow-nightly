"""Internal orchestration for recipe-driven evaluation."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from lmflow.pipeline.evaluation.recipe import (
    EvaluationBudget,
    EvaluationRecipe,
    EvaluationTask,
    ModelRunner,
    ModelRunOutput,
    VerificationOutcome,
)
from lmflow.pipeline.evaluation.result import (
    EvaluationFailure,
    EvaluationFailureType,
    EvaluationProvenance,
    EvaluationRecord,
    EvaluationResult,
    EvaluationSampleError,
    EvaluationUsage,
)
from lmflow.pipeline.evaluation.runtime import EvaluationRuntime


def _implementation_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        target = value
    return f"{target.__module__}.{target.__qualname__}"


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
            capability_profile=recipe.capability_profile,
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


def _validated_tasks(items: Iterable[EvaluationTask]) -> Iterable[EvaluationTask]:
    seen_task_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, EvaluationTask):
            raise TypeError(f"task_adapter item {index} must be EvaluationTask")
        task_id = item.task.task_id
        if task_id in seen_task_ids:
            raise ValueError(f"task_adapter returned duplicate task_id {task_id!r}")
        seen_task_ids.add(task_id)
        yield item


def evaluate_recipe(
    recipe: EvaluationRecipe,
    runner: ModelRunner,
    runtime: EvaluationRuntime,
    model: Any,
    dataset: Any,
    *,
    model_provenance: Mapping[str, Any],
    dataset_provenance: Mapping[str, Any],
) -> EvaluationResult:
    """Evaluate a dataset through a caller-selected execution runtime."""

    if not callable(getattr(runner, "run", None)):
        raise TypeError("runner must provide run()")
    if not callable(getattr(runner, "scaffold_provenance", None)):
        raise TypeError("runner must provide scaffold_provenance()")
    if not callable(getattr(runtime, "map", None)):
        raise TypeError("runtime must provide map()")
    items = _validated_tasks(recipe.task_adapter(dataset))
    records = runtime.map(
        lambda item: _evaluate_task(recipe, runner, model, item),
        items,
    )
    if not isinstance(records, tuple) or any(not isinstance(record, EvaluationRecord) for record in records):
        raise TypeError("evaluation runtime must return a tuple of EvaluationRecord values")

    provenance = EvaluationProvenance(
        recipe=recipe.provenance(),
        capability_profile=asdict(recipe.capability_profile),
        scaffold=runner.scaffold_provenance(),
        model=model_provenance,
        dataset=dataset_provenance,
        execution={
            "runner": _implementation_name(runner),
            "runtime": _implementation_name(runtime),
            "runtime_config": runtime.provenance() if callable(getattr(runtime, "provenance", None)) else {},
        },
    )
    return EvaluationResult(
        summary=_summarize(records),
        records=records,
        provenance=provenance,
    )


__all__ = ["evaluate_recipe"]
