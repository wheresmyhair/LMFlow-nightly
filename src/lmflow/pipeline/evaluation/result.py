"""Structured records returned by the Evaluator pipeline."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


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
        if self.artifact_ref is not None and (not isinstance(self.artifact_ref, str) or not self.artifact_ref.strip()):
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
    capability_profile: Mapping[str, Any]
    scaffold: Mapping[str, Any]
    model: Mapping[str, Any]
    dataset: Mapping[str, Any]
    execution: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recipe", _json_provenance(self.recipe, name="recipe provenance"))
        object.__setattr__(
            self,
            "capability_profile",
            _json_provenance(self.capability_profile, name="capability profile provenance"),
        )
        object.__setattr__(self, "scaffold", _json_provenance(self.scaffold, name="scaffold provenance"))
        object.__setattr__(self, "model", _json_provenance(self.model, name="model provenance"))
        object.__setattr__(self, "dataset", _json_provenance(self.dataset, name="dataset provenance"))
        object.__setattr__(self, "execution", _json_provenance(self.execution, name="execution provenance"))


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


__all__ = [
    "EvaluationFailure",
    "EvaluationFailureType",
    "EvaluationProvenance",
    "EvaluationRecord",
    "EvaluationResult",
    "EvaluationSampleError",
    "EvaluationUsage",
]
