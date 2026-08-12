"""Configuration and extension contracts for the Evaluator pipeline.

The contracts stop at evaluation orchestration. Model- and environment-
specific episode semantics belong to a ``ModelRunner`` implementation, which
can delegate to a shared Agentic episode executor.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from lmflow.pipeline.evaluation.result import EvaluationUsage, _json_mapping

if TYPE_CHECKING:
    from lmflow.agentic.contracts import TaskSpec


def _implementation_name(value: Any) -> str:
    target = value if isinstance(value, type) else type(value)
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        target = value
    return f"{target.__module__}.{target.__qualname__}"


def _validate_non_negative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CapabilityProfile:
    """External affordances made available under one evaluation condition."""

    name: str
    affordances: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("capability profile name must be a non-empty string")
        affordances = tuple(self.affordances)
        if any(not isinstance(value, str) or not value.strip() for value in affordances):
            raise ValueError("capability profile affordances must be non-empty strings")
        if len(affordances) != len(set(affordances)):
            raise ValueError("capability profile affordances must be unique")
        object.__setattr__(self, "affordances", affordances)
        object.__setattr__(self, "config", _json_mapping(self.config, name="capability profile config"))


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
    """Per-sample execution limits enforced by an evaluation runner."""

    max_model_calls: int
    max_tool_calls: int
    max_steps: int
    max_input_tokens: int
    max_output_tokens: int
    wall_time_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "max_model_calls",
            "max_tool_calls",
            "max_steps",
            "max_input_tokens",
            "max_output_tokens",
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


class ModelRunner(Protocol):
    """Run one model-visible task using existing episode execution semantics."""

    def run(
        self,
        model: Any,
        task: TaskSpec,
        *,
        capability_profile: CapabilityProfile,
        sampling: SamplingConfig,
        budget: EvaluationBudget,
    ) -> ModelRunOutput: ...

    def scaffold_provenance(self) -> Mapping[str, Any]: ...


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
    capability_profile: CapabilityProfile
    sampling: SamplingConfig
    budget: EvaluationBudget
    verifier: Verifier
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("recipe name must be a non-empty string")
        if not callable(self.task_adapter):
            raise TypeError("task_adapter must be callable")
        if not isinstance(self.capability_profile, CapabilityProfile):
            raise TypeError("recipe capability_profile must be a CapabilityProfile")
        if not isinstance(self.sampling, SamplingConfig):
            raise TypeError("recipe sampling must be SamplingConfig")
        if not isinstance(self.budget, EvaluationBudget):
            raise TypeError("recipe budget must be EvaluationBudget")
        if not callable(getattr(self.verifier, "verify", None)):
            raise TypeError("recipe verifier must provide verify()")
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


__all__ = [
    "CapabilityProfile",
    "EvaluationBudget",
    "EvaluationRecipe",
    "EvaluationTask",
    "LegacyEvaluationRecipe",
    "ModelRunOutput",
    "ModelRunner",
    "SamplingConfig",
    "VerificationOutcome",
    "Verifier",
]
