"""Offline contract tests for recipe-driven evaluation."""

from __future__ import annotations

import threading
import time

import pytest

from lmflow.agentic.contracts import TaskSpec
from lmflow.args import DatasetArguments, EvaluatorArguments, ModelArguments
from lmflow.datasets import Dataset
from lmflow.pipeline.evaluation import Evaluator
from lmflow.pipeline.evaluation.recipe import (
    CapabilityProfile,
    EvaluationBudget,
    EvaluationRecipe,
    EvaluationTask,
    ModelRunOutput,
    SamplingConfig,
    VerificationOutcome,
)
from lmflow.pipeline.evaluation.result import EvaluationFailureType, EvaluationSampleError, EvaluationUsage
from lmflow.pipeline.evaluation.runtime import LocalEvaluationRuntime


def _dataset() -> Dataset:
    return Dataset.create_from_dict(
        {
            "type": "text2text",
            "instances": [
                {"input": "one plus one", "output": "2"},
                {"input": "two plus two", "output": "4"},
                {"input": "three plus three", "output": "6"},
            ],
        }
    )


def _task_adapter(dataset: Dataset):
    for index, row in enumerate(dataset.to_list()):
        yield EvaluationTask(
            task=TaskSpec(
                task_id=f"math:{index}",
                messages=[{"role": "user", "content": row["input"]}],
                metadata={"index": index},
            ),
            verifier_material={"answer": row["output"]},
        )


def _budget(*, max_model_calls: int = 2) -> EvaluationBudget:
    return EvaluationBudget(
        max_model_calls=max_model_calls,
        max_tool_calls=1,
        max_steps=3,
        max_input_tokens=128,
        max_output_tokens=32,
        wall_time_seconds=5,
    )


class _ExactVerifier:
    def verify(self, task, output, *, verifier_material):
        assert "answer" not in task.metadata
        expected = verifier_material["answer"]
        return VerificationOutcome(
            metrics={"correctness": float(output.value == expected)},
            passed=output.value == expected,
            metadata={"expected_length": len(expected)},
        )


class _Runner:
    def __init__(self, outputs=None):
        self.outputs = outputs or {"math:0": "2", "math:1": "wrong", "math:2": "6"}
        self.tasks = []
        self.capability_profiles = []

    def scaffold_provenance(self):
        return {"role": "reference", "id": "toy", "revision": "v1"}

    def run(self, model, task, *, capability_profile, sampling, budget):
        self.tasks.append(task)
        self.capability_profiles.append(capability_profile)
        return ModelRunOutput(
            value=self.outputs[task.task_id],
            usage=EvaluationUsage(
                model_calls=1,
                tool_calls=0,
                steps=1,
                input_tokens=4,
                output_tokens=1,
                wall_time_seconds=0.01,
                cost=0.25,
            ),
            artifact_ref=f"artifacts/{task.task_id}.json",
            metadata={"finish_reason": "stop"},
        )


def _recipe(*, budget=None, verifier=None) -> EvaluationRecipe:
    return EvaluationRecipe(
        name="toy-direct",
        task_adapter=_task_adapter,
        capability_profile=CapabilityProfile(name="direct-answer"),
        sampling=SamplingConfig(temperature=0, top_p=1, seed=7),
        budget=budget or _budget(),
        verifier=verifier or _ExactVerifier(),
        metadata={"protocol": "toy-v1"},
    )


def _evaluator() -> Evaluator:
    return Evaluator(
        ModelArguments(model_name_or_path="example/model", model_revision="revision-1"),
        DatasetArguments(dataset_path=None, dataset_name="toy"),
        EvaluatorArguments(),
    )


def test_recipe_path_preserves_simple_evaluator_entrypoint_and_hidden_verifier_boundary():
    evaluator = _evaluator()
    runner = _Runner()

    result = evaluator.evaluate(object(), _dataset(), recipe=_recipe(), runner=runner)

    assert evaluator._legacy_initialized is False
    assert [task.messages[0]["content"] for task in runner.tasks] == [
        "one plus one",
        "two plus two",
        "three plus three",
    ]
    assert all("answer" not in task.metadata for task in runner.tasks)
    assert result.summary == {
        "total_samples": 3,
        "completed_samples": 3,
        "failed_samples": 0,
        "failure_rate": 0.0,
        "passed_samples": 2,
        "scored_samples": 3,
        "pass_rate": pytest.approx(2 / 3),
        "completed_pass_rate": pytest.approx(2 / 3),
        "metrics": {"correctness": pytest.approx(2 / 3)},
        "metric_counts": {"correctness": 3},
        "failures": {},
        "usage": {
            "model_calls": 3,
            "tool_calls": 0,
            "steps": 3,
            "wall_time_seconds": pytest.approx(0.03),
            "input_tokens": 12,
            "output_tokens": 3,
            "cost": pytest.approx(0.75),
        },
    }
    assert result.records[1].metrics == {"correctness": 0.0}
    assert result.records[2].artifact_ref == "artifacts/math:2.json"
    assert result.provenance.recipe["name"] == "toy-direct"
    assert result.provenance.capability_profile == {
        "name": "direct-answer",
        "affordances": [],
        "config": {},
    }
    assert result.provenance.model["model_name_or_path"] == "example/model"
    assert result.provenance.model["model_revision"] == "revision-1"
    assert result.provenance.execution["runner"].endswith("._Runner")
    assert result.provenance.execution["runtime"].endswith(".LocalEvaluationRuntime")
    assert result.provenance.scaffold == {"role": "reference", "id": "toy", "revision": "v1"}
    assert result.provenance.dataset["dataset_type"] == "text2text"
    assert result.to_dict()["records"][0]["status"] == "completed"


def test_recipe_can_be_configured_on_evaluator_for_two_argument_entrypoint():
    runner = _Runner()
    evaluator = Evaluator(
        ModelArguments(model_name_or_path="example/model"),
        DatasetArguments(dataset_path=None, dataset_name="toy"),
        EvaluatorArguments(),
        recipe=_recipe(),
        runner=runner,
    )

    result = evaluator(object(), _dataset())

    assert result.summary["passed_samples"] == 2
    assert evaluator._legacy_initialized is False


def test_evaluator_accepts_an_explicit_runtime_at_construction():
    evaluator = Evaluator(
        ModelArguments(model_name_or_path="example/model"),
        DatasetArguments(dataset_path=None, dataset_name="toy"),
        EvaluatorArguments(),
        recipe=_recipe(),
        runner=_Runner(),
        runtime=LocalEvaluationRuntime(max_concurrency=2),
    )

    result = evaluator.evaluate(object(), _dataset())

    assert result.provenance.execution["runtime"].endswith(".LocalEvaluationRuntime")
    assert result.provenance.execution["runtime_config"]["max_concurrency"] == 2


class _FailureRunner(_Runner):
    def run(self, model, task, *, capability_profile, sampling, budget):
        if task.task_id == "math:0":
            raise EvaluationSampleError(EvaluationFailureType.INVALID_TOOL_CALL, "malformed calculator call")
        if task.task_id == "math:1":
            raise ConnectionError("backend unavailable")
        return super().run(
            model,
            task,
            capability_profile=capability_profile,
            sampling=sampling,
            budget=budget,
        )


def test_recipe_isolates_structured_sample_failures_and_continues():
    result = _evaluator().evaluate(object(), _dataset(), recipe=_recipe(), runner=_FailureRunner())

    assert [record.status for record in result.records] == ["failed", "failed", "completed"]
    assert result.summary["failures"] == {"backend_failure": 1, "invalid_tool_call": 1}
    assert result.records[0].failure.failure_type is EvaluationFailureType.INVALID_TOOL_CALL
    assert result.records[1].failure.retryable is True
    assert result.summary["metrics"] == {"correctness": 1.0}


class _OverBudgetRunner(_Runner):
    def run(self, model, task, *, capability_profile, sampling, budget):
        output = super().run(
            model,
            task,
            capability_profile=capability_profile,
            sampling=sampling,
            budget=budget,
        )
        return ModelRunOutput(
            value=output.value,
            usage=EvaluationUsage(model_calls=2, tool_calls=0, steps=1, wall_time_seconds=0.01),
            artifact_ref=output.artifact_ref,
        )


class _ShouldNotRunVerifier:
    def verify(self, task, output, *, verifier_material):
        raise AssertionError("verifier must not run for a budget failure")


def test_recipe_enforces_reported_budget_before_hidden_verification():
    result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=_recipe(budget=_budget(max_model_calls=1), verifier=_ShouldNotRunVerifier()),
        runner=_OverBudgetRunner(),
    )

    assert result.summary["failures"] == {"budget_exhausted": 3}
    assert all(record.artifact_ref is not None for record in result.records)


class _ConcurrencyRunner(_Runner):
    def __init__(self):
        super().__init__({"math:0": "2", "math:1": "4", "math:2": "6"})
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def run(self, model, task, *, capability_profile, sampling, budget):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.03)
            return super().run(
                model,
                task,
                capability_profile=capability_profile,
                sampling=sampling,
                budget=budget,
            )
        finally:
            with self.lock:
                self.active -= 1


def test_recipe_bounds_concurrency_and_preserves_dataset_order():
    runner = _ConcurrencyRunner()
    result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=_recipe(),
        runner=runner,
        runtime=LocalEvaluationRuntime(max_concurrency=2),
    )

    assert runner.maximum_active == 2
    assert [record.task_id for record in result.records] == ["math:0", "math:1", "math:2"]


def test_recipe_rejects_duplicate_task_ids_while_consuming_the_task_stream():
    runner = _Runner()

    def duplicate_tasks(dataset):
        task = EvaluationTask(TaskSpec(task_id="duplicate", messages=[]), verifier_material="hidden")
        return [task, task]

    recipe = EvaluationRecipe(
        name="duplicates",
        task_adapter=duplicate_tasks,
        capability_profile=CapabilityProfile(name="direct-answer"),
        sampling=SamplingConfig(temperature=0, top_p=1, seed=0),
        budget=_budget(),
        verifier=_ExactVerifier(),
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        _evaluator().evaluate(object(), _dataset(), recipe=recipe, runner=runner)
    assert [task.task_id for task in runner.tasks] == ["duplicate"]


def test_legacy_metric_path_remains_available(monkeypatch):
    evaluator = _evaluator()
    initialized = []
    monkeypatch.setattr(evaluator, "_initialize_legacy_runtime", lambda: initialized.append(True))
    monkeypatch.setattr(evaluator, "_evaluate_nll", lambda model, dataset, verbose: 1.25)

    result = evaluator.evaluate(object(), _dataset(), metric="nll", verbose=False)

    assert result == 1.25
    assert initialized == [True]


def test_recipe_requires_runner_and_rejects_unpaired_runner():
    evaluator = _evaluator()
    with pytest.raises(ValueError, match="runner is required"):
        evaluator.evaluate(object(), _dataset(), recipe=_recipe())
    with pytest.raises(ValueError, match="recipe is required"):
        evaluator.evaluate(object(), _dataset(), runner=_Runner())


def test_unknown_legacy_metric_keeps_not_implemented_behavior_without_initializing_runtime():
    evaluator = _evaluator()

    with pytest.raises(NotImplementedError, match="metric unknown is not supported"):
        evaluator.evaluate(object(), _dataset(), metric="unknown")

    assert evaluator._legacy_initialized is False
