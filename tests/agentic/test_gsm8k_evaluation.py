"""Offline tests for the GSM8K direct and calculator evaluation recipes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import lmflow.pipeline as pipeline
from lmflow.agentic.gsm8k_evaluation import (
    GSM8K_CALCULATOR_CAPABILITY,
    GSM8K_CALCULATOR_TOOL,
    GSM8KCompletionRunner,
    create_gsm8k_calculator_recipe,
    create_gsm8k_direct_recipe,
    evaluate_arithmetic_expression,
)
from lmflow.args import DatasetArguments, EvaluatorArguments, ModelArguments
from lmflow.datasets import Dataset
from lmflow.pipeline.evaluation import Evaluator


def test_gsm8k_components_are_not_pipeline_exports():
    assert not hasattr(pipeline, "GSM8KCompletionRunner")
    assert not hasattr(pipeline, "create_gsm8k_direct_recipe")


def test_evaluator_uses_pipeline_evaluation_domain():
    assert Evaluator.__module__ == "lmflow.pipeline.evaluation.evaluator"
    with pytest.raises(ModuleNotFoundError):
        __import__("lmflow.evaluation")


class RecordingBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


def _completion(
    content="",
    *,
    tool_calls=None,
    finish_reason="stop",
    prompt_tokens=20,
    completion_tokens=5,
    cost=0.0,
):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "message": message,
        "finish_reason": finish_reason,
        "cost": cost,
        "raw_response": {
            "id": "fixture-response",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    }


def _calculator_call(expression, *, call_id="call-1", name="calculate", arguments=None):
    if arguments is None:
        arguments = json.dumps({"expression": expression})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _dataset():
    return Dataset.create_from_dict(
        {
            "type": "text2text",
            "instances": [
                {
                    "input": "A shelf has 18 books and receives 7 more. How many books are there?",
                    "output": "Add the quantities. #### 25",
                }
            ],
        }
    )


def _evaluator():
    return Evaluator(
        ModelArguments(model_name_or_path="Qwen/Qwen3-8B", model_revision="fixture-revision"),
        DatasetArguments(dataset_path=None, dataset_name="gsm8k-fixture"),
        EvaluatorArguments(),
    )


def _configured_evaluator(recipe, runner):
    return Evaluator(
        ModelArguments(model_name_or_path="Qwen/Qwen3-8B", model_revision="fixture-revision"),
        DatasetArguments(dataset_path=None, dataset_name="gsm8k-fixture"),
        EvaluatorArguments(),
        recipe=recipe,
        runner=runner,
    )


def test_direct_recipe_keeps_gold_hidden_and_reports_strict_metrics():
    backend = RecordingBackend([_completion("18 + 7 = 25. #### 25", cost=0.1)])
    recipe = create_gsm8k_direct_recipe()
    runner = GSM8KCompletionRunner(backend=backend, model_name="served-qwen")

    result = _configured_evaluator(recipe, runner).evaluate(object(), _dataset())

    request = backend.requests[0]
    assert request["tools"] == []
    assert "25" not in repr(request["messages"])
    assert request["model_kwargs"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 1024,
    }
    assert result.records[0].metrics == {
        "final_correctness": 1.0,
        "strict_correctness": 1.0,
        "tool_compliance": 1.0,
        "first_attempt_success": 1.0,
        "recovery": 0.0,
        "direct_answer_fallback": 0.0,
        "tool_error": 0.0,
    }
    assert result.summary["metrics"]["final_correctness"] == 1.0
    assert result.summary["usage"]["input_tokens"] == 20
    assert result.summary["usage"]["output_tokens"] == 5
    assert result.provenance.recipe["metadata"]["gold_visibility"] == "hidden_verifier_only"
    assert result.provenance.scaffold == {
        "role": "reference",
        "id": "lmflow.gsm8k.chat-completions",
        "revision": "v1",
    }
    assert result.provenance.capability_profile == {
        "name": "direct-answer",
        "affordances": [],
        "config": {},
    }


def test_calculator_recipe_reuses_visible_feedback_and_detects_recovery(tmp_path):
    backend = RecordingBackend(
        [
            _completion(
                "My first answer is #### 24",
                tool_calls=[_calculator_call("18 + 7")],
                finish_reason="tool_calls",
                cost=0.2,
            ),
            _completion("The calculator confirms the corrected answer. #### 25", cost=0.3),
        ]
    )
    recipe = create_gsm8k_calculator_recipe()
    runner = GSM8KCompletionRunner(
        backend=backend,
        model_name="served-qwen",
        artifact_dir=tmp_path,
    )

    result = _configured_evaluator(recipe, runner).evaluate(object(), _dataset())

    assert backend.requests[0]["tools"] == [GSM8K_CALCULATOR_TOOL]
    assert backend.requests[0]["model_kwargs"]["max_tokens"] == 1024
    assert backend.requests[1]["model_kwargs"]["max_tokens"] == 1024
    assert backend.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "calculate",
        "content": "Calculator result: 25",
    }
    assert "ground_truth" not in repr(backend.requests)
    record = result.records[0]
    assert record.passed is True
    assert record.metrics == {
        "final_correctness": 1.0,
        "strict_correctness": 1.0,
        "tool_compliance": 1.0,
        "first_attempt_success": 0.0,
        "recovery": 1.0,
        "direct_answer_fallback": 0.0,
        "tool_error": 0.0,
    }
    assert record.usage.model_calls == 2
    assert record.usage.tool_calls == 1
    assert record.usage.steps == 2
    assert record.usage.input_tokens == 40
    assert record.usage.output_tokens == 10
    assert record.usage.cost == pytest.approx(0.5)
    artifact = json.loads(Path(record.artifact_ref).read_text(encoding="utf-8"))
    assert artifact["task_id"] == "openai/gsm8k:test:0"
    assert "ground_truth" not in repr(artifact)
    assert artifact["scaffold"]["role"] == "reference"
    assert artifact["calls"][0]["tool_results"][0]["observation"] == "Calculator result: 25"
    assert result.provenance.capability_profile["affordances"] == [GSM8K_CALCULATOR_CAPABILITY]
    assert result.provenance.capability_profile["config"]["gold_access"] is False


def test_calculator_recipe_reports_direct_answer_fallback():
    backend = RecordingBackend([_completion("I can solve this directly. #### 25")])

    result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=create_gsm8k_calculator_recipe(),
        runner=GSM8KCompletionRunner(backend=backend, model_name="served-qwen"),
    )

    assert result.records[0].metrics["direct_answer_fallback"] == 1.0
    assert result.records[0].metrics["tool_compliance"] == 0.0
    assert result.records[0].metrics["final_correctness"] == 1.0


@pytest.mark.parametrize(
    ("tool_call", "message"),
    [
        (_calculator_call("18 + 7", name="calc_gsm8k_reward"), "function.name must be 'calculate'"),
        (_calculator_call("18 + 7", arguments="not-json"), "must be strict JSON"),
        (_calculator_call("18 + 7", arguments='{"expression":"1","expression":"2"}'), "duplicate JSON key"),
        (_calculator_call("18 + 7", arguments='{"expression":"1","extra":2}'), "contain only string field"),
        (_calculator_call("__import__('os').system('echo leaked')"), "unsupported syntax Call"),
    ],
)
def test_invalid_calculator_calls_are_structured_sample_failures(tool_call, message):
    backend = RecordingBackend([_completion(tool_calls=[tool_call], finish_reason="tool_calls")])

    result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=create_gsm8k_calculator_recipe(),
        runner=GSM8KCompletionRunner(backend=backend, model_name="served-qwen"),
    )

    assert result.summary["failures"] == {"invalid_tool_call": 1}
    assert message in result.records[0].failure.message


def test_timeout_and_model_call_exhaustion_have_distinct_failure_types(tmp_path):
    timeout_backend = RecordingBackend([TimeoutError("provider deadline")])
    timeout_result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=create_gsm8k_direct_recipe(),
        runner=GSM8KCompletionRunner(
            backend=timeout_backend,
            model_name="served-qwen",
            artifact_dir=tmp_path / "timeout",
        ),
    )
    exhausted_backend = RecordingBackend(
        [_completion(tool_calls=[_calculator_call("18 + 7")], finish_reason="tool_calls")]
    )
    exhausted_recipe = create_gsm8k_calculator_recipe()
    exhausted_recipe = type(exhausted_recipe)(
        name=exhausted_recipe.name,
        task_adapter=exhausted_recipe.task_adapter,
        capability_profile=exhausted_recipe.capability_profile,
        sampling=exhausted_recipe.sampling,
        budget=type(exhausted_recipe.budget)(
            max_model_calls=1,
            max_tool_calls=1,
            max_steps=1,
            max_input_tokens=4096,
            max_output_tokens=1024,
            wall_time_seconds=5,
        ),
        verifier=exhausted_recipe.verifier,
        metadata=exhausted_recipe.metadata,
    )
    exhausted_result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=exhausted_recipe,
        runner=GSM8KCompletionRunner(backend=exhausted_backend, model_name="served-qwen"),
    )

    assert timeout_result.summary["failures"] == {"timeout": 1}
    assert timeout_result.records[0].failure.retryable is True
    assert timeout_result.records[0].usage.model_calls == 1
    failure_artifact = json.loads(Path(timeout_result.records[0].artifact_ref).read_text(encoding="utf-8"))
    assert failure_artifact["failure"]["failure_type"] == "timeout"
    assert exhausted_result.summary["failures"] == {"budget_exhausted": 1}


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("18 + 7", "25"),
        ("(9 - 3) * 4", "24"),
        ("7 / 2", "3.5"),
        ("2 ** 10", "1024"),
        ("-3 + +5", "2"),
    ],
)
def test_arithmetic_calculator_accepts_only_bounded_numeric_expressions(expression, expected):
    assert evaluate_arithmetic_expression(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "sum([1, 2])",
        "x + 1",
        "[1, 2][0]",
        "2 ** 13",
        "1e200",
    ],
)
def test_arithmetic_calculator_rejects_code_and_unbounded_values(expression):
    with pytest.raises(ValueError):
        evaluate_arithmetic_expression(expression)


def test_same_dataset_has_different_recipe_level_capabilities_without_row_duplication():
    dataset = _dataset()
    direct_task = list(create_gsm8k_direct_recipe().task_adapter(dataset))[0]
    calculator_task = list(create_gsm8k_calculator_recipe().task_adapter(dataset))[0]

    assert direct_task.task.task_id == calculator_task.task.task_id
    assert direct_task.verifier_material == calculator_task.verifier_material == {"gold_answer": "25"}
    assert direct_task.task.tools == []
    assert calculator_task.task.tools == [GSM8K_CALCULATOR_TOOL]
    assert dataset.to_list()[0] == {
        "input": "A shelf has 18 books and receives 7 more. How many books are there?",
        "output": "Add the quantities. #### 25",
    }


@pytest.mark.parametrize("model_kwargs", [{"temperature": 0.7}, {"tool_choice": "auto"}])
def test_runner_rejects_sampling_overrides_that_would_undermine_recipe_provenance(model_kwargs):
    with pytest.raises(ValueError, match="sampling fields are controlled by the recipe"):
        GSM8KCompletionRunner(
            backend=RecordingBackend([]),
            model_name="served-qwen",
            model_kwargs=model_kwargs,
        )


def test_runner_can_require_the_first_calculator_call_through_provider_tool_choice():
    backend = RecordingBackend(
        [
            _completion(tool_calls=[_calculator_call("18 + 7")], finish_reason="tool_calls"),
            _completion("#### 25"),
        ]
    )
    result = _evaluator().evaluate(
        object(),
        _dataset(),
        recipe=create_gsm8k_calculator_recipe(require_tool_use=True),
        runner=GSM8KCompletionRunner(
            backend=backend,
            model_name="served-qwen",
        ),
    )

    assert backend.requests[0]["model_kwargs"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "calculate"},
    }
    assert "tool_choice" not in backend.requests[1]["model_kwargs"]
    assert result.records[0].metrics["tool_compliance"] == 1.0
    assert result.provenance.capability_profile["config"]["tool_use"] == "required"
