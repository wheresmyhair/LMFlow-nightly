import copy
import json

import pytest

from lmflow.agentic import (
    OpenAICompatibleCompletionBackend,
    atif_trajectory_to_conversation,
    gsm8k_example_to_task,
    run_gsm8k_tool_episode,
)
from lmflow.agentic.scaffolds.mini_swe_agent import (
    OpenAICompatibleCompletionBackend as LegacyOpenAICompatibleCompletionBackend,
)


class RecordingBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return copy.deepcopy(self.responses.pop(0))


def _task():
    return gsm8k_example_to_task(
        {
            "question": "A shelf has 18 books and receives 7 more. How many books are there?",
            "answer": "Add the two quantities. #### 25",
        },
        split="train",
        index=3,
    )


def _completion(*, content="", reasoning_content=None, tool_calls=None, finish_reason="stop", cost=0.0):
    message = {
        "role": "assistant",
        "content": content,
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "message": message,
        "finish_reason": finish_reason,
        "cost": cost,
        "raw_response": {"id": "fixture-response"},
    }


def _tool_call(answer, *, call_id="call-reward", name="calc_gsm8k_reward", arguments=None):
    if arguments is None:
        arguments = json.dumps({"answer": answer})
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def test_shared_completion_backend_preserves_the_legacy_import_path():
    assert OpenAICompatibleCompletionBackend is LegacyOpenAICompatibleCompletionBackend


def test_runs_reward_feedback_then_returns_a_scored_atif_trajectory():
    backend = RecordingBackend(
        [
            _completion(
                content="I will check the candidate answer.",
                reasoning_content="Eighteen plus seven is twenty-five.",
                tool_calls=[_tool_call("24")],
                finish_reason="tool_calls",
                cost=0.25,
            ),
            _completion(content="The corrected result is #### 25", cost=0.5),
        ]
    )

    trajectory = run_gsm8k_tool_episode(
        backend,
        _task(),
        model_name="fixture-model",
        model_kwargs={"temperature": 0.2},
        trajectory_id="trajectory-3-0",
        session_id="run-3",
    )

    assert len(backend.requests) == 2
    assert backend.requests[0]["model_name"] == "fixture-model"
    assert backend.requests[0]["model_kwargs"] == {"temperature": 0.2}
    assert backend.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-reward",
        "name": "calc_gsm8k_reward",
        "content": "Current parsed answer='#### 24' reward=0.0",
    }

    assert trajectory["trajectory_id"] == "trajectory-3-0"
    assert trajectory["session_id"] == "run-3"
    assert trajectory["extra"]["task_id"] == "openai/gsm8k:train:3"
    assert [step["source"] for step in trajectory["steps"]] == ["system", "user", "agent", "agent"]
    tool_step = trajectory["steps"][2]
    assert tool_step["reasoning_content"] == "Eighteen plus seven is twenty-five."
    assert tool_step["extra"]["provider"]["raw_response"] == {"id": "fixture-response"}
    assert tool_step["tool_calls"][0] == {
        "tool_call_id": "call-reward",
        "function_name": "calc_gsm8k_reward",
        "arguments": {"answer": "24"},
    }
    assert tool_step["observation"]["results"][0]["content"] == ("Current parsed answer='#### 24' reward=0.0")
    assert trajectory["steps"][-1]["metrics"] == {"reward": 1.0}
    assert trajectory["final_metrics"] == {
        "reward": 1.0,
        "reward_tool_calls": 1,
        "model_steps": 2,
        "completion_cost": 0.75,
    }

    conversation = atif_trajectory_to_conversation(trajectory)
    assert conversation["conversation_id"] == "trajectory-3-0"
    assert [message["role"] for message in conversation["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert conversation["messages"][-1]["content"] == "The corrected result is #### 25"


def test_scores_a_direct_final_answer_without_inventing_a_tool_call():
    backend = RecordingBackend([_completion(content="#### 25")])

    trajectory = run_gsm8k_tool_episode(
        backend,
        _task(),
        model_name="fixture-model",
        trajectory_id="direct-answer",
    )

    assert trajectory["final_metrics"] == {
        "reward": 1.0,
        "reward_tool_calls": 0,
        "model_steps": 1,
        "completion_cost": 0.0,
    }
    assert "tool_calls" not in trajectory["steps"][-1]


def test_records_an_incorrect_final_answer_as_zero_reward():
    backend = RecordingBackend([_completion(content="The answer is 24")])

    trajectory = run_gsm8k_tool_episode(
        backend,
        _task(),
        model_name="fixture-model",
        trajectory_id="incorrect-answer",
    )

    assert trajectory["final_metrics"]["reward"] == 0.0


@pytest.mark.parametrize(
    ("tool_call", "error"),
    [
        (_tool_call("25", name="calculator"), "function.name must be 'calc_gsm8k_reward'"),
        (_tool_call("25", arguments="not-json"), "must be strict JSON"),
        (_tool_call("25", arguments='{"answer":"25","answer":"24"}'), "duplicate JSON key"),
        (_tool_call("25", arguments="[]"), "must decode to an object"),
        ({**_tool_call("25"), "type": "custom"}, "type must be 'function'"),
    ],
)
def test_rejects_invalid_tool_calls(tool_call, error):
    backend = RecordingBackend([_completion(tool_calls=[tool_call], finish_reason="tool_calls")])

    with pytest.raises((TypeError, ValueError), match=error):
        run_gsm8k_tool_episode(
            backend,
            _task(),
            model_name="fixture-model",
            trajectory_id="invalid-tool-call",
        )


def test_rejects_duplicate_tool_call_ids_across_model_steps():
    backend = RecordingBackend(
        [
            _completion(tool_calls=[_tool_call("24")], finish_reason="tool_calls"),
            _completion(tool_calls=[_tool_call("25")], finish_reason="tool_calls"),
        ]
    )

    with pytest.raises(ValueError, match="duplicates 'call-reward'"):
        run_gsm8k_tool_episode(
            backend,
            _task(),
            model_name="fixture-model",
            trajectory_id="duplicate-call",
        )


def test_fails_when_the_step_budget_ends_without_a_final_answer():
    backend = RecordingBackend([_completion(tool_calls=[_tool_call("25")], finish_reason="tool_calls")])

    with pytest.raises(RuntimeError, match="max_steps=1 without a final answer"):
        run_gsm8k_tool_episode(
            backend,
            _task(),
            model_name="fixture-model",
            trajectory_id="step-limit",
            max_steps=1,
        )


def test_rejects_task_without_reward_tool_ground_truth_before_calling_backend():
    task = _task()
    task.environment = {}
    backend = RecordingBackend([])

    with pytest.raises(ValueError, match="must contain GSM8K reward-tool ground truth"):
        run_gsm8k_tool_episode(
            backend,
            task,
            model_name="fixture-model",
            trajectory_id="missing-ground-truth",
        )

    assert backend.requests == []
