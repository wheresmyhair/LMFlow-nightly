import copy
import json

import pytest

from lmflow.agentic import generate_gsm8k_tool_dataset, gsm8k_example_to_task
from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset


class RecordingBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return copy.deepcopy(self.responses.pop(0))


def _task(index, *, question, answer):
    return gsm8k_example_to_task(
        {
            "question": question,
            "answer": f"Compute the result. #### {answer}",
        },
        split="train",
        index=index,
    )


def _completion(content="", *, tool_calls=None, cost=0.0):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "message": message,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "cost": cost,
        "raw_response": {"id": "fixture-response"},
    }


def _reward_tool_call(answer, *, call_id="call-reward"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "calc_gsm8k_reward",
            "arguments": json.dumps({"answer": answer}),
        },
    }


def test_generates_raw_trajectories_and_selects_successful_sft_conversations(tmp_path):
    tasks = [
        _task(0, question="What is 18 plus 7?", answer="25"),
        _task(1, question="What is 3 plus 4?", answer="7"),
    ]
    backend = RecordingBackend(
        [
            _completion("#### 25", cost=0.25),
            _completion("#### 24", cost=0.5),
            _completion("#### 7", cost=0.75),
            _completion("#### 8", cost=1.0),
        ]
    )
    artifact_dir = tmp_path / "run"

    report = generate_gsm8k_tool_dataset(
        backend,
        tasks,
        artifact_dir=artifact_dir,
        model_name="fixture-model",
        session_id="run-1",
        model_kwargs={"temperature": 0.7},
        rollouts_per_task=2,
    )

    assert report == {
        "session_id": "run-1",
        "model_name": "fixture-model",
        "task_count": 2,
        "rollouts_per_task": 2,
        "trajectory_count": 4,
        "successful_trajectory_count": 2,
        "conversation_count": 2,
        "success_rate": 0.5,
        "completion_cost": 2.5,
        "model_steps": 4,
        "reward_tool_calls": 0,
    }
    assert json.loads((artifact_dir / "report.json").read_text(encoding="utf-8")) == report

    trajectories = [
        json.loads(line) for line in (artifact_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [trajectory["trajectory_id"] for trajectory in trajectories] == [
        "run-1:openai/gsm8k:train:0:rollout-0",
        "run-1:openai/gsm8k:train:0:rollout-1",
        "run-1:openai/gsm8k:train:1:rollout-0",
        "run-1:openai/gsm8k:train:1:rollout-1",
    ]
    assert [trajectory["final_metrics"]["reward"] for trajectory in trajectories] == [1.0, 0.0, 1.0, 0.0]

    dataset_document = json.loads((artifact_dir / "dataset" / "data.json").read_text(encoding="utf-8"))
    assert [instance["conversation_id"] for instance in dataset_document["instances"]] == [
        "run-1:openai/gsm8k:train:0:rollout-0",
        "run-1:openai/gsm8k:train:1:rollout-0",
    ]
    dataset = Dataset(
        DatasetArguments(
            dataset_path=str(artifact_dir / "dataset"),
            dataset_cache_dir=str(tmp_path / "cache"),
        )
    )
    assert dataset.get_type() == "conversation"
    assert len(dataset) == 2
    assert len(backend.requests) == 4
    assert all(request["model_kwargs"] == {"temperature": 0.7} for request in backend.requests)


def test_publishes_raw_diagnostics_when_no_trajectory_earns_reward(tmp_path):
    backend = RecordingBackend([_completion("#### 24")])
    artifact_dir = tmp_path / "run"

    report = generate_gsm8k_tool_dataset(
        backend,
        [_task(0, question="What is 18 plus 7?", answer="25")],
        artifact_dir=artifact_dir,
        model_name="fixture-model",
        session_id="run-zero",
    )

    assert report["trajectory_count"] == 1
    assert report["successful_trajectory_count"] == 0
    assert report["success_rate"] == 0.0
    assert json.loads((artifact_dir / "dataset" / "data.json").read_text(encoding="utf-8"))["instances"] == []
    assert len((artifact_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_episode_failure_removes_the_staging_directory_and_adds_context(tmp_path):
    backend = RecordingBackend(
        [
            _completion("#### 25"),
            _completion(tool_calls=[_reward_tool_call("25")]),
        ]
    )
    artifact_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="max_steps=1") as error:
        generate_gsm8k_tool_dataset(
            backend,
            [_task(0, question="What is 18 plus 7?", answer="25")],
            artifact_dir=artifact_dir,
            model_name="fixture-model",
            session_id="run-failure",
            rollouts_per_task=2,
            max_steps=1,
        )

    assert error.value.__notes__ == ["GSM8K task_id='openai/gsm8k:train:0', rollout_index=1"]
    assert not artifact_dir.exists()
    assert list(tmp_path.glob(".run.*.tmp")) == []


def test_rejects_duplicate_tasks_and_existing_output_before_calling_backend(tmp_path):
    task = _task(0, question="What is 18 plus 7?", answer="25")
    backend = RecordingBackend([])

    with pytest.raises(ValueError, match="duplicate task_id"):
        generate_gsm8k_tool_dataset(
            backend,
            [task, copy.deepcopy(task)],
            artifact_dir=tmp_path / "duplicate-run",
            model_name="fixture-model",
            session_id="run-duplicate",
        )

    existing_dir = tmp_path / "existing-run"
    existing_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_gsm8k_tool_dataset(
            backend,
            [task],
            artifact_dir=existing_dir,
            model_name="fixture-model",
            session_id="run-existing",
        )

    assert backend.requests == []


@pytest.mark.parametrize("rollouts_per_task", [True, 0, -1])
def test_rejects_invalid_rollout_count_without_calling_backend(tmp_path, rollouts_per_task):
    backend = RecordingBackend([])

    with pytest.raises((TypeError, ValueError), match="rollouts_per_task"):
        generate_gsm8k_tool_dataset(
            backend,
            [_task(0, question="What is 18 plus 7?", answer="25")],
            artifact_dir=tmp_path / "run",
            model_name="fixture-model",
            session_id="run-invalid",
            rollouts_per_task=rollouts_per_task,
        )

    assert backend.requests == []
