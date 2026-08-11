import importlib
import json

import pytest

from lmflow.agentic import load_gsm8k_tasks

generate_cli = importlib.import_module("lmflow.agentic.generate_gsm8k_dataset")


def _rows():
    return [
        {"question": "What is 1 plus 1?", "answer": "Add the values. #### 2"},
        {"question": "What is 2 plus 3?", "answer": "Add the values. #### 5"},
        {"question": "What is 4 plus 5?", "answer": "Add the values. #### 9"},
    ]


def test_loads_an_exact_official_dataset_range(monkeypatch, tmp_path):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _rows()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    tasks = load_gsm8k_tasks(
        split="train",
        start_index=1,
        limit=2,
        cache_dir=tmp_path / "cache",
    )

    assert calls == [
        (
            ("openai/gsm8k", "main"),
            {"split": "train", "cache_dir": str(tmp_path / "cache")},
        )
    ]
    assert [task.task_id for task in tasks] == [
        "openai/gsm8k:train:1",
        "openai/gsm8k:train:2",
    ]
    assert [task.metadata["index"] for task in tasks] == [1, 2]


def test_loads_local_jsonl_without_exposing_the_absolute_path(monkeypatch, tmp_path):
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _rows()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    tasks = load_gsm8k_tasks(
        split="candidate",
        start_index=0,
        limit=1,
        input_path=input_path,
    )

    assert calls == [
        (
            ("json",),
            {
                "data_files": {"candidate": str(input_path.resolve())},
                "split": "candidate",
                "cache_dir": None,
            },
        )
    ]
    assert tasks[0].task_id == "local:rows.jsonl:candidate:0"
    assert tasks[0].metadata["data_source"] == "local:rows.jsonl"


def test_loads_real_local_jsonl_through_hugging_face_datasets(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n",
        encoding="utf-8",
    )

    tasks = load_gsm8k_tasks(
        split="train",
        start_index=1,
        limit=2,
        input_path=input_path,
        cache_dir=tmp_path / "cache",
    )

    assert [task.task_id for task in tasks] == [
        "local:rows.jsonl:train:1",
        "local:rows.jsonl:train:2",
    ]
    assert tasks[0].messages[-1]["content"].startswith("What is 2 plus 3?")


def test_rejects_a_range_that_cannot_produce_the_requested_task_count(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: _rows())

    with pytest.raises(ValueError, match=r"requested rows \[2, 4\) exceed 'test' dataset size 3"):
        load_gsm8k_tasks(
            split="test",
            start_index=2,
            limit=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"start_index": True}, "start_index must be an integer"),
        ({"start_index": -1}, "start_index must be non-negative"),
        ({"limit": 0}, "limit must be positive"),
        ({"dataset_config": ""}, "dataset_config must be a non-empty string"),
    ],
)
def test_rejects_invalid_loader_arguments_before_loading(monkeypatch, kwargs, error):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("load_dataset must not be called")

    monkeypatch.setattr("datasets.load_dataset", unexpected_load)
    arguments = {
        "split": "train",
        "start_index": 0,
        "limit": 1,
        **kwargs,
    }

    with pytest.raises((TypeError, ValueError), match=error):
        load_gsm8k_tasks(**arguments)


def test_cli_connects_dataset_backend_and_batch_generator(monkeypatch, tmp_path, capsys):
    calls = {}
    tasks = [object(), object()]

    def fake_load_tasks(**kwargs):
        calls["load"] = kwargs
        return tasks

    class FakeBackend:
        def __init__(self, **kwargs):
            calls["backend"] = kwargs

        def __enter__(self):
            calls["entered"] = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            calls["exited"] = True

    def fake_generate(backend, task_values, **kwargs):
        calls["generate"] = (backend, task_values, kwargs)
        return {"trajectory_count": 6, "successful_trajectory_count": 4}

    monkeypatch.setattr(generate_cli, "load_gsm8k_tasks", fake_load_tasks)
    monkeypatch.setattr(generate_cli, "OpenAICompatibleCompletionBackend", FakeBackend)
    monkeypatch.setattr(generate_cli, "generate_gsm8k_tool_dataset", fake_generate)
    monkeypatch.setenv("FIXTURE_API_KEY", "secret-value")

    result = generate_cli.main(
        [
            "--artifact-dir",
            str(tmp_path / "run"),
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model-name",
            "fixture-model",
            "--session-id",
            "run-1",
            "--limit",
            "2",
            "--start-index",
            "3",
            "--rollouts-per-task",
            "3",
            "--max-steps",
            "5",
            "--temperature",
            "0.2",
            "--top-p",
            "0.9",
            "--max-tokens",
            "1024",
            "--seed",
            "17",
            "--timeout-seconds",
            "30",
            "--max-retries",
            "0",
            "--api-key-env",
            "FIXTURE_API_KEY",
        ]
    )

    assert result == 0
    assert calls["load"] == {
        "split": "train",
        "start_index": 3,
        "limit": 2,
        "dataset_name": "openai/gsm8k",
        "dataset_config": "main",
        "input_path": None,
        "cache_dir": None,
    }
    assert calls["backend"] == {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "secret-value",
        "timeout_seconds": 30,
        "max_retries": 0,
    }
    backend, generated_tasks, generate_kwargs = calls["generate"]
    assert isinstance(backend, FakeBackend)
    assert generated_tasks is tasks
    assert generate_kwargs == {
        "artifact_dir": str(tmp_path / "run"),
        "model_name": "fixture-model",
        "session_id": "run-1",
        "model_kwargs": {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1024,
            "seed": 17,
        },
        "rollouts_per_task": 3,
        "max_steps": 5,
    }
    assert calls["entered"] is True
    assert calls["exited"] is True
    assert json.loads(capsys.readouterr().out) == {
        "trajectory_count": 6,
        "successful_trajectory_count": 4,
    }


def test_cli_reports_loading_errors_without_creating_a_backend(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        generate_cli,
        "load_gsm8k_tasks",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("fixture load failure")),
    )

    class UnexpectedBackend:
        def __init__(self, **kwargs):
            raise AssertionError("backend must not be created")

    monkeypatch.setattr(generate_cli, "OpenAICompatibleCompletionBackend", UnexpectedBackend)

    result = generate_cli.main(
        [
            "--artifact-dir",
            str(tmp_path / "run"),
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--model-name",
            "fixture-model",
            "--session-id",
            "run-1",
            "--limit",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "error: fixture load failure\n"
