import os
from pathlib import Path
from types import SimpleNamespace

from lmflow.agentic.appworld_data_factory import finalize_appworld_candidate, summarize_appworld_candidates
from lmflow.agentic.appworld_episode import replay_appworld_episode, run_appworld_episode
from lmflow.agentic.appworld_protocol import APPWORLD_TINY_TASK_IDS, verify_manifest_digest


class FakeBackend:
    def __init__(self):
        self.responses = [
            "I will probe.\n```python\nfail()\n```",
            "I will recover and finish.\n```python\ncomplete()\n```",
        ]

    def complete(self, *, messages, tools, model_name, model_kwargs):
        content = self.responses.pop(0)
        return {
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
            "raw_response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        }


class FakeTracker:
    def __init__(self, success):
        self.success = success

    def to_dict(self, stats_only=False):
        result = {"success": self.success, "difficulty": 1, "num_tests": 1}
        if not stats_only:
            result.update(
                {
                    "passes": [{"requirement": "done", "label": None}] if self.success else [],
                    "failures": [] if self.success else [{"requirement": "done", "trace": "", "label": None}],
                }
            )
        return result


class FakeWorld:
    def __init__(self, root: Path, *, state_value: str = "changed"):
        self.task = SimpleNamespace(
            instruction="Complete the fake task.",
            supervisor=SimpleNamespace(first_name="Ada", last_name="Lovelace"),
            app_descriptions={"fake": "A fake app."},
        )
        self.output_directory = str(root / "output")
        self.output_db_home_path_on_disk = str(root / "output" / "dbs")
        Path(self.output_db_home_path_on_disk).mkdir(parents=True)
        self.requester = SimpleNamespace(request_tracker=SimpleNamespace(requests=[]))
        self.completed = False
        self.state_value = state_value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def execute(self, code):
        if code == "fail()":
            self.requester.request_tracker.requests.append(
                {"method": "post", "url": "/fake/login", "data": {"password": "synthetic"}}
            )
            return "Execution failed. Traceback:\nintentional"
        assert code == "complete()"
        self.requester.request_tracker.requests.append({"method": "post", "url": "/fake/complete", "data": {}})
        Path(self.output_db_home_path_on_disk, "state.json").write_text(self.state_value, encoding="utf-8")
        self.completed = True
        return "Marked complete."

    def task_completed(self):
        return self.completed

    def evaluate(self, suppress_errors=True):
        return FakeTracker(self.completed)


def test_episode_records_failure_recovery_state_and_training_projection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )

    def unexpected_freezegun_configure():
        raise AssertionError("custom world factories must not require AppWorld dependencies")

    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.configure_appworld_freezegun",
        unexpected_freezegun_configure,
    )
    world = FakeWorld(tmp_path)

    def factory(**kwargs):
        return world

    original_appworld_root = os.environ.get("APPWORLD_ROOT")
    try:
        result = run_appworld_episode(
            FakeBackend(),
            task_id=APPWORLD_TINY_TASK_IDS[0],
            model_name="Qwen/Qwen3-8B",
            model_revision="revision",
            trajectory_id="run:task",
            appworld_root=tmp_path,
            appworld_source=tmp_path,
            experiment_name="test",
            world_factory=factory,
        )
    finally:
        if original_appworld_root is None:
            os.environ.pop("APPWORLD_ROOT", None)
        else:
            os.environ["APPWORLD_ROOT"] = original_appworld_root

    verify_manifest_digest(result.artifact)
    metrics = result.artifact["metrics"]
    assert metrics["success"] is True
    assert metrics["tool_calls"] == 2
    assert metrics["valid_tool_calls"] == 1
    assert metrics["invalid_tool_calls"] == 1
    assert metrics["recovery_count"] == 1
    assert metrics["state_change_steps"] == 1
    assert metrics["termination_reason"] == "task_completed"
    assert metrics["usage"]["input_tokens"] == 20
    assert metrics["latency_seconds"]["initialization"] < 5
    assert metrics["latency_seconds"]["evaluation"] < 5
    assert result.artifact["action_steps"][0]["api_calls_redacted"][0]["data"]["password"] == "[REDACTED]"

    messages = result.training_projection["instances"][0]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["loss"] is False
    assert messages[-1]["loss"] is True
    assert messages[-1]["content"].endswith("\n\n")
    metadata = result.training_projection["instances"][0]["metadata"]
    assert metadata["replay_required"] is True
    assert metadata["eligible_for_success_only_sft"] is False
    assert metadata["eligible_for_success_plus_recovery_sft"] is False


def test_replay_gate_admits_verified_recovery_and_masks_failed_action(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    original_world = FakeWorld(tmp_path / "original")
    result = run_appworld_episode(
        FakeBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="teacher-model",
        model_revision="fixed",
        trajectory_id="pilot:task:candidate-0",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="pilot-original",
        world_factory=lambda **kwargs: original_world,
    )
    replay = replay_appworld_episode(
        result.artifact,
        appworld_root=tmp_path,
        experiment_name="pilot-replay",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "replay"),
    )

    verify_manifest_digest(replay)
    assert replay["replay_match"] is True
    assert replay["collateral_invariant_passed"] is True
    finalized = finalize_appworld_candidate(result, replay)
    assert finalized.admission["data_class"] == "B"
    assert finalized.admission["admitted_for_sft"] is True
    assert finalized.admission["sft_arms"] == {"success_only": False, "success_plus_recovery": True}
    assert finalized.admission["usage"]["trainable_output_tokens"] == 5
    messages = finalized.training_projection["instances"][0]["messages"]
    assert [message["loss"] for message in messages if message["role"] == "assistant"] == [False, True]
    assert finalized.training_projection["instances"][0]["metadata"]["hidden_verifier_material_included"] is False

    summary = summarize_appworld_candidates([finalized.admission])
    assert summary["accepted_yield"] == 1.0
    assert summary["class_counts"]["B"] == 1
    assert summary["trainable_output_tokens"] == {"reported_count": 1, "p50": 5, "p95": 5}


def test_replay_mismatch_is_excluded_as_invalid_infrastructure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.load_reference_prompt",
        lambda source: "USER:\nTask: {{ instruction }}",
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_episode.scaffold_identity",
        lambda source: {"id": "fake-reference", "prompt_sha256": "0" * 64},
    )
    monkeypatch.setattr("lmflow.agentic.appworld_episode.configure_appworld_freezegun", lambda: None)

    result = run_appworld_episode(
        FakeBackend(),
        task_id=APPWORLD_TINY_TASK_IDS[0],
        model_name="teacher-model",
        model_revision="fixed",
        trajectory_id="pilot:task:candidate-mismatch",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        experiment_name="pilot-original-mismatch",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "original-mismatch"),
    )
    replay = replay_appworld_episode(
        result.artifact,
        appworld_root=tmp_path,
        experiment_name="pilot-replay-mismatch",
        world_factory=lambda **kwargs: FakeWorld(tmp_path / "replay-mismatch", state_value="different"),
    )
    finalized = finalize_appworld_candidate(result, replay)

    assert replay["replay_match"] is False
    assert finalized.admission["data_class"] == "E"
    assert finalized.admission["collateral_rejected"] is True
    assert finalized.training_projection is None
