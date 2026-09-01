from pathlib import Path

import pytest

from lmflow.agentic.appworld_data_factory import run_appworld_data_factory
from lmflow.agentic.appworld_episode import (
    APPWORLD_EPISODE_FORMAT_VERSION,
    APPWORLD_REPLAY_FORMAT_VERSION,
    AppWorldEpisodeResult,
)
from lmflow.agentic.appworld_protocol import (
    APPWORLD_DATA_PILOT_TASK_IDS,
    APPWORLD_SCENARIO_CURRICULUM_TASK_IDS,
    verify_manifest_digest,
    with_manifest_digest,
)


class FakeDataset:
    def __init__(self, task_ids):
        self.task_ids = task_ids

    def to_dict(self):
        return {
            "type": "text_only",
            "instances": [{"task_id": task_id, "text": f"synthetic {task_id}"} for task_id in self.task_ids],
        }


def _fake_episode(tmp_path: Path, *, task_id: str, trajectory_id: str) -> AppWorldEpisodeResult:
    artifact = with_manifest_digest(
        {
            "format_version": APPWORLD_EPISODE_FORMAT_VERSION,
            "trajectory_id": trajectory_id,
            "task": {"task_id": task_id, "source_split": "train"},
            "runner_error": None,
            "evaluator_error": None,
            "model_steps": [
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ],
            "action_steps": [{"valid": True}],
            "metrics": {
                "success": True,
                "invalid_tool_calls": 0,
                "recovery_count": 0,
                "failure_type": None,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        }
    )
    training_projection = {
        "type": "conversation",
        "instances": [
            {
                "conversation_id": trajectory_id,
                "messages": [
                    {"role": "user", "content": "synthetic"},
                    {"role": "assistant", "content": "```python\ncomplete()\n```", "loss": True},
                ],
                "metadata": {"official_success": True, "replay_required": True},
            }
        ],
    }
    raw_output_directory = tmp_path / "raw" / trajectory_id.replace(":", "-")
    raw_output_directory.mkdir(parents=True)
    return AppWorldEpisodeResult(
        artifact=artifact,
        training_projection=training_projection,
        raw_output_directory=raw_output_directory,
        official_tracker=None,
    )


def test_data_factory_atomically_publishes_fixed_two_candidate_pilot(monkeypatch, tmp_path):
    dataset_manifest = with_manifest_digest({"format_version": "fake-dataset", "scenario_disjoint": True})
    monkeypatch.setattr(
        "lmflow.agentic.appworld_data_factory.load_pinned_appworld_data_pilot_dataset",
        lambda **kwargs: (FakeDataset(APPWORLD_DATA_PILOT_TASK_IDS), dataset_manifest),
    )
    episode_calls = []

    def fake_run_episode(backend, **kwargs):
        episode_calls.append(kwargs)
        return _fake_episode(
            tmp_path,
            task_id=kwargs["task_id"],
            trajectory_id=kwargs["trajectory_id"],
        )

    monkeypatch.setattr("lmflow.agentic.appworld_data_factory.run_appworld_episode", fake_run_episode)

    def fake_replay(artifact, **kwargs):
        return with_manifest_digest(
            {
                "format_version": APPWORLD_REPLAY_FORMAT_VERSION,
                "source_artifact_sha256": artifact["manifest_sha256"],
                "replay_error": None,
                "replay_match": True,
                "collateral_invariant_passed": True,
                "sealed_partial_signal": False,
            }
        )

    monkeypatch.setattr("lmflow.agentic.appworld_data_factory.replay_appworld_episode", fake_replay)
    target = tmp_path / "factory-run"
    report = run_appworld_data_factory(
        object(),
        artifact_dir=target,
        run_id="synthetic-pilot",
        appworld_root=tmp_path,
        appworld_source=tmp_path,
        teacher_model_name="teacher",
        teacher_model_revision="fixed",
        provider_identity={
            "provider_id": "synthetic",
            "endpoint_label": "offline-test",
            "api_contract": "openai-compatible",
            "pricing": {
                "input_usd_per_million_tokens": 1.0,
                "output_usd_per_million_tokens": 2.0,
                "effective_date": "2026-08-16",
                "source": "synthetic-test",
            },
        },
        model_kwargs={"temperature": 0},
        candidates_per_task=2,
        max_steps=5,
        candidate_seeds=(100, 101),
    )

    verify_manifest_digest(report)
    assert report["summary"]["candidate_count"] == 18
    assert report["summary"]["accepted_count"] == 18
    assert report["summary"]["class_counts"]["A"] == 18
    assert report["summary"]["duplicate_accepted_count"] == 17
    assert report["summary"]["cost_reported_for_all_candidates"] is True
    assert report["summary"]["cost_usd"] == pytest.approx(0.00036)
    assert [call["model_kwargs"]["seed"] for call in episode_calls] == [100, 101] * len(APPWORLD_DATA_PILOT_TASK_IDS)
    assert (target / "attempt-manifest.json").is_file()
    assert (target / "candidate-plan.json").is_file()
    assert (target / "candidates/00000/projection.json").is_file()
    assert (target / "candidates/00000/record.json").is_file()
    assert (target / "selections/success-only.json").is_file()
    assert (target / "selections/success-plus-recovery.json").is_file()
    assert (target / "artifact-manifest.sha256").is_file()
    with pytest.raises(FileExistsError):
        run_appworld_data_factory(
            object(),
            artifact_dir=target,
            run_id="synthetic-pilot",
            appworld_root=tmp_path,
            appworld_source=tmp_path,
            teacher_model_name="teacher",
            teacher_model_revision="fixed",
            provider_identity={
                "provider_id": "synthetic",
                "endpoint_label": "offline-test",
                "api_contract": "openai-compatible",
            },
            model_kwargs={"temperature": 0},
        )


def test_data_factory_rejects_provider_credentials_before_execution(tmp_path):
    with pytest.raises(ValueError, match="sensitive key"):
        run_appworld_data_factory(
            object(),
            artifact_dir=tmp_path / "never-created",
            run_id="synthetic-pilot",
            appworld_root=tmp_path,
            appworld_source=tmp_path,
            teacher_model_name="teacher",
            teacher_model_revision="fixed",
            provider_identity={
                "provider_id": "synthetic",
                "endpoint_label": "offline-test",
                "api_contract": "openai-compatible",
                "api_key": "must-not-be-recorded",
            },
            model_kwargs={"temperature": 0},
        )
    assert not (tmp_path / "never-created").exists()


def test_data_factory_routes_scenario_curriculum_and_preserves_failed_evidence(monkeypatch, tmp_path):
    dataset_manifest = with_manifest_digest({"format_version": "fake-curriculum", "scenario_disjoint": True})
    monkeypatch.setattr(
        "lmflow.agentic.appworld_data_factory.load_pinned_appworld_scenario_curriculum_dataset",
        lambda **kwargs: (FakeDataset(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS), dataset_manifest),
    )
    call_count = 0

    def fail_after_first_candidate(backend, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic provider failure")
        return _fake_episode(tmp_path, task_id=kwargs["task_id"], trajectory_id=kwargs["trajectory_id"])

    monkeypatch.setattr(
        "lmflow.agentic.appworld_data_factory.run_appworld_episode",
        fail_after_first_candidate,
    )
    monkeypatch.setattr(
        "lmflow.agentic.appworld_data_factory.replay_appworld_episode",
        lambda artifact, **kwargs: with_manifest_digest(
            {
                "format_version": APPWORLD_REPLAY_FORMAT_VERSION,
                "source_artifact_sha256": artifact["manifest_sha256"],
                "replay_error": None,
                "replay_match": True,
                "collateral_invariant_passed": True,
                "sealed_partial_signal": False,
            }
        ),
    )
    target = tmp_path / "curriculum-run"
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        run_appworld_data_factory(
            object(),
            artifact_dir=target,
            run_id="synthetic-curriculum",
            appworld_root=tmp_path,
            appworld_source=tmp_path,
            teacher_model_name="teacher",
            teacher_model_revision="fixed",
            provider_identity={
                "provider_id": "synthetic",
                "endpoint_label": "offline-test",
                "api_contract": "openai-compatible",
            },
            model_kwargs={"temperature": 0.2},
            candidates_per_task=2,
            candidate_seeds=(100, 101),
            task_set_id="scenario_curriculum_v1",
            scheduled_ordinals=(0, 1),
        )
    failed = list(tmp_path.glob("curriculum-run.failed-*"))
    assert not target.exists()
    assert len(failed) == 1
    assert call_count == 2
    assert (failed[0] / "candidates/00000/record.json").is_file()
    assert (failed[0] / "candidates/00001").is_dir()
    assert (failed[0] / "failure.json").is_file()
