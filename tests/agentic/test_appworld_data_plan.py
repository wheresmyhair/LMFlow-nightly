import os

import pytest

from lmflow.agentic.appworld_data_plan import (
    build_appworld_inherited_selections,
    build_appworld_presft_paired_baseline_artifacts,
    build_appworld_train_d1_d2_plan_bundle,
)
from lmflow.agentic.appworld_protocol import APPWORLD_SCENARIO_CURRICULUM_TASK_IDS


def _record(*, ordinal, task_id, seed, data_class, fit_16k, fit_32k, action_path):
    tags = ["inherited", "success-plus-recovery"]
    if data_class == "A":
        tags.append("success-only")
    if fit_16k:
        tags.append("short-16k")
    if not fit_32k:
        tags.append("over-32k")
    return {
        "attempt_id": f"source-{ordinal}",
        "candidate_id": f"candidate-{ordinal}",
        "plan_entry_id": f"{ordinal + 1:064x}",
        "ordinal": ordinal,
        "task_id": task_id,
        "sample_id": f"seed-{seed}",
        "record_ref": f"source-{ordinal}/record.json",
        "record_file_sha256": "a" * 64,
        "record_manifest_sha256": "b" * 64,
        "selection_tags": tags,
        "metadata": {
            "scenario_id": task_id.rsplit("_", maxsplit=1)[0],
            "difficulty": 1,
            "action_path_sha256": action_path,
            "data_class": data_class,
            "official_success": True,
            "official_test_count": 2,
            "official_test_pass_count": 2,
            "official_test_pass_fraction": 1.0,
            "replay_match": True,
            "collateral_invariant_passed": True,
            "hidden_verifier_material_included": False,
            "fit_16k": fit_16k,
            "fit_32k": fit_32k,
            "full_sequence_tokens": 8000 if fit_16k else 48000,
            "invalid_actions": 0 if data_class == "A" else 1,
            "valid_actions": 4,
            "recovery_count": 0 if data_class == "A" else 1,
            "state_change_steps": 1,
            "duplicate_actions": 0,
            "steps": 4,
            "provider_total_tokens": 1000 + ordinal,
            "provider_input_tokens": 800 + ordinal,
            "provider_output_tokens": 200,
            "latency_seconds": 3.0,
            "seed": seed,
            "trajectory_ref": f"source-{ordinal}/trajectory.json",
            "replay_ref": f"source-{ordinal}/replay.json",
            "admission_ref": f"source-{ordinal}/admission.json",
        },
    }


def test_inherited_selections_keep_quality_length_and_task_coverage_separate():
    records = [
        _record(
            ordinal=0,
            task_id="task-a",
            seed=100,
            data_class="B",
            fit_16k=True,
            fit_32k=True,
            action_path="a" * 64,
        ),
        _record(
            ordinal=1,
            task_id="task-a",
            seed=101,
            data_class="A",
            fit_16k=False,
            fit_32k=False,
            action_path="b" * 64,
        ),
        _record(
            ordinal=2,
            task_id="task-b",
            seed=100,
            data_class="A",
            fit_16k=True,
            fit_32k=True,
            action_path="c" * 64,
        ),
        _record(
            ordinal=3,
            task_id="task-b",
            seed=101,
            data_class="A",
            fit_16k=True,
            fit_32k=True,
            action_path="d" * 64,
        ),
    ]

    selections = build_appworld_inherited_selections(records)

    assert selections["inherited-success-only"]["selected_count"] == 3
    assert selections["inherited-success-plus-recovery"]["selected_count"] == 4
    assert selections["short-16k-success-only"]["selected_count"] == 2
    assert selections["short-16k-success-plus-recovery"]["selected_count"] == 3
    controlled = selections["over-32k-controlled-prefix-inputs"]
    assert controlled["selected_count"] == 1
    assert controlled["policy_identity"]["training_eligible"] is False
    coverage = selections["inherited-task-coverage"]
    assert coverage["selected_count"] == 2
    assert [record["candidate_id"] for record in coverage["selected"]] == ["candidate-1", "candidate-2"]


def test_presft_paired_baseline_reuses_exact_seed100_teacher_rows_without_execution():
    records = [
        _record(
            ordinal=ordinal * 2,
            task_id=task_id,
            seed=100,
            data_class="B" if ordinal == 7 else "A",
            fit_16k=True,
            fit_32k=True,
            action_path=f"{ordinal + 1:064x}",
        )
        for ordinal, task_id in enumerate(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS)
    ]

    artifacts = build_appworld_presft_paired_baseline_artifacts(records, code_commit="1" * 40)

    assert artifacts["task_selection"]["task_count"] == 12
    assert artifacts["task_selection"]["ordered_task_ids"] == list(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS)
    assert artifacts["plan"]["authorization"]["execution_allowed"] is False
    assert artifacts["plan"]["teacher_reference"]["summary"]["official_success_count"] == 12
    assert [arm["lineage_label"] for arm in artifacts["plan"]["student_arms"]] == [
        "Starting checkpoint",
        "Base",
    ]
    assert artifacts["run_sheet"]["budget_recommendation"]["recommended_hard_amount_cny"] == 16.0


@pytest.mark.optional_backend
def test_real_sealed_l0_builds_no_request_train_d1_d2_plan(tmp_path):
    appworld_root = os.environ.get("APPWORLD_ROOT")
    aggregate_root = os.environ.get("APPWORLD_L0_AGGREGATE")
    code_commit = os.environ.get("APPWORLD_PLAN_CODE_COMMIT")
    if not appworld_root or not aggregate_root or not code_commit:
        pytest.skip("pinned AppWorld and sealed L0 identities select the optional planning integration")

    target = tmp_path / "train-d1-d2-plan"
    result = build_appworld_train_d1_d2_plan_bundle(
        artifact_dir=target,
        appworld_root=appworld_root,
        l0_aggregate_dir=aggregate_root,
        code_commit=code_commit,
    )

    assert result["plan"]["authorization"]["execution_allowed"] is False
    assert result["plan"]["coverage_target"] == {
        "task_count": 72,
        "scenario_count": 24,
        "task_dataset_manifest_sha256": result["plan"]["coverage_target"]["task_dataset_manifest_sha256"],
        "inherited_task_count": 12,
        "generated_task_count": 60,
    }
    assert result["plan"]["generation"]["initial_paid_candidate_count"] == 60
    assert (target / "generation-candidate-plan.json").is_file()
    assert (target / "selections/inherited-task-coverage.json").is_file()
    assert (target / "presft-paired-baseline/plan.json").is_file()
    assert (target / "presft-paired-baseline/task-selection.json").is_file()
    assert result["artifact_file_count"] == 20
