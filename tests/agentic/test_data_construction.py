from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from lmflow.agentic.data_construction import (
    DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION,
    AdmissionProduct,
    CandidateContext,
    CandidatePlanEntry,
    StageProduct,
    build_candidate_plan,
    build_resume_manifest,
    build_selection_manifest,
    run_data_construction_attempt,
    verify_artifact_manifest,
    verify_manifest_digest,
    with_manifest_digest,
)


class FakeAdapter:
    def __init__(self, *, fail_ordinal: int | None = None):
        self.fail_ordinal = fail_ordinal
        self.interactions: list[int] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"adapter_id": "fake/v1", "quality_semantics": "fixture-local"}

    def interact(self, context: CandidateContext) -> StageProduct:
        self.interactions.append(context.plan_entry.ordinal)
        return StageProduct(
            artifact=with_manifest_digest(
                {
                    "format_version": "fake-trajectory/v1",
                    "candidate_id": context.candidate_id,
                    "task_id": context.plan_entry.task_id,
                }
            ),
            state={"ordinal": context.plan_entry.ordinal},
        )

    def verify(self, context: CandidateContext, interaction: StageProduct) -> StageProduct:
        if context.plan_entry.ordinal == self.fail_ordinal:
            raise RuntimeError("synthetic sensitive-looking failure text")
        return StageProduct(
            artifact=with_manifest_digest(
                {
                    "format_version": "fake-verification/v1",
                    "source_artifact_sha256": interaction.artifact["manifest_sha256"],
                    "passed": context.plan_entry.ordinal == 0,
                }
            )
        )

    def admit(
        self,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
    ) -> AdmissionProduct:
        del interaction
        accepted = verification.artifact["passed"] is True
        admission = with_manifest_digest(
            {
                "format_version": "fake-admission/v1",
                "candidate_id": context.candidate_id,
                "accepted": accepted,
            }
        )
        return AdmissionProduct(
            artifact=admission,
            state={"accepted": accepted},
            selection_tags=("accepted",) if accepted else (),
            record_metadata={
                "accepted": accepted,
                "dedup_key": context.plan_entry.task_id,
                "rank": context.plan_entry.ordinal,
            },
        )

    def project(
        self,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
        admission: AdmissionProduct,
    ) -> Mapping[str, Any] | None:
        del context, interaction, verification
        if admission.state["accepted"] is not True:
            return None
        return {"type": "conversation", "instances": [{"messages": []}]}

    def materialize_evidence(
        self,
        candidate_directory: Path,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
        admission: AdmissionProduct,
    ) -> Mapping[str, Any]:
        del candidate_directory, context, interaction, verification, admission
        return {"fixture": "no-extra-files"}

    def summarize(
        self,
        admissions: Sequence[AdmissionProduct],
        candidate_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {
            "candidate_count": len(candidate_records),
            "accepted_count": sum(admission.state["accepted"] is True for admission in admissions),
        }

    def build_selections(
        self,
        candidate_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        return {
            "accepted": build_selection_manifest(
                candidate_records,
                policy_id="accepted",
                policy_identity={
                    "eligibility": "fixture accepted tag",
                    "dedup_key": "task_id",
                    "ranking": "lower ordinal",
                },
                eligible=lambda record: "accepted" in record["selection_tags"],
                dedup_key=lambda record: record["metadata"]["dedup_key"],
                rank_key=lambda record: record["metadata"]["rank"],
            )
        }


def _dataset_manifest() -> dict[str, Any]:
    return with_manifest_digest({"format_version": "fake-dataset/v1", "ordered_task_ids": ["task-a", "task-b"]})


def _plan(dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return build_candidate_plan(
        (
            CandidatePlanEntry(ordinal=0, task_id="task-a", sample_id="seed-100", sampling={"seed": 100}),
            CandidatePlanEntry(ordinal=1, task_id="task-b", sample_id="seed-101", sampling={"seed": 101}),
        ),
        task_set_id="fixture-v1",
        task_dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
    )


def test_attempt_materializes_explicit_stages_and_deterministic_selection(tmp_path):
    dataset_manifest = _dataset_manifest()
    plan = _plan(dataset_manifest)
    target = tmp_path / "attempt-1"
    adapter = FakeAdapter()

    report = run_data_construction_attempt(
        adapter,
        artifact_dir=target,
        attempt_id="attempt-1",
        task_dataset={"type": "text_only", "instances": []},
        task_dataset_manifest=dataset_manifest,
        candidate_plan=plan,
        run_identity={"recipe_id": "fixture-recipe-v1"},
    )

    verify_manifest_digest(report)
    assert adapter.interactions == [0, 1]
    assert report["summary"] == {"accepted_count": 1, "candidate_count": 2}
    assert report["selection_refs"]["accepted"]["selected_count"] == 1
    assert (target / "candidates/00000/trajectory.json").is_file()
    assert (target / "candidates/00000/verification.json").is_file()
    assert (target / "candidates/00000/admission.json").is_file()
    assert (target / "candidates/00000/projection.json").is_file()
    assert not (target / "candidates/00001/projection.json").exists()
    assert (target / "selections/accepted.json").is_file()
    assert (target / "artifact-manifest.sha256").is_file()
    artifact_check = verify_artifact_manifest(target)
    assert artifact_check["file_count"] > 0
    record = json.loads((target / "candidates/00000/record.json").read_text(encoding="utf-8"))
    verify_manifest_digest(record)
    with pytest.raises(FileExistsError):
        run_data_construction_attempt(
            adapter,
            artifact_dir=target,
            attempt_id="attempt-1",
            task_dataset={"type": "text_only", "instances": []},
            task_dataset_manifest=dataset_manifest,
            candidate_plan=plan,
            run_identity={"recipe_id": "fixture-recipe-v1"},
        )


def test_artifact_manifest_rejects_tampering_and_unmanaged_files(tmp_path):
    dataset_manifest = _dataset_manifest()
    target = tmp_path / "attempt-tamper"
    run_data_construction_attempt(
        FakeAdapter(),
        artifact_dir=target,
        attempt_id="attempt-tamper",
        task_dataset={"type": "text_only", "instances": []},
        task_dataset_manifest=dataset_manifest,
        candidate_plan=_plan(dataset_manifest),
        run_identity={"recipe_id": "fixture-recipe-v1"},
        scheduled_ordinals=(0,),
    )

    trajectory_path = target / "candidates/00000/trajectory.json"
    original = trajectory_path.read_text(encoding="utf-8")
    trajectory_path.write_text(original + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="file digest mismatch"):
        verify_artifact_manifest(target)
    trajectory_path.write_text(original, encoding="utf-8")
    (target / "unmanaged.txt").write_text("unmanaged", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        verify_artifact_manifest(target)


def test_candidate_plan_rejects_duplicate_task_sample_slots():
    dataset_manifest = _dataset_manifest()
    with pytest.raises(ValueError, match="task_id/sample_id slots must be unique"):
        build_candidate_plan(
            (
                CandidatePlanEntry(ordinal=0, task_id="task-a", sample_id="seed-100"),
                CandidatePlanEntry(ordinal=1, task_id="task-a", sample_id="seed-100"),
            ),
            task_set_id="fixture-v1",
            task_dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
        )


def test_attempt_preserves_completed_and_current_stage_evidence_on_failure(tmp_path):
    dataset_manifest = _dataset_manifest()
    plan = _plan(dataset_manifest)
    target = tmp_path / "failed-attempt"

    with pytest.raises(RuntimeError, match="synthetic sensitive-looking failure text"):
        run_data_construction_attempt(
            FakeAdapter(fail_ordinal=1),
            artifact_dir=target,
            attempt_id="failed-attempt",
            task_dataset={"type": "text_only", "instances": []},
            task_dataset_manifest=dataset_manifest,
            candidate_plan=plan,
            run_identity={"recipe_id": "fixture-recipe-v1"},
        )

    failed = list(tmp_path.glob("failed-attempt.failed-*"))
    assert len(failed) == 1
    assert not target.exists()
    assert (failed[0] / "candidates/00000/record.json").is_file()
    assert (failed[0] / "candidates/00001/trajectory.json").is_file()
    assert not (failed[0] / "candidates/00001/verification.json").exists()
    failure = json.loads((failed[0] / "failure.json").read_text(encoding="utf-8"))
    verify_manifest_digest(failure)
    assert failure["failed_stage"] == "verify"
    assert failure["failed_ordinal"] == 1
    assert failure["completed_candidate_count"] == 1
    assert failure["exception_type"] == "RuntimeError"
    assert "synthetic sensitive-looking failure text" not in json.dumps(failure)


def _attempt_report(
    plan: Mapping[str, Any],
    *,
    attempt_id: str,
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return with_manifest_digest(
        {
            "format_version": DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION,
            "attempt_id": attempt_id,
            "plan_manifest_sha256": plan["manifest_sha256"],
            "candidate_count": len(candidate_records),
            "candidate_records": list(candidate_records),
        }
    )


def _resume_record(plan: Mapping[str, Any], *, ordinal: int, attempt_id: str, accepted: bool) -> dict[str, Any]:
    entry = plan["entries"][ordinal]
    return {
        "attempt_id": attempt_id,
        "candidate_id": f"{attempt_id}:{entry['task_id']}:{entry['sample_id']}",
        "plan_manifest_sha256": plan["manifest_sha256"],
        "plan_entry_id": entry["plan_entry_id"],
        "ordinal": ordinal,
        "task_id": entry["task_id"],
        "sample_id": entry["sample_id"],
        "record_ref": f"candidates/{ordinal:05d}/record.json",
        "record_file_sha256": "1" * 64,
        "record_manifest_sha256": "2" * 64,
        "selection_tags": ["accepted"] if accepted else [],
        "metadata": {"accepted": accepted},
    }


def test_resume_distinguishes_stable_plan_entries_from_retry_attempts():
    dataset_manifest = _dataset_manifest()
    plan = _plan(dataset_manifest)
    first = _attempt_report(
        plan,
        attempt_id="attempt-e",
        candidate_records=[_resume_record(plan, ordinal=0, attempt_id="attempt-e", accepted=False)],
    )
    second = _attempt_report(
        plan,
        attempt_id="attempt-a",
        candidate_records=[_resume_record(plan, ordinal=0, attempt_id="attempt-a", accepted=True)],
    )

    resume = build_resume_manifest(
        plan,
        [first, second],
        canonical_eligible=lambda record: record["metadata"]["accepted"] is True,
    )

    verify_manifest_digest(resume)
    assert resume["canonical_count"] == 1
    assert resume["unresolved_count"] == 1
    assert resume["canonical"][0]["attempt_id"] == "attempt-a"
    assert resume["unresolved"][0]["ordinal"] == 1

    duplicate = _attempt_report(
        plan,
        attempt_id="attempt-duplicate",
        candidate_records=[_resume_record(plan, ordinal=0, attempt_id="attempt-duplicate", accepted=True)],
    )
    with pytest.raises(RuntimeError, match="completed candidate was retried"):
        build_resume_manifest(
            plan,
            [first, second, duplicate],
            canonical_eligible=lambda record: record["metadata"]["accepted"] is True,
        )


def test_resume_rejects_unknown_plan_entries_and_inconsistent_attempt_counts():
    dataset_manifest = _dataset_manifest()
    plan = _plan(dataset_manifest)
    unknown = _resume_record(plan, ordinal=0, attempt_id="attempt-unknown", accepted=True)
    unknown["plan_entry_id"] = "f" * 64
    report = _attempt_report(plan, attempt_id="attempt-unknown", candidate_records=[unknown])
    with pytest.raises(ValueError, match="unknown candidate plan entry"):
        build_resume_manifest(plan, [report], canonical_eligible=lambda record: True)

    inconsistent = _attempt_report(plan, attempt_id="attempt-count", candidate_records=[])
    inconsistent["candidate_count"] = 1
    inconsistent.pop("manifest_sha256")
    inconsistent = with_manifest_digest(inconsistent)
    with pytest.raises(ValueError, match="candidate_count does not match"):
        build_resume_manifest(plan, [inconsistent], canonical_eligible=lambda record: True)


def test_sensitive_run_identity_fails_before_interaction(tmp_path):
    dataset_manifest = _dataset_manifest()
    plan = _plan(dataset_manifest)
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="sensitive key"):
        run_data_construction_attempt(
            adapter,
            artifact_dir=tmp_path / "never-created",
            attempt_id="attempt-sensitive",
            task_dataset={"type": "text_only", "instances": []},
            task_dataset_manifest=dataset_manifest,
            candidate_plan=plan,
            run_identity={"provider": {"api_key": "must-not-persist"}},
        )
    assert adapter.interactions == []
    assert not (tmp_path / "never-created").exists()
