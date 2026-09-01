"""Sealed AppWorld Train difficulty-1/2 data-construction planning.

The plan composes the immutable 24-candidate scenario-curriculum aggregate
with a frozen primary/fallback schedule for the remaining 60 tasks. It never
executes a provider request and keeps legacy source artifacts in place.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from lmflow.agentic.appworld_episode import appworld_artifact_to_semantic_conversation
from lmflow.agentic.appworld_protocol import (
    APPWORLD_REVISION,
    APPWORLD_SCENARIO_CURRICULUM_TASK_IDS,
    APPWORLD_SCENARIO_CURRICULUM_TASK_SET_SHA256,
    APPWORLD_TRAIN_D1_D2_EXPANSION_TASK_IDS,
    APPWORLD_TRAIN_D1_D2_TASK_IDS,
    load_pinned_appworld_train_d1_d2_dataset,
    load_pinned_appworld_train_d1_d2_expansion_dataset,
)
from lmflow.agentic.data_construction import (
    CandidatePlanEntry,
    build_candidate_plan,
    build_selection_manifest,
    file_sha256,
    json_copy,
    verify_artifact_manifest,
    verify_manifest_digest,
    with_manifest_digest,
    write_artifact_manifest,
)
from lmflow.agentic.scaffolds.appworld_react_code import APPWORLD_REACT_CODE_SCAFFOLD

APPWORLD_TRAIN_D1_D2_PLAN_FORMAT_VERSION = "lmflow.appworld-train-d1-d2-plan/v1"
APPWORLD_INHERITED_SOURCE_FORMAT_VERSION = "lmflow.appworld-inherited-source/v1"
APPWORLD_GENERATION_POLICY_FORMAT_VERSION = "lmflow.appworld-generation-policy/v1"
APPWORLD_PRESFT_PAIRED_BASELINE_PLAN_FORMAT_VERSION = "lmflow.appworld-presft-paired-baseline-plan/v1"
APPWORLD_PRESFT_PAIRED_BASELINE_METRICS_FORMAT_VERSION = "lmflow.appworld-presft-paired-metrics/v1"
APPWORLD_PRESFT_PAIRED_BASELINE_RUN_SHEET_FORMAT_VERSION = "lmflow.appworld-presft-paired-run-sheet/v1"
APPWORLD_L0_AGGREGATE_IDENTITY = {
    "aggregate_report_manifest_sha256": "003effa7af453493377f38ccc7b0aa8e7146a856c2eecbabbc41acb5bfff7c66",
    "candidate_matrix_manifest_sha256": "8111cdc1b50296e42e1c31ec40df73bf8b182ff3069ee0763d7598631a40deb5",
    "student_token_audit_manifest_sha256": "7aaccd7a56aef53d2449a197ad5195ee48e94d717ec76d136a4c0a30781c3fec",
    "success_only_selection_manifest_sha256": "85cf3f0bdc5de94ecfc420e678d14826481fff4c8580ee0a27739c68814c9ac0",
    "success_plus_recovery_selection_manifest_sha256": (
        "10cacf47b5dc20baae3d9883ced2c430bd520b4b9840d2da4b7c4c83402a23a3"
    ),
    "attempts_manifest_sha256": "c94a577c61fdaa54e36270e2bbd3180213093ee62394cbbce81ee70d2080c5c5",
    "artifact_manifest_file_sha256": "a62e3146e4369aae62112d05f3e160a38fff6ef6b65c55044ada61dd7d41aab7",
}
APPWORLD_L1_MODEL_IDENTITY = {
    "model_code": "ZHIPU/GLM-5.3-Flash",
    "model_revision": "provider-model-id-contract-2026-09-01",
    "thinking": {"enabled": True, "reasoning_effort": "max"},
}
APPWORLD_L1_SAMPLING_IDENTITY = {
    "temperature": 0.2,
    "max_completion_tokens": 3000,
    "max_steps": 50,
    "transparent_api_retry": False,
    "primary_seed": 100,
    "fallback_seed": 101,
}
APPWORLD_L1_FALLBACK_TRIGGERS = (
    "data_class_C",
    "data_class_D",
    "data_class_E",
    "contract_invalid_response",
    "replay_match_not_true",
    "collateral_invariant_not_true",
    "canonical_admission_missing",
)
APPWORLD_PRESFT_STUDENT_ARMS = (
    {
        "arm_id": "qwen3-4b-starting",
        "lineage_label": "Starting checkpoint",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    },
    {
        "arm_id": "qwen3-4b-base",
        "lineage_label": "Base",
        "model_id": "Qwen/Qwen3-4B-Base",
        "model_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "tokenizer_revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "tokenizer_json_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "tokenizer_config_sha256": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
        "chat_template_sha256": "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
    },
)


def _planning_implementation_identity(code_base_commit: str) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    source_paths = (
        module_root / "appworld_data_plan.py",
        module_root / "appworld_data_factory.py",
        module_root / "appworld_episode.py",
        module_root / "appworld_protocol.py",
        module_root / "data_construction.py",
        module_root / "scaffolds" / "appworld_react_code" / "scaffold.py",
    )
    return {
        "code_base_commit": code_base_commit,
        "execution_code_commit": "TBD-BLOCKING after product-code merge",
        "planning_source_files": [
            {"path": path.relative_to(module_root.parent.parent).as_posix(), "sha256": file_sha256(path)}
            for path in source_paths
        ],
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular non-symlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


def _read_manifest(path: Path, *, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    value = _read_json_object(path)
    verify_manifest_digest(value)
    if expected_manifest_sha256 is not None and value["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError(f"manifest identity mismatch: {path.name}")
    return value


def _resolve_source_ref(source_root: Path, reference: Any) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("source artifact reference must be a non-empty string")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"unsafe source artifact reference: {reference!r}")
    path = source_root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source artifact reference is missing or unsafe: {reference!r}")
    return path


def _new_json_file(path: Path, value: Mapping[str, Any]) -> None:
    payload = json_copy(value, name=str(path))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _validate_aggregate(aggregate_root: Path) -> dict[str, dict[str, Any]]:
    verification = verify_artifact_manifest(aggregate_root)
    if verification["manifest_file_sha256"] != APPWORLD_L0_AGGREGATE_IDENTITY["artifact_manifest_file_sha256"]:
        raise ValueError("sealed L0 aggregate artifact-manifest identity changed")
    files = {
        "aggregate_report": _read_manifest(
            aggregate_root / "aggregate-report.json",
            expected_manifest_sha256=APPWORLD_L0_AGGREGATE_IDENTITY["aggregate_report_manifest_sha256"],
        ),
        "candidate_matrix": _read_manifest(
            aggregate_root / "candidate-matrix.json",
            expected_manifest_sha256=APPWORLD_L0_AGGREGATE_IDENTITY["candidate_matrix_manifest_sha256"],
        ),
        "student_token_audit": _read_manifest(
            aggregate_root / "student-token-audit.json",
            expected_manifest_sha256=APPWORLD_L0_AGGREGATE_IDENTITY["student_token_audit_manifest_sha256"],
        ),
        "success_only": _read_manifest(
            aggregate_root / "selection-success-only.json",
            expected_manifest_sha256=APPWORLD_L0_AGGREGATE_IDENTITY["success_only_selection_manifest_sha256"],
        ),
        "success_plus_recovery": _read_manifest(
            aggregate_root / "selection-success-plus-recovery.json",
            expected_manifest_sha256=(
                APPWORLD_L0_AGGREGATE_IDENTITY["success_plus_recovery_selection_manifest_sha256"]
            ),
        ),
        "attempts": _read_manifest(
            aggregate_root / "attempts.json",
            expected_manifest_sha256=APPWORLD_L0_AGGREGATE_IDENTITY["attempts_manifest_sha256"],
        ),
    }
    report = files["aggregate_report"]
    if report.get("appworld_revision") != APPWORLD_REVISION:
        raise ValueError("sealed L0 aggregate AppWorld revision changed")
    if report.get("ordered_task_ids") != list(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS):
        raise ValueError("sealed L0 aggregate task set changed")
    if (
        report.get("canonical_complete_trajectory_count") != 24
        or report.get("quality_gate", {}).get("passed") is not True
    ):
        raise ValueError("sealed L0 aggregate quality gate is not reusable")
    if len(files["candidate_matrix"].get("rows", [])) != 24:
        raise ValueError("sealed L0 candidate matrix must contain exactly 24 rows")
    return files


def _validate_source_artifact(
    *,
    source_root: Path,
    row: Mapping[str, Any],
    plan_entry: Mapping[str, Any],
) -> dict[str, Any]:
    seed = row.get("seed")
    task_id = row.get("task_id")
    record_path = source_root / "candidates" / str(task_id) / f"seed-{seed}" / "record.json"
    record = _read_manifest(record_path, expected_manifest_sha256=str(row.get("record_manifest_sha256")))
    for key in ("candidate_id", "ordinal", "task_id", "seed"):
        if record.get(key) != row.get(key):
            raise ValueError(f"sealed L0 source record {key} differs from the candidate matrix")

    artifacts = {}
    for name in ("trajectory", "replay", "admission"):
        path = _resolve_source_ref(source_root, record.get(f"{name}_ref"))
        if file_sha256(path) != record.get(f"{name}_file_sha256"):
            raise ValueError(f"sealed L0 {name} file digest mismatch: {record['candidate_id']}")
        artifacts[name] = _read_manifest(path, expected_manifest_sha256=record.get(f"{name}_manifest_sha256"))
    admission = artifacts["admission"]
    if not (
        row.get("official_success") is True
        and admission.get("replay_match") is True
        and admission.get("collateral_invariant_passed") is True
        and admission.get("hidden_verifier_material_included") is False
        and admission.get("truncated") is False
        and admission.get("data_class") in {"A", "B"}
    ):
        raise ValueError(f"sealed L0 candidate is not reusable A/B evidence: {record['candidate_id']}")

    conversation_path = _resolve_source_ref(source_root, record.get("conversation_ref"))
    if file_sha256(conversation_path) != row.get("conversation_file_sha256"):
        raise ValueError(f"sealed L0 conversation digest mismatch: {record['candidate_id']}")
    conversation = _read_json_object(conversation_path)
    instances = conversation.get("instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise ValueError("sealed L0 conversation must contain exactly one instance")
    metadata = instances[0].get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("hidden_verifier_material_included") is not False:
        raise ValueError("sealed L0 conversation metadata is unsafe")
    derived = appworld_artifact_to_semantic_conversation(artifacts["trajectory"])
    if instances[0].get("messages") != derived["instances"][0]["messages"]:
        raise ValueError(f"sealed L0 semantic conversation drift: {record['candidate_id']}")
    official_evaluation = artifacts["trajectory"].get("official_evaluation")
    if not isinstance(official_evaluation, Mapping):
        raise ValueError(f"sealed L0 trajectory has no official evaluation: {record['candidate_id']}")
    official_test_count = official_evaluation.get("num_tests")
    official_test_pass_count = len(official_evaluation.get("passes", []))
    official_test_failure_count = len(official_evaluation.get("failures", []))
    if (
        not isinstance(official_test_count, int)
        or official_test_count < 1
        or official_test_pass_count + official_test_failure_count != official_test_count
    ):
        raise ValueError(f"sealed L0 official test counts are invalid: {record['candidate_id']}")

    data_class = str(admission["data_class"])
    tags = ["inherited", "success-plus-recovery"]
    if data_class == "A":
        tags.append("success-only")
    if row.get("fit_16k") is True:
        tags.append("short-16k")
    if row.get("fit_32k") is False:
        tags.append("over-32k")
    return {
        "attempt_id": source_root.name,
        "candidate_id": record["candidate_id"],
        "plan_manifest_sha256": plan_entry["plan_manifest_sha256"],
        "plan_entry_id": plan_entry["plan_entry_id"],
        "ordinal": record["ordinal"],
        "task_id": record["task_id"],
        "sample_id": f"seed-{record['seed']}",
        "record_ref": f"{source_root.name}/{record_path.relative_to(source_root).as_posix()}",
        "record_file_sha256": file_sha256(record_path),
        "record_manifest_sha256": record["manifest_sha256"],
        "selection_tags": tags,
        "metadata": {
            **json_copy(row, name="candidate matrix row"),
            "official_test_count": official_test_count,
            "official_test_pass_count": official_test_pass_count,
            "official_test_pass_fraction": official_test_pass_count / official_test_count,
            "source_root_id": source_root.name,
            "source_report_ref": f"{source_root.name}/report.json",
            "conversation_ref": f"{source_root.name}/{conversation_path.relative_to(source_root).as_posix()}",
            "trajectory_ref": f"{source_root.name}/{record['trajectory_ref']}",
            "replay_ref": f"{source_root.name}/{record['replay_ref']}",
            "admission_ref": f"{source_root.name}/{record['admission_ref']}",
        },
    }


def build_appworld_inherited_selections(candidate_records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build immutable L0 views without assigning cross-benchmark quality meaning."""

    def selection(
        policy_id: str,
        *,
        eligible: Any,
        dedup_key: Any,
        rank_key: Any,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        return build_selection_manifest(
            candidate_records,
            policy_id=policy_id,
            policy_identity={"benchmark": "appworld", **identity},
            eligible=eligible,
            dedup_key=dedup_key,
            rank_key=rank_key,
            output_order_key=lambda record: record["ordinal"],
        )

    action_dedup = lambda record: [record["task_id"], record["metadata"]["action_path_sha256"]]
    standard_rank = lambda record: (
        record["metadata"]["data_class"] != "A",
        record["metadata"]["invalid_actions"],
        record["metadata"]["provider_total_tokens"],
        record["metadata"]["seed"],
        record["candidate_id"],
    )
    return {
        "inherited-success-only": selection(
            "inherited-success-only",
            eligible=lambda record: "success-only" in record["selection_tags"],
            dedup_key=action_dedup,
            rank_key=standard_rank,
            identity={"eligible_classes": ["A"], "dedup": ["task_id", "action_path_sha256"]},
        ),
        "inherited-success-plus-recovery": selection(
            "inherited-success-plus-recovery",
            eligible=lambda record: "success-plus-recovery" in record["selection_tags"],
            dedup_key=action_dedup,
            rank_key=standard_rank,
            identity={"eligible_classes": ["A", "B"], "dedup": ["task_id", "action_path_sha256"]},
        ),
        "short-16k-success-only": selection(
            "short-16k-success-only",
            eligible=lambda record: {"success-only", "short-16k"}.issubset(record["selection_tags"]),
            dedup_key=action_dedup,
            rank_key=standard_rank,
            identity={"eligible_classes": ["A"], "maximum_full_sequence_tokens": 16384},
        ),
        "short-16k-success-plus-recovery": selection(
            "short-16k-success-plus-recovery",
            eligible=lambda record: {"success-plus-recovery", "short-16k"}.issubset(record["selection_tags"]),
            dedup_key=action_dedup,
            rank_key=standard_rank,
            identity={"eligible_classes": ["A", "B"], "maximum_full_sequence_tokens": 16384},
        ),
        "over-32k-controlled-prefix-inputs": selection(
            "over-32k-controlled-prefix-inputs",
            eligible=lambda record: "over-32k" in record["selection_tags"],
            dedup_key=lambda record: record["candidate_id"],
            rank_key=lambda record: (record["ordinal"],),
            identity={
                "eligible_classes": ["A", "B"],
                "minimum_full_sequence_tokens_exclusive": 32768,
                "training_eligible": False,
                "projection_status": "blocked_pending_exact_prefix_recipe",
                "silent_truncation_allowed": False,
            },
        ),
        "inherited-task-coverage": selection(
            "inherited-task-coverage",
            eligible=lambda record: "success-plus-recovery" in record["selection_tags"],
            dedup_key=lambda record: record["task_id"],
            rank_key=lambda record: (
                record["metadata"]["data_class"] != "A",
                record["metadata"]["seed"],
                record["candidate_id"],
            ),
            identity={
                "coverage_unit": "task_id",
                "ranking": ["prefer A over B", "prefer lower fixed seed", "candidate identity"],
                "preserve_unselected_source_candidates": True,
            },
        ),
    }


def build_appworld_presft_paired_baseline_artifacts(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    code_commit: str,
) -> dict[str, dict[str, Any]]:
    """Freeze the no-execution teacher/student diagnostic contract."""

    if not isinstance(code_commit, str) or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("code_commit must be a full 40-character Git SHA")
    teacher_records = sorted(
        (record for record in candidate_records if record.get("sample_id") == "seed-100"),
        key=lambda record: record["ordinal"],
    )
    if [record["task_id"] for record in teacher_records] != list(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS):
        raise ValueError("paired baseline teacher rows do not match the pinned L0 task set")

    task_rows = []
    for record in teacher_records:
        metadata = record["metadata"]
        if not (
            metadata.get("official_success") is True
            and metadata.get("replay_match") is True
            and metadata.get("collateral_invariant_passed") is True
            and metadata.get("hidden_verifier_material_included") is False
            and metadata.get("data_class") in {"A", "B"}
            and metadata.get("official_test_pass_fraction") == 1.0
        ):
            raise ValueError(f"paired teacher reference is not sealed successful evidence: {record['candidate_id']}")
        task_rows.append(
            {
                "ordinal": len(task_rows),
                "task_id": record["task_id"],
                "scenario_id": metadata["scenario_id"],
                "difficulty": metadata["difficulty"],
                "seed": 100,
                "teacher_reference": {
                    "candidate_id": record["candidate_id"],
                    "source_root_id": record["attempt_id"],
                    "record_ref": record["record_ref"],
                    "record_file_sha256": record["record_file_sha256"],
                    "record_manifest_sha256": record["record_manifest_sha256"],
                    "trajectory_ref": metadata["trajectory_ref"],
                    "replay_ref": metadata["replay_ref"],
                    "admission_ref": metadata["admission_ref"],
                    "data_class": metadata["data_class"],
                    "official_success": True,
                    "official_test_count": metadata["official_test_count"],
                    "official_test_pass_count": metadata["official_test_pass_count"],
                    "official_test_pass_fraction": metadata["official_test_pass_fraction"],
                    "valid_actions": metadata["valid_actions"],
                    "invalid_actions": metadata["invalid_actions"],
                    "recovery_count": metadata["recovery_count"],
                    "state_change_steps": metadata["state_change_steps"],
                    "duplicate_actions": metadata["duplicate_actions"],
                    "steps": metadata["steps"],
                    "input_tokens": metadata["provider_input_tokens"],
                    "output_tokens": metadata["provider_output_tokens"],
                    "latency_seconds": metadata["latency_seconds"],
                    "conservative_list_price_cost_cny": (
                        metadata["provider_input_tokens"] * 0.8 + metadata["provider_output_tokens"] * 2.8
                    )
                    / 1_000_000,
                },
            }
        )
    teacher_references = [row["teacher_reference"] for row in task_rows]
    selection = with_manifest_digest(
        {
            "format_version": APPWORLD_PRESFT_PAIRED_BASELINE_PLAN_FORMAT_VERSION,
            "source_split": "train",
            "diagnostic_label": "Train/pre-SFT diagnostic",
            "task_count": len(task_rows),
            "scenario_count": 4,
            "ordered_task_ids": list(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS),
            "ordered_task_ids_sha256": APPWORLD_SCENARIO_CURRICULUM_TASK_SET_SHA256,
            "seed": 100,
            "teacher_selection_rule": "reuse the sealed seed-100 canonical candidate for every pinned L0 task",
            "tasks": task_rows,
        }
    )
    teacher_summary = {
        "task_count": len(teacher_references),
        "official_success_count": sum(reference["official_success"] for reference in teacher_references),
        "official_test_pass_fraction_mean": sum(
            reference["official_test_pass_fraction"] for reference in teacher_references
        )
        / len(teacher_references),
        "valid_actions": sum(reference["valid_actions"] for reference in teacher_references),
        "invalid_actions": sum(reference["invalid_actions"] for reference in teacher_references),
        "recoveries": sum(reference["recovery_count"] for reference in teacher_references),
        "state_change_steps": sum(reference["state_change_steps"] for reference in teacher_references),
        "duplicate_actions": sum(reference["duplicate_actions"] for reference in teacher_references),
        "steps": sum(reference["steps"] for reference in teacher_references),
        "input_tokens": sum(reference["input_tokens"] for reference in teacher_references),
        "output_tokens": sum(reference["output_tokens"] for reference in teacher_references),
        "latency_seconds": sum(reference["latency_seconds"] for reference in teacher_references),
        "conservative_list_price_cost_cny": sum(
            reference["conservative_list_price_cost_cny"] for reference in teacher_references
        ),
    }
    metrics = with_manifest_digest(
        {
            "format_version": APPWORLD_PRESFT_PAIRED_BASELINE_METRICS_FORMAT_VERSION,
            "reporting_unit": "one task execution in one lineage arm",
            "identity_fields": [
                "run_id",
                "arm_id",
                "model_id",
                "model_revision",
                "tokenizer_revision_and_digests",
                "appworld_revision",
                "task_id",
                "scenario_id",
                "difficulty",
                "scaffold_and_projector_identity",
                "backend_and_context_profile",
                "sampling_and_termination_profile",
            ],
            "metric_groups": {
                "official_outcome": [
                    "official_success",
                    "official_test_pass_count",
                    "official_test_count",
                    "official_test_pass_fraction",
                    "official_evaluator_error",
                    "replay_match",
                    "collateral_invariant_passed",
                ],
                "execution_behavior": [
                    "valid_actions",
                    "invalid_actions",
                    "recovery_count",
                    "state_change_steps",
                    "duplicate_actions",
                    "repetition_rate",
                    "context_overflow_count",
                    "backend_error_count",
                    "termination_reason",
                    "failure_type",
                ],
                "work_and_latency": [
                    "steps",
                    "model_calls",
                    "input_tokens",
                    "output_tokens",
                    "model_latency_seconds",
                    "environment_latency_seconds",
                    "evaluation_latency_seconds",
                    "total_latency_seconds",
                ],
                "resource_and_cost": [
                    "teacher_api_cost_cny",
                    "student_gpu_active_hours",
                    "student_control_plane_hours",
                    "student_nominal_gpu_cost_cny",
                    "student_control_plane_billed_delta_cny",
                ],
            },
            "aggregation": {
                "required": ["per-task rows", "per-arm totals", "per-arm rates", "paired exact task deltas"],
                "small_sample_claim": "descriptive Train diagnostic only; no significance or held-out claim",
                "missing_values": "explicit null plus typed failure reason; never infer from HTTP or service logs",
            },
        }
    )
    common_student_profile = {
        "temperature": 0.2,
        "seed": 100,
        "max_completion_tokens": 3000,
        "max_steps": 50,
        "maximum_task_wall_seconds": 1200,
        "fresh_reset_per_task": True,
        "fresh_reset_replay": True,
        "official_evaluator_isolated": True,
        "collateral_invariant_required": True,
        "thinking": {"enabled": False, "transport": "chat_template_kwargs.enable_thinking"},
    }
    plan = with_manifest_digest(
        {
            "format_version": APPWORLD_PRESFT_PAIRED_BASELINE_PLAN_FORMAT_VERSION,
            "implementation": _planning_implementation_identity(code_commit),
            "authorization": {
                "gpu_or_cloud": "NOT_AUTHORIZED_FOR_THIS_RUN",
                "paid_api": "NOT_REQUIRED; teacher artifacts are reused",
                "execution_allowed": False,
            },
            "objective": "make the sealed L0 teacher-to-student pre-SFT gap directly auditable",
            "prohibited_claims": [
                "held-out benchmark score",
                "model selection",
                "SFT recipe tuning signal",
                "teacher gap alone proves RL is necessary",
            ],
            "task_selection_manifest_sha256": selection["manifest_sha256"],
            "metrics_manifest_sha256": metrics["manifest_sha256"],
            "teacher_reference": {
                "arm_id": "glm-5.3-flash-teacher-seed100",
                "model": copy.deepcopy(APPWORLD_L1_MODEL_IDENTITY),
                "task_count": 12,
                "execution": "sealed evidence reuse; no new provider request",
                "summary": teacher_summary,
            },
            "student_arms": [
                {**copy.deepcopy(arm), "execution_profile": copy.deepcopy(common_student_profile)}
                for arm in APPWORLD_PRESFT_STUDENT_ARMS
            ],
            "shared_protocol": {
                "appworld_revision": APPWORLD_REVISION,
                "source_split": "train",
                "task_set_sha256": APPWORLD_SCENARIO_CURRICULUM_TASK_SET_SHA256,
                "scaffold": copy.deepcopy(APPWORLD_REACT_CODE_SCAFFOLD),
                "projector": "lmflow.agentic.scaffolds.appworld_react_code canonical model-visible history",
                "official_evaluator": True,
                "hidden_verifier_material_visible_to_model": False,
            },
            "system_profile_differences": [
                {
                    "field": "backend",
                    "teacher": "Bailian OpenAI-compatible provider",
                    "students": "vLLM 0.25.1 candidate; exact remote runtime identity is TBD-BLOCKING",
                },
                {
                    "field": "thinking",
                    "teacher": "enabled with reasoning_effort=max in sealed L0",
                    "students": "disabled for both Qwen lineages",
                },
                {
                    "field": "tokenizer_and_chat_template",
                    "teacher": "provider protocol identity",
                    "students": "lineage-specific pinned tokenizer and chat-template digests",
                },
                {
                    "field": "cost_unit",
                    "teacher": "API token cost already incurred",
                    "students": "GPU active/control-plane hours and nominal/billed cost",
                },
            ],
            "entry_gates_after_diagnostic": {
                "SFT_motivation_only": True,
                "GRPO_requires_heldout_SFT_improvement": True,
                "GRPO_requires_train_K_group_reward_variance": True,
            },
        }
    )
    run_sheet = with_manifest_digest(
        {
            "format_version": APPWORLD_PRESFT_PAIRED_BASELINE_RUN_SHEET_FORMAT_VERSION,
            "plan_manifest_sha256": plan["manifest_sha256"],
            "status": "FROZEN_PREPARATION; DO_NOT_START_GPU_OR_CLOUD",
            "executor": "GPU/cloud management task 019feb6f-d996-71f1-8dea-d5b242c6102a",
            "ordered_stages": [
                "local bundle strict-set and SHA validation",
                "control-plane, mount, GPU, disk, nofile, environment and model identity preflight",
                "CPU-only task/scaffold/projector and artifact-contract tests",
                "Qwen3-4B Starting: one readiness task then remaining 11 tasks if gates pass",
                "stop service and prove GPU empty",
                "Qwen3-4B-Base: one readiness task then remaining 11 tasks if gates pass",
                "fresh-reset replay and isolated official evaluation for every task",
                "sanitized return manifest, GPU empty proof, control-plane shutdown",
            ],
            "gates": {
                "identity": "all code/model/tokenizer/data/scaffold/projector/runtime digests exact",
                "readiness_task": (
                    "artifact complete, parser path valid, reset/replay/evaluator executable; quality may fail"
                ),
                "per_task": "raw request/response before normalization/assert; step and episode evidence atomic",
                "quality": "never stops preservation; failures remain typed rows",
                "expansion": "exactly 12 tasks per student arm; no Dev9, extra model, seed, SFT or GRPO",
            },
            "fail_closed": {
                "identity_or_contract_failure": "stop before model load or next task; preserve raw evidence",
                "runtime_or_context_failure": "record typed system failure; do not change profile within the run",
                "official_replay_failure": (
                    "preserve environment evidence; do not advance that arm without owner review"
                ),
                "hard_deadline": "control-plane shutdown takes precedence over complete return",
            },
            "budget_recommendation": {
                "gpu": "one RTX 4090 24GB, one model service at a time",
                "maximum_control_plane_hours": 9.0,
                "assumed_price_cny_per_hour": 1.68,
                "nominal_maximum_cny": 15.12,
                "recommended_hard_amount_cny": 16.0,
                "stop_new_episode_reserve_minutes": 30,
                "price_and_available_budget_must_be_reconfirmed": True,
            },
            "TBD_BLOCKING": [
                "exact execution commit and clean/dirty bundle identity",
                "student model weight file manifests on the persistent mount",
                "remote Python/torch/transformers/vLLM/AppWorld environment manifest",
                "remote runner/config/template bundle digests",
                "current GPU SKU, price, mount, available budget and control-plane deadlines",
            ],
        }
    )
    return {
        "plan": plan,
        "task_selection": selection,
        "metrics_schema": metrics,
        "run_sheet": run_sheet,
    }


def _source_roots_manifest(
    candidate_records: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, Any],
    artifact_parent: Path,
) -> dict[str, Any]:
    attempt_rows = attempts.get("attempts")
    if not isinstance(attempt_rows, list):
        raise ValueError("sealed L0 attempts manifest is malformed")
    attempts_by_id = {row["run_id"]: row for row in attempt_rows}
    ordinals_by_root: dict[str, list[int]] = defaultdict(list)
    for record in candidate_records:
        ordinals_by_root[record["attempt_id"]].append(record["ordinal"])
    roots = []
    for source_root_id in sorted(ordinals_by_root):
        source_root = artifact_parent / source_root_id
        report_path = source_root / "report.json"
        attempt = attempts_by_id.get(source_root_id)
        if not isinstance(attempt, Mapping):
            raise ValueError(f"canonical source root is absent from the attempts manifest: {source_root_id}")
        evidence_manifest_sha256 = attempt.get("evidence_manifest_sha256")
        if not isinstance(evidence_manifest_sha256, str):
            raise ValueError(f"canonical source attempt has no evidence identity: {source_root_id}")
        if report_path.is_file() and not report_path.is_symlink():
            terminal_path = report_path
            terminal = _read_manifest(terminal_path)
            terminal_kind = "report"
            if terminal["manifest_sha256"] != evidence_manifest_sha256:
                raise ValueError(f"canonical source report differs from attempts evidence: {source_root_id}")
        else:
            checkpoint_paths = sorted((source_root / "control").glob("checkpoint-*.json"))
            matching_checkpoints = []
            for checkpoint_path in checkpoint_paths:
                checkpoint = _read_manifest(checkpoint_path)
                if checkpoint["manifest_sha256"] == evidence_manifest_sha256:
                    matching_checkpoints.append((checkpoint_path, checkpoint))
            if len(matching_checkpoints) != 1:
                raise ValueError(
                    f"canonical source without a report must have one matching checkpoint: {source_root_id}"
                )
            terminal_path, terminal = matching_checkpoints[0]
            terminal_kind = "checkpoint_after_report_publication_failure"
            if attempt.get("stop_reason") != "report_publication_failed":
                raise ValueError(f"canonical source report is missing without a publication failure: {source_root_id}")
        roots.append(
            {
                "source_root_id": source_root_id,
                "source_root_resolution": "sibling of the sealed L0 aggregate; absolute path not persisted",
                "terminal_evidence_kind": terminal_kind,
                "terminal_evidence_ref": terminal_path.relative_to(source_root).as_posix(),
                "terminal_evidence_file_sha256": file_sha256(terminal_path),
                "terminal_evidence_manifest_sha256": terminal["manifest_sha256"],
                "attempt_stop_reason": attempt.get("stop_reason"),
                "canonical_ordinals": sorted(ordinals_by_root[source_root_id]),
            }
        )
    return with_manifest_digest(
        {
            "format_version": APPWORLD_INHERITED_SOURCE_FORMAT_VERSION,
            "source_root_count": len(roots),
            "sources": roots,
        }
    )


def _generation_policy(candidate_plan: Mapping[str, Any]) -> dict[str, Any]:
    primary_ordinals = [entry["ordinal"] for entry in candidate_plan["entries"] if entry["sample_id"] == "seed-100"]
    fallback_ordinals = [entry["ordinal"] for entry in candidate_plan["entries"] if entry["sample_id"] == "seed-101"]
    return with_manifest_digest(
        {
            "format_version": APPWORLD_GENERATION_POLICY_FORMAT_VERSION,
            "candidate_plan_manifest_sha256": candidate_plan["manifest_sha256"],
            "task_count": len(APPWORLD_TRAIN_D1_D2_EXPANSION_TASK_IDS),
            "primary": {
                "seed": 100,
                "scheduled_ordinals": primary_ordinals,
                "candidate_count": len(primary_ordinals),
            },
            "fallback": {
                "seed": 101,
                "candidate_ordinals": fallback_ordinals,
                "candidate_count": len(fallback_ordinals),
                "trigger_any": list(APPWORLD_L1_FALLBACK_TRIGGERS),
                "A_or_B_primary_disables_fallback": True,
            },
            "attempt_semantics": {
                "pre_request_infrastructure_recovery_may_reuse_logical_slot": True,
                "post_response_primary_slot_reexecution": False,
                "multiple_canonical_eligible_executions_per_slot": False,
                "contract_invalid_or_noncanonical_primary_enables_seed101_slot": True,
            },
            "B_policy": {
                "meets_verified_task_coverage": True,
                "automatic_extra_call_for_A": False,
                "retain_optional_post_batch_arm": True,
            },
        }
    )


def build_appworld_train_d1_d2_plan_bundle(
    *,
    artifact_dir: str | os.PathLike[str],
    appworld_root: str | os.PathLike[str],
    l0_aggregate_dir: str | os.PathLike[str],
    code_commit: str,
) -> dict[str, Any]:
    """Validate inherited evidence and atomically publish a no-request plan."""

    if not isinstance(code_commit, str) or re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("code_commit must be a full 40-character Git SHA")
    target = Path(artifact_dir)
    if target.exists() or os.path.lexists(target):
        raise FileExistsError(f"artifact directory already exists: {target}")
    aggregate_root = Path(l0_aggregate_dir).resolve()
    if not aggregate_root.is_dir() or aggregate_root.is_symlink():
        raise ValueError("l0_aggregate_dir must be a non-symlink directory")

    aggregate = _validate_aggregate(aggregate_root)
    full_dataset, full_dataset_manifest = load_pinned_appworld_train_d1_d2_dataset(appworld_root=appworld_root)
    expansion_dataset, expansion_dataset_manifest = load_pinned_appworld_train_d1_d2_expansion_dataset(
        appworld_root=appworld_root
    )
    inherited_entries = tuple(
        CandidatePlanEntry(
            ordinal=ordinal,
            task_id=task_id,
            sample_id=f"seed-{seed}",
            sampling={"seed": seed, "source": "sealed-l0-inherited"},
        )
        for ordinal, (task_id, seed) in enumerate(
            (task_id, seed) for task_id in APPWORLD_SCENARIO_CURRICULUM_TASK_IDS for seed in (100, 101)
        )
    )
    inherited_plan = build_candidate_plan(
        inherited_entries,
        task_set_id="appworld_sealed_scenario_curriculum_v1",
        task_dataset_manifest_sha256=full_dataset_manifest["manifest_sha256"],
    )
    inherited_entry_by_ordinal = {
        entry["ordinal"]: {**entry, "plan_manifest_sha256": inherited_plan["manifest_sha256"]}
        for entry in inherited_plan["entries"]
    }
    rows = sorted(aggregate["candidate_matrix"]["rows"], key=lambda row: row["ordinal"])
    if [(row["task_id"], row["seed"]) for row in rows] != [
        (entry.task_id, entry.sampling["seed"]) for entry in inherited_entries
    ]:
        raise ValueError("sealed L0 candidate matrix does not match the inherited plan")
    candidate_records = []
    for row in rows:
        source_root = aggregate_root.parent / row["source_run_id"]
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError(f"sealed L0 source root is missing or unsafe: {row['source_run_id']}")
        candidate_records.append(
            _validate_source_artifact(
                source_root=source_root,
                row=row,
                plan_entry=inherited_entry_by_ordinal[row["ordinal"]],
            )
        )
    selections = build_appworld_inherited_selections(candidate_records)
    expected_counts = {
        "inherited-success-only": 22,
        "inherited-success-plus-recovery": 24,
        "short-16k-success-only": 21,
        "short-16k-success-plus-recovery": 22,
        "over-32k-controlled-prefix-inputs": 2,
        "inherited-task-coverage": 12,
    }
    for policy_id, expected_count in expected_counts.items():
        if selections[policy_id]["selected_count"] != expected_count:
            raise ValueError(f"sealed L0 selection count changed for {policy_id}")

    expansion_entries = tuple(
        CandidatePlanEntry(
            ordinal=ordinal,
            task_id=task_id,
            sample_id=f"seed-{seed}",
            sampling={
                "seed": seed,
                "temperature": APPWORLD_L1_SAMPLING_IDENTITY["temperature"],
                "max_completion_tokens": APPWORLD_L1_SAMPLING_IDENTITY["max_completion_tokens"],
                "extra_body": {"enable_thinking": True, "reasoning_effort": "max"},
            },
        )
        for ordinal, (task_id, seed) in enumerate(
            (task_id, seed) for task_id in APPWORLD_TRAIN_D1_D2_EXPANSION_TASK_IDS for seed in (100, 101)
        )
    )
    generation_plan = build_candidate_plan(
        expansion_entries,
        task_set_id="appworld_train_d1_d2_expansion_v1",
        task_dataset_manifest_sha256=expansion_dataset_manifest["manifest_sha256"],
    )
    generation_policy = _generation_policy(generation_plan)
    source_roots = _source_roots_manifest(candidate_records, aggregate["attempts"], aggregate_root.parent)
    paired_baseline = build_appworld_presft_paired_baseline_artifacts(candidate_records, code_commit=code_commit)

    plan = with_manifest_digest(
        {
            "format_version": APPWORLD_TRAIN_D1_D2_PLAN_FORMAT_VERSION,
            "implementation": _planning_implementation_identity(code_commit),
            "appworld_revision": APPWORLD_REVISION,
            "authorization": {
                "paid_api": "NOT_AUTHORIZED",
                "gpu_or_cloud": "NOT_AUTHORIZED",
                "execution_allowed": False,
            },
            "coverage_target": {
                "task_count": len(APPWORLD_TRAIN_D1_D2_TASK_IDS),
                "scenario_count": 24,
                "task_dataset_manifest_sha256": full_dataset_manifest["manifest_sha256"],
                "inherited_task_count": len(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS),
                "generated_task_count": len(APPWORLD_TRAIN_D1_D2_EXPANSION_TASK_IDS),
            },
            "inherited": {
                "candidate_count": len(candidate_records),
                "task_count": len(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS),
                "aggregate_identity": copy.deepcopy(APPWORLD_L0_AGGREGATE_IDENTITY),
                "candidate_plan_manifest_sha256": inherited_plan["manifest_sha256"],
                "source_roots_manifest_sha256": source_roots["manifest_sha256"],
                "reuse_without_generation": True,
                "source_artifacts_rewritten": False,
            },
            "generation": {
                "model": copy.deepcopy(APPWORLD_L1_MODEL_IDENTITY),
                "sampling": copy.deepcopy(APPWORLD_L1_SAMPLING_IDENTITY),
                "candidate_plan_manifest_sha256": generation_plan["manifest_sha256"],
                "generation_policy_manifest_sha256": generation_policy["manifest_sha256"],
                "initial_paid_candidate_count": 60,
                "maximum_fallback_candidate_count": 60,
                "current_price_and_budget": "TBD-BLOCKING before paid authorization",
            },
            "selection_policy": {
                "task_coverage": "A before B, then fixed seed and candidate identity",
                "A_only_view": True,
                "A_plus_B_view": True,
                "preserve_all_inherited_candidates": True,
            },
            "long_context": {
                "short_16k_candidate_count": 22,
                "over_32k_candidate_count": 2,
                "silent_truncation_allowed": False,
                "controlled_prefix_projection": "BLOCKED pending exact-prefix implementation and audit",
            },
            "presft_paired_baseline": {
                "execution_allowed": False,
                "plan_manifest_sha256": paired_baseline["plan"]["manifest_sha256"],
                "task_selection_manifest_sha256": paired_baseline["task_selection"]["manifest_sha256"],
                "metrics_manifest_sha256": paired_baseline["metrics_schema"]["manifest_sha256"],
                "run_sheet_manifest_sha256": paired_baseline["run_sheet"]["manifest_sha256"],
            },
        }
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        _new_json_file(staging / "plan.json", plan)
        _new_json_file(staging / "task-dataset.json", full_dataset.to_dict())
        _new_json_file(staging / "task-dataset-manifest.json", full_dataset_manifest)
        _new_json_file(staging / "expansion-dataset.json", expansion_dataset.to_dict())
        _new_json_file(staging / "expansion-dataset-manifest.json", expansion_dataset_manifest)
        _new_json_file(staging / "inherited-candidate-plan.json", inherited_plan)
        _new_json_file(
            staging / "inherited-candidate-index.json",
            with_manifest_digest(
                {
                    "format_version": APPWORLD_INHERITED_SOURCE_FORMAT_VERSION,
                    "candidate_count": len(candidate_records),
                    "candidate_records": candidate_records,
                }
            ),
        )
        _new_json_file(staging / "inherited-source-roots.json", source_roots)
        _new_json_file(staging / "generation-candidate-plan.json", generation_plan)
        _new_json_file(staging / "generation-policy.json", generation_policy)
        for policy_id, selection in selections.items():
            _new_json_file(staging / "selections" / f"{policy_id}.json", selection)
        _new_json_file(staging / "presft-paired-baseline" / "plan.json", paired_baseline["plan"])
        _new_json_file(
            staging / "presft-paired-baseline" / "task-selection.json",
            paired_baseline["task_selection"],
        )
        _new_json_file(
            staging / "presft-paired-baseline" / "metrics-schema.json",
            paired_baseline["metrics_schema"],
        )
        _new_json_file(staging / "presft-paired-baseline" / "run-sheet.json", paired_baseline["run_sheet"])
        write_artifact_manifest(staging)
        verification = verify_artifact_manifest(staging)
        staging.rename(target)
    except BaseException as error:
        if staging.exists():
            failed = target.with_name(f"{target.name}.failed-{staging.name.removeprefix('.').removesuffix('.tmp')}")
            staging.rename(failed)
            error.add_note(f"partial AppWorld plan evidence moved to {failed}")
        raise
    return {
        "plan": copy.deepcopy(plan),
        "artifact_file_count": verification["file_count"],
        "artifact_manifest_file_sha256": verification["manifest_file_sha256"],
    }


__all__ = [
    "APPWORLD_GENERATION_POLICY_FORMAT_VERSION",
    "APPWORLD_INHERITED_SOURCE_FORMAT_VERSION",
    "APPWORLD_L0_AGGREGATE_IDENTITY",
    "APPWORLD_L1_FALLBACK_TRIGGERS",
    "APPWORLD_L1_MODEL_IDENTITY",
    "APPWORLD_L1_SAMPLING_IDENTITY",
    "APPWORLD_PRESFT_PAIRED_BASELINE_METRICS_FORMAT_VERSION",
    "APPWORLD_PRESFT_PAIRED_BASELINE_PLAN_FORMAT_VERSION",
    "APPWORLD_PRESFT_PAIRED_BASELINE_RUN_SHEET_FORMAT_VERSION",
    "APPWORLD_PRESFT_STUDENT_ARMS",
    "APPWORLD_TRAIN_D1_D2_PLAN_FORMAT_VERSION",
    "build_appworld_inherited_selections",
    "build_appworld_presft_paired_baseline_artifacts",
    "build_appworld_train_d1_d2_plan_bundle",
]
