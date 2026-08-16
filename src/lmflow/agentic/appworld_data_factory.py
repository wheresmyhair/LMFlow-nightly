"""Benchmark-local admission and projection gates for AppWorld cold-start data."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lmflow.agentic.appworld_episode import (
    APPWORLD_REPLAY_FORMAT_VERSION,
    AppWorldEpisodeResult,
    replay_appworld_episode,
    run_appworld_episode,
)
from lmflow.agentic.appworld_protocol import (
    APPWORLD_DATA_PILOT_TASK_IDS,
    APPWORLD_REVISION,
    canonical_json_sha256,
    load_pinned_appworld_data_pilot_dataset,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.agentic.completion import CompletionBackend

APPWORLD_DATA_CANDIDATE_FORMAT_VERSION = "lmflow.appworld-data-candidate/v1"
APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION = "lmflow.appworld-data-factory-run/v1"
APPWORLD_DATA_CLASSES = ("A", "B", "C", "D", "E")


@dataclass
class AppWorldFinalizedCandidate:
    """A sealed admission record and an optional SFT projection."""

    admission: dict[str, Any]
    training_projection: dict[str, Any] | None


def _nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _candidate_class(artifact: Mapping[str, Any], replay: Mapping[str, Any]) -> str:
    metrics = artifact["metrics"]
    if artifact.get("runner_error") is not None or artifact.get("evaluator_error") is not None:
        return "E"
    if replay.get("replay_error") is not None or replay.get("replay_match") is not True:
        return "E"
    official_success = metrics.get("success") is True
    collateral_verified = replay.get("collateral_invariant_passed") is True
    invalid_actions = metrics.get("invalid_tool_calls", 0)
    recovery_count = metrics.get("recovery_count", 0)
    if official_success and collateral_verified and invalid_actions == 0:
        return "A"
    if official_success and collateral_verified and invalid_actions > 0 and recovery_count > 0:
        return "B"
    if not official_success and replay.get("sealed_partial_signal") is True:
        return "C"
    return "D"


def _align_action_loss_mask(
    training_projection: dict[str, Any],
    action_steps: Sequence[Mapping[str, Any]],
) -> None:
    instances = training_projection.get("instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise ValueError("AppWorld training projection must contain exactly one instance")
    messages = instances[0].get("messages")
    if not isinstance(messages, list):
        raise ValueError("AppWorld training projection messages are missing")
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if len(assistant_messages) < len(action_steps):
        raise ValueError("AppWorld training projection is missing assistant action messages")
    if not action_steps:
        return
    for message, action_step in zip(assistant_messages[-len(action_steps) :], action_steps, strict=True):
        message["loss"] = action_step.get("valid") is True


def _trainable_target_digest(training_projection: Mapping[str, Any]) -> str:
    messages = training_projection["instances"][0]["messages"]
    targets = [
        message["content"] for message in messages if message.get("role") == "assistant" and message.get("loss") is True
    ]
    return canonical_json_sha256(targets)


def _trainable_output_tokens(artifact: Mapping[str, Any]) -> int | None:
    model_steps = artifact.get("model_steps")
    action_steps = artifact.get("action_steps")
    if not isinstance(model_steps, list) or not isinstance(action_steps, list) or len(model_steps) != len(action_steps):
        return None
    values = []
    for model_step, action_step in zip(model_steps, action_steps, strict=True):
        if action_step.get("valid") is not True:
            continue
        usage = model_step.get("usage")
        value = usage.get("output_tokens") if isinstance(usage, Mapping) else None
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        values.append(value)
    return sum(values)


def finalize_appworld_candidate(
    result: AppWorldEpisodeResult,
    replay: Mapping[str, Any],
    *,
    cost_usd: float | None = None,
) -> AppWorldFinalizedCandidate:
    """Classify one executed/replayed candidate and emit only allowed projections."""

    if not isinstance(result, AppWorldEpisodeResult):
        raise TypeError("result must be an AppWorldEpisodeResult")
    verify_manifest_digest(result.artifact)
    if not isinstance(replay, Mapping):
        raise TypeError("replay must be a mapping")
    verify_manifest_digest(replay)
    if replay.get("format_version") != APPWORLD_REPLAY_FORMAT_VERSION:
        raise ValueError("replay is not an AppWorld replay artifact")
    if replay.get("source_artifact_sha256") != result.artifact["manifest_sha256"]:
        raise ValueError("replay source artifact does not match the candidate")
    if cost_usd is not None:
        if isinstance(cost_usd, bool) or not isinstance(cost_usd, int | float):
            raise TypeError("cost_usd must be a number when provided")
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("cost_usd must be finite and non-negative")

    artifact = result.artifact
    metrics = artifact["metrics"]
    data_class = _candidate_class(artifact, replay)
    admitted_for_sft = data_class in {"A", "B"}
    success_only_eligible = data_class == "A"
    success_plus_recovery_eligible = admitted_for_sft
    training_projection = None
    target_digest = None
    trainable_tokens = None
    if admitted_for_sft:
        training_projection = copy.deepcopy(result.training_projection)
        _align_action_loss_mask(training_projection, artifact["action_steps"])
        metadata = training_projection["instances"][0]["metadata"]
        metadata.update(
            {
                "data_class": data_class,
                "replay_required": False,
                "replay_manifest_sha256": replay["manifest_sha256"],
                "collateral_invariant_passed": True,
                "eligible_for_success_only_sft": success_only_eligible,
                "eligible_for_success_plus_recovery_sft": success_plus_recovery_eligible,
                "hidden_verifier_material_included": False,
            }
        )
        target_digest = _trainable_target_digest(training_projection)
        trainable_tokens = _trainable_output_tokens(artifact)

    usage = metrics.get("usage") if isinstance(metrics.get("usage"), Mapping) else {}
    truncated = any(step.get("finish_reason") == "length" for step in artifact.get("model_steps", []))
    collateral_rejected = metrics.get("success") is True and replay.get("collateral_invariant_passed") is not True
    admission = with_manifest_digest(
        {
            "format_version": APPWORLD_DATA_CANDIDATE_FORMAT_VERSION,
            "candidate_id": artifact["trajectory_id"],
            "task_id": artifact["task"]["task_id"],
            "source_artifact_sha256": artifact["manifest_sha256"],
            "replay_manifest_sha256": replay["manifest_sha256"],
            "data_class": data_class,
            "admitted_for_sft": admitted_for_sft,
            "sft_arms": {
                "success_only": success_only_eligible,
                "success_plus_recovery": success_plus_recovery_eligible,
            },
            "preference_eligible": False,
            "preference_reason": "requires a separately verified improved pair under the same task and reset",
            "replay_match": replay.get("replay_match") is True,
            "collateral_invariant_passed": replay.get("collateral_invariant_passed") is True,
            "collateral_rejected": collateral_rejected,
            "duplicate_target_sha256": target_digest,
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "trainable_output_tokens": trainable_tokens,
            },
            "truncated": truncated,
            "cost_usd": cost_usd,
            "failure_type": metrics.get("failure_type"),
            "hidden_verifier_material_included": False,
            "protected_source_artifact_required": True,
        }
    )
    if training_projection is not None:
        training_projection["instances"][0]["metadata"]["admission_manifest_sha256"] = admission["manifest_sha256"]
    return AppWorldFinalizedCandidate(admission=admission, training_projection=training_projection)


def summarize_appworld_candidates(admissions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pilot gates without exposing task verifier material."""

    for admission in admissions:
        if not isinstance(admission, Mapping):
            raise TypeError("every admission must be a mapping")
        verify_manifest_digest(admission)
        if admission.get("format_version") != APPWORLD_DATA_CANDIDATE_FORMAT_VERSION:
            raise ValueError("unexpected AppWorld candidate format")
    class_counts = Counter(admission["data_class"] for admission in admissions)
    admitted = [admission for admission in admissions if admission["admitted_for_sft"] is True]
    target_counts = Counter(
        admission["duplicate_target_sha256"]
        for admission in admitted
        if admission.get("duplicate_target_sha256") is not None
    )
    duplicate_count = sum(count - 1 for count in target_counts.values() if count > 1)
    trainable_tokens = [
        admission["usage"]["trainable_output_tokens"]
        for admission in admitted
        if isinstance(admission["usage"].get("trainable_output_tokens"), int)
    ]
    token_fields = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [admission["usage"].get(field) for admission in admissions]
        token_fields[field] = sum(values) if all(isinstance(value, int) for value in values) else None
    candidate_count = len(admissions)
    return {
        "candidate_count": candidate_count,
        "accepted_count": len(admitted),
        "accepted_yield": len(admitted) / candidate_count if candidate_count else None,
        "class_counts": {data_class: class_counts.get(data_class, 0) for data_class in APPWORLD_DATA_CLASSES},
        "collateral_rejection_count": sum(admission["collateral_rejected"] is True for admission in admissions),
        "replay_mismatch_count": sum(admission["replay_match"] is not True for admission in admissions),
        "duplicate_accepted_count": duplicate_count,
        "duplicate_accepted_rate": duplicate_count / len(admitted) if admitted else None,
        "usage": token_fields,
        "trainable_output_tokens": {
            "reported_count": len(trainable_tokens),
            "p50": _nearest_rank(trainable_tokens, 0.50),
            "p95": _nearest_rank(trainable_tokens, 0.95),
        },
        "truncation_rate": (
            sum(admission["truncated"] is True for admission in admissions) / candidate_count
            if candidate_count
            else None
        ),
        "cost_usd": (
            sum(admission["cost_usd"] for admission in admissions)
            if all(isinstance(admission.get("cost_usd"), int | float) for admission in admissions)
            else None
        ),
        "cost_reported_for_all_candidates": all(
            isinstance(admission.get("cost_usd"), int | float) for admission in admissions
        ),
        "cost_note": "provider pricing must be supplied and recorded for a paid pilot",
        "hidden_verifier_material_included": False,
    }


_SENSITIVE_IDENTITY_KEY_PARTS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
)


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def _reject_sensitive_identity_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if any(part in normalized_key for part in _SENSITIVE_IDENTITY_KEY_PARTS):
                raise ValueError(f"provider identity must not contain sensitive key {key!r}")
            _reject_sensitive_identity_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_identity_keys(item)


def _validate_provider_identity(provider_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provider_identity, Mapping):
        raise TypeError("provider_identity must be a mapping")
    copied = _json_copy(provider_identity, name="provider_identity")
    _reject_sensitive_identity_keys(copied)
    for key in ("provider_id", "endpoint_label", "api_contract"):
        if not isinstance(copied.get(key), str) or not copied[key].strip():
            raise ValueError(f"provider_identity.{key} must be a non-empty string")
    pricing = copied.get("pricing")
    if pricing is not None:
        if not isinstance(pricing, Mapping):
            raise TypeError("provider_identity.pricing must be a mapping")
        for key in ("input_usd_per_million_tokens", "output_usd_per_million_tokens"):
            value = pricing.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"provider_identity.pricing.{key} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"provider_identity.pricing.{key} must be finite and non-negative")
        for key in ("effective_date", "source"):
            if not isinstance(pricing.get(key), str) or not pricing[key].strip():
                raise ValueError(f"provider_identity.pricing.{key} must be a non-empty string")
    return copied


def _candidate_cost_usd(artifact: Mapping[str, Any], provider_identity: Mapping[str, Any]) -> float | None:
    pricing = provider_identity.get("pricing")
    if not isinstance(pricing, Mapping):
        return None
    usage = artifact["metrics"].get("usage")
    if not isinstance(usage, Mapping):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return (
        input_tokens * pricing["input_usd_per_million_tokens"]
        + output_tokens * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000


def _new_json_file(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_raw_outputs(source: Path, destination: Path) -> list[str]:
    copied = []
    for name in ("logs", "version", "evaluation", "misc"):
        source_path = source / name
        if not source_path.exists():
            continue
        destination_path = destination / name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        copied.append(name)
    return copied


def run_appworld_data_factory(
    backend: CompletionBackend,
    *,
    artifact_dir: str | os.PathLike[str],
    run_id: str,
    appworld_root: str | os.PathLike[str],
    appworld_source: str | os.PathLike[str],
    teacher_model_name: str,
    teacher_model_revision: str,
    provider_identity: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
    candidates_per_task: int = 2,
    max_steps: int = 50,
) -> dict[str, Any]:
    """Execute, replay, classify, and atomically publish the fixed train pilot."""

    for name, value in (
        ("run_id", run_id),
        ("teacher_model_name", teacher_model_name),
        ("teacher_model_revision", teacher_model_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if isinstance(candidates_per_task, bool) or not isinstance(candidates_per_task, int):
        raise TypeError("candidates_per_task must be an integer")
    if not 2 <= candidates_per_task <= 4:
        raise ValueError("candidates_per_task must be between 2 and 4 for the micro pilot")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 50:
        raise ValueError("max_steps must be an integer between 1 and 50")
    if not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping")
    sampling = _json_copy(model_kwargs, name="model_kwargs")
    provider = _validate_provider_identity(provider_identity)

    target = Path(artifact_dir)
    if target.exists():
        raise FileExistsError(f"artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        dataset, dataset_manifest = load_pinned_appworld_data_pilot_dataset(appworld_root=appworld_root)
        dataset_payload = dataset.to_dict()
        admissions = []
        candidate_records = []
        for task_id in APPWORLD_DATA_PILOT_TASK_IDS:
            for candidate_index in range(candidates_per_task):
                candidate_slug = f"candidate-{candidate_index:02d}"
                candidate_id = f"{run_id}:{task_id}:{candidate_slug}"
                result = run_appworld_episode(
                    backend,
                    task_id=task_id,
                    model_name=teacher_model_name,
                    model_revision=teacher_model_revision,
                    trajectory_id=candidate_id,
                    appworld_root=appworld_root,
                    appworld_source=appworld_source,
                    experiment_name=f"lmflow-appworld-data/{run_id}/{task_id}/{candidate_slug}",
                    model_kwargs=sampling,
                    max_steps=max_steps,
                    source_split="train",
                )
                replay = replay_appworld_episode(
                    result.artifact,
                    appworld_root=appworld_root,
                    experiment_name=f"lmflow-appworld-data-replay/{run_id}/{task_id}/{candidate_slug}",
                )
                finalized = finalize_appworld_candidate(
                    result,
                    replay,
                    cost_usd=_candidate_cost_usd(result.artifact, provider),
                )
                candidate_directory = staging / "candidates" / task_id / candidate_slug
                artifact_path = candidate_directory / "trajectory.json"
                replay_path = candidate_directory / "replay.json"
                admission_path = candidate_directory / "admission.json"
                _new_json_file(artifact_path, result.artifact)
                _new_json_file(replay_path, replay)
                _new_json_file(admission_path, finalized.admission)
                projection_ref = None
                projection_sha256 = None
                if finalized.training_projection is not None:
                    projection_path = candidate_directory / "conversation.json"
                    _new_json_file(projection_path, finalized.training_projection)
                    projection_ref = projection_path.relative_to(staging).as_posix()
                    projection_sha256 = _sha256_file(projection_path)
                raw_directory = candidate_directory / "raw_appworld"
                raw_sections = _copy_raw_outputs(result.raw_output_directory, raw_directory)
                admissions.append(finalized.admission)
                candidate_records.append(
                    {
                        "candidate_id": candidate_id,
                        "task_id": task_id,
                        "candidate_index": candidate_index,
                        "data_class": finalized.admission["data_class"],
                        "admitted_for_sft": finalized.admission["admitted_for_sft"],
                        "trajectory_ref": artifact_path.relative_to(staging).as_posix(),
                        "trajectory_file_sha256": _sha256_file(artifact_path),
                        "trajectory_manifest_sha256": result.artifact["manifest_sha256"],
                        "replay_ref": replay_path.relative_to(staging).as_posix(),
                        "replay_file_sha256": _sha256_file(replay_path),
                        "replay_manifest_sha256": replay["manifest_sha256"],
                        "admission_ref": admission_path.relative_to(staging).as_posix(),
                        "admission_file_sha256": _sha256_file(admission_path),
                        "admission_manifest_sha256": finalized.admission["manifest_sha256"],
                        "training_projection_ref": projection_ref,
                        "training_projection_sha256": projection_sha256,
                        "raw_appworld_ref": raw_directory.relative_to(staging).as_posix(),
                        "raw_appworld_sections": raw_sections,
                    }
                )

        factory_manifest = with_manifest_digest(
            {
                "format_version": APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "dataset_manifest_ref": "dataset_manifest.json",
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "dataset_projection_ref": "dataset.json",
                "dataset_projection_sha256": canonical_json_sha256(dataset_payload),
                "task_ids": list(APPWORLD_DATA_PILOT_TASK_IDS),
                "candidates_per_task": candidates_per_task,
                "candidate_count": len(candidate_records),
                "teacher": {
                    "model_name": teacher_model_name,
                    "model_revision": teacher_model_revision,
                    "provider": provider,
                    "sampling": sampling,
                },
                "appworld_revision": APPWORLD_REVISION,
                "execution": {
                    "serial": True,
                    "fresh_reset_per_candidate": True,
                    "fresh_reset_replay_required": True,
                    "official_evaluator_required": True,
                    "max_steps": max_steps,
                },
                "projection_policy": {
                    "A": "success-only and success-plus-recovery SFT arms",
                    "B": "success-plus-recovery arm with invalid assistant actions loss-masked",
                    "C": "sealed partial signal only; no SFT projection in this slice",
                    "D": "no SFT projection; preference requires a verified improved pair",
                    "E": "diagnostics only",
                },
                "protected_artifacts_outside_git": True,
                "hidden_verifier_material_in_dataset_or_conversation": False,
            }
        )
        report = with_manifest_digest(
            {
                "format_version": APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION,
                "run_id": run_id,
                "factory_manifest_ref": "factory_manifest.json",
                "factory_manifest_sha256": factory_manifest["manifest_sha256"],
                "candidate_records": candidate_records,
                "summary": summarize_appworld_candidates(admissions),
            }
        )
        _new_json_file(staging / "dataset.json", dataset_payload)
        _new_json_file(staging / "dataset_manifest.json", dataset_manifest)
        _new_json_file(staging / "factory_manifest.json", factory_manifest)
        _new_json_file(staging / "report.json", report)
        staging.rename(target)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "APPWORLD_DATA_CANDIDATE_FORMAT_VERSION",
    "APPWORLD_DATA_CLASSES",
    "APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION",
    "AppWorldFinalizedCandidate",
    "finalize_appworld_candidate",
    "run_appworld_data_factory",
    "summarize_appworld_candidates",
]
