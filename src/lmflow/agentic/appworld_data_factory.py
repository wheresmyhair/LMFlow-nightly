"""Benchmark-local admission and projection gates for AppWorld cold-start data."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
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
    APPWORLD_CONTEXT_BUDGET_EXHAUSTED,
    APPWORLD_DATA_PILOT_SCENARIOS,
    APPWORLD_DATA_PILOT_TASK_IDS,
    APPWORLD_REVISION,
    APPWORLD_SCENARIO_CURRICULUM_SCENARIOS,
    APPWORLD_SCENARIO_CURRICULUM_TASK_IDS,
    APPWORLD_TRAIN_D1_D2_EXPANSION_TASK_IDS,
    canonical_json_sha256,
    load_pinned_appworld_data_pilot_dataset,
    load_pinned_appworld_scenario_curriculum_dataset,
    load_pinned_appworld_train_d1_d2_expansion_dataset,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.agentic.completion import CompletionBackend
from lmflow.agentic.data_construction import (
    AdmissionProduct,
    CandidateContext,
    CandidatePlanEntry,
    StageProduct,
    build_candidate_plan,
    build_selection_manifest,
    run_data_construction_attempt,
)

APPWORLD_DATA_CANDIDATE_FORMAT_VERSION = "lmflow.appworld-data-candidate/v1"
APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION = "lmflow.appworld-data-factory-run/v3"
APPWORLD_DATA_CLASSES = ("A", "B", "C", "D", "E")
APPWORLD_DATA_FACTORY_TASK_SET_IDS = (
    "initial_pilot",
    "scenario_curriculum_v1",
    "train_d1_d2_expansion_v1",
)


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
    if metrics.get("failure_type") == APPWORLD_CONTEXT_BUDGET_EXHAUSTED:
        return "D"
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


class _AppWorldDataConstructionAdapter:
    """Keep AppWorld execution and quality semantics behind the shared lifecycle."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        run_id: str,
        appworld_root: str | os.PathLike[str],
        appworld_source: str | os.PathLike[str],
        teacher_model_name: str,
        teacher_model_revision: str,
        provider_identity: Mapping[str, Any],
        max_steps: int,
    ) -> None:
        self._backend = backend
        self._run_id = run_id
        self._appworld_root = appworld_root
        self._appworld_source = appworld_source
        self._teacher_model_name = teacher_model_name
        self._teacher_model_revision = teacher_model_revision
        self._provider_identity = provider_identity
        self._max_steps = max_steps

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter_id": "appworld-data-factory-v1",
            "adapter_format_version": APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION,
            "appworld_revision": APPWORLD_REVISION,
            "verification": "fresh-reset replay plus official evaluator facts",
            "admission": "AppWorld benchmark-local A-E v1",
            "projection": "AppWorld semantic conversation v2 with per-action loss",
        }

    def interact(self, context: CandidateContext) -> StageProduct:
        result = run_appworld_episode(
            self._backend,
            task_id=context.plan_entry.task_id,
            model_name=self._teacher_model_name,
            model_revision=self._teacher_model_revision,
            trajectory_id=context.candidate_id,
            appworld_root=self._appworld_root,
            appworld_source=self._appworld_source,
            experiment_name=f"lmflow-appworld-data/{self._run_id}/{context.plan_entry.ordinal:05d}",
            model_kwargs=context.plan_entry.sampling,
            max_steps=self._max_steps,
            source_split="train",
        )
        return StageProduct(artifact=result.artifact, state=result)

    def verify(self, context: CandidateContext, interaction: StageProduct) -> StageProduct:
        replay = replay_appworld_episode(
            interaction.artifact,
            appworld_root=self._appworld_root,
            experiment_name=f"lmflow-appworld-data-replay/{self._run_id}/{context.plan_entry.ordinal:05d}",
        )
        return StageProduct(artifact=replay, state=replay)

    def admit(
        self,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
    ) -> AdmissionProduct:
        finalized = finalize_appworld_candidate(
            interaction.state,
            verification.artifact,
            cost_usd=_candidate_cost_usd(interaction.artifact, self._provider_identity),
        )
        admission = finalized.admission
        artifact = interaction.artifact
        metrics = artifact["metrics"]
        action_codes = [step.get("code") for step in artifact.get("action_steps", [])]
        duplicate_action_count = len(action_codes) - len(set(action_codes))
        scenario_id = context.plan_entry.task_id.split("_", 1)[0]
        scenario = {
            **APPWORLD_DATA_PILOT_SCENARIOS,
            **APPWORLD_SCENARIO_CURRICULUM_SCENARIOS,
        }[scenario_id]
        tags = []
        if admission["sft_arms"]["success_only"] is True:
            tags.append("success-only")
        if admission["sft_arms"]["success_plus_recovery"] is True:
            tags.append("success-plus-recovery")
        usage = metrics.get("usage") if isinstance(metrics.get("usage"), Mapping) else {}
        metadata = {
            "benchmark": "appworld",
            "source_split": "train",
            "scenario_id": scenario_id,
            "difficulty": scenario["difficulty"],
            "data_class": admission["data_class"],
            "admitted_for_sft": admission["admitted_for_sft"],
            "official_success": metrics.get("success") is True,
            "replay_match": admission["replay_match"],
            "collateral_invariant_passed": admission["collateral_invariant_passed"],
            "hidden_verifier_material_included": admission["hidden_verifier_material_included"],
            "truncated": admission["truncated"],
            "invalid_actions": metrics.get("invalid_tool_calls", 0),
            "recovery_count": metrics.get("recovery_count", 0),
            "state_change_steps": metrics.get("state_change_steps", 0),
            "action_path_sha256": canonical_json_sha256(action_codes),
            "duplicate_action_count": duplicate_action_count,
            "duplicate_target_sha256": admission.get("duplicate_target_sha256"),
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "cost_usd": admission.get("cost_usd"),
        }
        return AdmissionProduct(
            artifact=admission,
            state=finalized,
            selection_tags=tuple(tags),
            record_metadata=metadata,
        )

    def project(
        self,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
        admission: AdmissionProduct,
    ) -> Mapping[str, Any] | None:
        del context, interaction, verification
        return admission.state.training_projection

    def materialize_evidence(
        self,
        candidate_directory: Path,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
        admission: AdmissionProduct,
    ) -> Mapping[str, Any]:
        del context, verification, admission
        raw_directory = candidate_directory / "raw_appworld"
        raw_sections = _copy_raw_outputs(interaction.state.raw_output_directory, raw_directory)
        return {
            "raw_appworld_ref": "raw_appworld",
            "raw_appworld_sections": raw_sections,
        }

    def summarize(
        self,
        admissions: Sequence[AdmissionProduct],
        candidate_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        del candidate_records
        return summarize_appworld_candidates([admission.artifact for admission in admissions])

    def build_selections(
        self,
        candidate_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]:
        def dedup_key(record: Mapping[str, Any]) -> list[Any]:
            return [record["task_id"], record["metadata"]["action_path_sha256"]]

        def rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
            metadata = record["metadata"]
            usage = metadata["usage"]
            total_tokens = usage.get("total_tokens")
            return (
                metadata["data_class"] != "A",
                metadata["truncated"],
                metadata["invalid_actions"],
                metadata["duplicate_action_count"],
                math.inf if total_tokens is None else total_tokens,
                record["ordinal"],
            )

        policies = {}
        for policy_id, tag, classes in (
            ("success-only", "success-only", ["A"]),
            ("success-plus-recovery", "success-plus-recovery", ["A", "B"]),
        ):
            policies[policy_id] = build_selection_manifest(
                candidate_records,
                policy_id=policy_id,
                policy_identity={
                    "benchmark": "appworld",
                    "eligible_classes": classes,
                    "eligibility": f"selection tag {tag!r}",
                    "dedup_key": ["task_id", "action_path_sha256"],
                    "ranking": [
                        "prefer A",
                        "prefer untruncated",
                        "fewer invalid actions",
                        "fewer duplicate actions",
                        "fewer provider tokens",
                        "lower plan ordinal",
                    ],
                },
                eligible=lambda record, tag=tag: tag in record["selection_tags"],
                dedup_key=dedup_key,
                rank_key=rank_key,
                output_order_key=lambda record: record["ordinal"],
            )
        return policies


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
    candidate_seeds: Sequence[int] | None = None,
    task_set_id: str = "initial_pilot",
    scheduled_ordinals: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Execute a pinned AppWorld slice through the shared construction lifecycle."""

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
    if task_set_id not in APPWORLD_DATA_FACTORY_TASK_SET_IDS:
        raise ValueError(f"task_set_id must be one of {APPWORLD_DATA_FACTORY_TASK_SET_IDS}")
    sampling = _json_copy(model_kwargs, name="model_kwargs")
    if candidate_seeds is None:
        sampling_profiles = [copy.deepcopy(sampling) for _ in range(candidates_per_task)]
    else:
        if isinstance(candidate_seeds, str | bytes) or not isinstance(candidate_seeds, Sequence):
            raise TypeError("candidate_seeds must be a sequence of integers")
        if len(candidate_seeds) != candidates_per_task:
            raise ValueError("candidate_seeds must contain one seed per candidate")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in candidate_seeds):
            raise TypeError("candidate_seeds must contain only integers")
        if len(set(candidate_seeds)) != len(candidate_seeds):
            raise ValueError("candidate_seeds must be unique")
        sampling_profiles = []
        for seed in candidate_seeds:
            profile = copy.deepcopy(sampling)
            profile["seed"] = seed
            sampling_profiles.append(profile)
    provider = _validate_provider_identity(provider_identity)

    if task_set_id == "initial_pilot":
        task_ids = APPWORLD_DATA_PILOT_TASK_IDS
        dataset, dataset_manifest = load_pinned_appworld_data_pilot_dataset(appworld_root=appworld_root)
    elif task_set_id == "scenario_curriculum_v1":
        task_ids = APPWORLD_SCENARIO_CURRICULUM_TASK_IDS
        dataset, dataset_manifest = load_pinned_appworld_scenario_curriculum_dataset(appworld_root=appworld_root)
    else:
        task_ids = APPWORLD_TRAIN_D1_D2_EXPANSION_TASK_IDS
        dataset, dataset_manifest = load_pinned_appworld_train_d1_d2_expansion_dataset(appworld_root=appworld_root)
    plan_entries = []
    for task_id in task_ids:
        for candidate_index, candidate_sampling in enumerate(sampling_profiles):
            seed = candidate_sampling.get("seed")
            sample_id = (
                f"seed-{seed}"
                if isinstance(seed, int) and not isinstance(seed, bool)
                else f"candidate-{candidate_index:02d}"
            )
            plan_entries.append(
                CandidatePlanEntry(
                    ordinal=len(plan_entries),
                    task_id=task_id,
                    sample_id=sample_id,
                    sampling=candidate_sampling,
                )
            )
    candidate_plan = build_candidate_plan(
        plan_entries,
        task_set_id=task_set_id,
        task_dataset_manifest_sha256=dataset_manifest["manifest_sha256"],
    )
    adapter = _AppWorldDataConstructionAdapter(
        backend,
        run_id=run_id,
        appworld_root=appworld_root,
        appworld_source=appworld_source,
        teacher_model_name=teacher_model_name,
        teacher_model_revision=teacher_model_revision,
        provider_identity=provider,
        max_steps=max_steps,
    )
    return run_data_construction_attempt(
        adapter,
        artifact_dir=artifact_dir,
        attempt_id=run_id,
        task_dataset=dataset.to_dict(),
        task_dataset_manifest=dataset_manifest,
        candidate_plan=candidate_plan,
        scheduled_ordinals=scheduled_ordinals,
        run_identity={
            "adapter_format_version": APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "task_set_id": task_set_id,
            "task_ids": list(task_ids),
            "teacher": {
                "model_name": teacher_model_name,
                "model_revision": teacher_model_revision,
                "provider": provider,
                "sampling": sampling,
                "candidate_sampling": sampling_profiles,
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
        },
    )


__all__ = [
    "APPWORLD_DATA_CANDIDATE_FORMAT_VERSION",
    "APPWORLD_DATA_CLASSES",
    "APPWORLD_DATA_FACTORY_TASK_SET_IDS",
    "APPWORLD_DATA_FACTORY_RUN_FORMAT_VERSION",
    "AppWorldFinalizedCandidate",
    "finalize_appworld_candidate",
    "run_appworld_data_factory",
    "summarize_appworld_candidates",
]
