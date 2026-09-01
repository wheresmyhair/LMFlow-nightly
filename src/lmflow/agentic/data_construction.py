"""Immutable lifecycle helpers for agentic data construction.

This module owns execution-neutral mechanics only: stable candidate plans,
attempt identities, evidence-first stage publication, resumable candidate
accounting, deterministic selection manifests, and atomic run publication.
Benchmark adapters continue to own environments, verifiers, admission classes,
quality rules, and training projections.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

DATA_CONSTRUCTION_PLAN_FORMAT_VERSION = "lmflow.data-construction-plan/v1"
DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION = "lmflow.data-construction-attempt/v1"
DATA_CONSTRUCTION_CANDIDATE_RECORD_FORMAT_VERSION = "lmflow.data-construction-candidate-record/v1"
DATA_CONSTRUCTION_SELECTION_FORMAT_VERSION = "lmflow.data-construction-selection/v1"
DATA_CONSTRUCTION_RESUME_FORMAT_VERSION = "lmflow.data-construction-resume/v1"

_POLICY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SENSITIVE_KEY_PARTS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "proxy_url",
    "secret",
)


def json_copy(value: Any, *, name: str) -> Any:
    """Return an isolated JSON-compatible copy with finite numeric values."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible and contain only finite numbers") from error


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the canonical encoding shared by manifests."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def with_manifest_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a JSON object and attach a digest that excludes the digest field."""

    if not isinstance(payload, Mapping):
        raise TypeError("manifest payload must be a mapping")
    manifest = json_copy(payload, name="manifest payload")
    if "manifest_sha256" in manifest:
        raise ValueError("manifest payload must not already contain manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def verify_manifest_digest(manifest: Mapping[str, Any]) -> None:
    """Fail closed unless a manifest carries its exact canonical digest."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    expected = _validate_sha256(manifest.get("manifest_sha256"), name="manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if canonical_json_sha256(payload) != expected:
        raise ValueError("manifest_sha256 does not match the manifest content")


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of one regular file."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_sensitive_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                raise ValueError(f"{path} must not contain sensitive key {key!r}")
            _reject_sensitive_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_keys(item, path=f"{path}[{index}]")


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class CandidatePlanEntry:
    """One stable candidate slot shared by all retry attempts."""

    ordinal: int
    task_id: str
    sample_id: str
    sampling: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        _non_empty_string(self.task_id, name="task_id")
        _non_empty_string(self.sample_id, name="sample_id")
        sampling = json_copy(self.sampling, name="sampling")
        if not isinstance(sampling, dict):
            raise TypeError("sampling must be a mapping")
        _reject_sensitive_keys(sampling, path="sampling")
        object.__setattr__(self, "sampling", sampling)

    def payload(self) -> dict[str, Any]:
        value = {
            "ordinal": self.ordinal,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "sampling": copy.deepcopy(dict(self.sampling)),
        }
        value["plan_entry_id"] = canonical_json_sha256(value)
        return value


def build_candidate_plan(
    entries: Sequence[CandidatePlanEntry],
    *,
    task_set_id: str,
    task_dataset_manifest_sha256: str,
) -> dict[str, Any]:
    """Freeze an ordered candidate schedule independently of execution attempts."""

    _non_empty_string(task_set_id, name="task_set_id")
    _validate_sha256(task_dataset_manifest_sha256, name="task_dataset_manifest_sha256")
    if isinstance(entries, str | bytes) or not isinstance(entries, Sequence) or not entries:
        raise ValueError("entries must contain at least one CandidatePlanEntry")
    if any(not isinstance(entry, CandidatePlanEntry) for entry in entries):
        raise TypeError("entries must contain only CandidatePlanEntry values")
    ordinals = [entry.ordinal for entry in entries]
    if ordinals != list(range(len(entries))):
        raise ValueError("candidate plan ordinals must be contiguous and ordered from zero")
    payloads = [entry.payload() for entry in entries]
    entry_ids = [payload["plan_entry_id"] for payload in payloads]
    if len(set(entry_ids)) != len(entry_ids):
        raise ValueError("candidate plan entries must be unique")
    sample_slots = [(entry.task_id, entry.sample_id) for entry in entries]
    if len(set(sample_slots)) != len(sample_slots):
        raise ValueError("candidate plan task_id/sample_id slots must be unique")
    return with_manifest_digest(
        {
            "format_version": DATA_CONSTRUCTION_PLAN_FORMAT_VERSION,
            "task_set_id": task_set_id,
            "task_dataset_manifest_sha256": task_dataset_manifest_sha256,
            "candidate_count": len(payloads),
            "entries": payloads,
        }
    )


def candidate_plan_entries(plan: Mapping[str, Any]) -> tuple[CandidatePlanEntry, ...]:
    """Validate and reconstruct a frozen candidate plan."""

    verify_manifest_digest(plan)
    if plan.get("format_version") != DATA_CONSTRUCTION_PLAN_FORMAT_VERSION:
        raise ValueError("unexpected data-construction plan format")
    raw_entries = plan.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("candidate plan entries are missing")
    entries = tuple(
        CandidatePlanEntry(
            ordinal=value["ordinal"],
            task_id=value["task_id"],
            sample_id=value["sample_id"],
            sampling=value["sampling"],
        )
        for value in raw_entries
    )
    rebuilt = build_candidate_plan(
        entries,
        task_set_id=plan["task_set_id"],
        task_dataset_manifest_sha256=plan["task_dataset_manifest_sha256"],
    )
    if rebuilt != dict(plan):
        raise ValueError("candidate plan entry identities do not match their payloads")
    return entries


@dataclass(frozen=True)
class CandidateContext:
    """Attempt-local identity for one stable candidate plan entry."""

    attempt_id: str
    candidate_id: str
    plan_entry: CandidatePlanEntry


@dataclass
class StageProduct:
    """A sealed stage artifact plus benchmark-local in-process state."""

    artifact: Mapping[str, Any]
    state: Any = None


@dataclass
class AdmissionProduct(StageProduct):
    """Admission evidence and non-authoritative indexing facts."""

    selection_tags: Sequence[str] = field(default_factory=tuple)
    record_metadata: Mapping[str, Any] = field(default_factory=dict)


class DataConstructionAdapter(Protocol):
    """Benchmark-owned semantics plugged into the shared artifact lifecycle."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def interact(self, context: CandidateContext) -> StageProduct: ...

    def verify(self, context: CandidateContext, interaction: StageProduct) -> StageProduct: ...

    def admit(
        self,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
    ) -> AdmissionProduct: ...

    def project(
        self,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
        admission: AdmissionProduct,
    ) -> Mapping[str, Any] | None: ...

    def materialize_evidence(
        self,
        candidate_directory: Path,
        context: CandidateContext,
        interaction: StageProduct,
        verification: StageProduct,
        admission: AdmissionProduct,
    ) -> Mapping[str, Any]: ...

    def summarize(
        self,
        admissions: Sequence[AdmissionProduct],
        candidate_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def build_selections(
        self,
        candidate_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Mapping[str, Any]]: ...


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


def _validated_stage_product(product: StageProduct, *, stage: str) -> StageProduct:
    if not isinstance(product, StageProduct):
        raise TypeError(f"{stage} must return StageProduct")
    artifact = json_copy(product.artifact, name=f"{stage} artifact")
    if not isinstance(artifact, dict):
        raise TypeError(f"{stage} artifact must be a mapping")
    verify_manifest_digest(artifact)
    product.artifact = artifact
    return product


def _candidate_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("candidate record must be a mapping")
    for key in ("attempt_id", "candidate_id", "plan_entry_id", "task_id", "sample_id", "record_ref"):
        _non_empty_string(record.get(key), name=f"candidate record {key}")
    ordinal = record.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("candidate record ordinal must be a non-negative integer")
    _validate_sha256(record.get("record_file_sha256"), name="candidate record_file_sha256")
    _validate_sha256(record.get("record_manifest_sha256"), name="candidate record_manifest_sha256")
    tags = record.get("selection_tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise TypeError("candidate record selection_tags must be a list of strings")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("candidate record metadata must be a mapping")
    return {
        "attempt_id": record["attempt_id"],
        "candidate_id": record["candidate_id"],
        "plan_entry_id": record["plan_entry_id"],
        "ordinal": record["ordinal"],
        "task_id": record["task_id"],
        "sample_id": record["sample_id"],
        "record_ref": record["record_ref"],
        "record_file_sha256": record["record_file_sha256"],
        "record_manifest_sha256": record["record_manifest_sha256"],
        "selection_tags": list(tags),
        "metadata": json_copy(metadata, name="candidate record metadata"),
    }


def build_selection_manifest(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    policy_id: str,
    policy_identity: Mapping[str, Any],
    eligible: Callable[[Mapping[str, Any]], bool],
    dedup_key: Callable[[Mapping[str, Any]], Any],
    rank_key: Callable[[Mapping[str, Any]], Any],
    output_order_key: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Select deterministic candidate references without interpreting quality."""

    if not isinstance(policy_id, str) or _POLICY_ID_PATTERN.fullmatch(policy_id) is None:
        raise ValueError("policy_id must be a path-safe stable identifier")
    identity = json_copy(policy_identity, name="policy_identity")
    if not isinstance(identity, dict):
        raise TypeError("policy_identity must be a mapping")
    _reject_sensitive_keys(identity, path="policy_identity")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    eligible_count = 0
    for record in candidate_records:
        if not isinstance(record, Mapping):
            raise TypeError("candidate_records must contain mappings")
        if not eligible(record):
            continue
        eligible_count += 1
        key = json_copy(dedup_key(record), name="dedup key")
        groups[canonical_json_sha256(key)].append(record)
    selected = []
    duplicates = []
    for digest in sorted(groups):
        ranked = sorted(
            groups[digest],
            key=lambda record: (
                rank_key(record),
                record["attempt_id"],
                record["candidate_id"],
            ),
        )
        selected.append(_candidate_reference(ranked[0]))
        duplicates.extend(_candidate_reference(record) for record in ranked[1:])
    if output_order_key is not None:
        selected.sort(key=output_order_key)
        duplicates.sort(key=output_order_key)
    return with_manifest_digest(
        {
            "format_version": DATA_CONSTRUCTION_SELECTION_FORMAT_VERSION,
            "policy_id": policy_id,
            "policy_identity": identity,
            "eligible_count": eligible_count,
            "selected_count": len(selected),
            "duplicate_count": len(duplicates),
            "selected": selected,
            "rejected_duplicates": duplicates,
        }
    )


def build_resume_manifest(
    plan: Mapping[str, Any],
    attempt_reports: Sequence[Mapping[str, Any]],
    *,
    canonical_eligible: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """Resolve completed plan slots across attempts and fail on double execution."""

    entries = candidate_plan_entries(plan)
    payloads = [entry.payload() for entry in entries]
    entry_payloads = {payload["plan_entry_id"]: payload for payload in payloads}
    by_entry: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    attempt_refs = []
    seen_attempts = set()
    for report in attempt_reports:
        verify_manifest_digest(report)
        if report.get("format_version") != DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION:
            raise ValueError("unexpected data-construction attempt format")
        if report.get("plan_manifest_sha256") != plan["manifest_sha256"]:
            raise ValueError("attempt report belongs to a different candidate plan")
        attempt_id = _non_empty_string(report.get("attempt_id"), name="attempt report attempt_id")
        if attempt_id in seen_attempts:
            raise ValueError(f"duplicate attempt_id in resume input: {attempt_id!r}")
        seen_attempts.add(attempt_id)
        records = report.get("candidate_records")
        if not isinstance(records, list):
            raise TypeError("attempt report candidate_records must be a list")
        candidate_count = report.get("candidate_count")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count != len(records):
            raise ValueError("attempt report candidate_count does not match candidate_records")
        attempt_refs.append(
            {
                "attempt_id": attempt_id,
                "manifest_sha256": report["manifest_sha256"],
                "candidate_count": candidate_count,
            }
        )
        seen_entries = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("attempt report candidate_records must contain mappings")
            if record.get("attempt_id") != attempt_id:
                raise ValueError("candidate record attempt_id differs from its attempt report")
            if record.get("plan_manifest_sha256") != plan["manifest_sha256"]:
                raise ValueError("candidate record belongs to a different candidate plan")
            plan_entry_id = record.get("plan_entry_id")
            expected_entry = entry_payloads.get(plan_entry_id)
            if expected_entry is None:
                raise ValueError("candidate record references an unknown candidate plan entry")
            if plan_entry_id in seen_entries:
                raise ValueError("attempt report contains multiple records for one candidate plan entry")
            seen_entries.add(plan_entry_id)
            for key in ("ordinal", "task_id", "sample_id"):
                if record.get(key) != expected_entry[key]:
                    raise ValueError(f"candidate record {key} differs from its candidate plan entry")
            _candidate_reference(record)
            by_entry[plan_entry_id].append(record)

    canonical = []
    unresolved = []
    for entry in entries:
        entry_payload = entry.payload()
        eligible_records = [record for record in by_entry[entry_payload["plan_entry_id"]] if canonical_eligible(record)]
        if len(eligible_records) > 1:
            raise RuntimeError(
                f"multiple canonical-eligible executions for plan ordinal {entry.ordinal}; "
                "the completed candidate was retried"
            )
        if eligible_records:
            canonical.append(_candidate_reference(eligible_records[0]))
        else:
            unresolved.append(entry_payload)
    return with_manifest_digest(
        {
            "format_version": DATA_CONSTRUCTION_RESUME_FORMAT_VERSION,
            "plan_manifest_sha256": plan["manifest_sha256"],
            "attempts": attempt_refs,
            "canonical_count": len(canonical),
            "unresolved_count": len(unresolved),
            "canonical": canonical,
            "unresolved": unresolved,
        }
    )


def write_artifact_manifest(root: str | os.PathLike[str]) -> None:
    """Seal every existing artifact file under ``root`` into a new manifest."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"artifact root must be a non-symlink directory: {root}")
    manifest_path = root / "artifact-manifest.sha256"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f"artifact tree contains an unsupported entry: {path}")
    with manifest_path.open("x", encoding="ascii", newline="\n") as output_file:
        for path in files:
            relative_path = path.relative_to(root).as_posix()
            if "\n" in relative_path or "\r" in relative_path:
                raise ValueError(f"artifact path contains a newline: {relative_path!r}")
            output_file.write(f"{file_sha256(path)}  {relative_path}\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def verify_artifact_manifest(artifact_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify an exact artifact file set against its recursive SHA-256 manifest."""

    root = Path(artifact_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"artifact_dir must be a non-symlink directory: {root}")
    manifest_path = root / "artifact-manifest.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"artifact manifest is missing or not a regular file: {manifest_path}")

    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="ascii").splitlines(), start=1):
        parts = raw_line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"malformed artifact manifest line {line_number}")
        digest, relative_path = parts
        _validate_sha256(digest, name=f"artifact manifest line {line_number} digest")
        pure_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or pure_path.is_absolute()
            or any(part in ("", ".", "..") for part in pure_path.parts)
            or relative_path == manifest_path.name
        ):
            raise ValueError(f"unsafe artifact manifest path on line {line_number}")
        if relative_path in expected:
            raise ValueError(f"duplicate artifact manifest path on line {line_number}")
        expected[relative_path] = digest

    actual = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree must not contain symlinks: {path}")
        if path.is_file() and path != manifest_path:
            actual[path.relative_to(root).as_posix()] = path
        elif not path.is_file() and not path.is_dir():
            raise ValueError(f"artifact tree contains an unsupported entry: {path}")
    missing = sorted(set(expected) - set(actual))
    unmanaged = sorted(set(actual) - set(expected))
    if missing or unmanaged:
        raise ValueError(f"artifact file set mismatch: missing={missing}, unmanaged={unmanaged}")
    for relative_path, expected_digest in expected.items():
        if file_sha256(actual[relative_path]) != expected_digest:
            raise ValueError(f"artifact file digest mismatch: {relative_path}")
    return {
        "file_count": len(expected),
        "manifest_file_sha256": file_sha256(manifest_path),
    }


def run_data_construction_attempt(
    adapter: DataConstructionAdapter,
    *,
    artifact_dir: str | os.PathLike[str],
    attempt_id: str,
    task_dataset: Mapping[str, Any],
    task_dataset_manifest: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    scheduled_ordinals: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run and atomically publish one immutable data-construction attempt."""

    _non_empty_string(attempt_id, name="attempt_id")
    dataset = json_copy(task_dataset, name="task_dataset")
    if not isinstance(dataset, dict):
        raise TypeError("task_dataset must be a mapping")
    dataset_manifest = json_copy(task_dataset_manifest, name="task_dataset_manifest")
    if not isinstance(dataset_manifest, dict):
        raise TypeError("task_dataset_manifest must be a mapping")
    verify_manifest_digest(dataset_manifest)
    plan = json_copy(candidate_plan, name="candidate_plan")
    if not isinstance(plan, dict):
        raise TypeError("candidate_plan must be a mapping")
    entries = candidate_plan_entries(plan)
    if plan["task_dataset_manifest_sha256"] != dataset_manifest["manifest_sha256"]:
        raise ValueError("candidate plan references a different task dataset manifest")
    identity = json_copy(run_identity, name="run_identity")
    adapter_identity = json_copy(adapter.identity, name="adapter.identity")
    if not isinstance(identity, dict) or not isinstance(adapter_identity, dict):
        raise TypeError("run and adapter identities must be mappings")
    _reject_sensitive_keys(identity, path="run_identity")
    _reject_sensitive_keys(adapter_identity, path="adapter.identity")

    if scheduled_ordinals is None:
        selected_entries = entries
    else:
        if isinstance(scheduled_ordinals, str | bytes) or not isinstance(scheduled_ordinals, Sequence):
            raise TypeError("scheduled_ordinals must be a sequence of integers")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in scheduled_ordinals):
            raise TypeError("scheduled_ordinals must contain only integers")
        if len(set(scheduled_ordinals)) != len(scheduled_ordinals):
            raise ValueError("scheduled_ordinals must be unique")
        if list(scheduled_ordinals) != sorted(scheduled_ordinals):
            raise ValueError("scheduled_ordinals must be in plan order")
        by_ordinal = {entry.ordinal: entry for entry in entries}
        try:
            selected_entries = tuple(by_ordinal[ordinal] for ordinal in scheduled_ordinals)
        except KeyError as error:
            raise ValueError(f"scheduled ordinal is outside the candidate plan: {error.args[0]}") from error
    if not selected_entries:
        raise ValueError("attempt must schedule at least one unresolved candidate")

    target = Path(artifact_dir)
    if target.exists() or os.path.lexists(target):
        raise FileExistsError(f"artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    current_stage = "initialize"
    current_context: CandidateContext | None = None
    candidate_records: list[dict[str, Any]] = []
    admissions: list[AdmissionProduct] = []
    try:
        _new_json_file(staging / "dataset.json", dataset)
        _new_json_file(staging / "dataset-manifest.json", dataset_manifest)
        _new_json_file(staging / "candidate-plan.json", plan)
        for entry in selected_entries:
            entry_payload = entry.payload()
            candidate_id = f"{attempt_id}:{entry.task_id}:{entry.sample_id}"
            current_context = CandidateContext(
                attempt_id=attempt_id,
                candidate_id=candidate_id,
                plan_entry=entry,
            )
            candidate_directory = staging / "candidates" / f"{entry.ordinal:05d}"
            candidate_directory.mkdir(parents=True)

            current_stage = "interact"
            interaction = _validated_stage_product(adapter.interact(current_context), stage=current_stage)
            trajectory_path = candidate_directory / "trajectory.json"
            _new_json_file(trajectory_path, interaction.artifact)

            current_stage = "verify"
            verification = _validated_stage_product(
                adapter.verify(current_context, interaction),
                stage=current_stage,
            )
            verification_path = candidate_directory / "verification.json"
            _new_json_file(verification_path, verification.artifact)

            current_stage = "admit"
            admission = adapter.admit(current_context, interaction, verification)
            if not isinstance(admission, AdmissionProduct):
                raise TypeError("admit must return AdmissionProduct")
            _validated_stage_product(admission, stage=current_stage)
            tags = tuple(admission.selection_tags)
            if any(not isinstance(tag, str) or _POLICY_ID_PATTERN.fullmatch(tag) is None for tag in tags):
                raise ValueError("selection tags must be path-safe stable identifiers")
            if len(set(tags)) != len(tags):
                raise ValueError("selection tags must be unique")
            metadata = json_copy(admission.record_metadata, name="admission.record_metadata")
            if not isinstance(metadata, dict):
                raise TypeError("admission.record_metadata must be a mapping")
            _reject_sensitive_keys(metadata, path="admission.record_metadata")
            admission_path = candidate_directory / "admission.json"
            _new_json_file(admission_path, admission.artifact)

            current_stage = "project"
            projection = adapter.project(current_context, interaction, verification, admission)
            projection_ref = None
            projection_file_sha256 = None
            if projection is not None:
                projected = json_copy(projection, name="training projection")
                if not isinstance(projected, dict):
                    raise TypeError("training projection must be a mapping")
                projection_path = candidate_directory / "projection.json"
                _new_json_file(projection_path, projected)
                projection_ref = projection_path.relative_to(staging).as_posix()
                projection_file_sha256 = file_sha256(projection_path)

            current_stage = "materialize_evidence"
            evidence = json_copy(
                adapter.materialize_evidence(
                    candidate_directory,
                    current_context,
                    interaction,
                    verification,
                    admission,
                ),
                name="materialized evidence",
            )
            if not isinstance(evidence, dict):
                raise TypeError("materialize_evidence must return a mapping")
            _reject_sensitive_keys(evidence, path="materialized evidence")

            current_stage = "candidate_record"
            record_path = candidate_directory / "record.json"
            record = with_manifest_digest(
                {
                    "format_version": DATA_CONSTRUCTION_CANDIDATE_RECORD_FORMAT_VERSION,
                    "attempt_id": attempt_id,
                    "candidate_id": candidate_id,
                    "plan_manifest_sha256": plan["manifest_sha256"],
                    "plan_entry_id": entry_payload["plan_entry_id"],
                    "ordinal": entry.ordinal,
                    "task_id": entry.task_id,
                    "sample_id": entry.sample_id,
                    "sampling": copy.deepcopy(dict(entry.sampling)),
                    "trajectory_ref": trajectory_path.relative_to(staging).as_posix(),
                    "trajectory_file_sha256": file_sha256(trajectory_path),
                    "trajectory_manifest_sha256": interaction.artifact["manifest_sha256"],
                    "verification_ref": verification_path.relative_to(staging).as_posix(),
                    "verification_file_sha256": file_sha256(verification_path),
                    "verification_manifest_sha256": verification.artifact["manifest_sha256"],
                    "admission_ref": admission_path.relative_to(staging).as_posix(),
                    "admission_file_sha256": file_sha256(admission_path),
                    "admission_manifest_sha256": admission.artifact["manifest_sha256"],
                    "projection_ref": projection_ref,
                    "projection_file_sha256": projection_file_sha256,
                    "selection_tags": list(tags),
                    "metadata": metadata,
                    "evidence": evidence,
                }
            )
            _new_json_file(record_path, record)
            record_index = {key: value for key, value in record.items() if key != "manifest_sha256"}
            record_index.update(
                {
                    "record_ref": record_path.relative_to(staging).as_posix(),
                    "record_file_sha256": file_sha256(record_path),
                    "record_manifest_sha256": record["manifest_sha256"],
                }
            )
            candidate_records.append(record_index)
            admissions.append(admission)

        current_stage = "summarize"
        summary = json_copy(adapter.summarize(admissions, candidate_records), name="attempt summary")
        if not isinstance(summary, dict):
            raise TypeError("summarize must return a mapping")
        _reject_sensitive_keys(summary, path="attempt summary")

        current_stage = "select"
        raw_selections = adapter.build_selections(candidate_records)
        if not isinstance(raw_selections, Mapping):
            raise TypeError("build_selections must return a mapping")
        selection_refs = {}
        for policy_id, selection in sorted(raw_selections.items()):
            if not isinstance(policy_id, str) or _POLICY_ID_PATTERN.fullmatch(policy_id) is None:
                raise ValueError("selection policy IDs must be path-safe stable identifiers")
            verify_manifest_digest(selection)
            if selection.get("format_version") != DATA_CONSTRUCTION_SELECTION_FORMAT_VERSION:
                raise ValueError("unexpected data-construction selection format")
            if selection.get("policy_id") != policy_id:
                raise ValueError("selection policy key and artifact identity differ")
            selection_path = staging / "selections" / f"{policy_id}.json"
            _new_json_file(selection_path, selection)
            selection_refs[policy_id] = {
                "ref": selection_path.relative_to(staging).as_posix(),
                "file_sha256": file_sha256(selection_path),
                "manifest_sha256": selection["manifest_sha256"],
                "selected_count": selection["selected_count"],
            }

        current_stage = "attempt_manifest"
        attempt_manifest = with_manifest_digest(
            {
                "format_version": DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION,
                "attempt_id": attempt_id,
                "run_identity": identity,
                "adapter_identity": adapter_identity,
                "task_dataset_ref": "dataset.json",
                "task_dataset_sha256": canonical_json_sha256(dataset),
                "task_dataset_manifest_ref": "dataset-manifest.json",
                "task_dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "plan_ref": "candidate-plan.json",
                "plan_manifest_sha256": plan["manifest_sha256"],
                "scheduled_ordinals": [entry.ordinal for entry in selected_entries],
                "candidate_count": len(candidate_records),
                "candidate_record_refs": [record["record_ref"] for record in candidate_records],
                "selection_refs": selection_refs,
                "partial_evidence_preserved_on_failure": True,
                "hidden_verifier_material_allowed_in_projection": False,
            }
        )
        report = with_manifest_digest(
            {
                "format_version": DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION,
                "attempt_id": attempt_id,
                "attempt_manifest_ref": "attempt-manifest.json",
                "attempt_manifest_sha256": attempt_manifest["manifest_sha256"],
                "plan_manifest_sha256": plan["manifest_sha256"],
                "candidate_count": len(candidate_records),
                "candidate_records": candidate_records,
                "summary": summary,
                "selection_refs": selection_refs,
            }
        )
        _new_json_file(staging / "attempt-manifest.json", attempt_manifest)
        _new_json_file(staging / "report.json", report)
        write_artifact_manifest(staging)
        staging.rename(target)
        return copy.deepcopy(report)
    except BaseException as error:
        if staging.exists():
            try:
                failure = with_manifest_digest(
                    {
                        "format_version": DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION,
                        "attempt_id": attempt_id,
                        "plan_manifest_sha256": plan["manifest_sha256"],
                        "failed_stage": current_stage,
                        "failed_ordinal": None if current_context is None else current_context.plan_entry.ordinal,
                        "failed_candidate_id": None if current_context is None else current_context.candidate_id,
                        "completed_candidate_count": len(candidate_records),
                        "exception_type": type(error).__name__,
                        "exception_message_persisted": False,
                    }
                )
                _new_json_file(staging / "failure.json", failure)
            except BaseException:
                pass
            staging_token = staging.name.removeprefix(f".{target.name}.").removesuffix(".tmp")
            failed_target = target.with_name(f"{target.name}.failed-{staging_token}")
            staging.rename(failed_target)
            error.add_note(f"partial data-construction evidence moved to {failed_target}")
        raise


__all__ = [
    "DATA_CONSTRUCTION_ATTEMPT_FORMAT_VERSION",
    "DATA_CONSTRUCTION_CANDIDATE_RECORD_FORMAT_VERSION",
    "DATA_CONSTRUCTION_PLAN_FORMAT_VERSION",
    "DATA_CONSTRUCTION_RESUME_FORMAT_VERSION",
    "DATA_CONSTRUCTION_SELECTION_FORMAT_VERSION",
    "AdmissionProduct",
    "CandidateContext",
    "CandidatePlanEntry",
    "DataConstructionAdapter",
    "StageProduct",
    "build_candidate_plan",
    "build_resume_manifest",
    "build_selection_manifest",
    "candidate_plan_entries",
    "canonical_json_sha256",
    "file_sha256",
    "json_copy",
    "run_data_construction_attempt",
    "verify_artifact_manifest",
    "verify_manifest_digest",
    "write_artifact_manifest",
    "with_manifest_digest",
]
