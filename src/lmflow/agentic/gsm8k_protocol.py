"""Pinned GSM8K baseline data, provenance, and report helpers.

This module deliberately stays at the GSM8K benchmark boundary. The generic
Evaluator owns per-run execution records; this layer pins source identity,
defines benchmark subsets, and aggregates one or more Evaluator results.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal

from datasets import load_dataset

from lmflow.datasets import Dataset
from lmflow.pipeline.evaluation.result import EvaluationResult

GSM8K_PROTOCOL_FORMAT_VERSION = "lmflow.gsm8k-evaluation-protocol/v2"
GSM8K_REPORT_FORMAT_VERSION = "lmflow.gsm8k-evaluation-report/v1"
GSM8K_DATASET_SOURCE = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
GSM8K_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_MODEL_ID = "Qwen/Qwen3-8B"
GSM8K_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GSM8K_SPLIT_SEED = 20260815
GSM8K_SOURCE_SPLIT_SIZES = {"train": 7473, "test": 1319}
GSM8K_PROTOCOL_SPLIT_SIZES = {
    "training": 6961,
    "development": 512,
    "development128": 128,
    "smoke": 16,
    "repeat": 128,
    "decision": 512,
    "heldout": 1319,
}

ProtocolSplit = Literal[
    "training",
    "development",
    "development128",
    "smoke",
    "repeat",
    "decision",
    "heldout",
]

_EXPECTED_SOURCE_CONTENT_SHA256 = {
    "train": "de809f480930a9568f011158cf15c6d8bad5eda51efb29216a324c30d5f76ff5",
    "test": "7d7e49d1579b2b660b2dc3d0a7557261e875e79980da2fd0d45c0625c7566811",
}


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON value using the protocol's canonical encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def with_manifest_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON copy with a digest over every field except the digest."""

    if not isinstance(payload, Mapping):
        raise TypeError("manifest payload must be a mapping")
    if "manifest_sha256" in payload:
        raise ValueError("manifest payload must not already contain manifest_sha256")
    copied = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    copied["manifest_sha256"] = canonical_json_sha256(copied)
    return copied


def verify_manifest_digest(manifest: Mapping[str, Any]) -> None:
    """Fail closed when a stored manifest is malformed or has changed."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    expected = manifest.get("manifest_sha256")
    if not _is_sha256(expected):
        raise ValueError("manifest must contain a SHA-256 manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_json_sha256(payload) != expected:
        raise ValueError("manifest_sha256 does not match the manifest content")


def canonical_gsm8k_instance_id(source_split: str, source_index: int) -> str:
    """Build an instance identity stable across subset materializations."""

    if source_split not in GSM8K_SOURCE_SPLIT_SIZES:
        raise ValueError(f"unsupported GSM8K source split {source_split!r}")
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise TypeError("source_index must be an integer")
    if not 0 <= source_index < GSM8K_SOURCE_SPLIT_SIZES[source_split]:
        raise ValueError(f"source_index is outside the pinned {source_split!r} split")
    return f"{GSM8K_DATASET_SOURCE}@{GSM8K_DATASET_REVISION}/{GSM8K_DATASET_CONFIG}/{source_split}/{source_index:07d}"


def _ranked_indices(source_split: str) -> list[int]:
    indices = range(GSM8K_SOURCE_SPLIT_SIZES[source_split])
    return sorted(
        indices,
        key=lambda index: (
            hashlib.sha256(f"{GSM8K_SPLIT_SEED}\0{canonical_gsm8k_instance_id(source_split, index)}".encode()).digest(),
            index,
        ),
    )


def gsm8k_protocol_indices(split: ProtocolSplit) -> tuple[str, tuple[int, ...]]:
    """Resolve one named protocol split to a source split and ordered indices."""

    if split not in GSM8K_PROTOCOL_SPLIT_SIZES:
        raise ValueError(f"unsupported GSM8K protocol split {split!r}")
    if split in {"development", "development128"}:
        return "train", tuple(_ranked_indices("train")[: GSM8K_PROTOCOL_SPLIT_SIZES[split]])
    if split == "training":
        development = set(_ranked_indices("train")[: GSM8K_PROTOCOL_SPLIT_SIZES["development"]])
        return "train", tuple(index for index in range(GSM8K_SOURCE_SPLIT_SIZES["train"]) if index not in development)
    if split == "heldout":
        return "test", tuple(range(GSM8K_SOURCE_SPLIT_SIZES["test"]))
    return "test", tuple(_ranked_indices("test")[: GSM8K_PROTOCOL_SPLIT_SIZES[split]])


def gsm8k_row_digest(question: str, answer: str) -> str:
    """Hash the complete official source row without exposing it in manifests."""

    if not isinstance(question, str) or not question:
        raise ValueError("GSM8K question must be a non-empty string")
    if not isinstance(answer, str) or not answer:
        raise ValueError("GSM8K answer must be a non-empty string")
    return canonical_json_sha256({"answer": answer, "question": question})


def _source_content_digest(rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    row_digests: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"GSM8K source row {index} must be a mapping")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question:
            raise ValueError(f"GSM8K source row {index} question must be a non-empty string")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"GSM8K source row {index} answer must be a non-empty string")
        row_digests.append(gsm8k_row_digest(question, answer))
    return canonical_json_sha256(row_digests), row_digests


def load_pinned_gsm8k_dataset(
    split: ProtocolSplit,
    *,
    cache_dir: str | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Load one pinned subset and return an LMFlow Dataset plus its manifest."""

    source_split, source_indices = gsm8k_protocol_indices(split)
    source_dataset = load_dataset(
        GSM8K_DATASET_SOURCE,
        GSM8K_DATASET_CONFIG,
        split=source_split,
        revision=GSM8K_DATASET_REVISION,
        cache_dir=cache_dir,
    )
    if len(source_dataset) != GSM8K_SOURCE_SPLIT_SIZES[source_split]:
        raise ValueError(
            f"pinned GSM8K {source_split!r} contains {len(source_dataset)} rows; "
            f"expected {GSM8K_SOURCE_SPLIT_SIZES[source_split]}"
        )
    source_rows = source_dataset.to_list()
    source_content_sha256, source_row_digests = _source_content_digest(source_rows)
    expected_source_digest = _EXPECTED_SOURCE_CONTENT_SHA256.get(source_split)
    if expected_source_digest is not None and source_content_sha256 != expected_source_digest:
        raise ValueError(f"pinned GSM8K {source_split!r} content digest does not match the protocol")

    instances: list[dict[str, Any]] = []
    instance_manifest: list[dict[str, Any]] = []
    for source_index in source_indices:
        row = source_rows[source_index]
        instance_id = canonical_gsm8k_instance_id(source_split, source_index)
        row_digest = source_row_digests[source_index]
        instances.append(
            {
                "input": row["question"],
                "output": row["answer"],
                "instance_id": instance_id,
                "source_index": source_index,
                "row_digest": row_digest,
            }
        )
        instance_manifest.append(
            {
                "instance_id": instance_id,
                "source_index": source_index,
                "row_sha256": row_digest,
            }
        )

    dataset = Dataset.create_from_dict({"type": "text2text", "instances": instances})
    ordered_instance_ids = [item["instance_id"] for item in instance_manifest]
    if split == "heldout":
        selection = {"algorithm": "canonical-source-order/v1", "seed": None}
    elif split == "training":
        selection = {
            "algorithm": "canonical-source-order-excluding-development/v1",
            "seed": GSM8K_SPLIT_SEED,
        }
    else:
        selection = {"algorithm": "sha256-ranked-canonical-id/v1", "seed": GSM8K_SPLIT_SEED}
    dataset_identity = {
        "source": GSM8K_DATASET_SOURCE,
        "config": GSM8K_DATASET_CONFIG,
        "revision": GSM8K_DATASET_REVISION,
        "source_split": source_split,
        "source_content_sha256": source_content_sha256,
        "protocol_split": split,
        "selection": selection,
        "instance_count": len(instance_manifest),
        "ordered_instance_ids_sha256": canonical_json_sha256(ordered_instance_ids),
        "instance_manifest_sha256": canonical_json_sha256(instance_manifest),
    }
    manifest = with_manifest_digest(
        {
            "format_version": GSM8K_PROTOCOL_FORMAT_VERSION,
            "identity_scope": "canonical_source",
            **dataset_identity,
            "dataset_protocol_sha256": canonical_json_sha256(dataset_identity),
            "source_instance_count": len(source_rows),
            "dataset_fingerprint": dataset.get_fingerprint(),
            "instances": instance_manifest,
        }
    )
    return dataset, manifest


def file_sha256(path: str | Path) -> str:
    """Hash a local identity file without retaining its machine path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def qwen3_tokenizer_identity(path: str | Path, *, revision: str) -> dict[str, Any]:
    """Capture the tokenizer and chat-template identity used by a served model."""

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("tokenizer revision must be a non-empty string")
    tokenizer_dir = Path(path)
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    with tokenizer_config_path.open(encoding="utf-8") as input_file:
        tokenizer_config = json.load(input_file)
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer_config.json must contain a non-empty chat_template")
    return {
        "name": GSM8K_MODEL_ID,
        "revision": revision,
        "tokenizer_config_sha256": file_sha256(tokenizer_config_path),
        "tokenizer_json_sha256": file_sha256(tokenizer_json_path),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> dict[str, float]:
    """Compute a Wilson score interval for a binary rate."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if not 0 <= successes <= total or total == 0:
        raise ValueError("successes and total must satisfy 0 <= successes <= total and total > 0")
    if confidence != 0.95:
        raise ValueError("only the pre-registered 95% confidence level is supported")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"confidence": confidence, "lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _result_dict(result: EvaluationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, EvaluationResult):
        return result.to_dict()
    if not isinstance(result, Mapping):
        raise TypeError("evaluation result must be EvaluationResult or a mapping")
    return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))


def summarize_evaluation_result(result: EvaluationResult | Mapping[str, Any]) -> dict[str, Any]:
    """Add all-instance rates, Wilson intervals, failures, and latency quantiles."""

    payload = _result_dict(result)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation result must contain at least one record")
    task_ids = [record.get("task_id") for record in records if isinstance(record, Mapping)]
    if len(task_ids) != len(records) or any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("every evaluation record must contain a task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("evaluation result task_id values must be unique")
    if any(record.get("status") not in {"completed", "failed"} for record in records):
        raise ValueError("evaluation record status must be completed or failed")

    status_counts = Counter(record.get("status") for record in records)
    failed_records = [record for record in records if record.get("status") == "failed"]
    if any(not isinstance(record.get("failure"), Mapping) for record in failed_records):
        raise ValueError("failed evaluation records must contain a failure mapping")
    failure_counts = Counter(record.get("failure", {}).get("failure_type") for record in failed_records)
    if any(not isinstance(name, str) or not name for name in failure_counts):
        raise ValueError("failed evaluation records must contain a failure_type")
    metric_names = sorted(
        {name for record in records if isinstance(record.get("metrics"), Mapping) for name in record["metrics"]}
    )
    metrics: dict[str, Any] = {}
    for name in metric_names:
        observed = [record["metrics"][name] for record in records if name in record.get("metrics", {})]
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in observed):
            raise ValueError(f"evaluation metric {name!r} must be numeric")
        metric: dict[str, Any] = {"observed_count": len(observed), "completed_mean": fmean(observed)}
        if all(value in {0, 1} for value in observed):
            successes = sum(int(value) for value in observed)
            metric.update(
                {
                    "successes": successes,
                    "all_instance_rate": successes / len(records),
                    "wilson_95": wilson_interval(successes, len(records)),
                }
            )
        metrics[name] = metric

    usage_records = [record["usage"] for record in records if isinstance(record.get("usage"), Mapping)]
    latencies = [usage["wall_time_seconds"] for usage in usage_records if usage.get("wall_time_seconds") is not None]
    usage_totals: dict[str, Any] = {"observed_count": len(usage_records)}
    for name in ("model_calls", "tool_calls", "steps", "input_tokens", "output_tokens", "cost"):
        values = [usage[name] for usage in usage_records if usage.get(name) is not None]
        usage_totals[name] = sum(values) if len(values) == len(usage_records) else None
    usage_totals["latency_seconds"] = (
        {
            "observed_count": len(latencies),
            "mean": fmean(latencies),
            "p50": _quantile(latencies, 0.5),
            "p95": _quantile(latencies, 0.95),
            "max": max(latencies),
        }
        if latencies
        else {"observed_count": 0}
    )
    passed = sum(record.get("passed") is True for record in records)
    return {
        "instance_count": len(records),
        "ordered_task_ids_sha256": canonical_json_sha256(task_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_counts": {str(key): value for key, value in sorted(failure_counts.items())},
        "passed": passed,
        "pass_rate": passed / len(records),
        "pass_rate_wilson_95": wilson_interval(passed, len(records)),
        "metrics": metrics,
        "usage": usage_totals,
    }


def classify_evaluation_cases(
    result: EvaluationResult | Mapping[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any]:
    """Classify per-instance correctness, formatting, tool, and runtime issues."""

    if profile_name not in {"direct-answer", "calculator-tool"}:
        raise ValueError("profile_name must be direct-answer or calculator-tool")
    records = _result_dict(result).get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation result must contain at least one record")
    label_counts = Counter()
    attention_cases = []
    for record in records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("every evaluation record must contain a task_id")
        labels: list[str] = []
        if record.get("status") == "failed":
            failure = record.get("failure")
            failure_type = failure.get("failure_type") if isinstance(failure, Mapping) else None
            if not isinstance(failure_type, str) or not failure_type:
                raise ValueError("failed evaluation records must contain a failure_type")
            labels.append(f"execution_failure:{failure_type}")
        elif record.get("status") == "completed":
            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError("completed evaluation records must contain metrics")
            final_correctness = metrics.get("final_correctness")
            strict_correctness = metrics.get("strict_correctness")
            if final_correctness not in {0, 1} or strict_correctness not in {0, 1}:
                raise ValueError("completed GSM8K records must contain binary correctness metrics")
            if final_correctness == 0:
                labels.append("incorrect_answer")
            elif strict_correctness == 0:
                labels.append("non_strict_correct")
            if profile_name == "calculator-tool":
                if metrics.get("direct_answer_fallback") == 1:
                    labels.append("direct_answer_fallback")
                if metrics.get("tool_compliance") == 0:
                    labels.append("tool_noncompliance")
            if metrics.get("recovery") == 1:
                labels.append("recovery")
            if metrics.get("tool_error") == 1:
                labels.append("tool_error")
        else:
            raise ValueError("evaluation record status must be completed or failed")
        label_counts.update(labels)
        if labels:
            attention_cases.append(
                {
                    "task_id": task_id,
                    "labels": labels,
                    "artifact_ref": record.get("artifact_ref"),
                }
            )
    return {
        "label_counts": dict(sorted(label_counts.items())),
        "attention_case_count": len(attention_cases),
        "attention_cases": attention_cases,
    }


def _record_correctness(record: Mapping[str, Any]) -> int:
    value = record.get("metrics", {}).get("final_correctness")
    return int(value == 1 or value == 1.0)


def paired_profile_comparison(
    direct_result: EvaluationResult | Mapping[str, Any],
    calculator_result: EvaluationResult | Mapping[str, Any],
    *,
    bootstrap_seed: int = GSM8K_SPLIT_SEED,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    """Compare direct and calculator outcomes on exactly the same instances."""

    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise TypeError("bootstrap_samples must be an integer")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    direct_records = _result_dict(direct_result).get("records")
    calculator_records = _result_dict(calculator_result).get("records")
    if not isinstance(direct_records, list) or not isinstance(calculator_records, list):
        raise ValueError("both evaluation results must contain records")
    direct_by_id = {record["task_id"]: record for record in direct_records}
    calculator_by_id = {record["task_id"]: record for record in calculator_records}
    if len(direct_by_id) != len(direct_records) or len(calculator_by_id) != len(calculator_records):
        raise ValueError("profile results must contain unique task_id values")
    ordered_ids = [record["task_id"] for record in direct_records]
    if set(ordered_ids) != set(calculator_by_id):
        raise ValueError("direct and calculator profiles must evaluate the same task_id set")

    differences: list[int] = []
    pairs = Counter()
    for task_id in ordered_ids:
        direct_correct = _record_correctness(direct_by_id[task_id])
        calculator_correct = _record_correctness(calculator_by_id[task_id])
        differences.append(calculator_correct - direct_correct)
        pair_name = {
            (1, 1): "both_correct",
            (1, 0): "direct_only",
            (0, 1): "calculator_only",
            (0, 0): "neither_correct",
        }[(direct_correct, calculator_correct)]
        pairs[pair_name] += 1
    if not differences:
        raise ValueError("paired comparison requires at least one record")

    random_source = random.Random(bootstrap_seed)
    sample_size = len(differences)
    bootstrap_deltas = sorted(
        sum(differences[random_source.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(bootstrap_samples)
    )
    return {
        "instance_count": sample_size,
        "ordered_task_ids_sha256": canonical_json_sha256(ordered_ids),
        "counts": {name: pairs[name] for name in ("both_correct", "direct_only", "calculator_only", "neither_correct")},
        "calculator_minus_direct": fmean(differences),
        "paired_bootstrap_95": {
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "lower": _quantile(bootstrap_deltas, 0.025),
            "upper": _quantile(bootstrap_deltas, 0.975),
        },
    }


def summarize_repeated_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize compatible GSM8K report files across sampling seeds."""

    if len(reports) < 2:
        raise ValueError("repeat summary requires at least two reports")
    for report in reports:
        verify_manifest_digest(report)
        if report.get("format_version") != GSM8K_REPORT_FORMAT_VERSION:
            raise ValueError("repeat report format_version does not match the GSM8K report protocol")
        for name in ("dataset_protocol_sha256", "protocol_sha256", "model_sha256", "execution_sha256"):
            if not _is_sha256(report.get(name)):
                raise ValueError(f"repeat report {name} must be a SHA-256")
    seeds = [report.get("sampling_seed") for report in reports]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("repeat report sampling_seed values must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("repeat reports must use distinct sampling_seed values")
    compatibility = {
        (
            report.get("dataset_protocol_sha256"),
            report.get("protocol_sha256"),
            report.get("model_sha256"),
            report.get("execution_sha256"),
        )
        for report in reports
    }
    if len(compatibility) != 1:
        raise ValueError("repeat reports do not share dataset, protocol, model, and execution identity")

    profile_names = set(reports[0].get("profiles", {}))
    if not profile_names or any(set(report.get("profiles", {})) != profile_names for report in reports):
        raise ValueError("repeat reports must contain the same profiles")
    profiles: dict[str, Any] = {}
    for profile_name in sorted(profile_names):
        values = [report["profiles"][profile_name]["statistics"]["pass_rate"] for report in reports]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        ):
            raise ValueError(f"repeat report profile {profile_name!r} pass_rate values must be in [0, 1]")
        profiles[profile_name] = {
            "values": values,
            "mean": fmean(values),
            "population_stddev": pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    paired_values = [report["paired"]["calculator_minus_direct"] for report in reports]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not -1 <= value <= 1
        for value in paired_values
    ):
        raise ValueError("repeat report paired deltas must be finite values in [-1, 1]")
    return with_manifest_digest(
        {
            "format_version": GSM8K_REPORT_FORMAT_VERSION,
            "run_count": len(reports),
            "seeds": seeds,
            "dataset_protocol_sha256": reports[0]["dataset_protocol_sha256"],
            "protocol_sha256": reports[0]["protocol_sha256"],
            "model_sha256": reports[0]["model_sha256"],
            "execution_sha256": reports[0]["execution_sha256"],
            "profiles": profiles,
            "paired_calculator_minus_direct": {
                "values": paired_values,
                "mean": fmean(paired_values),
                "population_stddev": pstdev(paired_values),
                "min": min(paired_values),
                "max": max(paired_values),
            },
        }
    )


__all__ = [
    "GSM8K_DATASET_CONFIG",
    "GSM8K_DATASET_REVISION",
    "GSM8K_DATASET_SOURCE",
    "GSM8K_MODEL_ID",
    "GSM8K_MODEL_REVISION",
    "GSM8K_PROTOCOL_FORMAT_VERSION",
    "GSM8K_PROTOCOL_SPLIT_SIZES",
    "GSM8K_REPORT_FORMAT_VERSION",
    "GSM8K_SOURCE_SPLIT_SIZES",
    "GSM8K_SPLIT_SEED",
    "canonical_gsm8k_instance_id",
    "canonical_json_sha256",
    "classify_evaluation_cases",
    "file_sha256",
    "gsm8k_protocol_indices",
    "gsm8k_row_digest",
    "load_pinned_gsm8k_dataset",
    "paired_profile_comparison",
    "qwen3_tokenizer_identity",
    "summarize_evaluation_result",
    "summarize_repeated_reports",
    "verify_manifest_digest",
    "wilson_interval",
    "with_manifest_digest",
]
