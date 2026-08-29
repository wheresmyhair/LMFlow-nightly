"""Offline tests for the pinned GSM8K evaluation protocol and reports."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

import lmflow.agentic.evaluate_gsm8k as evaluation_cli
import lmflow.agentic.gsm8k_protocol as protocol
from lmflow.agentic.gsm8k_protocol import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_REVISION,
    GSM8K_DATASET_SOURCE,
    GSM8K_PROTOCOL_SPLIT_SIZES,
    canonical_gsm8k_instance_id,
    canonical_json_sha256,
    classify_evaluation_cases,
    gsm8k_protocol_indices,
    load_pinned_gsm8k_dataset,
    paired_profile_comparison,
    qwen3_tokenizer_identity,
    summarize_evaluation_result,
    summarize_repeated_reports,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.datasets import Dataset


def test_protocol_splits_are_deterministic_disjoint_and_nested():
    train_source, training = gsm8k_protocol_indices("training")
    development_source, development = gsm8k_protocol_indices("development")
    development128_source, development128 = gsm8k_protocol_indices("development128")
    test_source, smoke = gsm8k_protocol_indices("smoke")
    _, repeat = gsm8k_protocol_indices("repeat")
    _, decision = gsm8k_protocol_indices("decision")
    _, heldout = gsm8k_protocol_indices("heldout")

    assert train_source == development_source == development128_source == "train"
    assert test_source == "test"
    assert len(training) == GSM8K_PROTOCOL_SPLIT_SIZES["training"]
    assert len(development) == GSM8K_PROTOCOL_SPLIT_SIZES["development"]
    assert len(development128) == GSM8K_PROTOCOL_SPLIT_SIZES["development128"]
    assert set(training).isdisjoint(development)
    assert set(training).union(development) == set(range(7473))
    assert development128 == development[:128]
    assert smoke == repeat[:16]
    assert repeat == decision[:128]
    assert heldout == tuple(range(1319))
    assert gsm8k_protocol_indices("decision")[1] == decision
    assert (
        canonical_json_sha256([canonical_gsm8k_instance_id("train", index) for index in development128])
        == "4dead906e475ea6e75a35186451d95a7a64d4faabb6dd0d458fd580d3846e633"
    )
    assert (
        canonical_json_sha256([canonical_gsm8k_instance_id("test", index) for index in smoke])
        == "6794b26ec1e879785b410f40b151d0673f5928e31bd1bda0045d7681f8683b0f"
    )


def test_canonical_instance_id_includes_source_revision_and_index():
    assert canonical_gsm8k_instance_id("test", 42) == (
        f"{GSM8K_DATASET_SOURCE}@{GSM8K_DATASET_REVISION}/{GSM8K_DATASET_CONFIG}/test/0000042"
    )


class _SourceDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def to_list(self):
        return copy.deepcopy(self._rows)


def test_pinned_loader_preserves_source_identity_and_hides_content_from_manifest(monkeypatch):
    rows = [{"question": f"Question {index}?", "answer": f"Reasoning. #### {index}"} for index in range(1319)]
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _SourceDataset(rows)

    monkeypatch.setattr(protocol, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(protocol, "_EXPECTED_SOURCE_CONTENT_SHA256", {})

    dataset, manifest = load_pinned_gsm8k_dataset("smoke", cache_dir="fixture-cache")

    assert calls == [
        (
            (GSM8K_DATASET_SOURCE, GSM8K_DATASET_CONFIG),
            {
                "split": "test",
                "revision": GSM8K_DATASET_REVISION,
                "cache_dir": "fixture-cache",
            },
        )
    ]
    assert len(dataset) == 16
    source_indices = [row["source_index"] for row in dataset.to_list()]
    assert source_indices == list(gsm8k_protocol_indices("smoke")[1])
    assert all(row["instance_id"].endswith(f"/{row['source_index']:07d}") for row in dataset.to_list())
    assert manifest["identity_scope"] == "canonical_source"
    assert manifest["revision"] == GSM8K_DATASET_REVISION
    assert manifest["instance_count"] == 16
    assert manifest["ordered_instance_ids_sha256"] == canonical_json_sha256(
        [row["instance_id"] for row in dataset.to_list()]
    )
    verify_manifest_digest(manifest)
    manifest_text = json.dumps(manifest)
    assert "Question " not in manifest_text
    assert "Reasoning" not in manifest_text
    assert "gold" not in manifest_text


def test_development128_loader_preserves_nested_source_identity(monkeypatch):
    rows = [{"question": f"Question {index}?", "answer": f"Reasoning. #### {index}"} for index in range(7473)]
    monkeypatch.setattr(protocol, "load_dataset", lambda *args, **kwargs: _SourceDataset(rows))
    monkeypatch.setattr(protocol, "_EXPECTED_SOURCE_CONTENT_SHA256", {})

    dataset, manifest = load_pinned_gsm8k_dataset("development128")

    source_indices = [row["source_index"] for row in dataset.to_list()]
    assert source_indices == list(gsm8k_protocol_indices("development")[1][:128])
    assert manifest["source_split"] == "train"
    assert manifest["protocol_split"] == "development128"
    assert manifest["selection"] == {
        "algorithm": "sha256-ranked-canonical-id/v1",
        "seed": protocol.GSM8K_SPLIT_SEED,
    }
    assert manifest["instance_count"] == 128
    assert manifest["ordered_instance_ids_sha256"] == canonical_json_sha256(
        [row["instance_id"] for row in dataset.to_list()]
    )
    verify_manifest_digest(manifest)


def test_evaluation_cli_accepts_development128_split():
    arguments = evaluation_cli._build_parser().parse_args(
        [
            "run",
            "--artifact-dir",
            "artifacts/run",
            "--run-id",
            "development128-run",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--served-model-name",
            "Qwen/Qwen3-8B",
            "--tokenizer-path",
            "models/Qwen3-8B",
            "--backend-version",
            "0.25.1",
            "--split",
            "development128",
        ]
    )

    assert arguments.split == "development128"
    assert arguments.tool_call_parser == "hermes"


def test_pinned_loader_rejects_content_drift(monkeypatch):
    rows = [{"question": f"Question {index}?", "answer": f"#### {index}"} for index in range(1319)]
    monkeypatch.setattr(protocol, "load_dataset", lambda *args, **kwargs: _SourceDataset(rows))
    monkeypatch.setattr(protocol, "_EXPECTED_SOURCE_CONTENT_SHA256", {"test": "0" * 64})

    with pytest.raises(ValueError, match="content digest"):
        load_pinned_gsm8k_dataset("smoke")


def test_qwen3_tokenizer_identity_hashes_template_without_persisting_path(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (tokenizer_dir / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")

    identity = qwen3_tokenizer_identity(tokenizer_dir, revision="tokenizer-revision")

    assert identity["revision"] == "tokenizer-revision"
    assert identity["chat_template_sha256"] == hashlib.sha256(b"{{ messages }}").hexdigest()
    assert str(tmp_path) not in repr(identity)


def _usage(seconds, *, model_calls=1, tool_calls=0, tokens=10):
    return {
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "steps": model_calls,
        "wall_time_seconds": seconds,
        "input_tokens": tokens,
        "output_tokens": tokens,
        "cost": 0.0,
    }


def _completed(task_id, correct, seconds=1.0):
    return {
        "task_id": task_id,
        "status": "completed",
        "passed": bool(correct),
        "metrics": {"final_correctness": float(correct), "strict_correctness": float(correct)},
        "usage": _usage(seconds),
        "artifact_ref": None,
        "failure": None,
        "metadata": {},
    }


def _failed(task_id, seconds=2.0):
    return {
        "task_id": task_id,
        "status": "failed",
        "passed": None,
        "metrics": {},
        "usage": _usage(seconds),
        "artifact_ref": None,
        "failure": {"failure_type": "timeout", "message": "deadline", "retryable": True},
        "metadata": {},
    }


def test_report_statistics_include_failures_in_rates_and_latency_quantiles():
    result = {"records": [_completed("a", 1, 1.0), _completed("b", 0, 2.0), _failed("c", 3.0)]}

    summary = summarize_evaluation_result(result)

    assert summary["instance_count"] == 3
    assert summary["status_counts"] == {"completed": 2, "failed": 1}
    assert summary["failure_counts"] == {"timeout": 1}
    assert summary["pass_rate"] == pytest.approx(1 / 3)
    assert summary["metrics"]["final_correctness"]["observed_count"] == 2
    assert summary["metrics"]["final_correctness"]["completed_mean"] == 0.5
    assert summary["metrics"]["final_correctness"]["all_instance_rate"] == pytest.approx(1 / 3)
    assert summary["usage"]["model_calls"] == 3
    assert summary["usage"]["latency_seconds"] == {
        "observed_count": 3,
        "mean": 2.0,
        "p50": 2.0,
        "p95": 2.9,
        "max": 3.0,
    }


def test_paired_profile_comparison_reports_all_four_outcomes():
    direct = {"records": [_completed("a", 1), _completed("b", 1), _completed("c", 0), _failed("d")]}
    calculator = {"records": [_completed("a", 1), _completed("b", 0), _completed("c", 1), _failed("d")]}

    comparison = paired_profile_comparison(direct, calculator, bootstrap_seed=7, bootstrap_samples=100)

    assert comparison["counts"] == {
        "both_correct": 1,
        "direct_only": 1,
        "calculator_only": 1,
        "neither_correct": 1,
    }
    assert comparison["calculator_minus_direct"] == 0.0
    assert comparison["paired_bootstrap_95"]["seed"] == 7


def test_case_classification_keeps_overlapping_quality_and_tool_labels():
    result = {
        "records": [
            _completed("correct", 1),
            {
                **_completed("format", 1),
                "metrics": {"final_correctness": 1.0, "strict_correctness": 0.0},
            },
            {
                **_completed("fallback", 0),
                "metrics": {
                    "final_correctness": 0.0,
                    "strict_correctness": 0.0,
                    "direct_answer_fallback": 1.0,
                    "tool_compliance": 0.0,
                    "recovery": 0.0,
                    "tool_error": 0.0,
                },
            },
            _failed("timeout"),
        ]
    }

    analysis = classify_evaluation_cases(result, profile_name="calculator-tool")

    assert analysis["label_counts"] == {
        "direct_answer_fallback": 1,
        "execution_failure:timeout": 1,
        "incorrect_answer": 1,
        "non_strict_correct": 1,
        "tool_noncompliance": 1,
    }
    assert analysis["attention_case_count"] == 3


def _repeat_report(seed, direct_rate, calculator_rate, delta):
    return with_manifest_digest(
        {
            "format_version": protocol.GSM8K_REPORT_FORMAT_VERSION,
            "dataset_manifest_sha256": "d" * 64,
            "dataset_protocol_sha256": "a" * 64,
            "protocol_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "execution_sha256": "d" * 64,
            "sampling_seed": seed,
            "profiles": {
                "direct-answer": {"statistics": {"pass_rate": direct_rate}},
                "calculator-tool": {"statistics": {"pass_rate": calculator_rate}},
            },
            "paired": {"calculator_minus_direct": delta},
        }
    )


def test_repeat_postprocessor_checks_compatibility_and_reports_variation():
    report = summarize_repeated_reports([_repeat_report(1, 0.5, 0.6, 0.1), _repeat_report(2, 0.7, 0.8, 0.1)])

    assert report["run_count"] == 2
    assert report["seeds"] == [1, 2]
    assert report["profiles"]["direct-answer"]["mean"] == pytest.approx(0.6)
    assert report["profiles"]["direct-answer"]["population_stddev"] == pytest.approx(0.1)
    verify_manifest_digest(report)


def test_manifest_digest_detects_mutation():
    manifest = with_manifest_digest({"value": 1})
    verify_manifest_digest(manifest)
    manifest["value"] = 2

    with pytest.raises(ValueError, match="does not match"):
        verify_manifest_digest(manifest)


class _FakeBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _FakeResult:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return copy.deepcopy(self.payload)


def test_baseline_publishes_digested_path_portable_artifacts(monkeypatch, tmp_path):
    dataset = Dataset.create_from_dict(
        {
            "type": "text2text",
            "instances": [
                {
                    "input": "Question?",
                    "output": "#### 1",
                    "instance_id": "canonical-1",
                    "source_index": 1,
                    "row_digest": "row-digest",
                }
            ],
        }
    )
    dataset_manifest = with_manifest_digest({"instance_count": 1, "dataset_protocol_sha256": "a" * 64})
    monkeypatch.setattr(
        evaluation_cli,
        "load_pinned_gsm8k_dataset",
        lambda split, cache_dir=None: (dataset, dataset_manifest),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "qwen3_tokenizer_identity",
        lambda path, revision: {
            "name": "fixture",
            "revision": revision,
            "tokenizer_config_sha256": "1" * 64,
            "tokenizer_json_sha256": "2" * 64,
            "chat_template_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(evaluation_cli, "OpenAICompatibleCompletionBackend", _FakeBackend)

    def fake_run_profile(*, profile_name, artifact_dir, **kwargs):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = artifact_dir / "record.json"
        artifact.write_text('{"model_visible":"safe"}\n', encoding="utf-8")
        correct = int(profile_name == "direct-answer")
        return _FakeResult({"records": [{**_completed("canonical-1", correct), "artifact_ref": str(artifact)}]})

    monkeypatch.setattr(evaluation_cli, "_run_profile", fake_run_profile)
    target = tmp_path / "run"

    report = evaluation_cli.run_gsm8k_baseline(
        artifact_dir=target,
        run_id="fixture-run",
        base_url="http://secret-host/v1",
        served_model_name="served-qwen",
        tokenizer_path=tmp_path / "machine-tokenizer-path",
        backend_version="0.25.1",
        bootstrap_samples=10,
    )

    assert target.is_dir()
    assert (target / "dataset_manifest.json").is_file()
    assert (target / "run_manifest.json").is_file()
    assert (target / "profiles/direct-answer/result.json").is_file()
    verify_manifest_digest(report)
    direct_result = json.loads((target / "profiles/direct-answer/result.json").read_text(encoding="utf-8"))
    assert direct_result["records"][0]["artifact_ref"] == "profiles/direct-answer/records/record.json"
    manifest_text = (target / "run_manifest.json").read_text(encoding="utf-8")
    assert "secret-host" not in manifest_text
    assert "machine-tokenizer-path" not in manifest_text
    assert "provider-specific runner/backend behavior" in manifest_text
    run_manifest = json.loads(manifest_text)
    assert run_manifest["execution"]["served_engine"]["tool_call_parser"] == "hermes"


def test_baseline_rejects_machine_paths_in_persisted_model_identity(tmp_path):
    with pytest.raises(ValueError, match="portable identity"):
        evaluation_cli.run_gsm8k_baseline(
            artifact_dir=tmp_path / "run",
            run_id="fixture-run",
            base_url="http://127.0.0.1:8000/v1",
            served_model_name="/home/example/model",
            tokenizer_path=tmp_path,
            backend_version="0.25.1",
        )
