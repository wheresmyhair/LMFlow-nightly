import copy
import json

import pytest

import lmflow.agentic.gsm8k_cold_start as cold_start
import lmflow.agentic.prepare_gsm8k_cold_start as cold_start_cli
from lmflow.agentic.gsm8k_cold_start import (
    GSM8KColdStartProjectionError,
    build_gsm8k_cold_start_payload,
    project_gsm8k_annotated_row,
    run_gsm8k_cold_start_factory,
)
from lmflow.agentic.gsm8k_protocol import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_REVISION,
    GSM8K_DATASET_SOURCE,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.args import DatasetArguments
from lmflow.datasets import Dataset


def _solution(call_count, *, mismatch=False):
    lines = []
    final = None
    for index in range(call_count):
        left = index + 2
        right = index + 3
        result = left + right
        annotation_result = result + 1 if mismatch and index == 0 else result
        lines.append(f"Step {index + 1}: {left}+{right} = <<{left}+{right}={annotation_result}>>{annotation_result}.")
        final = annotation_result
    return "\n".join(lines) + f"\n#### {final}"


def _row(source_index, call_count, *, mismatch=False):
    return {
        "input": f"Compute fixture {source_index}.",
        "output": _solution(call_count, mismatch=mismatch),
        "instance_id": f"{GSM8K_DATASET_SOURCE}@{GSM8K_DATASET_REVISION}/main/train/{source_index:07d}",
        "source_index": source_index,
        "row_digest": f"{source_index:064x}",
    }


def _source_manifest(instance_count):
    instances = [
        {
            "instance_id": f"{GSM8K_DATASET_SOURCE}@{GSM8K_DATASET_REVISION}/main/train/{index:07d}",
            "source_index": index,
            "row_sha256": f"{index:064x}",
        }
        for index in range(instance_count)
    ]
    return with_manifest_digest(
        {
            "format_version": "fixture",
            "identity_scope": "canonical_source",
            "source": GSM8K_DATASET_SOURCE,
            "config": GSM8K_DATASET_CONFIG,
            "revision": GSM8K_DATASET_REVISION,
            "source_split": "train",
            "protocol_split": "training",
            "instance_count": instance_count,
            "dataset_protocol_sha256": "a" * 64,
            "instances": instances,
        }
    )


def _fixture_dataset():
    rows = []
    source_index = 0
    for call_count in range(1, 5):
        for _ in range(3):
            rows.append(_row(source_index, call_count))
            source_index += 1
    rows.extend(
        [
            {
                **_row(source_index, 1),
                "output": "No calculator annotation.\n#### 5",
            },
            _row(source_index + 1, 5),
            _row(source_index + 2, 1, mismatch=True),
        ]
    )
    return Dataset.create_from_dict({"type": "text2text", "instances": rows})


def test_projects_public_annotations_to_authentic_calculator_and_direct_views():
    projection = project_gsm8k_annotated_row(_row(42, 2))

    assert projection.tool_call_count == 2
    assert projection.tool_conversation["conversation_id"].endswith(":calculator")
    assert projection.direct_conversation["conversation_id"].endswith(":direct")
    assert "tools" not in projection.direct_conversation
    assert "<<" not in projection.direct_conversation["messages"][-1]["content"]
    tool_messages = [message for message in projection.tool_conversation["messages"] if message["role"] == "tool"]
    assert [message["content"] for message in tool_messages] == ["Calculator result: 5", "Calculator result: 7"]
    assistant_calls = [
        message["tool_calls"][0]
        for message in projection.tool_conversation["messages"]
        if message["role"] == "assistant" and "tool_calls" in message
    ]
    assert [json.loads(call["function"]["arguments"]) for call in assistant_calls] == [
        {"expression": "2+3"},
        {"expression": "3+4"},
    ]
    assert all(call["function"]["name"] == "calculate" for call in assistant_calls)
    assert all("loss" not in message for message in projection.tool_conversation["messages"])
    assert projection.replay["data_class"] == "A"
    assert projection.replay["official_final_verifier_passed"] is True
    assert projection.replay["invalid_or_forged_observation_count"] == 0
    verify_manifest_digest(projection.replay)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("No annotation.\n#### 5", "no_annotation"),
        (_solution(5), "over_tool_budget"),
        (_solution(1, mismatch=True), "calculator_replay_mismatch"),
        ("Broken <<2+3>>5.\n#### 5", "malformed_annotation"),
    ],
)
def test_projection_rejects_non_admissible_rows(output, reason):
    row = _row(0, 1)
    row["output"] = output

    with pytest.raises(GSM8KColdStartProjectionError) as error:
        project_gsm8k_annotated_row(row)

    assert error.value.reason == reason


def test_builds_deterministic_balanced_nested_A_only_dataset():
    dataset = _fixture_dataset()
    source_manifest = _source_manifest(len(dataset))

    payload8, manifest8, replays8 = build_gsm8k_cold_start_payload(
        dataset,
        source_manifest,
        task_count=8,
    )
    payload12, manifest12, _ = build_gsm8k_cold_start_payload(
        dataset,
        source_manifest,
        task_count=12,
    )

    assert payload8["type"] == "conversation"
    assert len(payload8["instances"]) == 16
    assert manifest8["summary"] == {
        "source_training_count": 15,
        "eligible_A_count": 12,
        "rejected_count": 3,
        "rejection_counts": {
            "calculator_replay_mismatch": 1,
            "no_annotation": 1,
            "over_tool_budget": 1,
        },
        "selected_task_count": 8,
        "conversation_count": 16,
        "tool_conversation_count": 8,
        "direct_conversation_count": 8,
        "tool_call_count": 20,
        "selected_tool_call_distribution": {"1": 2, "2": 2, "3": 2, "4": 2},
        "official_replay_pass_count": 8,
        "data_class_counts": {"A": 8, "B": 0, "C": 0, "D": 0, "E": 0},
        "hidden_verifier_material_included": False,
    }
    assert [item["instance_id"] for item in manifest8["selected_instances"]] == [
        item["instance_id"] for item in manifest12["selected_instances"][:8]
    ]
    assert len(replays8) == 8
    assert all(replay["data_class"] == "A" for replay in replays8)
    verify_manifest_digest(manifest8)


def test_build_fails_closed_when_source_manifest_does_not_match_materialized_rows():
    dataset = _fixture_dataset()
    source_manifest = _source_manifest(len(dataset))
    source_manifest.pop("manifest_sha256")
    source_manifest["instances"][0]["row_sha256"] = "f" * 64
    source_manifest = with_manifest_digest(source_manifest)

    with pytest.raises(ValueError, match="does not match the materialized row"):
        build_gsm8k_cold_start_payload(
            dataset,
            source_manifest,
            task_count=8,
        )


def test_visible_dataset_and_provenance_exclude_reward_tool_and_hidden_keys():
    dataset = _fixture_dataset()
    payload, manifest, replays = build_gsm8k_cold_start_payload(
        dataset,
        _source_manifest(len(dataset)),
        task_count=8,
    )

    serialized_dataset = json.dumps(payload, sort_keys=True)
    serialized_provenance = json.dumps({"manifest": manifest, "replays": replays}, sort_keys=True)
    for forbidden in ("calc_gsm8k_reward", "ground_truth", "gold_answer"):
        assert forbidden not in serialized_dataset
        assert forbidden not in serialized_provenance
    assert '"reward"' not in serialized_dataset
    assert '"verifier_material"' not in serialized_provenance
    assert manifest["projection"]["hidden_verifier_material_included"] is False


def _fake_token_summary(conversations, **_kwargs):
    records = [
        {
            "conversation_id": conversation["conversation_id"],
            "total_tokens": 100,
            "assistant_loss_tokens": 40,
        }
        for conversation in conversations
    ]
    return (
        {
            "tokenizer": {
                "name": "Qwen/Qwen3-8B",
                "revision": "fixture",
                "tokenizer_config_sha256": "1" * 64,
                "tokenizer_json_sha256": "2" * 64,
                "chat_template_sha256": "3" * 64,
            },
            "block_size": 2048,
            "conversation_count": len(records),
            "total_tokens": 100 * len(records),
            "assistant_loss_tokens": 40 * len(records),
            "assistant_loss_tokens_mean": 40.0,
            "assistant_loss_tokens_p50": 40,
            "assistant_loss_tokens_p95": 40,
            "max_conversation_tokens": 100,
            "truncated_conversation_count": 0,
        },
        records,
    )


def test_factory_atomically_publishes_loadable_dataset_and_portable_manifests(tmp_path, monkeypatch):
    dataset = _fixture_dataset()
    source_manifest = _source_manifest(len(dataset))
    monkeypatch.setattr(cold_start, "load_pinned_gsm8k_dataset", lambda *_args, **_kwargs: (dataset, source_manifest))
    monkeypatch.setattr(cold_start, "_tokenize_conversations", _fake_token_summary)
    artifact_dir = tmp_path / "e0"

    report = run_gsm8k_cold_start_factory(
        artifact_dir=artifact_dir,
        run_id="e0-fixture",
        task_count=8,
        tokenizer_path=tmp_path / "tokenizer",
    )

    assert report["summary"]["selected_task_count"] == 8
    assert report["summary"]["tokens"]["assistant_loss_tokens"] == 640
    verify_manifest_digest(report)
    assert (artifact_dir / "dataset" / "data.json").is_file()
    assert (artifact_dir / "data_manifest.json").is_file()
    assert (artifact_dir / "source_dataset_manifest.json").is_file()
    assert (artifact_dir / "replay.jsonl").is_file()
    artifact_manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    verify_manifest_digest(artifact_manifest)
    assert "tokenizer_path" not in json.dumps(artifact_manifest)
    loaded = Dataset(
        DatasetArguments(
            dataset_path=str(artifact_dir / "dataset"),
            dataset_cache_dir=str(tmp_path / "cache"),
        )
    )
    assert loaded.get_type() == "conversation"
    assert len(loaded) == 16


def test_factory_removes_staging_output_on_tokenization_failure(tmp_path, monkeypatch):
    dataset = _fixture_dataset()
    source_manifest = _source_manifest(len(dataset))
    monkeypatch.setattr(cold_start, "load_pinned_gsm8k_dataset", lambda *_args, **_kwargs: (dataset, source_manifest))

    def fail_tokenization(*_args, **_kwargs):
        raise RuntimeError("mask failure")

    monkeypatch.setattr(cold_start, "_tokenize_conversations", fail_tokenization)
    artifact_dir = tmp_path / "e0"

    with pytest.raises(RuntimeError, match="mask failure"):
        run_gsm8k_cold_start_factory(
            artifact_dir=artifact_dir,
            run_id="e0-fixture",
            task_count=8,
            tokenizer_path=tmp_path / "tokenizer",
        )

    assert not artifact_dir.exists()
    assert list(tmp_path.glob(".e0.*.tmp")) == []


def test_cli_returns_report_without_exposing_tokenizer_path(tmp_path, monkeypatch, capsys):
    report = with_manifest_digest({"format_version": "fixture", "summary": {"selected_task_count": 8}})
    calls = []

    def fake_factory(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return report

    monkeypatch.setattr(cold_start_cli, "run_gsm8k_cold_start_factory", fake_factory)

    assert (
        cold_start_cli.main(
            [
                "--artifact-dir",
                str(tmp_path / "e0"),
                "--run-id",
                "e0-cli",
                "--tokenizer-path",
                str(tmp_path / "private-tokenizer-path"),
            ]
        )
        == 0
    )

    assert calls[0]["task_count"] == 8
    assert str(tmp_path / "private-tokenizer-path") not in capsys.readouterr().out
