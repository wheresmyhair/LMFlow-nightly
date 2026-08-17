"""Verified cold-start conversations from public GSM8K calculator annotations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean
from typing import Any

from lmflow.agentic.gsm8k import extract_gsm8k_answer
from lmflow.agentic.gsm8k_evaluation import (
    GSM8K_CALCULATOR_SYSTEM_PROMPT,
    GSM8K_CALCULATOR_TOOL,
    GSM8K_CALCULATOR_TOOL_NAME,
    GSM8K_DIRECT_SYSTEM_PROMPT,
    GSM8K_USER_PROMPT,
    CalculatorArithmeticError,
    CalculatorExpressionError,
    evaluate_arithmetic_expression,
)
from lmflow.agentic.gsm8k_protocol import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_REVISION,
    GSM8K_DATASET_SOURCE,
    GSM8K_MODEL_ID,
    GSM8K_MODEL_REVISION,
    canonical_json_sha256,
    file_sha256,
    load_pinned_gsm8k_dataset,
    qwen3_tokenizer_identity,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.datasets import Dataset
from lmflow.utils.conversation_template.qwen import QWEN3_TEMPLATE

GSM8K_COLD_START_FORMAT_VERSION = "lmflow.gsm8k-cold-start/v1"
GSM8K_COLD_START_SELECTION_SEED = 20260817
GSM8K_COLD_START_MAX_TOOL_CALLS = 4
GSM8K_COLD_START_DEFAULT_BLOCK_SIZE = 2048

_ANNOTATION_PATTERN = re.compile(r"<<([^<>]+)>>")
_THOUSANDS_SEPARATOR_PATTERN = re.compile(r"(?<=\d),(?=\d)")
_FORBIDDEN_VISIBLE_KEY_PARTS = (
    "gold_answer",
    "ground_truth",
    "hidden_verifier",
    "reward",
    "verifier_material",
)


class GSM8KColdStartProjectionError(ValueError):
    """One public GSM8K row cannot enter the verified cold-start dataset."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class GSM8KColdStartProjection:
    """One admitted task and its paired calculator/direct SFT views."""

    instance_id: str
    source_index: int
    row_sha256: str
    tool_call_count: int
    tool_conversation: dict[str, Any]
    direct_conversation: dict[str, Any]
    replay: dict[str, Any]


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error


def _decimal(value: str) -> Decimal:
    try:
        number = Decimal(value.replace(",", "").strip())
    except InvalidOperation as error:
        raise GSM8KColdStartProjectionError("non_numeric_annotation", "calculator result must be numeric") from error
    if not number.is_finite():
        raise GSM8KColdStartProjectionError("non_numeric_annotation", "calculator result must be finite")
    return number


def _normalize_expression(expression: str) -> str:
    normalized = _THOUSANDS_SEPARATOR_PATTERN.sub("", expression.strip())
    if not normalized:
        raise GSM8KColdStartProjectionError("malformed_annotation", "calculator expression must be non-empty")
    return normalized


def _parse_annotations(solution: str) -> list[tuple[re.Match[str], str, str]]:
    matches = list(_ANNOTATION_PATTERN.finditer(solution))
    if solution.count("<<") != len(matches) or solution.count(">>") != len(matches):
        raise GSM8KColdStartProjectionError("malformed_annotation", "GSM8K calculator annotation is malformed")
    if not matches:
        raise GSM8KColdStartProjectionError("no_annotation", "GSM8K row contains no calculator annotation")
    if len(matches) > GSM8K_COLD_START_MAX_TOOL_CALLS:
        raise GSM8KColdStartProjectionError(
            "over_tool_budget",
            f"GSM8K row requires {len(matches)} calculator calls; maximum is {GSM8K_COLD_START_MAX_TOOL_CALLS}",
        )

    parsed = []
    for match in matches:
        annotation = match.group(1)
        if "=" not in annotation:
            raise GSM8KColdStartProjectionError(
                "malformed_annotation", "GSM8K calculator annotation must contain an equals sign"
            )
        expression, expected = annotation.rsplit("=", 1)
        expression = _normalize_expression(expression)
        expected = expected.strip()
        if not expected:
            raise GSM8KColdStartProjectionError(
                "malformed_annotation", "GSM8K calculator annotation result must be non-empty"
            )
        parsed.append((match, expression, expected))
    return parsed


def _reject_visible_verifier_material(value: Any, *, path: str = "conversation") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if any(part in normalized_key for part in _FORBIDDEN_VISIBLE_KEY_PARTS):
                raise GSM8KColdStartProjectionError(
                    "visible_verifier_material",
                    f"{path} contains forbidden verifier key {key!r}",
                )
            _reject_visible_verifier_material(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_visible_verifier_material(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and "calc_gsm8k_reward" in value:
        raise GSM8KColdStartProjectionError(
            "visible_verifier_material",
            f"{path} contains the gold-derived reward tool",
        )


def _validate_canonical_row(row: Mapping[str, Any]) -> tuple[str, int, str, str, str]:
    if not isinstance(row, Mapping):
        raise TypeError("GSM8K source row must be a mapping")
    question = row.get("input")
    solution = row.get("output")
    instance_id = row.get("instance_id")
    source_index = row.get("source_index")
    row_sha256 = row.get("row_digest")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K source row input must be a non-empty string")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("GSM8K source row output must be a non-empty string")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("GSM8K source row instance_id must be a non-empty string")
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError("GSM8K source row source_index must be a non-negative integer")
    if not isinstance(row_sha256, str) or len(row_sha256) != 64:
        raise ValueError("GSM8K source row row_digest must be a SHA-256")
    try:
        int(row_sha256, 16)
    except ValueError as error:
        raise ValueError("GSM8K source row row_digest must be a SHA-256") from error
    return question, solution, instance_id, source_index, row_sha256


def project_gsm8k_annotated_row(row: Mapping[str, Any]) -> GSM8KColdStartProjection:
    """Project one canonical public train row into verified paired SFT views."""

    question, solution, instance_id, source_index, row_sha256 = _validate_canonical_row(row)
    annotations = _parse_annotations(solution)
    source_answer = extract_gsm8k_answer(solution, method="strict")
    if source_answer is None:
        raise GSM8KColdStartProjectionError(
            "missing_final_answer", "GSM8K source solution must contain a strict final answer"
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": GSM8K_USER_PROMPT.format(question=question),
        }
    ]
    replay_calls = []
    cursor = 0
    for ordinal, (match, expression, expected) in enumerate(annotations):
        call_id = f"gsm8k-calc-{source_index:07d}-{ordinal:02d}"
        try:
            observed = evaluate_arithmetic_expression(expression)
        except (CalculatorExpressionError, CalculatorArithmeticError) as error:
            raise GSM8KColdStartProjectionError(
                "calculator_execution_error",
                f"calculator replay failed for source index {source_index}",
            ) from error
        if _decimal(observed) != _decimal(expected):
            raise GSM8KColdStartProjectionError(
                "calculator_replay_mismatch",
                f"calculator replay disagrees with the public annotation for source index {source_index}",
            )

        messages.append(
            {
                "role": "assistant",
                "content": solution[cursor : match.start()],
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": GSM8K_CALCULATOR_TOOL_NAME,
                            "arguments": json.dumps(
                                {"expression": expression},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            }
        )
        observation = f"Calculator result: {observed}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": GSM8K_CALCULATOR_TOOL_NAME,
                "content": observation,
            }
        )
        replay_calls.append(
            {
                "call_id": call_id,
                "expression": expression,
                "observation": observation,
                "annotation_numeric_match": True,
            }
        )
        cursor = match.end()
    final_content = solution[cursor:]
    messages.append({"role": "assistant", "content": final_content})

    direct_solution = _ANNOTATION_PATTERN.sub("", solution)
    tool_answer = extract_gsm8k_answer(final_content, method="strict")
    direct_answer = extract_gsm8k_answer(direct_solution, method="strict")
    if tool_answer is None or _decimal(tool_answer) != _decimal(source_answer):
        raise GSM8KColdStartProjectionError(
            "final_verifier_mismatch", "calculator projection does not preserve the public final answer"
        )
    if direct_answer is None or _decimal(direct_answer) != _decimal(source_answer):
        raise GSM8KColdStartProjectionError(
            "final_verifier_mismatch", "direct projection does not preserve the public final answer"
        )

    tool_conversation = {
        "conversation_id": f"{instance_id}:calculator",
        "system": GSM8K_CALCULATOR_SYSTEM_PROMPT,
        "tools": [copy.deepcopy(GSM8K_CALCULATOR_TOOL)],
        "messages": messages,
    }
    direct_conversation = {
        "conversation_id": f"{instance_id}:direct",
        "system": GSM8K_DIRECT_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": GSM8K_USER_PROMPT.format(question=question)},
            {"role": "assistant", "content": direct_solution},
        ],
    }
    _reject_visible_verifier_material(tool_conversation)
    _reject_visible_verifier_material(direct_conversation)
    replay = with_manifest_digest(
        {
            "format_version": GSM8K_COLD_START_FORMAT_VERSION,
            "instance_id": instance_id,
            "source_index": source_index,
            "row_sha256": row_sha256,
            "data_class": "A",
            "official_final_verifier_passed": True,
            "calculator_protocol_passed": True,
            "fresh_reexecution": True,
            "calculator_calls": replay_calls,
            "tool_call_count": len(replay_calls),
            "invalid_or_forged_observation_count": 0,
            "hidden_verifier_material_included": False,
        }
    )
    return GSM8KColdStartProjection(
        instance_id=instance_id,
        source_index=source_index,
        row_sha256=row_sha256,
        tool_call_count=len(replay_calls),
        tool_conversation=tool_conversation,
        direct_conversation=direct_conversation,
        replay=replay,
    )


def _selection_rank(instance_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{instance_id}".encode()).digest()


def _select_balanced_projections(
    projections: Sequence[GSM8KColdStartProjection],
    *,
    task_count: int,
    selection_seed: int,
) -> list[GSM8KColdStartProjection]:
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise ValueError("task_count must be a positive integer")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("selection_seed must be an integer")
    by_call_count = {
        call_count: sorted(
            (projection for projection in projections if projection.tool_call_count == call_count),
            key=lambda projection: (_selection_rank(projection.instance_id, selection_seed), projection.instance_id),
        )
        for call_count in range(1, GSM8K_COLD_START_MAX_TOOL_CALLS + 1)
    }
    selected = []
    offsets = Counter()
    while len(selected) < task_count:
        made_progress = False
        for call_count in range(1, GSM8K_COLD_START_MAX_TOOL_CALLS + 1):
            offset = offsets[call_count]
            candidates = by_call_count[call_count]
            if offset >= len(candidates):
                continue
            selected.append(candidates[offset])
            offsets[call_count] += 1
            made_progress = True
            if len(selected) == task_count:
                break
        if not made_progress:
            raise ValueError(f"task_count={task_count} exceeds the {len(projections)} eligible GSM8K rows")
    return selected


def _nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _tokenize_conversations(
    conversations: Sequence[Mapping[str, Any]],
    *,
    tokenizer_path: str | os.PathLike[str],
    tokenizer_revision: str,
    block_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("token accounting requires transformers") from error

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)
    records = []
    for conversation in conversations:
        rendered_messages = []
        system = conversation.get("system")
        if system is not None:
            rendered_messages.append({"role": "system", "content": system})
        rendered_messages.extend(copy.deepcopy(conversation["messages"]))
        encoded = tokenizer.apply_chat_template(
            conversation=rendered_messages,
            tools=conversation.get("tools"),
            chat_template=QWEN3_TEMPLATE,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        assistant_masks = encoded.get("assistant_masks")
        if assistant_masks is None or len(assistant_masks) != len(input_ids):
            raise RuntimeError("Qwen3 tokenizer did not return an aligned assistant token mask")
        if any(mask not in {0, 1} for mask in assistant_masks):
            raise RuntimeError("Qwen3 assistant token mask must contain only 0/1 values")
        if len(input_ids) > block_size:
            raise ValueError(
                f"conversation {conversation['conversation_id']!r} has {len(input_ids)} tokens and exceeds "
                f"block_size={block_size}; refusing to truncate cold-start targets"
            )
        assistant_loss_tokens = sum(assistant_masks)
        if assistant_loss_tokens < 1:
            raise ValueError(f"conversation {conversation['conversation_id']!r} has no assistant loss tokens")
        records.append(
            {
                "conversation_id": conversation["conversation_id"],
                "total_tokens": len(input_ids),
                "assistant_loss_tokens": assistant_loss_tokens,
            }
        )

    total_tokens = [record["total_tokens"] for record in records]
    loss_tokens = [record["assistant_loss_tokens"] for record in records]
    identity = qwen3_tokenizer_identity(tokenizer_path, revision=tokenizer_revision)
    return (
        {
            "tokenizer": identity,
            "training_chat_template": {
                "name": "qwen3",
                "sha256": hashlib.sha256(QWEN3_TEMPLATE.encode("utf-8")).hexdigest(),
            },
            "block_size": block_size,
            "conversation_count": len(records),
            "total_tokens": sum(total_tokens),
            "assistant_loss_tokens": sum(loss_tokens),
            "assistant_loss_tokens_mean": fmean(loss_tokens),
            "assistant_loss_tokens_p50": _nearest_rank(loss_tokens, 0.5),
            "assistant_loss_tokens_p95": _nearest_rank(loss_tokens, 0.95),
            "max_conversation_tokens": max(total_tokens),
            "truncated_conversation_count": 0,
        },
        records,
    )


def build_gsm8k_cold_start_payload(
    dataset: Dataset,
    source_manifest: Mapping[str, Any],
    *,
    task_count: int,
    selection_seed: int = GSM8K_COLD_START_SELECTION_SEED,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build paired A-class conversations and portable selected-row provenance."""

    verify_manifest_digest(source_manifest)
    if source_manifest.get("source") != GSM8K_DATASET_SOURCE:
        raise ValueError("source manifest does not identify the pinned GSM8K source")
    if source_manifest.get("config") != GSM8K_DATASET_CONFIG:
        raise ValueError("source manifest does not identify the pinned GSM8K config")
    if source_manifest.get("revision") != GSM8K_DATASET_REVISION:
        raise ValueError("source manifest does not identify the pinned GSM8K revision")
    if source_manifest.get("protocol_split") != "training":
        raise ValueError("cold-start data must come from the pinned training split")
    if not callable(getattr(dataset, "to_list", None)):
        raise TypeError("dataset must be an LMFlow Dataset-compatible object")

    source_rows = dataset.to_list()
    source_instances = source_manifest.get("instances")
    if not isinstance(source_instances, list) or len(source_instances) != len(source_rows):
        raise ValueError("source manifest instances must align with the materialized training dataset")
    if source_manifest.get("instance_count") != len(source_rows):
        raise ValueError("source manifest instance_count must match the materialized training dataset")
    for row_index, (row, source_instance) in enumerate(zip(source_rows, source_instances, strict=True)):
        if not isinstance(source_instance, Mapping):
            raise TypeError(f"source manifest instances[{row_index}] must be a mapping")
        _, _, instance_id, source_index, row_sha256 = _validate_canonical_row(row)
        expected_identity = {
            "instance_id": instance_id,
            "source_index": source_index,
            "row_sha256": row_sha256,
        }
        if dict(source_instance) != expected_identity:
            raise ValueError(f"source manifest instances[{row_index}] does not match the materialized row")

    rejection_counts = Counter()
    projections = []
    for row in source_rows:
        try:
            projections.append(project_gsm8k_annotated_row(row))
        except GSM8KColdStartProjectionError as error:
            rejection_counts[error.reason] += 1
    selected = _select_balanced_projections(
        projections,
        task_count=task_count,
        selection_seed=selection_seed,
    )
    conversations = []
    selected_instances = []
    replays = []
    for projection in selected:
        conversations.extend([projection.tool_conversation, projection.direct_conversation])
        replays.append(projection.replay)
        selected_instances.append(
            {
                "instance_id": projection.instance_id,
                "source_index": projection.source_index,
                "row_sha256": projection.row_sha256,
                "tool_call_count": projection.tool_call_count,
                "tool_conversation_id": projection.tool_conversation["conversation_id"],
                "tool_conversation_sha256": canonical_json_sha256(projection.tool_conversation),
                "direct_conversation_id": projection.direct_conversation["conversation_id"],
                "direct_conversation_sha256": canonical_json_sha256(projection.direct_conversation),
                "replay_manifest_sha256": projection.replay["manifest_sha256"],
            }
        )

    dataset_payload = {"type": "conversation", "instances": conversations}
    summary = {
        "source_training_count": len(dataset),
        "eligible_A_count": len(projections),
        "rejected_count": len(dataset) - len(projections),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_task_count": len(selected),
        "conversation_count": len(conversations),
        "tool_conversation_count": len(selected),
        "direct_conversation_count": len(selected),
        "tool_call_count": sum(projection.tool_call_count for projection in selected),
        "selected_tool_call_distribution": dict(sorted(Counter(p.tool_call_count for p in selected).items())),
        "official_replay_pass_count": len(selected),
        "data_class_counts": {"A": len(selected), "B": 0, "C": 0, "D": 0, "E": 0},
        "hidden_verifier_material_included": False,
    }
    manifest_payload = {
        "format_version": GSM8K_COLD_START_FORMAT_VERSION,
        "source": {
            "dataset": GSM8K_DATASET_SOURCE,
            "config": GSM8K_DATASET_CONFIG,
            "revision": GSM8K_DATASET_REVISION,
            "protocol_split": "training",
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "dataset_protocol_sha256": source_manifest["dataset_protocol_sha256"],
        },
        "selection": {
            "algorithm": "sha256-ranked-round-robin-by-tool-call-count/v1",
            "seed": selection_seed,
            "task_count": task_count,
            "max_tool_calls": GSM8K_COLD_START_MAX_TOOL_CALLS,
        },
        "projection": {
            "data_class": "A",
            "tool_profile": "calculate(expression)",
            "paired_direct": True,
            "fresh_calculator_reexecution_required": True,
            "official_final_verifier_required": True,
            "assistant_messages_trainable": True,
            "system_user_tool_observation_trainable": False,
            "hidden_verifier_material_included": False,
        },
        "selected_instances": selected_instances,
        "ordered_instance_ids_sha256": canonical_json_sha256([p.instance_id for p in selected]),
        "dataset_sha256": canonical_json_sha256(dataset_payload),
        "summary": summary,
    }
    return dataset_payload, with_manifest_digest(manifest_payload), replays


def _new_json_file(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _new_jsonl_file(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        for value in values:
            output_file.write(
                json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
            )
            output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def run_gsm8k_cold_start_factory(
    *,
    artifact_dir: str | os.PathLike[str],
    run_id: str,
    task_count: int,
    tokenizer_path: str | os.PathLike[str],
    tokenizer_revision: str = GSM8K_MODEL_REVISION,
    selection_seed: int = GSM8K_COLD_START_SELECTION_SEED,
    block_size: int = GSM8K_COLD_START_DEFAULT_BLOCK_SIZE,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Atomically publish a verified paired GSM8K cold-start dataset."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    target = Path(artifact_dir)
    if target.exists():
        raise FileExistsError(f"artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        dataset, source_manifest = load_pinned_gsm8k_dataset("training", cache_dir=cache_dir)
        dataset_payload, data_manifest, replays = build_gsm8k_cold_start_payload(
            dataset,
            source_manifest,
            task_count=task_count,
            selection_seed=selection_seed,
        )
        token_summary, token_records = _tokenize_conversations(
            dataset_payload["instances"],
            tokenizer_path=tokenizer_path,
            tokenizer_revision=tokenizer_revision,
            block_size=block_size,
        )
        token_records_by_id = {record["conversation_id"]: record for record in token_records}
        selected_instances = copy.deepcopy(data_manifest["selected_instances"])
        for instance in selected_instances:
            instance["tool_tokens"] = token_records_by_id[instance["tool_conversation_id"]]
            instance["direct_tokens"] = token_records_by_id[instance["direct_conversation_id"]]

        _new_json_file(staging / "dataset" / "data.json", dataset_payload)
        _new_jsonl_file(staging / "replay.jsonl", replays)
        _new_json_file(staging / "source_dataset_manifest.json", source_manifest)
        _new_json_file(staging / "data_manifest.json", data_manifest)
        artifact_manifest = with_manifest_digest(
            {
                "format_version": GSM8K_COLD_START_FORMAT_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "dataset_ref": "dataset/data.json",
                "dataset_file_sha256": file_sha256(staging / "dataset" / "data.json"),
                "dataset_content_sha256": data_manifest["dataset_sha256"],
                "source_dataset_manifest_ref": "source_dataset_manifest.json",
                "source_dataset_manifest_sha256": source_manifest["manifest_sha256"],
                "data_manifest_ref": "data_manifest.json",
                "replay_ref": "replay.jsonl",
                "replay_file_sha256": file_sha256(staging / "replay.jsonl"),
                "data_manifest_sha256": data_manifest["manifest_sha256"],
                "selection": data_manifest["selection"],
                "projection": data_manifest["projection"],
                "model_target": {
                    "model_id": GSM8K_MODEL_ID,
                    "model_revision": GSM8K_MODEL_REVISION,
                    "tokenizer": token_summary["tokenizer"],
                    "block_size": block_size,
                },
                "selected_instances": selected_instances,
                "summary": {**data_manifest["summary"], "tokens": token_summary},
                "hidden_verifier_material_included": False,
            }
        )
        report = with_manifest_digest(
            {
                "format_version": GSM8K_COLD_START_FORMAT_VERSION,
                "run_id": run_id,
                "artifact_manifest_ref": "artifact_manifest.json",
                "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
                "data_manifest": data_manifest,
                "summary": artifact_manifest["summary"],
            }
        )
        _new_json_file(staging / "artifact_manifest.json", artifact_manifest)
        _new_json_file(staging / "report.json", report)
        staging.rename(target)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "GSM8K_COLD_START_DEFAULT_BLOCK_SIZE",
    "GSM8K_COLD_START_FORMAT_VERSION",
    "GSM8K_COLD_START_MAX_TOOL_CALLS",
    "GSM8K_COLD_START_SELECTION_SEED",
    "GSM8KColdStartProjection",
    "GSM8KColdStartProjectionError",
    "build_gsm8k_cold_start_payload",
    "project_gsm8k_annotated_row",
    "run_gsm8k_cold_start_factory",
]
