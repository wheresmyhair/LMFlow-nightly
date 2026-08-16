"""Run the pinned AppWorld tiny baseline and publish atomic artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from lmflow.agentic.appworld_episode import configure_appworld_freezegun, run_appworld_episode
from lmflow.agentic.appworld_protocol import (
    APPWORLD_CODE_VERSION,
    APPWORLD_DATA_VERSION,
    APPWORLD_PROTOCOL_FORMAT_VERSION,
    APPWORLD_REVISION,
    APPWORLD_TINY_SCENARIOS,
    APPWORLD_TINY_TASK_IDS,
    canonical_json_sha256,
    load_pinned_appworld_tiny_dataset,
    with_manifest_digest,
)
from lmflow.agentic.completion import OpenAICompatibleCompletionBackend
from lmflow.agentic.gsm8k_protocol import GSM8K_MODEL_REVISION, qwen3_tokenizer_identity
from lmflow.agentic.scaffolds.appworld_react_code.scaffold import (
    qwen3_reference_model_kwargs,
    scaffold_identity,
)

APPWORLD_REPORT_FORMAT_VERSION = "lmflow.appworld-tiny-report/v1"
APPWORLD_MODEL_ID = "Qwen/Qwen3-8B"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return parsed


def _reject_machine_path(value: str, *, name: str) -> None:
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{name} must be a portable identity, not an absolute machine path")


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


def _complete_scenarios(task_ids: Sequence[str]) -> list[str]:
    selected = set(task_ids)
    return [
        scenario_id
        for scenario_id in APPWORLD_TINY_SCENARIOS
        if {f"{scenario_id}_{variant}" for variant in (1, 2, 3)}.issubset(selected)
    ]


def _summarize_tasks(task_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success_count = sum(record["metrics"]["success"] is True for record in task_records)
    task_status_counts = Counter(record["metrics"]["task_status"] for record in task_records)
    failure_counts = Counter(
        record["metrics"]["failure_type"] for record in task_records if record["metrics"]["failure_type"] is not None
    )
    metric_totals = {
        name: sum(record["metrics"][name] for record in task_records)
        for name in (
            "steps",
            "model_calls",
            "tool_calls",
            "valid_tool_calls",
            "invalid_tool_calls",
            "api_call_attempts",
            "state_change_steps",
            "recovery_count",
        )
    }
    input_tokens = [record["metrics"]["usage"]["input_tokens"] for record in task_records]
    output_tokens = [record["metrics"]["usage"]["output_tokens"] for record in task_records]
    return {
        "task_count": len(task_records),
        "success_count": success_count,
        "success_rate": success_count / len(task_records),
        "task_status_counts": dict(sorted(task_status_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "totals": metric_totals,
        "usage": {
            "input_tokens": sum(input_tokens) if all(isinstance(value, int) for value in input_tokens) else None,
            "output_tokens": (sum(output_tokens) if all(isinstance(value, int) for value in output_tokens) else None),
            "reported_for_all_tasks": all(
                record["metrics"]["usage"]["reported_for_all_calls"] for record in task_records
            ),
        },
        "latency_seconds": {
            "total": sum(record["metrics"]["latency_seconds"]["total"] for record in task_records),
            "model": sum(record["metrics"]["latency_seconds"]["model"] for record in task_records),
            "environment": sum(record["metrics"]["latency_seconds"]["environment"] for record in task_records),
        },
    }


def run_appworld_tiny_baseline(
    *,
    artifact_dir: str | os.PathLike[str],
    run_id: str,
    appworld_root: str | os.PathLike[str],
    appworld_source: str | os.PathLike[str],
    base_url: str,
    served_model_name: str,
    tokenizer_path: str | os.PathLike[str],
    backend_version: str,
    task_ids: Sequence[str] | None = None,
    model_id: str = APPWORLD_MODEL_ID,
    model_revision: str = GSM8K_MODEL_REVISION,
    tokenizer_revision: str = GSM8K_MODEL_REVISION,
    model_artifact_sha256: str | None = None,
    model_label: str = "base",
    backend_id: str = "vllm-openai-compatible",
    endpoint_label: str = "local-vllm",
    served_max_model_len: int = 32768,
    served_dtype: str = "bfloat16",
    served_model_runner: str = "v1",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 1,
    max_steps: int = 50,
    max_completion_tokens: int = 3000,
    enable_thinking: bool = False,
    timeout_seconds: float = 300.0,
    max_retries: int = 2,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Execute a serial AppWorld run and atomically publish its directory."""

    for name, value in (
        ("run_id", run_id),
        ("base_url", base_url),
        ("served_model_name", served_model_name),
        ("backend_version", backend_version),
        ("model_id", model_id),
        ("model_revision", model_revision),
        ("tokenizer_revision", tokenizer_revision),
        ("model_label", model_label),
        ("backend_id", backend_id),
        ("endpoint_label", endpoint_label),
        ("served_dtype", served_dtype),
        ("served_model_runner", served_model_runner),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    _reject_machine_path(model_id, name="model_id")
    _reject_machine_path(served_model_name, name="served_model_name")
    if model_artifact_sha256 is not None:
        if not isinstance(model_artifact_sha256, str) or len(model_artifact_sha256) != 64:
            raise ValueError("model_artifact_sha256 must be a SHA-256 when provided")
        int(model_artifact_sha256, 16)
    for name, value in (
        ("served_max_model_len", served_max_model_len),
        ("tensor_parallel_size", tensor_parallel_size),
        ("max_num_seqs", max_num_seqs),
        ("max_steps", max_steps),
        ("max_completion_tokens", max_completion_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_steps > 50:
        raise ValueError("max_steps cannot exceed the official scaffold's 50-step limit")
    if not isinstance(enable_thinking, bool):
        raise TypeError("enable_thinking must be a bool")
    if isinstance(gpu_memory_utilization, bool) or not isinstance(gpu_memory_utilization, int | float):
        raise TypeError("gpu_memory_utilization must be a number")
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int | float):
        raise TypeError("timeout_seconds must be a number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    target = Path(artifact_dir)
    if target.exists():
        raise FileExistsError(f"artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        dataset, dataset_manifest = load_pinned_appworld_tiny_dataset(
            appworld_root=appworld_root,
            task_ids=task_ids,
        )
        selected_task_ids = dataset_manifest["selected_task_ids"]
        dataset_payload = dataset.to_dict()
        tokenizer = qwen3_tokenizer_identity(tokenizer_path, revision=tokenizer_revision)
        tokenizer["name"] = model_id
        model_identity = {
            "label": model_label,
            "model_id": model_id,
            "model_revision": model_revision,
            "served_model_name": served_model_name,
            "artifact_sha256": model_artifact_sha256,
            "tokenizer": tokenizer,
        }
        sampling = dict(qwen3_reference_model_kwargs(enable_thinking=enable_thinking))
        sampling["max_completion_tokens"] = max_completion_tokens
        scaffold = scaffold_identity(appworld_source)
        execution_identity = {
            "runner": "lmflow.agentic.appworld_episode.run_appworld_episode",
            "official_evaluator": "appworld.evaluator.Metric",
            "backend_id": backend_id,
            "backend_version": backend_version,
            "endpoint_label": endpoint_label,
            "concurrency": 1,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "served_engine": {
                "max_model_len": served_max_model_len,
                "dtype": served_dtype,
                "model_runner": served_model_runner,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_num_seqs": max_num_seqs,
            },
        }
        experiment_name = f"lmflow-appworld/{run_id}"
        task_records = []
        trackers = {}
        with OpenAICompatibleCompletionBackend(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ) as backend:
            for task_id in selected_task_ids:
                print(f"running AppWorld task {task_id}", file=sys.stderr, flush=True)
                result = run_appworld_episode(
                    backend,
                    task_id=task_id,
                    model_name=served_model_name,
                    model_revision=model_revision,
                    trajectory_id=f"{run_id}:{task_id}",
                    appworld_root=appworld_root,
                    appworld_source=appworld_source,
                    experiment_name=experiment_name,
                    model_kwargs=sampling,
                    max_steps=max_steps,
                )
                task_directory = staging / "tasks" / task_id
                artifact_path = task_directory / "trajectory.json"
                projection_path = task_directory / "conversation.json"
                _new_json_file(artifact_path, result.artifact)
                _new_json_file(projection_path, result.training_projection)
                copied_raw = _copy_raw_outputs(
                    result.raw_output_directory,
                    task_directory / "raw_appworld",
                )
                if result.official_tracker is not None:
                    trackers[task_id] = result.official_tracker
                task_records.append(
                    {
                        "task_id": task_id,
                        "artifact_ref": artifact_path.relative_to(staging).as_posix(),
                        "artifact_file_sha256": _sha256_file(artifact_path),
                        "artifact_manifest_sha256": result.artifact["manifest_sha256"],
                        "training_projection_ref": projection_path.relative_to(staging).as_posix(),
                        "training_projection_sha256": _sha256_file(projection_path),
                        "raw_appworld_ref": (task_directory / "raw_appworld").relative_to(staging).as_posix(),
                        "raw_appworld_sections": copied_raw,
                        "metrics": result.artifact["metrics"],
                    }
                )
                print(f"completed AppWorld task {task_id}", file=sys.stderr, flush=True)

        official_metrics = None
        if len(trackers) == len(selected_task_ids):
            from appworld.evaluator import Metric

            official_metrics = Metric.compute_metrics(trackers, include_details=True)
        complete_scenarios = _complete_scenarios(selected_task_ids)
        protocol_identity = {
            "format_version": APPWORLD_PROTOCOL_FORMAT_VERSION,
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "scaffold": scaffold,
            "sampling": sampling,
            "max_steps": max_steps,
            "gold_visibility": "official_evaluator_only",
        }
        run_manifest = with_manifest_digest(
            {
                "format_version": APPWORLD_PROTOCOL_FORMAT_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "dataset": {
                    "manifest_ref": "dataset_manifest.json",
                    "manifest_sha256": dataset_manifest["manifest_sha256"],
                    "dataset_protocol_sha256": dataset_manifest["dataset_protocol_sha256"],
                    "projection_ref": "dataset.json",
                    "projection_sha256": canonical_json_sha256(dataset_payload),
                    "selected_task_ids": selected_task_ids,
                },
                "protocol": protocol_identity,
                "protocol_sha256": canonical_json_sha256(protocol_identity),
                "model": model_identity,
                "model_sha256": canonical_json_sha256(model_identity),
                "execution": execution_identity,
                "execution_sha256": canonical_json_sha256(execution_identity),
                "appworld": {
                    "revision": APPWORLD_REVISION,
                    "code_version": APPWORLD_CODE_VERSION,
                    "data_version": APPWORLD_DATA_VERSION,
                },
                "known_limitations": [
                    "AppWorld's in-process environment is intentionally serialized because it owns global state.",
                    (
                        "Scenario goal completion is interpretable only for scenario groups whose three variants "
                        "are present."
                    ),
                    (
                        "Raw task artifacts can contain AppWorld protected data and must not be committed or "
                        "publicly redistributed unencrypted."
                    ),
                ],
            }
        )
        report = with_manifest_digest(
            {
                "format_version": APPWORLD_REPORT_FORMAT_VERSION,
                "run_id": run_id,
                "run_manifest_ref": "run_manifest.json",
                "run_manifest_sha256": run_manifest["manifest_sha256"],
                "dataset_manifest_ref": "dataset_manifest.json",
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "task_records": task_records,
                "summary": _summarize_tasks(task_records),
                "official_metrics": official_metrics,
                "complete_scenario_groups": complete_scenarios,
                "scenario_goal_completion_interpretable": (len(complete_scenarios) * 3 == len(selected_task_ids)),
            }
        )
        _new_json_file(staging / "dataset.json", dataset_payload)
        _new_json_file(staging / "dataset_manifest.json", dataset_manifest)
        _new_json_file(staging / "run_manifest.json", run_manifest)
        _new_json_file(staging / "report.json", report)
        staging.rename(target)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_appworld_install(
    *,
    appworld_root: str | os.PathLike[str],
    appworld_source: str | os.PathLike[str],
    task_id: str = APPWORLD_TINY_TASK_IDS[0],
) -> dict[str, Any]:
    """Verify source identity, task loading, initialization, reset, and evaluator use."""

    dataset, manifest = load_pinned_appworld_tiny_dataset(
        appworld_root=appworld_root,
        task_ids=[task_id],
    )
    scaffold = scaffold_identity(appworld_source)
    os.environ["APPWORLD_ROOT"] = str(Path(appworld_root).expanduser().resolve())
    configure_appworld_freezegun()
    from appworld import AppWorld

    snapshots = []
    evaluations = []
    for _ in range(2):
        with AppWorld(
            task_id=task_id,
            experiment_name="lmflow-appworld-install-verification",
            load_ground_truth=True,
            random_seed=100,
            raise_on_extra_parameters=True,
        ) as world:
            snapshots.append(world.execute("print(apis.api_docs.show_app_descriptions())"))
            evaluations.append(world.evaluate(suppress_errors=True).to_dict(stats_only=True))
    if snapshots[0] != snapshots[1] or evaluations[0] != evaluations[1]:
        raise RuntimeError("AppWorld reset verification was not deterministic")
    return with_manifest_digest(
        {
            "format_version": "lmflow.appworld-install-verification/v1",
            "python": sys.version.split()[0],
            "appworld_code_version": APPWORLD_CODE_VERSION,
            "appworld_revision": APPWORLD_REVISION,
            "appworld_data_version": APPWORLD_DATA_VERSION,
            "task_id": task_id,
            "dataset_instance_count": len(dataset),
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "scaffold_prompt_sha256": scaffold["prompt_sha256"],
            "reset_observation_sha256": hashlib.sha256(snapshots[0].encode()).hexdigest(),
            "reset_equal": True,
            "official_evaluation_stats": evaluations[0],
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify the pinned AppWorld tiny protocol.")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="Verify the pinned install, data, scaffold, reset, and evaluator.")
    verify.add_argument("--appworld-root", required=True)
    verify.add_argument("--appworld-source", required=True)
    verify.add_argument("--task-id", choices=APPWORLD_TINY_TASK_IDS, default=APPWORLD_TINY_TASK_IDS[0])

    run = commands.add_parser("run", help="Run selected tiny tasks and atomically publish artifacts.")
    run.add_argument("--artifact-dir", required=True, help="New output artifact directory.")
    run.add_argument("--run-id", required=True)
    run.add_argument("--appworld-root", required=True)
    run.add_argument("--appworld-source", required=True)
    run.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL; not persisted.")
    run.add_argument("--served-model-name", required=True)
    run.add_argument("--tokenizer-path", required=True)
    run.add_argument("--backend-version", required=True)
    run.add_argument("--task-id", action="append", choices=APPWORLD_TINY_TASK_IDS)
    run.add_argument("--model-id", default=APPWORLD_MODEL_ID)
    run.add_argument("--model-revision", default=GSM8K_MODEL_REVISION)
    run.add_argument("--tokenizer-revision", default=GSM8K_MODEL_REVISION)
    run.add_argument("--model-artifact-sha256")
    run.add_argument("--model-label", default="base")
    run.add_argument("--backend-id", default="vllm-openai-compatible")
    run.add_argument("--endpoint-label", default="local-vllm")
    run.add_argument("--served-max-model-len", type=_positive_int, default=32768)
    run.add_argument("--served-dtype", default="bfloat16")
    run.add_argument("--served-model-runner", choices=["v1", "v2"], default="v1")
    run.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    run.add_argument("--gpu-memory-utilization", type=_fraction, default=0.90)
    run.add_argument("--max-num-seqs", type=_positive_int, default=1)
    run.add_argument("--max-steps", type=_positive_int, default=50)
    run.add_argument(
        "--max-completion-tokens",
        type=_positive_int,
        default=3000,
        help="Per-step output cap; 3000 matches AppWorld's official Qwen3-8B profile.",
    )
    run.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking in the chat template; disabled by the official AppWorld profile.",
    )
    run.add_argument("--timeout-seconds", type=_positive_int, default=300)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; credentials are never accepted as arguments.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_appworld_install(
                appworld_root=args.appworld_root,
                appworld_source=args.appworld_source,
                task_id=args.task_id,
            )
        else:
            if args.max_retries < 0:
                raise ValueError("max_retries must be non-negative")
            result = run_appworld_tiny_baseline(
                artifact_dir=args.artifact_dir,
                run_id=args.run_id,
                appworld_root=args.appworld_root,
                appworld_source=args.appworld_source,
                base_url=args.base_url,
                served_model_name=args.served_model_name,
                tokenizer_path=args.tokenizer_path,
                backend_version=args.backend_version,
                task_ids=args.task_id,
                model_id=args.model_id,
                model_revision=args.model_revision,
                tokenizer_revision=args.tokenizer_revision,
                model_artifact_sha256=args.model_artifact_sha256,
                model_label=args.model_label,
                backend_id=args.backend_id,
                endpoint_label=args.endpoint_label,
                served_max_model_len=args.served_max_model_len,
                served_dtype=args.served_dtype,
                served_model_runner=args.served_model_runner,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                max_steps=args.max_steps,
                max_completion_tokens=args.max_completion_tokens,
                enable_thinking=args.enable_thinking,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                api_key=os.environ.get(args.api_key_env),
            )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
