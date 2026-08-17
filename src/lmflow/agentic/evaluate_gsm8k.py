"""Run and postprocess the pinned GSM8K direct/calculator baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from lmflow.agentic.completion import OpenAICompatibleCompletionBackend
from lmflow.agentic.gsm8k_evaluation import (
    GSM8K_CALCULATOR_SYSTEM_PROMPT,
    GSM8K_CALCULATOR_TOOL,
    GSM8K_DIRECT_SYSTEM_PROMPT,
    GSM8K_REFERENCE_SCAFFOLD,
    GSM8K_USER_PROMPT,
    GSM8KCompletionRunner,
    create_gsm8k_calculator_recipe,
    create_gsm8k_direct_recipe,
)
from lmflow.agentic.gsm8k_protocol import (
    GSM8K_DATASET_CONFIG,
    GSM8K_DATASET_SOURCE,
    GSM8K_MODEL_ID,
    GSM8K_MODEL_REVISION,
    GSM8K_PROTOCOL_FORMAT_VERSION,
    GSM8K_PROTOCOL_SPLIT_SIZES,
    GSM8K_REPORT_FORMAT_VERSION,
    canonical_json_sha256,
    classify_evaluation_cases,
    load_pinned_gsm8k_dataset,
    paired_profile_comparison,
    qwen3_tokenizer_identity,
    summarize_evaluation_result,
    summarize_repeated_reports,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.args import DatasetArguments, EvaluatorArguments, ModelArguments
from lmflow.pipeline.evaluation import Evaluator
from lmflow.pipeline.evaluation.recipe import EvaluationBudget, SamplingConfig
from lmflow.pipeline.evaluation.runtime import LocalEvaluationRuntime

_TEMPERATURE = 0.6
_TOP_P = 0.95
_TOP_K = 20
_MIN_P = 0.0
_ENABLE_THINKING = True
_DIRECT_BUDGET = EvaluationBudget(
    max_model_calls=1,
    max_tool_calls=0,
    max_steps=1,
    max_input_tokens=8192,
    max_output_tokens=4096,
    wall_time_seconds=180,
)
_CALCULATOR_BUDGET = EvaluationBudget(
    max_model_calls=4,
    max_tool_calls=4,
    max_steps=4,
    max_input_tokens=32768,
    max_output_tokens=8192,
    wall_time_seconds=300,
)


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _relative_result_payload(result: Any, *, root: Path) -> dict[str, Any]:
    payload = result.to_dict()
    resolved_root = root.resolve()
    for record in payload["records"]:
        artifact_ref = record.get("artifact_ref")
        if artifact_ref is None:
            continue
        artifact_path = Path(artifact_ref).resolve()
        try:
            relative_path = artifact_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"evaluation artifact is outside the run directory: {artifact_path}") from error
        record["artifact_ref"] = relative_path.as_posix()
    return payload


def _protocol_identity(tokenizer: Mapping[str, Any]) -> dict[str, Any]:
    scaffold = {
        **GSM8K_REFERENCE_SCAFFOLD,
        "direct_system_prompt": GSM8K_DIRECT_SYSTEM_PROMPT,
        "calculator_system_prompt": GSM8K_CALCULATOR_SYSTEM_PROMPT,
        "user_prompt_template": GSM8K_USER_PROMPT,
        "calculator_tool": GSM8K_CALCULATOR_TOOL,
        "chat_template_sha256": tokenizer["chat_template_sha256"],
        "enable_thinking": _ENABLE_THINKING,
    }
    return {
        "format_version": GSM8K_PROTOCOL_FORMAT_VERSION,
        "profiles": ["direct-answer", "calculator-tool"],
        "scaffold": scaffold,
        "sampling": {
            "temperature": _TEMPERATURE,
            "top_p": _TOP_P,
            "top_k": _TOP_K,
            "min_p": _MIN_P,
            "seed_policy": "run_manifest.sampling_seed",
        },
        "budgets": {
            "direct-answer": asdict(_DIRECT_BUDGET),
            "calculator-tool": asdict(_CALCULATOR_BUDGET),
        },
        "gold_visibility": "hidden_verifier_only",
        "calculator_feedback_visibility": "model_visible_arithmetic_result",
        "calculator_gold_access": False,
        "verifier": {
            "id": "lmflow.gsm8k.hidden-verifier",
            "revision": "v2",
            "primary_correctness": "decimal-numeric-equivalence",
            "strictness": "requires-final-marker-and-decimal-numeric-equivalence",
            "compatibility_metrics": ["reference_exact_correctness", "strict_exact_correctness"],
        },
    }


def _run_profile(
    *,
    profile_name: str,
    backend: Any,
    served_model_name: str,
    model_kwargs: Mapping[str, Any],
    model_args: ModelArguments,
    data_args: DatasetArguments,
    dataset: Any,
    split: str,
    sampling: SamplingConfig,
    runtime: LocalEvaluationRuntime,
    artifact_dir: Path,
) -> Any:
    if profile_name == "direct-answer":
        recipe = create_gsm8k_direct_recipe(
            split=split,
            data_source=GSM8K_DATASET_SOURCE,
            sampling=sampling,
            budget=_DIRECT_BUDGET,
            require_canonical_ids=True,
            correctness="numeric-equivalence",
        )
    elif profile_name == "calculator-tool":
        recipe = create_gsm8k_calculator_recipe(
            split=split,
            data_source=GSM8K_DATASET_SOURCE,
            sampling=sampling,
            budget=_CALCULATOR_BUDGET,
            require_canonical_ids=True,
            correctness="numeric-equivalence",
        )
    else:
        raise ValueError(f"unsupported GSM8K profile {profile_name!r}")
    runner = GSM8KCompletionRunner(
        backend=backend,
        model_name=served_model_name,
        model_kwargs=model_kwargs,
        artifact_dir=artifact_dir,
    )
    evaluator = Evaluator(
        model_args,
        data_args,
        EvaluatorArguments(),
        recipe=recipe,
        runner=runner,
        runtime=runtime,
    )
    return evaluator.evaluate(backend, dataset)


def run_gsm8k_baseline(
    *,
    artifact_dir: str | os.PathLike[str],
    run_id: str,
    base_url: str,
    served_model_name: str,
    tokenizer_path: str | os.PathLike[str],
    backend_version: str,
    split: str = "smoke",
    sampling_seed: int = 0,
    max_concurrency: int = 1,
    cache_dir: str | None = None,
    model_id: str = GSM8K_MODEL_ID,
    model_revision: str = GSM8K_MODEL_REVISION,
    tokenizer_revision: str = GSM8K_MODEL_REVISION,
    model_artifact_sha256: str | None = None,
    model_label: str = "base",
    backend_id: str = "vllm-openai-compatible",
    endpoint_label: str = "local-vllm",
    served_max_model_len: int = 16384,
    served_dtype: str = "bfloat16",
    tool_call_parser: str = "qwen3_xml",
    reasoning_parser: str = "qwen3",
    generation_config: str = "vllm",
    served_model_runner: str = "v1",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 4,
    timeout_seconds: float = 300.0,
    max_retries: int = 2,
    api_key: str | None = None,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    """Publish one atomic two-profile baseline run directory."""

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
        ("tool_call_parser", tool_call_parser),
        ("reasoning_parser", reasoning_parser),
        ("generation_config", generation_config),
        ("served_model_runner", served_model_runner),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if split not in GSM8K_PROTOCOL_SPLIT_SIZES:
        raise ValueError(f"unsupported GSM8K protocol split {split!r}")
    if split == "training":
        raise ValueError("the training split is excluded from evaluation runs")
    _reject_machine_path(model_id, name="model_id")
    _reject_machine_path(served_model_name, name="served_model_name")
    if isinstance(served_max_model_len, bool) or not isinstance(served_max_model_len, int):
        raise TypeError("served_max_model_len must be an integer")
    if served_max_model_len < 1:
        raise ValueError("served_max_model_len must be positive")
    if served_model_runner not in {"v1", "v2"}:
        raise ValueError("served_model_runner must be v1 or v2")
    for name, value in (("tensor_parallel_size", tensor_parallel_size), ("max_num_seqs", max_num_seqs)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(gpu_memory_utilization, bool) or not isinstance(gpu_memory_utilization, int | float):
        raise TypeError("gpu_memory_utilization must be a number")
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int | float):
        raise TypeError("timeout_seconds must be a number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if model_artifact_sha256 is not None:
        if not isinstance(model_artifact_sha256, str) or len(model_artifact_sha256) != 64:
            raise ValueError("model_artifact_sha256 must be a SHA-256 when provided")
        try:
            int(model_artifact_sha256, 16)
        except ValueError as error:
            raise ValueError("model_artifact_sha256 must be a SHA-256 when provided") from error

    target = Path(artifact_dir)
    if target.exists():
        raise FileExistsError(f"artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        dataset, dataset_manifest = load_pinned_gsm8k_dataset(split, cache_dir=cache_dir)
        tokenizer = qwen3_tokenizer_identity(tokenizer_path, revision=tokenizer_revision)
        tokenizer["name"] = model_id
        protocol = _protocol_identity(tokenizer)
        protocol_sha256 = canonical_json_sha256(protocol)
        model_identity = {
            "label": model_label,
            "model_id": model_id,
            "model_revision": model_revision,
            "served_model_name": served_model_name,
            "artifact_sha256": model_artifact_sha256,
            "tokenizer": tokenizer,
        }
        model_sha256 = canonical_json_sha256(model_identity)
        sampling = SamplingConfig(temperature=_TEMPERATURE, top_p=_TOP_P, seed=sampling_seed)
        model_kwargs = {
            "extra_body": {
                "top_k": _TOP_K,
                "min_p": _MIN_P,
                "chat_template_kwargs": {"enable_thinking": _ENABLE_THINKING},
            }
        }
        runtime = LocalEvaluationRuntime(max_concurrency=max_concurrency)
        model_args = ModelArguments(
            model_name_or_path=model_id,
            model_revision=model_revision,
            tokenizer_name=model_id,
        )
        data_args = DatasetArguments(
            dataset_path=None,
            dataset_name=GSM8K_DATASET_SOURCE,
            dataset_config_name=GSM8K_DATASET_CONFIG,
        )

        profile_payloads: dict[str, dict[str, Any]] = {}
        profile_reports: dict[str, dict[str, Any]] = {}
        with OpenAICompatibleCompletionBackend(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ) as backend:
            for profile_name in ("direct-answer", "calculator-tool"):
                print(
                    f"running {profile_name} on {dataset_manifest['instance_count']} instances",
                    file=sys.stderr,
                    flush=True,
                )
                result = _run_profile(
                    profile_name=profile_name,
                    backend=backend,
                    served_model_name=served_model_name,
                    model_kwargs=model_kwargs,
                    model_args=model_args,
                    data_args=data_args,
                    dataset=dataset,
                    split=split,
                    sampling=sampling,
                    runtime=runtime,
                    artifact_dir=staging / "profiles" / profile_name / "records",
                )
                result_payload = _relative_result_payload(result, root=staging)
                profile_payloads[profile_name] = result_payload
                result_ref = f"profiles/{profile_name}/result.json"
                profile_reports[profile_name] = {
                    "result_ref": result_ref,
                    "result_sha256": canonical_json_sha256(result_payload),
                    "statistics": summarize_evaluation_result(result_payload),
                    "case_analysis": classify_evaluation_cases(result_payload, profile_name=profile_name),
                }
                print(f"completed {profile_name}", file=sys.stderr, flush=True)

        paired = paired_profile_comparison(
            profile_payloads["direct-answer"],
            profile_payloads["calculator-tool"],
            bootstrap_samples=bootstrap_samples,
        )
        dataset_manifest_ref = "dataset_manifest.json"
        execution_identity = {
            "runner": "lmflow.agentic.gsm8k_evaluation.GSM8KCompletionRunner",
            "backend_id": backend_id,
            "backend_version": backend_version,
            "endpoint_label": endpoint_label,
            "max_concurrency": max_concurrency,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "provider_behavior": model_kwargs,
            "served_engine": {
                "max_model_len": served_max_model_len,
                "dtype": served_dtype,
                "tool_call_parser": tool_call_parser,
                "reasoning_parser": reasoning_parser,
                "generation_config": generation_config,
                "model_runner": served_model_runner,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_num_seqs": max_num_seqs,
            },
        }
        execution_sha256 = canonical_json_sha256(execution_identity)
        run_manifest = with_manifest_digest(
            {
                "format_version": GSM8K_PROTOCOL_FORMAT_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "dataset": {
                    "manifest_ref": dataset_manifest_ref,
                    "manifest_sha256": dataset_manifest["manifest_sha256"],
                    "dataset_protocol_sha256": dataset_manifest["dataset_protocol_sha256"],
                    "protocol_split": split,
                    "instance_count": dataset_manifest["instance_count"],
                },
                "protocol": protocol,
                "protocol_sha256": protocol_sha256,
                "sampling_seed": sampling_seed,
                "model": model_identity,
                "model_sha256": model_sha256,
                "execution": execution_identity,
                "execution_sha256": execution_sha256,
                "known_limitations": [
                    (
                        "The generic Evaluator provenance records runner implementation and runtime, but does not "
                        "yet capture provider-specific runner/backend behavior. This run manifest is authoritative "
                        "for top_k, min_p, thinking mode, backend identity, and endpoint behavior."
                    )
                ],
            }
        )
        report = with_manifest_digest(
            {
                "format_version": GSM8K_REPORT_FORMAT_VERSION,
                "run_id": run_id,
                "run_manifest_ref": "run_manifest.json",
                "run_manifest_sha256": run_manifest["manifest_sha256"],
                "dataset_manifest_ref": dataset_manifest_ref,
                "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
                "dataset_protocol_sha256": dataset_manifest["dataset_protocol_sha256"],
                "protocol_sha256": protocol_sha256,
                "model_sha256": model_sha256,
                "execution_sha256": execution_sha256,
                "sampling_seed": sampling_seed,
                "profiles": profile_reports,
                "paired": paired,
            }
        )

        _new_json_file(staging / dataset_manifest_ref, dataset_manifest)
        _new_json_file(staging / "run_manifest.json", run_manifest)
        for profile_name, result_payload in profile_payloads.items():
            _new_json_file(staging / "profiles" / profile_name / "result.json", result_payload)
        _new_json_file(staging / "report.json", report)
        staging.rename(target)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def summarize_run_directories(run_directories: Sequence[str | os.PathLike[str]]) -> dict[str, Any]:
    """Read and summarize compatible report.json files."""

    reports = []
    for directory in run_directories:
        report = _load_json(Path(directory) / "report.json")
        verify_manifest_digest(report)
        reports.append(report)
    return summarize_repeated_reports(reports)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or summarize the pinned GSM8K evaluation protocol.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run direct and calculator profiles and atomically publish artifacts.")
    run.add_argument("--artifact-dir", required=True, help="New output artifact directory.")
    run.add_argument("--run-id", required=True, help="Stable identity for this evaluation run.")
    run.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL; not persisted.")
    run.add_argument("--served-model-name", required=True, help="Model name sent to the completion endpoint.")
    run.add_argument("--tokenizer-path", required=True, help="Local tokenizer directory used only for identity hashes.")
    run.add_argument("--backend-version", required=True, help="Version of the serving engine.")
    run.add_argument(
        "--split",
        choices=["development", "development128", "smoke", "repeat", "decision", "heldout"],
        default="smoke",
    )
    run.add_argument("--sampling-seed", type=int, default=0)
    run.add_argument("--max-concurrency", type=_positive_int, default=1)
    run.add_argument("--cache-dir", help="Optional Hugging Face Datasets cache directory.")
    run.add_argument("--model-id", default=GSM8K_MODEL_ID)
    run.add_argument("--model-revision", default=GSM8K_MODEL_REVISION)
    run.add_argument("--tokenizer-revision", default=GSM8K_MODEL_REVISION)
    run.add_argument("--model-artifact-sha256", help="Required identity addition for unpublished adapters/checkpoints.")
    run.add_argument("--model-label", default="base")
    run.add_argument("--backend-id", default="vllm-openai-compatible")
    run.add_argument("--endpoint-label", default="local-vllm")
    run.add_argument("--served-max-model-len", type=_positive_int, default=16384)
    run.add_argument("--served-dtype", default="bfloat16")
    run.add_argument("--tool-call-parser", default="qwen3_xml")
    run.add_argument("--reasoning-parser", default="qwen3")
    run.add_argument("--generation-config", default="vllm")
    run.add_argument("--served-model-runner", choices=["v1", "v2"], default="v1")
    run.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    run.add_argument("--gpu-memory-utilization", type=_fraction, default=0.90)
    run.add_argument("--max-num-seqs", type=_positive_int, default=4)
    run.add_argument("--timeout-seconds", type=_positive_int, default=300)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--bootstrap-samples", type=_positive_int, default=5000)
    run.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; credentials are never accepted as arguments.",
    )

    summarize = commands.add_parser("summarize", help="Aggregate compatible repeated runs.")
    summarize.add_argument("run_directories", nargs="+", help="Run directories containing report.json.")
    summarize.add_argument("--output", help="New JSON output file; omit to print only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.max_retries < 0:
                raise ValueError("max_retries must be non-negative")
            if not args.api_key_env:
                raise ValueError("api_key_env must be a non-empty string")
            report = run_gsm8k_baseline(
                artifact_dir=args.artifact_dir,
                run_id=args.run_id,
                base_url=args.base_url,
                served_model_name=args.served_model_name,
                tokenizer_path=args.tokenizer_path,
                backend_version=args.backend_version,
                split=args.split,
                sampling_seed=args.sampling_seed,
                max_concurrency=args.max_concurrency,
                cache_dir=args.cache_dir,
                model_id=args.model_id,
                model_revision=args.model_revision,
                tokenizer_revision=args.tokenizer_revision,
                model_artifact_sha256=args.model_artifact_sha256,
                model_label=args.model_label,
                backend_id=args.backend_id,
                endpoint_label=args.endpoint_label,
                served_max_model_len=args.served_max_model_len,
                served_dtype=args.served_dtype,
                tool_call_parser=args.tool_call_parser,
                reasoning_parser=args.reasoning_parser,
                generation_config=args.generation_config,
                served_model_runner=args.served_model_runner,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                api_key=os.environ.get(args.api_key_env),
                bootstrap_samples=args.bootstrap_samples,
            )
        else:
            report = summarize_run_directories(args.run_directories)
            if args.output:
                _new_json_file(Path(args.output), report)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
