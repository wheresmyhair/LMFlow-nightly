"""Command-line entry point for GSM8K reward-tool data generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence

from lmflow.agentic.completion import OpenAICompatibleCompletionBackend
from lmflow.agentic.gsm8k_dataset import generate_gsm8k_tool_dataset, load_gsm8k_tasks


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _top_p(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate raw GSM8K-tool trajectories and a successful conversation dataset.",
    )
    parser.add_argument("--artifact-dir", required=True, help="New output artifact directory.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model-name", required=True, help="Model name sent to the completion endpoint.")
    parser.add_argument("--session-id", required=True, help="Stable identity for this generation run.")
    parser.add_argument("--split", default="train", help="Dataset split and task identity component.")
    parser.add_argument("--start-index", type=int, default=0, help="First source row index, inclusive.")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        required=True,
        help="Exact number of source rows to process; required to prevent accidental full-dataset runs.",
    )
    parser.add_argument("--dataset-name", default="openai/gsm8k", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default="main", help="Hugging Face dataset configuration.")
    parser.add_argument("--input-path", help="Local GSM8K JSON or JSONL file; overrides the dataset name.")
    parser.add_argument("--cache-dir", help="Optional Hugging Face Datasets cache directory.")
    parser.add_argument("--rollouts-per-task", type=_positive_int, default=1)
    parser.add_argument("--max-steps", type=_positive_int, default=4)
    parser.add_argument("--temperature", type=_non_negative_float, default=0.7)
    parser.add_argument("--top-p", type=_top_p, default=1.0)
    parser.add_argument("--max-tokens", type=_positive_int, default=4096)
    parser.add_argument("--seed", type=int, help="Optional provider-side sampling seed.")
    parser.add_argument("--timeout-seconds", type=_positive_int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; an unset variable uses the local-server default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if args.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not args.api_key_env:
            raise ValueError("api_key_env must be a non-empty string")

        tasks = load_gsm8k_tasks(
            split=args.split,
            start_index=args.start_index,
            limit=args.limit,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            input_path=args.input_path,
            cache_dir=args.cache_dir,
        )
        model_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        if args.seed is not None:
            model_kwargs["seed"] = args.seed

        with OpenAICompatibleCompletionBackend(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env),
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        ) as backend:
            report = generate_gsm8k_tool_dataset(
                backend,
                tasks,
                artifact_dir=args.artifact_dir,
                model_name=args.model_name,
                session_id=args.session_id,
                model_kwargs=model_kwargs,
                rollouts_per_task=args.rollouts_per_task,
                max_steps=args.max_steps,
            )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
