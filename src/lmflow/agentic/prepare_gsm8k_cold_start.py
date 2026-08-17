"""Prepare verified GSM8K calculator/direct cold-start conversations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from lmflow.agentic.gsm8k_cold_start import (
    GSM8K_COLD_START_DEFAULT_BLOCK_SIZE,
    GSM8K_COLD_START_SELECTION_SEED,
    run_gsm8k_cold_start_factory,
)
from lmflow.agentic.gsm8k_protocol import GSM8K_MODEL_REVISION


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paired verified calculator/direct SFT data from pinned public GSM8K train annotations."
    )
    parser.add_argument("--artifact-dir", required=True, help="New output artifact directory.")
    parser.add_argument("--run-id", required=True, help="Stable identity for this data construction run.")
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Local Qwen3 tokenizer directory; path is not persisted.",
    )
    parser.add_argument("--tokenizer-revision", default=GSM8K_MODEL_REVISION)
    parser.add_argument("--task-count", type=_positive_int, default=8)
    parser.add_argument("--selection-seed", type=int, default=GSM8K_COLD_START_SELECTION_SEED)
    parser.add_argument("--block-size", type=_positive_int, default=GSM8K_COLD_START_DEFAULT_BLOCK_SIZE)
    parser.add_argument("--cache-dir", help="Optional Hugging Face Datasets cache directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_gsm8k_cold_start_factory(
            artifact_dir=args.artifact_dir,
            run_id=args.run_id,
            task_count=args.task_count,
            tokenizer_path=args.tokenizer_path,
            tokenizer_revision=args.tokenizer_revision,
            selection_seed=args.selection_seed,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
