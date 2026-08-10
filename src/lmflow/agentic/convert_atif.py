"""Command-line entry point for ATIF file conversion."""

import argparse
import sys
from collections.abc import Sequence
from typing import Optional

from lmflow.agentic.atif_io import convert_atif_file_to_conversation_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an ATIF JSON or JSONL file to an LMFlow conversation dataset.",
    )
    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        required=True,
        help="ATIF .json or .jsonl input file.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        required=True,
        help="LMFlow conversation .json output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file after the full conversion succeeds.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        converted_count = convert_atif_file_to_conversation_dataset(
            args.input_path,
            args.output_path,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Converted {converted_count} ATIF trajectory or trajectories to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
