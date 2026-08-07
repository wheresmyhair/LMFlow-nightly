#!/usr/bin/env python3
"""Fail when a Sphinx warning reports a broken internal documentation link."""

import argparse
from pathlib import Path


BLOCKING_WARNING_MARKERS = (
    "[myst.xref_missing]",
    "[myst.iref_missing]",
    "[myst.xref_ambiguous]",
    "[myst.iref_ambiguous]",
    "[ref.",
    "[toc.",
)

BLOCKING_WARNING_MESSAGES = (
    "cross-reference target not found",
    "reference target not found",
    "undefined label:",
    "failed to create a cross reference",
    "toctree contains reference to",
    "document isn't included in any toctree",
    "duplicated entry found in toctree",
)


def find_blocking_warnings(warning_log: str) -> list[str]:
    """Return warnings related to internal references and navigation."""
    blocking_warnings = []
    for line in warning_log.splitlines():
        normalized_line = line.lower()
        if any(marker in normalized_line for marker in BLOCKING_WARNING_MARKERS) or any(
            message in normalized_line for message in BLOCKING_WARNING_MESSAGES
        ):
            blocking_warnings.append(line)
    return blocking_warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warning_log", type=Path, help="Sphinx warning log generated with -w")
    args = parser.parse_args()

    blocking_warnings = find_blocking_warnings(args.warning_log.read_text(encoding="utf-8"))
    if not blocking_warnings:
        print("Documentation internal-link check passed.")
        return 0

    print("Documentation internal-link check failed:")
    for warning in blocking_warnings:
        print(warning)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
