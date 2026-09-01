#!/usr/bin/env python3
"""Build a sealed, no-request AppWorld Train d1/d2 planning bundle."""

from __future__ import annotations

import argparse
import json

from lmflow.agentic.appworld_data_plan import build_appworld_train_d1_d2_plan_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--appworld-root", required=True)
    parser.add_argument("--l0-aggregate-dir", required=True)
    parser.add_argument("--code-base-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_appworld_train_d1_d2_plan_bundle(
        artifact_dir=args.artifact_dir,
        appworld_root=args.appworld_root,
        l0_aggregate_dir=args.l0_aggregate_dir,
        code_commit=args.code_base_commit,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
