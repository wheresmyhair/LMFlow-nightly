#!/usr/bin/env python3
"""Validate LMFlow Nightly change fragments and build modular release notes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


MODULES = ("core", "data", "agentic", "ci", "docs", "packaging", "tooling")
MODULE_TITLES = {
    "core": "Core",
    "data": "Data / protocol",
    "agentic": "Agentic",
    "ci": "CI/CD",
    "docs": "Documentation",
    "packaging": "Packaging / environment",
    "tooling": "Private tooling",
}
KINDS = ("feature", "fix", "maintenance", "documentation")
UPSTREAM_STATES = ("candidate", "private", "blocked")
REQUIRED_FIELDS = {"title", "module", "kind", "upstream", "summary", "review"}
OPTIONAL_FIELDS = {"breaking"}
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*\.json$")


class FragmentError(ValueError):
    """Raised when a change fragment does not satisfy the repository schema."""


@dataclass(frozen=True)
class Change:
    title: str
    module: str
    kind: str
    upstream: str
    summary: str
    review: tuple[str, ...]
    breaking: bool
    source: str


def _text(value: object, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FragmentError(f"{path}: {field!r} must be a non-empty string")
    return value.strip()


def _load_fragment(path: Path, root: Path) -> Change:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FragmentError(f"{path}: invalid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise FragmentError(f"{path}: the top-level JSON value must be an object")

    fields = set(raw)
    missing = REQUIRED_FIELDS - fields
    unknown = fields - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if missing:
        raise FragmentError(f"{path}: missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise FragmentError(f"{path}: unknown fields: {', '.join(sorted(unknown))}")

    module = _text(raw["module"], "module", path)
    kind = _text(raw["kind"], "kind", path)
    upstream = _text(raw["upstream"], "upstream", path)
    if module not in MODULES:
        raise FragmentError(f"{path}: module must be one of {', '.join(MODULES)}")
    if kind not in KINDS:
        raise FragmentError(f"{path}: kind must be one of {', '.join(KINDS)}")
    if upstream not in UPSTREAM_STATES:
        raise FragmentError(f"{path}: upstream must be one of {', '.join(UPSTREAM_STATES)}")
    if path.parent.name != module:
        raise FragmentError(f"{path}: module {module!r} must match directory {path.parent.name!r}")

    review = raw["review"]
    if not isinstance(review, list) or not review:
        raise FragmentError(f"{path}: 'review' must be a non-empty list")
    review_items = tuple(_text(item, "review item", path) for item in review)

    breaking = raw.get("breaking", False)
    if not isinstance(breaking, bool):
        raise FragmentError(f"{path}: 'breaking' must be true or false")

    return Change(
        title=_text(raw["title"], "title", path),
        module=module,
        kind=kind,
        upstream=upstream,
        summary=_text(raw["summary"], "summary", path),
        review=review_items,
        breaking=breaking,
        source=path.relative_to(root).as_posix(),
    )


def load_fragments(root: Path) -> list[Change]:
    if not root.is_dir():
        raise FragmentError(f"change fragment directory does not exist: {root}")

    changes = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        if path.parent.parent != root:
            raise FragmentError(f"{path}: fragments must use .changes/<module>/<slug>.json")
        if not SLUG_PATTERN.fullmatch(path.name):
            raise FragmentError(f"{path}: filename must use lowercase letters, digits, and hyphens")
        changes.append(_load_fragment(path, root))
    return changes


def render_notes(changes: Sequence[Change], version: str) -> str:
    if not changes:
        raise FragmentError("cannot build What's New without change fragments")

    upstream_counts = Counter(change.upstream for change in changes)
    lines = [
        f"# LMFlow Nightly {version} — What's New",
        "",
        "Generated from reviewed change fragments. Upstream status is recorded per change.",
        "",
        "## Export overview",
        "",
    ]
    for state in UPSTREAM_STATES:
        lines.append(f"- {state.title()}: {upstream_counts[state]}")

    for module in MODULES:
        module_changes = sorted(
            (change for change in changes if change.module == module),
            key=lambda change: (change.title.casefold(), change.source),
        )
        if not module_changes:
            continue
        lines.extend(["", f"## {MODULE_TITLES[module]}", ""])
        for change in module_changes:
            lines.extend(
                [
                    f"### {change.title}",
                    "",
                    f"- Kind: {change.kind}",
                    f"- Upstream: {change.upstream}",
                    f"- Breaking: {'yes' if change.breaking else 'no'}",
                    "",
                    change.summary,
                    "",
                    "Review focus:",
                ]
            )
            lines.extend(f"- {item}" for item in change.review)
            lines.extend(["", f"Fragment: `{change.source}`", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments-dir", type=Path, default=Path(".changes"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="validate all active change fragments")
    build = subparsers.add_parser("build", help="write modular What's New notes")
    build.add_argument("--version", required=True, help="release identifier, for example 2026.08")
    build.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        changes = load_fragments(args.fragments_dir)
        if args.command == "check":
            print(f"Validated {len(changes)} change fragment(s).")
            return 0

        notes = render_notes(changes, args.version)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(notes, encoding="utf-8")
        print(f"Wrote {len(changes)} change fragment(s) to {args.output}.")
        return 0
    except FragmentError as exc:
        print(f"Change fragment error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
