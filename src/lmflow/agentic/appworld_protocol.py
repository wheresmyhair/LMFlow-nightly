"""Pinned task and provenance protocol for the AppWorld tiny slice."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lmflow.datasets import Dataset

APPWORLD_PROTOCOL_FORMAT_VERSION = "lmflow.appworld-tiny-protocol/v1"
APPWORLD_REPOSITORY = "https://github.com/StonyBrookNLP/appworld.git"
APPWORLD_REVISION = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
APPWORLD_CODE_VERSION = "0.2.0.dev0"
APPWORLD_DATA_VERSION = "0.2.0"
APPWORLD_SOURCE_SPLIT = "dev"
APPWORLD_SOURCE_SPLIT_SIZE = 57
APPWORLD_SOURCE_SPLIT_SHA256 = "9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001"
APPWORLD_TINY_TASK_SET_SHA256 = "dafe2f2aad8a8cfabaa33550e9e9196d0a284b5bef2cca639b48e7c8e7dac67a"

# Three complete scenario groups make AppWorld's scenario-goal-completion
# metric meaningful while covering one group at each official difficulty.
APPWORLD_TINY_SCENARIOS = {
    "396c5a2": {"difficulty": 1, "coverage": ["spotify", "state-change"]},
    "6c2c621": {"difficulty": 2, "coverage": ["simple_note", "file_system", "cross-app"]},
    "530b157": {"difficulty": 3, "coverage": ["phone", "venmo", "cross-app"]},
}
APPWORLD_TINY_TASK_IDS = tuple(
    f"{scenario_id}_{variant}" for scenario_id in APPWORLD_TINY_SCENARIOS for variant in (1, 2, 3)
)


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using a stable protocol encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def with_manifest_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON copy with a digest over every existing field."""

    if not isinstance(payload, Mapping):
        raise TypeError("manifest payload must be a mapping")
    if "manifest_sha256" in payload:
        raise ValueError("manifest payload must not already contain manifest_sha256")
    copied = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    copied["manifest_sha256"] = canonical_json_sha256(copied)
    return copied


def verify_manifest_digest(manifest: Mapping[str, Any]) -> None:
    """Validate a manifest produced by :func:`with_manifest_digest`."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("manifest must contain a SHA-256 manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_json_sha256(payload) != expected:
        raise ValueError("manifest_sha256 does not match the manifest content")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_root(appworld_root: str | os.PathLike[str]) -> Path:
    root = Path(appworld_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AppWorld root does not exist: {root}")
    os.environ["APPWORLD_ROOT"] = str(root)
    return root


def _require_appworld() -> tuple[Any, Any, Any]:
    try:
        import appworld
        from appworld.task import Task, load_task_ids
    except ImportError as error:
        raise ImportError(
            "AppWorld is unavailable; synchronize the agentic lock and run scripts/agentic/bootstrap_appworld.sh"
        ) from error
    if appworld.__version__ != APPWORLD_CODE_VERSION:
        raise RuntimeError(f"AppWorld version {appworld.__version__!r} does not match pinned {APPWORLD_CODE_VERSION!r}")
    return appworld, Task, load_task_ids


def _validate_source(root: Path, load_task_ids: Any) -> dict[str, set[str]]:
    version_path = root / "data" / "version.txt"
    if not version_path.is_file():
        raise FileNotFoundError(f"missing AppWorld data version file: {version_path}")
    data_version = version_path.read_text(encoding="utf-8").strip()
    if data_version != APPWORLD_DATA_VERSION:
        raise ValueError(f"AppWorld data version {data_version!r} does not match pinned {APPWORLD_DATA_VERSION!r}")

    split_path = root / "data" / "datasets" / f"{APPWORLD_SOURCE_SPLIT}.txt"
    split_sha256 = _sha256_file(split_path)
    if split_sha256 != APPWORLD_SOURCE_SPLIT_SHA256:
        raise ValueError(f"pinned AppWorld {APPWORLD_SOURCE_SPLIT!r} task-list digest mismatch")
    source_ids = load_task_ids(APPWORLD_SOURCE_SPLIT, num_tasks_per_scenario=3)
    if len(source_ids) != APPWORLD_SOURCE_SPLIT_SIZE:
        raise ValueError(
            f"pinned AppWorld {APPWORLD_SOURCE_SPLIT!r} contains {len(source_ids)} tasks; "
            f"expected {APPWORLD_SOURCE_SPLIT_SIZE}"
        )
    missing = sorted(set(APPWORLD_TINY_TASK_IDS) - set(source_ids))
    if missing:
        raise ValueError(f"pinned AppWorld tiny tasks are absent from dev: {missing}")

    by_difficulty = {
        str(difficulty): set(load_task_ids(APPWORLD_SOURCE_SPLIT, difficulty=difficulty)) for difficulty in (1, 2, 3)
    }
    for scenario_id, scenario in APPWORLD_TINY_SCENARIOS.items():
        expected_difficulty = str(scenario["difficulty"])
        scenario_ids = {f"{scenario_id}_{variant}" for variant in (1, 2, 3)}
        if not scenario_ids.issubset(by_difficulty[expected_difficulty]):
            raise ValueError(f"AppWorld scenario {scenario_id!r} no longer has difficulty {expected_difficulty}")
    return by_difficulty


def canonical_appworld_instance_id(task_id: str) -> str:
    """Build a stable identity for a pinned AppWorld task."""

    if task_id not in APPWORLD_TINY_TASK_IDS:
        raise ValueError(f"task {task_id!r} is outside the pinned AppWorld tiny task set")
    return f"{APPWORLD_REPOSITORY}@{APPWORLD_REVISION}/data-{APPWORLD_DATA_VERSION}/{APPWORLD_SOURCE_SPLIT}/{task_id}"


def _normalize_task_ids(task_ids: Sequence[str] | None) -> tuple[str, ...]:
    if task_ids is None:
        return APPWORLD_TINY_TASK_IDS
    if isinstance(task_ids, str | bytes) or not isinstance(task_ids, Sequence):
        raise TypeError("task_ids must be a sequence")
    requested = tuple(task_ids)
    if not requested:
        raise ValueError("task_ids must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("task_ids must not contain duplicates")
    unknown = sorted(set(requested) - set(APPWORLD_TINY_TASK_IDS))
    if unknown:
        raise ValueError(f"tasks are outside the pinned AppWorld tiny task set: {unknown}")
    requested_set = set(requested)
    return tuple(task_id for task_id in APPWORLD_TINY_TASK_IDS if task_id in requested_set)


def load_pinned_appworld_tiny_dataset(
    *,
    appworld_root: str | os.PathLike[str],
    task_ids: Sequence[str] | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Load model-visible task facts and a ground-truth-free provenance manifest."""

    root = _configure_root(appworld_root)
    _, Task, load_task_ids = _require_appworld()
    _validate_source(root, load_task_ids)
    selected_ids = _normalize_task_ids(task_ids)
    if canonical_json_sha256(list(APPWORLD_TINY_TASK_IDS)) != APPWORLD_TINY_TASK_SET_SHA256:
        raise RuntimeError("AppWorld tiny task-set constant does not match its pinned digest")

    instances = []
    instance_manifest = []
    for task_id in selected_ids:
        scenario_id = task_id.rsplit("_", maxsplit=1)[0]
        difficulty = APPWORLD_TINY_SCENARIOS[scenario_id]["difficulty"]
        spec_path = root / "data" / "tasks" / task_id / "specs.json"
        task_spec_sha256 = _sha256_file(spec_path)
        task = Task.load(task_id=task_id, load_ground_truth=False)
        try:
            instance_id = canonical_appworld_instance_id(task_id)
            instances.append(
                {
                    "text": task.instruction,
                    "instance_id": instance_id,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "source_split": APPWORLD_SOURCE_SPLIT,
                    "task_spec_sha256": task_spec_sha256,
                }
            )
            instance_manifest.append(
                {
                    "instance_id": instance_id,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "task_spec_sha256": task_spec_sha256,
                }
            )
        finally:
            task.close()

    dataset = Dataset.create_from_dict({"type": "text_only", "instances": instances})
    protocol_identity = {
        "format_version": APPWORLD_PROTOCOL_FORMAT_VERSION,
        "source": {
            "repository": APPWORLD_REPOSITORY,
            "revision": APPWORLD_REVISION,
            "code_version": APPWORLD_CODE_VERSION,
            "data_version": APPWORLD_DATA_VERSION,
            "split": APPWORLD_SOURCE_SPLIT,
            "split_size": APPWORLD_SOURCE_SPLIT_SIZE,
            "split_task_list_sha256": APPWORLD_SOURCE_SPLIT_SHA256,
        },
        "tiny_task_set": {
            "task_ids": list(APPWORLD_TINY_TASK_IDS),
            "task_ids_sha256": APPWORLD_TINY_TASK_SET_SHA256,
            "selection_rule": (
                "manually frozen complete dev scenario groups: one official group per difficulty, "
                "chosen to cover state-changing and cross-app workflows"
            ),
            "tasks_per_scenario": 3,
            "scenario_count": 3,
            "scenarios": APPWORLD_TINY_SCENARIOS,
        },
        "gold_visibility": "ground_truth_not_loaded_for_dataset_projection",
    }
    manifest = with_manifest_digest(
        {
            **protocol_identity,
            "dataset_protocol_sha256": canonical_json_sha256(protocol_identity),
            "selected_task_ids": list(selected_ids),
            "selected_task_ids_sha256": canonical_json_sha256(list(selected_ids)),
            "instance_count": len(instance_manifest),
            "instances": instance_manifest,
            "selected_content_sha256": canonical_json_sha256([item["task_spec_sha256"] for item in instance_manifest]),
        }
    )
    return dataset, manifest


__all__ = [
    "APPWORLD_CODE_VERSION",
    "APPWORLD_DATA_VERSION",
    "APPWORLD_PROTOCOL_FORMAT_VERSION",
    "APPWORLD_REPOSITORY",
    "APPWORLD_REVISION",
    "APPWORLD_SOURCE_SPLIT",
    "APPWORLD_TINY_SCENARIOS",
    "APPWORLD_TINY_TASK_IDS",
    "canonical_appworld_instance_id",
    "canonical_json_sha256",
    "load_pinned_appworld_tiny_dataset",
    "verify_manifest_digest",
    "with_manifest_digest",
]
