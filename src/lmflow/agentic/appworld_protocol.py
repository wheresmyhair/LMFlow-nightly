"""Pinned task and provenance protocol for the AppWorld tiny slice."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lmflow.agentic.data_construction import canonical_json_sha256, verify_manifest_digest, with_manifest_digest
from lmflow.datasets import Dataset

APPWORLD_PROTOCOL_FORMAT_VERSION = "lmflow.appworld-tiny-protocol/v1"
APPWORLD_DATA_PILOT_PROTOCOL_FORMAT_VERSION = "lmflow.appworld-data-pilot-protocol/v1"
APPWORLD_SCENARIO_CURRICULUM_PROTOCOL_FORMAT_VERSION = "lmflow.appworld-scenario-curriculum-protocol/v1"
APPWORLD_REPOSITORY = "https://github.com/StonyBrookNLP/appworld.git"
APPWORLD_REVISION = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
APPWORLD_CODE_VERSION = "0.2.0.dev0"
APPWORLD_DATA_VERSION = "0.2.0"
APPWORLD_SOURCE_SPLIT = "dev"
APPWORLD_SOURCE_SPLIT_SIZE = 57
APPWORLD_SOURCE_SPLIT_SHA256 = "9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001"
APPWORLD_TINY_TASK_SET_SHA256 = "dafe2f2aad8a8cfabaa33550e9e9196d0a284b5bef2cca639b48e7c8e7dac67a"
APPWORLD_OFFICIAL_SPLITS = {
    "train": {
        "task_count": 90,
        "scenario_count": 30,
        "task_list_sha256": "93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37",
    },
    "dev": {
        "task_count": 57,
        "scenario_count": 19,
        "task_list_sha256": "9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001",
    },
    "test_normal": {
        "task_count": 168,
        "scenario_count": 56,
        "task_list_sha256": "c3af41497b6f2f0860a2ff8c09b335dca527e2cf48e59b4aabdb301b6b68db8f",
    },
    "test_challenge": {
        "task_count": 417,
        "scenario_count": 139,
        "task_list_sha256": "3c32b481042ac97f7d3477d53f5d196245c885c438d652944edc8a9a28e0f028",
    },
}

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

# The first complete train scenario in official order at each difficulty is a
# deterministic, scenario-disjoint paid-pilot candidate set. Actual provider
# calls and candidate count still require a separately approved run.
APPWORLD_DATA_PILOT_SCENARIOS = {
    "82e2fac": {"difficulty": 1},
    "692c77d": {"difficulty": 2},
    "6104387": {"difficulty": 3},
}
APPWORLD_DATA_PILOT_TASK_IDS = tuple(
    f"{scenario_id}_{variant}" for scenario_id in APPWORLD_DATA_PILOT_SCENARIOS for variant in (1, 2, 3)
)
APPWORLD_DATA_PILOT_TASK_SET_SHA256 = "d8a89fb3037ce6fe078d72517b80146c1c2cd1f6c007cad79beaa06aa3252327"

# The next two complete train scenarios in official order for difficulties 1
# and 2 form the first scenario-diverse curriculum candidate. Difficulty 3 is
# deliberately absent from this slice; the initial teacher pilot produced no
# verified success there, so it requires a separate data decision.
APPWORLD_SCENARIO_CURRICULUM_SCENARIOS = {
    "287e338": {"difficulty": 1, "official_scenario_index": 1},
    "27e1026": {"difficulty": 1, "official_scenario_index": 2},
    "2a163ab": {"difficulty": 2, "official_scenario_index": 1},
    "29caf6f": {"difficulty": 2, "official_scenario_index": 2},
}
APPWORLD_SCENARIO_CURRICULUM_TASK_IDS = tuple(
    f"{scenario_id}_{variant}" for scenario_id in APPWORLD_SCENARIO_CURRICULUM_SCENARIOS for variant in (1, 2, 3)
)
APPWORLD_SCENARIO_CURRICULUM_TASK_SET_SHA256 = "969c03630fe2f4f66d5a67aca5c7b91cda76c1146ee33c221699bc892a370da5"


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


def _scenario_id(task_id: str) -> str:
    return task_id.rsplit("_", maxsplit=1)[0]


def _validate_official_splits(root: Path, load_task_ids: Any) -> dict[str, tuple[str, ...]]:
    split_task_ids = {}
    split_scenarios = {}
    for split, expected in APPWORLD_OFFICIAL_SPLITS.items():
        split_path = root / "data" / "datasets" / f"{split}.txt"
        if _sha256_file(split_path) != expected["task_list_sha256"]:
            raise ValueError(f"pinned AppWorld {split!r} task-list digest mismatch")
        task_ids = tuple(load_task_ids(split, num_tasks_per_scenario=3))
        scenarios = {_scenario_id(task_id) for task_id in task_ids}
        if len(task_ids) != expected["task_count"]:
            raise ValueError(f"pinned AppWorld {split!r} task count changed")
        if len(scenarios) != expected["scenario_count"]:
            raise ValueError(f"pinned AppWorld {split!r} scenario count changed")
        split_task_ids[split] = task_ids
        split_scenarios[split] = scenarios
    split_names = tuple(APPWORLD_OFFICIAL_SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_scenarios[left] & split_scenarios[right]
            if overlap:
                raise ValueError(f"AppWorld scenarios overlap between {left!r} and {right!r}: {sorted(overlap)}")
    if not set(APPWORLD_DATA_PILOT_TASK_IDS).issubset(split_task_ids["train"]):
        raise ValueError("AppWorld data-pilot tasks are absent from train")
    if not set(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS).issubset(split_task_ids["train"]):
        raise ValueError("AppWorld scenario-curriculum tasks are absent from train")
    if set(APPWORLD_DATA_PILOT_TASK_IDS) & set(APPWORLD_SCENARIO_CURRICULUM_TASK_IDS):
        raise RuntimeError("AppWorld data-pilot and scenario-curriculum task sets overlap")
    for difficulty, scenario_id in enumerate(APPWORLD_DATA_PILOT_SCENARIOS, start=1):
        difficulty_ids = tuple(load_task_ids("train", difficulty=difficulty, num_tasks_per_scenario=3))
        if not difficulty_ids or _scenario_id(difficulty_ids[0]) != scenario_id:
            raise ValueError(f"AppWorld data-pilot difficulty-{difficulty} selection rule changed")
    for difficulty in (1, 2):
        difficulty_ids = tuple(load_task_ids("train", difficulty=difficulty, num_tasks_per_scenario=3))
        ordered_scenarios = tuple(dict.fromkeys(_scenario_id(task_id) for task_id in difficulty_ids))
        selected_scenarios = tuple(
            scenario_id
            for scenario_id, metadata in APPWORLD_SCENARIO_CURRICULUM_SCENARIOS.items()
            if metadata["difficulty"] == difficulty
        )
        if selected_scenarios != ordered_scenarios[1:3]:
            raise ValueError(f"AppWorld scenario-curriculum difficulty-{difficulty} selection rule changed")
    return split_task_ids


def canonical_appworld_instance_id(task_id: str) -> str:
    """Build a stable identity for a pinned AppWorld task."""

    if task_id not in APPWORLD_TINY_TASK_IDS:
        raise ValueError(f"task {task_id!r} is outside the pinned AppWorld tiny task set")
    return f"{APPWORLD_REPOSITORY}@{APPWORLD_REVISION}/data-{APPWORLD_DATA_VERSION}/{APPWORLD_SOURCE_SPLIT}/{task_id}"


def canonical_appworld_sliced_instance_id(task_id: str, *, source_split: str) -> str:
    """Build an identity for a pinned AppWorld development slice."""

    if source_split == APPWORLD_SOURCE_SPLIT:
        return canonical_appworld_instance_id(task_id)
    if source_split == "train" and task_id in {
        *APPWORLD_DATA_PILOT_TASK_IDS,
        *APPWORLD_SCENARIO_CURRICULUM_TASK_IDS,
    }:
        return f"{APPWORLD_REPOSITORY}@{APPWORLD_REVISION}/data-{APPWORLD_DATA_VERSION}/train/{task_id}"
    raise ValueError(f"task {task_id!r} is outside the pinned AppWorld {source_split!r} slice")


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


def _load_pinned_appworld_train_slice_dataset(
    *,
    root: Path,
    task_type: Any,
    split_task_ids: Mapping[str, tuple[str, ...]],
    format_version: str,
    task_set_key: str,
    task_ids: Sequence[str],
    task_set_sha256: str,
    scenarios: Mapping[str, Mapping[str, Any]],
    selection_rule: str,
) -> tuple[Dataset, dict[str, Any]]:
    if canonical_json_sha256(list(task_ids)) != task_set_sha256:
        raise RuntimeError(f"AppWorld {task_set_key} constant does not match its pinned digest")

    instances = []
    instance_manifest = []
    for task_id in task_ids:
        scenario_id = _scenario_id(task_id)
        difficulty = scenarios[scenario_id]["difficulty"]
        spec_path = root / "data" / "tasks" / task_id / "specs.json"
        task_spec_sha256 = _sha256_file(spec_path)
        task = task_type.load(task_id=task_id, load_ground_truth=False)
        try:
            instance_id = canonical_appworld_sliced_instance_id(task_id, source_split="train")
            instance = {
                "text": task.instruction,
                "instance_id": instance_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "difficulty": difficulty,
                "source_split": "train",
                "task_spec_sha256": task_spec_sha256,
            }
            instances.append(instance)
            instance_manifest.append({key: value for key, value in instance.items() if key != "text"})
        finally:
            task.close()

    dataset = Dataset.create_from_dict({"type": "text_only", "instances": instances})
    split_identity = {
        split: {
            **APPWORLD_OFFICIAL_SPLITS[split],
            "task_ids_sha256": canonical_json_sha256(list(task_ids)),
        }
        for split, task_ids in split_task_ids.items()
    }
    protocol_identity = {
        "format_version": format_version,
        "source": {
            "repository": APPWORLD_REPOSITORY,
            "revision": APPWORLD_REVISION,
            "code_version": APPWORLD_CODE_VERSION,
            "data_version": APPWORLD_DATA_VERSION,
            "split": "train",
        },
        "official_splits": split_identity,
        "scenario_disjoint": True,
        task_set_key: {
            "task_ids": list(task_ids),
            "task_ids_sha256": task_set_sha256,
            "selection_rule": selection_rule,
            "tasks_per_scenario": 3,
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
        },
        "gold_visibility": "ground_truth_not_loaded_for_dataset_projection",
    }
    manifest = with_manifest_digest(
        {
            **protocol_identity,
            "dataset_protocol_sha256": canonical_json_sha256(protocol_identity),
            "instance_count": len(instance_manifest),
            "instances": instance_manifest,
            "selected_content_sha256": canonical_json_sha256([item["task_spec_sha256"] for item in instance_manifest]),
        }
    )
    return dataset, manifest


def load_pinned_appworld_data_pilot_dataset(
    *,
    appworld_root: str | os.PathLike[str],
) -> tuple[Dataset, dict[str, Any]]:
    """Load the frozen train pilot without loading ground truth into the Dataset."""

    root = _configure_root(appworld_root)
    _, Task, load_task_ids = _require_appworld()
    split_task_ids = _validate_official_splits(root, load_task_ids)
    return _load_pinned_appworld_train_slice_dataset(
        root=root,
        task_type=Task,
        split_task_ids=split_task_ids,
        format_version=APPWORLD_DATA_PILOT_PROTOCOL_FORMAT_VERSION,
        task_set_key="pilot_task_set",
        task_ids=APPWORLD_DATA_PILOT_TASK_IDS,
        task_set_sha256=APPWORLD_DATA_PILOT_TASK_SET_SHA256,
        scenarios=APPWORLD_DATA_PILOT_SCENARIOS,
        selection_rule="first complete train scenario in official order at each difficulty",
    )


def load_pinned_appworld_scenario_curriculum_dataset(
    *,
    appworld_root: str | os.PathLike[str],
) -> tuple[Dataset, dict[str, Any]]:
    """Load the first scenario-diverse d1/d2 train curriculum candidate."""

    root = _configure_root(appworld_root)
    _, Task, load_task_ids = _require_appworld()
    split_task_ids = _validate_official_splits(root, load_task_ids)
    return _load_pinned_appworld_train_slice_dataset(
        root=root,
        task_type=Task,
        split_task_ids=split_task_ids,
        format_version=APPWORLD_SCENARIO_CURRICULUM_PROTOCOL_FORMAT_VERSION,
        task_set_key="scenario_curriculum_task_set",
        task_ids=APPWORLD_SCENARIO_CURRICULUM_TASK_IDS,
        task_set_sha256=APPWORLD_SCENARIO_CURRICULUM_TASK_SET_SHA256,
        scenarios=APPWORLD_SCENARIO_CURRICULUM_SCENARIOS,
        selection_rule=(
            "next two complete train scenarios in official order after the initial pilot "
            "for each of difficulties 1 and 2"
        ),
    )


__all__ = [
    "APPWORLD_CODE_VERSION",
    "APPWORLD_DATA_PILOT_PROTOCOL_FORMAT_VERSION",
    "APPWORLD_DATA_PILOT_SCENARIOS",
    "APPWORLD_DATA_PILOT_TASK_IDS",
    "APPWORLD_DATA_VERSION",
    "APPWORLD_OFFICIAL_SPLITS",
    "APPWORLD_PROTOCOL_FORMAT_VERSION",
    "APPWORLD_REPOSITORY",
    "APPWORLD_REVISION",
    "APPWORLD_SCENARIO_CURRICULUM_PROTOCOL_FORMAT_VERSION",
    "APPWORLD_SCENARIO_CURRICULUM_SCENARIOS",
    "APPWORLD_SCENARIO_CURRICULUM_TASK_IDS",
    "APPWORLD_SCENARIO_CURRICULUM_TASK_SET_SHA256",
    "APPWORLD_SOURCE_SPLIT",
    "APPWORLD_TINY_SCENARIOS",
    "APPWORLD_TINY_TASK_IDS",
    "canonical_appworld_instance_id",
    "canonical_appworld_sliced_instance_id",
    "canonical_json_sha256",
    "load_pinned_appworld_data_pilot_dataset",
    "load_pinned_appworld_scenario_curriculum_dataset",
    "load_pinned_appworld_tiny_dataset",
    "verify_manifest_digest",
    "with_manifest_digest",
]
