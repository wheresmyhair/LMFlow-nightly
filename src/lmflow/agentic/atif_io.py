"""File conversion helpers for ATIF conversation datasets."""

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Optional, Union

from lmflow.agentic.atif import atif_trajectory_to_conversation

PathLike = Union[str, os.PathLike]
_SUPPORTED_INPUT_SUFFIXES = frozenset({".json", ".jsonl"})


class _StrictJSONError(ValueError):
    """An error raised for JSON extensions rejected by the ATIF reader."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJSONError(f"duplicate object key {key!r}")
        value[key] = item
    return value


def _reject_non_finite_number(value: str) -> None:
    raise _StrictJSONError(f"non-finite number {value!r} is not valid JSON")


def _load_strict_json(document: str) -> Any:
    return json.loads(
        document,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite_number,
    )


def _record_identity(record: Any) -> str:
    if not isinstance(record, Mapping):
        return ""
    for key in ("trajectory_id", "session_id"):
        value = record.get(key)
        if isinstance(value, str):
            return f" ({key}={value!r})"
    return ""


def _record_location(input_path: Path, line_number: Optional[int]) -> str:
    if line_number is None:
        return f"{input_path}:document"
    return f"{input_path}:{line_number}"


def _ensure_atif_object(record: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{location}: each input record must be one ATIF object; use JSONL for multiple trajectories")
    return record


def _iter_atif_records(input_path: Path) -> Iterator[tuple[Optional[int], Mapping[str, Any]]]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        try:
            document = input_path.read_bytes().decode("utf-8")
            record = _load_strict_json(document)
        except UnicodeDecodeError as error:
            raise ValueError(f"{input_path}:document: input is not valid UTF-8: {error}") from error
        except (json.JSONDecodeError, _StrictJSONError) as error:
            raise ValueError(f"{input_path}:document: invalid JSON: {error}") from error
        yield None, _ensure_atif_object(record, f"{input_path}:document")
        return

    with input_path.open("rb") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                document = line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{input_path}:{line_number}: input is not valid UTF-8: {error}") from error
            try:
                record = _load_strict_json(document)
            except (json.JSONDecodeError, _StrictJSONError) as error:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {error}") from error
            yield line_number, _ensure_atif_object(record, f"{input_path}:{line_number}")


def _validate_paths(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if input_path.suffix.lower() not in _SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"input_path must use one of these extensions: {supported}")
    if output_path.suffix.lower() != ".json":
        raise ValueError("output_path must use the .json extension")
    if not input_path.is_file():
        raise ValueError(f"input_path is not a file: {input_path}")
    if not output_path.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output_path.parent}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different files")
    if os.path.lexists(output_path) and not overwrite:
        raise FileExistsError(f"output_path already exists: {output_path}")


def _publish_output(temporary_path: Path, output_path: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary_path, output_path)
        return

    try:
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise FileExistsError(f"output_path already exists: {output_path}") from error
    try:
        temporary_path.unlink()
    except OSError:
        # The output is already atomically visible. A hidden temporary link is
        # preferable to reporting failure after successfully publishing it.
        pass


def convert_atif_file_to_conversation_dataset(
    input_path: PathLike,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> int:
    """Convert one ATIF JSON or JSONL file into an LMFlow conversation dataset.

    A ``.json`` input contains exactly one ATIF trajectory object. A ``.jsonl``
    input contains one trajectory object per non-blank line. Records retain their
    input order. The output is published only after every record passes the
    existing fail-closed ATIF-to-conversation conversion.

    This generic file boundary does not infer the extra model-visibility facts
    required by deterministic ``llm_call_count=0`` steps. Exporters with those
    facts should call ``atif_trajectory_to_conversation`` directly.

    Returns
    -------
    int
        Number of converted trajectories.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    _validate_paths(input_path, output_path, overwrite)

    temporary_path: Optional[Path] = None
    converted_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            output_file.write('{"type":"conversation","instances":[')
            for line_number, trajectory in _iter_atif_records(input_path):
                location = _record_location(input_path, line_number)
                try:
                    conversation = atif_trajectory_to_conversation(trajectory)
                except ValueError as error:
                    identity = _record_identity(trajectory)
                    raise ValueError(f"{location}{identity}: {error}") from error

                if converted_count:
                    output_file.write(",")
                json.dump(
                    conversation,
                    output_file,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                converted_count += 1

            if converted_count == 0:
                raise ValueError(f"{input_path}: no ATIF trajectories found")

            output_file.write("]}\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        _publish_output(temporary_path, output_path, overwrite)
        temporary_path = None
        return converted_count
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
