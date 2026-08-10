import json

import pytest

from lmflow.agentic import convert_atif_file_to_conversation_dataset
from lmflow.agentic.convert_atif import main
from lmflow.args import DatasetArguments
from lmflow.datasets.dataset import Dataset


def _trajectory(trajectory_id="trajectory-1", answer="DONE"):
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "run-1",
        "trajectory_id": trajectory_id,
        "agent": {
            "name": "test-agent",
            "version": "1.0.0",
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "Solve the task"},
            {
                "step_id": 2,
                "source": "agent",
                "message": answer,
                "llm_call_count": 1,
            },
        ],
    }


def _tool_trajectory():
    trajectory = _trajectory("trajectory-tool", "")
    trajectory["agent"]["tool_definitions"] = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    trajectory["steps"][1] = {
        "step_id": 2,
        "source": "agent",
        "message": "",
        "llm_call_count": 1,
        "tool_calls": [
            {
                "tool_call_id": "call-1",
                "function_name": "bash",
                "arguments": {"command": "printf UNICODE_OK_✓"},
            }
        ],
        "observation": {
            "results": [
                {
                    "source_call_id": "call-1",
                    "content": "UNICODE_OK_✓",
                }
            ]
        },
    }
    trajectory["steps"].append(
        {
            "step_id": 3,
            "source": "agent",
            "message": "Completed",
            "llm_call_count": 1,
        }
    )
    return trajectory


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_converts_single_json_and_loads_with_lmflow_dataset(tmp_path):
    input_path = tmp_path / "trajectory.json"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_path = dataset_dir / "data.json"
    _write_json(input_path, _tool_trajectory())

    converted_count = convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert converted_count == 1
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["type"] == "conversation"
    assert output["instances"][0]["conversation_id"] == "trajectory-tool"
    assert output["instances"][0]["messages"][2]["content"] == "UNICODE_OK_✓"

    dataset = Dataset(
        DatasetArguments(
            dataset_path=str(dataset_dir),
            dataset_cache_dir=str(tmp_path / "cache"),
        )
    )
    assert dataset.get_type() == "conversation"
    assert len(dataset) == 1


def test_converts_jsonl_in_physical_line_order_and_skips_blank_lines(tmp_path):
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "data.json"
    records = [
        json.dumps(_trajectory("trajectory-1", "first"), ensure_ascii=False),
        "",
        json.dumps(_tool_trajectory(), ensure_ascii=False),
    ]
    input_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    converted_count = convert_atif_file_to_conversation_dataset(input_path, output_path)

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert converted_count == 2
    assert [item["conversation_id"] for item in output["instances"]] == [
        "trajectory-1",
        "trajectory-tool",
    ]
    assert output["instances"][1]["messages"][-1]["content"] == "Completed"

    dataset = Dataset(
        DatasetArguments(
            dataset_path=str(tmp_path),
            dataset_cache_dir=str(tmp_path / "cache"),
        )
    )
    assert dataset.get_type() == "conversation"
    assert len(dataset) == 2
    loaded_instances = dataset.to_list()
    tool_messages = loaded_instances[1]["messages"]
    assert tool_messages[1]["tool_calls"][0]["function"]["arguments"] == '{"command":"printf UNICODE_OK_✓"}'
    assert tool_messages[2]["content"] == "UNICODE_OK_✓"


def test_conversion_error_reports_line_and_identity_without_publishing_partial_output(tmp_path):
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "data.json"
    output_path.write_bytes(b"existing output")
    invalid = _trajectory("trajectory-bad")
    invalid["schema_version"] = "ATIF-v1.6"
    input_path.write_text(
        json.dumps(_trajectory()) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        convert_atif_file_to_conversation_dataset(
            input_path,
            output_path,
            overwrite=True,
        )

    assert f"{input_path}:2" in str(error.value)
    assert "trajectory_id='trajectory-bad'" in str(error.value)
    assert output_path.read_bytes() == b"existing output"
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_invalid_json_reports_physical_line_without_creating_output(tmp_path):
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "data.json"
    input_path.write_text(json.dumps(_trajectory()) + "\n{\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON") as error:
        convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert f"{input_path}:2" in str(error.value)
    assert not output_path.exists()
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_invalid_utf8_reports_physical_line_without_creating_output(tmp_path):
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "data.json"
    input_path.write_bytes(json.dumps(_trajectory()).encode("utf-8") + b"\n\xff\n")

    with pytest.raises(ValueError, match="input is not valid UTF-8") as error:
        convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert f"{input_path}:2" in str(error.value)
    assert not output_path.exists()
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ('{"schema_version":"ATIF-v1.7","schema_version":"ATIF-v1.7"}', "duplicate object key"),
        ('{"schema_version":NaN}', "non-finite number"),
    ],
)
def test_rejects_non_strict_json_extensions(tmp_path, document, message):
    input_path = tmp_path / "trajectory.json"
    output_path = tmp_path / "data.json"
    input_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize("content", ["", "\n\n"])
def test_rejects_empty_jsonl(tmp_path, content):
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "data.json"
    input_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="no ATIF trajectories found"):
        convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert not output_path.exists()


def test_rejects_json_array_with_jsonl_guidance(tmp_path):
    input_path = tmp_path / "trajectories.json"
    output_path = tmp_path / "data.json"
    _write_json(input_path, [_trajectory()])

    with pytest.raises(ValueError, match="use JSONL"):
        convert_atif_file_to_conversation_dataset(input_path, output_path)


def test_existing_output_requires_overwrite(tmp_path):
    input_path = tmp_path / "trajectory.json"
    output_path = tmp_path / "data.json"
    _write_json(input_path, _trajectory())
    output_path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        convert_atif_file_to_conversation_dataset(input_path, output_path)
    assert output_path.read_text(encoding="utf-8") == "sentinel"

    assert (
        convert_atif_file_to_conversation_dataset(
            input_path,
            output_path,
            overwrite=True,
        )
        == 1
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["type"] == "conversation"


def test_dangling_output_symlink_is_not_replaced_without_overwrite(tmp_path):
    input_path = tmp_path / "trajectory.json"
    output_path = tmp_path / "data.json"
    missing_target = tmp_path / "missing.json"
    _write_json(input_path, _trajectory())
    output_path.symlink_to(missing_target)

    with pytest.raises(FileExistsError, match="already exists"):
        convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert output_path.is_symlink()
    assert output_path.readlink() == missing_target
    assert not missing_target.exists()


def test_generic_file_converter_keeps_deterministic_steps_fail_closed(tmp_path):
    input_path = tmp_path / "trajectory.json"
    output_path = tmp_path / "data.json"
    trajectory = _tool_trajectory()
    trajectory["steps"][1]["llm_call_count"] = 0
    _write_json(input_path, trajectory)

    with pytest.raises(ValueError, match="requires model_visible_tool_names"):
        convert_atif_file_to_conversation_dataset(input_path, output_path)

    assert not output_path.exists()


def test_module_cli_reports_success_and_expected_errors(tmp_path, capsys):
    input_path = tmp_path / "trajectory.json"
    output_path = tmp_path / "data.json"
    _write_json(input_path, _trajectory())

    assert main(["--input-path", str(input_path), "--output-path", str(output_path)]) == 0
    captured = capsys.readouterr()
    assert "Converted 1 ATIF trajectory" in captured.out
    assert captured.err == ""

    assert main(["--input-path", str(input_path), "--output-path", str(output_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "already exists" in captured.err
