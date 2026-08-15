import hashlib
import json
from types import SimpleNamespace

import pytest

from lmflow.agentic.appworld_protocol import (
    APPWORLD_TINY_SCENARIOS,
    APPWORLD_TINY_TASK_IDS,
    APPWORLD_TINY_TASK_SET_SHA256,
    canonical_appworld_instance_id,
    canonical_json_sha256,
    verify_manifest_digest,
    with_manifest_digest,
)
from lmflow.agentic.scaffolds.appworld_react_code.scaffold import (
    extract_first_python_code,
    render_reference_messages,
    text_to_messages,
    verify_reference_checkout,
)


def test_tiny_task_set_is_three_complete_difficulty_groups():
    assert len(APPWORLD_TINY_TASK_IDS) == 9
    assert [scenario["difficulty"] for scenario in APPWORLD_TINY_SCENARIOS.values()] == [1, 2, 3]
    for scenario_id in APPWORLD_TINY_SCENARIOS:
        assert [task_id for task_id in APPWORLD_TINY_TASK_IDS if task_id.startswith(scenario_id)] == [
            f"{scenario_id}_1",
            f"{scenario_id}_2",
            f"{scenario_id}_3",
        ]
    assert canonical_json_sha256(list(APPWORLD_TINY_TASK_IDS)) == APPWORLD_TINY_TASK_SET_SHA256
    assert canonical_appworld_instance_id(APPWORLD_TINY_TASK_IDS[0]).endswith("/dev/396c5a2_1")


def test_manifest_digest_fails_closed_after_mutation():
    manifest = with_manifest_digest({"tasks": list(APPWORLD_TINY_TASK_IDS)})
    verify_manifest_digest(manifest)
    manifest["tasks"].append("unexpected")
    with pytest.raises(ValueError, match="does not match"):
        verify_manifest_digest(manifest)


def test_reference_code_parser_matches_first_complete_and_partial_blocks():
    code, fixed = extract_first_python_code("plan\n```python\nprint('first')\n```\n```python\nprint('second')\n```")
    assert code == "print('first')"
    assert fixed.endswith("print('first')\n```")

    code, fixed = extract_first_python_code("plan\n```python\nprint('partial')")
    assert code == "print('partial')"
    assert fixed.endswith("\n```")

    assert extract_first_python_code("no code") == ("", "no code")


def test_reference_prompt_rendering_preserves_role_messages():
    task = SimpleNamespace(
        instruction="Do the thing.",
        supervisor=SimpleNamespace(first_name="Ada", last_name="Lovelace"),
        app_descriptions={"notes": "A notes app."},
    )
    prompt = (
        "SYSTEM:\nBe precise.\n\nUSER:\n{{ main_user.first_name }}: {{ instruction }} "
        "{{ app_descriptions }}\n\nASSISTANT:\nReady."
    )
    messages = render_reference_messages(prompt, task)
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert "Ada: Do the thing." in messages[1]["content"]
    assert '"name": "notes"' in messages[1]["content"]
    assert text_to_messages("USER:\nhello") == [{"role": "user", "content": "hello"}]


def test_reference_checkout_verifies_file_digests(monkeypatch, tmp_path):
    source = tmp_path / "prompt.txt"
    source.write_text("pinned", encoding="utf-8")
    expected = hashlib.sha256(b"pinned").hexdigest()
    monkeypatch.setattr(
        "lmflow.agentic.scaffolds.appworld_react_code.scaffold._REFERENCE_FILES",
        {"prompt.txt": expected},
    )
    assert verify_reference_checkout(tmp_path) == {"prompt.txt": expected}
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_reference_checkout(tmp_path)


def test_manifest_encoding_is_canonical():
    assert canonical_json_sha256({"b": 1, "a": 2}) == hashlib.sha256(b'{"a":2,"b":1}').hexdigest()
    assert json.loads(json.dumps(with_manifest_digest({"a": 1})))
