import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "change_notes.py"
SPEC = importlib.util.spec_from_file_location("change_notes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
change_notes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = change_notes
SPEC.loader.exec_module(change_notes)


def _fragment(module: str = "agentic", title: str = "Agent loop") -> dict[str, object]:
    return {
        "title": title,
        "module": module,
        "kind": "feature",
        "upstream": "candidate",
        "summary": "Add a reviewable agent loop contract.",
        "review": ["Data contract", "Failure behavior"],
        "breaking": False,
    }


class ChangeNotesTest(unittest.TestCase):
    def _write(self, root: Path, module: str, name: str, data: dict[str, object]) -> Path:
        target = root / module / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")
        return target

    def test_loads_valid_fragments_and_ignores_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write(root, "agentic", "agent-loop.json", _fragment())
            (root / "_template.json").write_text("{}", encoding="utf-8")

            changes = change_notes.load_fragments(root)

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].source, "agentic/agent-loop.json")

    def test_rejects_module_directory_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = self._write(root, "core", "agent-loop.json", _fragment(module="agentic"))

            with self.assertRaisesRegex(change_notes.FragmentError, "must match directory"):
                change_notes.load_fragments(root)

        self.assertEqual(path.parent.name, "core")

    def test_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = _fragment()
            data["surprise"] = True
            self._write(root, "agentic", "agent-loop.json", data)

            with self.assertRaisesRegex(change_notes.FragmentError, "unknown fields"):
                change_notes.load_fragments(root)

    def test_render_groups_modules_and_summarizes_export_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write(root, "agentic", "agent-loop.json", _fragment())
            private_data = _fragment(module="ci", title="Private trigger")
            private_data["upstream"] = "private"
            self._write(root, "ci", "private-trigger.json", private_data)
            changes = change_notes.load_fragments(root)

        notes = change_notes.render_notes(changes, "2026.08")
        self.assertIn("# LMFlow Nightly 2026.08 — What's New", notes)
        self.assertIn("- Candidate: 1", notes)
        self.assertIn("- Private: 1", notes)
        self.assertLess(notes.index("## Agentic"), notes.index("## CI/CD"))
        self.assertIn("Fragment: `ci/private-trigger.json`", notes)

    def test_refuses_to_build_empty_notes(self):
        with self.assertRaisesRegex(change_notes.FragmentError, "without change fragments"):
            change_notes.render_notes([], "2026.08")


if __name__ == "__main__":
    unittest.main()
