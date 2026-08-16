"""Narrow adapter for AppWorld's official Simplified ReAct Code scaffold."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import Template

APPWORLD_REACT_CODE_SCAFFOLD = {
    "id": "appworld.simplified-react-code-agent",
    "repository": "https://github.com/StonyBrookNLP/appworld.git",
    "revision": "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
    "version": "0.2.0.dev0",
    "prompt_path": "experiments/prompts/react_code_agent/instructions.txt",
    "prompt_sha256": "c41f7852217d46586047a38f68e88b827cb9dcbe624e9651f9db8301547a534b",
    "agent_source_path": "experiments/code/simplified/agent.py",
    "agent_source_sha256": "e48934f66a23d00e32babbf8759343d21c277cc722e874b1d25e2a9bfc8fe017",
    "react_source_path": "experiments/code/simplified/react_code_agent.py",
    "react_source_sha256": "4b80c27e61b5d04859c447fa47e525025f2c10044a8799ffad650f42fbfc5963",
    "configuration": {
        "ignore_multiple_calls": True,
        "max_prompt_length": None,
        "max_output_length": None,
        "max_steps": 50,
        "random_seed": 100,
    },
    "qwen3_8b_without_reasoning": {
        "temperature": 0,
        "seed": 100,
        "max_completion_tokens": 3000,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}

_REFERENCE_FILES = {
    APPWORLD_REACT_CODE_SCAFFOLD["prompt_path"]: APPWORLD_REACT_CODE_SCAFFOLD["prompt_sha256"],
    APPWORLD_REACT_CODE_SCAFFOLD["agent_source_path"]: APPWORLD_REACT_CODE_SCAFFOLD["agent_source_sha256"],
    APPWORLD_REACT_CODE_SCAFFOLD["react_source_path"]: APPWORLD_REACT_CODE_SCAFFOLD["react_source_sha256"],
}
_FULL_CODE_PATTERN = re.compile(r"```python\n(.*?)```", flags=re.DOTALL)
_PARTIAL_CODE_PATTERN = re.compile(r".*```python\n(.*)", flags=re.DOTALL)
_ROLE_PATTERN = re.compile(r"(USER|ASSISTANT|SYSTEM):\n", flags=re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_checkout(source_root: str | Path) -> dict[str, str]:
    """Verify the public scaffold files required from the pinned checkout."""

    root = Path(source_root).expanduser().resolve()
    verified = {}
    for relative_path, expected_sha256 in _REFERENCE_FILES.items():
        source_path = root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"missing AppWorld reference scaffold file: {source_path}")
        actual_sha256 = _sha256_file(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"AppWorld reference scaffold digest mismatch for {relative_path}: {actual_sha256}")
        verified[relative_path] = actual_sha256
    return verified


def load_reference_prompt(source_root: str | Path) -> str:
    """Load the exact official prompt after checking its pinned source files."""

    verify_reference_checkout(source_root)
    prompt_path = Path(source_root).expanduser().resolve() / APPWORLD_REACT_CODE_SCAFFOLD["prompt_path"]
    return prompt_path.read_text(encoding="utf-8").lstrip()


def text_to_messages(input_text: str) -> list[dict[str, str]]:
    """Parse the role-delimited prompt exactly as the reference agent does."""

    if not isinstance(input_text, str) or not input_text:
        raise ValueError("input_text must be a non-empty string")
    messages: list[dict[str, str]] = []
    last_start = 0
    for match in _ROLE_PATTERN.finditer(input_text):
        last_end = match.start()
        if not messages:
            if last_end != 0:
                raise ValueError(f"start of prompt has no assigned role: {input_text[:last_end]}")
        else:
            messages[-1]["content"] = input_text[last_start:last_end]
        messages.append({"role": match.group(1).lower(), "content": ""})
        last_start = match.end()
    if not messages:
        raise ValueError("prompt contains no USER, ASSISTANT, or SYSTEM role marker")
    messages[-1]["content"] = input_text[last_start:]
    return messages


def render_reference_messages(prompt: str, task: Any) -> list[dict[str, str]]:
    """Render the official prompt using only model-visible AppWorld task data."""

    app_descriptions = json.dumps(
        [{"name": name, "description": description} for name, description in task.app_descriptions.items()],
        indent=1,
    )
    rendered = Template(prompt).render(
        instruction=task.instruction,
        main_user=task.supervisor,
        app_descriptions=app_descriptions,
    )
    return text_to_messages(rendered + "\n\n")


def extract_first_python_code(text: str) -> tuple[str, str]:
    """Return the first complete Python block, matching the official parser."""

    if not isinstance(text, str):
        raise TypeError("completion content must be a string")
    for match in _FULL_CODE_PATTERN.finditer(text):
        return match.group(1).strip(), text[: match.end()]
    partial_match = _PARTIAL_CODE_PATTERN.match(text)
    if partial_match:
        code = partial_match.group(1).strip()
        fixed_text = text if text.endswith("\n") else text + "\n"
        return code, fixed_text + "```"
    return "", text


def scaffold_identity(source_root: str | Path) -> dict[str, Any]:
    """Return a machine-path-free identity after verifying the checkout."""

    verified = verify_reference_checkout(source_root)
    return {
        **json.loads(json.dumps(APPWORLD_REACT_CODE_SCAFFOLD)),
        "verified_source_files": verified,
        "adapter": "lmflow.agentic.scaffolds.appworld_react_code",
    }


def qwen3_reference_model_kwargs(*, enable_thinking: bool = False) -> Mapping[str, Any]:
    """Return AppWorld's Qwen3 profile with an explicit thinking-mode choice."""

    if not isinstance(enable_thinking, bool):
        raise TypeError("enable_thinking must be a bool")
    model_kwargs = json.loads(json.dumps(APPWORLD_REACT_CODE_SCAFFOLD["qwen3_8b_without_reasoning"]))
    model_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] = enable_thinking
    return model_kwargs


__all__ = [
    "APPWORLD_REACT_CODE_SCAFFOLD",
    "extract_first_python_code",
    "load_reference_prompt",
    "qwen3_reference_model_kwargs",
    "render_reference_messages",
    "scaffold_identity",
    "text_to_messages",
    "verify_reference_checkout",
]
