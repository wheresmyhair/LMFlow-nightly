"""GSM8K-tool task and reward helpers for Agentic smoke runs."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Literal

from lmflow.agentic.contracts import TaskSpec

GSM8K_DATA_SOURCE = "openai/gsm8k"
GSM8K_AGENT_R1_REFERENCE_COMMIT = "b124aa46534cbf2fb8bc8af11405774984c42ac7"
GSM8K_REWARD_TOOL_NAME = "calc_gsm8k_reward"

GSM8K_AGENT_SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step. "
    "Before giving the final answer, call the `calc_gsm8k_reward` tool at least once with your answer. "
    "Use the tool feedback to refine the answer if needed. "
    "Put the final answer in the format `#### <answer>`."
)
GSM8K_AGENT_USER_PROMPT = "{question}\n\nThink step by step and use the reward tool before the final answer."

GSM8K_REWARD_TOOL = {
    "type": "function",
    "function": {
        "name": GSM8K_REWARD_TOOL_NAME,
        "description": "A tool for calculating the reward of GSM8K answers.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the question.",
                }
            },
            "required": ["answer"],
        },
    },
}

_SOLUTION_CLIP_CHARS = 300
_STRICT_ANSWER_PATTERN = re.compile(r"#### (-?[0-9.,]+)")
_FLEXIBLE_ANSWER_PATTERN = re.compile(r"(-?[0-9.,]+)")


def extract_gsm8k_answer(
    solution: str,
    *,
    method: Literal["strict", "flexible"] = "strict",
) -> str | None:
    """Extract the last numeric answer using the Agent-R1/veRL GSM8K rules."""

    if not isinstance(solution, str):
        raise TypeError("solution must be a string")
    if method not in {"strict", "flexible"}:
        raise ValueError("method must be 'strict' or 'flexible'")

    clipped = solution[-_SOLUTION_CLIP_CHARS:]
    if method == "strict":
        matches = _STRICT_ANSWER_PATTERN.findall(clipped)
        return matches[-1].replace(",", "") if matches else None

    for candidate in reversed(_FLEXIBLE_ANSWER_PATTERN.findall(clipped)):
        if candidate not in {"", "."}:
            return candidate
    return None


def score_gsm8k_answer(
    solution: str,
    ground_truth: str,
    *,
    method: Literal["strict", "flexible"] = "flexible",
) -> float:
    """Return the binary GSM8K score used by the Agent-R1 recipe."""

    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError("ground_truth must be a non-empty string")
    return float(extract_gsm8k_answer(solution, method=method) == ground_truth)


def gsm8k_example_to_task(
    example: Mapping[str, Any],
    *,
    split: str,
    index: int,
    data_source: str = GSM8K_DATA_SOURCE,
) -> TaskSpec:
    """Convert one official GSM8K row into the normalized tool task contract."""

    if not isinstance(example, Mapping):
        raise TypeError("example must be a mapping")
    question = example.get("question")
    answer = example.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("example.question must be a non-empty string")
    if not isinstance(answer, str):
        raise TypeError("example.answer must be a string")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if index < 0:
        raise ValueError("index must be non-negative")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a non-empty string")

    ground_truth = extract_gsm8k_answer(answer, method="strict")
    if ground_truth is None:
        raise ValueError("example.answer must contain a final answer in '#### <answer>' format")

    return TaskSpec(
        task_id=f"{data_source}:{split}:{index}",
        messages=[
            {"role": "system", "content": GSM8K_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": GSM8K_AGENT_USER_PROMPT.format(question=question)},
        ],
        tools=[copy.deepcopy(GSM8K_REWARD_TOOL)],
        environment={
            "tools_kwargs": {
                GSM8K_REWARD_TOOL_NAME: {
                    "ground_truth": ground_truth,
                }
            }
        },
        metadata={
            "data_source": data_source,
            "split": split,
            "index": index,
        },
    )


def run_gsm8k_reward_tool(
    arguments: Mapping[str, Any],
    *,
    ground_truth: str,
) -> tuple[str, dict[str, Any]]:
    """Execute one reward-tool call and return its observation and audit data."""

    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    answer = arguments.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)
    if not answer.startswith("#### "):
        answer = f"#### {answer}"

    reward = score_gsm8k_answer(answer, ground_truth, method="flexible")
    details = {
        "answer": answer,
        "ground_truth": ground_truth,
        "reward": reward,
    }
    return f"Current parsed answer={answer!r} reward={reward!r}", details


__all__ = [
    "GSM8K_AGENT_SYSTEM_PROMPT",
    "GSM8K_AGENT_USER_PROMPT",
    "GSM8K_AGENT_R1_REFERENCE_COMMIT",
    "GSM8K_DATA_SOURCE",
    "GSM8K_REWARD_TOOL",
    "GSM8K_REWARD_TOOL_NAME",
    "extract_gsm8k_answer",
    "gsm8k_example_to_task",
    "run_gsm8k_reward_tool",
    "score_gsm8k_answer",
]
