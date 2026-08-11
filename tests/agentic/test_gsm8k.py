import pytest

from lmflow.agentic import (
    GSM8K_REWARD_TOOL,
    extract_gsm8k_answer,
    gsm8k_example_to_task,
    run_gsm8k_reward_tool,
    score_gsm8k_answer,
)


def test_converts_official_example_to_tool_task_without_exposing_the_answer_in_messages():
    task = gsm8k_example_to_task(
        {
            "question": "A box has 18 pencils and receives 7 more. How many pencils are there?",
            "answer": "There are 18 + 7 pencils. #### 25",
        },
        split="train",
        index=12,
    )

    assert task.task_id == "openai/gsm8k:train:12"
    assert [message["role"] for message in task.messages] == ["system", "user"]
    assert "25" not in repr(task.messages)
    assert task.tools == [GSM8K_REWARD_TOOL]
    assert task.environment == {
        "tools_kwargs": {
            "calc_gsm8k_reward": {
                "ground_truth": "25",
            }
        }
    }
    assert task.metadata == {
        "data_source": "openai/gsm8k",
        "split": "train",
        "index": 12,
    }


@pytest.mark.parametrize(
    ("solution", "method", "expected"),
    [
        ("Reasoning. #### 1,234", "strict", "1234"),
        ("First #### 4, then #### -3.5", "strict", "-3.5"),
        ("The result is 19.", "flexible", "19."),
        ("No numeric answer", "flexible", None),
    ],
)
def test_extracts_answers_with_the_pinned_gsm8k_rules(solution, method, expected):
    assert extract_gsm8k_answer(solution, method=method) == expected


def test_only_considers_the_last_300_characters_when_scoring():
    solution = "#### 41" + ("x" * 301)

    assert extract_gsm8k_answer(solution, method="strict") is None


def test_scores_the_last_candidate_answer():
    assert score_gsm8k_answer("Try 12, then 13", "13") == 1.0
    assert score_gsm8k_answer("Try 12, then 14", "13") == 0.0
    assert score_gsm8k_answer("#### 1,000", "1000", method="strict") == 1.0


def test_reward_tool_matches_agent_r1_feedback_shape():
    observation, details = run_gsm8k_reward_tool({"answer": "25"}, ground_truth="25")

    assert observation == "Current parsed answer='#### 25' reward=1.0"
    assert details == {
        "answer": "#### 25",
        "ground_truth": "25",
        "reward": 1.0,
    }


def test_reward_tool_accepts_a_revised_incorrect_answer_without_terminating_the_episode():
    observation, details = run_gsm8k_reward_tool({"answer": "#### 24"}, ground_truth="25")

    assert observation == "Current parsed answer='#### 24' reward=0.0"
    assert details["reward"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"example": [], "split": "train", "index": 0}, TypeError, "example must be a mapping"),
        (
            {"example": {"question": "", "answer": "#### 1"}, "split": "train", "index": 0},
            ValueError,
            "question must be a non-empty string",
        ),
        (
            {"example": {"question": "Question", "answer": "missing marker"}, "split": "train", "index": 0},
            ValueError,
            "must contain a final answer",
        ),
        (
            {"example": {"question": "Question", "answer": "#### 1"}, "split": "train", "index": True},
            TypeError,
            "index must be an integer",
        ),
    ],
)
def test_rejects_malformed_dataset_examples(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        gsm8k_example_to_task(**kwargs)


def test_reward_tool_matches_agent_r1_missing_answer_recovery():
    observation, details = run_gsm8k_reward_tool({}, ground_truth="1")

    assert observation == "Current parsed answer='#### ' reward=0.0"
    assert details["answer"] == "#### "
    assert details["reward"] == 0.0
