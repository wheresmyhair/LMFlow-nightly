# GSM8K-tool Agentic bootstrap

LMFlow's GSM8K-tool integration follows Agent-R1's reward-tool recipe as of
commit `b124aa46534cbf2fb8bc8af11405774984c42ac7`. The model may call
`calc_gsm8k_reward` with a candidate answer, receive binary feedback, and revise
its answer before returning the final response.

`gsm8k_example_to_task()` converts one official `openai/gsm8k` row into the
shared `TaskSpec`. Ground truth is stored in environment arguments and is not
included in the initial model-visible messages. `run_gsm8k_tool_episode()` uses
the shared completion control plane and returns a linear ATIF v1.7 trajectory.
The existing ATIF adapter then produces an LMFlow conversation SFT example.

```python
from lmflow.agentic import (
    OpenAICompatibleCompletionBackend,
    atif_trajectory_to_conversation,
    gsm8k_example_to_task,
    run_gsm8k_tool_episode,
)

task = gsm8k_example_to_task(
    {
        "question": "A shelf has 18 books and receives 7 more. How many books are there?",
        "answer": "Add the quantities. #### 25",
    },
    split="train",
    index=0,
)

backend = OpenAICompatibleCompletionBackend(base_url="http://127.0.0.1:8000/v1")
trajectory = run_gsm8k_tool_episode(
    backend,
    task,
    model_name="Qwen/Qwen3-8B",
    trajectory_id="gsm8k-train-0-rollout-0",
    model_kwargs={"temperature": 0.7},
)
conversation = atif_trajectory_to_conversation(trajectory)
```

The episode records tool feedback, final reward, model-step count, reward-tool
call count, and completion cost reported by the backend. A direct final answer
without a tool call remains scoreable to match the reference recipe. Malformed
tool calls and duplicate call IDs fail closed. Reaching `max_steps` without a
final answer raises an error; batch-level failure artifacts and partial-output
policy belong to the later data-generation layer.

This runner is for data construction and evaluation through an OpenAI-compatible
control plane. It does not provide sampled token IDs or log-probabilities and
cannot serve as the token-native rollout source for online GRPO.
