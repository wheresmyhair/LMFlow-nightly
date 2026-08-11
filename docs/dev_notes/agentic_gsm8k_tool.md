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
    generate_gsm8k_tool_dataset,
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

report = generate_gsm8k_tool_dataset(
    backend,
    [task],
    artifact_dir="gsm8k-tool-run",
    model_name="Qwen/Qwen3-8B",
    session_id="gsm8k-tool-run",
    rollouts_per_task=4,
    model_kwargs={"temperature": 0.7},
)
```

The episode records tool feedback, final reward, model-step count, reward-tool
call count, and completion cost reported by the backend. A direct final answer
without a tool call remains scoreable to match the reference recipe. Malformed
tool calls and duplicate call IDs fail closed. Reaching `max_steps` without a
final answer raises an error.

The batch generator runs tasks and rollout indices in deterministic order. It
publishes one artifact directory only after all episodes finish successfully:

```text
gsm8k-tool-run/
  trajectories.jsonl
  report.json
  dataset/
    data.json
```

`trajectories.jsonl` retains every completed trajectory, including incorrect
answers. Only reward-1 trajectories are selected into the conversation dataset
for cold-start SFT. Point LMFlow's Dataset loader at the nested `dataset/`
directory so `report.json` is never interpreted as a training shard. The report
contains task, rollout, reward, model-step, tool-call, and provider-reported cost
aggregates. If an episode or conversion fails, the staging directory is removed
and no partial artifact is published. Concurrent generation, retry/resume, and
partial-success publication remain outside this synchronous baseline.

For a bounded run against an OpenAI-compatible endpoint, use the packaged CLI:

```bash
python -m lmflow.agentic.generate_gsm8k_dataset \
  --artifact-dir gsm8k-tool-run \
  --base-url http://127.0.0.1:8000/v1 \
  --model-name Qwen/Qwen3-8B \
  --session-id gsm8k-tool-run \
  --split train \
  --start-index 0 \
  --limit 16 \
  --rollouts-per-task 4
```

The required `--limit` makes the number of source tasks and resulting upper
bound on completion calls explicit before a paid run. The default source is
`openai/gsm8k` with configuration `main`; `--input-path rows.jsonl` selects a
local JSON/JSONL export. API credentials are read from `OPENAI_API_KEY` by
default, or from the environment variable named by `--api-key-env`; credentials
are never accepted as command-line values. Dataset rows are selected as the
exact contiguous interval `[start-index, start-index + limit)`, preserving the
original row indices in task and trajectory identities.

This runner is for data construction and evaluation through an OpenAI-compatible
control plane. It does not provide sampled token IDs or log-probabilities and
cannot serve as the token-native rollout source for online GRPO.

## Qwen3-8B cold-start SFT

The generated `dataset/` directory is directly consumable by LMFlow's existing
Finetuner. A single-GPU LoRA run can start with:

```bash
accelerate launch --config_file configs/accelerate_singlegpu_config.yaml \
  examples/finetune.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path gsm8k-tool-run/dataset \
  --output_dir output_models/qwen3-8b-gsm8k-tool-lora \
  --conversation_template qwen3 \
  --train_on_prompt false \
  --disable_group_texts true \
  --use_lora true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --block_size 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --bf16 true \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --use_flash_attention false \
  --validation_split_percentage 0 \
  --logging_steps 1 \
  --save_steps 100 \
  --dataloader_num_workers 2 \
  --report_to none \
  --do_train true \
  --seed 42
```

The explicit Qwen3 template keeps system, tool-call, tool-observation, and final
answer structure aligned with data construction. `train_on_prompt=false` trains
only assistant spans, including both the reward-tool action and the answer that
follows its feedback. The LoRA target set covers attention and MLP projections;
reduce rank or block size first when a development GPU has insufficient memory.
Large sharded checkpoints can also exceed a low process file-descriptor limit.
Check `ulimit -n` before launch; for example, WSL users can prefix the command
with `prlimit --nofile=65535:65535 --` when the configured hard limit permits.

For LISA, use distributed FSDP2 and replace the LoRA flags with LISA controls:

```bash
accelerate launch \
  --config_file configs/accelerate_fsdp2_config.yaml \
  --num_processes 2 \
  examples/finetune.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path gsm8k-tool-run/dataset \
  --output_dir output_models/qwen3-8b-gsm8k-tool-lisa \
  --conversation_template qwen3 \
  --train_on_prompt false \
  --disable_group_texts true \
  --use_lisa true \
  --lisa_activated_layers 1 \
  --lisa_interval_steps 20 \
  --lisa_layers_attribute model.model.layers \
  --block_size 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_steps 10 \
  --bf16 true \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --use_flash_attention false \
  --validation_split_percentage 0 \
  --logging_steps 1 \
  --save_steps 100 \
  --dataloader_num_workers 2 \
  --report_to none \
  --do_train true \
  --seed 42
```

The checked-in FSDP2 profile is a two-GPU, single-node starting point. Override
the Accelerate topology for the actual node and GPU count. Its full-state-dict
export favors a reloadable final artifact at the current 8B scale. FSDP LoRA
runs export the adapter through Transformers Trainer and should merge it with the
base model later in a non-sharded process. Larger models may require a separate
adapter-only or distributed-checkpoint export strategy to avoid rank-zero CPU
memory pressure. Treat the hyperparameters above as a reproducible bootstrap
recipe and select production values on a fixed development split rather than
from this smoke configuration.
