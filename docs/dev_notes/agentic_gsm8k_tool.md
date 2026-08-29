# GSM8K-tool Agentic bootstrap

LMFlow's GSM8K-tool integration follows Agent-R1's reward-tool recipe as of
commit `b124aa46534cbf2fb8bc8af11405774984c42ac7`. The model may call
`calc_gsm8k_reward` with a candidate answer, receive binary feedback, and revise
its answer before returning the final response.

This reward-tool path remains useful for reproducing the upstream behavior, but
it is not the formal calculator cold-start source below. Its binary observation
is derived from the gold answer, while the calculator protocol exposes only the
result of executing a model-supplied arithmetic expression.

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

## Pinned direct/calculator evaluation protocol

The decision baseline uses the official `openai/gsm8k` `main` configuration at
revision `740312add88f781978c0658806c59bc2815b9866`. It assigns every row a
canonical instance identity containing the dataset revision, source split, and
original source index. Subsetting and shuffling therefore do not change task
identity. Each dataset manifest contains the ordered identities, source indices,
row-content hashes, source-content hash, current LMFlow Dataset fingerprint, and
a digest over the complete manifest. It contains no question, solution, or gold
answer text.

The protocol reserves 512 deterministically hash-ranked official training rows
for `development`; `training` is the other 6,961 rows in source order. The
`development128` view is the first 128 rows of that same ranked development
view and supports paired Base/SFT checkpoint and recipe iteration. It remains
disjoint from every training row. The official 1,319-row test split is held out
from data construction. Its bounded evaluation views are nested: `smoke` (16
rows), `repeat` (128 rows), and `decision` (512 rows). `heldout` evaluates all
1,319 test rows in canonical source order. The split seed and selection
algorithm are part of the dataset manifest.

Run both the direct-answer and arithmetic-calculator conditions against an
already running OpenAI-compatible vLLM endpoint:

```bash
python -m lmflow.agentic.evaluate_gsm8k run \
  --artifact-dir artifacts/gsm8k-base-smoke-seed0 \
  --run-id gsm8k-base-smoke-seed0 \
  --base-url http://127.0.0.1:8000/v1 \
  --served-model-name Qwen/Qwen3-8B \
  --tokenizer-path /path/to/pinned/Qwen3-8B \
  --backend-version 0.25.1 \
  --served-max-model-len 16384 \
  --served-dtype bfloat16 \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --served-model-runner v1 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --split smoke \
  --sampling-seed 0 \
  --max-concurrency 2
```

The pinned Qwen3-8B chat template serializes calls as a JSON object inside
`<tool_call>...</tool_call>`, which matches vLLM's `hermes` parser. The
`qwen3_xml` parser targets the named `<function>/<parameter>` protocol and does
not match this model revision's template or the SFT serialization. In
particular, forced structured decoding with that parser can preserve JSON
string quotes inside a parameter value and turn `2 + 3` into the invalid
calculator expression `"2 + 3"`.

The fixed primary behavior is Qwen3 thinking enabled, temperature 0.6,
top-p 0.95, top-k 20, and min-p 0. The direct profile permits one model call;
the calculator profile permits up to four model and calculator calls. Both use
the same hidden verifier, and the calculator exposes arithmetic results only.
The reward tool and gold correctness feedback are absent from this held-out
protocol. The endpoint URL and local tokenizer path are used at runtime but are
not persisted; the report retains the served model name, model and tokenizer
revisions, tokenizer/config/chat-template hashes, backend identity and version,
provider behavior, prompts, tool schema, budgets, and sampling seed. Supply
`--model-artifact-sha256` when evaluating an unpublished SFT or GRPO adapter so
its identity does not depend on a machine path.

Each successful run is published atomically:

```text
gsm8k-base-smoke-seed0/
  dataset_manifest.json
  run_manifest.json
  report.json
  profiles/
    direct-answer/
      result.json
      records/*.json
    calculator-tool/
      result.json
      records/*.json
```

`report.json` treats failed or timed-out samples as unsuccessful for end-to-end
pass and binary metric rates. The pinned baseline selects the v2 verifier:
`final_correctness` and `strict_correctness` use
decimal numeric equivalence, so representations such as `2`, `2.0`, and `2.00`
agree; `reference_exact_correctness` and `strict_exact_correctness` retain the
Agent-R1 string-equality view for compatibility. The existing v1 recipe factory
default retains Agent-R1 exact scoring for callers that need historical
behavior; the run manifest and recipe provenance record the selected rule. The
report also includes completed-only metric means,
Wilson 95% intervals, failure categories, usage totals, latency p50/p95, and a
paired bootstrap interval for calculator-minus-direct correctness. Aggregate
repeat variation only across compatible run directories:

```bash
python -m lmflow.agentic.evaluate_gsm8k summarize \
  artifacts/gsm8k-base-repeat-seed1 \
  artifacts/gsm8k-base-repeat-seed2 \
  --output artifacts/gsm8k-base-repeat-summary.json
```

The generic Evaluator currently records runner implementation and runtime but
does not retain provider-specific runner/backend behavior such as vLLM `top_k`,
`min_p`, thinking mode, or served backend version. The GSM8K run manifest is the
authoritative provenance record for these fields until the Evaluator gains a
narrow optional runner-provenance mapping. Cross-run confidence intervals and
latency quantiles remain in this benchmark report layer; broader report
abstractions should wait for evidence from another benchmark.

Official GSM8K is common in model pretraining data, so a strong held-out score
does not by itself establish uncontaminated mathematical generalization. Use the
baseline as a capability-headroom and system-correctness gate. If direct
correctness is already at least 95%, calculator-minus-direct is below two
percentage points, and fewer than 5% of cases expose a trainable tool/recovery
gap, keep GSM8K to bounded SFT/GRPO correctness smokes and regression coverage.

## Verified public-annotation cold-start data

The first calculator-protocol SFT control uses only the pinned 6,961-row
`training` partition. It mechanically replaces public GSM8K
`<<expression=result>>` annotations with `calculate(expression)` calls. Every
observation is recomputed by the same restricted calculator used during
evaluation. A row enters class A only when every recomputed result matches the
public annotation and the projected final answer passes the strict numeric
verifier. Rows with no annotation, more than four calls, malformed arithmetic,
or any replay mismatch are excluded.

Each admitted task produces a calculator conversation and a paired direct
conversation. This preserves a direct-answer retention control without using
additional tasks. Gold answers, verifier inputs, and reward values are absent
from both model-visible messages and portable provenance. The replay file keeps
only audit outcomes, expressions, freshly computed observations, stable source
identity, and content digests.

Build the E0 replay/mask/leakage smoke with eight tasks and sixteen paired
conversations:

```bash
python -m lmflow.agentic.prepare_gsm8k_cold_start \
  --artifact-dir artifacts/gsm8k-cold-start-e0 \
  --run-id gsm8k-cold-start-e0 \
  --task-count 8 \
  --tokenizer-path /path/to/pinned/Qwen3-8B \
  --block-size 2048
```

The deterministic selector ranks stable source identities by SHA-256 and draws
round-robin from rows containing one through four calculator annotations. The
eight-task E0 therefore contains two examples at each call count and is a
prefix of a larger run with the same seed. Use `--task-count 32` for the initial
32 calculator + 32 paired-direct SFT-64 dataset.

The factory fails closed if source identities or row digests do not match the
pinned source manifest, if an assistant loss mask is missing, or if any
conversation would be truncated. It uses LMFlow's actual generation-aware
Qwen3 training template for token and mask accounting, and records both the
local tokenizer-file identity and LMFlow template digest without persisting a
machine-local path. Publication is atomic:

```text
gsm8k-cold-start-e0/
  artifact_manifest.json
  data_manifest.json
  source_dataset_manifest.json
  replay.jsonl
  report.json
  dataset/
    data.json
```

Keep these generated artifacts outside the product repository. The nested
`dataset/` directory is directly loadable by LMFlow and is the only path passed
to the Finetuner.

## Qwen3-8B cold-start SFT

The generated `dataset/` directory is directly consumable by LMFlow's existing
Finetuner. A single-GPU LoRA run can start with:

```bash
accelerate launch --config_file configs/accelerate_singlegpu_config.yaml \
  examples/finetune.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path artifacts/gsm8k-cold-start-sft64/dataset \
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

The explicit Qwen3 template keeps system, calculator-call, tool-observation, and
final-answer structure aligned with data construction. `train_on_prompt=false`
trains only assistant spans, including calculator actions and the reasoning that
follows observations. The LoRA target set covers attention and MLP projections;
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

## Token-native synchronous GRPO readiness

The synchronous GRPO reference consumes the existing `DataProto` fields
`input_ids`, `attention_mask`, `loss_mask`, and `old_log_probs`. The GSM8K
rollout adapter requests vLLM's provider-specific `return_token_ids` extension
together with sampled-token log-probabilities. It requires every log-prob token
label to use vLLM's `token_id:<id>` form. For vLLM 0.25.1 Chat Completions,
the returned ID must equal `chatcmpl-` plus the submitted request ID; both
identities are retained in rollout provenance. Decoded token text is never
accepted as sampled-token identity.

For a multi-turn calculator episode, each later vLLM prompt must preserve the
complete prior prompt and sampled output as an exact token prefix. The adapter
then appends only the new environment/tool suffix with `loss_mask=0`, followed
by the next sampled output with `loss_mask=1`. Any prefix drift caused by
structured-message re-rendering fails closed instead of silently re-tokenizing
the conversation.

The first calculator action is forced with a named provider `tool_choice` for
this bounded engineering acceptance. Its grammar-constrained output remains in
the exact training context but is excluded from policy loss. Later unforced
policy outputs are trainable. The current TRL correctness reference requires an
untruncated sampling distribution (`temperature=1`, `top_p=1`, and no
`top_k`, `min_p`, repetition, presence, frequency, or logit-bias transform) so
vLLM old log-probs and trainer policy log-probs have the same meaning.

`gsm8k_grpo_task_from_row()` returns the calculator `TaskSpec` and hidden gold
answer separately. The task and rollout provenance contain canonical source
identity, model/tokenizer/checkpoint versions, sampling seeds, request IDs,
finish reasons, token spans, and calculator execution counts, but no gold
answer or verifier reward. `gsm8k_correctness_rewards()` computes the quality
view, while `gsm8k_protocol_rewards()` separately requires correctness and at
least one successfully executed calculator call. There is no additive tool
bonus. Per-rollout records also expose model call counts, exact input/output
token totals, elapsed model time, and provider-reported cost for operational
diagnosis.

The readiness gate uses K=8 groups from pinned training tasks that do not occur
in the SFT data or development views. `summarize_gsm8k_group_variance()` reports
mixed reward groups and separately identifies mixed groups whose every rollout
has at least one unforced trainable token. A real optimizer update may consume
only the latter. If no such group exists, the GSM8K GRPO experiment exits
without expanding data or training budget.
