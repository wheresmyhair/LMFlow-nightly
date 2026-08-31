# TRL GRPO lifecycle bridge

LMFlow's sealed GRPO backend delegates the complete training lifecycle to
TRL 1.9.2 `GRPOTrainer.train()`. LMFlow supplies one complete, token-native
rollout batch through TRL's public `rollout_func` and a thin audited reward
function. TRL and Transformers continue to own gradient checkpointing, mixed
precision, gradient accumulation, optimizer and scheduler steps, Accelerate,
PEFT, callbacks, checkpoints, and model export.

The bridge is backend-internal and versioned. It does not define a public
`RolloutSource` or a general Agentic schema. The synchronous controller and
LMFlow's pure GRPO objective remain correctness references for differential
tests.

## Sealed input contract

The one-step bridge consumes an existing `DataProto` with:

| Location | Field | Meaning |
| --- | --- | --- |
| `batch` | `input_ids` | Actual prompt and sampled output token IDs, right padded. |
| `batch` | `attention_mask` | Binary active-token mask. |
| `batch` | `prompt_lengths` | Per-rollout boundary between conditioning context and the optimized completion. |
| `batch` | `loss_mask` | Binary policy/environment mask; prompt, tool observations, and padding are zero. |
| `batch` | `old_log_probs` | Actual sampled-policy token log-probabilities aligned with `input_ids`. |
| `batch` | `rewards` | One audited scalar training reward per rollout. |
| `non_tensor_batch` | `task_ids` | Stable task identity. |
| `non_tensor_batch` | `group_ids` | Complete rollout-group identity. |
| `non_tensor_batch` | `rollout_ids` | Unique rollout identity. |
| `meta_info` | `policy_version` | Policy/checkpoint version that produced the sealed batch. |
| `meta_info` | `logprob_provenance.behavior` | Source and policy version for sampled log-probabilities. |

Every group has one task identity and the same number of rollouts. Multi-turn
rollouts may have different conditioning token prefixes within a group because
each trajectory can contain a different earlier policy action and environment
observation. The group identity refers to the shared root task; each rollout's
actual conditioning prefix remains sealed and is passed to TRL unchanged.

Validation fails closed on incomplete groups, duplicate rollout IDs, stale
policy provenance, non-finite values, non-binary masks, non-contiguous padding,
invalid prompt boundaries, selected prompt/padding tokens, or a rollout with no
trainable policy token. The rollout and reward hooks are single-use so one
sealed batch cannot be consumed twice accidentally.

## Log-probability provenance

The bridge records three distinct meanings:

- `behavior` is the log-probability returned by the rollout backend for the
  token that was actually sampled. It remains available as
  `sampling_per_token_logps`.
- `trainer_old` is the denominator used by the clipped GRPO importance ratio.
  Version 1 explicitly selects `behavior` and copies it to
  `old_per_token_logps` after TRL has generated and scored the batch.
- `reference` belongs to the KL/reference-policy path. It is disabled in
  version 1 because `beta=0`.

TRL 1.9.2's public `rollout_func` preserves returned `logprobs` as
`sampling_per_token_logps`. In aligned, non-vLLM training it leaves
`old_per_token_logps` absent, and the loss otherwise falls back to recomputed
current-policy values. LMFlow therefore applies one narrow compatibility seam:
call `super()._generate_and_score_completions()` and, only when the sampled
field exists and the old field is absent, validate shape and finiteness and
inject a detached clone as `old_per_token_logps`. The sampled field is neither
deleted nor rewritten.

The package version is locked to `trl==1.9.2`. Reference tests pin the source
contract of the two private methods involved and exercise the observed output
keys and loss behavior. A TRL version mismatch, an already-present old field,
or private-output drift stops training before an optimizer step.

## Supported recipe

The first product slice intentionally supports one correctness recipe:

- one process and one optimizer step;
- one sealed generation batch with `per_device_train_batch_size=1` and gradient
  accumulation across every rollout;
- `num_iterations=1`, `loss_type="grpo"`, group reward scaling, token-level
  importance ratios, and symmetric clipping with epsilon 0.2;
- `beta=0`, no reference/KL term, no entropy term, no off-policy filtering, and
  no reward weighting;
- `use_vllm=False` inside TRL because rollout already happened in LMFlow;
- TRL's vLLM importance-sampling correction disabled;
- gradient checkpointing enabled and `use_cache=False`;
- dataset shuffling, Liger, truncated-completion masking, and multi-step reuse
  disabled.

The bridge also verifies that configured prompt/completion limits cover the
sealed token boundaries. Unsupported combinations fail during construction.
In particular, `use_vllm=True`, `beta>0`, `num_iterations>1`, multiple training
steps or epochs over the same rollout, distributed execution, asymmetric or
alternative clipping losses, and a pre-existing `old_per_token_logps` field
require separate contract evidence before they can be enabled.

## Verification boundary

The locked reference test runs the standard Trainer lifecycle with a tiny PEFT
model. It checks exact completion IDs, policy/environment masks, audited
rewards, group advantages, sampled and trainer-old log-probabilities, loss,
final accumulated gradients, one optimizer/scheduler step, active gradient
checkpointing, LoRA-only parameter changes, and frozen base parameters.

The GSM8K acceptance additionally projects the exact sealed 2-group, 16-rollout
batch into this bridge and compares the resulting TRL objective and gradients
against LMFlow's existing objective. Large-model update, adapter export, reload,
and generation-side publication remain separate runtime acceptance steps.
