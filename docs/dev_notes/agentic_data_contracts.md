# Agentic runtime data contracts

LMFlow keeps only one dedicated Agentic Python type: `TaskSpec`, the normalized
input passed from a dataset adapter to an agent. Dataset adapters own any
source-specific parsing and validation.

Rollout, reward, and training components exchange the existing
`lmflow.utils.protocol.DataProto` directly. Agentic code does not wrap it in a
parallel result or batch container.

## Task batching

`build_task_batch()` converts an iterable of normalized `TaskSpec` objects into
a `DataProto` for the dataset-adapter-to-agent handoff. It stores one-dimensional
object arrays in `non_tensor_batch`:

- `tasks`: the original `TaskSpec` objects;
- `task_ids`: their source identities.

The helper deliberately does not copy or recursively validate task payloads,
tokenize prompts, or construct rollout/training tensors. An agent consumes the
task objects, and downstream components keep `task_ids` when producing their
own token-native `DataProto` batches.

## Runtime field conventions

The first training loop uses these `DataProto` fields:

| Location | Field | Meaning |
| --- | --- | --- |
| `batch` | `input_ids` | Padded prompt, policy, and environment token ids. |
| `batch` | `attention_mask` | Tokens visible to the model. |
| `batch` | `loss_mask` | Zero excludes a token from the objective; positive values select or weight it. |
| `batch` | `rewards` | Sequence-level rollout rewards. |
| `batch` | `advantages` | Sequence-level or token-level algorithm advantages. |
| `batch` | `old_log_probs` | Optional rollout-policy log-probs aligned with `input_ids`. |
| `non_tensor_batch` | `tasks` | Normalized `TaskSpec` objects before agent execution. |
| `non_tensor_batch` | `task_ids` | Source task identity for every row. |
| `non_tensor_batch` | `group_ids` | Rollout group identity for algorithms such as GRPO. |
| `non_tensor_batch` | `rollout_ids` | Logical rollout request identity used for idempotent group assembly. |

Recipes populate algorithm fields such as `advantages` and `old_log_probs`,
may add returns or reference-policy outputs, and validate only the fields they
consume.
Model, tokenizer, scaffold, and policy revisions can live in `meta_info` when
they are constant for the whole batch, or in `non_tensor_batch` when they vary
by row.

The controller-local `RolloutGroupAssembler` additionally requires a
batch-constant `meta_info["policy_version"]`. Its group and rollout identities
are attempt-scoped; whole-group retries register new identities so late results
cannot contaminate a later update.

`DataProto` already checks batch-dimension alignment. The shared runtime does
not impose a second schema version, recursively validate metadata, or require
`loss_mask` to be binary. Persistent datasets and remote transports define
their own versioned manifests at those external boundaries.

For causal language-model updates, tensors aligned with `input_ids` use shape
`[batch, sequence]`. Column zero of `old_log_probs` is zero because the first
token has no preceding model logit, and `loss_mask[:, 0]` must also be zero.
Logits at column `t` score the target token at `input_ids[:, t + 1]`; prompt,
environment, and padding tokens remain present for context while their
`loss_mask` entries stay zero.
