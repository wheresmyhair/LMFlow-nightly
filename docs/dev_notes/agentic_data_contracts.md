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
| `non_tensor_batch` | `tasks` | Normalized `TaskSpec` objects before agent execution. |
| `non_tensor_batch` | `task_ids` | Source task identity for every row. |
| `non_tensor_batch` | `group_ids` | Rollout group identity for algorithms such as GRPO. |

Recipes add their own tensors, such as `old_log_probs`, `advantages`, returns,
or reference-policy outputs, and validate only the fields they consume.
Model, tokenizer, scaffold, and policy revisions can live in `meta_info` when
they are constant for the whole batch, or in `non_tensor_batch` when they vary
by row.

`DataProto` already checks batch-dimension alignment. The shared runtime does
not impose a second schema version, recursively validate metadata, or require
`loss_mask` to be binary. Persistent datasets and remote transports define
their own versioned manifests at those external boundaries.
