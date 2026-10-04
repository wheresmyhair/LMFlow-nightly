# Synchronous GRPO with the native TRL lifecycle

`GRPOPipeline` keeps the LMFlow `model = pipeline(model, dataset)` interface and
uses one TRL `GRPOTrainer.train()` call for the whole run. TRL owns accumulation,
gradient checkpointing, optimizer, scheduler and checkpoints. A fresh rollout
batch is requested after each optimizer update. No training loop is reimplemented.

For a real multi-step environment producer, see [AppWorld GRPO](agentic_appworld_grpo.md)
and `examples/appworld_grpo.py`. It reuses this lifecycle with fresh reset/replay,
official scalar rewards and per-call token evidence; serving remains caller-owned.

Run the offline CPU example in the Agentic environment:

```bash
PYTHONPATH=src python examples/grpo_tiny.py
```

The example creates a tiny local model, loads it through LMFlow `AutoModel`, and
uses an ordinary text Dataset. Its two-call toy rollout records sampled IDs and
generation scores, with a zero-loss environment observation between calls. Two
updates share one optimizer and scheduler. The toy reward is an engineering
demonstration, not a benchmark quality result. No download or GPU is required.

## Replace only the environment-specific pieces

```python
pipeline = GRPOPipeline(
    grpo_config,
    task_adapter=tasks_from_dataset,
    rollout_fn=rollout,
    reward_fn=reward,
    publish_policy=publish,
    policy_prefix="my-run",
)
model = pipeline(model, dataset)
pipeline.trainer.save_model(output_dir)
```

- `task_adapter(dataset)` returns existing `TaskSpec` values with unique IDs.
- `publish_policy(trainer, version)` publishes the current weights. Its receipt
  contains `policy_version`, `global_step`, `weight_digest` and `source`.
- `rollout_fn(requests, trainer)` samples under that publication and returns the
  existing sealed `DataProto` contract. The request contains tasks, task/group/
  rollout IDs and the publication receipt. Returned IDs must remain in request
  order, and `meta_info['policy_publication']` must confirm the loaded receipt.
- Optional `reward_fn(rollouts)` returns one reward per trajectory. Otherwise the
  returned DataProto already contains audited rewards. Reward aggregation and
  advantages remain native TRL operations.

Use the existing [sealed input fields](dev_notes/agentic_trl_grpo_lifecycle.md#sealed-input-contract).
`input_ids`, `attention_mask`,
`loss_mask` and `old_log_probs` have shape `[batch, sequence]`; `prompt_lengths`
and `rewards` have shape `[batch]`. Input rows use contiguous right padding,
with binary attention and optimization masks and zero loss on padding. Preserve
ordered task/group/rollout identities and current policy/publication/behavior
provenance. The example's `prompt_lengths=2` describes its fixed two-token toy
prompt. Real producers derive each row's boundary from the actual training-anchor
prompt IDs and pad according to this contract, retaining sampled IDs unchanged.

An in-process sampler uses the live Trainer model. An external server must load
the current published weights before generating and return verifiable load
evidence. Merely incrementing a policy string does not establish fresh sampling.
Caller code must preserve raw provider/environment evidence before validating it.

The backend-internal bridge preserves actual prompt/completion IDs, actual sampled
behavior log-probs and the optimization mask. Environment tokens remain in the
conditioning sequence with zero loss. Original per-call artifacts retain token
origin when model tokens are intentionally excluded from optimization. A flattened
trajectory must satisfy the token-native anchor/prefix rules of its producer;
re-tokenizing rendered text does not recover sampled token identity.

`max_completion_length` must cover the largest `active_length - prompt_lengths[row]`
in the sealed batch: the entire flattened suffix after the training-anchor prompt,
including model tokens and intervening environment/transport tokens even where
loss is zero. Per-call generation `max_tokens` and policy-only selected-token counts
are separate quantities. Preserve the producer's anchor/prefix rules; reject an
oversized or incompatible sequence instead of truncating its suffix to fit the cap.

A recorded completion may have no selected optimization tokens. Keep that member
and its reward in the complete group: it participates in native reward statistics
and advantages while contributing zero policy loss. The batch must contain at
least one selected completion token. Constant group rewards are valid and can
produce a zero-gradient step; the native step counter alone is not evidence of a
meaningful parameter update. Callers define which recorded terminal outcomes their
reward recipe accepts and must distinguish model failures from missing or corrupt
rollout evidence. Excluding forced model actions from loss also excludes direct
learning on those actions.

## Current support boundary

The bridge is locked to TRL 1.9.2. It uses the public, upstream-experimental
`rollout_func` and one version-locked compatibility seam after native generation/
scoring to select behavior log-probs as trainer-old log-probs. Sampling fields are
retained separately; KL/reference is disabled (`beta=0`). It does not override
input preparation, training steps or the Trainer loop.

Supported: single process, microbatch1, complete equal-sized task groups, one
optimization iteration per fresh batch, token-level GRPO clipping0.2,
group-scaled reward and temperature1. Generation batch size and accumulation must
both equal `number_of_tasks * num_generations`. The same task set may be sampled
again under each new policy; the previous trajectories may not be replayed.

Unsupported configurations are rejected: distributed training, reference/KL,
multiple optimization epochs over one rollout, internal vLLM/importance correction
and alternative objective transforms. An external vLLM caller is separate from
TRL's internal `use_vllm` option. Resume and automatic retry are not implemented;
use one fresh `train()` call per run. Persistent claims, crash recovery and remote
publication belong to the execution owner, and a failed export/reload after an
optimizer step does not authorize repeating that update.

`pipeline.trainer.lmflow_sync_bridge.history` records sealed/updated batches and
their log-prob/publication provenance. `final_publication` identifies the policy
after the last completed update. These in-memory records are not a durable
recovery ledger. Native training metrics are available in `pipeline.train_result`.
