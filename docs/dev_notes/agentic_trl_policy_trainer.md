# Legacy TRL precomputed-loss adapter

`TRLPolicyTrainer` is retained as a correctness reference for the existing
LMFlow GRPO objective. It does not own the product GRPO training lifecycle.
New sealed-rollout training uses the standard `GRPOTrainer.train()` path
described in [TRL GRPO lifecycle bridge](agentic_trl_grpo_lifecycle.md).

`TRLPolicyTrainer` is the first training-backend adapter for token-native
Agentic updates. It accepts a causal language model, a TRL `GRPOConfig`, and a
precomputed LMFlow `DataProto`; TRL-specific dataset rows, rollout results, and
trainer types do not enter the shared runtime contract.

The adapter maps the full LMFlow sequence to TRL's internal loss input as
follows:

- the first `input_ids` column is a one-token context anchor;
- the remaining columns are passed as TRL completion ids;
- `attention_mask` continues to control model-visible context;
- `loss_mask` is passed separately as the optimization mask;
- sequence-level or token-level `advantages` and optional `old_log_probs` keep
  their existing LMFlow alignment, with column zero removed from token-level
  tensors because it has no causal log-probability.

TRL's public `training_step()` always enters its own generation/scoring data
path. The adapter therefore calls the locked TRL 1.9.2 precomputed-batch loss
path and owns the corresponding Accelerate-backed optimizer step. The strict
version check keeps that private API dependency in one module. TRL is imported
lazily, so importing `lmflow.agentic` without the optional Agentic environment
still succeeds.

Version 1 intentionally supports the same narrow GRPO semantics as LMFlow's
current `grpo_policy_loss`: token-level importance sampling, symmetric clipping,
no KL term, no entropy term, no off-policy filtering, temperature 1, and one
microbatch per optimizer step. Unsupported TRL configuration fails during
adapter construction instead of silently changing the objective.

The adapter uses an initialization-only dataset and zero reward function to
satisfy `GRPOTrainer` construction. Neither is used by `compute_loss()` or
`train_step()`; rollout and reward remain upstream LMFlow responsibilities.
This manual step path must not be extended with additional Trainer lifecycle
management. It remains useful for objective and gradient differential tests
while the official-lifecycle backend is validated.
