# Minimal GRPO algorithm core

The first LMFlow-native GRPO slice keeps rollout generation, model forwarding,
and optimizer ownership outside the algorithm math. It provides two small
functions that consume the shared `DataProto` runtime container:

- `compute_group_advantages()` reads sequence-level `rewards` and explicit
  `group_ids`, then returns centered, group-scaled advantages;
- `grpo_policy_loss()` reads `loss_mask`, `advantages`, and optional
  `old_log_probs`, then returns a scalar clipped policy loss.

Groups do not need to occupy contiguous rows. Reward scaling uses sample
standard deviation and `epsilon=1e-4`, matching the locked TRL 1.9.2 reference
semantics. Constant-reward and singleton groups produce zero advantages. The
caller must compute advantages while each rollout group is complete, before a
group can be split across distributed ranks; this pure function performs no
collective communication.

The policy objective implements TRL 1.9.2 configured with `loss_type="grpo"`,
token-level importance ratios, symmetric clipping, and per-sequence
normalization before the batch mean. TRL 1.9.2 defaults to a different DAPO
normalization, so comparisons must set the loss type explicitly. Positive
`loss_mask` values can weight policy tokens; zero excludes a token. Every
training sequence must contain at least one selected token.

When `old_log_probs` is absent, the objective uses detached current log
probabilities. This supports the first strictly on-policy, single-iteration
update. Rollout-provided old log probabilities remain available for repeated
updates and later policy-version checks.

This module deliberately does not own model logits, token gathering,
`optimizer.step()`, distributed reduction, rollout transport, or checkpointing.
Those responsibilities remain in the trainer/runtime adapter. The CPU test
closes the first algorithm-math loop by computing group advantages, backpropagating
the scalar objective, and applying a real optimizer update.

## Reference validation

Required CPU tests use hand-computed cases, behavioral invariants, and PyTorch
finite-difference gradient checking without importing an external backend. An
optional differential suite runs only with exactly `trl==1.9.2`; it invokes
TRL's actual private GRPO loss implementation through a version-gated test shim
and compares advantages, scalar loss, per-token gradients, and an SGD parameter
update. The comparison explicitly sets original GRPO semantics and uses binary
masks, which are the common subset of the two implementations.

The TRL shim is test-only because its loss entry point is private and tightly
coupled to Trainer state. It must not be imported by LMFlow product code, and a
TRL upgrade must update the version gate and re-review the adapter before the
differential suite can run.
