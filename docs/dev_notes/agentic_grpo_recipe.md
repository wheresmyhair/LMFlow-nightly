# Minimal GRPO recipe

`run_grpo_step()` is the first narrow algorithm recipe connecting a rewarded
runtime batch to a real policy update. It accepts the existing `DataProto`
directly, computes group-relative advantages from `rewards` and `group_ids`,
writes the derived `advantages` tensor back to that batch, and calls the
selected policy trainer's `train_step()` method.

The recipe deliberately has no parallel batch type, registry, global step, or
controller lifecycle. Rollout and reward components remain responsible for
producing a complete group before distributed sharding. Scheduler, checkpoint,
gradient accumulation, rank synchronization, and policy publication remain
trainer/controller responsibilities.

Returned metrics follow the project naming plan. Trainer metrics without an
existing namespace receive the `train/` prefix, `selected_tokens` becomes
`train/tokens`, and the recipe adds population statistics for rewards and
advantages plus trajectory and group counts:

```text
train/loss
train/tokens
train/advantage_mean
train/advantage_std
reward/total_mean
reward/total_std
rollout/trajectories
rollout/groups
```

These values are local synchronous-step metrics. Distributed reduction and
rank-zero W&B emission are deferred until the FSDP execution slice, where
counts and sums can be reduced before computing global means.
