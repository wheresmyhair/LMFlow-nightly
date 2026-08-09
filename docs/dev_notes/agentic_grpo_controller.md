# Synchronous GRPO reference controller

`run_synchronous_grpo_step()` closes the Phase 0 in-process reference path:

```text
TaskSpec -> rollout requests -> token-native rollouts -> complete groups
         -> rewards -> group advantages -> policy trainer update
```

The function expands each input task into a group of at least two rollouts and
creates attempt-local integer `group_ids` and `rollout_ids`. The injected
`rollout_fn` receives those identities, repeated `TaskSpec` objects, `task_ids`,
and a batch-constant `policy_version` in one `DataProto`. It returns a
token-native `DataProto` carrying the same identities, task mapping, and policy
version. The controller validates each returned `rollout_id -> task_id` mapping.
Results may arrive in any row order; `RolloutGroupAssembler` restores registered
group order.

The injected `reward_fn` receives only complete groups and returns one real,
finite reward per row. The controller attaches those rewards, calls
`run_grpo_step()` to compute group-relative advantages, and delegates the real
optimizer update to the selected policy trainer. It returns recipe and group
assembly metrics without adding another batch or rollout result type.

This reference step intentionally requires every preselected group to complete
before it updates the policy. A partial synchronous response fails the whole
step, avoiding latency-based task selection. The timeout rejects results that
return after the deadline but cannot preempt a blocking Python rollout call.
Asynchronous scheduling, active cancellation, retries, vLLM lifecycle,
colocated GPU phase switching, checkpointing, and policy publication remain
Phase 3 controller responsibilities.
