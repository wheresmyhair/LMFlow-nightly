# Rollout group assembly

`RolloutGroupAssembler` is a controller-local ready queue for synchronous
group-relative algorithms. The controller registers the exact logical rollout
requests for a group before dispatch, then adds token-native `DataProto`
results as workers finish. A group becomes consumable only after every
registered rollout identity has returned.

Registration gives the assembler enough information to expire a group even if
no worker returns, reject unexpected results, and restore deterministic group
order after out-of-order completion. Required result fields are deliberately
small:

| Location | Field | Meaning |
| --- | --- | --- |
| `meta_info` | `policy_version` | Batch-constant policy revision; it must match the assembler. |
| `non_tensor_batch` | `group_ids` | Attempt-scoped group identity for each row. |
| `non_tensor_batch` | `rollout_ids` | Globally unique logical request identity for each row. |

`pop_ready()` consumes one or more complete groups without splitting group
boundaries. Multiple ready groups can then be concatenated into a training
batch, passed to `run_grpo_step()` to compute advantages, and arbitrarily
sharded only after those advantages exist.

Timeout and explicit cancellation close the entire group. Late results for a
closed group fail rather than entering a later update. A whole-group retry must
register new group and rollout identities; the old attempt remains tombstoned
for the lifetime of this policy-version assembler. Recreate the assembler when
the controller publishes a new policy version.

The component reports local pending/ready counts, terminal counters, rejected
duplicates and policy mismatches, dispatch-to-completion latency, and
first-result-to-last-result straggler wait. It does not launch retries, persist
requests, perform distributed collectives, switch colocated GPU phases, or
publish policy weights; those remain controller responsibilities.
