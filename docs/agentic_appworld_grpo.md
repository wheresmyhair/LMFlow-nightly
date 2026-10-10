# AppWorld synchronous GRPO producer

`AppWorldTokenNativeRollout` composes the existing AppWorld episode, semantic
projector, token recorder, fresh replay and official evaluator with the
[native synchronous GRPO Pipeline](agentic_synchronous_grpo.md). It is
benchmark-local and adds no training loop or serving scheduler.

## User entry point

Load an LMFlow Model and a pinned Train Dataset, configure TRL `GRPOConfig`, and
use `examples/appworld_grpo.py:build_pipeline`:

```python
# All paths, resource limits and identities come from the experiment config.
pipeline = build_pipeline(
    args, model,
    publish_policy=publish_policy,
    backend_for_policy=backend_for_policy,
    peft_config=lora_config,
    **experiment_config,
)
model = pipeline(model, dataset)
pipeline.trainer.save_model(adapter_output)
```

The example accepts a model/tokenizer revision, explicit prompt file/id/SHA256,
environment/source/artifact paths and step/context/output budgets. It does not
embed machine paths, billing, experiment ledgers or a fixed task/K matrix. Use
`load_pinned_appworld_train_d1_d2_dataset` (or another existing pinned Train
loader) and select rows before invoking the Pipeline. `appworld_grpo_tasks`
copies only task identity and spec digest; prompts are rendered from fresh public
environment metadata. Dev/test rows are rejected. Task spec bytes are checked
before any candidate starts.

## Prompt and token protocol

Passing `AppWorldPrompt` to `run_appworld_episode` supplies an explicit verified
UTF-8 prompt without changing the reference code parser or semantic projector.
Omitting it retains the previous reference-prompt default. A custom prompt's
digest and identity are recorded separately from the pinned reference identity.

For the current remove-demonstration protocol use
`appworld.react-code.remove-demonstration-preserve-apps/v1` and prompt SHA256
`5953342fe5d53a815690072767614b240fdad7e74102a6dfc024c7d235d4de38`.
Supply the reviewed prompt file; the example does not silently recreate it or
switch back to the historical one-shot scaffold. This is an agent protocol,
not an AppWorld evaluator requirement. The Qwen3 replay template must be the
same derived template on the serving side and in the local prompt renderer.

Every request retains actual prompt IDs, sampled output IDs, behavior logprobs,
finish reason and request identity. Canonical prompt equality and sampled-prefix
continuity must both pass. No text retokenization substitutes for sampled IDs.
Each new candidate has an independent environment and a deterministic distinct
seed derived from the configured base seed and rollout ID. Calls within one
candidate retain that seed, matching the existing episode API.

`policy_origin_mask` marks all genuinely sampled assistant tokens, including
reasoning when present. `loss_mask` controls optimization. By default every
sampled assistant token is optimized, including invalid actions; offline SFT's
valid-action filtering is not reused as an RL rule. A caller may supply an
explicitly named `select_calls(calls)` policy to make whole calls audit-only.
Their tokens/logprobs/origin remain intact. Fine-grained reasoning/action token
selection is not implemented here. Environment, initial history and deterministic
transport tokens have zero masks. Zero-optimization members retain their outcome
inside a complete group; the existing bridge still rejects an entirely zero-mask
batch.

`max_completion_length` counts the entire flattened suffix after the first actual
prompt, including intermediate environment/transport tokens. Checking only the
sum of sampled output lengths is insufficient. No history or completion is
silently truncated to fit training. Temperature/top-p are 1, with no sampling
warpers, under the current behavior-old native bridge support matrix.

## Publication and Trainer ownership

The existing `publish_policy(trainer, version)` callback exports/loads the
current policy and returns `policy_version`, `global_step`, `weight_digest`,
and `source`. The producer then calls
`backend_for_policy(receipt, trainer) -> (completion_backend, loaded_receipt)`.
It requires exact equality before sampling. The loaded receipt must describe
the actual sampling target, not merely echo a desired version. An external
runtime must prevent the endpoint from changing weights during the group and
release sampling resources before native training resumes. Endpoint URLs alone
cannot prove weight identity; service management remains the caller's job.

One native `GRPOTrainer.train()` owns all updates, accumulation, gradient
checkpointing, optimizer and scheduler. The next group is sampled only after
publication of the next policy. Final publication/export/reload still belongs to
the experiment runtime; this module does not implement recovery or distributed
training.

## Rewards, failures and artifacts

Each candidate executes once and replays its recorded actions from a fresh reset.
Reward is terminal `len(official passes) / official num_tests`, within [0, 1].
Only scalar counts, binary success and trajectory identity enter the DataProto.
Verifier requirements, traces and answer material remain in restricted local
episode audit files, never in model messages or training metadata. Replay/state
agreement is mandatory; official success and collateral success are not required
for admitting genuine unsuccessful RL trajectories. No tool/format bonuses or
teacher/auditor rewards are added.

Each batch directory is exclusively created from its identity digest. Received
raw responses and raw execution outputs are atomically saved before parsing or
validation; episode, token, selection and replay evidence precede completion.
Token evidence plus selection records reconstruct the padded training batch.
Only complete K-groups are returned. Backend, environment, evaluator, token,
length or replay failures preserve partial files and fail the current batch.
They do not become synthetic zero rewards or trigger automatic retries. A failed
candidate is never replaced with a copy of a historical episode. Existing claims
are never overwritten. A runtime may separately dispatch independent experiments;
this producer does not schedule them.

The local evidence directory may contain verifier-only material. Do not pass it
to a model, export it as training Conversations, or commit it to Git.

## Evidence limits

CPU fake-world tests cover fresh reset/replay, group completeness, genuine task
failure vs infrastructure failure, scalar isolation, masks, lengths and policy
receipts. A tiny CPU native-lifecycle test uses synthetic AppWorld actions to
check continuous integration, not real policy sampling or benchmark quality.
Historical real episodes may verify token conversion only; they do not prove a
fresh group or an updated policy. Real acceptance requires independently sampled
groups before and after an update, under published weights, and verified native
optimizer/scheduler continuity. Task success and reward variance are reported
separately from this engineering gate.
