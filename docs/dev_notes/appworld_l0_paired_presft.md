# AppWorld L0 paired pre-SFT diagnostic

Status: frozen preparation. Do not start a GPU, cloud instance, or paid API
request from this document. The executable bundle must retain the same
machine-readable identities and pass its own strict manifest checks before it
is handed to the GPU/cloud management task.

## Purpose and claim boundary

This Train-only diagnostic makes the cold-start teacher-to-student gap directly
auditable before L1 SFT. It is neither a held-out AppWorld score nor a model
selection experiment. Its results must not tune the SFT recipe. A teacher gap
alone is also insufficient to justify GRPO: entry still requires an SFT
improvement on the frozen held-out protocol and reward variance in Train
K-groups.

The 12 tasks are the four pinned L0 Train scenarios, in this exact order:

`287e338_{1,2,3}`, `27e1026_{1,2,3}`, `2a163ab_{1,2,3}`,
`29caf6f_{1,2,3}`.

Their ordered task-set digest is
`969c03630fe2f4f66d5a67aca5c7b91cda76c1146ee33c221699bc892a370da5`.
All tasks use AppWorld revision
`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`, the pinned Simplified ReAct
Code scaffold, fresh reset and replay, the isolated official evaluator, and the
collateral-state gate. Hidden verifier material remains unavailable to the
model.

## Arms

The teacher reference reuses the existing canonical seed-100 trajectories for
`ZHIPU/GLM-5.3-Flash`; it sends no new provider request. The sealed reference is
12/12 official success, comprising 11 class-A and one class-B trajectory. In
aggregate it has 127 steps, 126 valid actions, one invalid action/recovery, 18
state-changing steps, no duplicate actions, 877,954 input tokens, 17,015 output
tokens, and 590.867 seconds of episode latency. At the frozen conservative list
price, the already-incurred teacher cost for these 12 trajectories is
approximately CNY 0.7500052.

Two student lineages run independently on the same ordered tasks and seed:

- `Qwen/Qwen3-4B` at revision
  `1cfa9a7208912126459214e8b04321603b3df60c`, labelled Starting checkpoint;
- `Qwen/Qwen3-4B-Base` at revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`, labelled Base.

Both student arms use temperature 0.2, seed 100, at most 3,000 completion
tokens, at most 50 steps and at most 1,200 seconds per task. Thinking is
disabled for both student arms. The teacher reference used provider-side
thinking with `reasoning_effort=max`, so teacher-versus-student is explicitly a
system-profile comparison. Backend, tokenizer/chat template, precision,
context implementation, throughput and cost units are also recorded as
profile differences; the report must not present them as a pure checkpoint
ablation.

## Evidence and metrics

Each request is persisted before normalization or assertions. Each step then
persists the parsed action, tool observation, state-change evidence, token
usage, latency and termination metadata. An episode is sealed before fresh
replay, official evaluation and collateral checks. HTTP or service logs cannot
fill a missing artifact.

The report contains per-task rows, per-arm totals/rates and exact task-paired
deltas for four groups:

1. official success, test-pass count/fraction, replay, evaluator errors and
   collateral state;
2. valid/invalid actions, recovery, state changes, repetition, context
   overflow, backend failures and termination taxonomy;
3. steps, model calls, input/output tokens and model/environment/evaluator/total
   latency;
4. the sealed teacher API cost and student GPU-active/control-plane hours,
   nominal GPU cost and observed control-plane billing delta.

Missing evidence is an explicit null with a typed failure reason. Student
quality failure remains a complete diagnostic row and is not grounds to change
the frozen profile inside the run.

## Candidate cloud run sheet

Execution is serial on one 24 GiB RTX 4090 with one model service at a time.
After strict bundle, mount, GPU, disk, nofile, environment, model and tokenizer
identity checks, each lineage first runs one readiness task. A valid artifact
and executable reset/replay/evaluator path allows the remaining 11 tasks;
success is not a readiness requirement. The service is stopped and GPU-empty
is proved before loading the next lineage.

The planning budget is at most 9.0 control-plane hours. At the current planning
price of CNY 1.68/hour this is CNY 15.12 nominal, with a recommended CNY 16.00
hard amount and 30 minutes reserved before the hard deadline for sealing,
return and shutdown. GPU SKU, price, available budget, mount, deadlines,
environment, weight-file manifests and runner bundle digests are all
TBD-BLOCKING until the executor's live preflight.

Any identity or contract mismatch stops before the next model action. Replay
failure preserves the environment evidence and stops that arm for owner review.
The hard control-plane shutdown deadline takes precedence over complete
transfer. Dev9, additional models, an 8B arm, extra seeds, SFT and GRPO are
outside this run.
