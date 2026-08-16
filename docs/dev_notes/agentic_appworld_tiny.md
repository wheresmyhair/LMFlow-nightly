# AppWorld tiny end-to-end evidence

This note records the first AppWorld product slice as benchmark-local evidence.
It does not define a shared EpisodeExecutor, CapabilityProfile plugin, evaluator
contract, or report layer.

## Fixed upstream and compatibility

- Repository: `https://github.com/StonyBrookNLP/appworld.git`
- Code revision: `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`
- Code version: `0.2.0.dev0`
- Data version: `0.2.0`
- License: public source is Apache-2.0; protected source/data may be used for
  training and evaluation, but public redistribution of those materials or
  derivatives must preserve AppWorld's encryption restrictions.
- Python evidence: Python 3.12.3, Pydantic 2.13.4, FastAPI 0.136.3, SQLModel
  0.0.39, OpenAI 2.53.0, and the existing Torch/Transformers/TRL/vLLM pins pass
  task loading and the local AppWorld verifier.

The PyPI stable tag `0.1.3.post1` still pins Pydantic below 2 and SQLModel below
0.0.11, so it conflicts with the unified agentic environment. The pinned
upstream revision declares Python 3.11-3.14 and Pydantic 2 support. The single
environment therefore uses a hashed lock for AppWorld's declared dependencies,
then installs the exact Git+LFS checkout with no dependency resolution. A clean
lock dry-run resolves 247 packages without an unhashed requirement. The
bootstrap acceptance checked the four LFS bundle hashes, built the source with
the locked `uv-build`, downloaded a fresh data root, and passed the install
verification command.

`uv pip check` still reports LMFlow's known editable-package metadata mismatch:
the package metadata asks for older `datasets` and `evaluate`, plus optional
`cpm-kernels` and `bitsandbytes`, while the coordinated agentic lock intentionally
uses its tested complete stack. No AppWorld requirement is missing or
incompatible in that report.

## Tiny task protocol

The source split is official `dev`, whose pinned task-list SHA-256 is
`9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001`.
The tiny set contains three complete scenario groups so AppWorld's scenario goal
completion remains meaningful:

| Difficulty | Scenario | Tasks | Coverage |
|---|---|---|---|
| 1 | `396c5a2` | `396c5a2_1`, `396c5a2_2`, `396c5a2_3` | Spotify state change |
| 2 | `6c2c621` | `6c2c621_1`, `6c2c621_2`, `6c2c621_3` | Simple Note to file-system cross-app workflow |
| 3 | `530b157` | `530b157_1`, `530b157_2`, `530b157_3` | Phone and Venmo cross-app workflow |

The ordered task-ID SHA-256 is
`dafe2f2aad8a8cfabaa33550e9e9196d0a284b5bef2cca639b48e7c8e7dac67a`.
The repository stores IDs, selection facts, and digests only. The Dataset
projection loads instructions at runtime with `load_ground_truth=False`; ground
truth enters only the official evaluator after the episode.

## Reference scaffold and execution

The adapter uses the official `SimplifiedReActCodeAgent` behavior at the pinned
revision:

- exact verified prompt;
- one Python code block per model step and first-complete-block parsing;
- model-visible `Output:` framing between steps;
- AppWorld random seed 100 and default safety guards;
- at most 50 serial steps;
- official Qwen3-8B no-reasoning profile: temperature 0, seed 100,
  `max_completion_tokens=3000`, thinking disabled.

The `appworld-agents` package is not installed because it currently constrains
OpenAI to 1.99.8 or older. LMFlow's existing OpenAI-compatible completion
backend supplies model calls while the AppWorld environment and official
evaluator retain benchmark semantics.

AppWorld freezes task time with freezegun. In the unified environment, vLLM's
processor namespace contains lazy optional multimedia imports that freezegun
would otherwise traverse when both packages have already been imported. The
benchmark-local runner adds the `vllm` module prefix to freezegun's ignore list
before environment initialization; model serving runs in its separately
managed process and does not depend on AppWorld's frozen task clock.

Each task artifact records model/scaffold/AppWorld identities, sampling,
platform, initialization and reset provenance, model and execution latency,
token usage, code actions, redacted API-call audit data, state digests,
completion signal, official verifier result, failure class, and recovery count.
Here a tool call means one reference-scaffold Python execution action; valid and
invalid counts are determined from AppWorld's execution result. API request
attempts are counted separately. A recovery is a valid action immediately after
one or more failed actions.

The CLI writes to a temporary sibling directory, fsyncs JSON files, and renames
the directory only after every task and report succeeds. It also preserves the
AppWorld logs/version/evaluation sections, an LMFlow `text_only` task Dataset,
and one `conversation` training projection per trajectory. Database snapshots,
model files, caches, and ground-truth source are not copied into Git or the
published run directory. Structured API audit fields redact token/password
keys; AppWorld's raw local logs can still contain synthetic benchmark
credentials and therefore remain protected artifacts outside Git.

Example full tiny run (the model server is managed separately):

```bash
python -m lmflow.agentic.evaluate_appworld run \
  --artifact-dir /outside/the/repository/appworld-qwen3-8b-base-tiny \
  --run-id appworld-qwen3-8b-base-tiny \
  --appworld-root "$APPWORLD_ROOT" \
  --appworld-source "$APPWORLD_SOURCE" \
  --base-url http://127.0.0.1:18001/v1 \
  --served-model-name Qwen/Qwen3-8B \
  --tokenizer-path /path/to/pinned/qwen3-8b \
  --backend-version 0.25.1
```

Omit `--task-id` to run all nine tasks. Repeating `--task-id` selects an ordered
subset for development. Scenario goal completion is flagged as uninterpretable
when any selected scenario lacks one of its three variants. The default
per-step output cap is the official profile's 3000 tokens; a smaller
`--max-completion-tokens` is a recorded development budget rather than an
untracked server-side change. `--enable-thinking` records and enables Qwen3's
chat-template thinking mode; omitting it preserves the official AppWorld
no-reasoning profile.

## Local Base evidence

The pinned Qwen3-8B Base revision
`b968826d9c46dd6066d109eabc6255188de91218` was served on one local RTX 4090
with vLLM 0.25.1, BF16, a 16,384-token server context, V1 model runner, one
sequence, and no retries. An official-profile probe at 3,000 output tokens per
step produced eleven real AppWorld interactions before the local vLLM decode
stalled. That interrupted probe was not atomically published as a benchmark
result.

A controlled diagnostic run lowered only the recorded per-step output cap to
128 tokens and completed all nine fixed tasks. Its run-manifest SHA-256 is
`cd7012d867ee1809c2768141e3951a8f966fd8573141018d65fa146f4172baf4` and
report-manifest SHA-256 is
`11789196847a21c0eddd1b3641fbab787440a83b45f2c6876633c739d37acddf`.
The official evaluator reported task goal completion 0.0 and scenario goal
completion 0.0 across all three complete scenario groups. The run recorded:

- 177 model/action steps, 118 valid actions, and 59 invalid actions;
- 105 underlying API attempts, 31 recoveries, and 6 state-changing steps;
- 3 task-completed signals that still failed the official verifier;
- 2 trajectories that reached the 50-step limit and 4 model-backend errors;
- 921.34 seconds total episode time, including 786.63 seconds in model calls.

The 128-token result is the first reproducible Base tiny execution and failure
baseline. It is not an official-profile quality number and must not be compared
as though it used AppWorld's 3,000-token Qwen3-8B configuration. A future
official-profile rerun needs a stable local serving stack or separately approved
compute. The run manifest records the model revision and tokenizer hashes, but
the local weight artifact digest was not supplied and remains a provenance
limitation.

Validation on the fresh Python 3.12/AppWorld root covered deterministic double
reset, real one-step environment execution, state/evaluator access, nine focused
tests, the full LMFlow Agentic regression (`307 passed, 2 skipped`), Python
compilation, Ruff check/format, shell syntax, lock replay dry-run, and the
bootstrap from the exact Git+LFS revision. The full upstream test matrix was
intentionally not used as the acceptance boundary; AppWorld's core 1,652 tests
and common 76 tests passed separately, while remote-mode and MCP-specific tests
remain outside this local in-process slice.

## Local 4B checkpoint compatibility gate

The selected local 4B lineages are pinned independently:

- `Qwen/Qwen3-4B` revision
  `1cfa9a7208912126459214e8b04321603b3df60c`, reported as a starting
  checkpoint because it is post-trained;
- `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`, reported as Base.

Both exact snapshots load in the unified Python 3.12 environment with vLLM
0.25.1, BF16, a 32,768-token server context, V1 model runner, one sequence, and
0.70 GPU-memory utilization on the local RTX 4090. Each model used about 7.55
GiB for weights and retained about 8.99 GiB for KV cache; vLLM reported KV
capacity equivalent to about two 32,768-token requests, while the run remained
explicitly capped at one sequence. The gate ran serially with one model service
at a time and shut each service down after its artifact was published. No CUDA,
NVML, timeout, worker-restart, or HTTP error occurred.

Three protected one-task artifacts record behavior without claiming a tiny9
quality comparison:

- Qwen3-4B with the official no-thinking profile repeated the same valid API
  documentation query for all five allowed diagnostic steps. It made five API
  attempts, changed no state, consumed 40,040 tokens, and ended at the step
  limit. Its run-manifest SHA-256 is
  `41d9defadbc653928e53c42926327975d15563c0dee53fefb192233c40eff1a7` and
  report-manifest SHA-256 is
  `63628467a60fb8e1faf3d5fb13a9e080d4dcbd80132478c15a001806faaf8c62`.
- Qwen3-4B with thinking enabled broke the exact repetition and attempted
  successive repairs, but all ten Python actions ended in execution errors. It
  made 33 API attempts, changed no state, consumed 91,443 tokens, and used
  162.38 seconds of model time. Its run-manifest SHA-256 is
  `ea2e1c740a55e6b3eec629366734f23a75fc66a7221e7b2ffef5e7d964bff4d1` and
  report-manifest SHA-256 is
  `85515c5a680fcf7fcd325e2ca879bd74fa181b3481ce50a586004ed7e32b86c4`.
- Qwen3-4B-Base completed the serving and environment compatibility path but
  repeated one valid discovery action until the 3,000-token output cap. The
  single action changed no state and used 43.49 seconds of model time. Its
  run-manifest SHA-256 is
  `7b82c4e7af740fa2bb2afa624652030c5bbf1514f25a49f8da825869f3b03e7d` and
  report-manifest SHA-256 is
  `e64eb54943c64592aacce89f95ae86c40417783cb6249414b2746f78b543e36a`.

All three official evaluations returned task goal completion 0.0. These runs
establish 32K execution compatibility and concrete cold-start failure modes;
they do not replace a post-SFT tiny9 evaluation. The evidence favors using the
post-trained checkpoint for teacher-guided cold-start development while keeping
the raw Base lineage as the pure-base control. Thinking remains an explicit
recorded choice because it trades exact repetition for much longer outputs and
currently unsuccessful API repair behavior under the unchanged official
scaffold.

## Cold-start data factory gate

The benchmark-local data factory uses only the official `train` split. The
pinned data lists contain 90 train tasks in 30 scenarios, 57 dev tasks in 19
scenarios, 168 normal-test tasks in 56 scenarios, and 417 challenge-test tasks
in 139 scenarios. Their task-list SHA-256 values are respectively
`93d9fe71e7a2e3b7529803d4a20b604f4ebf5ae806f321081140238068189d37`,
`9fa976589300ea8905708257144d801d1604b06d85fb0181e381df8a3ba85001`,
`c3af41497b6f2f0860a2ff8c09b335dca527e2cf48e59b4aabdb301b6b68db8f`,
and `3c32b481042ac97f7d3477d53f5d196245c885c438d652944edc8a9a28e0f028`.
The scenario-ID intersection is empty for every split pair.

The unpaid pilot protocol deterministically freezes the first complete train
scenario in official order at each difficulty:

| Difficulty | Scenario | Tasks |
|---|---|---|
| 1 | `82e2fac` | `82e2fac_1`, `82e2fac_2`, `82e2fac_3` |
| 2 | `692c77d` | `692c77d_1`, `692c77d_2`, `692c77d_3` |
| 3 | `6104387` | `6104387_1`, `6104387_2`, `6104387_3` |

The ordered pilot task-ID SHA-256 is
`d8a89fb3037ce6fe078d72517b80146c1c2cd1f6c007cad79beaa06aa3252327`.
Two to four candidates per task produce an 18-36 trajectory micro pilot. A
provider, model, endpoint identity, sampling profile, effective pricing source,
and explicit paid-run approval are still required before execution. No online
teacher call has been made by this slice.

Every candidate follows a fail-closed path:

1. Start from the pinned revision and a fresh reset, run the unchanged official
   scaffold against the real environment, and invoke the official evaluator.
2. Start from another fresh reset and replay the exact recorded Python actions.
   Compare initial state, every action output digest, validity, API-call count,
   per-step state digest, final state, and the sealed official-evaluation
   summary.
3. Exclude any replay mismatch. Admit success only when both original and replay
   pass every official test and reach the same permitted state delta.
4. Classify as A verified success, B verified recovery, C sealed partial, D
   auditable failure, or E infrastructure/invalid. C, D, and E emit no SFT
   conversation in the current slice.
5. For B, keep failed actions and their visible observations as context with
   `loss:false`; train only valid recovery actions. A feeds both the
   success-only and success-plus-recovery arms, while B feeds only the latter.

The atomic factory report aggregates accepted yield, A-E counts, collateral
rejection, replay mismatch, duplicate accepted targets, token usage,
trainable-output-token p50/p95, truncation rate, and provider-derived USD cost.
Credentials are rejected from persisted provider identity. Raw trajectories,
replay evidence, AppWorld logs, and Dataset instructions remain protected local
artifacts outside Git; Conversation projections contain no verifier details or
hidden state. D can produce Preference only after a separately verified
improved pair under the same task and reset, which this first factory slice does
not synthesize.

The unified agentic virtual environment can retain an editable-install pointer
to a different LMFlow worktree. Tests and local entry points for this slice must
therefore resolve the active checkout explicitly, for example with
`PYTHONPATH="$PWD/src"`, unless the environment has just been installed from the
same worktree. This avoids silently importing stale AppWorld modules while still
keeping one long-lived Python 3.12 environment.

## GSM8K and AppWorld friction evidence

Shared friction observed in both benchmarks:

- dataset revision, ordered IDs, content digests, prompt/scaffold identity, and
  provider behavior all need explicit provenance outside model weights;
- the serving backend's chat-template and generation behavior remain more
  detailed than the current generic Evaluator provenance;
- atomic per-task artifacts, usage/latency accounting, failure classes, and
  machine-path-free manifests are useful across benchmarks;
- a successful control-plane trajectory still does not provide sampled token
  IDs or log probabilities required by online RL.

AppWorld-specific differences exposed by this slice:

- the environment has mutable multi-app database state and global in-process
  ownership, so reset and serial execution are correctness requirements;
- actions are Python REPL programs containing zero or more API requests, which
  creates distinct action-validity, API-attempt, state-change, and recovery
  concepts;
- termination signals and official verifier success can disagree;
- the official evaluator checks goal completion and collateral damage from DB
  deltas, and scenario metrics require complete variant groups;
- long observations and API discovery make context growth a first-class budget;
- task-level raw artifacts and verifier details are protected benchmark data;
- the exact install depends on Git LFS bundles and a separate versioned data
  download, unlike the ordinary Hugging Face Dataset path used by GSM8K.

Possible cross-benchmark review inputs, still unconfirmed:

- an episode execution boundary may need to expose reset, termination, state
  transition evidence, and recovery without making Dataset a runtime queue;
- capability identity may need scaffold/environment affordances in addition to
  native function schemas;
- benchmark evaluators may benefit from a narrow result-normalization hook while
  retaining their official semantics;
- atomic artifact publication and provider-behavior provenance are candidates
  for reuse after a third concrete path confirms them.

These are evidence items for the post-AppWorld architecture review. This slice
keeps them inside the AppWorld module and does not change the public Evaluator.
