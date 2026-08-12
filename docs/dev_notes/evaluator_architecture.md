# Evaluator architecture and compatibility boundary

LMFlow keeps one default `Evaluator` entry point:

```python
result = evaluator.evaluate(model, dataset)
```

Existing `accuracy`, `perplexity`, and negative-log-likelihood calls remain
legacy metric recipes. They keep their current return values and execution
semantics. Their Accelerate/DeepSpeed, `AutoConfig`, CUDA, output-file, and
optional W&B initialization is lazy, so constructing an evaluator no longer
activates those side effects when a recipe-driven evaluation is requested.

Recipe-driven evaluation uses the same class:

```python
result = evaluator.evaluate(model, dataset, recipe=recipe, runner=runner)
```

Applications may also provide `recipe` and `runner` to the evaluator
constructor, retaining the two-argument call `evaluator.evaluate(model,
dataset)`. A later CLI/config slice can select concrete recipes through
`EvaluatorArguments` without changing this user entry point.

## Core boundary

The evaluator core defines only the contracts required to orchestrate and
score an evaluation:

- `EvaluationRecipe` selects a dataset-to-task adapter, external
  `AgentCapability` values, sampling parameters, explicit limits, and a hidden
  verifier.
- `EvaluationTask` pairs a model-visible `TaskSpec` with verifier-only
  material. The evaluator sends only the `TaskSpec` to the runner.
- `ModelRunner` executes one task. Agentic runners should delegate to a shared
  episode executor or an existing benchmark harness instead of implementing a
  second agent loop in the evaluator.
- `Verifier` receives the completed runner output and the verifier-only
  material. Gold answers, hidden tests, and final scoring logic do not enter
  the model prompt through this interface.
- `EvaluationResult` contains aggregate metrics, ordered per-sample records,
  structured failures, optional artifact references, measured usage, and
  recipe/capability/model/dataset provenance.

Capabilities describe external affordances granted during the evaluation,
such as a calculator or search service. They do not claim that the checkpoint
has learned the corresponding competence. Recipe metrics must report learned
performance separately from tool availability and compliance.

## Budgets and failures

Every recipe declares limits for model calls, tool calls, steps, input tokens,
output tokens, wall time, and concurrency. A runner owns active enforcement of
episode-local limits because it controls model and tool execution. The
evaluator checks the runner's reported usage before verification and records
overruns as `budget_exhausted`.

Runners should raise `EvaluationSampleError` for expected typed failures and
use one of the stable categories: timeout, invalid tool call, backend failure,
or budget exhaustion. An ordinary `TimeoutError` is also classified as a
timeout. Failures are isolated to their sample; later samples continue and the
summary reports counts by category.

The core scheduler uses a bounded thread pool for synchronous runners and
preserves dataset order in the returned records. It does not implement retries,
resume, or artifact publication. Those policies need durable attempt identity
and will be added with the first concrete GSM8K runners rather than inferred by
this generic layer. A runner and model used with `max_concurrency > 1` must be
safe for concurrent calls or provide their own serialized backend boundary.

## Near-term recipes

GSM8K direct-answer and calculator-tool evaluation are the first concrete
recipes. They will use the same task content with different capability
profiles, and they will reuse the existing GSM8K scoring and episode
primitives. Their records must distinguish final and strict correctness, tool
compliance, first-attempt success, recovery, direct-answer fallback, invalid
calls, timeouts, calls, steps, tokens, latency, and cost when available.

The existing `calc_gsm8k_reward` tool is a data-generation primitive: it reads
ground truth and returns correctness feedback to the model. It must not serve
as the calculator capability in held-out evaluation because that would expose
verifier information. The calculator recipe needs an arithmetic-only tool;
ground truth remains exclusively in `EvaluationTask.verifier_material`.

AppWorld is the next environment used to test whether the capability and
runner boundaries generalize. A broad plugin registry or universal trajectory
schema remains out of scope until both environments provide concrete evidence
for shared extension points.

## Legacy migration

Callers that omit `recipe` and `runner` remain on the legacy path. A caller may
migrate one benchmark at a time by providing both values; passing only one is
an error. Legacy metrics will eventually become explicit compatibility recipes,
but their current behavior should first be covered by regression tests and
migrated without changing their public return values.
