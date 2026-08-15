# Evaluator architecture and compatibility boundary

LMFlow keeps one default `Evaluator` entry point:

```python
result = evaluator.evaluate(model, dataset)
```

`Evaluator` is also callable, so code following the general Pipeline notation
may use `result = evaluator(model, dataset)`.

Existing `accuracy`, `perplexity`, and negative-log-likelihood calls remain
legacy metric recipes. They keep their current return values and execution
semantics. Their Accelerate/DeepSpeed, `AutoConfig`, CUDA, output-file, and
optional W&B initialization is lazy, so constructing an evaluator no longer
activates those side effects when a recipe-driven evaluation is requested.

Recipe-driven evaluation uses the same pipeline. Recipe, runner, and runtime
configuration belong to evaluator construction, so the execution call stays
uniform:

```python
result = evaluator.evaluate(model, dataset)
```

The Python API still accepts per-call overrides for focused testing and
advanced integrations. User-facing scripts should configure those components
through pipeline args or constructor wiring and keep the two-argument call.

## Core boundary

`Evaluator` is a Pipeline implementation. Its facade, recipe contracts,
structured results, local runtime, orchestration, and legacy metric backend
live together under the `lmflow.pipeline.evaluation` domain package. The
Pipeline facade is implemented by
`lmflow.pipeline.evaluation.evaluator.Evaluator`. LMFlow does not expose a
parallel top-level `lmflow.evaluation` subsystem.

The package root intentionally exports only `Evaluator`. Recipe authors may
import the narrow contracts from `lmflow.pipeline.evaluation.recipe`, result
types from `lmflow.pipeline.evaluation.result`, and execution runtimes from
`lmflow.pipeline.evaluation.runtime`. The former
`lmflow.pipeline.evaluator.Evaluator` import remains a narrow compatibility
alias and emits `DeprecationWarning` when imported. Benchmark adapters should expose their
own recipe factories instead of widening the generic Pipeline namespace.

The evaluator package defines only the contracts required to orchestrate and
score an evaluation:

- `EvaluationRecipe` selects a dataset-to-task adapter, external
  `CapabilityProfile`, sampling parameters, explicit limits, and a hidden
  verifier. The runner records the concrete model-visible scaffold it executes.
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

`CapabilityProfile` describes external affordances granted during the evaluation,
such as a calculator or search service. They do not claim that the checkpoint
has learned the corresponding competence. Recipe metrics must report learned
performance separately from tool availability and compliance.

Scaffold identity is separate provenance. The GSM8K recipes currently define
one reference scaffold revision. Future seen-alternate and held-out-compatible
scaffolds must keep dataset, capability profile, budget, and verifier fixed
when their results are compared; this slice does not introduce a broad
`ScaffoldDistribution` API.

## Budgets and failures

Every recipe declares limits for model calls, tool calls, steps, input tokens,
output tokens, and wall time. A runner owns active enforcement of
episode-local limits because it controls model and tool execution. The
evaluator checks the runner's reported usage before verification and records
overruns as `budget_exhausted`.

Runners should raise `EvaluationSampleError` for expected typed failures and
use one of the stable categories: timeout, invalid tool call, backend failure,
or budget exhaustion. An ordinary `TimeoutError` is also classified as a
timeout. Failures are isolated to their sample; later samples continue and the
summary reports counts by category.

The evaluator delegates execution to an `EvaluationRuntime`. The default local
runtime is the synchronous correctness reference and can use bounded thread
concurrency while preserving dataset order. Dataset does not own worker,
queue, retry, or recovery behavior. The local runtime does not implement
retries or resume; those policies require durable attempt identity and remain
future run-level runtime work. A runner and model used with concurrent local
execution must be safe for concurrent calls or provide a serialized backend
boundary.

The core consumes task adapters as iterables and does not require `len()` or
full in-memory materialization. Duplicate task identities are rejected as the
stream is consumed. A future sealed Dataset view may provide manifest-level
identity validation before execution without making Dataset a scheduler.

## GSM8K recipes

GSM8K direct-answer and calculator-tool evaluation are the first concrete
recipes. Both consume the same `Dataset`; recipe-level adapters add different
model-visible prompts, tools, and capability profiles without copying a
capability configuration into every dataset row:

```python
from lmflow.agentic.gsm8k_evaluation import (
    GSM8KCompletionRunner,
    create_gsm8k_calculator_recipe,
    create_gsm8k_direct_recipe,
)
from lmflow.pipeline.evaluation import Evaluator

direct_evaluator = Evaluator(
    model_args,
    data_args,
    evaluator_args,
    recipe=create_gsm8k_direct_recipe(),
    runner=GSM8KCompletionRunner(
        backend=completion_backend,
        model_name="Qwen/Qwen3-8B",
        artifact_dir="output_dir/gsm8k-direct/records",
    ),
)
direct_result = direct_evaluator.evaluate(model, dataset)

tool_evaluator = Evaluator(
    model_args,
    data_args,
    evaluator_args,
    recipe=create_gsm8k_calculator_recipe(),
    runner=GSM8KCompletionRunner(
        backend=completion_backend,
        model_name="Qwen/Qwen3-8B",
        artifact_dir="output_dir/gsm8k-calculator/records",
    ),
)
tool_result = tool_evaluator.evaluate(model, dataset)
```

GSM8K components live in the Agentic benchmark adapter rather than in the
generic evaluator package. `lmflow.pipeline` continues to expose the generic
Evaluator instead of one pipeline per benchmark. A `text2text` dataset uses
`input` as the question and `output` as the gold solution; an official-shaped
dataset may use `question` and `answer`. The adapter extracts the final gold
answer and places it only in `EvaluationTask.verifier_material`.

Calculator tool use is optional by default, which makes direct-answer fallback
measurable. Pass `require_tool_use=True` to
`create_gsm8k_calculator_recipe()` to record a required-use capability profile
and request the calculator on the first model call when the selected provider
implements named function `tool_choice`.

Both recipes use the shared synchronous `CompletionBackend` control plane.
Sampling is fixed by `SamplingConfig`; runner-level model options cannot
override temperature, top-p, seed, or output-token limits. The runner actively
checks model-call, tool-call, step, reported input/output-token, and wall-time
limits. The total output-token budget is divided into an equal per-model-call
ceiling and unused tokens remain available to later calls. Provider timeouts,
invalid tool calls, budget exhaustion, and backend
failures become structured per-sample failures. When `artifact_dir` is set,
completed and failed records atomically publish raw completion/tool feedback,
usage, recipe limits, and model-visible task data under a task-id digest.

The hidden verifier reports final and strict correctness, tool compliance,
first answer-bearing assistant attempt success, later recovery, direct-answer
fallback in the calculator recipe, and recoverable calculator arithmetic
errors. Calls, steps, tokens, latency, and cost remain in `EvaluationUsage` and
the result summary. Token limits can only be checked against provider-reported
usage; a backend that omits usage leaves those fields unavailable.

The existing `calc_gsm8k_reward` tool remains a data-generation primitive: it
reads ground truth and returns correctness feedback to the model. Held-out
evaluation uses a separate arithmetic-only `calculate` tool that accepts a
bounded expression and has no access to the expected answer. Ground truth
remains exclusively in `EvaluationTask.verifier_material` and is absent from
prompts, completion requests, capability config, and raw runner artifacts.

The completion runner is checkpoint-neutral. Base, SFT, and SFT+GRPO models use
the same dataset, recipe, sampling, budget, and verifier; only model provenance
and the served checkpoint change. This slice uses the existing text control
plane and does not claim the token-native online-RL rollout contract.

The result records are the first concrete shape that a future
`Dataset[EvaluationRecord]` view can expose. This slice does not add an
`Experience` object or freeze a general typed-view framework. AppWorld is the
next environment used to test whether the capability, scaffold, runner, and
runtime boundaries generalize. Retry/resume and a run-level artifact manifest
also remain follow-up work. A broad plugin registry or universal trajectory
schema stays out of scope until both environments provide concrete evidence
for shared extension points.

## Legacy migration

Callers that omit `recipe` and `runner` remain on the legacy path. A caller may
migrate one benchmark at a time by providing both values; passing only one is
an error. Legacy metrics will eventually become explicit compatibility recipes,
but their current behavior should first be covered by regression tests and
migrated without changing their public return values.
