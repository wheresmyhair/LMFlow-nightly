# SWE-bench task and verification adapters

LMFlow provides a narrow bridge between an official SWE-bench instance and
the existing mini-swe-agent episode runner. The bridge keeps benchmark input,
model-visible context, rollout artifacts, and grading data at explicit
boundaries.

## Task preparation

`prepare_swe_bench_task()` accepts one SWE-bench instance and either a local
source repository or `PreparedRepositoryCache`. It copies only these fields
into the runtime task:

- `instance_id`
- `repo`
- `base_commit`
- `problem_statement`

The golden `patch`, hidden `test_patch`, `hints_text`, `FAIL_TO_PASS`, and
`PASS_TO_PASS` fields are deliberately excluded from `TaskSpec`. They remain
available to the external benchmark harness but cannot leak into the scaffold
through this adapter.

`base_commit` must be a full 40-character Git commit. With `source_repo`, the
repository must already exist locally. With `repository_cache`, the adapter
prepares the exact commit from `https://github.com/{repo}.git`; an explicit
`repository_url` can select a mirror or local test source. The episode runner
verifies the prepared revision again when it creates a fresh checkout, before
the first model request.

```python
from lmflow.agentic import PreparedRepositoryCache, prepare_swe_bench_task, run_swe_bench_episode

repository_cache = PreparedRepositoryCache("/datasets/repository-cache")
task = prepare_swe_bench_task(instance, repository_cache=repository_cache)
artifact_dir = run_swe_bench_episode(
    task=task,
    model=model,
    agent_config=agent_config,
    rollout_id="rollout-0001",
    workspace_root="/datasets/workspaces",
    artifact_dir="/datasets/artifacts/rollout-0001",
)
```

One successful call publishes the existing `trajectory.json` and
`model.patch` artifact pair. Episode checkout, patch export, and cleanup retain
the lifecycle documented in `agentic_swe_rollout.md`.

## Official prediction export

`swe_bench_prediction_from_artifact()` validates the artifact identity against
the prepared task and returns exactly the three fields consumed by the
official harness:

```python
prediction = swe_bench_prediction_from_artifact(
    task=task,
    artifact_dir=artifact_dir,
    model_name_or_path="Qwen/Qwen3-8B",
)
# instance_id, model_name_or_path, model_patch
```

The caller owns JSONL aggregation and submission to the official evaluator.
Prediction export only needs the prepared task identity and published artifact;
the local source repository may already have been evicted.

## Clean-workspace verification

`verify_swe_bench_artifact()` checks out the exact base revision in a new
`EpisodeWorkspace`, applies `model.patch` without fuzzy or three-way fallback,
and runs one argv-only verifier command through `ProcessSandbox`. The checkout
is removed after the command, and the source repository is never modified.

```python
result = verify_swe_bench_artifact(
    task=task,
    artifact_dir=artifact_dir,
    verifier_command=("python", "-m", "pytest", "-q"),
    workspace_root="/datasets/verification-workspaces",
)
```

A zero return code means only that the supplied verifier command succeeded.
This local helper does not infer SWE-bench resolution, apply hidden test
patches, or reimplement FAIL_TO_PASS/PASS_TO_PASS grading. The official
SWE-bench harness remains authoritative for benchmark results.

The current API intentionally handles one prepared task at a time. Batch
scheduling, dependency-environment caching, persistent verification reports,
cache eviction, and an official harness launcher are future orchestration
layers rather than responsibilities of this adapter.
