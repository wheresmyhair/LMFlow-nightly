# Minimal Bash agent

`MinimalBashAgent` is LMFlow's lightweight adaptation of the mini-swe-agent
control flow. The compatibility reference is
`SWE-agent/mini-swe-agent@a83fcae82d2a08f0ee0c688f9d137b3566c097f8`.
The adapted behavior and Bash tool definition retain the upstream MIT notice in
`THIRD_PARTY_NOTICES.md` and `LICENSES/mini-swe-agent-MIT.txt`.

LMFlow vendors only the small scaffold behavior. It does not add mini-swe-agent,
LiteLLM, Textual, provider SDKs, or container backends to the runtime
dependencies.

## Responsibility boundaries

```text
EpisodeWorkspace (pipeline-controlled repository state)
    -> BashAgentEnvironment (command execution backend)
        -> MinimalBashAgent (model-visible control flow)
    -> immutable artifacts, patch export, and independent verifier
```

`EpisodeWorkspace` owns the fixed base revision, checkout, reset, patch export,
and cleanup. It is not part of the model scaffold. `ProcessSandboxBashEnvironment`
adapts the existing `ProcessSandbox`; Docker, Singularity, or remote backends can
implement the same narrow environment protocol later.

The local adapter requests merged stdout/stderr capture from `ProcessSandbox`.
This preserves the write order that mini-swe-agent exposes to the model while
the raw `ProcessResult` still retains timeout, truncation, duration, command,
and return-code metadata.

## Pinned control-flow behavior

The current implementation preserves the behavior that affects scaffold
comparability:

- one or more structured Bash tool calls may be returned in a model response;
- calls execute sequentially and their observations are returned together;
- malformed or missing tool calls produce a corrective user message;
- repeated format errors, model-call count, billed cost, and wall time have
  explicit limits;
- a clean model turn resets the consecutive format-error counter;
- the Bash tool schema and submission marker follow the pinned upstream commit;
- the first successful command whose whitespace-trimmed first output line is
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` ends the rollout;
- commands run in fresh shell processes while repository files persist in the
  episode workspace.

The injected model callable receives independent OpenAI-style `messages` and
`tools` values and returns a normalized `BashAgentTurn`. Provider adapters keep
ownership of HTTP, authentication, retry, usage extraction, token-native output,
and API-specific raw-response handling. They should use
`bash_agent_turn_from_openai_message()` and pass billed cost, finish reason, and
the complete raw response. Recoverable parser failures raise
`BashAgentFormatError`, which the scaffold records and feeds back to the model.

## Raw facts and training projection

Each result separates two representations:

- `raw_trajectory` is the factual model-visible history. It records assistant
  responses, format-error feedback, exit state, accounting, and the fixed
  upstream revision. The outer data pipeline should persist it together with
  raw provider responses, `ProcessResult` values, patch, verifier logs, and
  provenance.
- `atif_trajectory` is an ATIF v1.7 projection for the existing conversation
  converter. It is present only when every relevant transition can be
  represented faithfully by the supported ATIF subset.

A recovered malformed model response currently makes `atif_trajectory` null.
The raw rollout remains available, but the pipeline must not silently train on
an incomplete projection. Likewise, if an early submission leaves later tool
calls in the same response unexecuted, the loss-ready ATIF projection is
withheld.

```python
with EpisodeWorkspace.create(
    episode_storage,
    source_repo=prepared_repo,
    revision=task_commit,
    task_id=task.task_id,
    rollout_id=rollout_id,
) as workspace:
    environment = ProcessSandboxBashEnvironment(ProcessSandbox(workspace.path))
    result = agent.run(
        task,
        environment=environment,
        trajectory_id=trajectory_id,
    )
    model_patch = workspace.export_patch_bytes()
```

## Current limits

- `TaskSpec` must contain already-rendered system and user messages. A frozen
  SWE-bench prompt/config importer is a separate data-source concern and must be
  versioned before comparative experiments.
- Provider adapters and immutable artifact persistence are not implemented in
  this module.
- Observation compaction follows the pinned 10,000-character head/tail policy;
  full output remains available only when the execution backend and artifact
  pipeline retain it.
- The local environment adapter is available now. Docker and Singularity remain
  optional future backends rather than core dependencies.
- Patch verification, baseline/oracle/candidate replay, reward assignment,
  batching, scheduling, and trajectory persistence belong to the outer data
  pipeline.
- `ProcessSandbox` provides process supervision, not a security boundary for
  actively malicious code.
