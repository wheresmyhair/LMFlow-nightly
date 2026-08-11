# Minimal Bash agent

`MinimalBashAgent` is the model-facing scaffold for a linear repository-agent
rollout. It owns the model/tool conversation, Bash observation formatting,
step limit, submission protocol, and ATIF projection. It does not own repository
preparation, revision selection, reset, patch export, verification, or cleanup.

The intended composition keeps those responsibilities separate:

```text
EpisodeWorkspace (pipeline-controlled repository state)
    -> ProcessSandbox (command execution)
        -> MinimalBashAgent (model-visible Bash loop)
    -> patch export and independent verifier
```

An alternative command backend can implement the narrow `BashCommandExecutor`
protocol and return the same `ProcessResult`. No workspace or Git type enters the
agent interface.

## Model contract

The injected model callable receives independent copies of OpenAI-style
`messages` and `tools`. It returns a normalized `BashAgentTurn`. The initial
implementation requires exactly one `BashToolCall` per model turn and rejects
missing, additional, or duplicate call IDs before executing the invalid call.
`bash_agent_turn_from_openai_message()` provides a strict, dependency-free
normalizer for an OpenAI-compatible assistant message. Provider clients retain
ownership of HTTP, authentication, retry, usage, and raw-response persistence.

The Bash tool definition and successful submission marker match the fixed
mini-swe-agent reference:

```text
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
optional submission text
```

The marker must be the first stdout line of a successful, non-timeout command.
The model-visible observation retains mini-swe-agent's `<returncode>` and
`<output>` structure. `ProcessResult` separately preserves stdout, stderr,
timeout, truncation, duration, and command metadata for the data pipeline.

## Pipeline composition

```python
with EpisodeWorkspace.create(
    episode_storage,
    source_repo=prepared_repo,
    revision=task_commit,
    task_id=task.task_id,
    rollout_id=rollout_id,
) as workspace:
    sandbox = ProcessSandbox(workspace.path)
    result = agent.run(
        task,
        sandbox=sandbox,
        trajectory_id=trajectory_id,
    )
    model_patch = workspace.export_patch_bytes()
```

Every model-generated step is recorded with `llm_call_count: 1`, a structured
Bash tool call, and its matching text observation. The resulting ATIF v1.7
document can be passed directly to `atif_trajectory_to_conversation()` for SFT.
Task, rollout, repository, verifier, and artifact provenance remain owned by the
outer data pipeline rather than the model scaffold.

## Current limits

- The caller supplies already-rendered system and user messages in `TaskSpec`.
- Provider clients, retry policy, malformed-response feedback, token-native
  generation results, and raw provider responses remain adapter concerns.
- Only the built-in Bash tool and one call per model turn are supported.
- A model or executor exception aborts the call; partial artifact persistence is
  left to the data pipeline.
- Patch verification, reward assignment, task-source adapters, batching,
  concurrency, and trajectory persistence are outside this module.
- `ProcessSandbox` provides process supervision, not a security boundary for
  actively malicious code.
