# Agentic episode workspace

`EpisodeWorkspace` owns one repository checkout for one rollout attempt. A task
may produce multiple rollouts, and every rollout receives a different checkout
even when the task and base revision are identical. This prevents one candidate
trajectory from changing the files observed by another candidate.

The initial lifecycle is deliberately small:

```text
create -> execute or reset -> export_patch -> cleanup
```

- `create()` resolves an exact base commit in a local source repository and
  creates a uniquely named checkout below a caller-owned storage directory.
- `reset()` builds a fresh checkout and then swaps it into place. This restores
  tracked files, removes untracked/ignored files, and recovers even if the
  previous checkout's Git metadata was deleted. A preparation failure leaves
  the previous checkout unchanged.
- `export_patch()` includes tracked changes, deletions, non-ignored new files,
  executable mode changes, and Git-encoded binary changes without removing the
  edits from the checkout. `export_patch_bytes()` preserves a patch containing
  non-UTF-8 text verbatim.
- `cleanup()` removes only the generated episode directory and is idempotent.
  The context-manager interface performs the same cleanup on normal return or
  exception.

Task and rollout identifiers are stored on the Python object but never copied
into path components. The directory name uses a digest plus a random suffix,
so identifiers containing separators or shell characters cannot redirect
workspace creation or cleanup.

## Connection to process execution

The checkout path can be passed directly to `ProcessSandbox`:

```python
with EpisodeWorkspace.create(
    episode_storage,
    source_repo=prepared_repo,
    revision=task_commit,
    task_id=task_id,
    rollout_id=rollout_id,
) as workspace:
    sandbox = ProcessSandbox(workspace.path)
    result = sandbox.run(["python", "-m", "pytest", "-q"])
    model_patch = workspace.export_patch()
```

`EpisodeWorkspace` controls repository state, while `ProcessSandbox` controls
the lifetime and resources of each command. Neither component currently
provides filesystem or network isolation for actively malicious code.

## Current limits

The checkout uses a local shared Git object store to avoid copying repository
history for every rollout. A caller-managed source repository must remain
available and its reachable objects must not be pruned while episodes are
active. `PreparedRepositoryCache` can provide that stable node-local source and
acquire an exact remote commit before rollout dispatch. The episode workspace
itself remains a checkout lifecycle rather than a cache or download manager.

This first slice does not implement dependency-environment caching, leases,
heartbeats, crash reaping, overlay/reflink snapshots, resource telemetry, or a
Bubblewrap/Linux-native sandbox. Those capabilities can be added around the
same lifecycle without changing the task, rollout, or patch semantics.
