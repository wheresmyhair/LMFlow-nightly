# Agentic prepared repository cache

`PreparedRepositoryCache` turns a repository source and an exact 40-character
Git commit into a reusable node-local bare repository. The returned path can be
passed directly to `EpisodeWorkspace.create()` as `source_repo`, so multiple
rollouts reuse fetched Git objects while retaining independent working trees.

```python
from lmflow.agentic import EpisodeWorkspace, PreparedRepositoryCache

cache = PreparedRepositoryCache("/datasets/lmflow-cache")
prepared_repo = cache.prepare(
    "https://github.com/django/django.git",
    "0123456789abcdef0123456789abcdef01234567",
)

with EpisodeWorkspace.create(
    "/datasets/episode-workspaces",
    source_repo=prepared_repo,
    revision="0123456789abcdef0123456789abcdef01234567",
    task_id="django__django-00001",
    rollout_id="rollout-0001",
) as workspace:
    ...
```

## Publication and concurrency

The cache key includes the cache format, normalized repository source, and
exact commit. A builder initializes a bare repository in a temporary directory,
fetches only the requested commit, pins it at `refs/lmflow/base`, writes a
`READY` marker, and atomically renames the completed entry into place. A
per-key POSIX advisory lock serializes processes preparing the same entry.
Failures remove their current staging directory and never publish a partial
entry. Published entries with a missing marker, unsafe symlink, or mismatched
commit fail closed; the cache does not silently overwrite them.

Consumers must treat the returned bare repository as immutable. The original
repository source can disappear after publication because fetched objects are
stored in the cache entry rather than referenced through Git alternates.

## Repository transports

The initial boundary accepts public HTTPS URLs, `file://` URLs, and local
filesystem paths. HTTPS credentials, URL query strings, fragments, SCP-style
SSH URLs, and interactive Git prompts are rejected. Standard HTTP(S) proxy
environment variables are forwarded to Git without being stored in the cache
metadata. The `READY` marker stores only a repository-source digest and the
commit.

## Current limits

Each repository/commit pair is a separate immutable entry. This favors simple
correctness and atomic publication over cross-commit object deduplication.
Dependency environments, Git LFS objects, submodule initialization, eviction,
leases, crash reaping, distributed cache coherence, weighted I/O admission,
and remote artifact storage remain orchestration responsibilities. A process
terminated during preparation may leave an unpublished staging directory;
later reaping can be added without changing published entry semantics.
