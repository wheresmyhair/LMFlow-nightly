# LMFlow Nightly Git workflow

## Repository role

`LMFlow-nightly` is an independent, full-history integration repository for fast LMFlow development. It keeps upstream history intact while allowing product work to move at a cadence independent of upstream pull-request latency.

The repository has four long-lived branches:

| Branch | Responsibility | Accepted changes |
| --- | --- | --- |
| `main` | Exact upstream mirror | Fast-forward synchronization from `OptimalScale/LMFlow` only |
| `nightly` | Default product integration branch | Reviewed product, data, CI/CD, documentation, packaging, and mature Agentic promotions |
| `agentic/integration` | Agentic staging branch | Reviewed `agentic/<topic>` feature PRs before milestone promotion to `nightly` |
| `tooling` | Private operational layer | Private bootstrap, preflight, topology, smoke, and maintainer tooling |

`main` must never receive a bulk merge from `nightly`. Upstream publication happens through module-scoped export branches and upstream pull requests.

## Short-lived branch names

- `core/<topic>`: reusable runtime and framework changes.
- `data/<topic>`: dataset, DataProto, batching, and protocol work.
- `agentic/<topic>`: Agentic SFT/RL product development based on `agentic/integration`.
- `ci/<topic>`: CI/CD, packaging, release, and repository automation.
- `docs/<topic>`: documentation-only work.
- `tooling/<topic>`: private tooling based on `tooling`.
- `sync/upstream-YYYYMMDD`: reviewed upstream synchronization.
- `release/YYYY.MM`: release freeze and modular What's New review.
- `export/<module>/<topic>`: clean upstream candidate based on `main`.
- `incubator/<topic>`: explicitly time-limited experiments.

Short-lived branches are deleted after merge or export. A domain prefix identifies ownership and review scope; it does not create another long-lived integration branch.

## Development flow

Core, data, CI/CD, documentation, and packaging work branches from `nightly` and returns through a pull request to `nightly`. A broadly reusable fix that is already upstream-ready may branch from `main`, but it must also be integrated into `nightly` promptly so the product trunk contains the fix.

Agentic feature work branches from `agentic/integration` as `agentic/<topic>` and returns to `agentic/integration` through focused pull requests. Keep the staging branch current by merging `nightly` into it through reviewed sync pull requests. When a coherent Agentic milestone has enough functional completeness and validation, open one promotion pull request from `agentic/integration` to `nightly`. After promotion, synchronize the resulting `nightly` head back into `agentic/integration` before starting the next milestone.

`agentic/integration` contains only Agentic product code, public tests, and product documentation. Private bootstrap, machine-specific configuration, smoke tooling, and internal recipes remain on `tooling` and never enter an Agentic promotion pull request.

Private operational work branches from `tooling` and returns to `tooling`. Product changes flow from `nightly` into `tooling`; private tooling changes never flow back into `nightly` or `main`.

Examples:

- DataProto chunk maintenance: `data/dataproto-chunk-fix` to `nightly`.
- Long-running CI/CD maintenance: `ci/<topic>` directly to `nightly`; private runner helpers use `tooling/<topic>` to `tooling`.
- Agentic RL development: `agentic/<topic>` to `agentic/integration`, followed by milestone promotion from `agentic/integration` to `nightly`.

## Release review and What's New

The pull request and its merge commit are the durable record for ordinary changes. Pull requests do not add separate change-fragment files or repeat module and upstream-publication metadata in their descriptions.

When a release review is needed, create `release/YYYY.MM` from `nightly` and curate `docs/releases/YYYY.MM.md` from the pull requests merged since the previous release. Group the resulting document by reviewer-friendly areas such as Core, Data / protocol, Agentic, CI/CD, Documentation, and Packaging only where that grouping helps review. A pull request may touch several areas and does not need a permanent module label.

The release review identifies the small set of changes worth proposing upstream. Approval is recorded by the reviewed release pull request and the subsequent module-scoped export pull request; changes that remain private or need more work require no metadata in their original development pull requests.

## Upstream publication

For each approved export item:

1. Update private `main` from upstream by fast-forward synchronization.
2. Create `export/<module>/<topic>` from `main`.
3. Cherry-pick or rebuild only the coherent public commits for that module.
4. Exclude private triggers, credentials, topology assumptions, and `tooling` content.
5. Test the export branch independently and open an upstream pull request.
6. After upstream merge, fast-forward private `main`, then propagate that upstream state through reviewed synchronization to `nightly`, `agentic/integration`, and `tooling`.

This process gives upstream reviewers focused pull requests and preserves an explicit record of everything intentionally withheld.

## Protection policy

- `main`: explicit mirror-only ruleset; deletion, history rewrites, and ordinary updates are blocked.
- `nightly`: pull request, resolved conversations, up-to-date branch, and required CI checks.
- `agentic/integration`: the same pull-request and required-check baseline as `nightly`; no direct feature pushes, deletion, or history rewrites.
- `tooling`: use the same pull-request and required-check baseline unless a private tool needs an explicitly documented exception.

Repository settings, ruleset changes, default-branch changes, and branch retirement are performed as separately reviewed administrative operations.
