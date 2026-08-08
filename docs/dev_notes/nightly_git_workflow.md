# LMFlow Nightly Git workflow

## Repository role

`LMFlow-nightly` is an independent, full-history integration repository for fast LMFlow development. It keeps upstream history intact while allowing product work to move at a cadence independent of upstream pull-request latency.

The repository has three long-lived branches:

| Branch | Responsibility | Accepted changes |
| --- | --- | --- |
| `main` | Exact upstream mirror | Fast-forward synchronization from `OptimalScale/LMFlow` only |
| `nightly` | Default product integration branch | Reviewed product, data, Agentic, CI, documentation, and packaging pull requests |
| `tooling` | Private operational layer | Private bootstrap, preflight, topology, smoke, and maintainer tooling |

`main` must never receive a bulk merge from `nightly`. Upstream publication happens through module-scoped export branches and upstream pull requests.

`agentic/integration` is a temporary compatibility branch. Keep it frozen until all active Agentic work has been retargeted to `nightly`, then retire its ruleset and branch in a separately reviewed cleanup.

## Short-lived branch names

- `core/<topic>`: reusable runtime and framework changes.
- `data/<topic>`: dataset, DataProto, batching, and protocol work.
- `agentic/<topic>`: Agentic SFT/RL product development.
- `ci/<topic>`: CI/CD, packaging, release, and repository automation.
- `docs/<topic>`: documentation-only work.
- `tooling/<topic>`: private tooling based on `tooling`.
- `sync/upstream-YYYYMMDD`: reviewed upstream synchronization.
- `release/YYYY.MM`: release freeze and modular What's New review.
- `export/<module>/<topic>`: clean upstream candidate based on `main`.
- `incubator/<topic>`: explicitly time-limited experiments.

Short-lived branches are deleted after merge or export. A domain prefix identifies ownership and review scope; it does not create another long-lived integration branch.

## Development flow

Product work branches from `nightly` and returns through a pull request to `nightly`. A broadly reusable fix that is already upstream-ready may branch from `main`, but it must also be integrated into `nightly` promptly so the product trunk contains the fix.

Private operational work branches from `tooling` and returns to `tooling`. Product changes flow from `nightly` into `tooling`; private tooling changes never flow back into `nightly` or `main`.

Examples:

- DataProto chunk maintenance: `data/dataproto-chunk-fix` to `nightly`, with `upstream: candidate`.
- Long-running CI maintenance: `ci/<topic>` to `nightly`; private runner helpers use `tooling/<topic>` to `tooling`.
- Agentic RL development: short-lived `agentic/<topic>` branches to `nightly`.

## Modular review and What's New

Every significant pull request adds `.changes/<module>/<slug>.json`. The fragment records:

- the owning module;
- feature, fix, maintenance, or documentation type;
- upstream disposition (`candidate`, `private`, or `blocked`);
- a concise outcome and explicit review focus;
- whether the change is breaking.

The CI workflow validates all active fragments. A `release/YYYY.MM` branch generates `docs/releases/YYYY.MM.md` with:

```bash
python .github/scripts/change_notes.py build \
  --version YYYY.MM \
  --output docs/releases/YYYY.MM.md
```

Reviewers approve the generated document module by module. Candidate entries become independent export branches; private and blocked entries remain visible so exclusions are deliberate and auditable.

## Upstream publication

For each approved candidate module:

1. Update private `main` from upstream by fast-forward synchronization.
2. Create `export/<module>/<topic>` from `main`.
3. Cherry-pick or rebuild only the coherent public commits for that module.
4. Exclude private triggers, credentials, topology assumptions, and `tooling` content.
5. Test the export branch independently and open an upstream pull request.
6. After upstream merge, fast-forward private `main`, then merge or rebase that upstream state into `nightly` and `tooling` through their normal review paths.

This process gives upstream reviewers focused pull requests and preserves an explicit record of everything intentionally withheld.

## Protection policy

- `main`: explicit mirror-only ruleset; deletion, history rewrites, and ordinary updates are blocked.
- `nightly`: pull request, resolved conversations, up-to-date branch, and required CI checks.
- `tooling`: protect after its CI trigger is present; use the same pull-request and required-check baseline unless a private tool needs an explicitly documented exception.

Repository settings, ruleset changes, default-branch changes, and branch retirement are performed as separately reviewed administrative operations.
