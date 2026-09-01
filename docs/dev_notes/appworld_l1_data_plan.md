# AppWorld L1 Train difficulty-1/2 data plan

Status: offline plan only. Provider execution remains disabled until a separate
paid-API run is frozen and handed off. This slice does not run Dev9, K-group
scans, SFT, GRPO, GPU work, or a new teacher request.

## Coverage identity

The target is every official AppWorld Train task at difficulty 1 or 2 in the
pinned revision `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`:

- 24 scenarios and 72 tasks;
- official difficulty order followed by official scenario and task order;
- ordered task-ID SHA-256
  `4b6e14cfe98be52dd1844cbed5fc4ad609da21b32c9db198bbd078137bcb69a9`.

The sealed L0 aggregate already covers four scenarios and 12 tasks. All 24
canonical L0 trajectories remain immutable inherited evidence; no L0 slot is
regenerated. The other 20 scenarios contain 60 tasks, with ordered task-ID
SHA-256
`bba246e4b220ada3ac0202ea2822511afd370c5a595e7fe442f276d46ae47b6d`.

## Generation schedule

Each of the 60 expansion tasks has two predeclared logical candidate slots.
Seed 100 is the primary slot and is the only initially scheduled provider call.
Seed 101 becomes eligible only after the primary yields class C/D/E, an invalid
provider contract, replay mismatch, collateral-state failure, or no canonical
admission. A primary class A or B result meets task coverage and disables the
fallback. Class B does not automatically buy an additional attempt to seek
class A.

Pre-request infrastructure recovery may reuse the same logical slot when no
response exists. Once a provider response has been persisted, the slot cannot
be silently re-executed. A fallback is a distinct predeclared identity. The
initial paid-call count is therefore 60 and the absolute fallback ceiling is 60
more; the actual price, amount cap and wall-time gates are TBD-BLOCKING before
paid authorization.

The teacher and sampling profile remains `ZHIPU/GLM-5.3-Flash`, model-contract
revision `provider-model-id-contract-2026-09-01`, temperature 0.2, at most
3,000 completion tokens, at most 50 steps, provider thinking enabled with
`reasoning_effort=max`, and no transparent API retry. Every candidate requires
fresh execution, fresh replay, isolated official evaluation, collateral-state
validation and hidden-material exclusion before admission.

## Immutable L0 views and long context

The inherited aggregate is accepted only when its strict artifact manifest and
aggregate/candidate/selection/attempt digests match the frozen identities. Each
canonical source record, trajectory, replay, admission and semantic
conversation is revalidated. A source that completed candidate evidence but
failed only while publishing its top-level report is identified by the exact
matching terminal checkpoint recorded in the aggregate; this does not relax
candidate admission.

The inherited views remain separate:

- 22 class-A trajectories and all 24 class-A/B trajectories;
- 21 class-A and 22 class-A/B trajectories fitting 16K in full;
- two over-32K trajectories, kept as controlled-prefix inputs but currently
  excluded from training until an exact prefix/loss-mask recipe is implemented;
- one deterministic task-coverage trajectory per L0 task, ranked class A before
  B and then by fixed seed and candidate identity.

Silent truncation is forbidden. The two over-32K tasks also have shorter
verified candidates, so quarantining the long examples does not reduce the
12-task inherited coverage.

## Selection after expansion

All canonical candidates are retained. The primary 72-task coverage view
selects one trajectory per task by class A before B, then fixed seed and
candidate identity. Class-A-only and class-A/B training views remain separate.
No aggregate candidate is overwritten or deleted by selection.

The separate [L0 paired pre-SFT diagnostic](appworld_l0_paired_presft.md)
freezes the existing teacher reference and two Qwen3-4B student lineages. It is
a Train diagnostic that motivates cold-start SFT; it is not held-out evidence
or a model-selection matrix.
