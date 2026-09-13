# UK-enabling shared graph contracts (amendments 25 and 26)

Lane journal. Append-only within this lane; historicize rather than
overwrite once the branch merges (CLAUDE.md, "Root journals are history").

## State

Bounded, source-only slice extracting the shared graph contract that
María's UK full-build graph (#901) consumes, as numbered amendments on
current `main`. No UK graph stage, calibration science or country
kernel is added here — those stay in #901.

- Worktree: `_worktrees/microcosm-uk-shared-graph-contracts-20260913`
- Branch: `uk-shared-graph-contracts-20260913`
- Base: `origin/main` `15ebde806cd1a262363f7217fe535c7234ff757f`
- Reviewed UK head: #901 `051fb972b19d319d58277bd63306d0d0e0947ce2`
  (draft, base `microcosm-us-launch-integration-20260909`, unchanged
  since 2026-09-10T21:03:10Z; re-verified via `gh` on 2026-09-13)
- Source review followed: `uk-parallel-review.md` (2026-09-12), section
  "Best independent implementation slice"

**Runtime is UNTESTED in this lane.** Instructions forbid pytest,
production imports, engine, native sources and installation here. Only
stdlib `ast`/`ruff`/CI-inventory source checks were run. A finite
invented-only runtime plan is at the end of this file for root review
*before* execution.

## Absence verified on base 15ebde806

| Contract | Present on main? | Evidence |
| --- | --- | --- |
| `WeightUpdate` (same-kind weight replacement) | **absent** | `grep -rni weightupdate .` over the worktree returns nothing; `decl.py:320` carries only `WeightTransition`, whose `__post_init__` requires `to_kind` strictly later in `WEIGHT_KINDS`, and `population.py:1944` rejects a non-forward move. A same-kind update is therefore unrepresentable. |
| `weight_update_receipt` / ordered-axis evidence | **absent** | no `weight_update` module under `packages/microcosm-graph/src/microcosm/graph/`. |
| `KernelContext.frame_metadata` | **absent** | `kernel.py:361-370` lists the complete field set; no metadata field. |
| `KernelContext.frame_mass_log` | **absent** (read side) | same field list. The *write* side already exists: `population._append_frame_mass_log` (`population.py:2117`) already ingests `receipt['frame_mass_log_append']`, so only the kernel's view of the incoming log is missing. |
| `KernelContext.frame_column_order` | **absent** | same field list. The executor projects `tables[entity]` in declaration order (`executor.py:600-628`), not the population version's own column order, so a consumer cannot reconstruct the frame layout. |

Frame-side prerequisites already on main: `Frame.mass_log`,
`Frame.metadata`, `MassChangeRecord` and `_freeze_metadata`
(`packages/microcosm-frame/src/microcosm/frame/bundle.py`).

Interface lock on base matches the files exactly:
`decl.py ed0a859adcae12510d5ba74d51c694617201f7b448b108a3f602410f5da44876`,
`kernel.py dbf57c137330f0f12744c557ee594586a1b308b6d6adaba1938e2b6efded21ca`.

## Actual consumer read before specifying shape

`packages/microcosm-build/src/microcosm/build/uk_runtime/graph_population.py`
at `051fb972` (SHA-256 `fc6f5b33127020f6e0529b39304715fe2028d47a6627b152b1e52e0d69f2efdc`):

- `context_frame` (L58-80) reads `context.frame_column_order.get(entity, ...)`,
  `context.weights`, `context.strata`, `getattr(context, "frame_mass_log", ())`
  and `getattr(context, "frame_metadata", {})`.
- `UKSampleNormalizationKernel` (L386-412 region) declares
  `WeightUpdate("household", weight_kind, "Normalize sampled source-family mass.")`
  with `mass="declared"` and places `weight_update_receipt(ids)` under
  `receipt["weight_update"]`.

## Done

- (nothing yet; baseline recorded)

## Next

- Amendment 25: `WeightUpdate` + ordered-axis receipt.
- Amendment 26: `KernelContext` frame metadata / mass log / column order.
