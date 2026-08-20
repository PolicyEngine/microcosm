# Final report: F1 deliverable 7 compiler-derived lineage surfaces

## Outcome

Deliverable 7 is complete. Every active production-shaped closure, segment,
and lineage-dashboard consumer now projects compiler outputs instead of the
held authored-class fixtures.

The initial retarget landed in `ef036572` (`F1-d7: retarget lineage surfaces
to compiler IR`). A continuation audit in `823a0100` found that whole-column
subtraction incorrectly discarded 20 early-family segments whose columns also
had graph writers, even though their exact compiler atoms were disjoint. The
exact-cell correction landed at `5875be22`. That commit was created by a
concurrent main-lane process while both lanes shared the Git index, so it also
contains unrelated main-lane corrections and does not have the D7 prefix; the
history was left intact and this prefixed handoff records the event.

The existing root `FINAL_REPORT.md` is an unrelated historical journal. This
scoped file is the required final output for the F1 deliverable-7 split-out.

## Retargeted consumers

- `tools/emit_lineage_dashboard.py` compiles the packaged US bundle and reads
  normalized `CompiledSpecIR` resources, typed inventory, the compiler
  producer graph, and both compiler predicate registries.
- `test_imputation_lineage_spec.py` reconstructs the graph, early-family, and
  take-up projections independently from compiler IR and checks the emitted
  dashboard exactly.
- `test_spec_engine_typed_closure.py`,
  `test_spec_engine_take_up_semantics.py`, and `test_us_spec_bundle.py` derive
  their production-shaped closure/segment fixtures from compiler outputs.
- Repository searches find no active test/tool reference to the held
  `us_imputation_lineage.yaml`, `us_pool_column_inventory.json`, authored
  lineage-class helpers, or their former closure fields. Synthetic unit
  mutations and explicit generation-0 parity oracles remain intentionally
  authored; they are not production-shaped closure consumers.
- The held 392-column f025 inventory was not revived. The approved RFC records
  that it predates predictor work and omits 56 later columns, so it cannot
  represent the current compiler closure.

## Exact compiler projection

The fail-closed dashboard now reports:

- 173 typed column contracts;
- 38 producer nodes and 227 producer output/write-scope occurrences;
- 241 lossless raw graph write segments;
- 170 final graph-owner segments covering 763 exact graph cells;
- 152 typed graph segments across 134 columns and 735 cells;
- 48 early-family segments across 48 columns and 48 cells;
- 14 take-up leaves across 13 columns and 26 cells; and
- 214 combined authority segments covering 809 unique
  `(predicate_space, column_key, atom)` cells and all 173 contracts.

Authority is cell-exact within each compiler predicate space. The 20 columns
shared by graph and early-family surfaces have disjoint atoms, so both
authorities are retained. Take-up uses its own compiler predicate space; no
compiler-sealed mapping authorizes it to erase a producer or family segment.
The column-surface distribution is 113 graph-only, 28 family-only, 11
take-up-only, 19 graph+family, one graph+take-up (Medicare), and one with all
three surfaces (Housing).

For graph cells, non-matrix predicates expand to exact atoms and resolve to a
unique DAG-maximal writer. Matrix cells must match the compiler's declared
`final_owner`/`owns_final` relation. The implementation rejects ambiguous
owners, graph coverage gaps, raw/final graph union mismatches, unknown typed
contracts, empty or duplicate atoms, peer-surface exact-cell collisions, and
missing typed-column closure.

## Verification

- All four D7 modules passed after the correction: **70 tests passed** in
  407.29 seconds.
- The exact dashboard projection regression passed alone in 85.94 seconds.
- A direct dashboard emission completed in 43.10 seconds and reported 183
  variables, 33 families, and compiler spec SHA-256
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`.
- Ruff check, Ruff format check, Python bytecode compilation, and `git diff
  --check` passed for the scoped files.
- An independent audit rebuilt the projection directly from `CompiledSpecIR`
  without importing the dashboard or its helpers and reproduced every count,
  the 809/809 unique cells, the complete 173-contract union, and the 20
  disjoint graph/family overlaps.
- A second independent review returned `APPROVE` and confirmed byte-identical
  output from separate processes with `PYTHONHASHSEED=1` and `8675309`
  (output SHA-256
  `03c08eb0e59b4ce2fa0d2ffe2bcf62e503005569b1171520a46068dec99ef7df`).

The prior main-lane seed-attestation mismatch is not a D7 blocker: concurrent
commit `5875be22` updated that owner-maintained pin to the current 213
production modules and 285 classified calls. This split-out does not claim
the main lane's repository-wide or certification gates.

## Commits and operating limits

- `9ed8f23d` — `F1-d7: start deliverable progress journal`
- `247decae` — `F1-d7: record compiler consumer inventory`
- `ef036572` — `F1-d7: retarget lineage surfaces to compiler IR`
- `aac2fe03` — `F1-d7: finalize compiler lineage handoff`
- `823a0100` — `F1-d7: reopen cell-scope lineage closure`
- `5875be22` — concurrent shared-index commit containing the exact-cell D7
  correction and main-lane F0 work
- the commit containing this report — prefixed final exact-cell handoff

No push, pool build, sample rung, restricted-data access, or publication was
performed. Validation ran sequentially within the split-out's 15 GiB RSS
ceiling.
