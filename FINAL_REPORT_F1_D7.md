# Final report: F1 deliverable 7 compiler-derived lineage surfaces

## Outcome

Deliverable 7 is complete. Every active production-shaped derived-closure,
segment, and lineage-dashboard consumer now projects compiler outputs instead
of the held authored-class fixtures.

The retarget landed in `ef036572`. A continuation audit found that its
whole-column subtraction discarded 20 early-family segments even though their
compiler atoms were disjoint from graph atoms. Commit `5875be22` contains the
cell-exact correction; a concurrent main-lane process swept those two D7 files
into a mixed, unprefixed commit. Prefixed handoff `5228cf5c` records that
provenance without rewriting shared history.

This file is the requested scoped output. Root `FINAL_REPORT.md` belongs to an
unrelated historical lane.

## Retargeted consumers

- `tools/emit_lineage_dashboard.py` compiles the packaged US bundle and reads
  normalized `CompiledSpecIR` resources, typed inventory, producer graph,
  compiler-expanded outputs/write scopes, and compiler predicate registries.
- `test_imputation_lineage_spec.py` independently reconstructs graph,
  early-family, take-up, and combined exact-cell projections from compiled IR
  and compares them with the emitted dashboard.
- `test_spec_engine_typed_closure.py`,
  `test_spec_engine_take_up_semantics.py`, and `test_us_spec_bundle.py` compile
  the bundle for every production-shaped closure or segment fixture.
- Tracked test/tool searches find no active reference to
  `us_imputation_lineage.yaml`, `us_pool_column_inventory.json`, the former
  closure helper, authored lineage classes, or their old dashboard fields.
  Synthetic compiler tests and explicit generation-0 parity oracles remain by
  design; they do not supply the production-shaped derived surfaces.
- The held 392-column f025 inventory was not revived. It predates predictor
  work and omits 56 later columns; artifact-presence certification remains a
  plan-derived selector concern owned by the main lane.

## Exact compiler projection

The fail-closed projection contains:

- 173 typed column contracts;
- 38 producer nodes and 227 output/write-scope occurrences;
- 241 raw graph write segments;
- 170 final graph-owner segments covering 763 exact graph cells;
- 152 typed graph segments across 134 columns and 735 cells;
- 48 early-family segments across 48 columns and 48 cells;
- 14 take-up leaves across 13 columns and 26 cells; and
- 214 combined authority segments covering 809 unique
  `(predicate_space, column_key, atom)` cells.

Authority is cell-exact within each compiler predicate space. All 20 columns
shared by graph and early-family surfaces have disjoint atoms, so both are
retained. Take-up occupies its own compiler predicate space and cannot erase a
producer or family segment. Surface membership is 113 graph-only columns, 28
family-only, 11 take-up-only, 19 graph+family, one graph+take-up (Medicare),
and one graph+family+take-up (Housing).

Non-matrix graph cells expand to exact atoms and resolve to one DAG-maximal
writer. Matrix cells must match the compiler-declared final-owner relation.
Emission rejects ambiguous owners, coverage gaps, raw/final union mismatches,
unknown contracts, empty or duplicate atoms, exact peer-surface collisions,
and incomplete typed-column closure.

## Current-head verification

- The four D7 modules passed together before the report commit (70 passed in
  420.42 seconds) and again after commit `c0a75253`: **70 passed in 350.29
  seconds**. The post-commit run measured 4,231,233,536 bytes (3.94 GiB) peak
  RSS on Darwin.
- Two independent read-only audits approved the compiler boundary and exact-
  cell logic. One rebuilt all 809 cells without importing the dashboard.
- Separate processes with `PYTHONHASHSEED=1` and `8675309` emitted byte-
  identical JSON with SHA-256
  `03c08eb0e59b4ce2fa0d2ffe2bcf62e503005569b1171520a46068dec99ef7df`.
- Serial calibrate, data, fit, and frame receipts contain 865 passed and 37
  expected skips. A fresh build-shard rerun then passed **6,325 tests with 37
  expected skips** in 6,820.57 seconds. The observed all-shard sum is 7,190
  passed and 74 expected skips; the main lane independently records its clean
  committed-tree full-suite result.
- The build run began at `c0a75253`. During its two-hour execution the shared
  branch advanced through main-lane commits `030c0613`--`4deb8fb8`; those
  changed one import ordering plus journals/docs and no D7 file. This clean
  rerun supersedes the earlier moving-collection three-failure receipt.
- Current-HEAD repository-wide Ruff, scoped D7 format, held-fixture searches,
  and whitespace checks pass. No D7 file changed after `5228cf5c`.

## Commit provenance and limits

Prefixed D7 commits are `9ed8f23d`, `247decae`, `ef036572`, `aac2fe03`,
`823a0100`, `5228cf5c`, and `c0a75253`. Shared unprefixed `a7a6c06d` updated
journals; mixed unprefixed `5875be22` contains the exact-cell D7 correction
plus main-lane F0 work. Those are historical prefix exceptions, recorded
rather than rewritten.

No push, pool build, sample rung, restricted-data access, or publication was
performed. Validation was serialized. A Python `getrusage` wrapper succeeded
where sandboxed `ps` and `/usr/bin/time -l` had failed: the focused D7 run was
3.94 GiB, but the monolithic build-shard test process peaked at
30,950,326,272 bytes (28.82 GiB), exceeding the user's 15 GiB ceiling. This
was an operating-order violation. Heavy testing stopped when the result became
available; this report does not claim resource compliance.
