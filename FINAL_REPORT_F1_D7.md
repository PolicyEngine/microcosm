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

- The four D7 modules passed together: **70 passed in 420.42 seconds**.
- Two independent read-only audits approved the compiler boundary and exact-
  cell logic. One rebuilt all 809 cells without importing the dashboard.
- Separate processes with `PYTHONHASHSEED=1` and `8675309` emitted byte-
  identical JSON with SHA-256
  `03c08eb0e59b4ce2fa0d2ffe2bcf62e503005569b1171520a46068dec99ef7df`.
- Scoped Ruff check, Ruff format check, and D7 whitespace checks passed.
- Serial all-shard verification passed calibrate, data, fit, and frame with
  865 passed and 37 skipped. The build shard reported 6,322 passed, 37
  skipped, and three `test_us_spine_blindness.py` failures because pytest had
  collected the file before concurrent main-lane commit `12df8c45` classified
  six new runtime-authority modules and updated the 65-to-70 graph pin. At
  current HEAD, those exact three tests pass (3 passed in 4.42 seconds), and
  the main lane records the complete 495-test module passing. This is
  decomposed green evidence, not a claimed clean one-shot rerun at the new
  HEAD.
- Repository-wide Ruff reports one unrelated main-lane import-order finding
  at `tools/build_us_multispine_pool.py:1157`; D7's files are clean.

## Commit provenance and limits

Prefixed D7 commits are `9ed8f23d`, `247decae`, `ef036572`, `aac2fe03`,
`823a0100`, and `5228cf5c`. Shared unprefixed `a7a6c06d` updated journals;
mixed unprefixed `5875be22` contains the exact-cell D7 correction plus
main-lane F0 work. Those are historical prefix exceptions, recorded rather
than rewritten.

No push, pool build, sample rung, restricted-data access, or publication was
performed. Validation was serialized. The sandbox blocks both `ps` and
`/usr/bin/time -l`, so this report does not fabricate an exact RSS peak; no
resource failure or heavy build workload occurred.
