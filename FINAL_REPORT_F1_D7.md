# Final report: F1 deliverable 7 compiler-derived lineage surfaces

## Outcome

Deliverable 7 is complete at commit `ef036572` (`F1-d7: retarget lineage
surfaces to compiler IR`). The lineage dashboard and every active
production-shaped closure/segment consumer now compile the packaged bundle and
read `CompiledSpecIR` outputs instead of reconstructing held authored-class
fixtures.

The existing root `FINAL_REPORT.md` is an unrelated historical journal. This
scoped file is the final output for the F1 deliverable-7 split-out and leaves
that prior report unchanged.

## Retargeted surfaces

- `tools/emit_lineage_dashboard.py` now uses `compile_spec(load_bundle(...))`,
  normalized `CompiledSpecIR.resource()` documents, `typed_inventory`, and the
  compiler producer graph.
- `test_imputation_lineage_spec.py` validates the emitted dashboard as an
  exact, deterministic JSON projection of the compiler IR.
- `test_spec_engine_typed_closure.py`,
  `test_spec_engine_take_up_semantics.py`, and `test_us_spec_bundle.py` derive
  their production-shaped closure and segment fixtures from compiled IR.
  Synthetic low-level mutation tests and explicit generation-0 parity oracles
  remain intentionally authored.
- No active consumer imports the held 392-column f025 inventory. That artifact
  predates later predictor work and is documented as missing 56 columns, so it
  is not represented as current compiler closure.

## Exact compiler projection

The dashboard reports these fail-closed surfaces:

- 173 typed column contracts;
- 38 producer nodes and 227 producer output/write-scope occurrences;
- 241 lossless raw write-event segments;
- 170 final graph-owner segments covering 763 exact cells;
- 150 graph authority segments across 132 typed columns after Medicare and
  Housing defer to take-up;
- 28 family-only authority segments across 28 typed columns;
- 14 take-up ownership leaves across 13 typed columns; and
- 192 exclusive authority variants covering all 173 typed columns exactly
  once by authority surface.

For non-matrix graph cells, the projection expands compiler predicates to
exact atoms and selects the unique DAG-maximal writer. Matrix cells must match
the compiler's `final_owner`/`owns_final` declaration. The implementation
rejects ambiguous owners, overlapping final cells, coverage gaps, and any
union mismatch with the raw compiler write surface. Family and take-up
variants remain distinct because those compiler authorities do not expose a
producer write policy; the dashboard does not invent one.

## Verification

- Four D7 modules passed before the implementation commit in 661.33 seconds.
- The exact dashboard semantic test independently proved the final-owner and
  authority partitions in 345.36 seconds.
- The same four modules passed on merged compiler-schema HEAD `da45dfcd` in
  426.33 seconds.
- Ruff check and Ruff format check passed on all five D7 Python files; `git
  diff --check` passed; emitted JSON has a deterministic encode/decode
  round-trip assertion.
- An independent adversarial review returned `APPROVE` with no blocking
  findings and separately verified the 241 raw writes, 170 graph-final
  segments, 763 exact cells, 173-column partition, and 14 take-up leaves.

The broader pre-merge `test_spec_engine_*.py` run completed in 2,559.03
seconds with every case passing except the main-lane seed-attestation count:
the test expected 208 production modules while the tree exposed 212. After the
compiler-schema merge, the isolated test still expects 208 while observing
213 (56.46 seconds). Deliverable 7 adds no production modules and deliberately
does not modify that deliverables-5/6/8-owned attestation. A repository-wide
green claim is therefore not made in this split-out report; the D7-owned suite
is green.

## Commits and operating limits

- `9ed8f23d` — `F1-d7: start deliverable progress journal`
- `247decae` — `F1-d7: record compiler consumer inventory`
- `ef036572` — `F1-d7: retarget lineage surfaces to compiler IR`

No push, pool build, sample rung, restricted-data access, or publication was
performed. Validation ran sequentially and stayed within the split-out's
15 GiB RSS ceiling.
