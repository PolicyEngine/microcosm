# SIPP nominal-equity first fit — progress

Branch `us-sipp-nominal-equity-first-fit-20260906`, dependent child of
`9ece4cadddd1933524231a46a959086b0ff74429`. Local implementation and
invented-fixture verification only. No genuine fit, no genuine source or
projection payload open, no push.

## State
Complete for this assignment. 38 tests pass under the PE-free
`R/constellation-env` interpreter with this worktree's shards loaded
explicitly. Report at `R/us-sipp-nominal-equity-first-fit-REPORT.md`.

## Done
- Read the controlling documents in order: root plan, v2 addendum, v3
  addendum, both Fable 5.1 reviews. v3's column-ownership fix is implemented;
  the compiler confirms `sipp.equity.support` is a real predecessor of DRAW.
- `sipp_equity_contract.py` + packaged `sipp_equity_contract_v1.json`: the
  closed pins, the admission rule, the tiny model configuration, and the
  concept/period refusal. `_contract_from_document` is the private test-only
  construction route.
- `sipp_december_readback.py`: public sibling of the December producer.
  Reuses seven private helpers as an explicit, tested internal dependency;
  never rereads the parent; validates the historical binding by comparison.
- `sipp_equity_graph.py`: prepared read-only source store (two objects,
  `objects/` and empty `tmp/`), the `sipp-december-equity-v1` Frame codec,
  target-specific admission with full coverage retained, exact Decimal
  evidence beside the checked float64 conversion at CREATE, three kernels and
  the graph builder.
- Invented fixtures and three test files; source-blindness and CI-bucket
  registration tests for modules outside `us_runtime`; docs and changelog.
- `sipp_december_source.py` and `sipp_measurements.py` byte-unchanged; their
  SHA-256s equal the accepted receipt's `implementation.module_sha256`.

## Next (not in this assignment)
- Independent review of this code, per v3.
- The separately reviewed genuine rung: real accepted store, packaged
  contract, reproduce 17,588 / 20 / 38 by execution, explicit recipient
  fixture, measured time and RSS in a new private namespace.
- Separate and outstanding: 2024 year bridge, engine mapping, scientific
  validation.
