# QBI ownership lane progress

## State

Complete for the authorized lane. The ownership evidence, realized-regime
persistence, exact-two SSTB role fix, and end-to-end ownership receipt
hardening are implemented. The merge of the `origin/main` snapshot at
`d69131a3` is fully resolved and verified; the combined authority is version
12 and preserves main's nine selective post-transfer calibrations alongside
this lane's complete origin/regime envelope
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1703-1707,4344-4687,5072-5754`).

All eight terminal amount cells have `qrf_transfer` provenance. All four
incidence checks and UBIA QED first turn red in the late transfer; BDC,
REIT/PTP, and W2 QED first turn red on the clone-1 producer. Frozen-donor
replay establishes `zero_inflated_positive` for all four amounts under every
realized availability pattern. Receipt checks persist and validate that result
going forward without claiming that a self-consistent receipt substitutes for
donor replay
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:1507-1767,1857-1972`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-723`).

Exactly the two SSTB booleans compare clone 1; every other physical target
remains clone 0. No amount-model change is included because the host-owned 1%
target × channel × pattern diagnostic/refit run does not yet exist
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2133-2160,3030-3131`).

## Done

- Inspected salvages `03e23e42`, `21f95b71`, `8942ef97`, and newest
  `321b3185`; recovered the useful regime/origin implementation and discarded
  superseded wrappers, stale journals, SHA-skipped output, and conflated
  ownership conclusions.
- Completed `uv sync --all-packages --extra us` with a writable uv cache.
- Authenticated the adjudication's failed publication, gates, three stage
  checkpoints, 13 bank target files, and late-transfer receipt.
- Reproduced all eight QBI amount cells and both SSTB clone-1 comparisons.
  Canonical evidence is byte-stable at SHA-256
  `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`.
- Recomputed all 52 cells (13 chained targets × four patterns) from the exact
  108,073-row frozen donor support; all 16 cells for the four red amounts are
  `zero_inflated_positive`.
- Reran the nine coupled invariants at transferred and terminal stages; all
  nine terminal counts are zero
  (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1377-1487`).
- Persisted full target origin/model-target/pattern/regime evidence through
  monolithic, banked, group, aggregate, signed, checkpoint, and ready-H5
  boundaries
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-723`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-4687,5072-5754,8766-9015`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:594-640`).
- Added the required four-target regression: deleting `origin.channel` from
  group, aggregate, or signed execution copies is rejected even after all
  enclosing receipts are rehashed
  (`packages/microcosm-build/tests/test_us_stacked_spine.py:6581-6720`).
- Implemented the supported exact-two terminal-role correction and direct
  fail-closed regressions
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2133-2160,3030-3131`).
- Resolved the semantic merge with authority/spec/checkpoint repins, retained
  both receipt systems, and added live producer-count, donor-route, predictor-
  catalog, sibling-catalog, and early-receipt validation
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-4687,5072-5754,5603-5617,8766-9015`).
- Replayed canonical evidence after the merge; zero mismatches and identical
  output digest.
- Focused merged ownership/receipt regression: 20 passed, 278 deselected; the
  two corrected live-count cases passed exactly.
- Preserved the intentional version split after the semantic merge: outer
  checkpoint materializer v11 and embedded stacked authority v12. The
  checkpoint identity regression now pins both values independently
  (`tools/build_us_multispine_pool.py:316-335`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1703-1707`;
  `packages/microcosm-build/tests/test_us_multispine_pool_tool.py:3393-3645`).
- Restored the canonical validation order after merging exact ownership and
  calibration receipts: row-count, target-surface, and calibration-summary
  failures retain their specific diagnostics; the focused eight-case
  compatibility regression passed
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4262-4298,4773-4857`).
- Completed the exhaustive package gate on the resolved tree: frame 294
  passed/36 skipped; fit 93 passed; calibrate 203 passed; data 275 passed/1
  skipped; build 6,323 passed/39 skipped across 6,362 unique tests. The exact
  package-wide build command passed 6,321 and skipped 39 before two trade-entry
  subprocesses hit unchanged 300-second limits under accumulated host load;
  the unchanged complete trade-entry file then passed 13/13 fresh, and the
  complete IMDB bulk file also passed in its fresh foreground run.
- Repository-wide `ruff check .`, targeted formatting checks, both cached and
  unstaged `git diff --check`, and spec-engine coverage (41,471/41,471 fields;
  40/40 inventory checks) pass.

## Next

1. Host handoff: instrument and compare the frozen 1% baseline, demonstrate a
   coupled amount refit with whole-pool reconciliation, and run 25%
   certification only after the 1% structural candidate clears the eight
   amount checks and nine coupled invariants.

No pool build, push, publication, exclusion, threshold tuning, or logbook-chain
operation has occurred in this lane.
