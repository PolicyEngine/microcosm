# us-composed-asec-bindings — PROGRESS (final)

## State

**Complete and committed.** Worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-composed-asec-bindings-20260906`,
branch `us-composed-asec-bindings-20260906`, base `origin/main` c1b83241,
fast-forwarded to the frozen parent ab86b5d2. The head is the commit carrying
this file; the report outside the repo records the exact SHA. Nothing was
pushed, merged, published or run on genuine sources.

See `us-composed-asec-bindings-REPORT.md` for the full report.

## Done

- [x] child worktree from fetched origin/main, ff-only to ab86b5d2, own `.venv`
      (`uv sync --all-packages --extra us`); fresh subprocesses resolve this child
- [x] read AGENTS/CLAUDE, `docs/us-composed-population-graph.md`,
      `graph_composed_population.py`, `graph_asec_prepared.py`,
      `graph_asec_income.py` and every helper they use
- [x] reproduced, on the real invented composed fixture, why the accepted
      prepared kernels refuse (`PREPARED_CONTEXT_METADATA` first, three more
      layers behind it) — kept as tests, none weakened
- [x] `graph_composed_asec_binding.py`: provenance-owner bind node; typed
      `asec_binding` document, framed `arm_rows` buffers, reviewed
      `selected_current_money`, three boolean arm cells
- [x] `graph_composed_asec_measures.py`: source-blind corrected-leaf and
      reported-income kernels writing only under the declared arm mask
- [x] new reviewed stage `composed_asec_binding_v1` + deliberate inventory
      extension (2 modules, 2 classifications, zero existing contract moved)
- [x] runtime classification: binding module as a provenance owner, measures
      module as an ordinary source-blind module
- [x] 32 tests on invented sources; docs, changelog fragment, CLAUDE.md pointer
- [x] affected-test run green (681 passed, exit 0), ruff clean, ci_test_groups
      `--verify` ok

## Remaining (for root)

- `test_us_graph_asec_income.py::test_fresh_warm_process_has_equal_artifacts_and_zero_calls`
  fails **at the frozen parent ab86b5d2 too** once the `us` extra is installed
  (reproduced in a temporary parent worktree; log in
  `R/us-composed-asec-bindings-parent-preexisting-failure.log`). Pre-existing,
  out of this lane's scope, deselected in the recorded run.
- No genuine run performed. The proposed genuine-small runner command is in the
  report and in `docs/us-composed-asec-binding.md`, not executed.

## Peer review answered (root's Fable review of 24f32366)

- **code-1 (material) fixed.** The `ASEC_DEMOGRAPHIC_SOURCE_APPEARED` refusal is
  removed. It was keyed on a fixture-only fact, and `graph.py`'s `_ROOT_CELLS`
  declares `A_SEX` as an int64 person cell on the legacy stacked path, so on
  genuine data the whole stage — and therefore the corrected money — would have
  refused at bind time. The binding now records the observed roster truthfully
  with two reasons (`UNBOUND_ABSENT` / `UNBOUND_UNDECIDED`) and `bound_here:
  false`; a new test drives the present-`A_SEX` path.
- **code-2 fixed.** `source_origin` joined `_SOURCE_EDGES`, so
  `NODE_EVIDENCE_PRODUCER` and `ARTIFACT_PRODUCER_SPLIT` now really do cover it,
  matching what the doc claimed.
- **code-3 fixed.** The journal no longer claims a stale phase or head.
- **code-4 fixed.** A malformed `arm_rows` buffer roster refuses with the typed
  `ARM_ROWS_BUFFER_ROSTER` instead of a bare `ValueError`.
- Two recorded mapping names were wrong and are corrected:
  `cps_carried.derive_us_cps_carried_tables` and
  `relationship_inputs.derive_us_relationship_inputs_from_manifest`.
- The review's evidence gaps are closed by execution: the new module runs 33
  tests green, and the affected set runs green end to end.
