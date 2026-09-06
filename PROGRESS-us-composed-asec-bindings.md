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
