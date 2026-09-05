# US relationship/eligibility pre-clone source graph adapters

Branch `us-person-source-graph-20260905`, base `ebb47671`.

## State

Reading the reviewed patterns (`graph_prior_year.py`,
`graph_early_preparation.py`, the registered `multispine_pool` observation
helpers) before any edit. Nothing implemented yet.

## Done

- Read AGENTS/CLAUDE/DESIGN and `docs/us-production-graph-development.md`.
- Read `graph_prior_year.py`, `graph_early_preparation.py`,
  `relationship_inputs.py`, `eligibility_inputs.py`,
  `_person_signal_summary.py`, and the pool observation helpers.

## Next

1. Capture the pre-edit direct-chain payload/receipt/context oracle.
2. Add registered `multispine_pool` relationship/eligibility person observers.
3. Add `graph_person_preparation.py` boundary + derive adapters.
4. Tests, docs, changelog, inventory, `--verify`, blindness fence.

## Limits (standing)

- No push/deploy/calibrate; no held-out reads; no acceptance-threshold change.
- Housing is authored in a separate worktree: not edited, not awaited.
- Nominal engineering parity only; no source-year price alignment here.
