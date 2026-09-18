# Lane journal: the US native build's byte-transport ceilings

Branch `native-byte-transports`, from `native-row-ceilings` (PR #949's head,
`c5ac78d2d`). Worktree `~/PolicyEngine/_worktrees/microcosm-native-byte-transports`.
Opened 2026-09-18. Session-handoff journal: accurate when written, historical
afterward (see `CLAUDE.md`, "Root journals are history, not state").

Design authority once written: `docs/us-native-byte-transports.md`. Lane report:
`experiments/native-byte-transports/out.md` (root `out.md` is another lane's
tracked report and is not touched).

## State (third session, 2026-09-18)

The moved base (`origin/native-row-ceilings` at `0319b1c73`, merged level with
main) is merged in at `77bafbb3a`; the venv is resynced to the merged lock
(policyengine-us 2.2.1). The second session's uncommitted edits were committed
unchanged (`2ed7b0715`) and then tested: two of its new tests failed, both
fixed in the commit after this journal entry -- the digest and segmented
encoders no longer require the segment to be at most the maximum (a digest has
no accumulation), and the budget's `checked_view` no longer re-encodes the
whole document through the 64 MiB encoder (`graph._json_matches` streams the
comparison). Pins re-derived after the merge: 124 inventory contracts and all
four `_ACCEPTED` entries unchanged, ten stage manifests build. A census
workflow (nine module families, adversarial verification, completeness critic)
is running read-only over this head; no source file moves until it returns.

Earlier state, first session, at `c5ac78d2d`: both prior reports read (#949 row ceilings,
#950 retention seal §8/§11), the transport design note, the prior census
(146 bounds, 91 byte-shaped rows), and every module that holds a named byte
bound on the 19-node and 45-node paths. `uv sync --all-packages --locked
--extra us` built `.venv` (Python 3.14.4; `uv.lock` identical to the
`microcosm-native-v5` venv the prior harnesses ran on). Baseline gates at the
untouched head: `tools/ci_test_groups.py --verify` ok; spec-engine coverage
42156/42156 and 41/41.

## Done

- Read at this head: `acs_person_coverage_authentication.py` (whole),
  `acs_native_coverage_binding.py` (whole), the roster transport in
  `survey_population_preparation.py` (`_roster_segments`, `_spill_roster`,
  `_roster_payload`, the three `raise ... from None` sites), `_bounded_json`
  and `_allocation_payload` in `graph_survey_population.py`,
  `survey_origin_budget._document`, `graph_survey_budget`,
  `graph_survey_calibration`, `graph_survey_age_artifact`,
  `puf_diagnostic_consumer.current_survey_recipient_matrix`, the housing
  prepared receipt, the child-property projections, and the fit adapter's
  draw document.
- Established reachability: the 45-node pilot ran with household roles and
  person status disabled (`_roles_disabled_pair`; pilot report §"The 45-node
  graph, derived from the code"), so the roles projection and status artifact
  are reachable only from the 49/51-node variants. The origin budget, numeric
  bounds and age artifact run in `survey_age_calibration`, not in either
  graph, but the origin budget is one of the three the brief names, and the
  numeric bounds consume its payload.

## Done (third session)

- Merged the moved base; resynced; committed and tested the second session's
  edits (ACS 116 passed with the row ceilings; child property 61; graph survey
  population 70 after the fix; origin budget 54 after the fix; spine blindness
  503; preparation 75; calibration and age 67; fit adapter, child source and
  predictors 95).
- Found and closed a consumer gap the second session missed: the budget's
  `checked_view` re-encoded the whole document through `_bounded_json`
  (`survey_origin_budget.py`, `FINAL_BUDGET_VIEW_DOCUMENT`), which would refuse
  at the same 5.7% of source the producer used to.
- Wrote the census brief (`experiments/native-byte-transports/census-brief.md`)
  and the seeded agreement sweep (`agreement_sweep.py`).

## Next

0. Act on the census: any bound it finds binding that this branch has not
   lifted; then pins, the design note, the report, the 1/15 run from a
   detached worktree at the finished head, the draft PR.
1. (first session's plan) Byte census: measurement script over the recovered 1/1000 artifact and the
   captured public ACS archive; agent fan-out per module family with an
   adversarial pass over every binding verdict; `census.json`.
2. Commit 1: the ACS serialno and body budgets.
3. The preparation consumer, the origin budget, the numeric bounds, the age
   artifact, the diagnostic matrix; catch-alls carry their cause; pins.
4. Tests as CI runs them; the 1/15 run; the design note; the report; draft PR
   against `native-row-ceilings`.
