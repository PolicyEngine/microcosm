# Lane journal: the US native build's byte-transport ceilings

Branch `native-byte-transports`, from `native-row-ceilings` (PR #949's head,
`c5ac78d2d`). Worktree `~/PolicyEngine/_worktrees/microcosm-native-byte-transports`.
Opened 2026-09-18. Session-handoff journal: accurate when written, historical
afterward (see `CLAUDE.md`, "Root journals are history, not state").

Design authority once written: `docs/us-native-byte-transports.md`. Lane report:
`experiments/native-byte-transports/out.md` (root `out.md` is another lane's
tracked report and is not touched).

## State

Orientation done at `c5ac78d2d`: both prior reports read (#949 row ceilings,
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

## Next

1. Byte census: measurement script over the recovered 1/1000 artifact and the
   captured public ACS archive; agent fan-out per module family with an
   adversarial pass over every binding verdict; `census.json`.
2. Commit 1: the ACS serialno and body budgets.
3. The preparation consumer, the origin budget, the numeric bounds, the age
   artifact, the diagnostic matrix; catch-alls carry their cause; pins.
4. Tests as CI runs them; the 1/15 run; the design note; the report; draft PR
   against `native-row-ceilings`.
