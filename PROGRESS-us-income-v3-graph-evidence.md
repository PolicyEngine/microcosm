# us-income-v3 graph-evidence export — PROGRESS

Owner: independent worker (bounded assignment, no subagents).
R = /Users/maxghenis/architecture-reviews/microcosm-20260904
Branch: us-income-v3-graph-evidence-20260906, dependent child of
I = _worktrees/microcosm-us-launch-current-main-20260905 @ 9ece4cadddd1933524231a46a959086b0ff74429
(I inspected clean before branching; I is not modified by this lane.)

## State
- [x] Step 0: inspect I status/branch, fetch origin, create fresh dependent worktree at 9ece
- [x] Step 1: verify the three pinned R SHAs (runner-v2 / roster-v2 / root approval) match the brief
- [x] Step 2: read the existing 1200-line exporter, its tests, docs and root review/integration receipt
- [x] Step 3: read runner-v2 + roster-v2 + the five-node graph source at 9ece for actual profile facts
- [ ] Step 4: income-v3 profile adapter reusing the reviewed helpers
- [ ] Step 5: invented five-node fixtures + refusal tests; old export tests stay green
- [ ] Step 6: dashboard-reader byte parse check
- [ ] Step 7: R report .md/.json + capture

## Facts established by reading (not assumed)
- Pins verified now: runner-v2 10bed5d8…4ea, roster-v2 fed9afc2…292, approval 303e1895…622.
- Existing exporter/test/doc bytes at 9ece equal the integration-receipt hashes
  (970d9554…55d, a7fb1fca…cf3, 9ceae447…071), so the old profile is unmodified here.
- New schemas: run microcosm.income_v3_first_genuine_graph_run.v1,
  worker microcosm.income_v3_first_genuine_graph_worker.v1.
- Five nodes: create, select_households, cps_carried_current, engine_current_money,
  reported_income; every kernel ref is @3.
- CREATE artifact outputs are five (adds housing_universe, income_observations);
  reported_income outputs are (reported_income, frame_context).
- Cold calls: len 6, prefix [create, source_preparation, select_households],
  set(calls[3:]) == {leaves, engine, reported}, leaves before engine. Warm calls [].
- Direct/graph accounting headers differ only in `inputs` (lineage), never in body or columns.

## Next
- Steps 4-7.
