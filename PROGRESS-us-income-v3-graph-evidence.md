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
- [x] Step 4: income-v3 profile adapter reusing the reviewed helpers
- [x] Step 5: invented five-node fixtures + refusal tests; old export tests stay green
- [x] Step 6: dashboard-reader byte parse check
- [x] Step 7: R report .md/.json + capture

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

## Done (later steps)
- Step 4: graph_evidence_income_v3.py. Loads the reviewed exporter by file path and
  reuses its helpers, writer loading and aggregate schema; the reviewed files stay
  byte-identical to the integration receipt. No default pins: freeze_income_v3_pins
  requires an accepted root output review and refuses without one.
- Step 5: test_graph_evidence_income_v3.py, invented five-node fixtures only.
  166 passed (59 old + 107 new); ruff clean; registered in tools/ci_test_groups.py.
- Step 6: the existing dashboard reader refuses only the new annotation fields
  (shape() enforces an exact key count) and parses everything else: two executions,
  five nodes, six binding checks. Three shapes to extend, listed in the report.
  That worktree was not modified.
- Extra: a read-only structural probe ran the profile's validators against the
  completed genuine run's aggregate receipts, 20/20, creating no candidate and
  hashing no opaque output. It caught two real contract errors: the carried
  housing-universe cells are a twelve entry digest map (plus a verified_runtime
  field on worker receipts), and the graph engine evaluation bytes legitimately
  differ from the direct ones. Both are fixed and covered by tests.
- Step 7: R/us-income-v3-graph-evidence-REPORT.md/.json and the separate
  us-income-v3-graph-evidence-capture.md.

## Next
- NONE here. No genuine export exists: the root output review is absent, so the
  freeze entry point refuses. No approval was forged, nothing was pushed, no PR,
  no merge, no deploy, no index promotion. Root audits the run, writes the review
  in the documented shape, declares the candidate, exports, peer-reviews the
  bytes, then extends the three dashboard shapes.
