# Progress

## State

Blocked by the task's explicit contract-safety condition. The declared stage
adds a Schedule-D memo component from long-term gains but never reduces
`non_sch_d_capital_gains`, so it cannot satisfy the requested conserved split
or direct-route range. Full findings are in `BLOCKED.md`.

## Done

- Confirmed the worktree is on `cgd-split-462` and starts clean.
- Read the complete stage declaration, share resource, executor, runtime, base
  builders, outer-stage manifest recorder, and relevant tests.
- Confirmed the declared 9.8526% parameter is a Schedule-D share of eligible
  long-term gains, not a partition share for the existing CGD total.
- Confirmed the executor adds only the Schedule-D output, leaves the direct
  route untouched, and fails loudly when the output already exists.
- Located the otherwise-correct builder insertion point after PUF QRF
  finalization and before QBI reconciliation, and confirmed how an outer stage
  would be recorded in `stage_run_context.json`.
- Documented the blocking semantic, magnitude, conservation, and grain
  conflicts in `BLOCKED.md` without changing implementation or tests.

## Next

- Obtain an approved stage-contract revision or clarification covering the
  conserved route split, person/tax-unit placement, and provenance-backed
  parameter.
- After that contract exists, wire the stage after QRF finalization, add the
  requested orchestration and semantic tests, then run the focused and full
  suites.
