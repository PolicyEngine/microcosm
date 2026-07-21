# Progress

## State

Implementation is wired under the adjudicated 3a-only contract. The declared
`capital_gain_distributions` executor now runs in both PUF support base paths
after `qrf_finalization` and before `qbi_reconciliation`; focused end-to-end
stage-record, semantic, and rerun coverage is green. Full-suite verification
is next.

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
  conflicts in `CONTRACT_FINDINGS.md` without changing implementation or tests.
- Recorded the user's adjudication: the conservation requirement is withdrawn,
  the existing executor behavior is authoritative, and QRF machinery is out of
  scope.
- Retitled the resolved adjudication record from `BLOCKED.md` to
  `CONTRACT_FINDINGS.md` while retaining its analysis.
- Traced the existing entity-grain convention: use `Frame.place` to sum PUF
  person inputs to tax units, then first-person carry the memo output because
  PolicyEngine re-aggregates that person input to the filing unit.
- Added `capital_gain_distributions` to the outer pipeline and monolithic base
  path at the adjudicated insertion point, calling the unchanged manifest
  executor through `run_source_stage`.
- Preserved both original PUF inputs and surfaced any pre-existing output to
  the executor so its existing overwrite refusal remains the rerun contract.
- Extended the existing capital-gain fixture through the actual named outer
  stage: tax-unit memo semantics, unchanged PUF inputs, first-person output
  placement, automatic pipeline/completed/stage-record entries, and the
  executor-owned rerun failure are pinned end to end.
- Updated the locked pipeline/base-order tests; the focused builder and
  executor suites pass (58 tests), and Ruff is clean for the touched files.
- Confirmed the release input-coverage generator is already byte-identical:
  both capital-gain route legs are required independently of builder stage
  presence, and its focused six-test sync/route guarantee suite passes. No
  generated manifest rewrite is needed.

## Next

- Run focused tests and the full suite, accounting only for the three declared
  pre-existing failures, then write the final report to the output file.
