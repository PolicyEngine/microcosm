# Progress

## State

In progress under the adjudicated 3a-only contract. Wire the declared
`capital_gain_distributions` memo stage into the PUF support base after
`qrf_finalization` and before `qbi_reconciliation`, preserving the executor's
existing memo semantics and fail-loud rerun behavior.

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

## Next

- Trace neighboring outer-stage entity-grain handling and the checkpoint/build
  manifest recorder.
- Wire the existing executor without changing QRF machinery.
- Extend the existing capital-gain-distributions fixtures with base-path,
  stage-record/build-manifest, semantics, and rerun coverage.
- Regenerate the release input-coverage manifest through its builder only if
  the newly materialized route changes generated output.
- Run focused tests and the full suite, accounting only for the three declared
  pre-existing failures, then write the final report to the output file.
