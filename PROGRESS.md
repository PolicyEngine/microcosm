# Progress

## State

Populace #548 round 4 is in progress on `ssi-gate-batch-547` at the requested
worktree. The production guard and both required regressions are implemented;
the complete requested pytest/Ruff gates and final report remain.

## Done

- Confirmed the clean requested worktree, branch, and starting HEAD
  `012733eaad2bd426d6fc42f5a9359dd63e8ecaa0`.
- Read `CLAUDE.md` and the GitNexus debugging workflow.
- Confirmed that the prior `PROGRESS.md` described an unrelated completed
  branch and replaced it with this round-4 journal.
- Located the repository convention for a root `FINAL_REPORT.md`; the final
  outcome will be written there.
- Attempted the required GitNexus debugging workflow. Its analyzer could not
  update the managed global registry, and its local graph resolved to an
  unrelated repository, so no graph result was trusted; the generated local
  index was removed and the corridor audit used direct source call sites.
- Verified that all 10 telemetry calls in the scoped terminal section precede
  the terminal raise, which itself precedes both the H5 write and certification
  manifest construction.
- Added a section-local `_TerminalBatchTelemetry` proxy. It turns any
  `stage`/`attach_artifact` exception into a release-failing batch line and
  continues evaluation, including when the run had otherwise been green.
- Moved input-coverage, input-mass-parity, and QRF-tail failure collection
  ahead of each block's reporting writes.
- Ruff check/format-check and `git diff --check` pass for the production file
  using a task-local uv cache.
- Verified the committed `012733e` `final_loss` scrub. The audit also found
  unscoped strict-JSON risks in sparse-only sibling loss keys; these will be
  named in the final report rather than silently changed.
- Added live-telemetry coverage to
  `test_main_writes_diagnostics_before_post_calibration_gate_failure` as the
  `telemetry` parameter. Its double crashes the calibration-diagnostics
  attachment, and the test pins the pre-existing failure, recorded telemetry
  line, later coverage/mass/QRF evaluations, terminal raise, and absence of H5
  and manifests.
- Added
  `test_release_calibration_diagnostics_writes_nan_final_loss_as_null`, using
  an actual `CalibrationResult` and the real diagnostics writer.
- Both new paths and the existing `merge|crash` paths pass in isolated pytest
  runs; the changed test file is Ruff-clean and formatted.

## Next

- Commit the regression step.
- Run the full requested pytest file and both Ruff gates over every changed
  source/test file.
- Write and commit `FINAL_REPORT.md`, historicize this journal, and verify the
  final worktree/commit chain without pushing.
