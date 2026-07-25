# Final report: populace #548 round 4

Date: 2026-07-25

## Outcome

Round 4 is complete on `ssi-gate-batch-547` in the requested worktree. No
push, release publication, certification, or other worktree/branch mutation
was performed.

Functional commits:

- `01926dddd156d153ef837c4b27970725bdddf049` — guard terminal-batch
  telemetry and secure gate lines before reporting.
- `49b65a110cfd9789feb349b543b561e178ff4588` — add the live-telemetry and
  real diagnostics-writer regressions.

## Design

The builder now creates a section-local `_TerminalBatchTelemetry` proxy
immediately before attaching `calibration_diagnostics.json`. Every current
`stage(...)` and `attach_artifact(...)` call through the terminal raise goes
through that proxy. If the underlying live telemetry raises, the proxy appends
an operation-labelled failure to `terminal_gate_failures` and returns, allowing
the remaining input-coverage, input-mass-parity, and QRF-tail gate groups to
run.

This is intentionally release-failing even when all substantive gates were
green: a telemetry crash becomes a recorded batch line and reaches the normal
terminal `RuntimeError`, rather than producing an opaque early abort.
Telemetry outside this narrowly scoped section retains its prior behavior.

For the three gate-result blocks, enforced failure lines now enter
`terminal_gate_failures` before the block writes or attaches its report and
before it emits a failed telemetry stage. This matches the already-established
SSI secure-first pattern.

Enforcement remains strict. Any collected substantive or telemetry line reaches
the terminal raise. That raise remains before `write_dataset(...)`, and
`_build_manifests(...)` remains later still, so a failure path writes neither
the release H5 nor certification manifests. The end-to-end regression also
asserts both absences.

## Tests added

- `test_main_writes_diagnostics_before_post_calibration_gate_failure[telemetry]`
  uses a live telemetry double that raises while attaching the already-written
  calibration diagnostics. It proves the terminal message contains both the
  pre-existing `ctc failed` line and the recorded telemetry-crash line, proves
  later coverage, mass-parity, and QRF-tail evaluations occur in order, and
  proves no H5 or manifest is written.
- `test_release_calibration_diagnostics_writes_nan_final_loss_as_null` creates
  an actual `CalibrationResult` with `final_loss = NaN`, calls the real
  `_write_release_calibration_diagnostics`, and verifies both the top-level
  diagnostic and `build.default_dataset.final_loss` serialize as JSON `null`.
  The test preserves the committed `012733e` boundary: `main()` supplies the
  already-scrubbed default-dataset value to the writer.

## Verification

- `uv run pytest packages/populace-build/tests/test_us_fiscal_refresh_builder.py -q`
  — **rc 0**.
- `uv run ruff check tools/build_us_fiscal_refresh_release.py packages/populace-build/tests/test_us_fiscal_refresh_builder.py`
  — **rc 0**.
- `uv run ruff format --check tools/build_us_fiscal_refresh_release.py packages/populace-build/tests/test_us_fiscal_refresh_builder.py`
  — **rc 0**.
- `git diff --check` — **rc 0**.

The managed sandbox denied the global uv cache, so the same commands used
`UV_CACHE_DIR=/private/tmp/populace-548-uv-cache`; no repository files were
created by that override.

## Unanticipated corridor findings

These were named and deliberately not changed outside the round-3-finding-2
scope:

1. The direct report writes for `input_coverage.json`,
   `input_mass_parity.json`, and `qrf_tail_concentration.json` remain
   unguarded. Their gate lines are now secured first, but a filesystem or
   serialization exception at those writes can still prevent later groups and
   the terminal raise.
2. Commit `012733e` correctly scrubs
   `build.default_dataset.final_loss`. The normal sparse payload separately
   retains `selection_final_loss`, `refit_initial_loss`, and
   `refit_final_loss`; in particular, the same non-finite `result.final_loss`
   still enters `refit_final_loss` and can trip strict JSON. This sibling-key
   issue was not silently repaired because the brief explicitly scoped finding
   1 out of this round.
3. `PolicyEngineUSEngine()` construction immediately before input coverage is
   outside the degraded evaluation guards. A constructor exception would still
   interrupt an already-degraded batch; it is evaluation infrastructure, not
   the optional telemetry surface requested here.

## Tooling note

The GitNexus debugging workflow was attempted. Its analyzer could not update
the managed global registry, and the resulting local graph resolved to an
unrelated repository, so no graph output was trusted. The generated local
index was removed, and all conclusions above came from direct source call-site
and test inspection.
