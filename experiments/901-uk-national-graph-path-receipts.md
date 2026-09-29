# Receipts: the UK national release role on the graph (PR-2 of the #901 line)

Branch `uk-national-graph-path` off main 6a70cd4ee (the #901 merge, 2026-09-29). Plan:
the "PR-2" paragraph of the #901 re-base plan (her ruling 2026-09-25); execution record in the
working notes. Rule: nothing here is a release; the parity run is a sampled rung and is never
exported.

## R1. Composition and the seam fixture (hermetic)

`packages/microcosm-build/tests/engine_free/uk/test_uk_national_graph.py` (8) runs the composed
national graph on the calibration seam suite's synthetic frame, bound as a checkpoint:

- The ordered problem is the seam's matrix row for row (`build_constraint_matrix` on the adapter's
  prepared frame with the register's target set), with the doctrine bound: `family_equal` loss
  weights, the national mass reason, the bounds, the register digests.
- The graph's calibrated weights equal an in-process `calibrate(...)` call under the national
  doctrine on the same frame and register; the calibration mass record is the seam's and is appended
  once.
- The gate battery evaluates the seven `UK_CALIBRATION_GATE_SCOPE` gates; the evidence block carries
  the seam manifest's fields.

Found while composing: the bound checkpoint's CREATE normaliser defaulted the frame's period to the
FRS vintage (2024), so the projection fence declared 2024 against the fixture's 2023 and blocked. The
checkpoint node now carries its H5's own `time_period` (a node parameter on `uk.full.spine_checkpoint`;
its key moves, no pin).

## R2. The driver (hermetic)

`test_uk_rowwise_national_role.py` (21) runs `microcosm-build-uk --release-role national` end to end
on the same fixture: the build record's bindings and provenance, the signed report (posture,
exclusions, release evidence, HMAC), the diagnostics build block, the frozen registries, the
manifest, local and remote staging (fake Hub), the Logbook row on `uk-frs-calibration` with the seam's
phases, the doctrine overrides, the unpinned-feed refusal and override, the incumbent evaluation after
staging, a blocked battery (report on disk, no H5, failed row), and the H5's weights equal to an
in-process calibrate call under the doctrine.

## R3. Retirement

`run_uk_calibration`, its attempt runner and battery runner, `UKNationalCalibrationStage`: gone.
The seam suites re-anchor on the graph's route as a library (`encode_uk_national_problem`,
`materialize_uk_national_rows`, `national_calibration_manifest`): `test_uk_national_calibration.py`
26, `test_uk_calibration_run.py` 12 (the checkpoint, scope, admin-anchor, provenance and Logbook-scope
contracts, plus the band-edge reconciliation and attempt-id tests the runner tests carried),
`test_uk_cgt_observation_period.py` 1.

## R4. Sweeps

(filled at the end: the touched suites, the uk group, ci_test_groups, ruff)

## R5. Licensed parity, seam vs graph

(filled from `data/ukds/acceptance/901-national-ab/`)
