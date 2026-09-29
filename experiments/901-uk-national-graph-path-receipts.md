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

## R4. Sweeps (2026-09-29, tree 218226898 with the PLAN_5 relaxation reverted)

- The touched suites during the work: 72 driver-side (the driver, the national graph, the
  preparation, the national role), 125 retirement-side (the seam suite, the national-calibration
  suite, the CGT observation period, the preparation, the rowwise candidate, the national graph and
  role), 451 regression (the certifier, the scoring route, the gate-register pins, the data contract,
  the battery bindings, the spine graph and evidence, the posture, the full gates and targets, the
  diagnostics), all green.
- The final tree: the whole engine-free suite including the uk directory in one process, 11,980
  passed, 119 skipped, one failure, `test_uk_cgt_projection::test_engine_is_reported_unavailable_without_the_uk_extra`,
  which asserts the uk extra is absent: it fails identically on main 6a70cd4ee in a venv that has the
  extra, and passes in the engine-free CI job, which installs none. The shared-spec lane
  (`test_country_spec`, `test_spec_engine_country_bundles`, `test_gate_battery_contract_pins`, the data
  contract, `test_stage_evidence`, `test_gate_battery_replay`): 432 passed. `tools/ci_test_groups.py
  --verify` ok, `ruff check` clean, every edited file formatted.
- Gate-register digests unchanged (`gates.json` untouched by this change); no pinned node key
  moved (the bound checkpoint's new `time_period` parameter moves that node's key, which nothing
  pins).

## R5. Licensed parity, seam vs graph (2026-09-29)

Harness `data/ukds/acceptance/901-national-ab/` (`run_national_ab.sh`, `compare_national_ab.py`,
`compare_a_vs_b2.json`). Input: the licensed 10 % smoke spine of the #901 pre-merge run
(`901-dense-10pct/spine-a/spine-a.h5`, 53,806 households, sha `a699e756…`), the pinned Chronicle
artifact `chronicle-uk-artifact-5324aa2`, the doctrine (no solve flags: 1,500 epochs, `family_equal`,
learning rate 0.02, seed 0), `--staging-local-only`. Both trees carried the PLAN_5 measurement
relaxation uncommitted (as the smoke run did), so the spine's gate-report digests match the declared
spec; reverted after the runs, never committed. A sampled rung, never exported.

- A = the seam: main at the #901 merge (6a70cd4ee, `repos/populace-main-901`),
  `--release-role national` dispatching to `run_uk_calibration`. 638 s wall.
- B = the graph: this branch at 218226898 (`repos/populace-901r`). 653 s wall (a first run at
  7f875a6f2, before the two evidence-shape fixes, took 627 s).

Both sides blocked at the terminal battery on the same gate, `uk_cgt_projection_entrants` (129,863
weighted sub-exempt gainers cross the frozen exempt amount by 2030 against the 73,000 bound): a
property of this 10 % spine under the engine's uprating, identical on both sides, not of either path.
So neither side wrote the H5 or the build record; the comparison stands on what both write before the
battery and on the battery itself.

Identical, both sides:
- The solve: the same 1,062 × 53,806 constraint matrix (CSR identical), the same row names and target
  vector, the same design weights and household axis, and final weights identical to the bit
  (max abs diff 0.0; total 29,024,853.117736887).
- The diagnostics file minus the build block: no differing field (initial loss 0.3107629418373108,
  final loss 0.008806911920052014, 53,806 non-zero, 97.74 % within 10 %, realized max ratio 10.0,
  every one of the 1,062 target rows incl. entity and measure descriptors, the UK block).
- The gate report minus attestation and release evidence: all seven outcomes and their details
  (six passed, the projection fence failed with the same numbers); both reports signed.
- The Logbook row: the same pipeline (`uk-frs-calibration`), the same identity digest
  (`2344c21d…`, so the graph's `run_config` is the seam's byte for byte) and the same input-pins
  digest, both `failed`, rung f100, seed 0.

The one remaining difference, deliberate: `build.measure_resolution.provider.mode` reads
`direct_h5` on the seam (the resolver simulated the input file) and `scratch_frame_export` on the
graph (the problem node exports the bound frame it holds to a scratch H5 for the engine), with the
matching `source_path`. Both statements are true of what each did; the measures they produced are the
same matrix. The two shape differences the first run showed (target descriptors read as compiled-row
callables on the household entity; `measure_resolution` carried the engine receipt instead of the
resolution loop's) were fixed in 218226898 and are gone in the re-run.

Not measured here: the H5 bytes and the build record (blocked before both). The hermetic driver test
checks the H5 (calibrated weights equal to an in-process solve, the seam's single mass record) and the
build record's every field; the readback node checks the written H5's content identity against the
calibrated population. An f100 national A/B on a spine that passes the projection fence is owed when
one exists (the fence is microcosm#970's; the sub-exempt tail of this smoke spine is the #1049 line's
concern, not this PR's).
