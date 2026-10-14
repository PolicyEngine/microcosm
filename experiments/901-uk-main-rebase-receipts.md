# #901 re-based onto main: receipts

Date: 2026-09-25. Branch `uk-full-build-graph-registration`, new history on `origin/main`
8f628b1e7 (main after #1006). Old head kept locally as the tag `uk-901-pre-main-rebase`
(2da4f421, stacked on #893's 2300e56b). Plan: `repos/uk-901-main-rebase-plan.md` (approved
2026-09-25); audit: `repos/uk-901-rebase-to-main-audit.md`.

## R0. What was ported and what was not

- Nothing from #893's shared code was needed by #901: the graph-kernel amendments the UK graph
  binds (platform-bitwise scope, keyed seeds, artifact inputs, rewrite ownership, the raw-bytes
  codec, the typed-artifact scope rule) were on main before #893 branched (merge-base 15ebde806).
- Folded in: Max's #918 (17 commits, cherry-picked with `-x`, authorship kept), renumbered from
  graph amendments 25/26 to 26/27 because main recorded the observer opt-in (#950/#951) as 25;
  `docs/graph-interface.lock` re-recorded (decl.py `b25ae4a6…`, kernel.py `24045ab2…`).
- #901's own shared additions (country-agnostic): `artifact_files`, `stage_evidence`, the
  gate-phase payload codec and `record_phase`, `calibrate.artifacts`,
  `calibrate.target_selection`, the `TargetSpec` dict codec.
- Left with #893: table identity, survey allocation, the CD benchmark, the legacy-QRF fit family,
  grouped bounds and target snapshots, the frame-checkpoint v4 schema, two performance fast paths.

## R1. Phase-1 spine consolidation: licensed A/B

Inputs identical on both sides: FRS 2024-25 raw tables, SPI 2022-23 PUT, HMRC collated tables
2023-24, WAS round 8, LCFS 2023-24 household and person, ETB 1977-2024, the five NTS tabs; no
sampling; `--no-staging`; checkpoints on. Script and outputs:
`data/ukds/acceptance/901-rebase-ab/` (`build_ab_spine.sh`, `spine-a/`, `spine-b/`, logs).

- A = main 8f628b1e7, `tools/build_uk_frs_spine.py`; B = branch 80f706050, the six-line shim over
  `uk_runtime/spine_build.py`.
- Both runs block at phase `transferred` on `uk_stage_student_loans_realization` (PLAN_5
  realization_deviation 1.1021725784754537 exceeds 1.0). That gate is main's own limitation at
  this head, recorded by #1006; neither side reaches an H5. Wall: A 284 s, B 308 s.
- The two 26-gate reports (`spine-a.spine_gates.json`, `spine-b.spine_gates.json`) are identical
  after removing the timestamped `release_id` and the signature that covers it: same gates, same
  outcomes, same evidence digests, same `blocked_at_phase`.
- The two content stores hold 8,076 stored tables each (`values.npy`); the multiset of their
  bytes is identical, with no table unique to either side. Node keys differ, as declared: the
  artifact declarations on the spine nodes move every key.
- Owed: the full-H5 A/B once a spine can be built at main's head, and the dense and national
  parity builds through the graph driver (see R4).

## R2. Phase results and test counts (JUnit-counted)

- Commit 2 (renumber, re-lock): `packages/microcosm-graph/tests` 690 passed / 1 skipped, twice
  (before and after the renumbering); `tools/graph_acceptance_burndown.py --verify` ok.
- Commit 3 (shared additions): 66 passed (new suites, gate-register pins, registry codec).
- Phase 1 (spine consolidation, 80f706050): targeted 82 passed; `spine-uk` group 2,825 passed /
  21 skipped; H2 parity 1/1 after regeneration (oracle `edb1659b…`).
- Phase 2 (graph build beside the drivers, 3df839f48): targeted 367 passed; whole `uk` group
  2,599 passed / 20 skipped; lock-hash test 24 passed; spot-check 203 passed.
- Phase 3 (dense role, 9bc39f8aa): whole `uk` group 2,654 passed / 20 skipped; role files 214
  passed; both drivers' `--help` exit 0.
- Phase 4a (national dispatch, 142268fa2): whole `uk` group 2,571 passed / 20 skipped;
  90 + 140 + 145 targeted.
- Phase 4b (HMRC tail retirement, ffe6ed7d2): retirement files 458 passed / 2 skipped;
  whole `uk` group 2,549 passed / 20 skipped; `shared-spec` 2,100 passed / 48 skipped;
  microcosm-data contract 254; H2 3 passed / 1 skipped; spot-check 337 passed / 1 skipped.
- Registry re-point (7eaf6ff3d): `test_frame_serializer_registry` green.
- Derived surfaces regenerated with their tools, never pasted: H2 spine parity fixture
  (`tools/graph_uk_spine_fixture.py`), release-input coverage manifest (`--check` current, 145
  required inputs, `source_stages.json` 73c3a14f…); gate-register digests did not move.
  `uv.lock` relocked; `APPROVED_UV_LOCK_SHA256` = `3bd27a6c…`.

## R3. Behaviour changes recorded for review

- Withdrawn 2026-09-28 (review item 1): a blocked *assembled* spine gate fails inside
  `run_graph`, and the driver now materialises the stored assembled report into
  `spine_gates.json` (`blocked_at_phase: "assembled"`, transferred `unreached`) before it
  fails, as main's tool did; a blocked *transferred* gate writes the file on the success path
  (R1). Pinned by `test_uk_frs_spine.py::test_driver_materializes_a_blocked_assembled_gate_report_before_failing`.
- `numerical_dependencies` pins installed versions into the H2 fixture.
- Withdrawn 2026-09-28 (review item 2): `_normalise_uk_local_bound_families` keeps main's
  refusal of a declaration that names nothing; only the graph's country-only target selection
  (`UKFullProblemKernel`, no local-surface spec selected) passes `allow_empty_local_binding=True`
  through `prepare_uk_full_solve`. Both behaviours pinned by
  `test_uk_local_rowwise.py::test_rowwise_binding_refuses_an_empty_declaration_unless_allowed`
  and `test_uk_full_solve_scope.py::test_zero_local_scope_uses_same_solver_and_has_no_fake_holdout`.
- The HMRC family names `spi_income_band_donors` (#1006) as a predecessor and admits its two
  operation kinds; the contract otherwise refused the drifted operation order.
- `tools/build_uk_rowwise_dataset.py` stays main's (its tests load it by path; it still serves
  `--candidate-clone-counts`).

## R4. Blocker for the dense and national licensed parity

The graph driver refuses every existing acceptance spine at its strict checkpoint gate
(`gates_manifest_sha256 differs from current declarations`), by #901's design, and a fresh spine
cannot be built at main's head (R1). Options put to María on 2026-09-25: wait for the #1012
lane to settle main's fence; a non-release measurement flag that accepts a stale gate manifest
and records the mismatch; or a measurement-only branch over #1012. The national role is parity
by identity regardless (same engine, moved code); the dense comparison needs a spine.

## R5. The 25 retired in-process tool tests, mapped onto the graph driver (2026-09-28)

Vahid's review counted 7 replaced, 8 partially covered and 10 lost. Re-derived here for all 25
against the pre-retirement bodies (`af01b990d~1:packages/microcosm-build/tests/engine_free/uk/test_uk_rowwise_candidate.py`),
each retired test is either restored on `full_build_cli.main` over the synthetic dense build
(`test_support/microcosm_build/uk_full_build_cli.py`: `run_dense_main`, multi-gate
`gate_payload`, `_FakeHub` from the rowwise helper, patches on `rowwise_staging`) or shown
covered by a named test. Unqualified `::` names below are in
`packages/microcosm-build/tests/engine_free/uk/test_uk_full_build_cli.py`. Totals: 15 rows
restored (rows 3, 8, 9, 10, 12, 14, 15, 18, 19, 20 to 25, through 13 new test functions and
one extended test; rows 9 and 10 live inside row 8's test), 10 rows covered by named tests (1,
2, 4, 5, 6, 7, 11, 13, 16, 17). Residual gaps are named on their rows; the two gaps rows 18 and
19 carried for a ruling (the run manifest's `sample` block, the size phases' epoch rows) were
closed on 2026-09-28 by restoring the tool's behaviour on the driver.

1. `candidate_build_writes_calibrated_h5_and_evidence`: covered. Manifest projection (schema 4,
   dense role and release id, doctrine block, identity pins, solve, weights, output digests,
   releasable) by `::test_dense_run_projects_the_rowwise_candidate_manifest`; the H5 and sidecars
   written and replayed by `::test_cli_cold_and_required_replay_recreate_dataset_and_sidecars`;
   the Logbook row (pipeline, rung, seed, iterating, artifact location, every verdict passed
   with a `.local_gates.json#/gates/` receipt) by
   `::test_main_runs_the_logbook_envelope_around_a_dense_build`; the real-solve calibrated
   weights, mass record and solver parity by
   `engine/uk/test_uk_full_target_graph.py::test_default_all_has_direct_matrix_and_solver_parity_and_replays`.
   The seeded adjudication row values are pinned by
   `test_uk_local_rowwise.py::test_committed_local_binding_register_references_committed_census`
   rather than through the driver.
2. `candidate_dry_run_plans_without_solve_or_write`: covered by
   `::test_dry_run_has_no_files_or_kernel_execution` (no files, no kernel execution, no Logbook
   row) and `test_uk_rowwise_candidate.py::test_graph_driver_dry_run_prints_the_operation_inventory`;
   the plan is the operation inventory by design, and the old plan's sampling, binding and
   cross-grain blocks are graph artifacts asserted through row 1.
3. `candidate_sampling_rung_receipt_and_engine_block_validation`: the `--engine-blocks` must
   equal `--n-clones` refusal restored as
   `::test_engine_blocks_must_be_positive_and_equal_the_clone_count`; the sampled-rung receipt
   covered by `test_uk_national_sampling.py::test_spine_sampling_is_stratified_keeps_families_and_normalizes`
   and `::test_source_sampling_cannot_be_reapplied_as_pool_sampling` (the graph samples the
   source spine, never the pool). The dry-run plan no longer carries a sampling block (row 2).
4. `candidate_f100_does_not_call_any_sampler`: covered by
   `test_uk_national_sampling.py::test_full_fraction_is_a_structural_no_op` and
   `::test_source_sampling_cannot_be_reapplied_as_pool_sampling`; the compact national sampler
   is not reachable from the graph's population node.
5. `candidate_engine_surface_reuses_one_resolver`: covered by
   `test_uk_full_measure.py::test_full_measure_reuses_one_resolver` (direct port).
6. `candidate_engine_surface_resolves_real_per_clone_blocks`: covered by
   `test_uk_full_measure.py::test_full_measure_resolves_real_per_clone_blocks` (direct port, same
   parametrisation).
7. `joint_candidate_f100_and_f001_end_to_end`: covered. The joint local/ladder/national matrix,
   solver parity and replay by
   `engine/uk/test_uk_full_target_graph.py::test_default_all_has_direct_matrix_and_solver_parity_and_replays`;
   the unbound-bridge and fan-out receipts by `test_uk_ledger_targets.py`; the rowwise dataset
   round trip and `clone_index` rename by `test_uk_rowwise_dataset.py::test_clone_uk_dataset_h5_roundtrip`
   and `test_uk_ladder_rowwise_clone.py::test_inherited_clone_index_is_replaced_like_the_pre_frame_writer`.
   Residual: the f001 leg's `rung_surface` counts are asserted through the driver only as a
   present manifest key (row 1); a sampled-rung run through the driver is not repeated.
8. `candidate_refusal_records_receipt_and_reraises`: restored as
   `::test_refusal_records_the_gate_and_error_receipt_pointers` (a blocked geography gate leaves
   `{verdict: failed, receipt: <local_gates.json>#/gates/<id>}` on the failed row; a raise
   leaves `pipeline_error` with `#/error_type`). The graph driver returns the block as status 1
   instead of re-raising, by design.
9. `candidate_binding_adjudication_failure_records_failed_row`: restored inside the same test
   (a refusal after `targets_bound` and before `solved` records the failed row and pointer); the
   binding refusal itself by `test_uk_local_rowwise.py::test_rowwise_binding_refuses_unadjudicated_committed_fence`.
10. `candidate_setup_failure_records_failed_row`: restored inside the same test (a preparation
    failure spools the failed row with the pointer and no manifest); `inputs_pinned` is not a
    graph-driver phase, the pins ride on the prepared build's `inputs` record (row 1).
11. `households_only_targets_come_from_compiled_chronicle_registry`: covered by
    `::test_households_only_binds_the_census_family_on_the_selection_node`, the uprating receipt
    cases in `test_uk_ledger_targets.py`, and the compiled-registry problem in
    `engine/uk/test_uk_full_target_graph.py` (both selection cases).
12. `candidate_dry_run_refuses_ladder_sidecar_collision`: restored as
    `::test_input_inside_the_output_directory_is_refused_before_anything_is_written` (an input
    under the output directory is refused by `_output_locations` before anything is written; the
    colliding file keeps its bytes), beside
    `::test_rejected_output_inside_source_never_writes_failure_sidecar` for the reverse direction.
13. `candidate_weight_ratio_failure_is_reported_and_blocks`: covered by
    `::test_blocked_gate_projects_an_unreleasable_manifest` and the restored row 14 (the
    weight-ratio failure line, the report's criticality and status, the failed row).
14. `candidate_block_partitions_failures_by_criticality`: restored as
    `::test_blocked_gates_partition_failures_by_criticality`.
15. `candidate_multi_block_engine_run_is_never_releasable`: restored as
    `::test_multi_block_engine_run_is_never_releasable` (end to end on the driver), beside
    `test_uk_rowwise_candidate.py::test_release_verdict_requires_single_block_engine`.
16. `size_candidate_exports_compact_links_and_cannot_claim_dense_release`: covered by
    `test_uk_full_calibration_graph.py::test_graph_preserves_numerical_path_and_complete_resume`
    (size search and refit nodes, the size artifact, complete resume) and
    `::test_changing_k_reuses_dense_but_source_bytes_invalidate_it`;
    `test_uk_rowwise_candidate.py::test_dense_candidate_manifest_has_no_size_sidecars` and
    `::test_size_cli_refuses_promotion_without_separate_certification`; the size outputs are
    read by `test_uk_size_evaluation.py::test_weight_tables_size_and_dense_spine_paths`.
    Residual: the K=300 export's link integrity and the two-seed manifest projection are not
    repeated through the driver.
17. `size_candidate_checkpoints_before_the_draw_and_resumes_from_it`: covered by
    `test_uk_full_calibration_graph.py::test_external_search_checkpoint_is_imported_without_repeating_solves`
    and `::test_reused_draw_skips_rng_and_rejects_changed_binding`, and the size-checkpoint cases
    in `test_uk_local_rowwise.py`. Residual: the `size_selection_checkpointed` and
    `size_selection_resumed` Logbook phases and the stderr progress lines are not pinned through
    the driver.
18. `candidate_build_stages_telemetry_locally_and_inventories_the_bundle`: restored by extending
    `::test_main_stages_the_bundle_locally_with_staging_local_only` with the retired inventory
    assertions (stdout manifest equals the on-disk one, run id equals build id, operation and
    pipeline ids, artifacts, fit summary, staged files with digests, sha256sums, sidecar
    inventory, sidecars never outputs, stage sequence) and the run manifest's `sample` block:
    `{"mode": "full"}` on the f100 rung, written through `rowwise_staging.stage_sample`, the
    one helper the national seam and the dense driver now share (the driver judges the rung on
    the effective fraction, pool times source spine); a rung below f100 stages a null sample,
    as the tool did, pinned by `::test_sampled_run_stages_a_null_sample_block`.
19. `size_candidate_stages_the_search_and_refit_phases`: restored as
    `test_uk_full_calibration_graph.py::test_size_kernels_forward_phased_epochs_to_the_registered_observer`:
    the size search and refit kernels forward their epochs, tagged `size_search` and
    `size_refit` by `dataset_size`, through the one observer
    `register_uk_calibration_kernels(progress_callback=)` registers (the driver's
    `_solve_observer`, so the staging rows and the stderr lines carry the phase); the observer is
    instance state, so the implementation hashes are those of an unobserved registry and the
    observed run replays under one without execution. The manifest projection records
    `graph.epoch_rows == "dense_solve,size_search,size_refit"` (pinned by
    `::test_dense_run_projects_the_rowwise_candidate_manifest`); the size-run manifest claims
    are rows 16 and 17.
20. `telemetry_content_refusal_never_aborts_the_solve`: restored as
    `::test_telemetry_content_refusal_never_aborts_the_solve` (drives the driver's own
    `_solve_observer` with synthetic epochs).
21. `invalid_local_telemetry_bundle_is_a_warning_not_the_runs_failure`: restored as
    `::test_invalid_local_telemetry_bundle_is_a_warning_not_the_runs_failure`.
22. `no_staging_records_both_opt_outs`: restored as `::test_no_staging_records_both_opt_outs`.
23. `remote_staging_uploads_telemetry_and_the_bundle_in_one_commit`: restored as
    `::test_remote_staging_uploads_telemetry_and_the_bundle_in_one_commit`, including the
    re-stage (`tools/stage_uk_rowwise_candidate.py`) and fetch (`tools/fetch_uk_staged_dataset.py`)
    legs.
24. `remote_staging_failure_is_recorded_and_the_build_still_succeeds`: restored as
    `::test_remote_staging_failure_is_recorded_and_the_build_still_succeeds`.
25. `no_staged_dataset_keeps_telemetry_remote_and_the_bundle_local`: restored as
    `::test_no_staged_dataset_keeps_telemetry_remote_and_the_bundle_local`.

One helper defect surfaced while restoring row 8: `arguments()` parsed through `cli.parse_args`,
which an earlier `run_dense_main` in the same test had already patched, so a second run reused
the first run's output directory; the helper now parses through the real parser bound at import.

## R6. Rebase onto main 937aca4ec after #1012 merged (2026-09-28)

- Replay: `git rebase origin/main` over the 33-commit series (pre-rebase head ae3179cff, local tag
  `uk-901-pre-1012-rebase`). Two stops. At the spine consolidation commit the shim-vs-tool conflict
  on `tools/build_uk_frs_spine.py` and the `uk_spine.json` fixture were staged by rerere from the
  trial merge (`repos/populace-901-1012trial`, 6103d2be2, never pushed); at the HMRC tail
  retirement commit the coverage manifest and `test_country_spec` were staged by rerere and
  `test_uk_graph.py` was resolved by hand (34 stages; the exclusion assertion stays retired). The
  trial's two modify/delete stops did not recur because #1012 merged on #998's layout.
- Lifted as the recipe recorded: #1012's three spine-tool hunks into `uk_runtime/spine_build.py`
  (the `UKSPIHousingShellStageTransform` import, the `impute_spi_housing_shell` and
  `price_domestic_energy` seed branches in `_declared_seeds`, the `implementations["spi_housing_shell"]`
  entry in `prepare_uk_spine_execution`), identical to main's delta on the tool; the H2 docstring count
  in `test_support/microcosm_graph/acceptance_h_parity.py`. Main already carried the 34 counts in
  `graph_kernels.py`, `tools/graph_uk_spine_fixture.py` and the three #1012 tests at their #998
  paths, so those lifts were moot. Both lifts are squashed into the commits they belong to
  (spine consolidation, tail retirement), so each commit keeps "spine_build = main's tool moved" true.
- Derived surfaces regenerated with their tools and found unchanged: the release-input coverage
  manifest (`--check` current, 145 required, 0 reviewed exclusions) equals the rerere resolution;
  the H2 fixture (oracle identity `9a069cfae…` on this machine) equals main's #1012 fixture with the
  rerere'd `uk_spine.json`. `uv lock --check` is quiet; `APPROVED_UV_LOCK_SHA256` stays `e299eef1…`.
- Roster: 34 stages, #1012's 36 minus the retired pair, in #1012's order (SPI block after
  `frs_brma`, `spi_housing_shell` after `hmrc_spi_income_spine`).
- Verification (targeted to the files the rebase touched, JUnit-counted): the 24 files the rebase touched or #1012 added (spine, roster, coverage, gates, evidence, driver, staging, pins) 729 passed / 0 failed, after one test fix: `test_uk_graph_evidence` had hard-coded `was_wealth` as the stage the assembled gate admits; the gate is roster-derived (the stage after `frs_brma`, now `frs_hmrc_spine_leaves` because #1012 moved the SPI block ahead of the WAS transfer, exactly where main's tool places `UK_SPINE_ASSEMBLED_FINAL_STAGE`), so the test now derives it and pins the new value; H2 spine parity, the shared parity check and the interface lock 5 passed;
  `tools/ci_test_groups.py --verify` ok (engine-free-shared 145, engine-free-us 142, engine-free-uk
  163, engine-us 78, engine-uk 34, integration-uk 1); `ruff check` and `ruff format --check` clean
  on the edited files. The rest is CI on the pushed head.
- Found by the targeted run: `tests/engine_free/shared/test_gate_battery_contract_pins.py` was a
  zero-byte file on this branch since the #998 re-homing (the national-dispatch commit's 17-line edit
  to the flat file had been replayed as a delete), so its 20 tests collected nothing on the pushed
  heads and CI. Restored from main's 585-line file with the branch's own hunk (the dense-line mirror
  reads `UK_ROWWISE_DENSE_POSTURE.gate_policy_suffix` instead of loading the retired tool by path);
  20 passed on this tree, no digest moved.

## R7. Licensed 10 % dense smoke run before merge (2026-09-28, María's ask)

Question put: has a dense licensed run been made end to end, and what does a full dense run cost
on the graph driver? Neither had been done (R4). Method: two measurement-only worktrees at main
937aca4ec (`repos/populace-main-1012`, main's tool) and at this branch (`repos/populace-901-measure`),
both carrying the #1014 lane's PLAN_5 relaxation patch uncommitted so a spine can be built at all;
harness and outputs under `data/ukds/acceptance/901-dense-10pct/` (`run_dense_10pct.sh`, the chain
scripts, `phase_timeline.py`, `compare_ab.sh`). Smoke settings ruled by María: K=2 clones, 250
epochs, `--sample-fraction 0.1 --sample-seed 7`, seed 42, staging local-only (phase timings, no
upload); never a release. Reference: the runbook's 3.5 h / 10 GB at K=15, 1,500 epochs on the
rowwise tool; the graph driver had never been timed.

- Spine A (main's tool, full data, no sampling): 413 s wall, 9.7 GB peak, 26/26 gates passed under
  the measurement patch, H5 188 MB (`spine-a/`).
- A = main's tool on spine-a (`tool-10pct/`): exit 0, 581 s wall, 7.4 GB peak. f010 rung: 5,278
  households → K=2 → 10,556 rows, 18,729 targets; 250 epochs, loss 0.699 → 0.354; 4 of 6 local
  gates fail (area support, per-family fit, target fit, weight ratio), `releasable: false`.
  Phases: target_compilation 494 s, cloning 4 s, surface_resolution 10 s, calibration 10 s,
  gate_battery 1 s, holdout 39 s, output_bundle 10 s.
- B = the graph driver on spine-a, same flags: exit 1 by the calibrated battery's verdict, 1,086 s wall, 7.7 GB peak (`graph-10pct/`). Every
  node through the calibrated gate battery ran: the same f010 sample (10,556 rows, 18,729 targets),
  the dense solve, the rotated holdout, the calibrated population and the 26-gate calibrated battery;
  the battery classed nine national/source checks as structural stops (as the full-build enforcement
  rule declares below f100: only the five local fit/support/weight checks are excused), so no export,
  package or certification node ran. Phases: target_compilation 997 s (the graph's node also carries
  the surface, the cross-grain reconciliation, the measures, the problem, the geography gate and the
  preflight battery, which the tool splits over compile 494 s + cloning 4 s + surface 10 s),
  calibration 76 s (dense 15.5 s, holdout 38.5 s, calibrated 2.2 s, battery 13.3 s; the tool: 10 s
  solve + 39 s holdout + 1 s battery + 10 s bundle).
- B2 = the graph driver end to end (`--spine-request`, the graph builds the spine): exit 1 by the same battery verdict, 1,445 s wall, 9.9 GB peak (`graph-spine-request-10pct/`). The graph
  built the 34-stage spine itself with the two gate nodes (assembled after `frs_brma`, transferred at the
  endpoint): the 26 spine gates carry the same ids and the same outcomes as main's tool report on
  spine-a (all passed), the same `gates_manifest_sha256`; the sampled pool and the solve are bit-identical to
  the input-H5 rung and to main's tool (final loss 0.3535366112843803 again), which is the full-H5
  spine parity R1 left owed, established through the solve rather than a payload compare (the graph
  materialises no spine H5 below the export). Node wall times: the 50 spine nodes sum to 296 s against
  the tool's 413 s spine build; `uk.full.target_compilation` 860 s; preflight battery 72 s; measures
  12.6 s; problem 7.3 s; dense 15.3 s; holdout 39.5 s; calibrated battery 14.4 s.
- Parity A vs B: bit-identical where both sides produce the same object: initial loss 0.6991022825241089 and final
  loss 0.3535366112843803 on both drivers, 18,729 target rows, 10,556 non-zero households, the same six
  local gate outcomes (four failed, two passed), the same rotated holdout (mean 0.7248643006468352, worst
  0.7320742950761016, five folds). The H5 payload compare waits on the export rung below.
- B3 = the graph driver on spine-a with a measurement-only export exception (below f100 the national
  checks are recorded without enforcement, as the tool's sampled rungs behave; `export-exception-measurement.patch`,
  applied in the measurement worktree only): exit 0, 1,143 s wall, 8.6 GB peak (`graph-10pct-export/`). The whole
  line ran: export, readback, package, certification readiness (`ready_for_external_review: false` at f010, as it
  must be), the schema-4 manifest, `sha256sums.txt`, the local bundle. H5 payload against A
  (`tools/compare_uk_h5_payload.py`, receipt `ab_payload_compare.json`): same 40,238,817 bytes, keys equal,
  `person`, `benunit` and `time_period` payload-identical, `household` identical in rows, column order and
  every value, with eleven geography code columns stored as pandas `string` on the graph against `str` on
  the tool (the file digests differ for that alone); the six local gates agree.
- Full-run proxy: the solve path is byte-identical (same losses to the last digit at every rung), so the per-epoch
  and per-row costs are the tool's; the graph adds a fixed cost per run, at this rung about 500 s
  (target_compilation 860–997 s against the tool's 494 s compile + 14 s cloning and surface, of
  which the historical validation-period registry compile and the preflight battery are the
  identifiable parts) plus about 15 s across the calibration segment. Against the runbook's
  3.5 h / 10 GB reference at K=15 and 1,500 epochs on the tool, a full graph run projects to about
  3.7 h; peak memory 7.7 GB (input-H5) and 9.9 GB (spine-request) at this rung against the tool's
  7.4 GB. Not exploding; the fixed overhead is the target-compilation node, which is the one place
  worth profiling before a full run.

Defects the run surfaced, none visible to the unit suites or CI:

1. Main (#971, merged 2026-09-23, not this branch): every real full-UK ladder is refused at the
   locations step because `draw_uk_ladder_locations` compares the ladder's composite
   local-authority vintage (`ew:2023_april_lad;scotland:2019_council_area;ni:2014_lgd`, the only
   vintage the artifact tool writes) with the EW-only names-resource constant; the fixture ladders
   carry the plain string, so the tests pass. Both drivers hit it. Measured with a second
   uncommitted patch (`la-vintage-measurement.patch`) in both trees; fix drafted for main
   (`repos/uk-971-ladder-vintage-defect-draft.md`).
2. This branch, fixed (1214fb616): `full_targets` compiled the national register for {2023, 2025,
   calibration year} fail-closed; the pinned feed carries OBR facts from fiscal 2024 onward only,
   so the build refused at `uk.full.target_compilation`. Historical validation periods are now
   best-effort and recorded in `register_completeness`; the calibration year stays fail-closed.
3. This branch, fixed (265f8acf4): on the raw-spine path the sample node's transferred-gate
   admission lacked `spine_gate_synthetic_smoke`, so the first end-to-end build failed at
   `uk.full.sample` after the whole spine had run.
4. This branch, fixed (cd6829a71): the joint-surface helpers filtered the national register to
   country rows (pre-#906), leaving the region legs unparented in the cross-grain reconciliation;
   they now pass the whole register as the rowwise tool does.
5. This branch, fixed (da9b2472a): the local surface crosses the target_compilation → problem
   boundary as JSON, so each spec's schema-8 hierarchy came back as a mapping and the problem
   assembly refused it; the stored surface is now decoded the way ``TargetSpec.from_dict`` does.
6. This branch, fixed (f9562fc01): the driver handed the gate batteries the spine adapter as the
   coverage engine, whose ``variables()`` enumerates the frame's non-computed columns, so the
   release input-coverage gate read sixteen live inputs as manifest drift; the batteries now get
   ``PolicyEngineUKCoverageEngine`` as the release-cut producer hands it.
7. This branch, fixed (7003aaee3): the preflight parity gate for production 2023 reads the 2023
   registry, which fix 2 had skipped; a validation period now keeps its partial registry, as the
   release-cut producer does, with the unsupported references recorded.
8. This branch, fixed (86137886d): the calibration segment ran for the first time on real data and the
   executor refused the holdout report at the calibrated gate battery and the certification node
   (amendment 19: a platform-bitwise artifact needs a platform-bitwise consumer); the holdout kernel
   alone declared that class while the dense solve it rotates declares the default; it now matches,
   and a test walks every typed edge of the composed graph on the synthetic spec.
9. This branch, fixed (055591b54): the dense solve ran (250 epochs, loss 0.35355 against main's
   0.35354 on the same spine), the holdout and the calibrated population followed, and the
   calibrated battery refused in its receipt: it handed the diagnostics the selected registry, wider
   than the solved one below f100 (the rung drops cells unreachable in the sample), so the
   local-authority UC rows were neither compiled nor skipped; it now partitions the specs that entered
   the solve, as the rowwise tool's diagnostics registry does, and the driver names a battery's recorded
   exception instead of tripping on the missing artifact.
