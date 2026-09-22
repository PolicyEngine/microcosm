# microcosm#823 consolidation receipts

## R1 — the national role rebuilds the v20 seam-built dataset bit for bit (2026-09-20)

Comparison run: `tools/build_uk_rowwise_candidate.py --release-role national` on the
v20 inputs — spine-s (`4d9752fdcd92…`, 175,920,391 bytes), Chronicle feed
`c5e5bf8` (facts `6d039dd869dc…`, manifest `20ac5d22e7d8…`), the committed feed
pin — under the ruled doctrine (1,500 epochs, `family_equal`, learning rate 0.02,
seed 0) with no solve flags, `--staging-local-only`; code = `eba0e152` (the
retirement commit, this PR's head minus the receipts commit) plus the signed
self-employment 20-30k deferral (`90bdb809`, cherry-picked as `3828183c`,
`git_dirty` false, engine 2.98.0) — the deferral is not in this PR (it is the
v20 branch's own commit) and without it the target-fit gate blocks while the
weights still compare, so the same inputs on the bare head are unverified past
the gate. Output `data/ukds/acceptance/823-consolidation/national-v20-twin/`,
attempt id `uk-frs-calibration-attempt-20260920T165920Z-5acd7515`. Reference: `runs/uk-623-first-calibrated/spine-assessment-v20/`
(`uk-frs-calibration-attempt-20260918T174754Z-3f928f19`, built by the retired
`tools/calibrate_uk_national_dataset.py` with `--epochs 1500
--target-weight-rule family_equal`).

- Household weights: `numpy.array_equal` on 52,846 households — **equal**.
- `calibration_diagnostics.json`: 638 target rows (name, target, initial and
  final estimate, relative error) — **equal**; final loss 0.010414168864537107,
  initial loss 0.3009223937988281, ESS 9,304.74, within-10 % 96.39 %, realized
  max weight ratio 10.0 — **equal**.
- Register: 705 compiled, 67 excluded, 638 calibrated, version `f536021bbcd9` —
  **equal**.
- Build record: `calibration` block, `input_posture`, `spine_provenance`,
  `run_config.ledger`, `run_config.doctrine` — **equal**; six calibration-seam
  gates all `passed` on both; `run_config.doctrine_overrides` is `{}` on the
  twin where v20 carried `{epochs 256 → 1500, target_weight_rule uniform →
  family_equal}` (the constants moved in B1).
- Wall time 456 s (v20: 852 s on a loaded machine), peak RSS 9.3 GB.

Rowwise manifest: `uk_national_calibrated_candidate`, `release_role national`,
`release_id microcosm-uk-2024-25-national`, `staged_dataset.status skipped`
(local-only), `staging_delivery.mode local_only`; frozen
`national_target_registry.json` version `f536021bbcd9`.

## R2 — the same build with remote staging and read-back (2026-09-20)

Same inputs, code and doctrine as R1, remote staging on at the driver's default
300 s cadence with `--staging-read-back`; output
`data/ukds/acceptance/823-consolidation/national-v20-twin-remote/`, attempt id
`uk-frs-calibration-attempt-20260920T170811Z-ce339e7c`, wall 550 s.

- Household weights and the 638 diagnostics rows equal to v20 (and to R1);
  register `f536021bbcd9`; six gates passed; `doctrine_overrides` `{}`.
- Telemetry: `policyengine/populace-uk-staging` `runs/<attempt id>/` holds
  `run_manifest.json`, `progress.json` (status `completed`), `events.ndjson`,
  `calibration_progress.json`, `artifacts/fit_summary.json`,
  `artifacts/staged_dataset.json`; `staging_delivery` in both the build record
  and the manifest: mode `local_and_remote`, 35 attempts / 35 successes,
  `read_back passed`, no error code; no telemetry warning in the run log.
- Bundle: `policyengine/populace-uk-private` `staged/<attempt id>/` (revision
  `40439a3f…`) holds the five outputs (`microcosm_uk_2024_25.h5` 176,040,337
  bytes, `build_record.json`, `calibration_diagnostics.json`,
  `microcosm_uk_2024_25.terminal_gates.json`, `national_target_registry.json`),
  the manifest, `staged_manifest.json` and `sha256sums.txt`, in one commit;
  `staged_dataset.status uploaded`.
- `tools/fetch_uk_staged_dataset.py --run-id <attempt id>` returned every file
  digest-verified and the fetched H5 is byte-equal to the local one (the copy
  was deleted after the check).

## R3 — the end-of-build evaluation and the first release-cut certification (2026-09-21)

María's direction (2026-09-21): certify national candidates; warn loudly, naming
every measure absent on the incumbent, instead of refusing; evaluate at the end
of every national build without blocking staging; never publish on an unpassed
evaluation. Code: scratch branch `uk-823-rehearsal-eval` (never pushed) = this
branch's `acd93c12` (the driver hook, PR #965) + PR #967's `92edabdd`,
`e0d80c37`, `4173ab38` (the v20 deferral, the scoring-route fix, the six A16
retirements) + `9f81db51`/`a58b6c29` (the common-surface scorer, #967) + the
certifier pin fix (#967 `3f2fe52d`, picked as `d45e91be`).

**Build.** `tools/build_uk_rowwise_candidate.py --release-role national` on
spine-s (`4d9752fdcd92…`) and feed `c5e5bf8`, no solve flags (1,500 epochs,
`family_equal`, learning rate 0.02, seed 0; override receipt `{}`),
`--staging-local-only`, `--incumbent-h5` the enhanced FRS 1.57.3 copy
(`ef34c1ae2821…`, label `enhanced_frs_2024_25_v1_57_3`). Attempt
`uk-frs-calibration-attempt-20260921T152658Z-6609d768`, 15:26:47Z → 15:35:30Z
(8m43s including the evaluation), exit 0. 644 targets (v20's 638 plus the six
retired A16 rows), 52,846 households, final loss 0.010453, max absolute relative
error 0.2524 (the signed 20-30k self-employment deferral, inside its window),
median 0.00109; the six calibration-seam gates passed. Output directory
`data/ukds/acceptance/823-consolidation/national-eval-rehearsal/`; the full
compiled register is frozen beside the solve register as
`national_contract_registry.json`.

**Evaluation.** `score_vs_incumbent.json` (sha `749598dff826…`, listed in the
sums, `outputs.score_receipt`, manifest `evaluation` block; telemetry events
`dataset_staging` < `incumbent_evaluation` < `complete`). Verdict **passed**:
candidate full loss 0.009559 against the incumbent's 0.212540; 501 candidate
wins to 23 on the 524 common targets (register `2331a9386a2e`). 120 targets
pruned from both arms and named on stderr, as the v21 head-to-head had them:
`dwp_universal_credit` 93, `dwp_two_child_limit` 15, `ons_household_composition`
10, `hmrc_cgt` 2, over six measures the enhanced FRS never carried
(`benunit.uc_calibration_administrative_family_type`, `…_child_count`,
`…_has_child_under_one`, `benunit.uc_tcl_affected_benunit_proxy`,
`household.ons_household_type`, `person.capital_gains_asset_type`). One defect
surfaced: the telemetry copy of the receipt was refused as a record array (the
per-target drift list), so this run's telemetry carries no `score_vs_incumbent`
artifact; fixed on this branch as `6104926e` (the staged copy keeps the
verdict, the pruned block and the aggregates and drops the arrays).

**Certification.** `tools/certify_uk_release_cut.py` with the spine H5, the
feed, the accepted input-mass reference
(`686-spine-swap/uk_input_mass_reference_2024_25_v1_56_16.json`) and the
receipt. The first attempt refused before any gate ran: the certifier's Logbook
pin roles carried a digest without `size_bytes`, so it had never run end to end
(fixed on #967 as `3f2fe52d`, with a stub-battery test). The second attempt
(`uk-frs-release-certification-attempt-20260921T153934Z`, 15:39Z → 16:50Z) ran
the 20-gate release-cut battery: **12 passed, 7 failed, 1 evidence absent**,
`GateBatteryBlockedError`, no certification composed (error receipt under
`logbook-receipts/`). Passed: `uk_aggregate_admin`, the three Ledger compile
parity gates, `uk_qrf_tail_concentration`, `uk_release_family_build_stages`,
`uk_release_input_coverage_manifest_current`, `uk_support`,
`uk_take_up_signal`, `uk_target_surface`, `uk_target_surface_local_default_2025`,
`uk_uc_capital_coherence`. The rest fall into three groups:

- *Certifier evidence adaptation, not the candidate.* `uk_release_input_coverage`
  (19 failures over 13 stage families: 13 "final household weights have kind
  'calibrated', expected reviewed kind 'importance'" and 6 "lacks the reviewed
  mass-conserving MassChangeRecord": the gate re-evaluates stage-boundary
  expectations on the calibrated frame). `uk_uc_deduction_combination_enum_domain`
  and `uk_student_loan_plan_enum_domain` fail closed with
  `'PolicyEngineUKCoverageEngine' object has no attribute '_variable'`: the
  evaluator (`battery_bindings.py`) reads `engine._variable(column)` while the
  certifier supplies the coverage engine, which exposes `variables()`,
  `variable_entities()` and `default_values()`. `uk_weights_audit` is
  evidence-absent because spine-s's sidecar carries no `fit_weight_records`
  block (the rehydration returns `None` for a pre-#757 sidecar).
- *Stale reviewed registers.* `uk_export_surface` compares against
  `UK_CANDIDATE_DATASET_NAME = "microcosm_uk_2024"` and the June allow-list: 49
  exported columns outside the enhanced FRS surface without an entry (person 32,
  household 12, benunit 5: the `hmrc_spi_*` income components, reported benefit
  flags, council tax, support-channel and clone ids, source keys), the calibrated
  weight not exported as `household.household_weight`, and one reference-only
  column to drop (`person.incapacity_benefit_reported`). `uk_input_mass_parity`:
  two columns beyond the ±452 % fence (`dfe_education_spending` +187,609 %,
  `jsa_income_reported` +909 %) and two stale exclusions now within tolerance
  (`charitable_investment_gifts`, `owned_land`), over 126 checked columns and 65
  candidate-only ones.
- *Data findings.* `uk_degenerate_release_surface`: two persisted all-zero
  columns (`person.incapacity_benefit_reported`, `person.is_in_approved_training`).
  `uk_nonnegative_columns`: `housing_water_and_electricity_consumption` has 239
  values below zero (minimum −14,165).

The evaluation path therefore works end to end and the receipt is certifiable
in shape; the cut does not certify today because the release-cut battery had
never been run on a candidate of this shape. María's calls: the three certifier
adaptations, the export allow-list and dataset name, the two input-mass
breaches and the two stale exclusions, the two all-zero columns and the negative
consumption values. Assembly and the inspect publication wait on a green
certification and her go.

## R4 — the release-cut battery on the R3 candidate with the certifier-evidence fixes (2026-09-21)

María's direction (2026-09-21): retire the stale exclusions, fix the fit-weight
records, give the coverage engine an enum-domain method, evaluate the
spine-stage contract on the spine's importance weights and the calibration
checks on the calibrated weights, and retire `hmrc_cgt_gains` from the June
path. Code: the four commits pushed to PR #967 (`447a24bd` stale exclusions,
`f6b8ce0c` June CGT family + pass-through semantics, `e77cc65c` enum-domain
accessor + build-state half on the spine frame + `--spine-sha256`, `dc42375b`
fit-weight collector), picked onto the scratch branch `uk-823-rehearsal-eval`
(never pushed) as `7222f78a`, `fd358654`, `be818e8e`, `d07ac0e7` on top of R3's
`d45e91be`.

**Certification.** The same candidate and inputs as R3 (candidate
`5b88e4accbb1…`, spine-s `4d9752fdcd92…` now pinned by `--spine-sha256`, feed
`c5e5bf8`, the accepted input-mass reference, the R3 score receipt), outputs
written beside R3's as `microcosm_uk_2024_25.release_cut_gates.r4.json` (sha
`802290cefe08…`). Attempt
`uk-frs-release-certification-attempt-20260921T185339Z`, 18:53Z → 19:09Z
(16 minutes; R3 took 71), the 20-gate release-cut battery: **15 passed, 4
failed, 1 evidence absent**, `GateBatteryBlockedError` at the terminal phase,
no certification composed. Against R3:

- `uk_release_input_coverage` failed → **passed**: the family build-state half
  read the spine frame (`details.family_build_state_frame: "spine"`), so the
  13 weight-kind failures are gone, and the 6 mass-record failures are gone
  with the invented E5 reasons (five families now `weights_pass_through`) and
  the retired June CGT family. The column and effective-mass halves still
  read the calibrated release frame.
- `uk_uc_deduction_combination_enum_domain` and
  `uk_student_loan_plan_enum_domain` failed-closed → **passed**: one column
  each checked against the engine's enum domain through the new accessor,
  zero invalid values.
- `uk_input_mass_parity` still fails, now on the two breaches alone
  (`dfe_education_spending` +187,609 %, `jsa_income_reported` +909 %):
  `stale_exclusions: []`, `expired_exclusions: []`, reviewed exclusions `{}`.
- `uk_weights_audit` still evidence-absent (`fit_weight_records`): the
  collector fix lands in the spine builder, so the evidence appears only in a
  spine built from `dc42375b` onward; spine-s cannot be retrofitted.
- Unchanged: `uk_export_surface` (June allow-list and
  `UK_CANDIDATE_DATASET_NAME`), `uk_degenerate_release_surface` (two all-zero
  columns), `uk_nonnegative_columns` (239 negative
  `housing_water_and_electricity_consumption` values). The twelve R3 passes
  still pass.

The certifier-evidence group from R3 is closed except for the spine rebuild.
What blocks a certified cut is now entirely María's calls: the export
allow-list and dataset name, the two input-mass breaches, the two all-zero
columns, the negative consumption values, and a spine rebuild for the weights
audit.
