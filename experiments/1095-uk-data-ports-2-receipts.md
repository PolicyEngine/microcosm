# microcosm#1095 receipts: the second ports PR

Plan: `repos/microcosm-1095-ports-2-plan.md` (approved 2026-10-05). Branch `uk-data-ports-1095-2`, off
main `07bbb656b`, rebased on 2026-10-06 onto main `581b5699e` (which merged #1107), on 2026-10-07
onto main `a50b9cc0d` (which merged #1081), again on 2026-10-07 onto main `966f0d9d7` (which merged
#1111, #1118, #1119, #1120 and #1129), and on 2026-10-08 onto main `e3e3d881f` (which merged #1125,
#1128, #888 and #1115). Each rebase replayed every commit, regenerating the coverage manifest, the
stage digests and the H2 fixture after each one. Each commit's other changes are line-for-line
those of its pre-rebase original, with three exceptions:
- in the second rebase, B4's removal of `CVPAY` from `property_income` became #1081's
  `frs_property_income`, which already leaves it out, and B4's spine test keeps both rules'
  assertions;
- in the third, B0's policyengine-uk concept coverage was regenerated on main's new concept schema
  (the liquid financial assets concept), still 285 inputs with 40 covered;
- in the fourth, B0 also reviews that concept's unmapped reason, and #1120's engine test of it,
  against policyengine-uk 2.122.2. The third rebase had left both on 2.100.0's facts, and CI's
  `engine-uk` lane failed on the test from then on. 2.122.2 has two more person-level money stocks,
  `car_list_price` and `lifetime_isa_balance`. Universal Credit now splits household capital by
  claimant and partner count and counts a Lifetime ISA as person-level capital. The concept stays
  unmapped.

Max ruled on #1086 on 2026-10-07 (option 2): each country carries its own policyengine-core pin.
P, a lock-only commit that moves no engine version, opened this branch until it landed on its own
as #1143 (merged 2026-10-08), as the ruling asked. B0 was replayed onto it and re-locked from its own
lock, so the UK fork keeps policyengine-uk 2.122.2 with core 3.32.19 and the US fork keeps core
3.32.5. Nothing else in the lock moved, and every later commit replays unchanged. On 2026-10-08 the
branch was rebased onto main `4cb43524a`, #1143's merge, whose tree is P's, so the head's tree did not
change.

Commits, in order:

- P, now #1143 (merged 2026-10-08): per-country policyengine-core pins (#1086). The extras that install
  policyengine-us pin core 3.32.5 and conflict with the extras that install policyengine-uk, so uv
  locks the two sides apart and refuses an install that asks for both. A lock-level test reads
  `uv.lock` through `uv export --frozen`.
- C1 `4092df00e`: the national target references regenerate from the pinned feed again (#1089 had
  hand-edited the generated file). Cherry-pickable alone.
- C2 `a08357994`: a calendar-month access-fund award above twice the largest annual-coded award is
  read as annual (Vahid's should-fix on #1100).
- C3 `04fdc3cfc`: the LCFS household counts are named for what they are and counted from `age`; the
  stage notes are quoted in full.
- C4 `53c112f23`: Great Britain pension-age Housing Benefit spending (`dwp.hb.amount_pension_age`)
  at the calendar-2025 window, through the new selector key `source_measure_id_by_opening_year`.
- C5 `7a67b506d`: `is_blind` from the FRS blind registration question (uk-data#523).
- C6 `c9d995466`: `uc_is_in_startup_period` from the FRS claim and trade start dates (uk-data#527).
- B0 `c5cdf2d04`: policyengine-uk 2.122.2 and policyengine-core 3.32.19 on the UK side of the lock.
- B1 `d7b42f944`: the severe disability flag stored as the tax credit condition (uk-data#494).
- B2 `6900190ee`: `is_claimant_or_partner` and `is_hbai_dependent_child` from the adult and child
  tables (uk-data#524/#486).
- B3 `37775c25d`: `pension_credit_reported_capital` (uk-data#513).
- B4 `e98503389`: shared rent liability and the rent boarders and lodgers pay (uk-data#511/#512/#506).
- B5 `3a90d8d0b`: the mixed-age couple Pension Credit saving at survey-year facts (uk-data#519).
- X `57a0ccf2f`: the West Midlands £12,570–15,000 income-tax cell deferred four weeks in the
  target-fit register (approved 2026-10-06).
- Y `caae8de35`: the two UC payment bands policyengine-uk 2.122.2 empties join the uk-data#452
  measure exclusions (approved 2026-10-07, expiring with the class on 2026-11-26).
- R1 `a97df8fc6`, from the review of #1121: the access-fund repair is recorded in the root stage's
  checkpoint evidence (paid awards by period code, the threshold and the repaired count), and a tab
  with calendar-month awards but no annual-coded award refuses.
- L `b5c73d12b`: X's register entry names its upstream fix, pe-uk#2174 (the review of #1121
  agreed the fix belongs in policyengine-uk's `total_income`).

## Where the plan changed during implementation

- **C5, children.** The plan said children stay False. The FRS 2024-25 child table asks the same
  `SPCREG1` question (fewer than 10 records registered), and the engine reads `is_blind` for
  children in Tax-Free Childcare, so the rule reads both tables.
- **C6, the draw.** `frs_spine`'s notes said stochastic flags are absent from the root stage. The
  start-up period needs raw claim, interview and job dates that only this stage reads, so it now
  carries one identity-keyed draw, declared as `assign_binary_from_rate`; the notes say so.
- **B1, mirror rather than stop persisting.** The plan preferred handing the flag to the engine's
  formula. The incumbent coverage manifest marks it required and the export surface reference holds
  it, so stopping would need a signed reclassification and an export exclusion. The stored flag now
  equals the formula by construction (the engine-lane test checks every category), which is also
  what policyengine-uk documents for datasets.
- **B2, the UC take-up population.** Max's WIP branch `uk-take-up-per-person-spa` rewrites it to read
  each person's `is_WA_adult` from the engine. This PR leaves it to that branch; B0 only keeps the
  current design working (the State Pension age now comes from the timetable by date of birth).
- **B4, CVPAY.** #1081 merged first (2026-10-07), so B4 is rebased on it and keeps only the payer's
  `rent_paid_as_boarder` / `rent_paid_as_lodger` and the shared-rent liability. The householder's
  receipt stays uncredited, as #1081 documents: policyengine-uk 2.122.2 has no householder-side input
  for it.

## Measurement

Each arm built the spine from its commit in the measurement tree and calibrated it on the national
role, with local staging only and Chronicle feed 825406f. The control is A9 (`d89a3ddfe`, main's code
before this PR).
Arms B2 and head were built from `8e81ddfd5` and `65a08ddb6`, before the rebase onto #1081. Their
spines therefore lack #1081's property-income rules (losses count as zero, the sub-letting rent is
floored), which a calibration of the current head would include.

- **Arm E (`52f164c41`, C1–C6 before the rebase).** Loss 0.00702 (A9 0.00715). 99.0% of 1,169 targets
  are within 10%, ESS 4,848, and all 7 calibrated-seam gates pass.
  - `dwp.hb.amount_pension_age`: target £7.049bn, design £5.636bn, calibrated £7.052bn.
  - `obr.vat`: +24.6% (A9 +24.4%), still the watch item.
  - Input mass passes. `access_fund` sits at £65.9m against the reference's £159.6m; this is the C2
    repair.
- **C3 is value-identical.** The 14 non-raked LCFS outputs differ for 137 of 63,936 common households,
  and every one is explained by an upstream predictor move:
  - 76 have a member in a UC start-up period (C6 moves UC, and so HBAI net income);
  - 61 have a registered-blind member (C5 reaches HBAI net income through the Council Tax Reduction
    non-dependant exemption);
  - fewer than 10 have a survey-income change.

  The four raked columns move for every household by the global raking factor.
- **Arm B0 (built as `f6e73f61d`, the engine bump).** Loss 0.00769, 98.4% within 10%, ESS 4,843; 6 of 7 gates
  pass.
  - `uk_target_fit` fails on two UC payment bands that hold no benefit unit under policyengine-uk
    2.122.2: singles at £26,400–27,600 a year and childless couples at £24,000–25,200.
  - Each rests on fewer than 10 FRS households. The same spine under policyengine-uk 2.100.0 keeps
    every supporting award in its band, so the engine alone moves them out, through three
    corrections: the published LHA rates move awards to neighbouring bands, the 2019 mixed-age
    couple saving keeps couples who report Pension Credit off UC, and Carer Support Payment counts
    as unearned income. The register already held 19 UC payment-band exclusions of this class
    (uk-data#452).
  - `obr.pension_credit` at design weights falls from £4.61bn to £4.13bn.
  - `obr.vat`: +23.8%.
- **#1107's rollout checks on arm B0.**
  - The Child Benefit stage exports registered claims (`opt_out_charge_share` 1).
  - The fully charged opt-out pool is exhausted: 599.6k families against the 627.0k target, a 27.4k
    shortfall.
  - In payment at the stage's design weights: 6.32m families (published 6.87m) and 11.67m children
    (11.73m).
  - `obr.child_benefit` calibrates to 0.0%.
- **Arm B2 (built as `8e81ddfd5`).** Identical fit to B0. The stored severe flag drives only tax credit elements
  the model does not pay in 2025, and the supplied person types equal the engine's inference.
- **Arm head (built as `65a08ddb6`).** Loss 0.00765, 98.3% within 10%, ESS 4,957; 6 of 7 gates pass.
  - `uk_target_fit` fails on the two UC bands.
  - It also fails on `hmrc.spi_region.income_tax_by_region_12570_15000@E12000005`, West Midlands
    income tax at £12,570–15,000: target £55.1m, design £339.2m, calibrated +27.4%. The cell was
    +8.6% on A9, +8.9% on E and +23.6% on B0 and B2.
  - `obr.vat`: +23.4%.
- **The West Midlands cell, diagnosed.** The regional cells slice on policyengine-uk's
  `total_income`, which leaves out `other_investment_income` although `income_tax` charges it as
  savings interest.
  - Fewer than 10 SPI records whose income is mostly other investment income sit in this band with
    their whole tax. The design value is 6.1 times the target (6.0 times on A9; every other
    region's cell is 0.6–1.1 times).
  - Banded on total income including other investment income, the cell falls in line with the
    other regions.
  - The engine bump moves the solver's pull from +8.9% to +23.6%. A counterfactual arm, the head
    with CVPAY back in property income (never pushed), lands at +25.4%. So CVPAY is about 2 of the
    3.8 points from B2 to head, and the rest comes from B3–B5 moving the calibrated weights.
  - X defers the cell four weeks (expiring 2026-11-03) while policyengine-uk's `total_income` is
    corrected upstream to include other investment income (pe-uk#2174, filed 2026-10-07).
- **Y excludes the two UC bands.** Replaying the target-fit gate on the head arm's evidence without
  them passes, with X in force, and the other six seam gates passed on that arm. The replay keeps
  the head arm's weights, which were solved with both bands bound; a calibration of the new head,
  the national build and the certifier have not been run.

## Verification

- Touched-surface tests per commit (counts in each commit's run).
- B0: the full suite on policyengine-uk 2.122.2, 13,372 passed, 119 skipped. The two failures do
  not come from the bump: the extras test expects no UK engine, and the live Hugging Face pointer
  now names `microcosm-uk-2024-25-national`, which fails on main too.
- Head: the full suite, 13,378 passed, 128 skipped, and the same two environment failures.
- `ruff check` and `ruff format --check` pass on all changed Python files.
- `tools/ci_test_plan.py verify` and `tools/graph_acceptance_burndown.py --verify` ok.
- #1086: on the shared lock, CI's US lane failed one test on both Pythons
  (`test_batched_scoring_matches_whole_pool_policyengine_us`, a Maryland child-care rate lookup under
  policyengine-us 2.2.1 on core 3.32.19). With P the US fork stays on core 3.32.5: a US install
  from the new lock resolves policyengine-us 2.2.1, core 3.32.5 and spm-calculator 1.0.0, and
  `test_us_post_export_scoring.py` passes there (4 tests, including that one). The lock-level test
  passes at P and at B0.
- Review round 1 (2026-10-07): CI's `engine-free` lane failed on `a0f292aec` in
  `test_fully_armed_battery_evaluates_gate_for_gate`. The battery's register clock (2026-10-04)
  predated X's approval, so the gate read X as not yet in force. X and Y now move the clock to
  their approval dates, as the test's note asks. The engine-free CI job's 496 test files were rerun
  locally on the rebased head (counts in the round's run).
- Rebase onto `e3e3d881f` (2026-10-08): from the third rebase on, CI's `engine-uk` lane failed on
  #1120's liquid-asset test (the fourth exception above). On the rebased head the lane's 40 test
  files pass in full, run past the lane's stop at the first failure: 170 passed. The `engine-free`
  lane's 509 test files, with `tools/orrery-contract` installed as CI does: 14,929 passed, 133
  skipped, and the same two environment failures.
- Review rounds 2 and 3 (2026-10-07 and 2026-10-08): C6's linked share is now a household-weighted
  average, `np.average` over the linked rows with the `gross4` household weights. It is not a microdf
  `MicroSeries`, because microdf reaches an environment only with an engine and the FRS spine runs
  in the engine-free lane. B0's forward-year note cites pe-uk#2173. The FRS spine tests pass (101),
  and no generated surface moved.
- Head calibration: queued on 2026-10-08 at tree `c7fe76e8a`, which is this head's tree. Its results
  are owed before merge.

## Signature drafts

`data/ukds/acceptance/1095-uk-data-ports-2/signed-difference-drafts.md` (not committed until signed).
