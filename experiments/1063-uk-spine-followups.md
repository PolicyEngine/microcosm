# microcosm#1063 receipts: UK spine follow-ups

Plan: `repos/microcosm-1063-implementation-plan.md` (approved 2026-10-01) and its continuation
(2026-10-02, rulings R1–R3 below). Branch `uk-spine-followups-1063`, stacked on `uk-pensions-1069`
at 004b5c9cc (PR #1084); the PR targets that branch until #1084 merges.

Commits, in order:

- 80261bf11 c1: the SPI income band donors as a mass-conserving support channel (items 1, 2, 4).
- e900bb68b c2: `mass_increasing_support` retired from the coverage validator (item 3).
- cccec91bf c8: `child_benefit_take_up`, a new stage after the Pension Credit redraw (fold-in).
- 245702287 c7: salary sacrifice predicted only for people who are paid (fold-in).
- ba702f3e0 c3+c4: WAS wealth coherent with tenure and with its own totals (items 5, 6).
- 9161ad6f5: the identity receipts (E7, E8) pass on a spine from main.
- b98667cfb c5+c6: the per-band residential walk and Table 7 by claimant status (items 7, 8).
- c53686355: the two plan-2 measure exclusions renewed to 2026-11-03 (R1).
- 991134b0c c5b: `cgt_residential_split`, the residential flag carried as weight (item 7, R3);
  replaces the walk of c5.
- 48eb73bf4 c9: the release-cut certifier's blockers (R2).
- c10: this note.

## Rulings (2026-10-02)

- R1 plan-2 measure exclusions: renewed one month (approved 2026-10-02, expires 2026-11-03), the
  coverage gap staying with microcosm#868.
- R2 export surface: every declared stage output the incumbent lacks is allow-listed; only
  `incapacity_benefit_reported` is dropped, at the release boundary.
- R3 residential flag: no closing rows, sub-bands or weight-keyed eligibility; the probability is
  carried as weight (the incidence clone and US PUF-clone pattern), calibration decides.

## Builds

Every arm is a licensed local build in the detached measurement tree `repos/populace-1063-shard`
(`data/ukds/acceptance/1063-spine-followups/scripts/arm.sh <commit> <name> [calibrate]`): the spine
with the build_930 recipe (the five NTS tabs, `--was-person-tab`, `--no-staging`), then the national
calibration (local staging only) against the Chronicle artifact 505e0e7 and the 2026-09-30 build as the
incumbent. Nothing is released from an arm. The engine is policyengine-uk at the repo's pin.

The arms, in order:

- C (e0b41a59b, the walk): all 27 spine gates pass. Residential gains gap £38.8m, 0.3% of the £12.2bn
  target (main: £469m, 3.8%); count gap 1,428 against a bound of 10,331; the ten largest flagged stakes
  carry 18.5% of residential gains (22.7% on the 2026-09-30 build); largest flagged stake £557m; the
  largest stake in the pool £10.9bn, 89% of the target. The open band (£5m and over) holds 84 rows with
  £47.7bn of gains, 5 flagged, achieved £1.27bn against £1.44bn expected: the deterministic bound of a
  band whose heaviest row weighs 254 households and whose largest gain is £522m is vacuous (£132.6bn),
  which is why the walk was replaced (R3). Table 7 by claimant status: claimants realise 87.2 / 5.3 /
  7.5 against the same targets to five decimals; non-claimants within 0.4 points of their restored
  shares. Calibration: loss 0.00853 (main 0.00836), ESS 5,217 (5,198), max-to-median weight 1,115
  (1,079; bound 1,151), projection entrants 6,715 (6,711; bound 73,000), 1,090 targets.
- B (b561270df, WAS and identities): all 28 spine gates pass including the new
  `uk_stage_was_wealth_coherence`: zero mortgage debt off mortgaged tenure, zero main-residence value
  off owner tenures, zero owners without one, zero identity violations. The rule drops the donor's
  off-tenure mortgage mass, £127bn of £1,266bn (10.1%), all other-property mortgages. Mortgaged
  households owing more than their home: 5.4% against the donor's 2.7% (26% on the 2026-09-30 build).
  Renters with property wealth: private 16.5% against the donor's 10.3%, social 4.6% against 1.1%
  (10.7% before). The calibration stopped before an H5 on
  `hmrc/state_pension_income_band_50_000_to_70_000@2025`, a target-fit exclusion inside its bound on
  this spine, which the pensions branch retired at b19f14d98; dense loss at epoch 1,500: 0.0087.
- C2 (0ce1a5235, the residential split, on the arm C spine): all 28 spine gates pass. 2,276
  households carry a liable gainer, none more than one, so 2,276 residential arms; the residential
  count and gains are identities on Table 8a in every band (solve error 1e-16 on the count, 7e-10 on
  the gains, identity error 0); 106 arms weigh less than one household (smallest 0.14, largest
  1,263); the ten largest residential arms carry 9.4% of residential gains (18.5% on the walk, 22.7%
  on the 2026-09-30 build); the largest residential arm stake is £176m (£557m flagged on the walk);
  the largest liable stake in the pool stays £10.9bn, 89% of the target. Calibration: dense loss
  0.00909 (C 0.00853, main 0.00836); on the family fold the max-to-median weight is 610 (C 618, main
  609) and the ESS 5,207 (C 5,209, main 5,191), with 7,028 rows folded (4,752 support copies and the
  2,276 arms); the unfolded ESS fraction falls from 0.0895 to 0.0862 because the arms are extra rows.
  The seam battery blocked the H5 on `uk_target_fit`: the North West £12,570–15,000 income-tax cell
  (`hmrc.spi_region.income_tax_by_region_12570_15000@E12000002@2025`) at +25.3% against the 25% bound
  (23.2% on main, 22.6% on C; its South East twin 22.6% and 23.3%: the regional band cells hover just
  under the bound on every arm and the solver moves a point among them), and the stale
  `hmrc/state_pension_income_band_50_000_to_70_000@2025` exclusion back inside the bound (24.9%; the
  pensions branch retired it at b19f14d98, so the stack does not carry it).
- Stack (ebff84125, the final-build candidate): the spine passed its gates in six minutes; the first
  calibration refused the Chronicle artifact 505e0e7 because the stack pins the pensions feed
  (facts 28b7105…, artifact 825406f), and was re-run against it; readings below when they land.

Child Benefit (c8, trial on the 1 October spine at design weights): 86.7% of eligible children claimed
for (HMRC 86.6%; the frame's age mix implies 87.1%), 9.0% of claiming families opted out, all from
families charged the whole benefit, 11.1m children in payment against 13.5m before.

## The certifier rehearsal (R5, 2026-09-30 build, 15 passed / 5 failed of 20)

- `uk_export_surface`: label `microcosm_uk_2024`; `household.household_weight` reported missing and
  every id column reported as an extra (the certifier read every frame column); 49 columns outside the
  enhanced-FRS surface without an allow-list entry; `incapacity_benefit_reported` to be dropped.
- `uk_degenerate_release_surface`: `incapacity_benefit_reported` and `is_in_approved_training`
  all-zero. FRS 2024-25 (SN 9563) carries `TRAIN2` (codes 1–8 and 10 are schemes, 9 none of these) and
  `TRAINEE` in place of `TRAIN`; 28 adults report a scheme, so the column carries signal once read.
- `uk_nonnegative_columns`: `housing_water_and_electricity_consumption` 250 rows below zero (minimum
  −14,165), inherited from LCFS donor rows whose diary nets refunds below zero; the support clip took
  its floor from the donor's realised range.
- `uk_input_mass_parity` against the enhanced FRS: `adult_ema` +786% (38.9m vs 4.4m),
  `dfe_education_spending` +187,061% (£98.8bn vs £52.8m).
- `uk_qrf_tail_concentration`: `charitable_investment_gifts` top-100 rows carry 100% of the mass over
  168 carriers.

c9 addresses each at its source (see the commit); the certifier is to be re-run on the final build.

## Residential split (c5b)

Each liable gainer's residential probability `p` (the logistic solved to Table 8a's individuals-basis
count and gains, with the stock shift) is carried as weight: a household with one liable gainer becomes
a residential arm at `p·w` beside the incumbent at `(1 − p)·w`; with `k` gainers, `2^k` arms at
product weights (`k > 3` refused). At design weights the residential count and gains are identities on
Table 8a in every gain band (no draw, no seed, no error bound); the asset-type stage types the arms and
keeps the BADR claims and the Table 7 fit. The £522m copy (weight 20.8, `p` 0.2%) becomes a residential
arm at 0.04 households beside a non-residential arm at 20.76. Family folds, the geography identity
kernel, the export surface, national sampling, the student-loans receipt and the E8 identity receipt
carry the new layer.

## Still owed

- Arm C2 and the final build with the certifier; the identity tool with `--spi-tab` on it.
- Her calls: Child Benefit opt-out pool order (fully charged first, then the taper, as built; or one
  pool over every charged family); `other_residential_property_value` (WAS `DVHseVal`) excludes
  buy-to-let (`DVBltVal`), which sits in the drawn remainder; the Table 8a rows stay fenced
  (promotion to bound targets is the #970 leverage question, after the final build).
- Findings outside this PR: SPI-channel rows receive `child_benefit_reported` from the stage-2 QRF
  without regard to children (1.13m weighted reporter units with no eligible child); the frame has
  0.67m reporter families with an adult above the £60,000 charge start against HMRC's 0.31m–0.44m
  liable individuals; the E8 donor recompute takes the licensed tape (`--spi-tab`) rather than storing
  the propensity table in the receipt (small cells).
- policyengine-uk: #1996 (VAT) filed; `is_renting` omitting `RENT_FROM_HA` and the engine ignoring
  `child_benefit_opts_out` to file on her go.
