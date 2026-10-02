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
  On the calibrated weights (review item 4, `scripts/residential_readout.py` on the C2 solution): the
  two bound national totals hold (count 202,628 against 202,630, gains £12.246bn against £12.242bn),
  and within them the solver moves mass between bands: £250–500k 3,680 → 4,256 taxpayers, £500k–1m
  2,004 → 2,173 (gains £1.25bn → £1.67bn), £1–2m 730 → 552 (£0.94bn → £0.77bn), £2–5m 427 → 274
  (£1.24bn → £0.87bn), the open band 134 → 139; the bands below £250k move by under 3%. So the
  by-band identities are a design-weight property; the release carries the national totals. Row
  level with the arms in place: the smallest positive calibrated weight is 0.012 (an arm, 0.050; the
  smallest design arm 0.144), the positive-weight median 26.6 (arms 11.8), no row at zero.
- Stack (ebff84125, the final-build candidate): the spine passed its gates in six minutes; the first
  calibration refused the Chronicle artifact 505e0e7 because the stack pins the pensions feed
  (facts 28b7105…, artifact 825406f). Against that feed the dense loss is 0.00818 (main 0.00836), but
  the seam battery blocked the H5 on one cell with no support at all:
  `dwp/uc_payment_dist/SINGLE_annual_payment_28_800_to_30_000@2025` (single, no children, monthly
  award £2,400–2,500; target 1,654), initial estimate 0 against 763 at design weights on every
  main-based arm and on every pensions arm. Traced: the cell is supported by one FRS household (a
  disabled single council tenant in Yorkshire, design weight 760, with its incidence clone at 3.3); on
  the stack its UC award is £678 a year lower, which moves it into the 27,600–28,800 band that the
  uk-data#452 class already excludes. Nothing on the household's own inputs changed; its consumption,
  energy, land and savings draws all moved at once, because the SPI income band donors now insert
  1,920 rows where 480 stood and every stage drawn after them that is not identity-keyed reads its
  rows in a new order. The cell joins the measure exclusions beside its sibling (approved 2026-10-02,
  expiring 2026-11-26 with the class, for her signature).
- Stack 2 (558325016, after review round 1): the spine passes; dense loss 0.00769; the battery blocked
  on two further cells. `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_22_800_to_24_000`
  has two supporting households on every main-based spine (design weights 1,025 and 1,999 against
  2,761) and both awards moved out of the band when the stages drawn after the reworked WAS chain read
  their rows in a new order; it joins the uk-data#452 measure exclusions. `hmrc.spi_region.
  income_tax_by_region_12570_15000@E12000008@2025` (South East) sits on its target at design weights
  is 11% short at design weights (69.6m against 78.3m; 62.4m on main, 58.3m on stack 1) and the
  solver pulls it to +32.0%. The pull comes from the national employment-income bands: at design
  weights the £12,570–15,000 and £15,000–20,000 employment-income bands are 20% and 31% short (the
  earners gap the 2026-09-30 build found), the solver fills them by raising the weights of low-paid
  employees everywhere (three-fold on the 200 households that carry most of this cell's gain), and
  the South East income-tax cell, carried by the same households, overshoots while its total-income
  sibling lands at +11.2% (main pulls the same cell from −20% to +22.6%). The cell is deferred four
  weeks in the target-fit register (it was deferred on 2026-09-18 and retired on microcosm#1012 at
  +24.5%); the fix is on the employment-income side, not on the draws. Separately, every regional
  £12,570–15,000 income-tax cell moves 10–30% at design weights between spines that share the same
  incomes (North West 63.0m, 44.5m, 52.0m; East 41.3m, 53.5m, 53.2m) because the salary-sacrifice and
  pension draws after the SPI chain are positional; identity-keying those draws would make the
  design values stable between spines but would not move where the solver lands.
- Stack 3 (2fe7d8bbb, the Table 3.6 amounts projected by HMRC's band growth): all 7 seam gates pass
  and the H5 is written; dense loss 0.00713 (main 0.00836). The three lowest employment rows fit at
  −3.9%, −11.9% and −10.0% (stack 2: −8.9%, −23.1%, −18.9%; main −9.9%, −23.2%, −19.6%). The South
  East £12,570–15,000 income-tax cell lands at +30.3% under its deferral; the North West twin at
  +14.4%. Folded weights: max-to-median 1,028 (main 609, stack 1 710, stack 2 1,116; bound 1,151),
  ESS 4,953 (main 5,191, stack 1 5,222), median positive weight 40.0 (main 55.8), heaviest household
  41,085 (an East Midlands household in its thirties carrying 10–12% of that region's 30–39
  population, £20–30k taxpayers and band-B council tax rows; on stack 1 the heaviest was an 85–89
  Pension Credit unit at 29,690). The rise in the fold between stack 1 and stack 2 came with the WAS
  mortgage rework and the first thin-cell exclusion; the margin to the bound is now thin and belongs
  in the review.
- Final build (2fe7d8bbb): running at the time of writing; the readings follow.

The South East cell, with the count side read: the frame has 0.278m South East taxpayers with total
income in £12,570–15,000 at design weights against HMRC's projected 0.364m (nationally 2.31m against
2.97m; North West 0.240m against 0.333m, London 0.235m against 0.340m, East 0.256m against 0.273m),
and the ones it has pay £250 of income tax each against HMRC's £215. The bound count rows make the
solver scale these households up by a third to hit the count (final error 0.000), the total-income
sibling overshoots to +11% and the tax cell to +30%. So root 2 is a deficit of taxpayers just above
the frozen allowance, concentrated in the South East, London and the North West, not an employment
problem: the fix is the lowest band's support in the frame (the SPI support channel's coverage of
that band, which the pension-age prior share of 0.2 reduced, and the FRS's small incomes around the
allowance), a spine change for the follow-up.

Child Benefit children in payment (review item 5): the trial's 11.1m is at design weights on the
1 October spine, where the eligible-child base is 14.04m; the claim rate (86.7%) and the opted-out
family share (9.0%) match HMRC, and the opted-out families carry 1.82 children each, so claimed
children are 12.17m and 1.09m of them are in opted-out families. On the 2026-09-30 build's calibrated
weights the eligible base is 14.79m, which at the same rates gives 11.7m in payment against HMRC's
11.73m: the gap is the design-weight child base, not the rates or the opt-out ages.

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

## The employment-income band shortfall (ruling 2026-10-02: fix, do not defer)

The three lowest `hmrc/employment_income_income_band_*` rows are SPI 2023-24 Table 3.6 amounts: the
employment income received by taxpayers whose total income falls in the band, projected to 2025. The
frame's taxpayer counts and total income in those bands sit within a few percent of HMRC; what was
short was the share of the band's income that is pay, 20–31% at design weights. Two roots:

- The registry projected the Table 3.6 amount rows with flat national indices (average earnings for
  employment, mixed income for self-employment, the pension index for private pensions) while the
  count, total-income and tax rows by band use HMRC's own band-specific growth (Income Tax
  Liabilities Table 2.5). An amount held in a fixed nominal band moves with the band's membership:
  HMRC projects the £12,570–15,000 band's total income +5.0% from 2023-24 to calendar 2025, the
  £15,000–20,000 band's −2.3% and the £20,000–30,000 band's +0.3%, against the flat +10.7%. The
  39 amount references now use the band growth; the three lowest employment targets move from
  £18.7bn, £53.8bn and £190.4bn to £17.7bn, £47.5bn and £172.5bn, and the frame's design-weight gaps
  from −19.8%, −31.2% and −25.1% to −15.5%, −22.1% and −17.4%. Dividend and savings-interest amounts
  keep their own indices (they move with rates, not membership). The SPI tape re-banded under the
  engine's own indices agrees with HMRC's direction in every band.
- The remaining gap is the frame's composition of the low bands. Banding the frame on core income
  only (pay, profits, private pensions, State Pension), its earners in the £12,570–15,000 band are
  1.36m, HMRC's 1.36m exactly, with £17.8bn of pay against the band-aware £17.7bn; adding the
  non-employment leaves (savings interest, dividends, property, miscellaneous) lifts 0.14m of them and
  £2.2bn of pay out of the band, and the engine's broader banding concept a further £0.6bn. In the
  £15,000–20,000 band the core-only comparison still leaves pay 8.6% and earners 14% short, with State
  Pensioners 7% above HMRC's count. The places to look, in order: the SPI stage-2 rewrite of dividend,
  property and savings income onto FRS rows (`hmrc_spi_income_spine`; on the tape the low earners who
  sit in these bands carry about £350 of such income, the frame's FRS-channel low earners show 7% with
  dividends averaging £17,500), the total-income concept the band measure uses against SPI's, and the
  SPI channel's age mix in the low bands (58% of the band on main, 43% on the stack). Spine work, a
  follow-up.

The South East £12,570–15,000 income-tax cell's +32% is this pull: the solver raised the weights of
the low-paid households that carry it (three-fold on the 200 that carry most of its gain) to fill the
under-projected employment rows; its own target is already band-aware.

## Review round 1 (Vahid, 2026-10-02)

- The five internal disability carriers leave the export allow-list and are dropped at the release
  boundary with `incapacity_benefit_reported`; the input-mass evidence pin and the gate digests follow
  the c9 register entries.
- WAS mortgages: only the main-residence mortgage (`TotMortR8`) is tenure-stratified; the mortgages
  on other property (`HMortGR8` less `TotMortR8`) are drawn on every tenure without a stratum, and
  `mortgage_debt` is derived as their sum, so a renter's or an outright owner's buy-to-let keeps its
  mortgage beside the property. The chain runs eight segments; the coherence gate holds the
  main-residence mortgage off the mortgaged tenure at zero and the mortgage identity on every row.
- The split's docstring and notes say what calibration binds: the two national residential rows.
- The plan-2 renewal stays on this PR (her ruling).

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
