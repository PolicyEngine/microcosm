# microcosm#1095 receipts: the third ports PR

Plan: `repos/microcosm-1095-ports-3-plan.md` (approved 2026-10-09). Branch `uk-data-ports-1095-3`, off
main `d50220d1f` (after #1121 and #1160). It was rebased on 2026-10-09 onto main `a224dc110` (which
merged #1087, #1162, #1144 and #1147) and replayed on 2026-10-10 onto main `7639ef8b0` (which merged
#1161, #1169, #1171, #1172 and #1180 to #1182), with María's rulings of that day folded into D1, D3 and
T1. Each replay regenerated the coverage manifest, the stage digests and the H2 fixture after every
commit, and every other change is line-for-line its pre-replay original. Each target commit carries the
national references and the compile-parity receipts regenerated against Chronicle feed 825406f.

Commits, in order:

- R0 `232eac6fa`: #1121's head calibration (arm head-r3) recorded in its receipts.
- D1 `18d1fa560`: `uc_is_in_gainful_self_employment` from the FRS main job and earnings (uk-data#525).
- D5 `3b41d6156`: `is_looked_after_by_local_authority` from the FRS foster links.
- D6 `533703ffa`: loss-making landlords stay in the CGT residential signal.
- D2 `d026554b2`: the UC take-up population on policyengine-uk 2.122.2's age rule (uk-data#486, #524).
- D3 `6dd8dac18`: property in UC and Pension Credit recorded capital, with UC reporters keeping their
  receipt.
- D4 `c5f358230`: Pension Credit take-up solved over Great Britain, with newly entitled units drawn
  (uk-data#510).
- T5 `d04ab10c7`: SPI savings interest and its band targets at the engine's projection index (uk-data#541).
- T6 `14ae2c349`: the PIP daily-living caseload rows scoped to England and Wales.
- T3 `6f4c3bcbf`: `obr.council_tax` on OBR table 4.1 row 15, over Great Britain.
- T4 `e1ee7f0ca`: `obr.ni` a diagnostic beside the class rows (uk-data#537).
- T2 `b85d5a4a3`: the salary-sacrifice relief bands by income tax rate class (uk-data#533).
- T1 `a2e5adbbf`: UC award bands in whole pence (uk-data#530).
- T6b `e3ac4a5ce`: the two PIP caseload rows held as diagnostics on `obr.pip`.
- W `a7e5bc56f`: the West Midlands low-band income-tax deferral retired.
- R: this file.

## María's rulings of 2026-10-10

The first head (below) failed two calibrated-seam gates. After its attribution she ruled:

- **D1.** SPI-channel rows take the earnings route alone: a profit above zero and above pay. The SPI
  draw does not read the donor's employment status, so on those rows the status says nothing about
  the drawn trade. The main-job route returns when the SPI draw follows employment status
  (uk-data#529, microcosm#840).
- **D3.** A unit that reports Universal Credit keeps its receipt and records no UC property share.
  Pension Credit keeps property.
- **T6.** The two PIP daily-living rows are diagnostics on `obr.pip`: checked, never fitted (T6b).
- **T1.** The whole-pence band edges stay. The "£2,500.01 or over" band stays unbound, as on main.
- **The support floor** for sparse UC bands moves to its own PR. This PR no longer needs it to pass.

W is not one of her rulings. The gate requires it on this head (see Measurement), and it follows the
earlier retirements of stale deferrals.

## Where the plan changed during implementation

- **D5, the allowance.** The plan asked for the carer's fostering allowance (ALLOW3) as well, if the
  counts supported it. FRS 2024-25 records 13 children under 16 as looked after, and ALLOW3 covers
  fewer than 10 of them. So the rule reads the relationship code alone: foster child (5) toward an
  adult in the child's own benefit unit, or foster parent (9) from that adult toward the child.
- **D2, Max's scaffolding.** The plan reused Max's test scaffolding
  (`test_support/microcosm_build/uk_take_up_population.py`). It is not on main, so D2 uses none of it.
  His `uk-take-up-per-person-spa` branch solves a different part, the State Pension age by date of
  birth that builds from 2026-27 need, and it will need a rebase onto D2.
- **D3, one share helper.** The plan pointed the UC share at `household_family_role_counts`. That
  helper counts claimants and partners only, and the Pension Credit share needs members at or over its
  qualifying age. So one helper, `_household_property_share`, takes each proxy's owner mask and serves
  both.
- **D4, the entitlement year.** The stage stores the mixed-age saving at the survey year. It then reads
  entitlement, Pension Credit capital and deemed income at the calibration year through a second engine
  read, as uk-data does.
- **T3, the geography pin.** The generator's substring rule reads "scotland" in the row's concept,
  `obr.council_tax_receipts_england_scotland_wales`, and would pin Scotland. A new exact-id pin keeps
  the row at the UK stamp Chronicle gives OBR's table.
- **T3 and T4, the parity rationales.** Their compile-parity rows carry their own reasons: a different
  series for council tax, and the ruling for the NICs total, as `obr.state_pension` has.
- **T2, the total.** The Table 6.2 all-rates total became a diagnostic on the higher-rate band, the
  band that carries most of the relief.
- **T6, diagnostics by month.** The caseload facts are dated by month, so diagnostic references now
  accept a month-dated fact.
- **T1, the top band and the floor.** The plan bound the top band and added a support-floor rule. The
  measurement below changed both.

## Measurement

Each arm built the spine from its commit in the measurement tree and calibrated it on the national
role, with local staging only and Chronicle feed 825406f (rebuilt with the original's manifest). The
control is main `d50220d1f`. Figures are aggregates; cells under 10 records are suppressed. The
family-folded weight ratio is the largest support-family weight over the positive median, against the
June fence of 1,151.

### Before the rulings

D1 and D3 were as first built here: SPI rows flagged by the donor's status, and property counted for
every unit.

- **Control (`d50220d1f`).** Loss 0.00712, 98.3% of 1,167 targets within 10%, ESS 4,874, weight ratio
  1,059. All 7 calibrated-seam gates and the certifier's input-mass gate pass.
- **Arm 1 (built as `341a02bd4`: D1, D5, D6).** Loss 0.00701, 98.5% within 10%, ESS 4,846, weight ratio
  1,129. All 7 gates and input mass pass. Design UC falls from £52.55bn to £51.96bn.
  - D1, at the arm's calibrated weights in 2025: the minimum income floor applies to 1.66m FRS-channel
    people (control 1.84m) and 2.49m SPI-channel people (control 1.73m).
  - D1, the SPI channel: 1.67m weighted SPI-channel people are flagged gainfully self-employed with no
    self-employment income. This led to the D1 ruling.
  - D5: 22k FRS-channel children are looked after at the arm's calibrated weights; the SPI channel
    holds fewer than 10 records.
  - D6: residential CGT gains move from £5.906bn to £5.898bn in the FRS channel and from £6.842bn to
    £6.858bn in the SPI channel.
- **D2, on #1121's head-r3 export.** The new population and the old differ by fewer than 10 benefit
  units. Against the engine's age half of `is_uc_eligible`, two named classes remain: reg 8 claimants
  aged 16-17 (32 units, 18k weighted), whom the engine includes, and mixed-age couples on the Pension
  Credit route (67 units, 39k), whom it excludes. Disagreement with the engine falls from 103 units to
  99.
- **D3, on #1121's head-r3 export, before recalibration.** The property shares add £1,247bn of UC
  capital over 5.18m benefit units, and £427bn of Pension Credit capital over 1.18m. Counted for every
  unit, UC falls £2.78bn (3.7%) and about 329k units lose paid UC; about 157k weighted UC reporters
  move above £16,000 on imputed property. With UC reporters keeping their receipt, UC falls £0.80bn,
  and the 179k units that lose UC are all non-reporters. Pension Credit falls £0.18bn either way.
- **Arm 2 (built as `ae9795d7c`: D2, D3 for every unit, D4).** Loss 0.00708, 98.4% within 10%, ESS 4,875,
  weight ratio 1,041. All 7 gates and input mass pass. Design UC falls to £50.65bn and design Pension
  Credit to £4.26bn; both calibrate to target.
  - On the arm's spine, 90 FRS-channel rows that report UC go above £16,000 through property alone.
    70 of them are in households with no property income, and 45 are in households with two or more
    benefit units, where the engine's proxy shares property over every claimant or partner.
  - D4: newly entitled units are flagged at a share of 0.3695 against the 0.37 rate.
  - The arm pipeline leaves two gates this PR changes to the release-cut certifier, so they were run
    on the arm's exported file with their own evaluators (`scripts/certifier_gates_check.py`).
    `uk_uc_capital_coherence` and `uk_take_up_signal` pass, and the capital gate fails on the control's
    export, which has no property shares.
- **The first head (built as `a42edaa9e`: T5, T6, T3, T4, T2, and T1 with the top band bound).** Loss
  0.00726, 98.2% of 1,174 targets within 10%, ESS 4,822. Two gates fail:
  - `uk_target_fit`: couples without children have no household in the top UC band, so that row is at
    −100%;
  - `uk_weight_ratio`: 1,230 against 1,151.
- **Attribution, calibrate-only on that head's spine** (T6 to T1 change no stage):
  - T5 alone: all 7 gates pass, loss 0.00716, weight ratio 1,061. It moves the top income cells,
    inside the 25% gate: the £500,000–1,000,000 employment-income band from −2.5% to −13.3%, and
    Scotland's £200,000-and-over total income from −8.3% to −15.7%.
  - T5 and T6: the two PIP rows fit within 0.1%, but the weight ratio is 1,241. They push the
    heaviest support family, a single FRS household, from 1,061 times the positive median.
  - Adding T3 and T4: 1,221. `obr.council_tax` moves from −9.8% to −7.3%.
  - Adding T2: 1,257. The three salary-sacrifice bands fit within 0.1%. Their design values are basic
    £1.05bn against £1.6bn, higher £2.94bn against £5.5bn, additional £2.46bn against £1.8bn.
  - Adding T1's top band: `obr.universal_credit` falls from 0.0% to −5.6% (£71.8bn against £76.1bn).
    Unbound, the calibration holds 2.57 times DWP's count of lone parents in the band (224k against
    87k) and 1.99 times for couples with children, £13.1bn of UC. Bound, the band holds £5.5bn
    (`scripts/uc_top_band_check.py`).
  - A measurement-only support floor of 10 holds out 17 bound rows with fewer than 10 source households.
    `uk_target_fit` then passes and loss is 0.00687, but the weight ratio is 1,247 and the UC total
    −4.6%.

### The head after the rulings

Main's evidence reader refuses real run manifests (#1187), so a national build from main stops before
its first phase. This head was therefore calibrated with #1188's fix on top, for measurement only;
the fix touches no calibration code.

- **Final head (`a7e5bc56f`).** The spine was built from `0324e9e84`, whose tree differs from the head
  only in compile-parity receipts and W's register entry, and calibrated from the head plus #1188's
  one commit. Loss 0.00694 (control 0.00712), 98.3% of 1,168 targets within 10%, ESS 4,843, weight
  ratio 1,067. All 7 calibrated-seam gates pass, and so does the certifier's input-mass gate, with no
  stale, expired or dormant exclusion.
  - `obr.universal_credit` and `obr.pip` calibrate to 0.0%, and 109 of 112 UC cells are within 10%
    (at most 14.6%). Design UC is £51.78bn and design Pension Credit £4.27bn.
  - `obr.council_tax` is at −7.0% and `obr.vat` at +22.7%. The three salary-sacrifice bands are within
    0.3%. T5's top income cells sit at −11.5% and −15.2%.
  - **W.** The West Midlands £12,570–15,000 income-tax cell calibrates to +23.9%, inside the 25%
    bound. With its deferral still in the register, the same weights failed `uk_target_fit` on that
    one point: a stale deferral. W retires it and the gate passes. The cell sat at +28.9% on the
    control and +26.8% on the first head. Its design value is still 6.2 times its target
    (pe-uk#2174), so a later change can push it back over the bound.
  - D1, at calibrated weights in 2025: the minimum income floor applies to 1.63m FRS-channel people
    (control 1.84m) and 1.24m SPI-channel people (control 1.73m). 1.58m SPI-channel people are flagged,
    against 3.35m when the flag followed the donor's status.
  - D3, the stage's receipt: 51 FRS-channel benefit units that report UC keep their receipt, and 44
    of them would have gone above £16,000 with their property share. Fewer than 10 SPI-channel
    reporter units are affected. 12 FRS-channel reporter units hold financial capital above £16,000 as
    surveyed.
  - D4: newly entitled units are flagged at a share of 0.3695 against the 0.37 rate.
  - D5: 18k FRS-channel children are looked after at calibrated weights.
  - On the exported file, `uk_uc_capital_coherence` passes with the reporter rule, and
    `uk_take_up_signal` passes: `would_claim_uc` sits at 0.560 against 0.55, over 55,763 units.

## Verification

- Touched-surface tests per commit (counts in each commit's run).
- Each replay left every commit's hand-written change line-for-line identical; only the regenerated
  surfaces differ.
- Final head `a7e5bc56f`:
  - the `engine-free` lane's 534 test files, with `policyengine_uk` blocked as in CI: 16,146 passed
    and 139 skipped. The three failures are environmental:
    - the live Hugging Face pointer test fails on main too;
    - the telemetry ambient-token exchange needs the hosted service;
    - main's new US head-to-head scorer test needs Python 3.14.7, and this machine has 3.14.6;
  - the `engine-uk` lane's 40 test files, run in full: 174 passed;
  - against the pinned feed, 12 passed: the national compile-parity receipts regenerate identically
    and pass their gate, the committed national and local references regenerate, and the NICs and PIP
    diagnostics resolve their exact facts.
- `ruff check` and `ruff format --check` pass on the 61 changed Python files, and so do
  `tools/ci_test_plan.py verify` and `tools/graph_acceptance_burndown.py --verify`.

## Signature drafts

`data/ukds/acceptance/1095-uk-data-ports-3/signed-differences-draft-d1-d5.md` (D1 and D5, net-new
columns) and `signed-differences-draft-d3.md` (D3's redrafted UC entry and the new Pension Credit
entry). They are not committed until signed.

## Left for later

- The sparse-band support floor, as its own PR.
- SPI incomes drawn by employment status (uk-data#529), with microcosm#840. It removes D1's SPI rule.
- The WAS property draw: a landlord stratum and a benefit-income predictor.
- The low UC awards below the top band, which the unbound band hides.
- The weight year, and the renewals of the exclusions that expire from 15 October.
