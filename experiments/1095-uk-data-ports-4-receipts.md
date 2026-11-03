# microcosm#1095 receipts: the fourth ports PR

Branch `uk-data-ports-1095-4`, rebased on 2026-10-14 onto main `121fd4dd6` (#1256, #1279). The four
arms and the lanes below ran on the branch as it stood on main `65d7d153a` (#1260, the LCFS 2024-25
tabs); the tree ids are the ones measured. #1256, merged since, draws the SPI channel's incomes
afresh, so a control and the head are rebuilt on `121fd4dd6` ("On the current main", below).
Chronicle feed 942a1fa throughout. The engine lock moves from policyengine-uk 2.123.0 to
2.130.1: the branch first locked 2.127.0, and after Vahid's review moved to 2.130.1, the first release
published after it (2.128.0 to 2.130.0 never reached PyPI). 2.131.0, published at 12:29 UTC on
14 October (trading losses), is left for the next bump.

Every figure here is a weighted aggregate or a publisher's figure. Survey support is counted in
distinct source households, FRS and SPI channels together, and given as a band: `<10`, `10–29` or
`30+`. A cell under 10 has no weighted figure, and a target row's figure is given only where 30+
source households stand behind it. Effective sample sizes are not given. A spine holds copies of
survey households, so its row counts overstate support and are not given.

## Commits, in order

Which items port merged incumbent code and which go first (item 5 of the review):

- B1 `00040d929`, B2 `c2d305fd0`: the lock at policyengine-uk 2.127.0 and the stage texts that cite
  it.
- R1 `3e62fa607`, `111391500`, `00519cf54`, `01c771ad2`: FRS EMPSTATI 11 is OTHER_INACTIVE, Max
  Ghenis's commits from #1097. **Ports merged code**: uk-data#526.
- R2 `d309afb85`: approved training from the FRS TRAINEE flag on dependants. **Goes first**:
  uk-data#561 is an open PR; the rule is its rule.
- R3 `6c3a0add8`: statutory paternity pay from the FRS adult table's SPPADJ. **Goes first**:
  uk-data#575 is an open issue.
- R4 `f19416e42`: salary sacrifice limited to the pay it comes out of. **Ports merged code**:
  uk-data#518, merged on 14 October after this commit was written; the review compared the two.
- T1 `2da556a5a`: the national 15–19 age bands split into 15, 16–17 and 18–19. **Goes first**:
  uk-data#563 is an open issue.
- E1 `0f0bb869a`: release weights written at the dataset's year. **Goes first**: uk-data#538 is the
  incumbent's open issue.
- E2 `e9aa8d333`, `1e88ac52c`: release identifiers inside the engine's 32-bit range. **Goes first**:
  uk-data#558 is an open issue.
- E3 `ac6088f2c`, `ae08ba3bd`: the calibration measures the engine on the release's identifiers.
  Microcosm only.
- B3 `3623ca57d`: the lock at policyengine-uk 2.130.1 and what the build follows (item 2).
- M `84ce97a2a`: each person's `months_since_last_birthday` stored in the spine. Microcosm only: the
  incumbent leaves it to the engine's formula.
- X `e3342783a`: `dwp.uc.households_children_5_or_more` held out of the calibration until
  26 November 2026 (item 4).
- `0912388b6`: the derived surfaces regenerated on main after the rebase onto #1256 (the gate
  digests, the coverage manifest and the H2 fixture; the rendered stages are unchanged).
- This commit: the receipts.

## María's rulings

Of 2026-10-13, at the plan:

- The engine bump is in this PR.
- Statutory paternity pay is included.
- The 15–19 band splits into 15, 16–17 and 18–19.
- Additional-dwelling purchases and `obr.sdlt` wait for HMRC's stamp tax tables in Chronicle. This PR
  only measures household SDLT by part (below).
- The measure exclusions that lapse from 15 October are not renewed here. The arms pass
  `--review-date 2026-10-14`, for measurement only.

Of 2026-10-14, during the work:

- #1097's `experiments/` folder is not carried. Four of its eight commits are cherry-picked; the
  other four touched only that folder.
- The incumbent's two 15–19 rows move to `excluded` in the registry accounting.
- The UK side takes the policyengine-core that uv resolves, 3.33.3.

Of 2026-10-14, on Vahid's review:

- The lock moves to 2.130.1 in this PR, with an arm of its own.
- R3 stays, and its changelog fragment and these receipts say what the engine reads and that the
  annual amount is an open question.
- `dwp.uc.households_children_5_or_more` is held out with a reason and the sparse-band expiry,
  26 November 2026.
- The receipts carry no effective sample size, and no commit on the branch does: the earlier
  receipts commit is replaced, not edited.
- The fix storing `months_since_last_birthday` is folded into this PR.

## Arms

Each arm builds the spine and runs the national calibration on one tree, on feed 942a1fa with the
review date above.

- **control**: main `65d7d153a`, policyengine-uk 2.123.0.
- **bump**: main plus the three lock commits and nothing else, policyengine-uk 2.130.1. A
  measurement-only local commit (`c1a8d91c8`), not in the PR: B1, B2 and B3 picked onto main with
  the generated surfaces regenerated. Its spine was built before B3 took the rate-class change below,
  which changes no stage.
- **head without M**: the PR at B3 plus X (`a3ae4faf1`, measurement only, not in the PR).
- **head**: `1da8d681f`, X before the rebases onto #1242 and #1256 (now `e3342783a`).

What each calibration reached:

- **control**: loss 0.00583; 98.8% of 1,214 rows within 10%; all 7 calibrated-seam gates pass.
- **bump**: loss 0.00593; 98.6% of 1,214; all 7 gates pass.
- **head without M**: loss 0.00588; 98.7% of 1,217; all 7 gates pass.
- **head**: loss 0.00590; 98.8% of 1,217; all 7 gates pass. It writes a release file, and E1, E2 and M
  are checked on it.

The head has three rows more than main: T1's six in place of the incumbent's two, less the row X
holds out.

The input-mass gate, run as the certifier runs it on the head's calibrated weights, passes.

**The rebase onto #1242.** #1242 came after the arms. It adds the UK local release path: target-weight
rules that the national role refuses, size-stage penalties that are off unless asked for, and the
local doctrine. It renews the `dwp.uc.households_housing_element` measure exclusion to 14 November,
which the arms already held at their review date, and its gate-register edits concern the release-cut
certifier. So the national calibration the arms ran computes the same on the rebased branch. The
rebase changed one file of this branch: the packaged-exclusions test, which now tells X's ruling
apart from the housing-element renewal of the same date by tracking issue (folded into X). The
derived surfaces regenerate unchanged on the rebased head.

## On the current main

#1256 draws the SPI channel's incomes at the FRS twin's conditional rank, which changes the spine
every arm above calibrates. So a control and the head were rebuilt on main `121fd4dd6`, with the same
feed and review date:

- **control** (`121fd4dd6`, policyengine-uk 2.123.0): loss 0.00584; 98.4% of 1,214 rows within 10%;
  all 7 calibrated-seam gates pass.
- **head** (`0912388b6`, this commit's parent): loss 0.00599; 98.6% of 1,217; all 7 gates pass.

On the head's written file the checks below hold as before. The stored weights times 1.0072 equal the
solve's to 16 significant figures, and policyengine-uk's 2025 weights equal them to single precision.
The identifiers run 1 to n with the build identifiers equal to the spine's, and the engine returns all
five as written. Every other column equals the spine's, and the input-mass gate passes. With M, the
file read as the engine reads it differs from its own solve on 3 rows: private-school pupils by 3.4
points, and the two land rows by 0.3 and 0.1. The engine returns the stored months for every person.

The decomposition into the bump, the spine fixes and the stored months was measured on `65d7d153a`
and is not repeated on this base.

## B. The engine bump, 2.123.0 to 2.130.1

Nothing the UK build writes is renamed, removed or re-typed. What the engine returns does change.
Up to 2.127.0:

- `uc_reported_capital` and `pension_credit_reported_capital` become formulas of the new input
  `benunit_reported_capital`. A stored value still wins, and a stored `uc_reported_capital` is now
  projected by GDP per head, as Pension Credit's already was (pe-uk#2201). That completes #1200's
  property rule for Universal Credit.
- A claimant whose partner cannot claim is assessed as a single person (pe-uk#2084, UC Regs 2013
  reg 3(3)).
- DWP's announced 2025-26 rates replace CPI projections (pe-uk#1925).
- The Housing Benefit Guarantee Credit passport needs receipt (pe-uk#1927), and each family's
  Council Tax Reduction uses its own scheme (pe-uk#2015).
- A unit's own claimant or partner no longer counts as its child or young person (pe-uk#2285).
- Carer's Allowance stops above the earnings limit for reported recipients too (pe-uk#2312).
- Savings and dividends are banded above property income (pe-uk#2352).
- The employer Class 1 secondary threshold is £5,000 a year (pe-uk#2165).
- The road-fuel litre proxies are re-based on DESNZ pump prices (pe-uk#2186).

From 2.128.0 to 2.130.1, the releases the review named:

- **Legacy benefits close** (pe-uk#2119, 2.128.0): a family moves onto Universal Credit when DWP
  closes a legacy benefit it receives, tax credits from 2025-26. Whether it claims is
  `would_claim_uc_at_legacy_closure`, which the build leaves at the engine's default, true.
  uk-data#534 draws it at DWP's Move to UC claim rates, which are not yet a Chronicle fact, so the
  calibration's 2025 Universal Credit rows rest on that default. The concept mapping declares it.
- **Statutory pay in Universal Credit** (pe-uk#2365, 2.128.0): sick, maternity and paternity pay
  count as UC earned income, and paternity pay in household benefits and HBAI income.
- **Gift Aid** (pe-uk#2271, 2.128.0) extends the basic rate band instead of reducing allowances.
  Two parts of the build follow. The CGT taxable-income measure is net income less the allowances,
  as the engine now stacks gains on it. And the salary-sacrifice relief by rate class extends a
  donor's bands by the grossed-up gift, as `earned_income_tax` now does (ITA 2007 s.414(2)(b)). The
  first head arm on 2.130.1 refused at that measure: its classes no longer added up to the engine's
  tax for donors. B3 carries the extension, the three relief rows declare it, and hand-computed
  donor cases hold it against the engine.
- **The benefit cap**: its ESA and disability exceptions follow the claimant and partner (2.129.0),
  and its earnings exception counts statutory pay (pe-uk#2399, 2.130.1).
- A reported income-related ESA or income-based JSA award ends in remunerative work (2.129.1).
- A child on the working parent childcare entitlement keeps the universal hours (pe-uk#2177). The
  universal-only childcare row now leaves the extended entitlement out itself, as the published
  difference does.
- Universal Credit's regulation 77 for company owners (2.130.0): the `owned_company_*` inputs
  default to no company, and the build sets none.
- Capital gains get their own projection path (2.130.0). With no scenario passed it copies GDP per
  head exactly (`legacy_gdp_per_capita`), so the build's gains are projected as before.

**The spine moves with the engine.** The stages read it: the take-up and reporter draws read
entitlement, the consumption and wealth stages read its indices, and the CGT stages read liability.
So the bump changes the spine although it changes no stage. At each spine's own weights:

- Universal Credit held by reporters (`universal_credit_reported`) rises 2.3%, from £42.2bn to
  £43.2bn.
- Pension contributions by salary sacrifice rise 2.3%, from £16.91bn to £17.30bn, and employee
  contributions 0.5%.
- Gains that qualify for Business Asset Disposal Relief fall 0.75%, and all capital gains rise 0.25%.
- Every other total checked moves by less than 0.5%: pay, residential gains, main-home, other
  residential and non-residential property, financial and corporate wealth, recorded capital for
  Universal Credit and Pension Credit, the three take-up flags, petrol, diesel, domestic energy and
  transport spending.

**Before calibration**, at the spine's weights, 169 of the 1,214 rows move by more than 1%: 56
Universal Credit rows, 35 CGT rows, 30 regional SPI rows, 14 Pension Credit rows and 34 others. The
rows the review named, and the largest moves:

- Universal Credit is £52.1bn against the control's £51.9bn (+0.4%), on 0.1% fewer households. The
  legacy closure and statutory pay as earned income both reach it; the arm measures them together.
- The benefit cap: 0.8% fewer capped households.
- ESA is £4.63bn against £4.77bn (−3.0%), with 5.1% fewer income-related claimants.
- OBR's Pension Credit line is £4.52bn against £4.24bn (+6.5%), with 1.05m benefit units against
  1.01m. Carer's Allowance is £3.06bn against £3.24bn (−5.5%).
- CGT liability rises 0.3%, and income tax 0.2%.

**After calibration** the two arms fit alike:

- loss 0.00593 against the control's 0.00583;
- 98.6% of the 1,214 rows within 10%, against 98.8%;
- all 7 calibrated-seam gates pass on both.

Fifteen rows are outside 10% on both arms (three of them rest on fewer than 10 source households),
and the bump has two more: `dwp.uc.households_children_1` and
`dwp.uc.households_couple_with_children`. The Universal Credit family counts trade places with each
other: the number of households on Universal Credit is 3.9% under DWP's on the control and 4.0%
under on the bump.

Row figures in this file are given only for rows with 30+ source households behind them. Of the
bump's 1,214 rows, 79 rest on 10–29 and 36 on fewer than 10; many of them are the sparse Universal
Credit award bands the support-floor PR is for.

Re-derived with no change to data: the uprating pins (the version stamp; every pinned value, the ONS
population growth E1 divides by included, is unchanged), the concept coverage (288 inputs to 299:
eleven owned-company inputs and `would_claim_uc_at_legacy_closure` join,
`uc_has_limited_capability_for_work` leaves; the build writes none of them), the stage texts that
cite the locked engine, the H2 fixture, the coverage manifest's source digests and
`APPROVED_UV_LOCK_SHA256`. The target references and the compile-parity receipt regenerate
identically on feed 942a1fa.

## R1 to R4. The spine fixes

The head without M against the bump, each at its spine's own weights.

- **R1. EMPSTATI 11.** 2.02m adults move from LONG_TERM_DISABLED to OTHER_INACTIVE: the first
  group falls from 5.19m to 3.18m, and no other status changes (30+ source households). The ESA
  rows read reported ESA and do not move.
- **R2. Approved training.** The flag leaves adults on employment schemes, who are mostly aged 20
  or over, and lands on dependants aged 16 to 19 in the child table. Each group rests on 10–29 FRS
  source households, and the two share fewer than 10, so no weighted count is given. Those
  dependants now pass the qualifying-young-person test, which moves their families up one size in
  the engine's Universal Credit child counts. Its effect on the fit is below (X).
- **R3. Statutory paternity pay.** Fewer than 10 source households report it, so no weighted figure
  is given. Statutory maternity pay is unchanged.
- **R4. Salary sacrifice within pay.** On the control's and the bump's spines some rows held a
  sacrifice above their pay (30+ source households across the FRS and SPI channels, fewer than 10
  of them FRS). The stage now refuses a spine where any does, and the head's spines built. On
  2.130.1 the totals barely move: contributions by salary sacrifice are £17.30bn on both spines,
  and employee contributions fall 0.25%. Before calibration, with R1 to R3 beside it, the
  additional-rate relief row rises 7.5% and the basic- and higher-rate rows fall 1.7% and 1.6%;
  users below the £2,000 cap rise 2.1% and those above it fall 2.3%. After calibration all seven
  salary-sacrifice rows are within 0.1% of HMRC's on the head.

Together, before calibration, 124 of the common rows move by more than 1%: 60 Universal Credit
rows, 26 CGT rows, 13 State Pension rows, 8 two-child-limit rows and 17 others. Universal Credit is
£52.5bn against the bump's £52.1bn (+0.7%), on 1.0% more households. No fix touches the State
Pension: those rows move because the fixes reweight the spine's households (the stages read the
engine), and without M the engine spreads birth months by those weights. At the spine's weights
Universal Credit held by reporters falls 0.7%, from £43.2bn to £42.9bn; every other total checked
moves by less than 0.5%.

After calibration: loss 0.00588 against the bump's 0.00593; 98.7% of 1,217 rows within 10%,
against 98.6% of 1,214; all 7 gates pass on both.

## R2 and the row held out (X)

DWP counts 76,300 Universal Credit households with five or more children. It counts 55,300
households under the two-child limit with five or more children (37,000 with five and 18,300 with
six or more). So about 21,000 large families on Universal Credit are outside the limit: their third
and later children were all born before 6 April 2017, or an exception applies.

On main both rows pass the fit gate through an artefact of the missing training flag. A dependant
aged 16 to 19 in training is a child in the FRS and in the first row's count, but failed the
engine's qualifying-young-person test, so the two-child-limit rows counted one child fewer in that
family. Fewer than 10 source households were in the first row and in neither of the
two-child-limit size rows. They were the only way the frame could tell DWP's two counts apart.

R2 gives those dependants the flag. The two counts then agree on every family, and the frame has
fewer than 10 source households with five or more children on Universal Credit and outside the
limit. The calibration cannot meet both DWP counts, and the two-child-limit rows carry six times the
loss weight. On the earlier arms, on 2.127.0, `dwp.uc.households_children_5_or_more` fell outside
the fit gate's 25% bound, and a blocked calibration writes no release file.

The fix in R2 is right, and the earlier fit stood on fewer than 10 source households. María ruled
to hold the row out (X): `calibration_measure_exclusions.json` carries it with the reason, tracking
microcosm#1095 and the sparse-band expiry, 26 November 2026, and the packaged-exclusions test pins
the entry. The two-child-limit rows stay bound. With the row held out, the head and the head without
M pass all 7 gates and write a release file.

## R3 on the locked engine

policyengine-uk 2.130.1 reads `statutory_paternity_pay` in Class 1 National Insurance earnings, in
the Universal Credit and Tax-Free Childcare paid-work tests, as Universal Credit earned income, and
in household benefits and HBAI income (pe-uk#2365). So the column raises the net income of the
households that report it, as it should. On 2.127.0 it was only a National Insurance base and
lowered their net income; the changelog fragment says so. Fewer than 10 source households report
it, so no weighted figure is given.

The annual amount is still open (uk-data#575): the stage takes SPPADJ, a weekly amount, times the
weeks in a year (365.25/7), as it does sick and maternity pay. For a payment that lasts at most two
weeks, a grossed weekly amount from a reference week is the FRS's usual annual estimate in
aggregate, but not per person.

## T1. Ages 15 to 19

**Measured first, on the control's file.** The control fits each sex's five-year band within 0.1%
and binds nothing inside it. Against the ONS mid-year estimates the three new rows bind, it holds:

- female: 3.7% too few at 15, 19.1% too many at 16–17, 18.0% too few at 18–19;
- male: 26.1% too many at 15, 14.0% too many at 16–17, 28.1% too few at 18–19.

Every one of the six cells rests on 30+ source households. This is uk-data#563's finding on this
frame: a fit to the band leaves it holding too many dependants under 18 and too few adults of 18
and 19.

**On the head** the six rows fit within 0.1%.

The contract has 400 targets (396) and 1,283 active references (1,279). In the registry accounting
the incumbent's two 15–19 rows move from mapped to excluded and the six rows are contract-only
declarations; the compile-parity receipt signs the two as fixture-only and the six as ledger-only.
Within the `ons_population` family every other row's loss weight falls to 149/153 of what it was.

## E1. Weights at the dataset's year

policyengine-uk projects a single-year dataset forward. From the dataset's own year it multiplies
`household_weight` by one plus the ONS population growth of each later year; no other weight is
stored, and the person and benefit-unit weights are formulas of it. The 2024-25 dataset's weights
are solved at 2025, where that growth is 0.72%. Written as solved, they read 0.72% high at 2025 on
every weighted count and total.

The release write now divides the solved weights by the engine's own factor, 1.0072. The division
is the last record of the file's mass log, and the readback expects exactly it. The solve, its
gates, its diagnostics, the spine checkpoints and the engine scratch file keep the solved weights.
The growth is unchanged in 2.130.1, so the factor holds.

**On the head's file:**

- the stored weights times 1.0072 equal the solve's weights on every household, to 16 significant
  figures;
- loaded in policyengine-uk 2.130.1, `household_weight` at 2025 equals the solve's weights to the
  engine's single precision (within 6 parts in 100 million);
- every other column of the file equals the spine's column, row for row, the reviewed export drops
  aside;
- the readback passes and records the solve year, the dataset year and the factor.

**Engine tests** hold the mechanism against the installed engine: the vendored growth equals the
engine's parameter at each pinned instant; `household_weight` is the only weight in the engine's
uprating file, and the person and benefit-unit weights are formulas; a written release read at 2025
returns the solved weights; and a file written as solved reads high by exactly the factor.

**Readers of a written file** follow the same projection:

- the rule-1 scorer reads each side with the weights the engine gives it at the scored year, and
  its score block records each file's year and factor. The incumbent `enhanced_frs_2024_25` is a
  2024 dataset too, so every one of its estimates is now read 0.72% higher than before. **Scores
  are not comparable with earlier ones;**
- the release-cut certifier runs its gates on the solved weights and refuses a file divided for
  another year;
- `tools/evaluate_uk_incumbent_surface.py` reads the candidate the same way;
- the assembler's weights file pairs with the H5 row by row, so it carries the file's own weights,
  and the build manifest says which year they stand at.

## E2 and E3. Identifiers inside 32 bits

**Before.** On main's file more than half of the rows of each entity carry an identifier above
2,147,483,647. policyengine-uk stores its five identifier variables as 32-bit integers, so those
wrapped: `calculate("household_id")` did not return the file's identifiers.

**E2.** The release boundary renumbers. Per entity the identifier is one plus the rank of the build
identifier, the person table's two memberships map through the same arrays, and row order does not
change. The build identifiers stay in the file as `person_build_id`, `benunit_build_id` and
`household_build_id`. Both readbacks refuse an identifier that repeats, is below 1 or is out of
range, naming only the column.

**E3.** The engine file the calibration measures on carries the release's identifiers, so any
engine read keyed on an identifier gives the solve what it gives the release. A per-clone engine
block takes the ranks its rows have in the whole frame, so the local solve numbers its rows as the
local release does. The resolver's own frame keeps the build identifiers, and nothing joins engine
outputs to it by identifier. Measured before M on 2.127.0, E2 alone made the release differ from the
solve on two-child-limit and Universal Credit award rows by up to a tenth or more, because the
engine dealt birth months by a hash of `person_id`; E3 removed that. With M the birth months no
longer depend on identifiers, and E3 remains for any other engine read keyed on one.

**On the head's file:** each entity's identifiers are 1 to n, and the build identifiers beside them
equal the spine's, row for row; policyengine-uk 2.130.1 returns all five identifier variables
exactly as written, with no warning on load.

## M. The birth months stored

policyengine-uk places each person's date of birth from the whole age and
`months_since_last_birthday`. The date decides the two-child limit (born before or after 6 April
2017), the State Pension scheme (reaching pension age before or after 6 April 2016) and the Pension
Credit qualifying age. Left unset, the engine spreads each single year of age and sex evenly over
the year by weight, in an order set by a hash of `person_id`. So the months a file gets depend on
the weights it carries: the solve measures the engine at the spine's weights and the release
carries the calibrated ones, so a release read a different spread of birth dates from the one its
own solve fitted. That was so on main too.

The root stage now stores the months, spread the engine's way once: at the FRS weights, in an order
set by an identity-keyed draw on the person's stable identifier. Every later engine read takes the
stored value, and a copy of a person keeps their months. An engine test holds the stage's spread
equal to the engine's own `stratified_uniform`, and shows that stored months do not move with the
weights while unset ones do.

**On the spines.** Every person on the head's spine carries a value in [0, 12), and every copy of a
source person carries the same one. Storing the months moves the stages' engine reads: Universal
Credit held by reporters falls a further 0.75%, from £42.9bn to £42.6bn. Before calibration 98 of
the 1,217 rows move by more than 1%: 61 Universal Credit rows, 18 State Pension rows, 11
two-child-limit rows, 4 Pension Credit rows, 3 benefit-cap rows and a childcare row. After it the
two arms fit alike: loss 0.00590 against 0.00588, 98.8% of rows within 10% against 98.7%, all 7
gates on both.

**The release against its own solve.** Each reading takes a written file as policyengine-uk 2.130.1
reads it and compares every row's relative error with the solve's own, in percentage points.

- **Without M:** 55 of the 1,217 rows differ, 8 by more than 1 point. Read from the file the loss is
  0.0080; the solve's is 0.0059. The State Pension rows move most: people on the new State Pension
  at £300 a week or more read 18.8 points higher, and men aged 70 to 74 on the pre-2016 scheme 10.9
  points lower. Private-school pupils read 4.5 points lower, and two Pension Credit rows move by up
  to 0.4 points.
- **With M:** 3 rows differ, 1 by more than 1 point; the loss is 0.0076 against the solve's 0.0059.
  No State Pension, two-child-limit or Pension Credit row moves. The engine returns the stored
  months for every person, and the written column equals the spine's.

What is left depends on the weights themselves, and is main's too: the engine imputes private-school
attendance by each household's income rank at the file's own weights, so private-school pupils read
4.1 points lower, and two land rows 0.3 and 0.1 points lower. Every row named here rests on 30+
source households.

## Household SDLT by part

`obr.sdlt` binds `stamp_duty_land_tax`. Split by part on calibrated files, with policyengine-uk
2.130.1 at the solved weights:

- **The bump's file** (main's stages): £15.04bn in all, against OBR's £15.03bn. £9.80bn is on
  main-home purchases. The rest, £5.24bn, is additional dwellings together with non-residential
  purchases and rent.
- **The head's file**: £15.03bn in all. £9.70bn is on main-home purchases and £5.33bn on the rest.

Non-residential purchases and rent each rest on fewer than 10 source households, so they are not
split out of the rest, and neither is any part that would give them back. `corporate_sdlt`, which
the bound measure leaves out, is £3.1bn on both. At the spine's weights household SDLT is £6.4bn,
so the calibration reaches OBR's total by raising it about 2.3 times through the weights. That is
the gap the purchases work is for: it waits for HMRC's Annual Stamp Tax Statistics in Chronicle.

## Tests

- CI's engine-uk job list on `1da8d681f`: 205 tests, all pass.
- CI's engine-free job list on `1da8d681f`: 17,475 tests. 17,318 pass and 155 skip; two fail on this
  machine only: a US scorer test that needs Python 3.14.7, and a test that reads the live Hugging
  Face pointer.
- The feed-gated tests on feed 942a1fa, on `1da8d681f`: 99 tests, 97 pass and 2 skip.
- `tools/ci_test_plan.py verify`, `tools/graph_acceptance_burndown.py --verify`,
  `tools/pin_uk_uprating_engine_values.py --check`, `tools/refresh_concept_coverage.py --check` and
  `ruff check .` pass on `1da8d681f`.
- On the head rebased onto #1242 (`cb922d69f`): CI's engine-uk job list, 205 tests, all pass; and
  the engine-free test files that rebase touched, 837 tests, of which 836 pass and one, feed-gated,
  skips.
- On this commit's parent (`0912388b6`, rebased onto #1256): CI's engine-uk job list, 207 tests, all
  pass; and the engine-free test files the rebases or this PR's last commits touch, #1256's SPI tests
  among them, 966 tests, of which 964 pass and two skip. CI runs every lane on the pushed branch.

## Not in this PR

- Additional-dwelling purchases and `obr.sdlt` (waiting on HMRC's Annual Stamp Tax Statistics in
  Chronicle, and on pe-uk#2241 and pe-uk#2322).
- DWP's Move to UC claim rates as a Chronicle fact, for `would_claim_uc_at_legacy_closure`
  (uk-data#534).
- policyengine-uk 2.131.0.
- The sparse-band support floor (its own PR, before the sparse-band exclusions expire on 26 November;
  X's row has the same expiry).
- The renewals of the measure exclusions that lapse from 15 October.
- The signed-difference entries for the new and changed columns: R3's and M's net-new columns, E2's
  three build-identifier columns, and the columns R1, R2 and R4 change. Their drafts stay outside the
  repository until signed.
