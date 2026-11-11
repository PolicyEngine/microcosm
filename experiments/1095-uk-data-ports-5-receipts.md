# microcosm#1095 receipts: the fifth ports PR

Branch `uk-data-ports-1095-5` on main `5d4eb37b3` (#1286), Chronicle feed 735820c throughout. The
engine lock moves from policyengine-uk 2.130.1 to 2.132.6, frozen for this PR on María's ruling of
15 October; later releases go to the next bump.

Every figure here is a weighted aggregate or a publisher's figure. Survey support is counted in
distinct source households, FRS and SPI channels together, and a cell under 10 has no weighted
figure. Effective sample sizes and weight ratios are not given. A spine holds copies of survey
households, so its row counts overstate support and are not given. The support floor's per-row
support counts stay on the machine that measured them; this file gives each band's category only.

## Commits, in order

Which items port merged incumbent code and which go first. Every uk-data item below is open on
16 October (uk-data#528, #546 and #589 are open PRs; #568 and #582 are open issues), so each port
goes first and follows its upstream's rule:

- `b5d82c2af` (#1240): mobile-home NEED cells start at their mean when they drew no kWh, so main's
  spine builds again after #1270. Microcosm only.
- V1 `985aae5b9`: a test pins how the spine stores birth months (Vahid's second should on #1271).
- B1 `cd9bb6ec0`: the lock at policyengine-uk 2.132.6.
- B2 `55d3a6091`: HBAI income before the reset to zero (pe-uk#2402, merged).
- B3 `08ebfa175`: the ESA support group from the FRS ESA group. Microcosm only.
- L1 `374b7de2a`, L2 `2d38875ef`, L3 `1f2961929`, and L3's follow-ups `ca050e0cc`, `01b6b2927`,
  `81203a14b`: self-employment losses. **Goes first**: uk-data#528 is an open PR; pe-uk#2085 is
  merged.
- U2a `88e355305`: Vahid's #1127 commit, with his authorship (#1126). U2b `870cb0c9d`: the fitter
  follows pe-uk#2177 (merged). `94ed6eaac`: the H2 fixture under the moved anchor. **Goes first**
  beside uk-data#546, an open PR with a different anchor.
- U1 `502840383`: the five-or-more-children count becomes a diagnostic. Microcosm only.
- E1 `8cc8532bb`: `obr.esa` binds the whole incapacity line. **Goes first**: uk-data#582 is open.
- D1a `e4b10872b` (#868): borrowers below the threshold, courses dated from leaving education.
  Microcosm only.
- O1 `8589c6603`: six owner-manager inputs (pe-uk#1965, merged). Microcosm only.
- U4 `caae98e2d`: private-school attendance stored at the spine's weights. **Goes first**:
  uk-data#568 is open.
- U2c `45c1bdeb0`: the four childcare take-up rates refitted.
- C1 `933ee4eef` (#1284): the council-tax stock on the household basis of its area. Microcosm only.
- P1 `dbfa76d1a`: the parity fixture's Scottish rows in their seats. **Goes first** with
  uk-data#589's mapping (an open PR).
- U3 `7f862f94e`: the sparse-band support floor. Microcosm only.
- `f57c50091` (#868): before 2026 a repayer never takes Plan 5, which nobody repays before April
  2026. Found on the head's first calibration (below). Microcosm only.
- Vahid's review of #1299: `968a9f945` pins each ESA group code and the SPI stage's ESA and
  trading-loss resets; `d918388e2` cites the FRS variable listing for the ESA group codes;
  `84e1ffece` corrects the childcare fitter's docstring on the universal row; `09a2c8f1b` says in
  the #1240 fragment that a started NEED cell replaces the spend its households drew;
  `18be8d205` notes when the Plan 5 repayment year needs revisiting. No build output moves.
- This commit: the receipts.

O2, the directors' dividends, was on the branch and is taken out (below); it stays open in #1095.

## María's rulings

Of 2026-10-15, at the plan: the engine bump to 2.132.6, frozen; LOSSBF to `loss_relief` and
`ni_class_4_losses_brought_forward`, with the HMRC self-employment rows measured after the relief;
#1287 out; the #866 renewal in its own small PR; three heavy builds.

Of 2026-10-16, "go with your recommendations on all of them": D1 (academic year 2024/25; the SLOS
cohort; start year from EDUCLEFT; plan 2 for England and Wales starters 2012/13 to 2022/23), O1
(DIRPERC at least 50%), O2 (the larger of the two dividends; the questionnaire does not exclude
own-company shares), the SPI rows' owner-manager inputs drawn in the stage-2 fill, U2a (#1127's
mean), U2c (a bound hit keeps the bound), U4 (DfE's series), U1 (diagnostic), E1 (the Income
Support part stays a residual), C1 (the households-per-dwelling assumption, signed), U3 (floor 10,
drop not pool, expiry 2027-01-08) and P1 (the remap, signed).

Of 2026-10-16, after the head's calibration: O2 leaves this PR and stays open in #1095 with what
the head found; the support floor's ceilings keep a little headroom over the head's drops; and
the Plan 5 repayer rule (`f57c50091`) is confirmed.

## Arms

Each arm builds its spine from the commit named, locally, on feed 735820c, and runs the national
role's calibration on it (`tools/build_uk_rowwise_candidate.py --release-role national`). The
review date is 2026-10-16 on the head arms, the date of María's signatures, and 2026-10-15 on the
control.

- **control** (`b5d82c2af`: main `5d4eb37b3` plus the NEED fix, policyengine-uk 2.130.1): loss
  0.29148 to 0.00607; 98.6% of 1,226 rows within 10%; all 7 calibrated-seam gates pass.
- **head with O2, first run** (calibrated on a spine built with O2, before the Plan 5 fix):
  blocked at the battery. `slc.borrowers.plan_5_above_threshold` binds academic year 2024/25,
  where SLC publishes zero, and a few young repayers still held Plan 5, so the row's miss held
  the loss near 0.30. `f57c50091` fixes the stage. The weight-ratio gate failed too (next arm).
- **head with O2** (the Plan 5 fix in): loss 0.26546 to 0.00583; 98.4% of 1,217 rows within 10%;
  6 of the 7 calibrated-seam gates pass. `uk_weight_ratio` fails: the largest family-folded
  weight passes the reviewed maximum ratio to the median. The record-level figures stay local.
- **head** (built at `da8892ed5`, a measurement commit that is the head with O2 except that the
  directors' dividends are left out of `dividend_income`; taking O2 out of the branch changes
  nothing else the build reads, so this is the head's spine and calibration): loss 0.26817 to
  0.00584; 98.4% of 1,216 rows within 10%; all 7 calibrated-seam gates pass.

On the head's written file:

- **E1 and E2**: the weights sit at the dataset's year with the engine's projection factor, the
  identifiers run 1 to n inside 32 bits with the build identifiers beside them, every other column
  equals the spine's (the reviewed export drops aside), and policyengine-uk reads back the
  identifiers and the solved 2025 weights.
- **M**: the stored birth months are the spine's, row for row.
- **Written against the solve**: of the 1,216 solved rows only the two ONS land rows, declared
  weight-share formulas, read differently from the file (by at most 0.22 points). The
  private-school pupils row, 3.4 points off on #1271's head, now matches the solve (U4). The
  file's loss over the written register is higher than the solve's because that register, the
  scoring surface, keeps the bands the support floor drops.
- **Input mass**: the certifier's input-mass gate passes on the calibrated weights.

## O2. The directors' dividends, taken out

DIVIDGRO, the gross dividends a director received from their own company, was taken into
`dividend_income` as the larger of the accounts' dividends and the director's. On the head's
spine it added about £5.8bn of dividends on FRS-channel people at the spine's weights, more than
doubling that channel's dividends. The directors move up the income bands, the bands they leave
lose support, and the national solve concentrates weight on fewer households, so the weight-ratio
gate fails (the "head with O2" arm). With only the directors' dividends left out, all seven
gates pass. How FRS directors' dividends sit beside the SPI channel's high-income copies, which
carry the tape's dividends already, is left to its own PR; the item stays open in #1095 with this
finding. O1, the owner-manager inputs, stays: its income share reads job DIVIDEND and does not
touch `dividend_income`.

## U2c. The childcare take-up refit

The four rates, fitted on the bump arm's spine (policyengine-uk 2.132.6, feed 735820c, 2024 base
year) by the version-4 fitter: Tax-Free Childcare 0.9797, extended hours 0.6579, universal hours
0.4353 and targeted hours 0.7769. L-BFGS-B converged inside the bounds of 0 and 1 for every rate.

- **The Tax-Free Childcare rate is near its bound, not at it.** María's U2c ruling, that a rate at
  its bound keeps the bound and the receipt records the residual, applies to a rate the
  optimizer stops at 0 or 1; 0.98 is an interior optimum, so the rule does not apply. The rate
  sits near 1 because HMRC's two rows pull against each other: even at a rate of 1 the frame's
  Tax-Free Childcare spend is 0.88 of HMRC's while its children with used accounts are 1.13 of
  HMRC's. The fit settles where the two misses balance, at spend 0.86 and children 1.11, so the
  residual is the frame's spend per child, about 0.78 of HMRC's, not the rate.
- **The universal rate is low because the universal-only pool is large.** With every scheme at
  a rate of 1 the frame already gives 1.77 times DfE's universal-only children. At the fitted
  extended-hours rate about a third of the children eligible for extended hours take none, and
  they join the universal-only pool, so the standalone rate that meets DfE's count is 0.44.

## U3. The support floor on the head spine

The floor's receipt as the head's national problem binds it: 19 bands dropped, 3.1% of the
scope's published total, against the ceilings of 24 rows and 5%, a margin of five rows and about
two points. The twelve remaining per-row band entries were compiled beside the register, measured
above the floor and left the problem again. By family, in DWP's published households (the monthly
average the rows bind):

- COUPLE_NO_CHILDREN: 10 bands, 42,229 households.
- SINGLE: 8 bands, 71,069 households.
- COUPLE_WITH_CHILDREN: 1 band, 12,836 households.
- LONE_PARENT: none.

The register census agrees: 1,291 compiled rows, 1,216 solved, 75 excluded (56 per-row entries
and the 19 bands). The 21 per-row band entries, sorted on the spine with O2 and confirmed on the
head's (support counts stay local; the category is the only thing given):

- (a) below the floor, so the floor drops the band and the entry retires (9): COUPLE_NO_CHILDREN
  22,800–24,000, 24,000–25,200, 26,400–27,600, 27,600–28,800 and 28,800–30,000; SINGLE
  21,600–22,800, 26,400–27,600, 27,600–28,800 and 28,800–30,000.
- (b) at or above the floor but out of reach with every weight at 10 times its prior: none.
- (c) neither, so the entry stays and is renewed per row, or not, before 26 November (12):
  LONE_PARENT 12,000–13,200, 13,200–14,400, 14,400–15,600, 15,600–16,800, 16,800–18,000,
  18,000–19,200, 19,200–20,400 and 21,600–22,800; SINGLE 8,400–9,600, 9,600–10,800,
  13,200–14,400 and 14,400–15,600.

The floor also drops ten bands the solve bound before: COUPLE_NO_CHILDREN 10,800–12,000,
19,200–20,400, 20,400–21,600, 21,600–22,800 and 25,200–26,400; COUPLE_WITH_CHILDREN 28,800–30,000;
SINGLE 20,400–21,600, 22,800–24,000, 24,000–25,200 and 25,200–26,400. Which bands sit under the
floor can move from one spine to the next, so the ceilings keep a margin over the head's drops.

## C1. The council-tax stock on the household basis

The factors, each area's published households over its chargeable dwellings on feed 735820c, run
from 0.980 to 1.023 across the eleven areas; the largest shift from one is 0.0227, and the
tripwire refuses 0.035. On the head's calibration, against the control:

- the 97 council-tax stock rows miss by 0.128% on average (control 0.644%), and none by more than
  2% (control: 10 rows);
- the 11 household controls miss by 0.061% on average (control 0.072%).

Both acceptance conditions of the plan hold: each family within 0.2% on average, and no stock row
past 2%.

## D1b. The plan 2 stock

The head spine's Plan 2 England stock, at design weights, is about 41% of SLC's academic year
2024/25 stock of borrowers liable to repay; the top-up walk's whole pool, measured before the
head, would bring it to about 47%. The plan's acceptance, within 25% of SLC, cannot be met by the
cohort rule, so the two #868 exclusions stay until María rules on them (they expire on 3 November).

## Tax credits (Vahid's fourth nit on #1271)

Child tax credit and working tax credit are zero from 2025 on the control's file (policyengine-uk
2.130.1) and on the head's (2.132.6), so the legacy closure takes effect on both engines.
Universal Credit at 2025 is £77.7bn on both at their calibrated weights.

## Tests

- CI runs on every push; the final head's run is in the PR.
- The touched surfaces at the head: 1,479 tests pass, 1 skipped (engine-free), and the SPI engine
  tests pass, 88 with 1 skipped across the SPI income and state-pension-age files.
- CI's engine-uk job list at the head's code (`f57c50091`): 213 tests, all pass. The review
  commits add two tests and change no build code. At their head, the FRS spine, SPI income
  (engine-free and engine), student-loan, take-up tool and contract, energy pricing and graph
  tests pass: 1,073 with 2 skipped. Mutations Vahid listed now fail them: reading only code 3
  as outside the support group, and deleting either SPI-stage reset.
- The feed-gated tests (`CHRONICLE_UK_FACTS` on feed 735820c) pass at the PR's first push: 103,
  2 skipped. The engine-free lane there ran 18,895 tests; its three failures were a contract hash
  the next push repinned and two tests that read this machine's environment (a live release
  pointer and a spawned-worker US scorer), which pass in CI.
