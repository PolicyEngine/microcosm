# microcosm#1359 receipts: alcohol, tobacco and gambling inputs for policyengine-uk's excise duties

Plan: `repos/microcosm-1359-implementation-plan.md`, approved 2026-10-23. María's rulings of that day:

- Quantities are levelled to published FY2024-25 figures. Spending, the division totals and the VAT base stay at survey level.
- Diary incidence is kept; there is no redraw.
- Alcohol volume follows spending, with the bias documented (decision 2).
- FY2024-25 is the quantity basis (decision 1).
- The OBR alcohol and tobacco rows are measured at FY2024-25 (decision 3).
- Gambling is implemented on the policyengine-uk bump that carries it, and the branch is built on that bump (2026-10-24).

Branch `uk-excise-quantities-1359`, stacked on microcosm#1383 (`uk-pe-uk-bump-2572` at `74a2a378e`). That PR locks policyengine-uk 2.151.0, the first release with the gambling duties (PolicyEngine/policyengine-uk#2590).

## Commits

- `db9b2aeff`: the tobacco and on-trade alcohol shares drawn inside the LCFS chain.
- `1f014b3dc`: the `level_excise_quantities` step and its gate, dormant until declared.
- `5398cb86b`: the engine checks of the excise inputs.
- `602905bbc`: the clearances are read in the units Chronicle stores them in.
- `9e62036a0`: the pub lean of the spending rule is documented.
- `46c0820cd`: the gambling stakes share of recreation.
- `e44af4a97`: draught shares and the gambling duty bases.
- `ecc3aaec1`: the re-pin to `200bf6d`, the vendored excise facts and the five measured OBR rows.
- `8c49c19d7`: the step is declared and its inputs are exported.
- `1c96edc93`: the charter-H2 fixture is regenerated.

## What the stage does

**Three scratch shares.** The `lcfs_consumption` chain draws them on the donor's diary lines, after its twenty targets: the eighteen spend targets and the two heating fuels microcosm#1383 draws.

- tobacco within P602's off-trade alcohol and tobacco lines;
- on-trade alcohol within P611;
- gambling stakes (the seven games-of-chance stake lines) within P609.

Appending targets to the identity-keyed chain leaves every earlier draw bit-identical. A household's parts are its shares of the totals it drew, so no total moves.

**The `level_excise_quantities` step.** It shares each published FY2024-25 national figure out to households in proportion to the spending that buys it, at prior weights (one factor per input, the `level_road_fuel` scaling). The figures come from `excise_facts.json`, vendored from the pinned Chronicle feed:

- litres of beer, cider, wine, spirits and other fermented products: HMRC's clearances in litres of pure alcohol over the engine's own strength for each product, which policyengine-uk multiplies back;
- the beer and cider draught shares: draught-relief clearances (alone or with small producer relief) over all clearances;
- cigarettes a week and grams a week of hand-rolling tobacco, cigars and other tobacco: HMRC's clearances over the engine's 52 weeks;
- the thirteen gambling duty bases. Each single-duty base is HMRC's cash receipts for the duty, or Machine Games Duty's liabilities for the band, over the engine's FY2024-25 rate, so the engine charges back the published receipts. General Betting Duty's base, its receipts over 15 percent, is split by the Gambling Commission's gross gambling yield for Great Britain.

Sparkling cider, heated tobacco and spread betting have no published quantity of their own and keep the engine's zero.

**Deviation from the plan's Group C.** The plan took the Gambling Commission's yield by sector as each base, with HMRC's receipts as the check. The yield covers Great Britain only and includes undutiable parts (on-course betting, society lotteries). Receipts over rates give the dutiable UK base directly, and reproduce the receipts the way the clearances reproduce alcohol and tobacco duty. The yield now only splits General Betting Duty.

## The spending rule for alcohol volume

No on-trade volume share can be archived: MESAS is a NielsenIQ and Circana dashboard with no fixed file. So a pound buys the same volume in a pub as in a shop. Against a published on-trade volume share of about 27 percent, measured on the LCFS 2024-25 donor at survey weights:

- a pound spent in a pub is given about 1.6 times the volume it buys (1.4 to 1.7 times for shares of 25 to 30 percent);
- a pound spent in a shop is given about 0.78 times;
- about 11 percent of the volume, and so of the duty, sits with other households (9 to 13 percent);
- the bottom three income deciles bear about 7 percent too little of the duty, and the top decile about 7 percent too much.

## The national figures the step levels to

These come from the declared rows of the pinned feed (`200bf6d`, FY2024-25, UK unless stated). They are public figures only.

- **Alcohol, litres of product a year, at the engine's strengths:**
  - beer 4.63bn (4.0%) and cider 0.67bn (4.5%);
  - wine 1.27bn (12%), spirits 0.33bn (40%) and other fermented products 0.24bn (5%);
  - draught shares: beer 0.397 and cider 0.233.
- **Tobacco, a year:** cigarettes 13.58bn sticks; hand-rolling tobacco 4,028 tonnes, cigars 395 tonnes and other tobacco 301 tonnes. The engine's inputs are a week, so each is divided by 52.
- **Gambling duty bases, a year:**
  - remote gaming £5.54bn, the National Lottery £7.77bn, land-based casinos £0.70bn, bingo £0.25bn and pool betting £0.05bn;
  - Machine Games Duty bands: lower £0.14bn, standard £3.02bn, higher £2.4m;
  - General Betting Duty's £4.76bn base, split by the Commission's Great Britain yield: land-based fixed odds 32.1 percent, remote UK horse racing 18.7, remote other 45.5, exchange 2.6, horse and dog pools 1.2.

## The stage on a real spine

A stage-only run on microcosm#1333's spine (78,440 households), both runs on policyengine-uk 2.151.0. microcosm#1383's head (`74a2a378e`, the `abf0921` feed) is set against the branch head (the `200bf6d` feed); see `docs/evidence/uk-excise-quantities-1359/`.

- **Existing columns.** Thirteen of the twenty-two columns #1383 writes are byte-identical.
- **Columns the re-pin moves.** Every other existing column moves through the re-pin, not the excise step:
  - petrol and diesel move by −4.227 percent, exactly the September Consumer Trends revision of 07.2.2;
  - the seven divisions #1383's levelling holds at the shared frame-to-donor ratio move by −0.149 percent each, because the lower road fuel lowers that ratio. The seven are furnishings, health, transport, communication, recreation, education and miscellaneous.
- **New outputs.** The head adds the 24 excise inputs, and fit records go from 25 to 28.
- **Gates.** All eight LCFS stage-health gates pass at the head, including `uk_stage_lcfs_consumption_excise_quantities`.
- **The spending that shares out the quantities** (prior weights, frame):

  | Spending | Total | Households with any |
  |---|---|---|
  | Tobacco | £3.68bn | 8.2% |
  | Off-trade alcohol | £16.28bn | 46.4% |
  | On-trade alcohol | £15.69bn | 30.4% |
  | Gambling stakes | £2.27bn | 14.6% |
- **The shares on the frame against the donor:**

  | Share | Frame | Donor | Frame positive | Donor positive |
  |---|---|---|---|---|
  | Tobacco, of P602's off-trade lines | 18.4% | 20.6% | 8.2% | 10.3% |
  | On-trade alcohol, of P611 | 18.6% | 13.9% | 30.4% | 31.0% |
  | Gambling stakes, of P609 | 1.5% | 1.5% | 14.6% | 17.0% |

  Every share rests on 30 or more donors. Tobacco incidence on the frame sits about a fifth below the donor's. Because the levels come from the clearances, that moves only who carries the duty, not how much there is.

## The re-pin

The feed moves from `abf0921` to `200bf6d` (1,502,291 rows, up 2,990). Chronicle #384 retired the June Consumer Trends edition, so the feed carries September's. `docs/uk-chronicle-feed-repin.md` has the full entry. In outline:

- **Vendored resources.** `excise_facts.json` is new, with 98 rows. `ons_household_expenditure_facts.json` takes the September values. Every other resource carries the same rows under the new feed identity.
- **National surface.**
  - Calendar-2024 road fuel (07.2.2, bound) moves from £37.025bn to £35.460bn, −4.2 percent. It is also the level the stage sets petrol and diesel to.
  - The measured calendar-2025 electricity row moves from £26.522bn to £28.840bn, and the gas row from £15.339bn to £18.958bn.
  - The five OBR excise rows are added: beer and cider £3.746bn, wine £4.686bn and spirits £4.113bn, all accrued; tobacco £7.927bn and betting and gaming £3.616bn, both cash. Each is measured at FY2024-25, and the national compile-parity receipt lists them as ledger-only.
- **Local surface.** Byte-identical, apart from the membership's feed label.

## Tests

Run on the restacked head and kept to the surfaces the change touches:

- **The LCFS stage, its stage-health gate and battery bindings:** 294 tests once the step was restacked; with the stage manifest, 168 once the gambling share was.
- **The re-pin:** the pin, the contract and references, the measure register and its tool, the holds, vendoring, the coverage manifest and the resource lists. 245 tests; the six feed-gated ones skip.
- **The declaration:** the spec, schema and re-emission, the gates and contract mirrors, the export surface, the graph, diagnostics and the LCFS stage. 863 tests.
- **Engine, the locked policyengine-uk 2.151.0:** the five engine files the stage touches, 65 of 65.
- **Charter-H2 parity** (`test_acceptance_h_parity.py`, engine and engine-free): 3 of 3.
- **Integration smoke** (`HF_STAGING_READ_TOKEN= bash tools/run_integration_tests.sh`): passes.

Before the restack, the full engine-free UK and shared suites ran once on the earlier head (14,175 tests); their only failures were environmental. The vendoring and national-reference regeneration checks against the full feed were not run: each holds the feed twice, about 50 GB. Instead, the tools ran on the pin, and their outputs were checked by diff against the base.

## Disclosure

Nothing here rests on fewer than ten households. The donor shares' parent residuals (P602 beside its lines, the P611 clip) are withheld as "<10" in every receipt. The support band is reported as a band. No minimum, maximum or single record appears.

## Open items

- **Stacked on microcosm#1383.** This PR's CI runs once #1383 merges. #1383 must be merged without deleting its branch, or this PR closes with it.
- **The signatures** of the five measure exclusions (expiring 2027-01-08) and the `matched_period` hold (expiring 2026-12-11) were approved by María on 2026-10-24 as drafted.
- **The measurement arms**, a control on #1383's head and the head, each a spine plus a national calibration. They will give the duty against OBR, decomposed, the distribution by decile and the drift at final weights. The exclusion reasons will then be restated with those figures.
- **The scoring register.** The uk-candidate-eval register is re-frozen before the next national solve, since the national register version moves with the five new references.
