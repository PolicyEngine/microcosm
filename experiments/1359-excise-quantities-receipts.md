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
- `352af5e41`: these receipts.

Vahid's review at `e07424ddf`, addressed on the restacked branch:

- `9e143959b`: Lottery Duty and the other gambling bases keyed to their own stakes, and the known biases stated in the stage notes.
- `303d1abf0`: the engine's duty on the national levels pinned against OBR, and the exclusion reasons restated with it.

On 2026-10-24 the branch was rebased onto main after microcosm#1383 merged, so the hashes above are the pre-rebase ones.

## What the stage does

**Five scratch shares.** The `lcfs_consumption` chain draws them on the donor's diary lines, after its twenty targets: the eighteen spend targets and the two heating fuels microcosm#1383 draws.

- tobacco within P602's off-trade alcohol and tobacco lines;
- on-trade alcohol within P611;
- gambling stakes (the seven games-of-chance stake lines) within P609;
- the National Lottery's share of those stakes (scratchcards and draws, c94316 and c94319);
- the betting and gaming share of the rest (football pools, bingo, and bookmaker and tote betting with gaming machines: c94311, c94312 and c94314). Society lotteries (c94313) and the Irish Lottery (c94315) pay no UK gambling duty.

Appending targets to the identity-keyed chain leaves every earlier draw bit-identical. A household's parts are its shares of the totals it drew, so no total moves.

**The `level_excise_quantities` step.** It shares each published FY2024-25 national figure out to households in proportion to the spending that buys it, at prior weights (one factor per input, the `level_road_fuel` scaling). The figures come from `excise_facts.json`, vendored from the pinned Chronicle feed:

- litres of beer, cider, wine, spirits and other fermented products: HMRC's clearances in litres of pure alcohol over the engine's own strength for each product, which policyengine-uk multiplies back;
- the beer and cider draught shares: draught-relief clearances (alone or with small producer relief) over all clearances;
- cigarettes a week and grams a week of hand-rolling tobacco, cigars and other tobacco: HMRC's clearances over the engine's 52 weeks;
- the thirteen gambling duty bases. Each single-duty base is HMRC's cash receipts for the duty, or Machine Games Duty's liabilities for the band, over the engine's FY2024-25 rate, so the engine charges back the published receipts. General Betting Duty's base, its receipts over 15 percent, is split by the Gambling Commission's gross gambling yield for Great Britain. Lottery Duty's base follows the National Lottery stakes and the other twelve follow the betting and gaming stakes.

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

## The duty those levels give, against OBR

`docs/evidence/uk-excise-quantities-1359/outputs/national_reconciliation.json`. One household holds every national level, run through policyengine-uk 2.151.0 at FY2024-25. Every duty is linear in its inputs, so this is the frame's duty at prior weights, and the final-weight arms will add only the drift. The engine test `test_the_national_levels_sit_at_the_stated_gaps_from_obr` pins each gap to half a percentage point.

| OBR row | Engine on the levels | OBR FY2024-25 | Gap |
|---|---|---|---|
| Beer and cider (accrued) | £4.050bn | £3.746bn | +8.1% |
| Wine and other fermented (accrued) | £4.808bn | £4.686bn | +2.6% |
| Spirits (accrued) | £4.205bn | £4.113bn | +2.2% |
| Alcohol total | £13.062bn | £12.545bn | +4.1% |
| Tobacco (cash) | £8.167bn | £7.927bn | +3.0% |
| Betting and gaming (cash) | £3.619bn | £3.616bn | +0.1% |

- **Alcohol and tobacco.** The gaps are what the engine leaves out. For beer, that is one strength per product with no small producer relief or lower-strength rate. For tobacco, it is one pack price for the ad valorem element and the minimum excise tax, and cash receipts against the year's clearances.
- **Betting and gaming is an identity.** Each base is receipts over the engine's own rate, so the engine charges the receipts back. Its £3m gap is Machine Games Duty's liabilities (£612.3m) against its cash receipts (£609.2m); HMRC's receipts for all the duties are OBR's £3.616bn. Only weight drift can move the row, so it says nothing about whether the gambling inputs are right. Its exclusion has its own reason.
- **Household tax.** The three duties add £24.8bn to `household_tax` at FY2024-25, and `household_net_income` falls by the same.

## The stage on a real spine

Stage-only runs on microcosm#1333's spine (78,440 households), all on policyengine-uk 2.151.0; see `docs/evidence/uk-excise-quantities-1359/`.

- **Against #1383's head** (`74a2a378e`, the `abf0921` feed), the restacked head (the `200bf6d` feed):
  - thirteen of the twenty-two columns #1383 writes are byte-identical;
  - every other existing column moves through the re-pin, not the excise step. Petrol and diesel move by −4.227 percent, exactly the September Consumer Trends revision of 07.2.2. The seven divisions #1383's levelling holds at the shared frame-to-donor ratio move by −0.149 percent each, because the lower road fuel lowers that ratio. The seven are furnishings, health, transport, communication, recreation, education and miscellaneous;
  - the head adds the 24 excise inputs, and fit records go from 25 to 28.
- **The gambling split against the restacked head:** 33 of the 46 columns are byte-identical; only the 13 gambling inputs move. Fit records go from 28 to 30.
- **The rebase onto main** (after microcosm#1383 merged), against the split head: 26 of the 46 columns are byte-identical. #1383's last commits level each division on the spend the chain drew against the donor's same spend, so the seven levelled divisions move, recreation from £148.7bn to £151.0bn. The 13 gambling inputs follow recreation, since the stakes are a share of it. The alcohol and tobacco inputs, the spending they follow and every share's incidence are unchanged.
- **Gates.** All eight LCFS stage-health gates pass on every head run, including `uk_stage_lcfs_consumption_excise_quantities`.
- **The spending that shares out the quantities** (prior weights, frame):

  | Spending | Total | Households with any |
  |---|---|---|
  | Tobacco | £3.68bn | 8.2% |
  | Off-trade alcohol | £16.28bn | 46.4% |
  | On-trade alcohol | £15.69bn | 30.4% |
  | Gambling stakes | £2.31bn | 14.6% |
  | of which the National Lottery | £1.69bn | 10.7% |
  | of which betting and gaming | £0.43bn | 2.0% |
- **The shares on the frame against the donor:**

  | Share | Frame | Donor | Frame positive | Donor positive |
  |---|---|---|---|---|
  | Tobacco, of P602's off-trade lines | 18.4% | 20.6% | 8.2% | 10.3% |
  | On-trade alcohol, of P611 | 18.6% | 13.9% | 30.4% | 31.0% |
  | Gambling stakes, of P609 | 1.5% | 1.5% | 14.6% | 17.0% |
  | The National Lottery, of the stakes | 73.3% | 51.2% | 10.7% | 11.4% |
  | Betting and gaming, of the other stakes | 69.6% | 72.9% | 2.0% | 3.6% |

  Every share rests on 30 or more donors.

## Who carries the duty

`docs/evidence/uk-excise-quantities-1359/outputs/excise_incidence.json`, at prior weights.

**Each duty is one multiple of the spending that shares it out.** The levels are national while spending stays at diary level, so every household that carries a duty pays the same multiple of its own spending:

| Duty | Duty | Spending it follows | Multiple | Households carrying it |
|---|---|---|---|---|
| Alcohol | £13.06bn | £31.97bn | 0.41 | 57.6% |
| Tobacco | £8.17bn | £3.68bn | 2.22 | 8.2% |
| Lottery | £0.93bn | £1.69bn | 0.55 | 10.7% |
| Other gambling | £2.69bn | £0.43bn | 6.25 | 2.0% |

The diary records a fraction of what is bought. The National Lottery's base, ticket sales, is 4.6 times the National Lottery stakes it follows, and the other gambling bases are 34 times the betting and gaming stakes. So a tobacco or betting household pays more duty than the diary says it spent. VAT on the same goods stays on the survey base.

**The duty sits on fewer households than buy.** 8.2 percent of households carry tobacco, against 11.9 percent of UK adults who smoked cigarettes in 2023 (ONS, *Adult smoking habits in the UK: 2023*). 10.7 percent carry the National Lottery and 2.0 percent betting and gaming, against 47 percent of adults in Great Britain who gambled in the past four weeks of 2025, or 27 percent once lottery-draw-only players are left out (Gambling Commission, *GSGB annual report 2025*). The share of households with a smoker or a gambler is above the adult rate, so these comparisons understate the gap. With the levels fixed, the duty per carrying household is higher by the same factor.

**Where the frame loses incidence.** Part of the gap is the diary's two weeks, and part is the frame's. Weighted shares of households:

| Part | Parent | Donor positive | Frame positive | Donor rate within parent | Frame rate within parent |
|---|---|---|---|---|---|
| Tobacco | Alcohol and tobacco | 10.3% | 8.2% | 18.9% | 15.5% |
| Gambling stakes | Recreation | 17.0% | 14.6% | 17.5% | 14.8% |
| National Lottery | Stakes | 11.4% | 10.7% | 67.1% | 73.0% |
| Other stakes | Stakes | 7.7% | 4.2% | 45.3% | 28.7% |
| Betting and gaming | Other stakes | 3.6% | 2.0% | 46.5% | 46.6% |

The parents carry almost the donor's incidence: alcohol and tobacco spending is positive for 52.7 percent of the frame against 54.6 percent of donors. The loss is in the share draws. A share comes out zero more often on the frame than on the donor among households that carry the parent. Within the stakes, the chain more often draws all of a household's stakes as the National Lottery's, so other stakes, and with them betting and gaming, lose almost half their incidence. The regime gate of the share draw is the place to look. A prevalence redraw, like `redraw_zero_road_fuel`, is the follow-up the review names.

**Other known biases**, in the stage notes:

- Other tobacco (c22131) includes e-cigarette refills, so the tobacco inputs also follow vaping spending, and a household that only vapes carries tobacco duty. ONS counts 9.8 percent of adults in Great Britain using e-cigarettes in 2023. Vaping Products Duty, from 1 October 2026, needs a key of its own.
- The draught shares are national constants, so a household that buys alcohol only in shops takes the same draught relief as one that drinks in pubs.
- Remote UK horse racing is split out by the horse-racing share of all remote betting yield, which includes non-UK racing. From April 2027, when the engine charges other remote betting at 25 percent, too much of the base stays at 15 percent, so the rise is understated.

## The gambling split

The review asked for two shares: the lottery and the rest. On the donor the rest includes society lotteries (c94313) and the Irish Lottery (c94315). Neither pays UK gambling duty, and more than half of the donors with non-lottery stakes play only those. So the second share is betting and gaming within the rest, and the society and Irish lotteries carry no base.

Keyed this way, the duties follow the right products. But on the diary's incidence, the betting and gaming duties concentrate:

- **One key** (the restacked head): all £3.62bn of gambling duty on the 14.6 percent of households with any stakes, about £850 a year each.
- **The split** (this head):
  - Lottery Duty, £0.93bn, on the 10.7 percent with National Lottery stakes: about £300 each;
  - the other duties, £2.69bn, on the 2.0 percent with betting and gaming stakes: about £4,700 each.

National revenue doesn't move, but the distribution of any betting or gaming reform does. Until a prevalence redraw lifts betting incidence, it lands on 2 percent of households. The choice between the split, a lottery-only split and one key is María's.

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

For the review's changes:

- **The LCFS stage** (`test_uk_lcfs_consumption.py`): 132 of 132.
- **The surfaces the split moves:** the stage-health gate, the spec and its re-emission, diagnostics, the coverage manifest, battery bindings, the contract mirrors and engine-free H2 parity. 732 of 732.
- **Engine:** the excise inputs with the new reconciliation, 10 of 10. H2 parity and the coverage manifest on the engine, 54 of 54.
- `tools/ci_test_plan.py verify`: ok. The battery digests do not move, since no gate changes.

After the rebase onto main, with the follow-ups to Vahid's third review of microcosm#1383:

- **The stage, the spec and the declaration**, with contract mirrors and both H2 parity files: 931 of 931.
- **The re-pin and the targets**: 353, of which 7 feed-gated checks skip.
- **The withholding, the gates, Orrery and the build CLI**: 631 of 631.
- **Engine**: the excise inputs, H2 parity and the coverage manifest, 64 of 64.
- `tools/ci_test_plan.py verify`: ok. The battery digests do not move.

Before the restack, the full engine-free UK and shared suites ran once on the earlier head (14,175 tests); their only failures were environmental. The vendoring and national-reference regeneration checks against the full feed were not run: each holds the feed twice, about 50 GB. Instead, the tools ran on the pin, and their outputs were checked by diff against the base.

## Disclosure

Nothing here rests on fewer than ten households. The donor shares' parent residuals (P602 beside its lines, the P611 clip) are withheld as "<10" in every receipt. Donor support is reported as a band, and the incidence script refuses a share on fewer than 30 donors. No minimum, maximum or single record appears.

## Open items

- **Stacked on microcosm#1383.** This PR's CI runs once #1383 merges. #1383 must be merged without deleting its branch, or this PR closes with it.
- **The signatures.** The five measure exclusions (expiring 2027-01-08) and the `matched_period` hold (expiring 2026-12-11) were approved by María on 2026-10-24 as drafted. The exclusion reasons restated for the review need her approval again.
- **The measurement arms**, a control on #1383's head and the head, each a spine plus a national calibration. They will give the duty against OBR at final weights, which only adds drift to the table above. They will also give each duty by equivalised income decile beside ONS *Effects of taxes and benefits*, and the drift at final weights.
- **The scoring register.** The uk-candidate-eval register is re-frozen before the next national solve, since the national register version moves with the five new references.
- **Decisions for María:** the gambling keys (the split, a lottery-only split or one key), and whether tobacco and gambling spending should be levelled to Consumer Trends 02.2 and 09.4.3, as road fuel is to 07.2.2. Levelling would move the VAT base.
- **Follow-ups:** a prevalence redraw for tobacco and gambling, which would also correct the share draws' loss of incidence; a key for Vaping Products Duty; draught relief keyed to on-trade spending.
- **The betting and gaming binding note** in `uk_population_targets.json` still puts the row's gap down to spread betting and the two receipts series. Its exclusion reason now gives the gap correctly. Correcting the note re-derives every surface that pins the contract's hash, so it waits for the next change to the contract.
