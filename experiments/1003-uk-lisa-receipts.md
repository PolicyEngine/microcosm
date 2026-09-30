# microcosm#1003 receipts: Lifetime ISA holdings from the WAS round-8 person tab

Plan: `repos/uk-1003-lisa-plan-2026-09-29.md`, approved 2026-09-29. María's rulings:

- 2026-09-23: a new person-grain stage `was_lisa` right after `was_wealth`; the household total
  capped pro rata at `gross_financial_wealth`, with a receipt.
- 2026-09-29: a declared credibility rule for impossible donor records; ownership from a
  weighted ridge logistic with person-keyed draws; the balance from the holders-only house QRF.

Branch `uk-1003-lisa` in `repos/populace-1003`, off main `5187fce25` (no UK change since
`6a70cd4ee`, the #901 merge). Commits: C1 `88182f311` (runtime), C2 `0c7f19002` (manifest,
graph, driver, fixture), C3 `360e7846c` (gates, export cells, support bounds). The licensed
builds, sidecars and measurement scripts live under `data/ukds/acceptance/1003-lisa/`, outside
the tree. `docs/evidence/uk-lisa-1003/` has aggregate extracts, and
[`docs/uk-lisa-1003.md`](../docs/uk-lisa-1003.md) is the topic doc.

Disclosure control follows the lane convention:

- Donor counts under 10 are suppressed, and so is any count that would reveal one by
  subtraction.
- Balance quantiles and means are rounded to two significant figures, and weighted medians of
  financial wealth to three.
- No single respondent's value is quoted.

Builds. Every licensed build used the SPI-first spine recipe of the #930 lane plus
`--was-person-tab`, with seed 42 and the same pinned inputs. Both trees carry one
measurement-only relaxation, never committed, while #1049 is open on main: the PLAN_5 student-loan
realisation deviation limit is raised from 1.0 to 2.0 in `uk/gates.json`. The builds are:

- `spine-ctl`: main `5187fce25`.
- `spine-lisa`: the branch head `360e7846c`.
- `spine-lisa-2`: the same tree again, for determinism.
- `spine-lisa-b`: the head with the four financial predictors removed from both models. This is
  measurement only and was not committed (Part F).
- `spine-lisa-d`: the head with the donor's income predictors uprated to the spine year (review
  round 1, item 3). This is measurement only and was not committed (Part F).

Rebased 2026-09-30 onto main `c5a1cba87`, after #932, #1057 and #1045. The licensed builds here
were measured before that, at `360e7846c` on main `5187fce25`. Every spine change in #1045 is
downstream of `was_lisa` (`cgt_support_split` replaces `cgt_band_donors`), so the stage-time
population and the stage evidence in Parts A to D and F stand. The H5 cross-check in Part C and
the twin in Part E describe the old base's downstream stages.

## Part A. The donor audit

The stage reads two licensed tabs, pinned in `uk/spec/sources.yaml` and checked by size and
sha256 at runtime:

- the round-8 household tab, the `was_wealth` donor with the same pins;
- the person tab `was_round_8_person_eul_may_2025_230525.tab` (sha256 `1ca6fd37…d71a`,
  356,930,989 bytes).

It reads 14 of the person tab's 4,079 columns. The person tab was extracted from the private
mirror's `was_2006_22.zip` (at `b3322dd5`), together with its `mrdoc/` documentation.

Join and pin check (`clean_was_lisa_donor` on the pinned tabs):

- 32,271 persons read in 15,128 households. Every person joins exactly one household on
  `CASER8`.
- The released person values `DVFLISAvR8` sum to the household's `DVFLISAVR8_aggr` on all
  15,128 households. The largest absolute difference is £0; the tolerance is £1.
- 4,996 dependent children are excluded. None carries a LISA value.
- The donor is 27,275 non-dependent adults, weighted to 50.2 million.
- There are no missing or negative values and no sentinel recodes of the earnings predictor.
- The released ownership flag `fisa_binary3r8_i` agrees with a positive value on every record.
- There are 141 released holders, 0.93% of weighted adults.

Response classes after the credibility rule are recorded in the stage evidence and never used as
predictors:

- 26,888 observed non-holders.
- 246 non-holders whose ownership ONS imputed.
- 107 holders with an exact reported value.
- 13 holders with a banded value or whose ownership ONS imputed. Both classes are under 10 and are
  reported together.
- 21 holders recoded by the rule.

Most released holders reported an exact value. For every holder who reported a band or had one
imputed, ONS's imputed value lies inside the band (the plan's audit against the round-8 codebook).
The released split between exact and banded values is not published: together with the classes
after the rule it would reveal, by subtraction, a count under 10 among the recoded holders.

The credibility rule (`clean_was_lisa_donor.credibility_rule`; its basis is in the topic doc):

- 21 holders in age bands starting at 45 or above are recoded to non-holders. They are 4.3% of
  the weighted released holders and 26.2% of the weighted LISA mass.
- Fewer than 10 holders above £40,000 stay holders but leave the balance fit.
- Together these records are about 5% of the weighted holders and about a third of the weighted
  LISA mass.
- After the rule there are 120 holders, 0.890% of weighted adults.

Credible holder balances, weighted:

- p10 £500
- p25 £1,500
- median £6,000
- p75 £10,000
- p90 £10,000
- weighted mean £6,100

Donor ownership by group (weighted share of adults):

| Group | Share |
| --- | --- |
| all adults | 0.890% |
| 16–24 | 2.09% |
| 25–34 | 2.39% |
| 35–44 | 1.70% |
| 45–54 | 0 (the rule) |
| 55+ | 0 (the rule) |
| female / male | 0.72% / 1.07% |
| private renters / other tenures | 2.74% / 0.54% |
| household net income tertiles 1–2 / tertile 3 | 0.45% / 1.78% |

The donor's first age group spans 16 to 24, not 18 to 24. WAS counts adults from 16, and the
public file's band 15–19 cannot separate the 16- and 17-year-olds, who cannot hold a LISA, from
the 18- and 19-year-olds. That band is about 30% of the group's weight and holds fewer than 10 of
its holders. The model keeps the band, because it carries the 18–19-year-old holders, and applies
the group's coefficient to recipients aged 18 to 24.

## Part B. The ownership model

Held-out comparison on the cleaned donor: 5 folds by household (fold seed 7), run through the
stage's own `fit_ownership_model` with the committed declaration. The alternatives are:

- the house regime gate (`HistGradientBoostingClassifier` at library defaults, as
  `RegimeGatedQRF` fits it);
- the same classifier tuned for rare events;
- the donor's weighted rates by age group.

Each entry is the probability-weighted ownership share of held-out adults, and the last row is
the held-out weighted log loss:

| Held out | Donor | Stage logistic | House gate | Tuned boosting | Age-group rates |
| --- | --- | --- | --- | --- | --- |
| all adults | 0.890% | 0.873% | 0.582% | 0.718% | 0.884% |
| 16–24 | 2.09% | 2.03% | 1.28% | 1.53% | 2.09% |
| 25–34 | 2.39% | 2.33% | 1.32% | 1.86% | 2.36% |
| 35–44 | 1.70% | 1.67% | 1.07% | 1.48% | 1.69% |
| 45–54 | 0 | 0.006% | 0.098% | 0.015% | 0 |
| 55+ | 0 | 0.009% | 0.096% | 0.015% | 0 |
| private renters | 2.74% | 2.69% | 1.35% | 1.87% | 1.26% |
| other tenures | 0.54% | 0.52% | 0.43% | 0.50% | 0.81% |
| income tertiles 1–2 | 0.45% | 0.53% | 0.35% | 0.41% | 0.80% |
| income tertile 3 | 1.78% | 1.56% | 1.05% | 1.34% | 1.04% |
| log loss × 1000 | | 39.4 | 53.3 | 41.4 | 44.0 |

How the models compare:

- **House gate.** It is miscalibrated for an outcome held by under one per cent of adults. It
  reproduces 0.58% against 0.89%, understates each age group under 45 by more than a third, and
  has the worst held-out log loss.
- **Tuned boosting.** Tuning closes part of the gap, not all of it.
- **Age-group rates.** They are calibrated by age but blind to tenure and income.
- **Stage logistic.** It matches every age and tenure margin within 0.07 points, with the lowest
  log loss.

Every arm sees only the predictors the stage reads. Since review round 1 the stage no longer
reads self-employment income, which no model used, so the two boosting arms lost it as a
feature; the logistic and the age-group rates are unchanged.

A plan-time probe found the logistic's fit stable for the ridge constant C from 0.1 to 10. María
ruled for the logistic on 2026-09-29.

Balance, held out on the credible holders (the committed holders-only QRF, drawn at the stage's
person-keyed quantile):

- Held-out draws: p10 £500, p25 £950, median £5,000, p75 £8,800, p90 £15,000, mean £5,700.
- The donor's own values: £500, £1,500, £6,000, £10,000 and £10,000, mean £6,100.

With under 100 training holders per fold the held-out draws are noisy. Their median sits £1,000
below the donor's, their p90 sits above it, and their mean is 7% lower.

The ownership model on the full donor (stage evidence), in terms of standardised coefficients:

- The age groups under 45 carry +0.88 to +1.06, and 45 and over −1.03 and −0.89.
- Log gross financial wealth has +0.97, log earnings +0.76, log savings +0.42, and private
  renting +0.40.
- Log stocks-and-shares ISA has −0.28 and the number of children −0.18.
- Sex (−0.09), cash ISA and household net income are small.

The weighted mean probability matches the donor share (0.8897% against 0.8902%), as the
unpenalised intercept guarantees up to solver tolerance. By age group, the model's mean
probability is within 0.02 points of the donor share. The solver converged in 22 iterations.

## Part C. Realised holdings against the donor

Stage evidence of `spine-lisa`: 57,733 persons at stage time, 45,920 of them adults. The 11,813
persons under 18 are set to zero. Weighted ownership share of adults:

| Group | Donor | Spine | Spine, arm B |
| --- | --- | --- | --- |
| all adults | 0.890% | 1.25% | 1.06% |
| model's expected share on the spine | | 1.36% | 1.09% |
| 18–24 (donor 16–24) | 2.09% | 4.14% | 2.91% |
| 25–34 | 2.39% | 3.02% | 3.00% |
| 35–44 | 1.70% | 2.61% | 2.15% |
| 45–54 | 0 | 0.020% | 0.020% |
| 55+ | 0 | 0.020% | 0.006% |
| female | 0.72% | 1.07% | 0.91% |
| male | 1.07% | 1.44% | 1.22% |
| private renters | 2.74% | 4.01% | 3.53% |
| other tenures | 0.54% | 0.68% | 0.55% |
| household net income tertiles 1–2 | 0.45% | 0.87% | 0.71% |
| household net income tertile 3 | 1.78% | 2.01% | 1.77% |
| FRS-channel rows | | 1.38% | 1.12% |
| SPI-channel rows | | 1.11% | 1.00% |

Arm B is the measurement build without the financial predictors (Part F). Income tertiles are
computed on each side's own distribution. The donor's lowest tertile has fewer than 10 holders,
so tertiles 1 and 2 are merged throughout. On the spine the merged share is the mean of its two
first tertiles' shares, which carry equal weight by construction.

The spine holds 1.4 times the donor's share. Part F traces the excess to the spine's WAS
financial draws.

Owner balances, weighted:

| Owners | p10 | p25 | Median | p75 | p90 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| donor credible holders | £500 | £1,500 | £6,000 | £10,000 | £10,000 | £6,100 |
| spine | £1,000 | £2,400 | £6,500 | £10,000 | £15,000 | £6,700 |
| spine, arm B | £500 | £1,400 | £5,000 | £9,000 | £10,000 | £5,600 |

No drawn balance exceeds £20,000, and every balance lies within the credible donor range. The
weighted LISA total at stage time is £4.45 billion. That figure is 2020-22 pounds on the 2024-25
spine population and is not uprated.

Co-holding: 24% of the donor's weighted holder households have two or more holders, against 4%
on the spine. Persons are drawn independently given their predictors (topic doc, coverage
section). This is receipted and not modelled.

Cross-check against the H5. `lisa_receipt.py` recomputes the realised figures from the written
`spine-lisa.h5`, which holds 116,077 persons and 92,321 adults. It finds 1.258% of weighted
adults holding, an owner mean of £6,800 and £4.55 billion. The H5 is the spine after the CGT
stages:

- `cgt_incidence_clone` copies every household and splits its weight.
- `cgt_band_donors` adds 392,000 households of support mass drawn from existing households.

The copies carry their source's LISA cells, so shares move by under 0.01 points and the total by
the donors' mass.

## Part D. Coherence, support and release checks

Coherence cap (`cap_lifetime_isa_to_financial_wealth`, pro rata within the household):

- Fewer than 10 households had a LISA total above their `gross_financial_wealth` draw, and their
  persons were scaled.
- No owner was cleared.
- £23 million weighted was removed, 0.51% of the weighted LISA mass.
- After the cap no household total exceeds its gross financial wealth. On the written H5 the
  household total equals the sum of its persons to £0.

Support: the stage's `support_clip` on `lifetime_isa_balance` clipped no row low and no row high
out of 57,733. The committed `uk/was_lisa_support_bounds.json` bounds the column at
[0, 30,000]: the credible donor range, rounded outward to one significant figure by
`tools/build_uk_was_lisa_support_bounds.py`, whose `--check` passes on the pinned tabs.

Release-side checks on the written H5, with the functions the terminal gates use:

- None of the three cells is degenerate. Their dtypes are bool, float64 and float64.
- The support gate on the committed bounds passes.
- Every value is non-negative.
- All three cells are on the UK export allow-list.
- `has_lifetime_isa` is true exactly where `lifetime_isa_balance` is positive, on every row.

Spine gate battery:

- 27 of 27 gates pass on `spine-lisa`; the control passes 26 of 26.
- The new gate is `uk_stage_was_lisa_support` (`stage_health`, `check: support_clip`,
  `release_blocking`, phase `transferred`). It passes with no failures on one column checked.
- Neither build is a release candidate, so `shippable` is false on both by construction.

## Part E. Twin diff and determinism

Twin: `spine-ctl` against `spine-lisa`. The payload compare and classification ran with
`tools/compare_uk_h5_payload.py` and `tools/classify_uk_payload_diff.py`, against the declared
expectation `spine-lisa-payload-expectation.json`.

The two H5s have equal store keys and equal row counts: 53,806 households, 62,469 benefit units
and 116,077 persons.

- `benunit` and `time_period` are payload-identical.
- `household` differs only by `household_lifetime_isa_balance`.
- `person` differs only by `has_lifetime_isa` and `lifetime_isa_balance`.
- Dropping the new columns leaves frames equal to the control's.
- The root mass log grows from 15 to 16 records. The one added record is the `was_lisa` receipt:
  household mass 29,422,433 conserved, declared factor 1.0. Every control record is still present.

The classifier observed the four expected changes and no expected change went unobserved. It
reported two unexpected items: the household and person `__column_order__`, which the
expectation format cannot declare because the new columns join mid-table. Both were adjudicated
by the drop-and-compare above. Verdict: true.

Determinism: `spine-lisa-2` was rebuilt from the same tree and inputs.

- `tools/compare_uk_h5_payload.py` reports `payload_identical: true` against `spine-lisa`.
- The file digests differ (`587128b5…` and `be100ef3…`). HDF5 writes differ at the byte level.
- Every store key is equal, the root mass log is equal, and the `was_lisa` stage evidence is
  equal.

## Part F. The financial predictors carry the spine's financial-wealth level

This part is an open ruling for María.

The logistic reproduces the donor when held out (Part B), yet on the spine its expected share is
1.36% against the donor's 0.89%. The model is not the cause; its inputs differ.

- **Wealth draws.** The spine's WAS financial draws sit well above the donor's at every age.
- **Coefficients.** The two largest standardised financial coefficients (log gross financial
  wealth +0.97, log savings +0.42) carry that level into ownership.

`fw_probe.py` compares the stage-time spine with the stage's own cleaned donor, in weighted
medians and weighted means of log(1 + x) over adults. The stage-time spine is the candidate H5
without the CGT band donors; a clone and its original carry identical cells and their weights sum
to the stage-time weight.

| Age | Median gross financial wealth, donor | Median, spine | Ratio | Mean log, gross financial wealth, donor / spine | Mean log, savings, donor / spine | Private renters, donor / spine |
| --- | --- | --- | --- | --- | --- | --- |
| 18–24 (donor 16–24) | £12,200 | £48,800 | 4.0 | 9.21 / 10.37 | 5.72 / 6.75 | 19.0% / 30.3% |
| 25–34 | £8,820 | £26,300 | 3.0 | 8.75 / 9.87 | 5.02 / 5.93 | 26.0% / 35.1% |
| 35–44 | £12,000 | £30,200 | 2.5 | 9.13 / 10.10 | 5.13 / 5.99 | 21.0% / 23.6% |
| 45–54 | £15,000 | £47,000 | 3.1 | 9.25 / 10.43 | 5.53 / 6.43 | 17.7% / 15.4% |
| 55+ | £31,800 | £70,300 | 2.2 | 9.94 / 10.69 | 5.79 / 6.60 | 8.3% / 7.0% |

- **Channels.** The gap is on both channels. The median for 18–24 is £46,100 on FRS-channel rows
  and £52,300 on SPI-channel rows; for 35–44 it is £32,000 and £28,300.
- **Earnings.** Mean log earnings are 0.36 to 0.47 higher on the spine in every age group. Spine
  incomes are 2024-25 pounds and the donor's are 2020-22 pounds.
- **Private renting.** Among adults under 35 the private-renting share is 9 to 11 points higher
  on the spine.
- **Not introduced here.** The twin (Part E) shows every `was_wealth` column byte-identical to
  main. The register's standing wealth adjudication (`spine_swap_signed_differences.json`,
  `was-wealth-qrf-incidence`) already records the spine's savings mean at 1.97 times the donor
  per household.
- **Uprating.** No `was_wealth` operation uprates the donor's 2020-22 values, its wealth or its
  income predictors. Spine households, whose incomes are 2024-25 pounds, therefore condition on
  richer donors. How much of the gap that explains is a question for the follow-up.

Four options, measured or implied on the same inputs:

- **(a) Keep the approved design.** On the spine:
  - Ownership is 1.25% (660,000 weighted holders on 52.7 million adults at stage time) and twice
    the donor's share in the youngest group (18–24 on the spine, 16–24 in the donor).
  - Owner balances run higher: median £6,500, p90 £15,000.
  - The cap removes 0.5% of the LISA mass.
- **(b) Drop the four financial predictors** (gross financial wealth, savings, cash ISA,
  stocks-and-shares ISA) from both models. This is arm B, built and measured but not committed.
  - Ownership falls to 1.06% (560,000), and household net income takes over part of the wealth
    signal (its coefficient rises from +0.03 to +0.74).
  - The age profile is closer to the donor's but still above it at every age under 45, where the
    spine's earnings and private-renting share are also above the donor's.
  - Balances fall to the donor's shape (median £5,000, p90 £10,000).
  - Coherence breaks. Ownership no longer follows the household's financial wealth, so the cap
    binds on 76 households, removes 12.7% of the weighted LISA mass, and clears fewer than 10
    owners whose household has no financial wealth.
- **(c) Keep the predictors and recalibrate the intercept by age group** so the spine's expected
  share in each age group under 45 equals the donor's. This is not built. It is a declared,
  receipted step after the fit.
  - It implies 0.82% of the spine's adults (430,000). The spine's adults are older than the
    donor's: 45% are 55 or over, against 40%.
  - Within each age group, the ranking by financial wealth, and with it the coherence, stays as
    in (a).
  - Balances stay as in (a), because the balance forest still conditions on the financial draws.
  - It anchors the 2024-25 spine to the 2020-22 donor rate, and holdings have grown since:
    HMRC reports LISA subscriptions 20.1% (£472 million) higher in 2024 to 2025
    ([annual savings statistics, September 2026 commentary](https://www.gov.uk/government/statistics/annual-savings-statistics-2026/commentary-for-annual-savings-statistics-september-2026)).
- **(d) Uprate the donor's income predictors to the spine year** (review round 1, item 3). This
  is arm D, built and measured but not committed. The house `uprate_donor_columns` step moves
  the WAS person donor's `employment_income` and `household_net_income` from 2021 to 2024 by the
  engine's OBR average-earnings index (factor 1.185), as the LCFS and ETB stages do.
  - The model's expected share falls from 1.36% to 1.32%, and ownership from 1.25% to 1.23%
    (650,000 holders). The youngest group falls from 4.14% to 4.07%, 25–34 is 2.96% and 35–44 is
    2.58%.
  - The coefficients are unchanged, since the standardisation absorbs the shift; only the
    spine's incomes read lower against the donor's.
  - Balances stay as in (a): median £6,500, p90 £15,000. The cap binds on 10 households and
    removes 0.6% of the LISA mass.
  - So the nominal income basis explains about 0.02 of the 0.36 points between the spine and
    the donor. The financial-wealth level carries the rest.

Against HMRC's administrative count (review round 1, item 2). HMRC's Individual Savings Account
statistics (September 2026 release, Table 9.4, adult ISAs) report the number of Lifetime ISA
accounts subscribed to in each tax year, across the UK. A person can pay into only one LISA a
year, so each subscribed account is a distinct subscriber, and every subscriber is a holder.

| Tax year | LISA accounts subscribed to | Amount subscribed |
| --- | --- | --- |
| 2020 to 2021 | 553,000 | £1,482 million |
| 2021 to 2022 | 662,000 | £1,700 million |
| 2023 to 2024 | 964,000 | £2,346 million |
| 2024 to 2025 (provisional) | 1,136,000 | £2,818 million |

- **The donor.** The WAS donor implies about 447,000 holders in Great Britain over April 2020 to
  March 2022 (467,000 before the credibility rule). That is below the subscribers alone in either
  fieldwork year: 81% of 2020-21's and 68% of 2021-22's. Northern Ireland is about 3% of UK
  adults, so the survey undercounts LISA holders against the administrative record, by a margin
  the non-subscribing holders only widen.
- **The spine year.** For 2024-25 the subscribers alone number 1.14 million.
  - The approved design's 660,000 holders are 58% of that.
  - Arm B's 560,000 are 49%.
  - Option (c)'s 430,000 are 38%.
  - Arm D's 650,000 are 57%.
- **The stock.** The government reports over 1.3 million LISA accounts open in 2023-24 (its
  response to the Treasury Committee, 11 September 2025, recommendation 2). A person can hold
  several accounts.
- **Market value.** Table 9.6 publishes no separate LISA market value: LISAs sit inside the cash
  and stocks-and-shares totals. So there is no administrative check on the £4.45 billion.
- **What follows for the options.** Every option leaves the spine below the administrative level,
  and option (c), which re-anchors to the donor's rate, moves furthest from it. A level fixed by
  HMRC's counts would need those counts vendored through Chronicle, as every calibration fact is.
  That is a question for the ruling, not something this PR does.

Whichever option she picks, the `was_wealth` level is a follow-up in its own right: it moves every
reader of the financial draws, not only this stage.

## Part G. The annual-contribution gap

WAS measures holdings and values at interview, and has no LISA-specific amount paid in during a
tax year. A balance accumulates payments, bonuses, returns and withdrawals, so it does not
identify one year's payments. Holding a LISA does not show that the person paid in that year.

The stage creates no contribution column and derives no contribution or bonus from a balance. The
issue's references for a future extension are carried in the topic doc as references only:

- Moneybox and CBI Economics on deposits by income band;
- Moneybox's written evidence on the share of customers at the annual limit;
- HMRC's annual savings statistics and the separate LISA administrative source.
