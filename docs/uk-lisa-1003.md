# Lifetime ISA holdings on the UK spine (microcosm#1003)

The `was_lisa` spine stage gives every adult of the UK spine a Lifetime ISA
(LISA) holding imputed from the Wealth and Assets Survey (WAS) round 8. It adds
three exported cells:

| Cell | Meaning |
| --- | --- |
| `person.has_lifetime_isa` | The person holds one or more LISAs. |
| `person.lifetime_isa_balance` | The current value across the person's LISAs, in pounds (zero for non-holders). |
| `household.household_lifetime_isa_balance` | The sum of its persons' balances. |

The fields describe holdings: who has a LISA and how much is in it. They do not
describe annual contributions (see [The annual-contribution gap](#the-annual-contribution-gap)).
Measured receipts are in
[`experiments/1003-uk-lisa-receipts.md`](../experiments/1003-uk-lisa-receipts.md).

## Source

WAS round 8 (ONS, UK Data Service SN 7215, DOI 10.5255/UKDA-SN-7215-20)
interviewed households in Great Britain between April 2020 and March 2022. Each
adult is asked which ISA types they hold (`FISA`, code 3 is a Lifetime ISA) and
the current value of their LISAs (`FLISAV`), or, when they do not know it, a
band on a LISA-specific card (`FLISAB`, eight bands from under £1,000 to
£10,000 or more). ONS imputes item non-response and imputes a value inside the
reported band, and releases the completed value as `DVFLISAvR8`
(`= FLISAVR8_i`). The household file's `DVFLISAVR8_aggr` is the sum of those
person values.

The stage reads both licensed tabs, pinned by size and sha256 in
`uk/spec/sources.yaml`:

- the round-8 household tab, the `was_wealth` donor with the same pins;
- the round-8 person tab `was_round_8_person_eul_may_2025_230525.tab`, of whose
  4,079 columns it reads 14, joined to its household on the anonymised case
  number `CASER8`.

The stage refuses a person tab whose released values do not sum to the
household aggregate on every household, so the variable the model is trained
on is provably the one ONS aggregates.

## Missing answers, banded answers and ONS imputation

The released value has no missing or negative entries on the pinned tab, and
the ownership flag (`fisa_binary3r8_i`) agrees with a positive value on every
record. The stage still refuses a missing or negative value rather than reading
it as a zero balance. Each donor adult keeps a response class, recorded in the
stage evidence and never used as a predictor:

- observed non-holder, or non-holder whose ownership ONS imputed;
- holder with an exact reported value;
- holder with a banded value (ONS imputed the value inside the band; the band
  itself was reported or imputed);
- holder whose value ONS imputed without a band, or whose ownership ONS
  imputed;
- holder recoded by the credibility rule below.

On the pinned tab, most released holders reported an exact value. For every
holder who reported a band or had one imputed, ONS's imputed value lies inside
the band. The model trains on the ONS-completed values, as the `was_wealth`
stage trains on ONS-completed household aggregates.

## Credibility rule

Some released records are impossible under the product rules, and the stage
declares a rule for them (`clean_was_lisa_donor.credibility_rule`):

- **Age.** LISAs exist from 6 April 2017 and can be opened only by adults under
  40 ([The Individual Savings Account (Amendment No. 2) Regulations 2017,
  explanatory memorandum](https://www.legislation.gov.uk/uksi/2017/466/pdfs/uksiem_20170466_en.pdf);
  [gov.uk: who can open a Lifetime ISA](https://www.gov.uk/lifetime-isa/who-can-open-a-lifetime-isa)),
  so no holder was older than 44 by the end of round-8 fieldwork. Holders
  recorded in an age band starting at 45 or above are read as a misreported ISA
  type and recoded to non-holders. Their value stays inside the household's
  gross financial wealth, where WAS already counts it.
- **Balance.** Payments are limited to £4,000 a year with a 25% bonus
  ([gov.uk: Lifetime ISA](https://www.gov.uk/lifetime-isa)). Five tax years to
  March 2022 allow £25,000 of payments and bonus. The one-off 2017-18 transfer
  of Help to Buy ISA savings held at 5 April 2017 did not count towards the
  limit and attracted the bonus
  ([HM Treasury, Lifetime ISA technical note, September 2016, para 1.25](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/553333/Lifetime_ISA_technical_note_September_2016_update.pdf));
  a Help to Buy ISA accepted £1,200 in its first month and £200 a month from
  1 December 2015 ([UK Finance](https://www.ukfinance.org.uk/help-buy-isa)),
  so at most about £4,400 could be transferred. About £30,500 was therefore
  attainable before investment returns. Holders above the declared ceiling of
  £40,000 stay holders but leave the balance fit.

On the pinned tab the rule recodes 21 holders aged 45 or over and keeps a
handful of holders above the ceiling out of the balance fit. Together these
records are about 5% of weighted holders but about a third of the weighted LISA
mass, so the rule matters for totals and not for the typical balance. The
weighted median holder balance is £6,000 with or without it, and ONS
publishes a £6,000 median for households holding LISAs in April 2020 to March
2022 ([Financial wealth: wealth in Great Britain](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/datasets/financialwealthwealthingreatbritain),
Table 5.1). No age limit is imposed on recipients: the model decides, and it
gives adults aged 45 or over a very small probability.

## How the stage imputes

`was_lisa` runs right after `was_wealth`, so every household of the spine,
including the SPI support copies and income-band donors, has its final incomes
and its own WAS financial draws before the LISA draw.

1. **Who holds.** A weighted logistic regression with a ridge penalty on the
   slopes, fitted on the donor's non-dependent adults with the WAS household
   weight each person carries. Predictors: five age groups (the public file has
   five-year age bands only), the logarithm of the person's earnings, household
   net income, gross financial wealth, cash ISA, stocks-and-shares ISA and
   savings, private renting, sex and the number of children. The intercept is
   unpenalised, so the model's weighted mean probability on the donor equals the
   donor's weighted ownership share. A recipient aged 18 or over holds when a
   uniform keyed on the person id falls below their probability, so adding or
   reordering rows never moves a draw.
2. **How much.** The house regime-gated quantile regression forest, fitted on
   the credible holders only, drawn at a second person-keyed quantile. A
   balance is positive exactly when the person holds.
3. **Coherence.** A household whose LISA total exceeds its `gross_financial_wealth`
   draw has its persons' balances scaled down pro rata, and ownership is
   re-derived from the capped balance. The number of households, persons and
   pounds affected is receipted.

The ownership part is a logistic because the house regime gate (a
gradient-boosting classifier) is miscalibrated for an outcome held by under one
per cent of adults. Held out by household on the donor, the gate reproduced
0.58% ownership against the donor's 0.89% and understated each age group under
45 by more than a third. The logistic reproduced 0.87% and every age and tenure
margin within 0.07 points (María's ruling of 2026-09-29; the
comparison is in the receipts, Part B).

Household predictors use the `was_wealth` definitions. Tenure is the
private-renter flag on both sides (WAS `DVPriRntR8 == 1`; the spine's
`tenure_type` of `RENT_PRIVATELY`), not the engine's `is_renting`.

## Reconciliation with the existing wealth columns

WAS counts every ISA, LISAs included, inside gross financial wealth
(`HFINWR8_SUM`, the spine's `gross_financial_wealth`) and therefore inside net
financial wealth. LISAs are not inside `savings` (savings accounts),
`cash_isa`, `stocks_and_shares_isa` or `corporate_wealth`. The LISA columns are
a detail of gross financial wealth: they must never be added to it, to
`savings`, or to the ISA columns. The engine's `total_wealth` does not read
`gross_financial_wealth`, and no engine variable reads the LISA columns today;
the engine's means-test capital reads `savings` and `corporate_wealth`, so
neither cash ISAs nor LISAs reach it. Innovative-finance ISAs and ISAs of
unknown type remain inside gross financial wealth only.

## Coverage and vintage

- Balances are April 2020 to March 2022 pounds from a Great Britain donor and
  are not uprated to the spine year. LISA holdings have grown since; a
  published LISA stock or count series would be needed to uprate them.
- WAS undercounts LISA holders against HMRC's record. The donor implies about
  447,000 holders in Great Britain over April 2020 to March 2022. HMRC counts
  553,000 LISA accounts subscribed to in 2020-21 and 662,000 in 2021-22 across
  the UK, and every subscriber is a holder (a person can pay into one LISA a
  year). For 2024-25 HMRC's provisional count is 1.14 million, against the
  candidate spine's 660,000 holders
  ([Individual Savings Account statistics, September 2026](https://www.gov.uk/government/statistics/annual-savings-statistics-2026),
  Table 9.4; receipts, Part F).
- Northern Ireland adults are predicted from the Great Britain model.
- Conditional ownership odds from 2020-22 are applied to the 2024-25 spine. A
  holder in 2024-25 can be up to about 47; the donor, interviewed earlier,
  cannot show holders that old, so those ages carry only the model's small
  probability.
- People in one household are drawn independently given their predictors. In
  the donor, 24% of weighted holder households have two or more holders after
  the credibility rule; on the candidate spine, 4% do (receipts, Part C).

## On the candidate spine

Measured on the licensed candidate spine (receipts, Parts C to F):

- 1.25% of weighted adults hold a LISA, against the donor's 0.89%. Every age
  group under 45 is above the donor's share, the youngest most: 4.1% of the
  spine's adults aged 18 to 24, against 2.1% of the donor's aged 16 to 24. The
  donor's first group includes 16- and 17-year-olds, who cannot hold a LISA,
  because the public file bands age 15 to 19.
- The weighted median owner balance is £6,500, against the donor's £6,000, and
  the p90 is £15,000 against £10,000.
- The excess comes from the inputs, not the fit. The spine's WAS financial
  draws, from the `was_wealth` stage and unchanged here, have two to four times
  the donor's median gross financial wealth at every age. Financial wealth is
  the ownership model's strongest predictor.
- Receipts Part F measures the alternatives: dropping the financial predictors,
  or recalibrating the intercept by age group.
- The coherence cap binds on fewer than 10 households and removes 0.5% of the
  weighted LISA mass.

## What the fields support

The fields support analysis of who holds LISAs and how much they hold, for
example the distribution of LISA wealth, or a means-test or wealth measure that
counts LISA balances once the engine reads them. They do not support reliable
scoring of reforms to the annual payment limit or the bonus rate, which depend
on contributions; reforms to withdrawals or to first-home purchases, which need
withdrawal and purchase flows and eligibility; or a cash-versus-stocks-and-
shares split, which WAS does not record.

## The annual-contribution gap

WAS measures holdings and values at interview. It has no LISA-specific amount
paid in during a tax year. A balance accumulates payments, bonuses, returns and
withdrawals, so it does not identify the payments of a particular year, and
holding a LISA does not show whether the person paid in that year. The stage
therefore creates no contribution column and derives no contribution or bonus
from a balance.

References for a future contribution extension, carried from the issue as
references only (not requests to obtain data or to contact providers or HMRC):

- Moneybox and CBI Economics,
  [The value of Lifetime ISAs](https://www.moneyboxapp.com/wp-content/uploads/2025/10/CBI-Economics-The-Value-of-Lifetime-ISAs.pdf#page=17),
  pp. 17-18: LISA deposit totals and means by income band and employment status
  for 2024-25, from proprietary provider data with extrapolation.
- Moneybox,
  [written evidence to Parliament, section 9](https://committees.parliament.uk/writtenevidence/136651/html/):
  the share of its customers who reached the annual limit in 2024.
- HMRC, [Annual savings statistics 2026](https://www.gov.uk/government/statistics/annual-savings-statistics-2026),
  ISA subscription totals and counts (Table 9.4), with the
  [methodology](https://www.gov.uk/government/statistics/annual-savings-statistics-2026/annual-savings-statistics-background-and-methodology)
  describing the separate LISA administrative source.

A contribution extension would build on the ownership and balance fields
added here.
