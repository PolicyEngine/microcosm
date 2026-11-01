# ISA subscriptions per adult (microcosm#1415)

The `isa_subscriptions` stage gives every adult of the UK spine what they paid
into ISAs in the year, by type, and the per-person ISA balances behind them. It
runs last, on final incomes, and nothing in the spine reads its outputs.

| Column | Meaning |
|---|---|
| `person.cash_isa_balance` | The person's share of the household's cash ISA balance (`was_wealth`), in pounds. |
| `person.stocks_and_shares_isa_balance` | The person's share of the household's stocks-and-shares ISA balance. |
| `person.cash_isa_subscriptions` | Paid into cash ISAs in 2024-25. |
| `person.stocks_and_shares_isa_subscriptions` | Paid into stocks-and-shares ISAs in 2024-25. |
| `person.lifetime_isa_payments` | Paid into the person's Lifetime ISA in 2024-25 (at most £4,000). |

The `was_lisa` stage also anchors Lifetime ISA holders to HMRC's payers (below,
and [the LISA notes](uk-lisa-1003.md)).

The pinned policyengine-uk 2.164.2 reads all five:

- `lifetime_isa_payments`, whose 25% bonus is `lifetime_isa_bonus`;
- the subscriptions, against the limits of policyengine-uk#2712
  (`isa_subscriptions_above_limit` and `cash_isa_subscriptions_above_limit` are
  what a limit leaves out);
- the person balances, which its household `cash_isa` and
  `stocks_and_shares_isa` sum.

The release does not store those household totals. A stored total would
override the sum, so the means tests, which count the household `cash_isa`,
read the sum of the person balances.

## Sources

- HMRC, [Annual savings statistics 2026](https://www.gov.uk/government/statistics/annual-savings-statistics-2026)
  (September 2026), through the pinned Chronicle feed (`53815b2`, which carries
  PolicyEngine/chronicle#392's ISA statistics):
  - Table 9.7: individuals subscribing in 2023-24 by total income and size of
    subscription, for each type paid into (cash only, stocks and shares only,
    both). The tables head their columns "(thousands)" but count people.
  - Table 9.8 by age and sex, and Table 9.9 by region.
  - Table 9.4: accounts and amounts subscribed by type to 2024-25 (the latest
    year provisional), including Lifetime ISAs.
  - Table 9.6: market values, and Tables 9.10 to 9.12: holders by market value,
    both checks only.
- HMRC, the cash ISA limit reduction tax information and impact note: 22% of
  cash subscribers under 65 paid in more than £12,000 in 2022-23 (a check).
- HMRC, Lifetime ISA statistics, Table 1a: withdrawals (a check).
- ONS Wealth and Assets Survey round 8 (UKDS SN 7215), the person tab's ISA
  values, for the split of household balances across adults.

How the tables relate, on HMRC's own figures:

- Table 9.7's type blocks are the types paid into: cash only plus both
  reproduces Table 9.4's 2023-24 cash accounts, and stocks and shares only plus
  both its stocks-and-shares accounts. Lifetime ISAs, Innovative Finance ISAs
  and Junior ISAs are outside Table 9.7; Help to Buy ISAs are inside cash.
- Table 9.7's bands imply more money than Table 9.4 records. Valued at their
  floors they give at least £114.3bn (cash only alone £78.1bn), against Table
  9.4's £100.6bn of cash plus stocks and shares, so the two cannot both bind.
- From 6 April 2024 (SI 2024/350) a person may pay into more than one ISA of a
  type in a year, but still only one Lifetime ISA, so Table 9.4's 2024-25
  accounts are an upper bound on people.

## Method

1. **Balances per adult.** Each household's cash and stocks-and-shares ISA
   totals are split across its adults by the holding pattern of a WAS round-8
   donor household with the same number of adults (up to four), drawn at
   household-keyed uniforms and conditioned so every subscriber holds the type
   they pay into. Person values sum exactly to the household totals, which are
   unchanged.
2. **Who subscribed in 2023-24.** Adults aged 18 or over in households holding
   the type are eligible: cash only needs a cash balance, stocks and shares only
   a stocks-and-shares balance, and both needs both. For each class, a small
   iterative proportional fit rakes the eligible weighted mass to Table 9.7's
   income bands (on the CGT stage's income proxy), Table 9.8's age and sex and
   Table 9.9's regions (overseas dropped, each margin on Table 9.7's total).
   Keyed walks select subscribers cell by cell, both first, then stocks and
   shares only, then cash only. A cell without enough eligible mass is reported
   (`shortfall_weight_unmet`); no holder is invented.
3. **How much.** Within each income band and class, amounts follow Table 9.7's
   subscription bands: uniform inside a band, exactly £20,000 at the top. The
   rank blends the person's balance rank with a keyed uniform.
4. **The both class.** Each person's amount is split by their balance shares,
   shifted on the logit scale so the class's stocks-and-shares total equals
   Table 9.4's 2023-24 stocks-and-shares amount less the stocks-and-shares-only
   payments (about 27% stocks and shares). A shift the floor stops is named.
5. **Growth to 2024-25.** Each class's people grow by Table 9.4's account
   growth (cash accounts for cash only, stocks-and-shares accounts for stocks and
   shares only, both together for both), added from eligible holders not yet
   subscribing. Amounts then scale, one factor per type, to Table 9.4's 2024-25
   amounts (£95.6bn of cash and £37.2bn of stocks and shares). Subscribers
   already at the £20,000 overall limit stay there. A person whose combined
   amount lies between £18,000 (Table 9.7's last band floor below the limit) and
   the limit moves on a line from their scaled £18,000 up to it, so the range just
   below the limit stays filled; the rest are capped at it.
6. **Lifetime ISA payments.** Every holder aged 18 to 47 with room under the
   overall limit pays in; a holder with none pays nothing. 18 to 47 are the ages
   that can hold one in 2024-25, since accounts opened at 18 to 39 from April
   2017. Amounts lie in (0, £4,000], drawn by a truncated-exponential tilt so
   the total equals Table 9.4's 2024-25 Lifetime ISA amount (£2,818m from
   1,136,000 accounts), so the baseline leaves nothing above the overall limit.
   The rank blends the balance and income ranks with a keyed uniform.

## Decisions (28 October 2026)

- **Measure, bind none.** No ISA fact enters the weight solve: binding the cells
  the stage draws to would be circular. A reviewed fence declares every vendored
  ISA fact measured only; the checks below are reported and never block a build.
- **Table 9.7's people, the stocks-and-shares residual.** Table 9.7 keeps its
  people in every band; the both class's split takes its stocks-and-shares
  total from Table 9.4's residual.
- **People by accounts.** Growth uses Table 9.4's account growth, the upper
  bound on people; the flat-people bound is reported beside it.
- **Lifetime ISAs here.** Holders are re-anchored to HMRC's payers and payments
  drawn in this change. The evidence is the next pair build.

## Decisions (29 October 2026, on the review of microcosm#1421)

- **Cash grows to Table 9.4's level, holding and tapering to the limit.**
  Growing by Table 9.4's ratio carried Table 9.7's cash excess into 2024-25 and,
  with the subscribers at £20,000 unable to grow, scaled everyone else by about
  2.05. On populations built from HMRC's own tables that gave £127bn of cash
  against Table 9.4's £95.6bn, 51% of cash subscribers above £12,000 and 40% of
  all subscribers at the limit. Scaling to Table 9.4's 2024-25 amounts holds the
  subscribers at the limit, since a plain factor below one would have moved every
  one of them to about £19,100. One factor of about 0.89 below the limit still
  left nobody between about £17,750 and £20,000, so amounts above £18,000 now
  taper up to the limit. That gives a cash factor of about 0.878, 36.9% of cash
  subscribers above £12,000, 28.8% at the limit (28.8% in 2023-24) and amounts
  up to about £19,950 below it.
- **The Lifetime ISA anchor's reach is counted on the pair.** Three later stages
  read the whole frame (`uc_reporter_redraw`, `uc_take_up` and
  `pension_credit_take_up`), and a Lifetime ISA counts as Universal Credit and
  Pension Credit capital, so the added holders can move their draws. The pair
  build's national counts, at its final weights, the benefit units whose
  entitlement changes without the added holders' balances, and the units that
  move in or out of the take-up pools. Where the financial-wealth cap binds, an
  added holder also scales the household's existing balances down pro rata.

## What the tables leave open

- **The cash excess.** Table 9.7 at band midpoints implies about £92.5bn of
  cash (£84.5bn cash only and £8.1bn from the both class) against Table 9.4's
  £69.5bn, about £23bn above it. The stage keeps Table 9.7's people and bands,
  so its 2023-24 cash total sits above Table 9.4's, and the receipt reports by
  how much; the 2024-25 total meets Table 9.4.
- **The impact note's year.** On HMRC's tables 36.9% of 2024-25 cash
  subscribers pay in more than £12,000, against the impact note's 22% of under-65
  cash subscribers in 2022-23. Age explains little of that: Table 9.8 puts about a
  quarter of cash-only subscribers at 66 or over, who would need about 80% above
  £12,000 to bring the under-65s to 22%. Most of it is the year: Table 9.4's
  cash subscriptions were £41.6bn in 2022-23, £69.5bn in 2023-24 and £95.6bn in
  2024-25, so the 22% does not validate 2024-25 either way. Amounts follow Table
  9.7's bands within income band and class, not age.
- **The both class's residual share** depends on where the stocks-and-shares-only
  payments sit inside their bands: about 27% stocks and shares at band
  midpoints, 50% at band floors and 7% at band tops.
- **One year of growth.** The individual tables are 2023-24, grown one year
  across the April 2024 rule change.
- **Balances.** WAS balances are April 2020 to March 2022 pounds from a Great
  Britain donor, not uprated; Northern Ireland adults take the Great Britain
  patterns.

## Gates

`uk_stage_isa_subscriptions_invariants` holds the rules any population keeps:
the limits, ages, holding what is paid into, finite and non-negative amounts and
person balances that sum to the household totals. It is no population fact
check, so the synthetic smoke build blocks on it as every licensed build does.
`uk_stage_isa_subscriptions` holds the 2024-25 totals, the both class's residual
and the Lifetime ISA amount to Table 9.4. A synthetic population cannot reach
HMRC's national totals, so it is a population fact check.

## Checks (reported on the pair build's national file)

- Subscription cells before and after calibration, by class and band, and above
  £12,000.
- Holders against Tables 9.10 to 9.12, and market values against Table 9.6.
- The impact note's 22% of under-65 cash subscribers above £12,000.
- Lifetime ISA payers and amounts against Table 9.4, and `lifetime_isa_bonus`
  against 25% of payments; the 2025-26 house-purchase withdrawals (99,750) as a
  loose check on holders.
- The flat-people growth bound.
- The Lifetime ISA anchor's reach: benefit units whose Universal Credit or
  Pension Credit entitlement changes without the added holders' balances, at
  the final weights, and the units that move in or out of the take-up pools.

## What the fields support

The fields support the subscriptions a limit leaves out: the overall £20,000
limit and a separate cash limit (the £12,000 under-65 setting is a reform in
policyengine-uk), through `isa_subscriptions_above_limit` and
`cash_isa_subscriptions_above_limit`. policyengine-uk 2.164.2 reads neither in
any tax or spending variable, so a limit reform changes those amounts but costs
nothing until the engine models where the excess goes. The subscriptions are
flows the engine does not uprate, so a later year reads 2024-25 amounts. In
particular the cash subscriptions above £12,000 in 2027-28, which a £12,000 cash
limit from April 2027 would leave out, are 2024-25 levels here, not a 2027-28
base: the dashboard's cash ISA card should not read them as one (Vahid's review
of microcosm#1421). policyengine-uk's own docstring notes a producer could grow
them and cap each person at the limit in force; this stage does not, since
HMRC's tables stop at 2024-25. The
Lifetime ISA bonus does reach government spending, so its rate and payment
limit score. The fields also support analyses of who subscribes how much. They
do not support withdrawals or first-home purchase rules, Innovative Finance or
Junior ISAs, or behavioural responses to a limit.
