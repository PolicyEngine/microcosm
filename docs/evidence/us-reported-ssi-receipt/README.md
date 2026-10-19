# Reported SSI receipt on the pinned US default: measured effect

Run 2026-10-18 with `measure_receives_ssi.py` (in this folder).
`summary_aggregates.json` holds the aggregates; it records no household-level
values.

## Setup

- **Data:** `populace_us_2024.h5`, sha256 `6496cc43…aee84`. This is
  PolicyEngine-US's `DEFAULT_DATASET` (`populace-us-2024-spm-20260909`; the
  same bytes as `-20260915`).
- **Model:** PolicyEngine-US `origin/main` at `fab18b7d6b` and
  PolicyEngine/policyengine-us#10145 at `f450d4ab00`, with policyengine-core
  3.33.2. The #10145 head branches from an older main, and the two trees
  differ in 59 files, including the New Mexico and Texas supplements and
  other state taxes. The receipt deltas reported here are identical on both
  trees; among the reported levels, only Colorado's differs. The tables use
  the #10145 head.
- **Variants:** each adds person columns to an in-memory copy of the dataset:

  | Variant | Change |
  |---|---|
  | `base` | None; `receives_ssi` takes the engine default `False`. |
  | `receipt` | `receives_ssi` from positive `SSI_VAL`, excluding on-behalf-of-child-only payees (`RESNSSI1` 3/4 with `RESNSSI2` 0/3/4). The pinned default has no row whose only payee code is in `RESNSSI2`, so this equals `cps_carried.reported_ssi_receipt` on it. |
  | `receipt_asec_only` | The same, on ASEC rows only; PUF tax-detail clones stay `False`. |
  | `receipt_any` | `SSI_VAL > 0` without the payee exclusion. |
  | `criteria_anchor` | Diagnostic: `meets_ssi_disability_criteria = True` for under-65 own-reason reporters on the PUF channel; no receipt. |
  | `receipt_criteria_anchor` | `receipt` plus `criteria_anchor`. |
  | `receipt_assets0` | Diagnostic upper bound: `receipt`, with no liquid assets for anyone in a reporter's household. |

- **Households simulated:** every household with a reporter (any row with
  `SSI_VAL > 0`) plus every Colorado household (2,541 households and 7,764
  rows).

`receives_ssi` is a person input, and the static model has no interaction
between households, so households without a reporter cannot change. The
weighted deltas below are therefore the national deltas. The run checks this
on Colorado's non-reporter households, whose deltas are exactly zero on every
aggregate except the `medicaid` dollar value. That value spreads a fixed
total over enrollees, so it is not reported. Colorado totals are complete; the
main-branch Colorado supplement reproduces the US hub's full-dataset $233.22M.

The variants change inputs only; nothing is recalibrated. These are
fixed-weight sensitivities on today's default, not the effects of a rebuilt
default. In a rebuild the SSI take-up priors are re-solved against SSA
counts, so the federal SSI changes in the two diagnostic variants overstate
a rebuilt effect. The default has no ACS rows, so the ACS mapping does not
enter these numbers.

## Survey reporters versus modeled SSI (2026, `base`)

| | Weighted people | Modeled SSI |
|---|---:|---:|
| Own-reason SSI reporters | 5.15M | 2.79M paid; $28.4B against $49.1B reported |
| On-behalf-of-child payees (excluded) | 0.45M | 0.16M paid $1.5B of their own |

In January 2026, 2.35M own-reason reporters receive no modeled payment:

| First failing test | Weighted people |
|---|---:|
| Not aged, blind or disabled (under 65 with no `meets_ssi_disability_criteria`) | 1.03M |
| Countable income at or above the benefit | 0.78M |
| Resource test | 0.45M (median countable resources $9,087) |
| Not aged, blind or disabled (other) | 0.09M |

By channel, ASEC rows anchor `meets_ssi_disability_criteria` for every
under-65 reporter (2.48M of 2.48M). PUF tax-detail clones do not (0.82M of
1.95M), although they carry the reporter's `SSI_VAL` and the take-up anchor.
Among reporters who fail the income test, ASEC rows have a median countable
income of $1,750 a month (464,000 people, 65% with Social Security). PUF
clones have $5,265 a month (316,000 people), from PUF-imputed incomes
(`income_fail_by_channel.py`).

## Effect of adding the column (2026, change from `base`)

| Measure | `receipt` | `receipt_asec_only` | `receipt_any` | `criteria_anchor` | `receipt_criteria_anchor` | `receipt_assets0` |
|---|---:|---:|---:|---:|---:|---:|
| Federal `ssi` | $0 | $0 | $0 | +$6,240M | +$6,240M | +$1,927M |
| `household_net_income` | +$339M | +$141M | +$454M | +$5,444M | +$5,729M | +$2,066M |
| `snap` | +$261M | +$97M | +$310M | −$607M | −$397M | +$48M |
| `early_head_start` | +$1,156M | +$433M | +$1,329M | +$268M | +$1,156M | +$1,156M |
| `head_start` | $0 | $0 | +$93M | $0 | $0 | $0 |
| `free_school_meals` | +$63M | +$43M | +$121M | $0 | +$63M | +$65M |
| `aca_ptc` | −$863M | −$220M | −$923M | −$201M | −$786M | −$863M |
| `medicaid_enrolled` (people) | +1,408k | +369k | +1,576k | +350k | +1,393k | +1,443k |
| `is_ssi_recipient_for_medicaid` (people) | +2,228k | +761k | +2,510k | +566k | +2,205k | +2,316k |
| `mi_ssp` | +$6.6M | $0 | +$9.5M | $0 | +$5.0M | +$3.3M |
| `wa_ssp` | +$7.6M | +$0.3M | +$7.6M | +$10.6M | +$18.2M | +$13.2M |
| `wy_ssp` | +$0.3M | +$0.3M | +$0.3M | $0 | +$0.3M | +$0.3M |
| `co_state_supplement` | $0 | $0 | $0 | $0 | $0 | $0 |

`ca_state_supplement`, `ma_state_supplement`, `nj_ssp`, `pa_ssp` and `me_ssp`
move only in the diagnostic variants. The 2024 deltas have the same signs.
`summary_aggregates.json` has both years for both commits.

## Colorado

Main pays Colorado's AND-CS supplement ($233.22M in 2026) to 11 records. None
of them belongs to a household with a reported SSI amount: they are
modeled-eligible people with `takes_up_ssi_if_eligible = False`. #10145 pays
$0 with or without `receives_ssi`. Every Colorado reporter with positive
weight who passes AND-CS's disability and age test already has modeled SSI
plus countable income equal to the $994 grant standard, so the supplement is
$0.
