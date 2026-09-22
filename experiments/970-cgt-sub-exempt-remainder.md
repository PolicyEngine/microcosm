# UK CGT sub-exempt gainers: remainder amounts, incidence anchor and projection fence (microcosm#970)

*2026-09-22. Branch `uk-970-cgt-sub-exempt`. Measurement receipts to be filled in from the licensed rebuild;
the numbers quoted for the v20 candidate come from the `uk-equalising-cgt` dashboard run of 2026-09-21
(PolicyEngine/uk-equalising-cgt PR #3) and the HMRC Capital Gains Tax statistics 2026 release.*

## Question

The staged v20 candidate (`staged/uk-spine-assessment-v20-calibration`) carried 11.66 million weighted people
with capital gains at or below the £3,000 annual exempt amount, 10.82 million of them at exactly £3,000.00: the
amounts stage capped every gainer beyond HMRC's published taxpayer mass at the exempt amount, and the equal-mass
incidence clone left half of all household mass on gainer households. policyengine-uk freezes the exempt amount
and uprates gains by per-capita GDP growth, so every one of them became a CGT taxpayer in the first projected
year: 11.54 million taxpayers in 2026-27 against 597 thousand on the incumbent Enhanced FRS and 551 thousand
in HMRC Table 1 for 2024-25. Issue #970 fixes this in three parts, and this note records what each changes.

## What changed

1. **Amounts** (`hmrc_cgt_gains_spine`): the remainder takes amounts from the Advani-Summers Table A1
   within-band gains distribution restricted to the quantiles between each band's zero crossing and its
   crossing of the exempt amount, rank-preserving within the band and deterministic (no seed consumed).
   The table is used as published (2017-18 nominal), with no uprating; for the first income band the
   crossings sit at quantiles 0.1805 and 0.2354.
2. **Incidence** (`cgt_incidence_anchor`, new 31st spine stage, weights only): the sub-exempt and
   loss-making clone households are trimmed to the reporter composition implied by the redrawn liable mass
   and the Advani-Summers crossings (sub-exempt = liable × (q_AEA − q_0) / (1 − q_AEA), losses =
   liable × q_0 / (1 − q_AEA)), with a factor rising in the band's incidence and capped at one; every unit
   of mass a clone loses goes to its paired original, so pair mass, household mass and every non-CGT
   aggregate are conserved. Liable clones and band donors are untouched. Gate:
   `uk_stage_cgt_incidence_anchor_composition`.
3. **Fence** (`uk_cgt_projection_entrants`, calibration seam): the weighted sub-exempt gainers whose
   uprated gains cross the frozen exempt amount are counted year by year to 2030; the largest count must not
   exceed the vendored HMRC Table 2.1a taxpayers in the £3,000–£5,999 band (73,000 in 2024-25).

## Expected end state on a rebuilt candidate

Liable mass unchanged (about 551.6 thousand), sub-exempt remainder about 43 thousand, loss-makers about
136 thousand, entrants by uprating in the tens of thousands per year, dashboard 2026-27 taxpayers in the
555–600 thousand range.

## Measurements

| receipt | v20 (before) | rebuilt candidate |
| --- | --- | --- |
| `hmrc_cgt_gains_spine.allocation.remainder` persons / mass | cap at £3,000; 10.82m weighted | _pending_ |
| `cgt_incidence_anchor.after.sub_exempt` / `.loss` / `liable_mass` | no anchor | _pending_ |
| `uk_cgt_projection_entrants.max_entrants` (year) | would have failed: 10.8m (2025) | _pending_ |
| pass-2 score (head-to-head) | 495–24 | _pending_ |
| dashboard 2026-27 CGT taxpayers | 11.54m | _pending after the staged upload_ |

## Status

Code, manifests, fixtures and pins are on the branch; the synthetic smoke build, the licensed "would have
failed v20" seam receipt and the v22 rebuild remain to be run and recorded here.
