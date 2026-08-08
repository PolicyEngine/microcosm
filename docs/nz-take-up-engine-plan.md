# microcosm-nz v1 plan: the New Zealand dollar take-up engine

Working plan for [epic #343](https://github.com/PolicyEngine/microcosm/issues/343). Companion inventory: [nz-calibration-targets.md](./nz-calibration-targets.md). Rules: [rulespec-nz](https://github.com/TheAxiomFoundation/rulespec-nz) through the Frame `RulesEngine` protocol via the Axiom adapter (shared with the Belgium pilot under #259 — NZ adds coverage, not adapter code).

## Product definition

For each major NZ transfer, estimate **predicted entitlement dollars** (rules × calibrated population, full take-up) against **actual dollars paid** (administrative expenditure), and present the gap by income percentile and family type against a living-income threshold. The first published cut is the **Accommodation Supplement**, where survey evidence puts receipt among the potentially eligible at ~44% (MSD Income Support Survey 2022, findings pack 7) versus ~87% for Working for Families and ~97% for Best Start — the sharpest quantified take-up wedge in the system.

## Method (Belgium pilot recipe, recalibrated)

1. **Donor pool**: the populace-us support pool (57,240 households / 166,302 persons in the BE pilot vintage). Donor records are US support records and are never presented as NZ microdata; support-stratum labels ship with the artifact.
2. **Reweighting**: microcosm-calibrate against the v1 target set below. Two BE-pilot solver lessons apply verbatim: initialize design weights at the destination population scale (or `target_loss_cap` saturates and gradients vanish), and consume total rows, not component+total double counts, from multi-row target tables.
3. **Rules leg**: rulespec-nz composed modules via the Axiom adapter. Child support is excluded from composed outputs until the income-shares re-encode lands (rulespec-nz#74).
4. **Periods**: tax year 2026-27 primary (1 Apr 2026 rates are current across the encoded surface); PPL/rates-rebate use their 1 July steps.

## v1 target set

Sixteen margins, all public, drawn from the inventory's ranked list (numbers refer to inventory rows):

| Margin | Source | Kind |
|---|---|---|
| Age (5-yr bands) × sex × region ERP | Stats NZ subnational estimates 2025 (S7) | count |
| Single-year age national ERP (collapsed bands) | Stats NZ national estimates (S8) | count |
| Household composition / family type | Census 2023 (S2/S3) | count |
| Tenure (own / rent) × region | Census 2023 (S1/S2) | count |
| Individual taxable income by band | IRD I1 | count + $ |
| Age × income band | IRD I1 tab 4 | count |
| Wage/salary distribution | IRD I2 | $ |
| Main-benefit recipients by benefit × age group | MSD M1 | count |
| AS recipients by W&I region | MSD M1d | count |
| NZS/VP recipients | MSD O4 | count |
| WfF recipient families + expenditure by credit type | IRD I3 | count + $ |
| Benefit expenditure by program | MSD M2 (2024/25 schedule) | $ |
| Rent distribution by TA × bedrooms | MBIE H3 | $ |
| Student loan borrowers | IRD I4 | count |
| Minimum/average earnings anchors | MBIE O1 + QES S12 | $ |
| Ethnicity × age × region (equity cuts) | Census O5 | count |

Deliberate non-targets: Treasury DistributionExplorer values (full-entitlement modelled — that's the comparison output, not a calibration input); household projections (stale 2013-base); anything IDI-derived.

## The joint-income problem

NZ publishes no couple/family joint income distribution (the inventory's top structural gap). The WfF and AS abatement surfaces depend on it. v1 strategy: inherit the donor pool's within-household income correlation structure, calibrate the individual margins hard, then check the implied family-income distribution against the one published conditional slice we have — the IRD WfF recipient-family income distribution (I3, 2020+). If the implied distribution misses that slice materially, escalate to a copula adjustment as a general microcosm-fit operator (never NZ-specific code).

## Validation gates (phase 2, before any published number)

1. **Per-case**: rulespec-nz vs Treasury IncomeExplorer on the AN25-01 stylized families; the Python `emtr` repo's `test/ref/emtr_output_{1..6}.csv` worked examples (28 columns including `as_amount`, `ftc_abated`, `iwtc_abated`, `winter_energy`, `emtr`) are ready-made fixtures.
2. **Population EMTR**: reproduce AN25-01's distribution — 94% of individuals below 50% EMTR; ~13% of couple-parent and ~30% of sole-parent families above — within a documented tolerance. Note AN25-01 excludes student-loan repayments from EMTRs; match that treatment in the comparison run.
3. **Aggregates**: WfF, main benefits, NZS, AS totals vs the M2/I3 dollar schedules (expect gaps = take-up, which is the product — so the gate is on *gross entitlement vs DistributionExplorer*, and on demographics vs Stats NZ, not on matching actual spend).
4. **Upstream gate**: rulespec-nz#73 (verified value fixes) merged.

## Take-up computation

- **Full-entitlement denominator**: our own rules × population (cross-checked against DistributionExplorer's modelled dollars — same full-take-up assumption, independent stack).
- **Actual numerator**: M2 expenditure by program (+ I3 for WfF; T4/T8 as secondary anchors).
- **Distributional actuals**: recipient counts (M1) × implied average payment, with the uniform-average-payment assumption flagged.
- **Outputs**: dollar gap by program; gap by income percentile × family type; AS gap against the ~44%/~87%/~97% survey anchors; living-income overlay.
- **AS specifics**: entitlement needs rents (H3 bond data by TA/bedrooms), the Area 1–4 crosswalk (H4), and area maxima (H4b). The April 2027 area redraw + homeowner 30%→40% entry change (H4c) is a known series break — encode both vintages when 2027 rules land.

## Risks / honesty constraints

- Donor-pool provenance: every output carries the US-support-stratum banner; nothing is presented as NZ microdata.
- Joint-income imputation is the weakest link; the WfF-recipient-slice check is the tripwire.
- FamilyBoost has no admin statistics — modelled only, flagged.
- Regional dollar splits assume uniform average payment within program.
- legislation.govt.nz WAF-blocks scripted fetch; rules provenance rides on rulespec-nz's corpus pins, not live fetches.
