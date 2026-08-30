# New Zealand transport and take-up plan

Working plan for [epic #343](https://github.com/PolicyEngine/microcosm/issues/343). Companion inventory: [nz-calibration-targets.md](./nz-calibration-targets.md). Rules: [rulespec-nz](https://github.com/TheAxiomFoundation/rulespec-nz) through the Frame `RulesEngine` protocol via the Axiom adapter shared with the Belgium pilot under #259.

## Implementation status: 29 August 2026

The spec-only [NZ country package](../packages/microcosm-build/src/microcosm/build/nz/country_package.json)
now compiles through the shared country seam. It is a scaffold, not a calibrated
population or live release. Follow the [country-expansion playbook](country-expansion-playbook.md)
for the three parallel evidence lanes and the build/publication gates.

The donor pin is the public US Build P release
`populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`, file
`populace_us_2024.h5`; the package records its SHA-256 and byte size. The first
rules bridge is the WFF official-Budget reform contract at rulespec-nz commit
`3b663b3e6eb6408351154990be0c4b92d42c92da`. It requires 11 evidenced Family
inputs and permits 10 adapter-padding zeros outside the requested outputs'
dependency graphs. Automatic unit derivation is not yet bound by the adapter.

The [closed export contract](../packages/microcosm-build/src/microcosm/build/nz/export_contract.json)
now has a real-runtime smoke: three synthetic families in two households,
household weights 2 and 7, and family weights resolved by the Frame as 2, 7,
and 7. The two upstream companion cases produce entitlement changes of
NZD 568.50 and NZD 438.50; Frame-weighted fixture totals are NZD 1,137 and
NZD 3,069.50. These are test-fixture results, not NZ population estimates.
The smoke verifies tax-year bounds, missing-input refusal, formula-owned
export exclusions, and exact Decimal-input HDF round trips. The generic
adapter now accepts canonical RuleSpec roots and `AxiomPeriod` bounds.
Its native float64 input boundary accepts Decimal values only when nine-place
rounding recovers them exactly; it does not claim unrestricted Decimal128 ABI
support. Treasury source activation remains pending immutable corpus
publication and protected validation.

The release contract names `policyengine/populace-nz` and declares calibration,
target, and comparison dashboard capabilities. Build readiness still requires
cell-pinned Chronicle facts, period/currency/family bridges, shared transport
operators, and reviewed NZ calibration thresholds. The WFF reform checks are
the first bounded official-score validation slice; the intended take-up
research product below remains separate.

Run the smoke against the pinned rules checkout and a real Axiom dense native
extension (engine commit `bb4b5684870547756078a62f1866a77c5b56f7f3`):

```bash
POPULACE_RULESPEC_NZ=/absolute/path/rulespec-nz \
  uv run --no-sync pytest \
  packages/microcosm-build/tests/test_country_spec.py -k NewZealandAxiomTransport
```

Install `microcosm-build`, `microcosm-frame[axiom]`, the Axiom Python wrapper,
and its built native extension into the workspace environment first;
`--no-sync` retains the separately installed native build. The test reports
a skip, not a pass, when the real extension or rules checkout is absent.

## Product definition

For each major NZ transfer, estimate **predicted entitlement dollars** (rules × calibrated population, full take-up) against **actual dollars paid** (administrative expenditure), and present the gap by income percentile and family type against a living-income threshold. The first published cut is the **Accommodation Supplement**, where survey evidence puts receipt among the potentially eligible at ~44% (MSD Income Support Survey 2022, findings pack 7) versus ~87% for Working for Families and ~97% for Best Start — the sharpest quantified take-up wedge in the system.

## Method (Belgium pilot recipe, recalibrated)

1. **Donor pool**: the immutable populace-us release pinned in the NZ country package. Historical BE pilot record counts do not describe this pin; the loader must inventory the authenticated artifact. Donor records are US support records and are never presented as NZ microdata; support-stratum labels ship with the artifact.
2. **Reweighting**: microcosm-calibrate against the v1 target set below. Two BE-pilot solver lessons apply verbatim: initialize design weights at the destination population scale (or `target_loss_cap` saturates and gradients vanish), and consume total rows, not component+total double counts, from multi-row target tables.
3. **Rules leg**: rulespec-nz composed modules via the Axiom adapter. Child support is excluded from composed outputs until the income-shares re-encode lands (rulespec-nz#74).
4. **Periods**: tax year 2026-27 primary (1 Apr 2026 rates are current across the encoded surface); PPL/rates-rebate use their 1 July steps.

## Candidate calibration and validation inputs

The public inventory below contains both calibration candidates and held-out
validation inputs. Observed values and source cells live in Chronicle; the NZ
package contains unactivated authoring references until every selected cell has
a reviewed Frame binding and target-period basis. Recipient counts constrain a
receipt/take-up layer, not full-entitlement outputs.

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
| WfF recipient families by credit type | IRD I3 | receipt-state count |
| WfF and benefit entitlement/expenditure schedules | IRD I3 + MSD M2 | held-out dollar validation |
| Rent distribution by TA × bedrooms | MBIE H3 | input/validation, not a hard expenditure target |
| Student loan borrowers | IRD I4 | count |
| Minimum/average earnings anchors | MBIE O1 + QES S12 | input/macro validation |
| Ethnicity × age × region (equity cuts) | Census O5 | count |

Deliberate non-targets: Treasury DistributionExplorer values and official Budget
reform scores (independent model comparisons); programme expenditure and
entitlement-dollar schedules (held-out take-up and accounting evidence);
household projections (stale 2013-base); anything IDI-derived.

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
