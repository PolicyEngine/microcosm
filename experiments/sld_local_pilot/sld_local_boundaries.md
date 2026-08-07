# SLD layer: declared boundaries

Per-district weights are calibrated to ACS 5-year demographics and household-income brackets only: population by age band (S0101), household counts and household income brackets (B19001). All tax and program detail is inherited from the state-and-national calibrated artifact solve and is NOT district-calibrated.

## Vintages

- ACS window: 2020-2024 ACS 5-year estimates (2024 dollars)
- District boundaries: 2024_state_legislative_districts
- Membership source: Census 2024 SLD block-equivalency files
- Period alignment: district targets are 2020-2024 5-year survey aggregates applied to the 2024 build-year artifact; this alignment is declared, not adjusted

## Membership assignment

District membership is derived at the 2024 boundary vintage: exact or population-weighted within-tract lookup for rows carrying certified tract geography, seeded block-overlap draws for ACS-spine rows conditional on (PUMA, congressional district, county) where that cell has block support, degrading through declared coarser conditioning (PUMA+county, PUMA+CD, PUMA) where it does not; the realized method mix below is the honest record. Rows whose certified tract is absent from the ladder also degrade to the seeded path and are counted.

- puma_cd_county_draw: 12,972
- puma_county_draw: 718
- tract_exact: 490
- tract_split_draw: 371
- unassigned row share: 0.0000%
- unassigned weight share: 0.0000%
- certified-tract rows degraded to seeded draws: 0

## Household universe

- gq_marker_present: True
- n_group_quarters_rows: 1617
- statement: household counts and income brackets bind on housing-unit households only (ACS TYPEHUGQ 1); group-quarters rows support the population age bands

## Income instrument

Income brackets bind on a declared ACS money-income analog built from artifact input columns (members aged 15 and over); the published median household income (B19013) is validation-only — a linear reweighting operator cannot honestly target a median.

Declared omissions:
- public_assistance_cash: TANF cash assistance is engine-computed, not an artifact input; omitted from the bracket instrument
- supplemental_security_income: SSI is engine-computed (seeded take-up), not an artifact input; omitted from the bracket instrument

Declared exclusions:
- capital_gains: ACS money income excludes capital gains and losses
- farm_income_column: the model's farm_income is Schedule J income averaging, separate from self-employment; farm self-employment enters via farm_operations_income
- in_kind_transfers: SNAP, housing subsidies, and other in-kind transfers are not money income
- tax_credits: refundable tax credits are not ACS money income
- tip_income_column: the model's employment_income input already includes tips; adding the tip_income memo column would double count
- under_15_income: ACS money income counts persons aged 15 and over; income carried by younger household members is excluded from the analog
- artifact columns absent from the recipe resolution: survivor_benefits, financial_assistance

## Doctrine

- anchor_rule: artifact_calibrated_weights
- max_weight_ratio: 100.0
- min_initial_weight: 0.0001
- scale_rule: default_target_loss_scales
- target_loss_cap: 10.0
- target_weight_rule: uniform

## Small-area tails

- sldu: 29 districts, 0 target rows past the loss cap at final, 0 pushed out
- sldl: 75 districts, 0 target rows past the loss cap at final, 0 pushed out
- Thin districts (< 150 rows): 13
- minimum effective sample size (sldl): 20.8
- minimum effective sample size (sldu): 52.8

## Consumption

Three lenses: household-level results use artifact rows directly; statewide results use the artifact's calibrated weights; by-district results use the per-district sidecar weights. District weight columns are valid only within their own district's rows.
