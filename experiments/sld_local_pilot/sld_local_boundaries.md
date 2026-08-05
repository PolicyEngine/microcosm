# SLD layer: declared boundaries

Per-district weights are calibrated to ACS 5-year demographics and household-income brackets only: population by age band (S0101), household counts and household income brackets (B19001). All tax and program detail is inherited from the state-and-national calibrated artifact solve and is NOT district-calibrated.

## Vintages

- ACS window: 2020-2024 ACS 5-year estimates (2024 dollars)
- District boundaries: 2024_state_legislative_districts
- Membership source: Census 2024 SLD block-equivalency files
- Period alignment: district targets are 2020-2024 5-year survey aggregates applied to the 2024 build-year artifact; this alignment is declared, not adjusted

## Membership assignment

District membership is derived at the 2024 boundary vintage: exact or population-weighted within-tract lookup for rows carrying certified tract geography, seeded block-overlap draws conditional on (PUMA, congressional district, county) for ACS-spine rows.

- puma_cd_county_draw: 12,972
- puma_county_draw: 718
- tract_exact: 490
- tract_split_draw: 371

## Income instrument

Income brackets bind on a declared ACS money-income analog built from artifact input columns; the published median household income (B19013) is validation-only — a linear reweighting operator cannot honestly target a median.

Declared omissions:
- public_assistance_cash: TANF cash assistance is engine-computed, not an artifact input; omitted from the bracket instrument
- supplemental_security_income: SSI is engine-computed (seeded take-up), not an artifact input; omitted from the bracket instrument

## Small-area tails

- sldu: 29 districts, 0 target rows past the loss cap at final, 0 pushed out
- sldl: 75 districts, 0 target rows past the loss cap at final, 0 pushed out
- Thin districts (< 150 rows): 13

## Consumption

Three lenses: household-level results use artifact rows directly; statewide results use the artifact's calibrated weights; by-district results use the per-district sidecar weights. District weight columns are valid only within their own district's rows.
