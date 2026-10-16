The registers that change the UK solve are in place before the first local release build (microcosm#1301).

**Employer NICs.** `obr.ni_employer` joins `uk/calibration_measure_exclusions.json` as a measured diagnostic (approved 2026-10-16, expires 2027-01-16, the three-month window of the `obr.vat` engine-factor diagnostic). The engine's `ni_employer` is gross of the Employment Allowance, which OBR's receipts are net of, and the row's calendar-2025 window takes a quarter of FY2024-25 that the engine prices on FY2025-26 rules. The row is tracked on policyengine-uk#2459, which splits the allowance out of policyengine-uk#1918. It leaves both lines' objective; `tools/diagnose_uk_uc_support.py` no longer asks its protected probe for it.

**High-value homes.** The family's region rows control it on the local line, so:
- `tools/generate_uk_local_target_references.py` defers its constituency level whole (`high_value_homes_constituency_grain_deferred`, `defer_if_compiles`), after the English band-H precedent, with a roster guard of 575 England and Wales constituencies;
- a new signed small-cell rule, `hvcts_la_household_floor` in `uk/local_small_cell_rule.json` (prefix `microcosm.high_value_homes.`, floor 400 homes, about two household weights at 150,000 households, expires 2027-04-01), defers the local-authority cells under the floor and keeps 192, 101 and 57 cells over £1m, £1.5m and £2m, whose deferred homes still count in their region rows;
- the `high_value_home_stock_derived_estimate` adjudication records where the family binds, and that the seat rank correlations, now out of sample, decide its renewal.

**Islands.** Orkney (`local_authority/S12000023`) and Shetland (`local_authority/S12000027`) are pre-signed gate-only exclusions of the area-support floor in `uk/local_area_support_exclusions.json`, scoped to the local line and expiring 2027-01-16. Their cells stay bound.
