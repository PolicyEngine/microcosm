# UK target vintage audit for the 2025 calibration (microcosm#1123, C0)

Audited 2026-10-07 against the pinned Chronicle feed (825406f) and each
publisher's release calendar. Every UK target the 2025 compile binds from a
fact older than 2025 was checked for a newer or revised edition covering
2025.

## Source-preference rule

The rule applies in this order:

1. The latest published, revised official estimate for the calibration
   period, for example ONS mid-2025 population estimates, NRS 2025 households
   and the LFS 2025 households for England.
2. Failing that, the latest official estimate for an earlier period, bridged
   by a declared index built from newer estimates. An example is Wales
   mid-2024 households rolled forward by the growth in Wales's mid-2025
   population estimate.
3. A projection, used for shares only and never for levels.
4. A declared hold in `uk/uprating_holds.json`.

Every bound number comes through Chronicle. A newer edition that Chronicle
does not carry is requested there before the re-pin (C3), so the re-pin
happens once.

## Local families, households and population

These are covered by PolicyEngine/chronicle#313:

- LFS 2025 households for England and its regions;
- NRS 2025 households for Scotland;
- Welsh Government mid-2024 households;
- NISRA projections and LPS dwelling stock;
- mid-2025 local-authority ages;
- the monthly UC cubes;
- PIPR.

The UK, country and region single-year-of-age sheets of ONS mid-2025
(`mye25tablesuk.xlsx`, 1 October 2026) were added in the #313 comment, item 1.
They feed `ons.population.uk_total`, the region age bands and the sex-by-age
rows. These rows compile at mid-2024 today (UK total 69,281,437). The same
release revises mid-2024 to about 69.26m.

## National families held at an older period

### Newer edition published, not yet in Chronicle

Each of these was requested in the PolicyEngine/chronicle#313 comment:

- Student loans in England 2025-26 (18 June 2026, corrected 2 July 2026):
  FY2025-26 repayments by plan (item 3).
- ONS preliminary UK national balance sheet estimates (10 June 2026): land
  for 2025, flagged preliminary (item 6). The layout has not been verified
  yet.
- DfI public transport statistics Northern Ireland 2025-26 (16 September
  2026): bus passenger receipts (item 7).
- ISC census, January 2025 and January 2026 (item 8). The publisher page
  refused automated fetching, so its figures must be verified from the
  publisher copy.
- ONS public sector employment, June 2026 quarter (15 September 2026): the
  four 2025 quarters, for a calendar-2025 mean (item 5).

### Newer data already in Chronicle but not bound

These are rebound in C3:

- `ons/public_sector_employment_2026` (March 2026 quarter);
- `hmrc/hydrocarbon_oils_quantities_june_2026`: the FY2025-26 fuel duty
  total, provisional. The cars-only target takes the car share of the OBR
  April 2024 note applied to this total, and is declared as such;
- `obr/efo_receipts_march_2026` (2025-26 estimates);
- SLC student support 2025: the provisional 2025/26 rows (item 4 asks
  Chronicle to emit them).

### Current edition is the latest

Each of these takes a declared index or a reviewed hold:

- **CGT 2024-25** (27 August 2026): a `matched_period` hold. The binding
  measures the rows under the 2024-25 rules. 2024-25 includes disposals
  brought forward ahead of the October 2024 rate change, so it is not
  projected. The 2025-26 outturn is due around August 2027.
- **Salary-sacrifice reliefs 2024-25:** the employer NICs relief is restated
  to the 2025 employer rate with the sacrificed amount anchored at 2024-25
  (#1069 c11). The income-tax and employee-NIC reliefs follow it as a
  `reviewed_no_index` hold (C6b): their rates do not change between 2024-25
  and 2025-26. Growing all the reliefs with earnings is an open ruling.
- **Public transport support and revenue for 2024-25:** DfT BUS05 and ORR
  (next editions November 2026), Scottish and Welsh bus statistics, and
  NITHC 2024/25. These are `reviewed_no_index` holds, except the England fare
  receipts, which the BUS0415 fares index carries.

## In-year snapshots

Counts dated inside 2025 are declared `in_year_snapshot`, not uprated:

- council-tax dwelling stock (England October 2025, Scotland September 2025);
- the benefit cap (November 2025);
- the two-child limit (April 2025);
- the UC payment distribution and the Scottish under-one cell (December
  2025);
- New Style JSA (August 2025);
- DfE funded early education (January 2025).

The UC caseload headline binds the calendar-2025 window.

## Where the outcome is recorded

`uk/uprating_holds.json` records each held family's kind, reason and expiry.
`uk/target_doctrine_exceptions.json` lists the holds a later #1123 change
closes, each naming that change. The compile refuses an undeclared, expired
or stale hold.

## Outcome of the re-pin (942a1fa, 2026-10-09)

- **Moved to a 2025 edition:**
  - the national population rows (ONS mid-2025);
  - local-authority ages (mid-2025, every authority including Northern Ireland);
  - SLC repayments (FY2025-26);
  - land values (preliminary end-2025, bound explicitly as preliminary);
  - public sector employment (mean of the four 2025 quarters);
  - ISC (January 2025 census day, an `in_year_snapshot` hold);
  - DfI receipts (FY2025-26).
- **Still held, with a declared reason:**
  - SLC student support: the provisional 2025/26 early-year tables report totals by level and domicile, not the products bound; the final edition is due 26 November 2026.
  - Cars fuel duty: the cars share needs a fiscal-to-calendar roll-forward of HMRC's provisional FY2025-26 total, which is not built yet.
  - Constituency ages: no mid-2025 constituency edition is published yet. They are `control_rescaled` under the mid-2025 regional controls.
