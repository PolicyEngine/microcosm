# Household reported money income: ACS and prepared ASEC

`microcosm.build.us_runtime.household_reported_income` measures the *publisher's
own* household total on two arms and states what makes them different. The
native ACS arm reads `HINCP`; the prepared ASEC arm reads `asec_HTOTVAL`.
Neither arm sums components. The module runs on explicitly supplied projections
of native source columns: it builds no Frame, loads no archive, imports no
country model and reads no engine output.

`microcosm.build.us_runtime.acs_household_source_columns` declares the
explicitly versioned household source-column contract that makes `HINCP`
available, and reads it additively.

## Primary source authority

A SHA-256 authenticates locally retrieved bytes; it is not a publisher
signature. Each digest below has a stated provenance:

- **ACS PUMS Data Dictionary** — retrieved from the publisher on 2026-09-06.
  The retrieved bytes equal the pin already carried by
  `acs_2024_housing_donor.json` and `acs_2024_housing_universe.json`.
- **ACS PUMS User Guide** — retrieved from the publisher on 2026-09-06. No
  prior repository pin existed; this contract introduces it.
- **ACS Subject Definitions** — not retrieved here. The copy already retrieved
  for the ACS rent source authority was checked, and its digest equals the pin
  already carried by `acs_2024_housing_donor.json`.
- **CPS ASEC 2025 dictionary** — retrieved from the publisher on 2026-09-06.
  The retrieved bytes equal the pin already carried by
  `asec_current_money_domains_v1.json` and `asec_housing_status.py`. The 2023
  and 2024 dictionary digests in `ASEC_HTOTVAL_VINTAGES` are copied from that
  same existing authenticated metadata and were not re-retrieved.
- **B19001 group metadata** — retrieved from the Census API on 2026-09-06. The
  retrieved bytes equal the content-addressed capture already held by the
  congressional-district reference evidence bundle.

| Document | URL | SHA-256 |
|---|---|---|
| 2024 ACS PUMS Data Dictionary (October 16, 2025) | `https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf` | `929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff` |
| 2024 ACS PUMS User Guide | `https://www2.census.gov/programs-surveys/acs/tech_docs/pums/2024ACS_PUMS_User_Guide.pdf` | `70d485604dfd89534eef7d491a219ab7d336f98e724732985a0a859eac5d9787` |
| 2024 ACS Subject Definitions | `https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf` | `6e0c5b1a5d3ebf7e300e583ac6424cbd5f09de50d69e7ef03aa6d070d9c4a7a8` |
| CPS ASEC 2025 public-use data dictionary | `https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf` | `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f` |
| Census API group metadata, 2024 ACS 1-year `B19001` | `https://api.census.gov/data/2024/acs/acs1/groups/B19001.json` | `d73a050438fb75f43c1b7738d0aa42c3c7ad17e4fb62b2661294652fbd05aafa` |

### ACS `HINCP`, PUMS dictionary PDF page 15

> HINCP          Numeric     7
>          Household income (past 12 months, use ADJINC to adjust HINCP to
>          constant dollars)
>                bbbbbbb          .N/A (GQ/vacant)
>                0                .No household income
>                -59999           .Loss of -$59,999 or more
>                -59998..-1       .Loss of $1 to -$59,998
>                1..9999999       .Total household income in dollars (Components
>                                 .are rounded)

The applicable universe is therefore occupied housing units, read off the
printed not-applicable statement: the publisher declares the field N/A for
group-quarters and vacant records, so a blank there is an out-of-universe state
and never a zero. `0` is a distinct declared value
meaning no household income. `-59999` is a declared open-ended loss floor;
`-59998..-1` are ordinary signed losses. Values outside `[-59999, 9999999]`
are outside the published range.

### ACS `ADJINC`, PUMS dictionary PDF page 2 (housing record)

> ADJINC         Character   7
>          Adjustment factor for income and earnings dollar amounts (6 implied
>          decimal places)
>                1015250 .2024 factor (1.015250)

The 2024 1-year dictionary enumerates exactly one code. The same entry appears
identically on the person record at PDF page 33.

### ACS adjustment, PUMS User Guide section F, PDF page 16

> Divide ADJINC by 1,000,000 to obtain the inflation adjustment factor and
> multiply it to the PUMS variable value to adjust it to 2024 dollars.
> Variables requiring ADJINC on the Housing Unit file are FINCP and HINCP.

This is the authority for applying the factor **exactly once**, from the
**housing-unit** file, to `HINCP`. `ADJHSG` is a different factor for housing
dollar amounts and is not used here.

### ACS reference period, Subject Definitions printed pages 96 and 98

> Income of Households – This includes the income of the householder and all
> other individuals 15 years old and over in the household, whether they are
> related to the householder or not. … Although the household income statistics
> cover the past 12 months, the characteristics of individuals and the
> composition of households refer to the time of interview.

> Adjusting Income for Inflation – Income components were reported for the 12
> months preceding the interview month. Monthly Consumer Price Indices (CPI)
> factors were used to inflation-adjust these components to a reference
> calendar year (January through December).

So the ACS amount is a **rolling** twelve-month total restated to calendar-2024
dollars. The published concept is built from **eight** income types (wage or
salary; self-employment; interest, dividends, net rental, royalty or estate and
trust income; Social Security or Railroad Retirement; SSI; public assistance;
retirement, survivor or disability; all other income) and explicitly excludes
capital gains.

### ASEC `HTOTVAL`, ASEC 2025 dictionary PDF page 12

> HTOTVAL                      8 108             (-999999:99999999)
> total household income
> Values: 0 = none
>         negative dollar amount
>         positive dollar amount
> Universe: All Households

`PTOTVAL` at PDF page 48 is the person analogue, `total persons income`, with
`Universe: All Persons aged 15+`. The published household total's contributing
members are that 15-and-over universe; this module never sums `PTOTVAL` into
`HTOTVAL`, and the household field carries no declared open-ended bound.

`CAP_VAL` at PDF page 44 is `capital gains value` with `Universe: CAP_YN = 1`.
Capital gains are outside the money-income concept on both arms. `CAP_VAL` is
therefore neither part of either household total nor the ACS all-other-income
component `OIP`, and no unallocated residual may be labelled "other income".

### What the executed ACS person roster does and does not carry

`acs_pums._PERSON_REQUIRED` carries six of the eight published income types:
`WAGP`, `SEMP`, `SSP`, `SSIP`, `RETP`, `INTP`. `PAP` and `OIP` are absent. A
household total assembled from those six would be an incomplete total wearing
a complete total's name, which is why both arms use the publisher aggregate.
The publisher also gives no basis in these documents for splitting `SSP` into
Social Security and Railroad Retirement, or `RETP` into private and public
pensions: the dictionary and subject definitions describe each as one combined
type. This module makes no such split.

## The ACS household source-column contract

`AcsHouseholdSourceColumns` declares a roster with a version and its own
digest.

- **Version 1** restates the executed 2024 spine roster exactly as
  `acs_pums` runs it. A test asserts it against the loader's own tuples, so
  drift fails rather than silently redefining version 1. Nothing in this lane
  reads or rewrites it, and the executed loader's bytes are unchanged, so its
  implementation identity and any existing source cache stay valid.
- **Version 2** adds exactly `HINCP` and the housing-record `ADJINC`.

`read_acs_household_income_columns` performs the version 2 read additively from
the same pinned archive. It reuses the executed loader's archive walk and its
occupancy rule, drops vacant records with a count, keeps group-quarters
placeholders so the measurement can mark them out of universe, retains literal
values as pandas nullable `Int64`, leaves Census blanks missing, and applies no
adjustment factor. The two reads align on native `SERIALNO`, never on row
position, because the version 1 loader may apply its smoke subset.

## The measurement

```python
measured = measure_acs_household_reported_income(
    household_keys=("2024HU0000001", "2024GQ0000001"),
    hincp=table["HINCP"],            # literal Int64, blanks preserved
    adjinc=table["ADJINC"],          # housing-record factor, Int64
    housing_unit_kind=table["TYPEHUGQ"],
)
measured = measure_asec_household_reported_income(
    household_keys=("2024:1",),
    htotval=household["asec_HTOTVAL"],
    income_year=household["income_year"],
)
```

`HouseholdReportedIncome` carries the household value (`values`, float64 in
2024 dollars, `NaN` where unavailable), the availability and quality axes, the
literal raw amounts, the source origin, the source and target price periods,
the single transformation, and JSON-safe evidence. Arrays are read-only;
`raw_values` and `evidence` return independent copies.

`AVAILABILITY` is `observed` / `unavailable_out_of_universe` /
`unavailable_in_universe`. Only the ACS arm can report the out-of-universe
code, because only the ACS dictionary declares one. `QUALITY` is
`exact_publisher_total` / `declared_open_ended_bound` / `not_applicable`; only
the ACS arm can report a declared bound, at `HINCP == -59999`.

Price handling is stated, not implied. The ACS arm multiplies once by the
declared `1015250 / 1000000`; requiring that exact enumerated code is what
refuses an input a caller already adjusted. The ASEC arm uses the packaged,
byte-pinned `asec_current_money_price_basis_v1.json` C-CPI-U basis
(`163.6` / `170.0` / `174.4`, target 2024): the 2024 identity year copies its
amount with no arithmetic, and earlier years multiply once by
`174.4 / source`. Zeros never enter arithmetic. The stale CPI-U-RS prose the
Subject Definitions use for *across-year* comparisons is not the ASEC
restatement and is not reused here.

`align_household_values` joins by exact native string key, so a leading-zero
identifier keeps its spelling; integer keys refuse before a zero can be lost.

## The two-arm surface

`two_arm_comparability(acs, asec)` is the whole consumer API. It returns both
origins and an explicit list of source-dependent differences — reference
period, price restatement, household universe, declared bounds — and produces
no pooled total, ratio or reconciliation. A consumer that wants a comparison
must state which difference it is willing to ignore; nothing here does that
for it.

## B19001

`b19001_household_income_distribution_spec` returns a
`microcosm.build.quality_aggregates.DistributionSpec` whose sixteen bins
reproduce the pinned Census group metadata exactly, with the same lower
inclusive / upper exclusive rule the published benchmark block declares. These
are **reserved same-source holdout observations**: `B19001_2024_METADATA`
records `targets_allowed = False` and `tuning_allowed = False`. Building this
spec activates no national or congressional-district target and scores no
candidate.

## Limits

Fixture checks establish engineering contracts only. No genuine ACS or ASEC
microdata was read or executed for this contract, so nothing here establishes
a real join's completeness, an income distribution's accuracy, agreement with
any published aggregate, or release eligibility. The measurement is verified
on tiny invented inputs.

Extending this to a measured genuine small rung needs exactly one new input:
the version 2 household read over the already-pinned `csv_hus.zip` archive,
which yields `HINCP`, `ADJINC` and `TYPEHUGQ` keyed by `SERIALNO`. The ASEC arm
needs no new input — `asec_HTOTVAL` is already attached to the prepared source
by `asec_household_observations`.
