# US County Files: household measures and schema manifest

This covers microcosm#922 items 6, 7 and 8: the household measures that ship
beside each County Files H5, and the schema manifest that documents every
delivered table and column. Two tools produce them:

- `tools/us_household_measures.py` computes household earnings, gross income
  and net income with policyengine-us and writes them as a sidecar file.
- `tools/us_release_schema_manifest.py` lists every table, column, dtype, row
  count and null count of the H5 and its sidecars, diffs them against an
  earlier schema, and writes JSON and Markdown.

Neither tool builds, calibrates, stages or publishes anything.

## Files for each year

Each delivered year (2026 and 2027) has:

| File | What it holds | Read with |
|---|---|---|
| `<name>.h5` | The six entity tables (`person`, `household`, `tax_unit`, `spm_unit`, `family`, `marital_unit`) and `_time_period`, the file's year | `pandas.read_hdf` (needs PyTables); runs in policyengine-us |
| `<name>.household_measures.csv.gz` | One row per household: the three measures and their five components | `pandas.read_csv`, no optional dependency |
| `<name>.household_measures.json` | The receipt: the H5's and sidecar's sha256, the year, the engine version, the resolved definitions and the checks | any JSON reader |
| `schema_manifest.json`, `schema_manifest.md` | Every table and column of the two data files, and the changes from the prior schema | any JSON reader; Markdown for publication |

```python
import pandas as pd

person = pd.read_hdf("county_files_2026.h5", "person")
household = pd.read_hdf("county_files_2026.h5", "household")
measures = pd.read_csv(
    "county_files_2026.household_measures.csv.gz", float_precision="round_trip"
)
household = household.merge(measures, on="household_id", validate="one_to_one")
```

`float_precision="round_trip"` reads every value back exactly as written.
pandas' default parser can differ from the written value in the last binary
digit (about one part in 10^16), which is enough to break an exact equality
such as the gross income identity below.

## Household measures

The measures are formula outputs, so they can't sit in the entity tables. The
export gate (`PolicyEngineUSEngine.write_dataset`) refuses formula-owned
columns there: a stored output would become an input and mask reforms. They
ship as a sidecar instead of as an extra key inside the H5. policyengine-us
2.2.1 would ignore an extra key, because `USSingleYearDataset` reads only the
six entity tables and `_time_period`. So both options keep the H5 runnable;
the sidecar was chosen for four reasons:

- the certified H5 stays byte-identical, so every sha256 that a release
  manifest or receipt records for it stays valid;
- no in-place HDF5 write touches a multi-GB release file;
- no loader that lists the H5's keys can trip over an unexpected one;
- a gzip CSV reads with `pandas.read_csv` alone, as well as from R, Stata or
  a spreadsheet.

### Columns

| Column | Dtype | Definition |
|---|---|---|
| `household_id` | int64 | The household's id in the H5's `household` table; each appears exactly once |
| `year` | int64 | The H5's `_time_period`, the year the engine computed |
| `earnings` | float64 | Labor income: `employment_income + self_employment_income + sstb_self_employment_income + farm_operations_income + partnership_self_employment_net_earnings`, summed over the household's members |
| `gross_income` | float64 | Pre-tax market and benefit income: `household_market_income + household_benefits` |
| `net_income` | float64 | Income after taxes and transfers: `household_net_income` |
| `household_market_income` | float64 | Income from non-government sources (below) |
| `household_benefits` | float64 | Government benefits (below) |
| `household_refundable_tax_credits` | float64 | `income_tax_refundable_credits + household_refundable_state_tax_credits` |
| `household_tax_before_refundable_credits` | float64 | `employee_payroll_tax + self_employment_tax + income_tax_before_refundable_credits + flat_tax + household_state_tax_before_refundable_credits + local_income_tax_before_refundable_credits + local_occupational_tax` |
| `household_health_costs` | float64 | Out-of-pocket health costs such as CHIP premiums, counted only when health benefits are (zero under current law) |

Every amount is annual US dollars for the file's year, computed with
policyengine-us 2.2.1 `Microsimulation` at the H5's own `_time_period`, under
the release's SPM selection (`{"geography_kind": "county"}`). Values are
nominal for that year, not deflated. policyengine-core stores float variables
as float32, so each engine amount carries about seven significant digits
(within $0.06 at $1 million); the CSV writes them in full float64 precision.

**Earnings** sums wages and salaries (`employment_income`) and the
self-employment sources that policyengine-us's `taxable_self_employment_income`
gathers under 26 USC 1402(a): Schedule C profit (`self_employment_income`,
and `sstb_self_employment_income` for specified service businesses), Schedule
F profit (`farm_operations_income`) and general partners' self-employment
earnings (`partnership_self_employment_net_earnings`). It is taken before the
employer-share deduction that the self-employment tax base applies, and it is
negative when self-employment losses exceed wages. Partnership self-employment
earnings are already included in `partnership_income`, and therefore in market
income through `partnership_s_corp_income`. Passive partnership and
S-corporation income also enters market income, but does not enter earnings.

On a person row from the ACS, the sum counts self-employment once. ACS
reports one total (`SEMP`) that covers farm, non-farm and own-partnership
self-employment. The local release build refuses to materialize, calibrate
or package a file unless each ACS person's leaves add up to that total
(`acs_self_employment_identity_gate`, microcosm#1380). The total sits wholly
on `self_employment_income` or, after QBI reconciliation, wholly on
`sstb_self_employment_income`, and farm operations income and partnership
self-employment earnings are zero on those rows. Rows from donor surveys carry
the components that those surveys report separately. An earnings concept
that leaves out the SSTB leaf would therefore miss ACS self-employment that
QBI reconciliation routed there. The County pilot's earnings measure
(`experiments/us-county-pilot-20261024/county_targets.py`) zeroes the SSTB
and farm leaves on ACS rows, guarding against donor draws stacked on `SEMP`
before the gate existed. It includes both leaves on donor rows. The systematic
difference on donor rows is that the pilot omits partnership self-employment
earnings, which this column includes.

**Gross income** is `household_market_income + household_benefits`. Under
policyengine-us 2.2.1 for 2026 and 2027:

- market income sums `employment_income`, `self_employment_income`,
  `sstb_self_employment_income`, `partnership_s_corp_income`,
  `gi_cash_assistance`, `farm_operations_income`, `farm_rent_income`,
  `capital_gains`, `interest_income`, `rental_income`, `dividend_income`,
  `pension_income`, `debt_relief`, `illicit_income`,
  `retirement_distributions`, `miscellaneous_income`, `alimony_income`,
  `strike_benefits` and `ak_permanent_fund_dividend`
  (`gov.household.market_income_sources`);
- benefits sum `social_security`, `ssi`, `snap`, `wic`, `free_school_meals`,
  `reduced_price_school_meals`, `acp`, `ebb`, `tanf`,
  `high_efficiency_electric_home_rebate`,
  `residential_efficiency_electrification_rebate`,
  `unemployment_compensation`, `child_support_received`,
  `workers_compensation`, `educational_assistance`, `financial_assistance`,
  `survivor_benefits`, `household_head_start_benefits`, `basic_income`,
  `trump_dividend`, `housing_assistance`, `household_state_benefits`,
  `commodity_supplemental_food_program` and `household_health_benefits`
  (`gov.household.household_benefits`).

Two of those benefits are zero under current law, because their switches are
off: `household_health_benefits` (see net income below) and
`household_head_start_benefits`
(`gov.simulation.include_head_start_benefits_in_net_income`). The receipt
records both switches.

Refundable tax credits are not part of gross income; they enter net income.
So a household whose refundable credits (an EITC, say) exceed its taxes has
net income above its gross income.
Each receipt records both lists as the engine resolved them for its year
(`engine.resolved_parameters`), and the receipt, not this page, is
authoritative for a delivered file.

**Net income** is the engine's `household_net_income`:

```text
household_market_income + household_benefits + household_refundable_tax_credits
  - household_tax_before_refundable_credits - household_health_costs
```

Health coverage is not counted. `gov.simulation.include_health_benefits_in_net_income`
is false, so `household_health_benefits` (Medicaid, CHIP and other coverage
values) and `household_health_costs` are both zero. The federal refundable
credits (`gov.irs.credits.refundable`: the EITC, the refundable American
Opportunity and child tax credits, the recovery rebate and the refundable
payroll tax credit) exclude the ACA premium tax credit.

### Checks

`compute` refuses to write a sidecar, and `verify` refuses an existing one,
unless every check holds for every row:

- the columns and dtypes are exactly those in the table above;
- every `household_id` in the H5's `household` table appears exactly once,
  and the sidecar names no other household;
- every `year` equals the H5's `_time_period`;
- every value is finite;
- `gross_income == household_market_income + household_benefits` exactly;
- `net_income` equals its five components to within $0.01 plus one
  millionth of the sum of their absolute values. The engine adds them in
  float32, which can differ from the float64 sum by up to about 3 parts in
  10^7 of that scale;
- (`verify`) the H5's and the sidecar's sha256 equal those the receipt
  records.

Before scoring, `compute` also requires every household and group row to have
a member person, and every group unit to sit inside one household.
policyengine-core places members by sorted id position, so a memberless row
would shift later households' members onto the wrong rows. These structural
checks run in `compute` only.

Before computing anything, `compute` refuses an installed engine that defines
`household_net_income`'s adds and subtracts, or `household_market_income`'s
source, differently from what this page documents. It also refuses one where
any earnings variable is not a person variable, or any component not a
household variable.

### Batching

`compute` scores the H5 in batches of whole households (default 5,000), each
in a fresh `Microsimulation`. Every measure is a per-household quantity, and
policyengine-core forbids formula-time randomness, so a batch gives each
household the value a whole-file engine would when its dependencies remain
within that household. Two engine features need extra guards. A formula may
aggregate over its simulation's whole population (the income deciles and the
Medicaid SLCSP state sums); a multi-batch run refuses as soon as any branch of
a batch engine computes one of them (microcosm#956), and the receipt records
any that a single-batch run computed. A person's
`medicaid_claiming_tax_unit_id` may name a tax unit in another household; a
multi-batch run refuses such a link before scoring. Batches must partition
every entity: each group unit has to nest in one household, the same premise
as the post-export scorer. Each batch holds only the file's year, using
policyengine-us's supported `extend_single_year_dataset(..., end_year=year)`
API instead of retaining inputs through the CPI-U projection horizon.

## Commands

For each delivered year, in the US engine environment
(`uv sync --all-packages --locked --extra us`):

```bash
uv run python tools/us_household_measures.py compute \
    --h5 <release dir>/<name>.h5 --expect-year 2026
uv run python tools/us_household_measures.py verify --h5 <release dir>/<name>.h5
uv run python tools/us_release_schema_manifest.py \
    --h5 <release dir>/<name>.h5 \
    --sidecar <release dir>/<name>.household_measures.csv.gz \
    --prior <earlier schema_manifest.json or table,column,dtype CSV> \
    --output-json <release dir>/schema_manifest.json \
    --output-md <release dir>/schema_manifest.md
```

`compute` loads the H5's tables once (as `USSingleYearDataset` does). Its
estimated peak memory is roughly up to twice the entity tables' uncompressed
in-memory size, plus one batch engine: a fixed-format table read may briefly
hold two copies. This is a rough bound, not a full-release measurement; engine
branches and temporary arrays also use memory. The tool reuses
`policyengine_us.system.system` for definition checks and receipt metadata.
`--batch-households` trades batch-engine memory against per-batch overhead.
Estimated runtime is the observed per-batch time times the number of batches,
plus H5 reads, hashing and sidecar I/O; it is not a measured full-release time.
The receipt records the run's actual wall time and peak RSS.
The manifest tool needs no engine. It reads each `table`-format table in
chunks of about 256 MiB, resolving a column with inconsistent chunk dtypes
through a whole-column read. It reads each `fixed`-format table whole once
because pandas' object-block inference can depend on the other columns in
the block. Neither tool overwrites an existing output.

## Schema manifest

The manifest lists, for each file, its sha256 and size and every table: its
storage format, row count and columns. Keys whose names start with `_`
(`_time_period`, `_populace_staging_metadata`) and the HDF5 root attributes
are listed as file metadata, with their values when short, not as tables. For each column it gives the dtype
pandas reads it as and its null count. A column that Microcosm's frame codec
stores as object-backed nullable booleans also carries
`logical_dtype: "boolean"`. Dtypes are pandas 3 names (`str` for strings);
run the tool in the workspace environment so they compare with earlier
manifests.

The diff keys columns by `(table, column)` and reports tables and columns
added and removed, dtype changes, and renames. A rename is reported only when
it is declared (`DECLARED_RENAMES`, plus any `--renames` JSON) and the prior
has the old name but not the new one, while the current file has the new name
but not the old one. A rename entry carries both dtypes. A declaration that
does not apply is listed with the reason. Dtypes are compared as pandas reads
them; a changed column whose current file stores nullable booleans also shows
its `logical_dtype`.

Known changes from the sample column list sent on August 18 (a 2024 file):

- `person.would_claim_wic` is now `person.takes_up_wic_if_eligible`, the name
  policyengine-us uses for the WIC take-up input (a declared rename).
- `person.weeks_worked` is not in the 2026 or 2027 person tables.
  policyengine-us 2.2.1 defines it by formula from 2025 on (`formula_2025`,
  which carries the previous year's value forward). The build's input-only
  projection holds this formula-owned variable back at every period.
  policyengine-us PR #9716, which closed issue #9660, deleted that formula
  upstream. The column can ship once Microcosm's policyengine-us pin moves
  past that change.
