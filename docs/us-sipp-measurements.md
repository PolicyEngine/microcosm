# SIPP source measurements

`microcosm.build.sipp_measurements` provides pure measurement functions
for explicitly supplied lexical 2023 SIPP rows. It is additive: none of the six
existing SIPP consumers calls it yet. The module lives above `us_runtime` to avoid that package’s existing eager engine
initialization. There is no source loader, fitting operation,
PolicyEngine calculation, donor product, cache, or year-transfer implementation.

`SippSource` carries a caller-supplied data digest, the fixed official master
metadata digest, survey year 2023 and reference year 2022. This is identity metadata,
not proof of source authentication. A future adapter must authenticate the bytes,
retain original source observations, bind this module's code identity and
`MEASUREMENT_RULE_ID`, and pass the complete required source projection. A filename
is never source authority. No mutable research report is a runtime dependency.

## Input and result contract

Wrap each mapping in `SippRow`. Source values are strings or nulls; absent columns,
null values and lexical blanks are distinct. Numeric parsing permits ordinary
signed decimal notation, at most 64 characters, with whole-dollar integrality for
amount/count fields. Nonfinite values, exponent notation, booleans, out-of-domain
numbers and field-specific missing-item codes never become valid amounts. Weight
parsing preserves fractional decimal values. Negative earnings and vehicle equity
remain signed within their respective domains.

Person identity is `(SSUID, PNUM)`; months are 1–12. SSUID remains an opaque string;
PNUM and residence are checked against their published numeric domains. Duplicate
person-months refuse, including alternate lexical spellings of the same numeric
key. Each annual call requires exactly one person. All rows in a call belong to the
one explicitly supplied source/year identity; the adapter owns that binding.

The `contradictions` tuple on `SippRow` carries concise reason codes from any other
source evidence checked by the caller. A caller must not omit a known contradiction
when projecting fields. The earnings function additionally recognizes supplied
public job-spell bounds and the parallel `TPEARN` recode as possible contradictions
to its no-job zero route. Absence of public job slots is never evidence of zero.

Frozen `Measurement` and `Observation` records contain the source identity, rule ID,
concept, period basis, units, value or unknown, allocation evidence, origin, reasons,
and only the requested field/control observations. Required fields are snapshotted;
results do not reference the caller's mappings or retain unrelated survey columns.
Raw-field evidence distinguishes presence and original lexical type. Flags 1–9
retain their separate meanings. In particular, flag 9 is
`component_derived_allocation_unresolved`, not reported or necessarily wholly
imputed. A usable edited measurement is not approval to fit a model.

Unknown values remain diagnostic results. A strict adapter calls `require_value()`
and receives `MeasurementError` before use when the requested value is unknown.
Invalid identities and unsupported source/target periods refuse the operation
immediately. These are local measurement errors, not household exclusion policies.

## Annual flows and annual-once components

`annual_earnings(rows, source=...)` returns
`sipp_current_total_earnings_alt_annual`, associated with `current_total_earnings`:
wages/salaries plus business profits/losses, on the alternate equal-month basis.
It never writes or relabels wage-only `employment_income` and introduces no prior
wages. It sums each of twelve unique reference-month values once. Missing months
or unresolved monthly measurements return an unknown annual value.

Required source fields are `SSUID`, `PNUM`, `MONTHCODE`, `TPEARN_ALT` and
`APEARN_ALT`. A valid numeric amount with an applicable flag is retained even if
RMESR says no job. A structural zero requires **all** of:

- A lexical blank `TPEARN_ALT` and `APEARN_ALT=0`.
- `RMNUMJOBS=0` and `RMESR` in 6, 7 or 8.
- Applicable, non-NIU/non-suppressed `AMNUMJOBS` and `AMESR` flags.
- Valid interview age `TAGE>=15` and valid unique source/person/month keys.
- No explicit contradictory supplied evidence.

`TAGE_EHC` is monthly age, not the interview-age guard. Blank earnings with
count zero and employed status remain unknown. Unpaid family work is a possible
explanation, not a blanket zero authorization. Derived zeros preserve the raw blank,
NIU flag and control-quality evidence. All-published, mixed-origin and
all-structural-zero complete years have distinct statuses. Published positive and
negative amounts cancelling to zero retain their published origin.

`annual_retirement(rows, source=...)` requires `TRETINCAMT`, `ARETINCAMT` and
`TAGE_EHC`, in addition to the keys. It sums twelve usable monthly recodes and
returns `sipp_broad_retirement_recode_annual`. It does not reconstruct the recode,
add annual components a second time, subtract rollovers again, infer zeros at age
boundaries, or call the result regular/taxable pension income. Unexplained monthly
variation is not by itself an invalid value.

`annual_once(rows, field, source=...)` supports only these annual source components:

| Amount / flag | Applicable receipt route and route flags |
| --- | --- |
| `TLIFEAMT` / `ALIFEAMT` | `ELIFEYN=1`, `ALIFEYN`; preserve `ELIFEREG`/`ALIFEREG` when supplied. |
| `TLMPAMT` / `ALMPAMT` | Any of `ELMPTYP1YN/2YN/4YN=1`, shared `ALMPTYPYN`. |
| `TROLLAMT` / `AROLLAMT` | `EROLLOVR1=1` or `EROLLOVR2=1`, corresponding `AROLLOVR1/2`. |
| `TIRA_INC_AMT` / `AIRA_INC_AMT` | `EIRA_INC_YN=1`, `AIRA_INC_YN`. |
| `TTHR_INC_AMT` / `ATHR_INC_AMT` | `ETHR_INC_YN=1`, `ATHR_INC_YN`. |

A source annual value does not require twelve copies. Agreeing usable repetitions
produce one annual value, while missing repetitions and all flag/route evidence
remain visible. Distinct valid amounts, contradictory applicable annual receipt
routes, suppressed or invalid source observations refuse that measurement.
Missing annual repeats are not invented monthly income. Mixed allocation flags
remain mixed evidence and never become reported by taking a first/max flag.
These primitives do not resolve regularity or recipient retirement concepts.

## December households and weights

`december_households(rows, source=..., complete_membership=True,
copied_fields=(...))` constructs `(SSUID, ERESIDENCEID)` membership from **all**
December members, before target selection. The explicit completeness argument is
a caller assertion; the adapter must establish it from the full source roster.
Missing/invalid residence or duplicate person-months refuse. Non-December records
are not substituted for December membership or weights.

Every member's `WPFINWGT`, including valid zero, enters the arithmetic household
mean. Fractional values are preserved with a fixed local Decimal context: 80-digit
precision, half-even rounding, exponent bounds -999999 to 999999, and explicit
flags/traps. Annual sums use the same context. Caller rounding, traps and exponent
bounds do not affect results or receive internal flags; nonterminating means round
at that precision. Missing/invalid member weights make the requested mean unknown rather
than excluding that member. Each person's own December weight remains separately
available. `select_household_targets` selects these already measured household
objects without changing membership, measurements or weights; absent target people
refuse.

The optional copied fields are `TVEH_NUM`/`AVEH_NUM`, `THVAL_VEH`/`AHVAL_VEH`,
`THEQ_VEH`/`AHEQ_VEH` and `THDEBT_VEH`/`AHDEBT_VEH`. Every member must agree on
parsed value/class and allocation status. Equivalent numeric spellings can agree.
Missing copies yield unknown; differing copies/statuses additionally identify a
conflict. No first, maximum, mean, filled-zero or summed-copy value is substituted.

`TVEH_NUM` counts cars/trucks/vans owned during the reference period.
`THVAL_VEH` includes other vehicles such as boats and recreational vehicles. A
zero car count does not force vehicle value to zero. A flag-9 household recode stays
component-derived regardless of a subset of reported car-component flags.

The returned stock/count/flow bases retain reference 2022 and nominal source-dollar
units where relevant. A request for target year 2024 refuses with
`YEAR_BRIDGE_REQUIRED`; this module implements no transfer bridge. Household annual
income, oldest-member covariates, source-to-recipient concept mappings, scientific
fit quality/population decisions and all six consumer integrations remain separate.

## Primary definitions and verification

The code-owned master identity is the [official 2023 API dictionary](https://api.census.gov/data/2023/sipp/variables.json),
SHA-256 `b62876a0763a0b45f2c4873b055b35c264d3a67640b8a8b3590b650aa1db679e`.
The [2023 SIPP dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf)
has SHA-256 `0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f`.
The [October 2024 edition of the 2023 guide](https://www2.census.gov/programs-surveys/sipp/tech-documentation/methodology/2023_SIPP_Users_Guide_OCT24.pdf)
has SHA-256 `37a20da6ac81d5dea28f744f02b96975f846f44b53c23588abd88bea7d985d86`.

Relevant dictionary printed pages: 8 (allocation), 560/568 (IRA/Thrift), 2534 (monthly
employment), 2625 (job count), 2676 (alternate earnings), 2871 (retirement recode),
2999–3002 (lumps/rollovers), 3089/3091 (life regularity/amount). Relevant guide
printed pages: 32–33 (residences), 49 (age timing), 72–73 (additional jobs/unpaid
family work), 120–121 (retirement construction), 128–129 (earnings components and
month-length basis), 160/162 (December and mean-member weights).

`test_us_sipp_measurements.py` uses invented lexical records. It covers structural
zero disqualifiers, losses and missing codes, complete/partial years, annual-once
repetitions and routes, varying retirement, split residences, copied-value/status
conflicts, decimal and zero weights, post-measurement target selection, ownership,
and source/target-period refusal. These tests establish software behavior only;
no source prevalence, national/CD fitness, or launch certification is asserted.
