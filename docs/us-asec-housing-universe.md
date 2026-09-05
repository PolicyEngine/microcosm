# Source housing-unit evidence

The ASEC housing-universe attachment identifies consistent occupied housing-unit
observations without changing the person population, money, tax units or model
tenure. It retains separate interview, physical-unit, household-kind and native
tenure codes. This is evidence for a future reference comparison; the attachment
does not certify a dataset or a congressional-district denominator.

The [2023](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf),
[2024](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf)
and [2025](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
ASEC dictionaries, printed pages 6A-2 through 6A-4, define the four source
controls. The artifact reuses the pinned dictionary identities and zero-origin
evidence from the [housing-status contract](us-asec-housing-status.md). It adds
no source joins or pins. Income cohorts 2022, 2023 and 2024 identify the existing
money-source scope; these housing controls describe the following year's
interview, not annual housing histories.

## ASEC interpretation

An occupied HU requires `H_HHTYPE=1` (interview), `H_LIVQRT` in 1–7 (HU quarters)
and `HRHTYPE` in 1–8 (primary household). Consistent interview records with
other-unit quarters 8–12 and GQ household kind 9–10 are outside the HU universe.
HU quarters with GQ kind, and other-unit quarters with primary-household kind,
remain unresolved. These conservative mapping decisions do not assert that
Census cannot publish such combinations.

The artifact holds seven derived `uint8` arrays alongside immutable raw
`int64` observations and household/income-year/native-key identity:

| Array | Codes |
| --- | --- |
| `interview_scope` | 0 unavailable; 1 interview; 2 type-A noninterview; 3 type-B/C noninterview |
| `physical_unit` | 0 unavailable; 1 HU quarters; 2 other-unit quarters |
| `household_kind` | 0 native source zero or not established; 1 primary household; 2 GQ |
| `tenure_subtype` | 0 NIU or unresolved zero; 1 owned/being bought; 2 cash rent; 3 no cash rent |
| `occupied_hu` | 0 not established; 1 occupied HU; 2 outside HU |
| `hu_tenure_class` | 0 outside or unestablished HU; 1 owner; 2 renter; 3 occupied HU with unresolved tenure |
| `unresolved_reasons` | Bit mask described below |

Household kind maps native `HRHTYPE` independently of interview status. Positive
kind evidence remains visible on contradictory noninterviews and on rows with
unknown interview status; it does not establish occupied-HU applicability.

Reason bits are 1 outside interview scope, 2 unknown interview code, 4 unknown
quarters, 8 unknown household kind, 16 HU quarters/GQ-kind disagreement,
32 context conflict, 64 occupied HU with unknown tenure, and 128 other-unit
quarters/primary-kind disagreement. Bits can coexist. Context conflicts include
an interview with household kind zero, or a noninterview with positive household
kind or tenure. Noninterview is never interpreted as vacancy. Zero remains an
observed source code with unresolved origin where the existing contract says
so; it is not permission to fill missing evidence.

An otherwise consistent HU with tenure zero remains in the HU total with
unresolved tenure. Both cash and no-cash renters belong to the source renter
class, while their native subtypes remain distinct. This follows the
[ACS tenure definition](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf),
pages 44–45, for a future B25003 comparison. It does not change model tenure or
resolve other cross-survey residence differences.

## Source authority and composition

`load_authenticated_housing_universe` requires the exact
`AuthenticatedCurrentMoneySource` type, validates it before and after capture,
and reads the household observations it already authenticates. Both the legacy
and restored person-income source kinds are supported, with distinct source
identities. The artifact binds the existing source-verification identity, its
new definition and implementation, dictionary evidence, raw ordered keys and
derived codes. `classify_synthetic_housing_universe` creates explicitly synthetic
evidence that cannot be attached through the authenticated API.

`attach_housing_universe` accepts exactly `CurrentMoneyTaxUnitResult` or
`HousingStatusAttachedAsec`. It retains that chosen parent, the original tax
result and the original ready money object. The complete supported chain is
student controls → tax units → housing status → housing universe; tax units →
housing universe is also supported. Existing student controls retain their
interview-week meaning; this attachment adds no annual tax-student validation.

Seven `asec_housing_universe_` aliases extend an owned copy of the entire Frame.
Validation requires the original parent after dropping only those aliases,
including all entity tables, axes, weights, strata, metadata and mass history.
Mutating the child cannot mutate its sealed parents. Source identity, household
order, income-year bytes and raw context must match the chosen parent. No
source observations or money fields are rewritten.

The primitive artifact codec accepts only exact canonical bytes matching an
independently issued expected object. A newly checksummed forgery does not
acquire authority. Reads require a regular file with the exact expected size
and use a nonblocking descriptor before inspecting its type. There is no
candidate HDF parsing or self-authenticating receipt. Writes create a new file
exclusively.

`verify_housing_universe_rows` can compare selected or cloned household rows
against the original authenticated household IDs. It checks their exact raw
context and derived codes, including `uint8` dtype, and checks a carried
`household_source_id` when present. The caller must supply original lineage IDs;
their assembly provenance remains a separate obligation. This check issues no
whole-Frame authority for a subset. Ordinary Frame checkpoints preserve the
columns, but a checkpoint alone does not authenticate them.

## Pure ACS interpretation

`classify_acs_housing_universe(households, tenure_valid=mask)` is an unbound
semantic helper. It requires ordinary NumPy `int64` `TYPEHUGQ`, `NP` and `TEN`
columns and an explicit positional Boolean validity vector. Nullable, float,
object or Boolean source columns are refused. Valid tenure codes must be 1–4;
an invalid tenure's integer payload is an uninterpreted placeholder, including
when it happens to be zero. This helper does not read or authenticate ACS files.

`TYPEHUGQ=1` and `NP>0` establishes an occupied HU; types 2/3 are GQ and outside
the HU universe. Type 1 with no persons leaves occupancy unestablished. Native
ACS tenure 1/2 maps to owner and 3/4 to renter, preserving all four subtypes.
An occupied HU with invalid tenure remains in the HU total as unresolved.
Blank GQ tenure introduces no conflict. CPS interview and household-kind
observations are not supplied by these inputs, so their output axes remain
explicitly unobserved (zero). The header gives source-specific codebooks and
the output includes `TEN_valid` as a `uint8` validity column.

The ASEC and ACS codes therefore need their own codebooks. A shared stacked
adapter must later bind authenticated ACS lineage and preserve source and
validity masks through cloning, pruning and export. These helpers do not add
nullable graph dtypes, a complete stacked denominator, B19001 household-money
income, or a release verdict.
