# Child-care attendance: NSECE source and population integration

Related: [Microcosm #915](https://github.com/PolicyEngine/microcosm/issues/915).

The US fiscal refresh builder can now produce three **person-level** inputs:
`childcare_attending_days_per_month`, `childcare_days_per_week`, and
`childcare_hours_per_day`. The source model covers children ages 0–12. It does
not change PolicyEngine-US defaults. Attendance, subsidy eligibility, provider
pricing, and benefit receipt are separate concepts.

The build runs `with_us_childcare_attendance_inputs` after the childcare expense
producer and before release validation. The generated release input contract
requires all three attendance columns. Licensed local source paths are explicit
build inputs; CI does not download or redistribute survey records. This PR
provides build integration and a qualified local population candidate; it does
not publish a replacement population or certify national CCDF spending.

## Source and mapping

The [2024 NSECE V1 release](https://www.childandfamilydataarchive.org/cfda/archives/cfda/studies/39466/datadocumentation)
contains household DS5 and calendar DS4 TSVs. Each contains 6,403 households.
The loader verifies the exact source hashes in `us/childcare_attendance_source.json`
before parsing. This separate `SourceStageSpec` compatibility resource preserves
the byte-frozen generation-0 source manifest. It contains declarative operations,
source pins, income bands, and price-year conventions.

| Field | Meaning |
| --- | --- |
| `HH4_METH_CASEID` | One-to-one household/calendar join |
| `HHC4_AGE_AT_USAGE_X` | Child age in months in the reference week |
| `HHC4_METH_WEIGHT_X` | Child design weight for donor draws |
| `HH4_METH_WEIGHT` | Household design weight for sibling dependence |
| `HH4_MISSING_STATUS_CC_X` | Missing, partial, or complete calendar |
| `HH4_CHCAL_R_X_Z` | 672 successive 15-minute blocks, starting Monday midnight |
| `HH4_TYPEOFCARE_AGG_X_Y` | Child/provider care type |
| `HH4_RPARENT` | Whether respondent care is parental care |
| `HH4_REGION` | Census region |
| `HH4_PARWORK_STATUS` | Work status of parents of any under-13 household child |
| `HH4_METH_QUEXVERSION` | Main, summer/typical-May, or new-school-year instrument |
| `HH4_ECON_INCOME_ANNUAL` | Published household pretax income for 2023 |
| `HHC4_NPC_HRSWEEK_TOC1..5_X` | Regular-care weekly hours for the noncalendar bridge |

Regular ECE includes provider types 1–5; type 7 is irregular care. K–8 schooling
(type 6) is excluded. Certain unpaid-care gap codes count as ECE. Respondent
care depends on `HH4_RPARENT`; school gap code 68 is classifiable as non-ECE only
at age six or older. Ambiguous codes remain unknown. A complete parental,
self-care, or school-only calendar is a measured zero donor. Missing calendars
never become observed zeros.

Attendance uses the union of classified ECE blocks. Days count days with any
ECE; hours per day equal weekly ECE hours divided by days. Monthly days use
`floor(days_per_week * 52 / 12 + 0.5)` (five days becomes 22). This represents a
typical week, not an observed month or a provider-specific schedule.

## Noncalendar reconstruction and joint transfer

Summer and new-school-year instruments have no calendars. The summer instrument
refers to a typical May week; this is **not measured summer attendance**. The
bridge preserves their published regular weekly hours and borrows days and
irregular hours jointly from ten nearest complete-calendar donors (including all distance ties), using
log regular hours, matching covariates, and survey weights. Regular-care
participation must agree. Zero regular hours does not establish zero irregular
care. Bridged rows are labeled `summary_bridge`, never `complete`.

Default matching uses age, Census region, parent work, and household income band.
The declared sparse-cell hierarchy drops region, then income, then parent work;
age is always retained. Chosen levels are recorded. Empty support or incompatible
observations fail; there is no invented full-time schedule.

The imputer draws a complete schedule jointly with child survey weights.
Observed cells, including observed zeros, constrain matching and are preserved.
Stable source person IDs keep clones identical and assignments independent of
row order. The native adapter losslessly encodes integer IDs temporarily.
A fitted mixture of independent child ranks and a shared household rank models
sibling dependence while preserving each child's conditional donor distribution.
It is fitted on youngest sibling pairs in fully observed households using
household weights and evaluated with household-separated folds.

## ASEC target harmonization

`harmonize_asec_childcare_predictors` resolves `PEPAR1` and `PEPAR2` against
`A_LINENO` within physical households. It counts measured last-week work among
parents of any under-13 household child, matching the NSECE unit. Unrelated
working adults do not become parents. Dangling parent pointers fail.

Regions derive from the shared Census state mapping. Household income is the
sum of raw `PTOTVAL`, expressed in 2023 dollars using annual CPI-U. The pinned
BuildP parent omits raw income for its 2022/2023 source cohorts. The optional
ASEC cache recovers a temporary income array from the existing pinned Census
archives, joining exact 22-digit `PERIDNUM` plus source year and checking raw
age, line number, and any already observed income. Original population columns,
including raw missingness, remain unchanged. No downloader runs inside the stage.

## Build and reproduction

Obtain the two ICPSR TSVs under the archive's terms and the pinned ASEC CSVs in
`education_assistance_source.py` (its existing fetch helper verifies them).
Run the complete local candidate and source diagnostics:

```bash
uv sync --all-packages --locked --extra us
uv run python tools/prepare_us_childcare_attendance.py \
  --household-tsv /local/39466-0005-Data.tsv \
  --calendar-tsv /local/39466-0004-Data.tsv \
  --asec-population-h5 /local/populace_us_2024.h5 \
  --population-sha256 48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e \
  --asec-source-cache /local/asec \
  --production-stage --extended-assessment \
  --inherit-outside-domain-baseline \
  --output-checkpoint /local/attendance-checkpoint.h5 \
  --output-native-h5 /local/attendance-candidate.h5 \
  --report /local/preparation.json
```

All output paths must be new. The checkpoint retains per-cell provenance,
matching levels, source and parent receipts, weights, strata, and mass history.
Native export adds only the three inputs and a receipt, reloads the result, and
verifies every original entity column, household weight, and time period.
Code hashes and environment versions accompany the aggregate preparation report.

Supply the same source inputs to the normal fiscal build using
`--childcare-attendance-household-tsv`, `--childcare-attendance-calendar-tsv`,
`--childcare-attendance-asec-cache`, and
`--childcare-attendance-inherit-outside-domain-baseline` alongside its usual flags.
Source receipts enter the fiscal source-coverage report. The frozen pool ABI
continues to describe the earlier pool simulation; attendance is supplied by this
subsequent fiscal-build stage and enforced by the final release input contract.
Existing release input
gates still apply; a build without required attendance inputs cannot substitute
an engine default for a persisted input.

Unknown values outside ages 0–12 stay null in the source model. The explicit
outside-domain export policy fills only these missing cells with the pinned
engine's existing baseline and labels them `inherited_engine_baseline_outside_age_0_12`.
This permits an attendance-only candidate without asserting observed
nonattendance for teens or disability-related older-child care. Observed older
values are preserved; unresolved under-13 values always fail export.

Compare every state model against the exact parent:

```bash
uv run python tools/validate_us_childcare_population.py \
  --parent-h5 /local/populace_us_2024.h5 \
  --parent-sha256 48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e \
  --candidate-checkpoint /local/attendance-checkpoint.h5 \
  --year 2026 --report /local/population-comparison.json
```

Both arms use fixed source ages and incomes without aging or uprating. Direct
state subsidy variables avoid conflating attendance with the separate household
aggregation issue [PolicyEngine-US #9405](https://github.com/PolicyEngine/policyengine-us/issues/9405).
Outputs describe potential modeled benefits, not caseload or spending estimates.

## Validation and limits

See [the aggregate experiment](../experiments/us-childcare-attendance/README.md).
Diagnostics include five-fold household cross-validation, instrument selection,
income/age/work/region comparisons, masked-calendar reconstruction, sibling joint
attendance, full target support, and all-state benefit comparisons. Development
used these diagnostics; they are not an untouched external acceptance sample.

Calendar selection remains unidentifiable for excluded ambiguous/partial cases.
Conditional matching assumes their schedules resemble supported children with
similar covariates. Bridged schedules are modeled, despite observed regular hours.
Source weights and good predictive means do not establish national validity.
Provider-specific schedules, true summer care, and older-child attendance need
additional evidence. Attendance alone cannot fix other missing CCDF inputs.
The reports retain `production_ready: false` to distinguish this candidate from
a calibrated, independently reviewed and published population release.

CI uses synthetic records only. Tests cover survey parsing, unknown/zero
separation, weighted joint draws, observed-cell and clone preservation, parent
links, verified income joins, noncalendar reconstruction, sibling ranks, build
orchestration, and native engine export. New runtime modules are classified in
the existing inventory and remain subject to the full source-spine AST guard.
