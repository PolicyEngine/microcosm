# Census person columns the 2022/2023 ASEC inputs lack (#720, layer 1)

Status as of 2026-09-23, branch `us-asec-720-coverage-recodes` (stacked on
`us-spm-role-stage`, #959). Code: `microcosm.build.us_runtime.asec_census_person_columns`.

## The defect

The base pools three processed CPS ASEC inputs, pinned in
`us_runtime/asec_sources.py`:

| income year | file | SHA-256 | person rows | `NOW_*` recodes |
|---|---|---|---:|---:|
| 2022 | `census_cps_2022.h5` | `7ccca976…` | 146,133 | 2 of 18 |
| 2023 | `census_cps_2023.h5` | `cb578173…` | 144,265 | 2 of 18 |
| 2024 | `census_cps_2024.h5` | `ec36604c…` | 142,125 | 18 of 18 |

The 2022 and 2023 files were extracted with an older column list. Besides 16
`NOW_*` at-interview coverage recodes they lack `A_EXPRRP`, `PTOTVAL`,
`A_ENRLW`, `A_FTPT`, `A_FAMREL`, `A_FAMTYP` and `PECOHAB`; the 2022 file also
lacks `LKWEEKS`. `asec_pool.pool_asec_sources` concatenates the per-year
person tables, so a column only the 2024 file carries is NaN on every
2022/2023 row, and each reader treats NaN as "no" or zero:

- `cps_carried._fill_health_coverage_inputs` maps seven reported-coverage
  flags from `NOW_*` codes. They were False for every 2022/2023 person, and
  the #744 gate (`us_reported_coverage_vintage_signal_gate`) refused route A's
  release with 14 failures (seven flags times two vintages).
- `asec_pool._with_relationship_recode` found no `A_EXPRRP` for those years
  and derived one from line, spouse and parent pointers; it never produces the
  Census partner/roommate code (13) and differs from Census's own recode for
  about 14% of 2022/2023 persons.
- microunit's tax-unit construction (fed by `microcosm.frame.units`) read
  `PTOTVAL` as 0 and saw no enrollment (`A_ENRLW`, `A_FTPT`) for those years;
  `eligibility_inputs` set `is_full_time_college_student` False for all of
  them.

## The fix

The official Census archives the inputs were extracted from are already pinned
build inputs. The base stage passes them as `--asec-education-source`, and
`spm_role_source.ASEC_SPM_ROLE_SOURCES` pins each archive's SHA-256 and its
person member's name, size and SHA-256. `restore_asec_census_person_columns`
runs in `asec_pool._prepare_year_input` for each vintage. It runs straight
after the H5 is read, before the relationship recode and before any
smoke-household limit. It appends every reviewed column the H5 lacks, reads
those columns from that year's pinned member, and joins them on exact
`PERIDNUM`. It uses `spm_role_source.read_pinned_asec_person_columns`, the SPM
role stage's own reader. That reader now takes a column list, and its default
read is unchanged.

`build_us_puf_support_base.py` and `build_us_asec_pooled_source_base.py` bind
every pooled year to its Census person file. They use the
`--asec-education-source` mapping or, for an unmapped year, a fetch from the
official archive that is verified against the same pins. They resolve the
file the same way the SPM role stage does. Every base built from `--asec-h5`
inputs therefore carries the columns. The source-construction frame, the
`asec_raw_stage` checkpoint, pre-clone enrichment and the export all see them.

### Refusals (`AsecCensusPersonColumnsError`, a `ValueError`)

The restoration refuses any of the following:

- The H5 lacks `PERIDNUM`, `PH_SEQ`, `P_SEQ`, `A_LINENO` or `A_AGE`. It also
  refuses repeated column names or a repeated row index.
- The archive or member fails its pins, or the member's person count differs
  from the pin.
- The member lacks a reviewed column. A reviewed column parses as anything but
  integers (a blank, a fraction or text), or a value falls outside the column's
  reviewed Census codes.
- A `PERIDNUM` is not an exact 22-digit string, or either side repeats one.
- An H5 person has no member row, or a member person is absent from the H5.
  The join must be one-to-one and total over the complete member.
- `PH_SEQ`, `P_SEQ`, `A_LINENO` or `A_AGE` disagree after the join.
- A reviewed column the H5 already carries has missing or non-integer values,
  or differs from the member on any row. Such a column is never overwritten.
  On 2024, where the H5 carries all eleven, this check validates the join on
  real data.
- Any existing H5 column changes position, dtype or value. This condition is
  checked explicitly, not asserted.

No value is predicted, imputed or defaulted.

## Column review

Start: the 24 columns the 2026-08-23 offline fix appended
(`_buildo-runtime/inputs/asec-720/receipt_720.json`). A column is restored
only if build code reads it. Readers come from a grep of `packages/*/src` and
`tools/` on 2026-09-23. `test_every_restored_column_cites_readers_that_really_read_it`
resolves every citation and requires the cited function or constant to name
the column.

| column | restored | reader, or why not |
|---|---|---|
| `NOW_MCAID` | yes | `cps_carried._fill_health_coverage_inputs` → `has_medicaid_health_coverage_at_interview` |
| `NOW_NONM` | yes | same → `has_non_marketplace_direct_purchase_health_coverage_at_interview` |
| `NOW_CHAMPVA` | yes | same → `has_champva_health_coverage_at_interview` |
| `NOW_MIL` | yes | same → `has_tricare_health_coverage_at_interview` |
| `NOW_VACARE` | yes | same → `has_va_health_coverage_at_interview` |
| `NOW_OTHMT` | yes | same → `has_other_means_tested_health_coverage_at_interview` |
| `NOW_IHSFLG` | yes | same → `has_indian_health_service_coverage_at_interview` |
| `A_EXPRRP` | yes | `asec_pool._with_relationship_recode` (the Census recode replaces the derived fallback); `frame.units.MICROUNIT_REQUIRED_COLUMNS` → microunit `_prepare_household_people` (tax-unit head choice); `acs_transfer._person_head_feature`; `us_late_producer_registry._transfer_input_inventory` |
| `PTOTVAL` | yes | `frame.units._is_microunit_optional_column` → microunit `_precompute_tax_unit_inputs` |
| `A_ENRLW` | yes | `frame.units._is_microunit_optional_column` → microunit `_is_full_time_student` |
| `A_FTPT` | yes | same; `eligibility_inputs.derive_us_eligibility_inputs_from_manifest` (`is_full_time_college_student`) |
| `NOW_COV`, `NOW_DIR`, `NOW_MCARE`, `NOW_MRKS`, `NOW_MRKUN`, `NOW_PCHIP`, `NOW_PRIV`, `NOW_PUB` | no | No build reader. Restoring them would only widen the exported person table. A test fails if code starts reading one. |
| `NOW_CAID` | no | Read only by the native current-survey health projection (`current_survey_health_coverage`, through `current_survey_health_source`). That projection captures its own pinned ASEC person member and never reads the pooled H5 frame. A test fails if any other module names the column. |
| `A_FAMTYP`, `A_FAMREL`, `PECOHAB` | no | Read only as optional cross-checks by the SPM role derivation (`spm_role_source._OPTIONAL_RAW_CHECKS`). That derivation reads this same member itself and derives the role from it, so restoring these columns would make the check compare the member with itself. #38's partner derivation should restore them together with its reader. |
| `LKWEEKS` | no | Already restored for 2022, the only vintage without it, by `weeks_unemployed.fill_asec_2022_weeks_unemployed_source` from the same pinned `pppub23.csv`. The 2023 and 2024 inputs carry it. On the real run it equals the 8/23 file on all 146,133 rows. |

`CENSUS_TAX_ID` was never appended: the extractor derives it, and the Census
files do not carry it.

### Restorations after #720

`ASEC_CENSUS_PERSON_COLUMNS_BEYOND_720` lists restored columns that were not
in the offline fix. The review test keeps the #720 columns exactly the
offline fix's list, and requires every later column to be declared there.

| column | restored | reader |
|---|---|---|
| `A_LFSR` | yes | `immigration._assign_ssn_card_codes`: the `immigration_status` stage's worker EAD spill uses labor-force codes 1-4 at ages 16+. `immigration.US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS` lists it, and the stage refuses a person table without it. |

None of the three H5 inputs carries `A_LFSR`, so it is appended for every
vintage. Its reviewed codes, observed on 2026-09-26 in all three pinned
members after their archive and member SHA-256 checks passed, are 0, 1, 2, 3,
4 and 7. No value is missing, and code 0 is the only code for persons under 15.

| member | 0 | 1 | 2 | 3 | 4 | 7 | persons |
|---|---|---|---|---|---|---|---|
| `pppub23.csv` | 29,948 | 66,156 | 2,539 | 2,064 | 408 | 45,018 | 146,133 |
| `pppub24.csv` | 28,915 | 66,185 | 2,326 | 2,225 | 410 | 44,204 | 144,265 |
| `pppub25.csv` | 28,155 | 65,222 | 2,214 | 2,422 | 408 | 43,704 | 142,125 |

A base-stage pool built without a Census person source keeps the #720 hole
and has no `A_LFSR`, so its `immigration_status` stage refuses. It does not
fall back to another worker definition.

## Real-data evidence (2026-09-23, commit `39b8e7b63`)

The real-data run used route A's exact base inputs and flags
(`route-a/run-47976be6ce75/base-config.json`) with
`--stage source_construction` only. Its checkpoints are under
`_recovered/scratch-backup/893/overnight-20260923/asec-720-fix/`. The run took
69 s at 11.2 GB peak RSS. The inputs matched their pins: the H5 files
`7ccca976…`, `cb578173…` and `ec36604c…`, and the archives `d2e00025…`,
`cdb39cda…` and `318845a2…`.

### The #744 gate

The gate ran on `derive_us_cps_carried_inputs` of the ASEC raw-stage frame.
"Before" is route A's raw-stage checkpoint (commit `47976be6c`); "after" is
this branch.

| | 2022 | 2023 | 2024 | gate |
|---|---:|---:|---:|---|
| Medicaid reporters, before | 0 | 0 | 24,844 | **FAIL, 14 failures** |
| Medicaid reporters, after | 28,184 | 26,367 | 24,844 | **PASS, 0 failures** |
| 2026-08-23 receipt | 28,184 | 26,367 | 24,844 | pass |

The other restored flags after the fix, as 2022 / 2023 / 2024 reporters:

- non-marketplace direct purchase: 8,666 / 8,435 / 8,486
- TRICARE: 3,882 / 4,318 / 4,413
- VA: 1,271 / 1,178 / 1,270
- IHS: 687 / 651 / 685
- CHAMPVA: 260 / 335 / 368
- other means-tested: 275 / 331 / 312

There are no nulls. `has_esi` and marketplace are unchanged. Every 2024
reported-coverage count is unchanged.

### Equality with the 2026-08-23 corrected inputs

- Restoring each pinned H5 as the pool does yields the corrected files'
  columns exactly, in value, dtype and row order, for all 11 restored columns
  in both 2022 and 2023. Every pre-existing column is identical too. The
  corrected files differ only in the 13 columns listed above as not restored.
- The whole raw-stage frame equals the 8/23 corrected raw-stage frame on
  every entity table: family, household, marital unit, person, SPM unit and
  tax unit, including the 226,102 tax units. It also matches on all 176
  shared person columns except exactly the 12 unrestored columns the 8/23
  frame also carried. `LKWEEKS` is identical.
- On 2024 the H5 carries all eleven columns, and all eleven were verified
  equal to `pppub25.csv` on 142,125 persons.

### Structural effect (intended; flag for downstream)

| | before | after |
|---|---:|---:|
| tax units, all vintages | 231,007 | 226,102 |
| tax units 2022 / 2023 / 2024 | 78,221 / 78,089 / 74,697 | 75,797 / 75,608 / 74,697 |
| tax units per person 2022 / 2023 / 2024 | 0.535 / 0.541 / 0.526 | 0.519 / 0.524 / 0.526 |
| 2022 / 2023 persons whose `A_EXPRRP` changes | — | 20,595 / 20,561 |
| `A_EXPRRP == 13` persons 2022 / 2023 | 0 / 0 | 4,312 / 4,387 |

Household, family, SPM-unit, marital-unit and person counts are unchanged, as
is every 2024 value. The 8/23 run produced the same 226,102 tax units. The
older vintages' tax-unit rate moves toward the 2024 rate, which always used
the Census recode.

## Coverage

- Covered: every `tools/build_us_puf_support_base.py` run that pools
  `--asec-h5` inputs, in both staged and monolithic runs and with a support
  spine spec, and `tools/build_us_asec_pooled_source_base.py`.
- Covered only after regeneration: `tools/build_us_multispine_pool.py` reads
  the SHA-pinned `asec_raw_stage.checkpoint.h5` that the base's source
  construction writes. A checkpoint built before this change keeps the hole,
  and the #744 gate refuses a release built from it.
- Not affected: `--base-h5` builds, which start from an existing frame, and the
  enrichment and qualification lanes, which start from published parents
  (`build_us_spm_role_enrichment.py`,
  `build_us_acs_donor_receipt_qualification.py`). ACS-spine rows come from ACS
  PUMS, not these H5 files.
- Unchanged: the #744 gate, poverty and SPM outcomes (neither a target nor a
  gate), and the reviewed `is_unmarried_partner_of_household_head` exclusion.
  That exclusion describes the inputs and remains true of them. With the
  Census recode restored, `A_EXPRRP == 13` now appears in every vintage, but
  it still conflates partners and roommates, so the exclusion's conclusion
  stands.
