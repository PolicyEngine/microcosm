# Schema manifest: the 2024 county file (County Files proof of concept)

Manifest kind `microcosm_us_release_schema_manifest`, schema version 1. Dtypes are as pandas reads each file; null counts count missing values in every row.

## Notes

These notes cover what the tables below do not show. Apart from this section, the file is the manifest tool's output, unchanged, and `schema_manifest.json` is as the tool wrote it.

### Empty strings in six household columns

`block_geoid`, `tract_geoid`, `place_fips`, `sldu`, `sldl` and `cbsa_code` hold an empty string, not a null, for all 1,529,668 ACS households. That is why the Nulls column shows 0 for them. The ACS public-use file locates a household only to its PUMA, and the build does not place ACS households below the PUMA, so the release fills these columns with `''` on ACS rows. Read `''` as missing.

The 57,240 donor households take all six from the census block each was placed in. Some donors are blank as well:

| Column | Blank donor households | Why |
| --- | ---: | --- |
| `block_geoid`, `tract_geoid`, `sldu` | 0 | |
| `place_fips` | 15,382 | The block is in no incorporated place or census-designated place. |
| `sldl` | 1,435 | Nebraska (884) and DC (551) have no lower-house districts. |
| `cbsa_code` | 5,343 | Outside Connecticut (4,435), the household's county is in no metropolitan or micropolitan area. All 908 Connecticut donor households are blank (below). |

In Connecticut, a blank `cbsa_code` does not mean the household is outside a metropolitan or micropolitan area. The donor households got their geography before 30 September 2026, from code that looked a block's CBSA up by its 2020 county. The 2023 delineations list Connecticut by its nine planning regions instead, so no Connecticut block found a CBSA. The build has used the planning regions since that date, but the donor households keep their earlier geography.

### Which columns the model reads

policyengine-us 2.2.1 sets a column as an input only if its name is a model variable. It loads the other columns and ignores them; these are the 254 ignored columns the README and the report mention. In this file the split follows letter case: the model reads every lowercase column except 40 build identifiers and derived columns, and no uppercase column.

| Kind | Columns | Which |
| --- | ---: | --- |
| Read by the model | 198 | 153 person, 19 household, 14 tax_unit, 10 spm_unit, 1 family, 1 marital_unit |
| Raw CPS ASEC fields | 162 | Uppercase, under their Census names: 161 person and household `H_TENURE`. Null on ACS rows, except eight (below). |
| Raw ACS 2024 PUMS fields | 51 | Uppercase; null on donor rows. `ST` is the Census `STATE` and `ACS_MIL` the Census `MIL`. Listed below. |
| Build identifiers | 29 | For each of the six tables, `{table}_source_id`, `{table}_support_channel`, `{table}_support_clone_index` and `{table}_spine`; plus `source_year`, `source_household_id`, `source_person_id`, `source_row_id` and `CENSUS_TAX_ID` (see below). `household_spine` is `acs_2024_1yr` for ACS households and `asec_puf` for donor households. |
| Derived columns | 12 | `county_fips_2023`, `puma_geoid`, `acs_social_security_income`, `acs_retirement_income`, `acs_interest_dividend_rental_income`, `acs_monthly_contract_rent`, `acs_monthly_gross_rent`, `acs_annual_property_tax`, `tax_unit_role_input`, `filing_status_input`, `parent_1_id`, `parent_2_id` |

The 51 ACS fields are, for persons, `SPORDER`, `ADJINC`, `PWGTP`, `AGEP`, `CIT`, `COW`, `DDRS`, `DEAR`, `DEYE`, `DOUT`, `DPHY`, `DREM`, `HINS3`, `HINS4`, `HINS5`, `HINS6`, `HINS7`, `INTP`, `MAR`, `ACS_MIL`, `OIP`, `PAP`, `RELSHIPP`, `RETP`, `SCHG`, `SEMP`, `SEX`, `SSIP`, `SSP`, `WAGP`, `WKHP`, `WKL`, `WKWN`, `YOEP`, `ESR`, `INDP`, `OCCP`, `POBP` and `FWKHP`, and for households `SERIALNO`, `ST`, `PUMA`, `NP`, `ADJHSG`, `TEN`, `RNTP`, `GRNTP`, `TAXAMT`, `TYPEHUGQ`, `FS` and `VEH`.

Four more points:

- Eight CPS ASEC names have no nulls: `PH_SEQ`, `A_LINENO`, `A_AGE`, `A_MARITL`, `A_SPOUSE`, `PEPAR1`, `PEPAR2` and `A_EXPRRP`. On ACS rows the build fills them from ACS fields so that it can form tax units.
- The model reads raw survey values only under its own names. On ACS rows, for example, `employment_income_before_lsr` is `WAGP` times `ADJINC`.
- The model reads `county_fips` (2020 counties). It does not read `county_fips_2023`.
- Two CPS ASEC names do not always hold the Census value. `A_EXPRRP` (relationship to the reference person) is the Census value only for donors from income year 2024; for income years 2022 and 2023, the donor build derived it from line, spouse and parent numbers. `TAX_ID` may hold the tax unit that PolicyEngine's CPS extractor built: the extractor copies the Census `TAX_ID` into `CENSUS_TAX_ID` and then overwrites `TAX_ID`. `CENSUS_TAX_ID` is filled only for income-year 2024 donors. That `TAX_ID` was overwritten in this file is an inference from the extractor's code; we have not compared values.

### Removed columns, the rename and the dtype changes

These compare the file with the sample column list sent with the County Files agreement.

- `has_marketplace_health_coverage` and `weeks_worked`: policyengine-us 2.2.1 has a formula for each, and the release leaves out any column the model can compute with a formula in some year. It keeps id columns, and inputs that policyengine-us says the data must supply even though it has a fallback formula, such as `is_spm_independent_minor_role`. The formula for `has_marketplace_health_coverage` returns `has_marketplace_health_coverage_at_interview`, which the file keeps. The formula for `weeks_worked` applies from 2025.
- `medicare_part_b_premiums`, a stored Medicare Part B premium: policyengine-us 2.2.1 has no variable by this name and computes the Part B premium itself (`medicare_part_b_premium`), so the release no longer stores it. It held values for the 166,321 donor persons only.
- 128 dtype changes are int64 to float64: 127 person columns and household `H_TENURE`. All are CPS ASEC fields with no value on ACS rows. An int64 column cannot hold a missing value, so when the build stacks donor and ACS rows, pandas stores these columns as float64. Only the storage type changes. The sample file had no ACS rows.
- The other six are columns that are null on ACS rows, where the release fills the model's default, 0, and stores the model's type, int64 or float64: household `household_vehicles_owned` (int32 to int64) and `household_vehicles_value` (float32 to float64); person `cps_race` (int16 to int64), `fsla_overtime_premium` (float32 to float64) and `treasury_tipped_occupation_code` (int16 to int64); and tax_unit `second_home_mortgage_origination_year` (float64 to int64).

## Files

| File | Format | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| populace_us_2024_acs_local.h5 | hdf5 | 10,637,811,669 | 82314a3b5a6a95762faf6eb4f5b499ce56608ad6a39a27b75e11c75b4945bd3c |

## Metadata

| File | Item | Value |
| --- | --- | --- |
| populace_us_2024_acs_local.h5 | key `_populace_staging_metadata` | {"artifact_kind": "calibrated_local_area_artifact", "entity_hdf_format": "fixed_nullable", "household_weight_kind": "calibrated"} |
| populace_us_2024_acs_local.h5 | key `_time_period` | 2024 |

## Tables

| Table | File | Storage | Rows | Columns |
| --- | --- | --- | ---: | ---: |
| person | populace_us_2024_acs_local.h5 | fixed | 3,587,263 | 368 |
| household | populace_us_2024_acs_local.h5 | fixed | 1,586,908 | 41 |
| tax_unit | populace_us_2024_acs_local.h5 | fixed | 2,175,528 | 19 |
| spm_unit | populace_us_2024_acs_local.h5 | fixed | 1,589,568 | 14 |
| family | populace_us_2024_acs_local.h5 | fixed | 1,594,874 | 5 |
| marital_unit | populace_us_2024_acs_local.h5 | fixed | 2,867,105 | 5 |

### `person`

| Column | Dtype | Logical dtype | Nulls |
| --- | --- | --- | ---: |
| `PERIDNUM` | str |  | 3,420,942 |
| `PH_SEQ` | int64 |  | 0 |
| `P_SEQ` | float64 |  | 3,420,942 |
| `A_LINENO` | int64 |  | 0 |
| `PF_SEQ` | float64 |  | 3,420,942 |
| `PEAFEVER` | float64 |  | 3,420,942 |
| `PENATVTY` | float64 |  | 3,420,942 |
| `PEINUSYR` | float64 |  | 3,420,942 |
| `PEPAR1` | int64 |  | 0 |
| `PEPAR2` | int64 |  | 0 |
| `PRDTHSP` | float64 |  | 3,420,942 |
| `PRDTRACE` | float64 |  | 3,420,942 |
| `PRCITSHP` | float64 |  | 3,420,942 |
| `PECOHAB` | float64 |  | 3,530,060 |
| `PEIO1COW` | float64 |  | 3,420,942 |
| `PEDISEAR` | float64 |  | 3,420,942 |
| `PEDISEYE` | float64 |  | 3,420,942 |
| `PEDISREM` | float64 |  | 3,420,942 |
| `PEDISPHY` | float64 |  | 3,420,942 |
| `PEDISDRS` | float64 |  | 3,420,942 |
| `PEDISOUT` | float64 |  | 3,420,942 |
| `A_AGE` | int64 |  | 0 |
| `A_SEX` | float64 |  | 3,420,942 |
| `A_ENRLW` | float64 |  | 3,530,060 |
| `A_MJOCC` | float64 |  | 3,420,942 |
| `A_HSCOL` | float64 |  | 3,420,942 |
| `A_FTPT` | float64 |  | 3,530,060 |
| `A_FNLWGT` | float64 |  | 3,420,942 |
| `A_FAMREL` | float64 |  | 3,530,060 |
| `A_FAMTYP` | float64 |  | 3,530,060 |
| `A_SPOUSE` | int64 |  | 0 |
| `A_MARITL` | int64 |  | 0 |
| `A_HRS1` | float64 |  | 3,420,942 |
| `A_EXPRRP` | int64 |  | 0 |
| `ACTC_CRD` | float64 |  | 3,420,942 |
| `AGI` | float64 |  | 3,420,942 |
| `ANN_VAL` | float64 |  | 3,420,942 |
| `CAID` | float64 |  | 3,420,942 |
| `CAP_VAL` | float64 |  | 3,420,942 |
| `CHAMPVA` | float64 |  | 3,420,942 |
| `CHSP_VAL` | float64 |  | 3,420,942 |
| `CSP_VAL` | float64 |  | 3,420,942 |
| `CTC_CRD` | float64 |  | 3,420,942 |
| `DIS_SC1` | float64 |  | 3,420,942 |
| `DIS_SC2` | float64 |  | 3,420,942 |
| `DIS_VAL1` | float64 |  | 3,420,942 |
| `DIS_VAL2` | float64 |  | 3,420,942 |
| `DIV_VAL` | float64 |  | 3,420,942 |
| `DST_SC1` | float64 |  | 3,420,942 |
| `DST_SC2` | float64 |  | 3,420,942 |
| `DST_SC1_YNG` | float64 |  | 3,420,942 |
| `DST_SC2_YNG` | float64 |  | 3,420,942 |
| `DST_VAL1` | float64 |  | 3,420,942 |
| `DST_VAL2` | float64 |  | 3,420,942 |
| `DST_VAL1_YNG` | float64 |  | 3,420,942 |
| `DST_VAL2_YNG` | float64 |  | 3,420,942 |
| `EIT_CRED` | float64 |  | 3,420,942 |
| `FEDTAX_AC` | float64 |  | 3,420,942 |
| `FEDTAX_BC` | float64 |  | 3,420,942 |
| `FRSE_VAL` | float64 |  | 3,420,942 |
| `HRSWK` | float64 |  | 3,420,942 |
| `IHSFLG` | float64 |  | 3,420,942 |
| `INT_VAL` | float64 |  | 3,420,942 |
| `I_ERNVAL` | float64 |  | 3,420,942 |
| `I_SEVAL` | float64 |  | 3,420,942 |
| `LKWEEKS` | float64 |  | 3,420,942 |
| `MARG_TAX` | float64 |  | 3,420,942 |
| `MCARE` | float64 |  | 3,420,942 |
| `MIL` | float64 |  | 3,420,942 |
| `MOOP` | float64 |  | 3,420,942 |
| `NOW_CAID` | float64 |  | 3,530,060 |
| `NOW_CHAMPVA` | float64 |  | 3,530,060 |
| `NOW_COV` | float64 |  | 3,530,060 |
| `NOW_DIR` | float64 |  | 3,530,060 |
| `NOW_GRP` | float64 |  | 3,420,942 |
| `NOW_IHSFLG` | float64 |  | 3,530,060 |
| `NOW_MCAID` | float64 |  | 3,530,060 |
| `NOW_MCARE` | float64 |  | 3,530,060 |
| `NOW_MIL` | float64 |  | 3,530,060 |
| `NOW_MRK` | float64 |  | 3,420,942 |
| `NOW_MRKS` | float64 |  | 3,530,060 |
| `NOW_MRKUN` | float64 |  | 3,530,060 |
| `NOW_NONM` | float64 |  | 3,530,060 |
| `NOW_OTHMT` | float64 |  | 3,530,060 |
| `NOW_PCHIP` | float64 |  | 3,530,060 |
| `NOW_PRIV` | float64 |  | 3,530,060 |
| `NOW_PUB` | float64 |  | 3,530,060 |
| `NOW_VACARE` | float64 |  | 3,530,060 |
| `OI_OFF` | float64 |  | 3,420,942 |
| `OI_VAL` | float64 |  | 3,420,942 |
| `PAW_VAL` | float64 |  | 3,420,942 |
| `PEMCPREM` | float64 |  | 3,420,942 |
| `PEN_SC1` | float64 |  | 3,420,942 |
| `PEN_SC2` | float64 |  | 3,420,942 |
| `PHIP_VAL` | float64 |  | 3,420,942 |
| `PMED_VAL` | float64 |  | 3,420,942 |
| `PNSN_VAL` | float64 |  | 3,420,942 |
| `POCCU2` | float64 |  | 3,420,942 |
| `POTC_VAL` | float64 |  | 3,420,942 |
| `PTOTVAL` | float64 |  | 3,530,060 |
| `RESNSS1` | float64 |  | 3,420,942 |
| `RESNSS2` | float64 |  | 3,420,942 |
| `RESNSSI1` | float64 |  | 3,420,942 |
| `RESNSSI2` | float64 |  | 3,420,942 |
| `RETCB_VAL` | float64 |  | 3,420,942 |
| `RNT_VAL` | float64 |  | 3,420,942 |
| `SEMP_VAL` | float64 |  | 3,420,942 |
| `SPM_ACTC` | float64 |  | 3,420,942 |
| `SPM_BBSUBVAL` | float64 |  | 3,420,942 |
| `SPM_CAPHOUSESUB` | float64 |  | 3,420,942 |
| `SPM_CAPWKCCXPNS` | float64 |  | 3,420,942 |
| `SPM_CHILDCAREXPNS` | float64 |  | 3,420,942 |
| `SPM_CHILDSUPPD` | float64 |  | 3,420,942 |
| `SPM_EITC` | float64 |  | 3,420,942 |
| `SPM_ENGVAL` | float64 |  | 3,420,942 |
| `SPM_EQUIVSCALE` | float64 |  | 3,420,942 |
| `SPM_FAMTYPE` | float64 |  | 3,420,942 |
| `SPM_FEDTAX` | float64 |  | 3,420,942 |
| `SPM_FEDTAXBC` | float64 |  | 3,420,942 |
| `SPM_FICA` | float64 |  | 3,420,942 |
| `SPM_GEOADJ` | float64 |  | 3,420,942 |
| `SPM_HAGE` | float64 |  | 3,420,942 |
| `SPM_HHISP` | float64 |  | 3,420,942 |
| `SPM_HMARITALSTATUS` | float64 |  | 3,420,942 |
| `SPM_HRACE` | float64 |  | 3,420,942 |
| `SPM_ID` | float64 |  | 3,420,942 |
| `SPM_MEDXPNS` | float64 |  | 3,420,942 |
| `SPM_NUMADULTS` | float64 |  | 3,420,942 |
| `SPM_NUMKIDS` | float64 |  | 3,420,942 |
| `SPM_NUMPER` | float64 |  | 3,420,942 |
| `SPM_POOR` | float64 |  | 3,420,942 |
| `SPM_POVTHRESHOLD` | float64 |  | 3,420,942 |
| `SPM_RESOURCES` | float64 |  | 3,420,942 |
| `SPM_SCHLUNCH` | float64 |  | 3,420,942 |
| `SPM_SNAPSUB` | float64 |  | 3,420,942 |
| `SPM_STTAX` | float64 |  | 3,420,942 |
| `SPM_TENMORTSTATUS` | float64 |  | 3,420,942 |
| `SPM_TOTVAL` | float64 |  | 3,420,942 |
| `SPM_WCOHABIT` | float64 |  | 3,420,942 |
| `SPM_WEIGHT` | float64 |  | 3,420,942 |
| `SPM_WFOSTER22` | float64 |  | 3,420,942 |
| `SPM_WICVAL` | float64 |  | 3,420,942 |
| `SPM_WKXPNS` | float64 |  | 3,420,942 |
| `SPM_WNEWHEAD` | float64 |  | 3,420,942 |
| `SPM_WNEWPARENT` | float64 |  | 3,420,942 |
| `SPM_WUI_LT15` | float64 |  | 3,420,942 |
| `SSI_VAL` | float64 |  | 3,420,942 |
| `SSI_YN` | float64 |  | 3,420,942 |
| `SS_VAL` | float64 |  | 3,420,942 |
| `SS_YN` | float64 |  | 3,420,942 |
| `STATETAX_A` | float64 |  | 3,420,942 |
| `STATETAX_B` | float64 |  | 3,420,942 |
| `TAX_INC` | float64 |  | 3,420,942 |
| `UC_VAL` | float64 |  | 3,420,942 |
| `VET_VAL` | float64 |  | 3,420,942 |
| `WC_VAL` | float64 |  | 3,420,942 |
| `WICYN` | float64 |  | 3,420,942 |
| `WKSWORK` | float64 |  | 3,420,942 |
| `WSAL_VAL` | float64 |  | 3,420,942 |
| `TAX_ID` | float64 |  | 3,420,942 |
| `PEIOOCC` | float64 |  | 3,420,942 |
| `CENSUS_TAX_ID` | float64 |  | 3,530,060 |
| `source_year` | int64 |  | 0 |
| `source_household_id` | int64 |  | 0 |
| `source_person_id` | str |  | 0 |
| `source_row_id` | int64 |  | 0 |
| `person_tax_unit_id` | int64 |  | 0 |
| `tax_unit_role_input` | str |  | 0 |
| `is_related_to_head_or_spouse` | bool |  | 0 |
| `person_spm_unit_id` | int64 |  | 0 |
| `person_family_id` | int64 |  | 0 |
| `person_marital_unit_id` | int64 |  | 0 |
| `person_household_id` | int64 |  | 0 |
| `person_id` | int64 |  | 0 |
| `age` | float64 |  | 0 |
| `is_female` | bool |  | 0 |
| `employment_income_before_lsr` | float64 |  | 0 |
| `self_employment_income_before_lsr` | float64 |  | 0 |
| `taxable_interest_income` | float64 |  | 0 |
| `qualified_dividend_income` | float64 |  | 0 |
| `non_qualified_dividend_income` | float64 |  | 0 |
| `long_term_capital_gains_before_response` | float64 |  | 0 |
| `short_term_capital_gains` | float64 |  | 0 |
| `social_security_retirement` | float64 |  | 0 |
| `social_security_disability` | float64 |  | 0 |
| `social_security_survivors` | float64 |  | 0 |
| `social_security_dependents` | float64 |  | 0 |
| `taxable_private_pension_income` | float64 |  | 0 |
| `tax_exempt_private_pension_income` | float64 |  | 0 |
| `taxable_ira_distributions` | float64 |  | 0 |
| `alimony_income` | float64 |  | 0 |
| `miscellaneous_income` | float64 |  | 0 |
| `rental_income` | float64 |  | 0 |
| `farm_operations_income` | float64 |  | 0 |
| `unemployment_compensation` | float64 |  | 0 |
| `health_insurance_premiums_without_medicare_part_b` | float64 |  | 0 |
| `other_medical_expenses` | float64 |  | 0 |
| `over_the_counter_health_expenses` | float64 |  | 0 |
| `has_marketplace_health_coverage_at_interview` | bool |  | 0 |
| `has_non_marketplace_direct_purchase_health_coverage_at_interview` | bool |  | 0 |
| `has_medicaid_health_coverage_at_interview` | bool |  | 0 |
| `has_esi` | bool |  | 0 |
| `has_champva_health_coverage_at_interview` | bool |  | 0 |
| `has_tricare_health_coverage_at_interview` | bool |  | 0 |
| `has_va_health_coverage_at_interview` | bool |  | 0 |
| `has_other_means_tested_health_coverage_at_interview` | bool |  | 0 |
| `has_indian_health_service_coverage_at_interview` | bool |  | 0 |
| `self_employment_income_last_year` | float64 |  | 0 |
| `previous_year_income_available` | bool |  | 0 |
| `is_household_head` | bool |  | 0 |
| `is_separated` | bool |  | 0 |
| `is_surviving_spouse` | bool |  | 0 |
| `takes_up_medicare_if_eligible` | bool |  | 0 |
| `pre_subsidy_rent` | float64 |  | 0 |
| `is_disabled` | bool |  | 0 |
| `is_blind` | bool |  | 0 |
| `is_full_time_college_student` | bool |  | 0 |
| `own_children_in_household` | float64 |  | 0 |
| `veterans_benefits` | float64 |  | 0 |
| `is_pregnant` | bool |  | 0 |
| `child_support_received` | float64 |  | 0 |
| `child_support_expense` | float64 |  | 0 |
| `disability_benefits` | float64 |  | 0 |
| `workers_compensation` | float64 |  | 0 |
| `weeks_unemployed` | float64 |  | 0 |
| `traditional_401k_contributions_desired` | float64 |  | 0 |
| `roth_401k_contributions_desired` | float64 |  | 0 |
| `traditional_ira_contributions_desired` | float64 |  | 0 |
| `roth_ira_contributions_desired` | float64 |  | 0 |
| `self_employed_pension_contributions_desired` | float64 |  | 0 |
| `taxable_401k_distributions` | float64 |  | 0 |
| `taxable_403b_distributions` | float64 |  | 0 |
| `tax_exempt_ira_distributions` | float64 |  | 0 |
| `keogh_distributions` | float64 |  | 0 |
| `taxable_sep_distributions` | float64 |  | 0 |
| `ssn_card_type` | str |  | 0 |
| `immigration_status_str` | str |  | 0 |
| `person_source_id` | int64 |  | 0 |
| `person_support_channel` | str |  | 0 |
| `person_support_clone_index` | int64 |  | 0 |
| `tax_exempt_interest_income` | float64 |  | 0 |
| `long_term_capital_gains_on_collectibles` | float64 |  | 0 |
| `non_sch_d_capital_gains` | float64 |  | 0 |
| `alimony_expense` | float64 |  | 0 |
| `salt_refund_income` | float64 |  | 0 |
| `charitable_cash_donations` | float64 |  | 0 |
| `charitable_non_cash_donations` | float64 |  | 0 |
| `real_estate_taxes` | float64 |  | 0 |
| `home_mortgage_interest` | float64 |  | 0 |
| `investment_interest_expense` | float64 |  | 0 |
| `investment_income_elected_form_4952` | float64 |  | 0 |
| `student_loan_interest` | float64 |  | 0 |
| `educator_expense` | float64 |  | 0 |
| `qualified_tuition_expenses` | float64 |  | 0 |
| `casualty_loss` | float64 |  | 0 |
| `unreimbursed_business_employee_expenses` | float64 |  | 0 |
| `estate_income` | float64 |  | 0 |
| `farm_income` | float64 |  | 0 |
| `farm_rent_income` | float64 |  | 0 |
| `partnership_income` | float64 |  | 0 |
| `s_corp_income` | float64 |  | 0 |
| `partnership_self_employment_net_earnings` | float64 |  | 0 |
| `estate_income_would_be_qualified` | bool |  | 0 |
| `farm_operations_income_would_be_qualified` | bool |  | 0 |
| `farm_rent_income_would_be_qualified` | bool |  | 0 |
| `partnership_s_corp_income_would_be_qualified` | bool |  | 0 |
| `rental_income_would_be_qualified` | bool |  | 0 |
| `self_employment_income_would_be_qualified` | bool |  | 0 |
| `sstb_self_employment_income_would_be_qualified` | bool |  | 0 |
| `business_is_sstb` | bool |  | 0 |
| `qualified_bdc_income` | float64 |  | 0 |
| `qualified_reit_and_ptp_income` | float64 |  | 0 |
| `sstb_self_employment_income_before_lsr` | float64 |  | 0 |
| `sstb_unadjusted_basis_qualified_property` | float64 |  | 0 |
| `sstb_w2_wages_from_qualified_business` | float64 |  | 0 |
| `unadjusted_basis_qualified_property` | float64 |  | 0 |
| `w2_wages_from_qualified_business` | float64 |  | 0 |
| `schedule_d_capital_gain_distributions` | float64 |  | 0 |
| `is_incapable_of_self_care` | bool |  | 0 |
| `pre_subsidy_care_expenses` | float64 |  | 0 |
| `educational_assistance` | float64 |  | 0 |
| `is_pursuing_credential_for_american_opportunity_credit` | bool |  | 0 |
| `attends_eligible_educational_institution_for_american_opportunity_credit` | bool |  | 0 |
| `is_enrolled_at_least_half_time_for_american_opportunity_credit` | bool |  | 0 |
| `has_american_opportunity_credit_1098_t_or_exception` | bool |  | 0 |
| `has_american_opportunity_credit_institution_ein` | bool |  | 0 |
| `weekly_hours_worked_before_lsr` | float64 |  | 0 |
| `hours_worked_last_week` | float64 |  | 0 |
| `is_snap_abawd_discretionary_exempt` | bool |  | 0 |
| `bank_account_assets` | float64 |  | 0 |
| `stock_assets` | float64 |  | 0 |
| `bond_assets` | float64 |  | 0 |
| `meets_ssi_disability_criteria` | bool |  | 0 |
| `takes_up_head_start_if_eligible` | bool |  | 0 |
| `takes_up_ssi_if_eligible` | bool |  | 0 |
| `tip_income` | float64 |  | 0 |
| `treasury_tipped_occupation_code` | int64 |  | 0 |
| `cps_race` | int64 |  | 0 |
| `is_hispanic` | bool |  | 0 |
| `detailed_occupation_recode` | int16 |  | 0 |
| `has_never_worked` | bool |  | 0 |
| `is_military` | bool |  | 0 |
| `is_computer_scientist` | bool |  | 0 |
| `is_farmer_fisher` | bool |  | 0 |
| `is_executive_administrative_professional` | bool |  | 0 |
| `hourly_wage` | float64 |  | 0 |
| `is_paid_hourly` | bool |  | 0 |
| `is_union_member_or_covered` | bool |  | 0 |
| `fsla_overtime_premium` | float64 |  | 0 |
| `takes_up_medicaid_if_eligible` | bool |  | 0 |
| `other_health_insurance_premiums` | float64 |  | 0 |
| `health_insurance_premiums` | float64 |  | 0 |
| `is_self_employed` | bool |  | 0 |
| `is_spm_independent_minor_role` | bool |  | 0 |
| `receives_wic` | bool |  | 0 |
| `detailed_industry_recode` | int16 |  | 0 |
| `major_industry_recode` | int16 |  | 0 |
| `worked_last_year` | bool |  | 0 |
| `strike_benefits` | float64 |  | 0 |
| `passive_partnership_s_corp_income` | float64 |  | 0 |
| `person_spine` | str |  | 0 |
| `SPORDER` | float64 |  | 166,321 |
| `ADJINC` | float64 |  | 166,321 |
| `PWGTP` | float64 |  | 166,321 |
| `AGEP` | float64 |  | 166,321 |
| `CIT` | float64 |  | 166,321 |
| `COW` | float64 |  | 1,542,526 |
| `DDRS` | float64 |  | 317,477 |
| `DEAR` | float64 |  | 166,321 |
| `DEYE` | float64 |  | 166,321 |
| `DOUT` | float64 |  | 674,473 |
| `DPHY` | float64 |  | 317,477 |
| `DREM` | float64 |  | 317,477 |
| `HINS3` | float64 |  | 166,321 |
| `HINS4` | float64 |  | 166,321 |
| `HINS5` | float64 |  | 166,321 |
| `HINS6` | float64 |  | 166,321 |
| `HINS7` | float64 |  | 166,321 |
| `INTP` | float64 |  | 674,473 |
| `MAR` | float64 |  | 166,321 |
| `ACS_MIL` | float64 |  | 755,441 |
| `OIP` | float64 |  | 674,473 |
| `PAP` | float64 |  | 674,473 |
| `RELSHIPP` | float64 |  | 166,321 |
| `RETP` | float64 |  | 674,473 |
| `SCHG` | float64 |  | 2,850,008 |
| `SEMP` | float64 |  | 674,473 |
| `SEX` | float64 |  | 166,321 |
| `SSIP` | float64 |  | 674,473 |
| `SSP` | float64 |  | 674,473 |
| `WAGP` | float64 |  | 674,473 |
| `WKHP` | float64 |  | 1,820,131 |
| `WKL` | float64 |  | 714,273 |
| `WKWN` | float64 |  | 1,820,131 |
| `YOEP` | float64 |  | 3,104,761 |
| `ESR` | float64 |  | 714,273 |
| `INDP` | float64 |  | 1,542,526 |
| `OCCP` | float64 |  | 1,542,526 |
| `POBP` | float64 |  | 166,321 |
| `FWKHP` | float64 |  | 166,321 |
| `ssi_reported` | float64 |  | 0 |
| `acs_social_security_income` | float64 |  | 674,473 |
| `acs_retirement_income` | float64 |  | 674,473 |
| `acs_interest_dividend_rental_income` | float64 |  | 674,473 |
| `parent_1_id` | float64 |  | 166,321 |
| `parent_2_id` | float64 |  | 166,321 |
| `years_since_us_entry` | float64 |  | 0 |
| `takes_up_wic_if_eligible` | bool |  | 0 |

### `household`

| Column | Dtype | Logical dtype | Nulls |
| --- | --- | --- | ---: |
| `household_id` | int64 |  | 0 |
| `state_fips` | int64 |  | 0 |
| `H_TENURE` | float64 |  | 1,529,668 |
| `tenure_type` | str |  | 0 |
| `household_source_id` | int64 |  | 0 |
| `household_support_channel` | str |  | 0 |
| `household_support_clone_index` | int64 |  | 0 |
| `congressional_district_geoid` | int64 |  | 0 |
| `block_geoid` | str |  | 0 |
| `tract_geoid` | str |  | 0 |
| `county_fips` | str |  | 0 |
| `place_fips` | str |  | 0 |
| `sldu` | str |  | 0 |
| `sldl` | str |  | 0 |
| `cbsa_code` | str |  | 0 |
| `net_worth` | float64 |  | 0 |
| `auto_loan_balance` | float64 |  | 0 |
| `auto_loan_interest` | float64 |  | 0 |
| `qualified_passenger_vehicle_loan_interest` | float64 |  | 0 |
| `household_vehicles_owned` | int64 |  | 0 |
| `household_vehicles_value` | float64 |  | 0 |
| `household_spine` | str |  | 0 |
| `SERIALNO` | str |  | 57,240 |
| `ST` | str |  | 57,240 |
| `PUMA` | str |  | 57,240 |
| `puma_geoid` | str |  | 57,240 |
| `puma` | str |  | 0 |
| `NP` | float64 |  | 57,240 |
| `ADJHSG` | float64 |  | 57,240 |
| `TEN` | float64 |  | 238,500 |
| `RNTP` | float64 |  | 1,250,798 |
| `GRNTP` | float64 |  | 1,250,798 |
| `TAXAMT` | float64 |  | 598,224 |
| `TYPEHUGQ` | float64 |  | 57,240 |
| `FS` | float64 |  | 57,240 |
| `VEH` | float64 |  | 238,500 |
| `acs_monthly_contract_rent` | float64 |  | 1,250,798 |
| `acs_monthly_gross_rent` | float64 |  | 1,250,798 |
| `acs_annual_property_tax` | float64 |  | 598,224 |
| `county_fips_2023` | str |  | 0 |
| `household_weight` | float64 |  | 0 |

### `tax_unit`

| Column | Dtype | Logical dtype | Nulls |
| --- | --- | --- | ---: |
| `tax_unit_id` | int64 |  | 0 |
| `filing_status_input` | str |  | 0 |
| `tax_unit_source_id` | int64 |  | 0 |
| `tax_unit_support_channel` | str |  | 0 |
| `tax_unit_support_clone_index` | int64 |  | 0 |
| `domestic_production_ald` | float64 |  | 0 |
| `unrecaptured_section_1250_gain` | float64 |  | 0 |
| `first_home_mortgage_balance` | float64 |  | 0 |
| `second_home_mortgage_balance` | float64 |  | 0 |
| `first_home_mortgage_interest` | float64 |  | 0 |
| `second_home_mortgage_interest` | float64 |  | 0 |
| `first_home_mortgage_origination_year` | float64 |  | 0 |
| `second_home_mortgage_origination_year` | int64 |  | 0 |
| `health_savings_account_ald` | float64 |  | 0 |
| `takes_up_eitc` | bool |  | 0 |
| `would_file_taxes_voluntarily` | bool |  | 0 |
| `takes_up_aca_if_eligible` | bool |  | 0 |
| `selected_marketplace_plan_benchmark_ratio` | float64 |  | 0 |
| `tax_unit_spine` | str |  | 0 |

### `spm_unit`

| Column | Dtype | Logical dtype | Nulls |
| --- | --- | --- | ---: |
| `spm_unit_id` | int64 |  | 0 |
| `spm_unit_pre_subsidy_childcare_expenses` | float64 |  | 0 |
| `receives_housing_assistance` | bool |  | 0 |
| `takes_up_housing_assistance_if_eligible` | bool |  | 0 |
| `spm_unit_tenure_type` | str |  | 0 |
| `spm_unit_energy_subsidy` | float64 |  | 0 |
| `spm_unit_source_id` | int64 |  | 0 |
| `spm_unit_support_channel` | str |  | 0 |
| `spm_unit_support_clone_index` | int64 |  | 0 |
| `takes_up_tanf_if_eligible` | bool |  | 0 |
| `takes_up_snap_if_eligible` | bool |  | 0 |
| `receives_snap` | bool |  | 0 |
| `receives_tanf` | bool |  | 0 |
| `spm_unit_spine` | str |  | 0 |

### `family`

| Column | Dtype | Logical dtype | Nulls |
| --- | --- | --- | ---: |
| `family_id` | int64 |  | 0 |
| `family_source_id` | int64 |  | 0 |
| `family_support_channel` | str |  | 0 |
| `family_support_clone_index` | int64 |  | 0 |
| `family_spine` | str |  | 0 |

### `marital_unit`

| Column | Dtype | Logical dtype | Nulls |
| --- | --- | --- | ---: |
| `marital_unit_id` | int64 |  | 0 |
| `marital_unit_source_id` | int64 |  | 0 |
| `marital_unit_support_channel` | str |  | 0 |
| `marital_unit_support_clone_index` | int64 |  | 0 |
| `marital_unit_spine` | str |  | 0 |

## Changes from the prior schema

Compared with the sample column list sent with the County Files agreement: `prior_sample_columns.csv`.

### Columns added

| Table | Column | Dtype |
| --- | --- | --- |
| family | `family_spine` | str |
| household | `ADJHSG` | float64 |
| household | `FS` | float64 |
| household | `GRNTP` | float64 |
| household | `NP` | float64 |
| household | `PUMA` | str |
| household | `RNTP` | float64 |
| household | `SERIALNO` | str |
| household | `ST` | str |
| household | `TAXAMT` | float64 |
| household | `TEN` | float64 |
| household | `TYPEHUGQ` | float64 |
| household | `VEH` | float64 |
| household | `acs_annual_property_tax` | float64 |
| household | `acs_monthly_contract_rent` | float64 |
| household | `acs_monthly_gross_rent` | float64 |
| household | `county_fips_2023` | str |
| household | `household_spine` | str |
| household | `puma` | str |
| household | `puma_geoid` | str |
| marital_unit | `marital_unit_spine` | str |
| person | `ACS_MIL` | float64 |
| person | `ADJINC` | float64 |
| person | `AGEP` | float64 |
| person | `CIT` | float64 |
| person | `COW` | float64 |
| person | `DDRS` | float64 |
| person | `DEAR` | float64 |
| person | `DEYE` | float64 |
| person | `DOUT` | float64 |
| person | `DPHY` | float64 |
| person | `DREM` | float64 |
| person | `ESR` | float64 |
| person | `FWKHP` | float64 |
| person | `HINS3` | float64 |
| person | `HINS4` | float64 |
| person | `HINS5` | float64 |
| person | `HINS6` | float64 |
| person | `HINS7` | float64 |
| person | `INDP` | float64 |
| person | `INTP` | float64 |
| person | `MAR` | float64 |
| person | `OCCP` | float64 |
| person | `OIP` | float64 |
| person | `PAP` | float64 |
| person | `POBP` | float64 |
| person | `PWGTP` | float64 |
| person | `RELSHIPP` | float64 |
| person | `RETP` | float64 |
| person | `SCHG` | float64 |
| person | `SEMP` | float64 |
| person | `SEX` | float64 |
| person | `SPORDER` | float64 |
| person | `SSIP` | float64 |
| person | `SSP` | float64 |
| person | `WAGP` | float64 |
| person | `WKHP` | float64 |
| person | `WKL` | float64 |
| person | `WKWN` | float64 |
| person | `YOEP` | float64 |
| person | `acs_interest_dividend_rental_income` | float64 |
| person | `acs_retirement_income` | float64 |
| person | `acs_social_security_income` | float64 |
| person | `detailed_industry_recode` | int16 |
| person | `is_spm_independent_minor_role` | bool |
| person | `major_industry_recode` | int16 |
| person | `parent_1_id` | float64 |
| person | `parent_2_id` | float64 |
| person | `passive_partnership_s_corp_income` | float64 |
| person | `person_spine` | str |
| person | `receives_wic` | bool |
| person | `ssi_reported` | float64 |
| person | `strike_benefits` | float64 |
| person | `worked_last_year` | bool |
| person | `years_since_us_entry` | float64 |
| spm_unit | `receives_snap` | bool |
| spm_unit | `receives_tanf` | bool |
| spm_unit | `spm_unit_spine` | str |
| tax_unit | `tax_unit_spine` | str |

### Columns removed

| Table | Column | Dtype |
| --- | --- | --- |
| person | `has_marketplace_health_coverage` | bool |
| person | `medicare_part_b_premiums` | float64 |
| person | `weeks_worked` | float64 |

### Columns renamed

| Table | From | To | Dtype | Why |
| --- | --- | --- | --- | --- |
| person | `would_claim_wic` | `takes_up_wic_if_eligible` | bool | policyengine-us names the WIC take-up input takes_up_wic_if_eligible; it defines no would_claim_wic. |

### Dtypes changed

| Table | Column | Before | After |
| --- | --- | --- | --- |
| household | `H_TENURE` | int64 | float64 |
| household | `household_vehicles_owned` | int32 | int64 |
| household | `household_vehicles_value` | float32 | float64 |
| person | `ACTC_CRD` | int64 | float64 |
| person | `AGI` | int64 | float64 |
| person | `ANN_VAL` | int64 | float64 |
| person | `A_FNLWGT` | int64 | float64 |
| person | `A_HRS1` | int64 | float64 |
| person | `A_HSCOL` | int64 | float64 |
| person | `A_MJOCC` | int64 | float64 |
| person | `A_SEX` | int64 | float64 |
| person | `CAID` | int64 | float64 |
| person | `CAP_VAL` | int64 | float64 |
| person | `CHAMPVA` | int64 | float64 |
| person | `CHSP_VAL` | int64 | float64 |
| person | `CSP_VAL` | int64 | float64 |
| person | `CTC_CRD` | int64 | float64 |
| person | `DIS_SC1` | int64 | float64 |
| person | `DIS_SC2` | int64 | float64 |
| person | `DIS_VAL1` | int64 | float64 |
| person | `DIS_VAL2` | int64 | float64 |
| person | `DIV_VAL` | int64 | float64 |
| person | `DST_SC1` | int64 | float64 |
| person | `DST_SC1_YNG` | int64 | float64 |
| person | `DST_SC2` | int64 | float64 |
| person | `DST_SC2_YNG` | int64 | float64 |
| person | `DST_VAL1` | int64 | float64 |
| person | `DST_VAL1_YNG` | int64 | float64 |
| person | `DST_VAL2` | int64 | float64 |
| person | `DST_VAL2_YNG` | int64 | float64 |
| person | `EIT_CRED` | int64 | float64 |
| person | `FEDTAX_AC` | int64 | float64 |
| person | `FEDTAX_BC` | int64 | float64 |
| person | `FRSE_VAL` | int64 | float64 |
| person | `HRSWK` | int64 | float64 |
| person | `IHSFLG` | int64 | float64 |
| person | `INT_VAL` | int64 | float64 |
| person | `I_ERNVAL` | int64 | float64 |
| person | `I_SEVAL` | int64 | float64 |
| person | `MARG_TAX` | int64 | float64 |
| person | `MCARE` | int64 | float64 |
| person | `MIL` | int64 | float64 |
| person | `MOOP` | int64 | float64 |
| person | `NOW_GRP` | int64 | float64 |
| person | `NOW_MRK` | int64 | float64 |
| person | `OI_OFF` | int64 | float64 |
| person | `OI_VAL` | int64 | float64 |
| person | `PAW_VAL` | int64 | float64 |
| person | `PEAFEVER` | int64 | float64 |
| person | `PEDISDRS` | int64 | float64 |
| person | `PEDISEAR` | int64 | float64 |
| person | `PEDISEYE` | int64 | float64 |
| person | `PEDISOUT` | int64 | float64 |
| person | `PEDISPHY` | int64 | float64 |
| person | `PEDISREM` | int64 | float64 |
| person | `PEINUSYR` | int64 | float64 |
| person | `PEIO1COW` | int64 | float64 |
| person | `PEIOOCC` | int64 | float64 |
| person | `PEMCPREM` | int64 | float64 |
| person | `PENATVTY` | int64 | float64 |
| person | `PEN_SC1` | int64 | float64 |
| person | `PEN_SC2` | int64 | float64 |
| person | `PF_SEQ` | int64 | float64 |
| person | `PHIP_VAL` | int64 | float64 |
| person | `PMED_VAL` | int64 | float64 |
| person | `PNSN_VAL` | int64 | float64 |
| person | `POCCU2` | int64 | float64 |
| person | `POTC_VAL` | int64 | float64 |
| person | `PRCITSHP` | int64 | float64 |
| person | `PRDTHSP` | int64 | float64 |
| person | `PRDTRACE` | int64 | float64 |
| person | `P_SEQ` | int64 | float64 |
| person | `RESNSS1` | int64 | float64 |
| person | `RESNSS2` | int64 | float64 |
| person | `RESNSSI1` | int64 | float64 |
| person | `RESNSSI2` | int64 | float64 |
| person | `RETCB_VAL` | int64 | float64 |
| person | `RNT_VAL` | int64 | float64 |
| person | `SEMP_VAL` | int64 | float64 |
| person | `SPM_ACTC` | int64 | float64 |
| person | `SPM_BBSUBVAL` | int64 | float64 |
| person | `SPM_CAPHOUSESUB` | int64 | float64 |
| person | `SPM_CAPWKCCXPNS` | int64 | float64 |
| person | `SPM_CHILDCAREXPNS` | int64 | float64 |
| person | `SPM_CHILDSUPPD` | int64 | float64 |
| person | `SPM_EITC` | int64 | float64 |
| person | `SPM_ENGVAL` | int64 | float64 |
| person | `SPM_FAMTYPE` | int64 | float64 |
| person | `SPM_FEDTAX` | int64 | float64 |
| person | `SPM_FEDTAXBC` | int64 | float64 |
| person | `SPM_FICA` | int64 | float64 |
| person | `SPM_HAGE` | int64 | float64 |
| person | `SPM_HHISP` | int64 | float64 |
| person | `SPM_HMARITALSTATUS` | int64 | float64 |
| person | `SPM_HRACE` | int64 | float64 |
| person | `SPM_ID` | int64 | float64 |
| person | `SPM_MEDXPNS` | int64 | float64 |
| person | `SPM_NUMADULTS` | int64 | float64 |
| person | `SPM_NUMKIDS` | int64 | float64 |
| person | `SPM_NUMPER` | int64 | float64 |
| person | `SPM_POOR` | int64 | float64 |
| person | `SPM_POVTHRESHOLD` | int64 | float64 |
| person | `SPM_RESOURCES` | int64 | float64 |
| person | `SPM_SCHLUNCH` | int64 | float64 |
| person | `SPM_SNAPSUB` | int64 | float64 |
| person | `SPM_STTAX` | int64 | float64 |
| person | `SPM_TENMORTSTATUS` | int64 | float64 |
| person | `SPM_TOTVAL` | int64 | float64 |
| person | `SPM_WCOHABIT` | int64 | float64 |
| person | `SPM_WEIGHT` | int64 | float64 |
| person | `SPM_WFOSTER22` | int64 | float64 |
| person | `SPM_WICVAL` | int64 | float64 |
| person | `SPM_WKXPNS` | int64 | float64 |
| person | `SPM_WNEWHEAD` | int64 | float64 |
| person | `SPM_WNEWPARENT` | int64 | float64 |
| person | `SPM_WUI_LT15` | int64 | float64 |
| person | `SSI_VAL` | int64 | float64 |
| person | `SSI_YN` | int64 | float64 |
| person | `SS_VAL` | int64 | float64 |
| person | `SS_YN` | int64 | float64 |
| person | `STATETAX_A` | int64 | float64 |
| person | `STATETAX_B` | int64 | float64 |
| person | `TAX_ID` | int64 | float64 |
| person | `TAX_INC` | int64 | float64 |
| person | `UC_VAL` | int64 | float64 |
| person | `VET_VAL` | int64 | float64 |
| person | `WC_VAL` | int64 | float64 |
| person | `WICYN` | int64 | float64 |
| person | `WKSWORK` | int64 | float64 |
| person | `WSAL_VAL` | int64 | float64 |
| person | `cps_race` | int16 | int64 |
| person | `fsla_overtime_premium` | float32 | float64 |
| person | `treasury_tipped_occupation_code` | int16 | int64 |
| tax_unit | `second_home_mortgage_origination_year` | float64 | int64 |
