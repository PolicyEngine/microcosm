"""Runtime helpers for the US dataset build.

This module is the declarative description of how the US population dataset
is assembled — the stage order, what each stage produces, and **which primary
survey each imputation draws from**, with citations. It replaces the
imperative driver's implicit structure with one reviewable object:
:func:`us_plan` returns the :class:`~populace.build.plan.StagePlan` whose
donor graph is the published sources diagram.

Every donor here is a primary source. Incumbent production datasets are not
build inputs; release comparisons against them live in the external benchmark
repo.

Implementations are injected: :func:`us_plan` requires one callable per
declared stage and refuses to assemble without all of them — there is no
stub, default, or fallback implementation (a plan you can run with a missing
stage would be the silent-fallback bug as a framework feature). Source-stage
content lives in the packaged JSON manifest loaded as
:data:`US_SOURCE_MANIFEST`; executable Python belongs only in shared Populace
runtimes that interpret those specs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files

from populace.build.plan import DonorSpec, Stage, StagePlan
from populace.build.source_manifest import (
    SourceManifest,
    SourceStageSpec,
    SupportSpineManifest,
    SupportSpineSpec,
    load_source_manifest,
    load_support_spine_manifest,
)
from populace.build.us_runtime.alimony import (
    ALIMONY_ASEC_ARCHIVED_DERIVATION_URL,
    ALIMONY_PUF_ARCHIVED_DERIVATION_URL,
    US_ALIMONY_NONCONSTANT_PERSON_COLUMNS,
    US_ALIMONY_OUTPUT_COLUMNS,
    US_ALIMONY_STAGE_NAME,
    derive_us_alimony_from_asec,
    derive_us_alimony_from_puf,
    us_alimony_signal_gate,
    us_alimony_stage_spec,
    us_alimony_summary,
)
from populace.build.us_runtime.asec_pool import (
    AsecSource,
    build_pooled_asec_unit_frame,
    load_asec_h5_tables,
    pool_asec_sources,
)
from populace.build.us_runtime.casualty_losses import (
    US_CASUALTY_LOSS_NONCONSTANT_PERSON_COLUMNS,
    US_CASUALTY_LOSS_OUTPUT_COLUMNS,
    US_CASUALTY_LOSS_STAGE_NAME,
    derive_us_casualty_loss_from_puf,
    us_casualty_loss_signal_gate,
    us_casualty_loss_stage_spec,
    us_casualty_loss_summary,
)
from populace.build.us_runtime.child_support import (
    CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL,
    CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL,
    CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL,
    CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL,
    US_CHILD_SUPPORT_NONCONSTANT_PERSON_COLUMNS,
    US_CHILD_SUPPORT_OUTPUT_COLUMNS,
    US_CHILD_SUPPORT_REQUIRED_SOURCE_COLUMNS,
    US_CHILD_SUPPORT_STAGE_NAME,
    derive_us_child_support_from_asec,
    derive_us_child_support_from_manifest,
    impute_us_child_support_to_puf_support_from_manifest,
    us_child_support_signal_gate,
    us_child_support_stage_spec,
    us_child_support_summary,
    with_us_child_support_inputs,
)
from populace.build.us_runtime.childcare import (
    US_CHILDCARE_OUTPUT_COLUMNS,
    US_CHILDCARE_REQUIRED_SOURCE_COLUMNS,
    US_CHILDCARE_STAGE_NAME,
    derive_us_childcare_from_manifest,
    impute_us_childcare_to_puf_support_from_manifest,
    us_childcare_signal_gate,
    us_childcare_stage_spec,
    us_childcare_summary,
    with_us_childcare_inputs,
)
from populace.build.us_runtime.congressional_district_geography import (
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
    SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID,
    assign_congressional_districts_to_households,
    congressional_district_assignment_summary,
    congressional_district_distribution_from_ledger_facts,
    with_household_congressional_districts,
)
from populace.build.us_runtime.congressional_district_vintage import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
    load_congressional_district_vintage_crosswalk,
    translate_congressional_district_facts_to_current_vintage,
)
from populace.build.us_runtime.cps_carried import (
    CPS_CARRIED_FORMULA_OWNED_COLUMNS,
    CPS_CARRIED_PERSON_INPUTS,
    CPS_CARRIED_SPM_UNIT_INPUTS,
    derive_us_cps_carried_inputs,
)
from populace.build.us_runtime.demographics import (
    AGE_BANDS,
    DEMOGRAPHICS_SCHEMA_VERSION,
    AgeBand,
    compute_age_distribution,
    demographics_payload,
    write_demographics,
)
from populace.build.us_runtime.disability_benefits import (
    DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL,
    DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL,
    DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL,
    DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL,
    US_DISABILITY_BENEFITS_NONCONSTANT_PERSON_COLUMNS,
    US_DISABILITY_BENEFITS_OUTPUT_COLUMNS,
    US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS,
    US_DISABILITY_BENEFITS_STAGE_NAME,
    derive_us_disability_benefits_from_asec,
    derive_us_disability_benefits_from_manifest,
    impute_us_disability_benefits_to_puf_support_from_manifest,
    us_disability_benefits_signal_gate,
    us_disability_benefits_stage_spec,
    us_disability_benefits_summary,
    with_us_disability_benefits,
)
from populace.build.us_runtime.domestic_production import (
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL,
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL,
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL,
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL,
    US_DOMESTIC_PRODUCTION_ALD_NONCONSTANT_TAX_UNIT_COLUMNS,
    US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS,
    US_DOMESTIC_PRODUCTION_ALD_STAGE_NAME,
    derive_us_domestic_production_ald_from_puf,
    us_domestic_production_ald_signal_gate,
    us_domestic_production_ald_stage_spec,
    us_domestic_production_ald_summary,
)
from populace.build.us_runtime.education_inputs import (
    US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS,
    US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    US_EDUCATION_INPUTS_OUTPUT_COLUMNS,
    US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS,
    US_EDUCATION_INPUTS_STAGE_NAME,
    derive_us_education_inputs_from_manifest,
    us_education_inputs_signal_gate,
    us_education_inputs_stage_spec,
    us_education_inputs_summary,
    with_us_education_inputs,
)
from populace.build.us_runtime.educator_expenses import (
    EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL,
    EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL,
    EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL,
    EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL,
    US_EDUCATOR_EXPENSE_NONCONSTANT_PERSON_COLUMNS,
    US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS,
    US_EDUCATOR_EXPENSE_STAGE_NAME,
    derive_us_educator_expense_from_puf,
    us_educator_expense_signal_gate,
    us_educator_expense_stage_spec,
    us_educator_expense_summary,
)
from populace.build.us_runtime.eligibility_inputs import (
    US_ELIGIBILITY_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS,
    US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS,
    US_ELIGIBILITY_INPUTS_STAGE_NAME,
    derive_us_eligibility_inputs_from_manifest,
    us_eligibility_inputs_signal_gate,
    us_eligibility_inputs_stage_spec,
    us_eligibility_inputs_summary,
    with_us_eligibility_inputs,
)
from populace.build.us_runtime.farm_business_income import (
    FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL,
    US_FARM_BUSINESS_INCOME_NONCONSTANT_PERSON_COLUMNS,
    US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS,
    US_FARM_BUSINESS_INCOME_STAGE_NAME,
    derive_us_farm_business_income_from_puf,
    us_farm_business_income_signal_gate,
    us_farm_business_income_stage_spec,
    us_farm_business_income_summary,
)
from populace.build.us_runtime.fiscal_targets import (
    SOI_VARIABLE_MAP,
    US_FISCAL_LEDGER_PARITY_REGISTRY,
    US_FISCAL_LEDGER_PARITY_REPORT,
    US_FISCAL_MACRO_REALISM_BANDS,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_LEDGER_REFERENCES,
    US_FISCAL_TARGET_REFERENCES,
    US_FISCAL_TARGET_REGISTRY,
    US_FISCAL_TARGET_SPECS,
    US_FISCAL_TARGET_SUPPORT_EXCLUSIONS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES,
    US_JCT_TAX_EXPENDITURE_TARGET_SPECS,
    US_SOI_FISCAL_TARGET_REFERENCES,
    US_SOI_FISCAL_TARGET_SPECS,
    US_STATE_INCOME_TAX_TARGET_REFERENCES,
    US_STATE_INCOME_TAX_TARGET_SPECS,
    SimpleTaxExpenditureReform,
    compile_us_fiscal_target_registry,
)
from populace.build.us_runtime.form_4952 import (
    FORM_4952_ARCHIVED_DERIVATION_URL,
    FORM_4952_ARCHIVED_EXPORT_URL,
    FORM_4952_ARCHIVED_IMPUTATION_URL,
    FORM_4952_ARCHIVED_PERSON_ALLOCATION_URL,
    FORM_4952_ARCHIVED_PUF_ARTIFACT_URL,
    US_FORM_4952_NONCONSTANT_PERSON_COLUMNS,
    US_FORM_4952_OUTPUT_COLUMNS,
    US_FORM_4952_STAGE_NAME,
    derive_us_form_4952_election_from_puf,
    us_form_4952_election_signal_gate,
    us_form_4952_election_stage_spec,
    us_form_4952_election_summary,
)
from populace.build.us_runtime.geography_ladder import (
    GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR,
    GEOGRAPHY_LADDER_VINTAGES_ATTR,
    US_BLOCK_LADDER_DERIVED_LAYERS,
    US_BLOCK_LADDER_KIND,
    US_BLOCK_LADDER_SCHEMA_VERSION,
    US_GEOGRAPHY_LADDER_COLUMNS,
    US_NYC_COUNTY_FIPS,
    UsBlockLadder,
    assign_us_geography_ladder,
    load_us_block_ladder,
    us_geography_ladder_assignment_summary,
    us_geography_ladder_gate,
    with_household_us_geography_ladder,
)
from populace.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_NONCONSTANT_PERSON_COLUMNS,
    US_HOURS_WORKED_OUTPUT_COLUMNS,
    US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS,
    US_HOURS_WORKED_STAGE_NAME,
    derive_us_hours_worked_from_manifest,
    us_hours_worked_signal_gate,
    us_hours_worked_stage_spec,
    us_hours_worked_summary,
    with_us_hours_worked_inputs,
)
from populace.build.us_runtime.immigration import (
    IMMIGRATION_STATUS_VALUES,
    SSN_CARD_TYPE_VALUES,
    US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS,
    US_IMMIGRATION_OUTPUT_COLUMNS,
    US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS,
    US_IMMIGRATION_STAGE_NAME,
    UndocumentedControls,
    derive_us_immigration_status_from_manifest,
    us_immigration_composition_gate,
    us_immigration_composition_summary,
    us_immigration_stage_spec,
    with_us_immigration_inputs,
)
from populace.build.us_runtime.input_mass import (
    us_input_mass_totals,
)
from populace.build.us_runtime.medicaid_take_up import (
    US_MEDICAID_ENROLLMENT_SUBSTITUTIONS,
    US_MEDICAID_ENROLLMENT_TARGET_ROLE,
    US_MEDICAID_ENROLLMENT_TARGET_TABLE,
    US_MEDICAID_ENROLLMENT_TOLERANCE,
    US_MEDICAID_TAKE_UP_ANCHOR,
    US_MEDICAID_TAKE_UP_STAGE,
    US_MEDICAID_TAKE_UP_VARIABLE,
    MedicaidEnrollmentSubstitution,
    apply_us_medicaid_enrollment_substitutions,
    us_medicaid_take_up_diagnostics,
    us_medicaid_take_up_gate,
    with_us_medicaid_take_up,
    write_us_medicaid_take_up_diagnostics,
)
from populace.build.us_runtime.misc_itemized import (
    US_MISC_ITEMIZED_NONCONSTANT_PERSON_COLUMNS,
    US_MISC_ITEMIZED_OUTPUT_COLUMNS,
    US_MISC_ITEMIZED_STAGE_NAME,
    derive_us_misc_itemized_from_puf,
    us_misc_itemized_signal_gate,
    us_misc_itemized_stage_spec,
    us_misc_itemized_summary,
)
from populace.build.us_runtime.nonzero_shares import (
    nonzero_share,
    us_nonzero_shares,
)
from populace.build.us_runtime.org_wages import (
    BLS_STATE_UNION_REPRESENTATION_RATE_2024,
    FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES,
    FLSA_OVERTIME_OCCUPATION_CODES,
    ORG_2024_DONOR_CONTENT_SHA256,
    ORG_2024_DONOR_FILENAME,
    ORG_PREDICTORS,
    US_ORG_WAGES_NONCONSTANT_PERSON_COLUMNS,
    US_ORG_WAGES_OUTPUT_COLUMNS,
    US_ORG_WAGES_REQUIRED_SOURCE_COLUMNS,
    US_ORG_WAGES_STAGE_NAME,
    derive_flsa_overtime_premium,
    derive_us_org_occupation_inputs,
    fetch_org_2024_donor,
    impute_us_org_wages,
    load_org_2024_donor,
    us_org_wages_signal_gate,
    us_org_wages_stage_spec,
    us_org_wages_summary,
    with_us_org_wages_inputs,
)
from populace.build.us_runtime.other_health_insurance import (
    OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL,
    US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES,
    US_OTHER_HEALTH_INSURANCE_NONCONSTANT_PERSON_COLUMNS,
    US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
    US_OTHER_HEALTH_INSURANCE_REQUIRED_SOURCE_COLUMNS,
    US_OTHER_HEALTH_INSURANCE_STAGE_NAME,
    derive_us_other_health_insurance_from_asec,
    derive_us_other_health_insurance_from_manifest,
    impute_us_other_health_insurance_to_puf_support_from_manifest,
    us_other_health_insurance_signal_gate,
    us_other_health_insurance_stage_spec,
    us_other_health_insurance_summary,
    with_us_other_health_insurance_inputs,
)
from populace.build.us_runtime.parity_reference import (
    ECPS_PARITY_KNOWN_GAPS_RESOURCE,
    ECPS_PARITY_REFERENCE_RESOURCE,
    EcpsParityReference,
    EcpsParitySource,
    ParityKnownGap,
    load_ecps_parity_known_gaps,
    load_ecps_parity_reference,
)
from populace.build.us_runtime.pregnancy import (
    US_PREGNANCY_NONCONSTANT_PERSON_COLUMNS,
    US_PREGNANCY_OUTPUT_COLUMN,
    US_PREGNANCY_REQUIRED_SOURCE_COLUMNS,
    US_PREGNANCY_STAGE_NAME,
    derive_us_pregnancy_from_manifest,
    us_pregnancy_signal_gate,
    us_pregnancy_stage_spec,
    us_pregnancy_summary,
    with_us_pregnancy_inputs,
)
from populace.build.us_runtime.puf_support import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_PUF_SUPPORT_FIT_NAME,
    US_PUF_SUPPORT_STAGE_NAME,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.build.us_runtime.qbi_inputs import (
    QBI_ARCHIVED_ASSUMPTIONS_URL,
    QBI_ARCHIVED_CLONE_URL,
    QBI_ARCHIVED_DERIVATION_URL,
    QBI_ARCHIVED_EXPORT_URL,
    QBI_ARCHIVED_IMPUTATION_URL,
    QBI_ARCHIVED_PUF_ARTIFACT_URL,
    QBI_ARCHIVED_SIMULATION_URL,
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_NONCONSTANT_PERSON_COLUMNS,
    US_QBI_NONNEGATIVE_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
    US_QBI_STAGE_NAME,
    us_qbi_inputs_signal_gate,
    us_qbi_inputs_stage_spec,
    us_qbi_inputs_summary,
    with_us_qbi_input_reconciliation,
)
from populace.build.us_runtime.reform_coverage_smoke import (
    us_reform_coverage_smoke_gate,
)
from populace.build.us_runtime.reform_validation import (
    REFORM_VALIDATION_SCHEMA_VERSION,
    ReformValidationSpec,
    in_sample_reform_specs,
    load_default_reform_specs,
    out_of_sample_reform_specs,
    reform_validation_payload,
    write_reform_validation,
)
from populace.build.us_runtime.register_consistency import (
    us_register_consistency_gate,
    us_register_contradictions,
)
from populace.build.us_runtime.release_input_coverage import (
    POST_REFERENCE_ECPS_REQUIRED_INPUTS,
    SSI_COUNTABLE_RESOURCE_ASSETS,
    US_RELEASE_INPUT_COVERAGE_RESOURCE,
    ReformCoverageProbe,
    ReleaseInputColumn,
    ReleaseInputCoverageManifest,
    assert_release_input_coverage_manifest_current,
    load_release_input_coverage_manifest,
    us_release_input_coverage_gate,
    us_release_input_coverage_required_columns,
    us_release_input_coverage_reviewed_exclusions,
    us_release_reform_coverage_probes,
)
from populace.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_NONCONSTANT_PERSON_COLUMNS,
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    US_RETIREMENT_CONTRIBUTION_STAGE_NAME,
    derive_us_retirement_contributions_from_manifest,
    impute_us_retirement_contributions_to_puf_support_from_manifest,
    us_retirement_contributions_signal_gate,
    us_retirement_contributions_stage_spec,
    us_retirement_contributions_summary,
    with_us_retirement_contribution_inputs,
)
from populace.build.us_runtime.scf_auto_loans import (
    QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET,
    SCF_2022_FULL_EXTRACT_MEMBER,
    SCF_2022_FULL_EXTRACT_MEMBER_SHA256,
    SCF_2022_FULL_EXTRACT_URL,
    SCF_2022_FULL_EXTRACT_ZIP_SHA256,
    SCF_AUTO_LOAN_AMOUNT_COLUMNS,
    SCF_AUTO_LOAN_RATE_COLUMNS,
    US_SCF_AUTO_LOAN_NONCONSTANT_HOUSEHOLD_COLUMNS,
    US_SCF_AUTO_LOAN_OUTPUT_COLUMNS,
    fetch_scf_2022_full_extract,
    impute_us_scf_auto_loans,
    load_scf_2022_auto_loan_donor,
    qualified_auto_loan_interest_proxy,
    us_scf_auto_loans_signal_gate,
    us_scf_auto_loans_stage_spec,
    us_scf_auto_loans_summary,
    with_us_scf_auto_loan_inputs,
)
from populace.build.us_runtime.scf_wealth import (
    SCF_FINANCIAL_ASSET_TARGET_COMPONENTS,
    SCF_WEALTH_PREDICTORS,
    US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
    US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS,
    US_SCF_WEALTH_STAGE_NAME,
    fetch_scf_2022_summary_extract,
    impute_us_scf_financial_assets,
    load_scf_2022_financial_asset_donor,
    us_scf_wealth_signal_gate,
    us_scf_wealth_stage_spec,
    us_scf_wealth_summary,
    with_us_scf_wealth_inputs,
)
from populace.build.us_runtime.sipp_tips import (
    CENSUS_OCCUPATION_CODE_TO_TTOC,
    SIPP_2023_TIP_DONOR_REVISION,
    SIPP_2023_TIP_DONOR_SHA256,
    SIPP_2023_TIP_DONOR_URL,
    SIPP_TIP_OUTPUT_COLUMNS,
    SIPP_TIP_PREDICTORS,
    US_SIPP_TIPS_NONCONSTANT_PERSON_COLUMNS,
    US_SIPP_TIPS_OUTPUT_COLUMNS,
    US_SIPP_TIPS_REQUIRED_SOURCE_COLUMNS,
    US_SIPP_TIPS_STAGE_NAME,
    derive_treasury_tipped_occupation_code,
    fetch_sipp_2023_tip_donor,
    impute_us_sipp_tips,
    load_sipp_2023_tip_donor,
    us_sipp_tips_signal_gate,
    us_sipp_tips_stage_spec,
    us_sipp_tips_summary,
    with_us_sipp_tip_inputs,
)
from populace.build.us_runtime.sipp_vehicles import (
    SIPP_2023_VEHICLE_DONOR_REVISION,
    SIPP_2023_VEHICLE_DONOR_SHA256,
    SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    SIPP_2023_VEHICLE_DONOR_URL,
    US_SIPP_VEHICLE_NONCONSTANT_HOUSEHOLD_COLUMNS,
    US_SIPP_VEHICLE_OUTPUT_COLUMNS,
    fetch_sipp_2023_vehicle_donor,
    load_sipp_2023_vehicle_donor,
    us_sipp_vehicles_signal_gate,
    us_sipp_vehicles_stage_spec,
    us_sipp_vehicles_summary,
    with_us_sipp_vehicle_inputs,
)
from populace.build.us_runtime.snap_discretionary_exemption import (
    US_SNAP_DISCRETIONARY_EXEMPTION_NONCONSTANT_PERSON_COLUMNS,
    US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN,
    US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS,
    US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME,
    derive_us_snap_discretionary_exemption_from_manifest,
    us_snap_discretionary_exemption_signal_gate,
    us_snap_discretionary_exemption_stage_spec,
    us_snap_discretionary_exemption_summary,
    with_us_snap_discretionary_exemption_inputs,
)
from populace.build.us_runtime.snap_take_up import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_SNAP_TAKE_UP_RAW_COLUMN,
    US_SNAP_TAKE_UP_STAGE_NAME,
    derive_us_snap_take_up_from_manifest,
    us_snap_take_up_signal_gate,
    us_snap_take_up_stage_spec,
    us_snap_take_up_summary,
    with_us_snap_take_up_inputs,
)
from populace.build.us_runtime.source_coverage import (
    LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT,
    US_SOURCE_COVERAGE,
    hard_target_package_aliases,
    source_gap_family_ids,
    us_source_coverage_diagnostics,
    us_source_coverage_gate,
    validation_only_family_ids,
    write_us_source_coverage_diagnostics,
)
from populace.build.us_runtime.source_runtime import (
    disaggregate_us_puf_aggregate_records_from_manifest,
    us_source_operation_handlers,
)
from populace.build.us_runtime.take_up import (
    US_TAKE_UP_SHARE_BAND,
    SeededTakeUpResult,
    us_take_up_participation_diagnostics,
    us_take_up_signal_gate,
    us_take_up_summary,
    with_us_take_up_inputs,
    write_us_take_up_participation_diagnostics,
)
from populace.build.us_runtime.take_up_contract import (
    TakeUpContract,
    TakeUpProgram,
    assert_take_up_contract_current,
    assert_take_up_treatments_consistent,
    count_calibrated_take_up_programs,
    load_take_up_contract,
    seeded_take_up_programs,
)
from populace.build.us_runtime.validation_input_coverage import (
    US_VALIDATION_PROVISION_INPUT_LEAVES,
    ValidationInputLeaf,
    assert_validation_leaf_registry_current,
    us_source_stage_outputs,
    us_validation_input_coverage_gate,
)
from populace.frame import Frame

__all__ = [
    "BuildConfig",
    "AsecSource",
    "BASE_ASEC_SUPPORT_CHANNEL",
    "CPS_CARRIED_FORMULA_OWNED_COLUMNS",
    "CPS_CARRIED_PERSON_INPUTS",
    "CPS_CARRIED_SPM_UNIT_INPUTS",
    "SimpleTaxExpenditureReform",
    "ReformValidationSpec",
    "REFORM_VALIDATION_SCHEMA_VERSION",
    "LEDGER_US_SOURCE_COVERAGE_CONTRACT_COMMIT",
    "AgeBand",
    "AGE_BANDS",
    "DEMOGRAPHICS_SCHEMA_VERSION",
    "compute_age_distribution",
    "CONGRESSIONAL_DISTRICT_GEOID_COLUMN",
    "CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR",
    "CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR",
    "CURRENT_CONGRESSIONAL_DISTRICT_PREFIX",
    "CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE",
    "SOI_CONGRESSIONAL_DISTRICT_RECORD_SET_ID",
    "SOURCE_CONGRESSIONAL_DISTRICT_PREFIX",
    "GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR",
    "GEOGRAPHY_LADDER_VINTAGES_ATTR",
    "US_BLOCK_LADDER_DERIVED_LAYERS",
    "US_BLOCK_LADDER_KIND",
    "US_BLOCK_LADDER_SCHEMA_VERSION",
    "US_GEOGRAPHY_LADDER_COLUMNS",
    "US_NYC_COUNTY_FIPS",
    "UsBlockLadder",
    "assign_us_geography_ladder",
    "load_us_block_ladder",
    "us_geography_ladder_assignment_summary",
    "us_geography_ladder_gate",
    "with_household_us_geography_ladder",
    "demographics_payload",
    "write_demographics",
    "US_DONORS",
    "US_FISCAL_MACRO_REALISM_BANDS",
    "US_FISCAL_LEDGER_PARITY_REGISTRY",
    "US_FISCAL_LEDGER_PARITY_REPORT",
    "US_FISCAL_TARGET_REGISTRY",
    "US_FISCAL_TARGET_REFERENCES",
    "US_FISCAL_TARGET_SPECS",
    "US_FISCAL_TARGET_SUPPORT_EXCLUSIONS",
    "US_FISCAL_TARGET_COVERAGE_REQUIREMENTS",
    "US_FISCAL_TARGET_LEDGER_REFERENCES",
    "US_JCT_TAX_EXPENDITURE_REFORMS",
    "US_JCT_TAX_EXPENDITURE_TARGET_SPECS",
    "US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES",
    "SOI_VARIABLE_MAP",
    "IMMIGRATION_STATUS_VALUES",
    "SSN_CARD_TYPE_VALUES",
    "US_IMMIGRATION_NONCONSTANT_PERSON_COLUMNS",
    "US_IMMIGRATION_OUTPUT_COLUMNS",
    "US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS",
    "US_IMMIGRATION_STAGE_NAME",
    "UndocumentedControls",
    "US_HOURS_WORKED_NONCONSTANT_PERSON_COLUMNS",
    "US_HOURS_WORKED_OUTPUT_COLUMNS",
    "US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS",
    "US_HOURS_WORKED_STAGE_NAME",
    "derive_us_hours_worked_from_manifest",
    "us_hours_worked_signal_gate",
    "us_hours_worked_stage_spec",
    "us_hours_worked_summary",
    "with_us_hours_worked_inputs",
    "US_SNAP_TAKE_UP_OUTPUT_COLUMN",
    "US_SNAP_TAKE_UP_RAW_COLUMN",
    "US_SNAP_TAKE_UP_STAGE_NAME",
    "derive_us_snap_take_up_from_manifest",
    "us_snap_take_up_signal_gate",
    "us_snap_take_up_stage_spec",
    "us_snap_take_up_summary",
    "with_us_snap_take_up_inputs",
    "US_SNAP_DISCRETIONARY_EXEMPTION_NONCONSTANT_PERSON_COLUMNS",
    "US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN",
    "US_SNAP_DISCRETIONARY_EXEMPTION_REQUIRED_SOURCE_COLUMNS",
    "US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME",
    "derive_us_snap_discretionary_exemption_from_manifest",
    "us_snap_discretionary_exemption_signal_gate",
    "us_snap_discretionary_exemption_stage_spec",
    "us_snap_discretionary_exemption_summary",
    "with_us_snap_discretionary_exemption_inputs",
    "US_PREGNANCY_NONCONSTANT_PERSON_COLUMNS",
    "US_PREGNANCY_OUTPUT_COLUMN",
    "US_PREGNANCY_REQUIRED_SOURCE_COLUMNS",
    "US_PREGNANCY_STAGE_NAME",
    "derive_us_pregnancy_from_manifest",
    "us_pregnancy_signal_gate",
    "us_pregnancy_stage_spec",
    "us_pregnancy_summary",
    "with_us_pregnancy_inputs",
    "US_ELIGIBILITY_INPUTS_NONCONSTANT_PERSON_COLUMNS",
    "US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS",
    "US_ELIGIBILITY_INPUTS_REQUIRED_SOURCE_COLUMNS",
    "US_ELIGIBILITY_INPUTS_STAGE_NAME",
    "derive_us_eligibility_inputs_from_manifest",
    "us_eligibility_inputs_signal_gate",
    "us_eligibility_inputs_stage_spec",
    "us_eligibility_inputs_summary",
    "with_us_eligibility_inputs",
    "ALIMONY_ASEC_ARCHIVED_DERIVATION_URL",
    "ALIMONY_PUF_ARCHIVED_DERIVATION_URL",
    "US_ALIMONY_NONCONSTANT_PERSON_COLUMNS",
    "US_ALIMONY_OUTPUT_COLUMNS",
    "US_ALIMONY_STAGE_NAME",
    "derive_us_alimony_from_asec",
    "derive_us_alimony_from_puf",
    "us_alimony_signal_gate",
    "us_alimony_stage_spec",
    "us_alimony_summary",
    "US_CASUALTY_LOSS_NONCONSTANT_PERSON_COLUMNS",
    "US_CASUALTY_LOSS_OUTPUT_COLUMNS",
    "US_CASUALTY_LOSS_STAGE_NAME",
    "derive_us_casualty_loss_from_puf",
    "us_casualty_loss_signal_gate",
    "us_casualty_loss_stage_spec",
    "us_casualty_loss_summary",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL",
    "DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL",
    "US_DOMESTIC_PRODUCTION_ALD_NONCONSTANT_TAX_UNIT_COLUMNS",
    "US_DOMESTIC_PRODUCTION_ALD_OUTPUT_COLUMNS",
    "US_DOMESTIC_PRODUCTION_ALD_STAGE_NAME",
    "derive_us_domestic_production_ald_from_puf",
    "us_domestic_production_ald_signal_gate",
    "us_domestic_production_ald_stage_spec",
    "us_domestic_production_ald_summary",
    "US_CHILDCARE_OUTPUT_COLUMNS",
    "US_CHILDCARE_REQUIRED_SOURCE_COLUMNS",
    "US_CHILDCARE_STAGE_NAME",
    "derive_us_childcare_from_manifest",
    "impute_us_childcare_to_puf_support_from_manifest",
    "us_childcare_signal_gate",
    "us_childcare_stage_spec",
    "us_childcare_summary",
    "with_us_childcare_inputs",
    "CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL",
    "CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL",
    "CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL",
    "CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL",
    "US_CHILD_SUPPORT_NONCONSTANT_PERSON_COLUMNS",
    "US_CHILD_SUPPORT_OUTPUT_COLUMNS",
    "US_CHILD_SUPPORT_REQUIRED_SOURCE_COLUMNS",
    "US_CHILD_SUPPORT_STAGE_NAME",
    "derive_us_child_support_from_asec",
    "derive_us_child_support_from_manifest",
    "impute_us_child_support_to_puf_support_from_manifest",
    "us_child_support_signal_gate",
    "us_child_support_stage_spec",
    "us_child_support_summary",
    "with_us_child_support_inputs",
    "DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL",
    "DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL",
    "DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL",
    "DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL",
    "US_DISABILITY_BENEFITS_NONCONSTANT_PERSON_COLUMNS",
    "US_DISABILITY_BENEFITS_OUTPUT_COLUMNS",
    "US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS",
    "US_DISABILITY_BENEFITS_STAGE_NAME",
    "derive_us_disability_benefits_from_asec",
    "derive_us_disability_benefits_from_manifest",
    "impute_us_disability_benefits_to_puf_support_from_manifest",
    "us_disability_benefits_signal_gate",
    "us_disability_benefits_stage_spec",
    "us_disability_benefits_summary",
    "with_us_disability_benefits",
    "US_MISC_ITEMIZED_NONCONSTANT_PERSON_COLUMNS",
    "US_MISC_ITEMIZED_OUTPUT_COLUMNS",
    "US_MISC_ITEMIZED_STAGE_NAME",
    "derive_us_misc_itemized_from_puf",
    "us_misc_itemized_signal_gate",
    "us_misc_itemized_stage_spec",
    "us_misc_itemized_summary",
    "FORM_4952_ARCHIVED_DERIVATION_URL",
    "FORM_4952_ARCHIVED_EXPORT_URL",
    "FORM_4952_ARCHIVED_IMPUTATION_URL",
    "FORM_4952_ARCHIVED_PERSON_ALLOCATION_URL",
    "FORM_4952_ARCHIVED_PUF_ARTIFACT_URL",
    "US_FORM_4952_NONCONSTANT_PERSON_COLUMNS",
    "US_FORM_4952_OUTPUT_COLUMNS",
    "US_FORM_4952_STAGE_NAME",
    "derive_us_form_4952_election_from_puf",
    "us_form_4952_election_signal_gate",
    "us_form_4952_election_stage_spec",
    "us_form_4952_election_summary",
    "QBI_ARCHIVED_ASSUMPTIONS_URL",
    "QBI_ARCHIVED_CLONE_URL",
    "QBI_ARCHIVED_DERIVATION_URL",
    "QBI_ARCHIVED_EXPORT_URL",
    "QBI_ARCHIVED_IMPUTATION_URL",
    "QBI_ARCHIVED_PUF_ARTIFACT_URL",
    "QBI_ARCHIVED_SIMULATION_URL",
    "US_QBI_BOOLEAN_OUTPUT_COLUMNS",
    "US_QBI_NONCONSTANT_PERSON_COLUMNS",
    "US_QBI_NONNEGATIVE_OUTPUT_COLUMNS",
    "US_QBI_OUTPUT_COLUMNS",
    "US_QBI_STAGE_NAME",
    "us_qbi_inputs_signal_gate",
    "us_qbi_inputs_stage_spec",
    "us_qbi_inputs_summary",
    "with_us_qbi_input_reconciliation",
    "US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS",
    "US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS",
    "US_EDUCATION_INPUTS_OUTPUT_COLUMNS",
    "US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS",
    "US_EDUCATION_INPUTS_STAGE_NAME",
    "derive_us_education_inputs_from_manifest",
    "us_education_inputs_signal_gate",
    "us_education_inputs_stage_spec",
    "us_education_inputs_summary",
    "with_us_education_inputs",
    "EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL",
    "EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL",
    "EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL",
    "EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL",
    "US_EDUCATOR_EXPENSE_NONCONSTANT_PERSON_COLUMNS",
    "US_EDUCATOR_EXPENSE_OUTPUT_COLUMNS",
    "US_EDUCATOR_EXPENSE_STAGE_NAME",
    "derive_us_educator_expense_from_puf",
    "us_educator_expense_signal_gate",
    "us_educator_expense_stage_spec",
    "us_educator_expense_summary",
    "FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL",
    "FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL",
    "US_FARM_BUSINESS_INCOME_NONCONSTANT_PERSON_COLUMNS",
    "US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS",
    "US_FARM_BUSINESS_INCOME_STAGE_NAME",
    "derive_us_farm_business_income_from_puf",
    "us_farm_business_income_signal_gate",
    "us_farm_business_income_stage_spec",
    "us_farm_business_income_summary",
    "US_RETIREMENT_CONTRIBUTION_NONCONSTANT_PERSON_COLUMNS",
    "US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS",
    "US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS",
    "US_RETIREMENT_CONTRIBUTION_STAGE_NAME",
    "derive_us_retirement_contributions_from_manifest",
    "impute_us_retirement_contributions_to_puf_support_from_manifest",
    "us_retirement_contributions_signal_gate",
    "us_retirement_contributions_stage_spec",
    "us_retirement_contributions_summary",
    "with_us_retirement_contribution_inputs",
    "QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET",
    "SCF_2022_FULL_EXTRACT_MEMBER",
    "SCF_2022_FULL_EXTRACT_MEMBER_SHA256",
    "SCF_2022_FULL_EXTRACT_URL",
    "SCF_2022_FULL_EXTRACT_ZIP_SHA256",
    "SCF_AUTO_LOAN_AMOUNT_COLUMNS",
    "SCF_AUTO_LOAN_RATE_COLUMNS",
    "US_SCF_AUTO_LOAN_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_SCF_AUTO_LOAN_OUTPUT_COLUMNS",
    "fetch_scf_2022_full_extract",
    "impute_us_scf_auto_loans",
    "load_scf_2022_auto_loan_donor",
    "qualified_auto_loan_interest_proxy",
    "us_scf_auto_loans_signal_gate",
    "us_scf_auto_loans_stage_spec",
    "us_scf_auto_loans_summary",
    "with_us_scf_auto_loan_inputs",
    "SCF_FINANCIAL_ASSET_TARGET_COMPONENTS",
    "SCF_WEALTH_PREDICTORS",
    "US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS",
    "US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS",
    "US_SCF_WEALTH_STAGE_NAME",
    "fetch_scf_2022_summary_extract",
    "impute_us_scf_financial_assets",
    "load_scf_2022_financial_asset_donor",
    "us_scf_wealth_signal_gate",
    "us_scf_wealth_stage_spec",
    "us_scf_wealth_summary",
    "with_us_scf_wealth_inputs",
    "CENSUS_OCCUPATION_CODE_TO_TTOC",
    "SIPP_2023_TIP_DONOR_REVISION",
    "SIPP_2023_TIP_DONOR_SHA256",
    "SIPP_2023_TIP_DONOR_URL",
    "SIPP_TIP_OUTPUT_COLUMNS",
    "SIPP_TIP_PREDICTORS",
    "US_SIPP_TIPS_NONCONSTANT_PERSON_COLUMNS",
    "US_SIPP_TIPS_OUTPUT_COLUMNS",
    "US_SIPP_TIPS_REQUIRED_SOURCE_COLUMNS",
    "US_SIPP_TIPS_STAGE_NAME",
    "derive_treasury_tipped_occupation_code",
    "fetch_sipp_2023_tip_donor",
    "impute_us_sipp_tips",
    "load_sipp_2023_tip_donor",
    "us_sipp_tips_signal_gate",
    "us_sipp_tips_stage_spec",
    "us_sipp_tips_summary",
    "with_us_sipp_tip_inputs",
    "SIPP_2023_VEHICLE_DONOR_REVISION",
    "SIPP_2023_VEHICLE_DONOR_SHA256",
    "SIPP_2023_VEHICLE_DONOR_SIZE_BYTES",
    "SIPP_2023_VEHICLE_DONOR_URL",
    "US_SIPP_VEHICLE_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_SIPP_VEHICLE_OUTPUT_COLUMNS",
    "fetch_sipp_2023_vehicle_donor",
    "load_sipp_2023_vehicle_donor",
    "us_sipp_vehicles_signal_gate",
    "us_sipp_vehicles_stage_spec",
    "us_sipp_vehicles_summary",
    "with_us_sipp_vehicle_inputs",
    "BLS_STATE_UNION_REPRESENTATION_RATE_2024",
    "FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES",
    "FLSA_OVERTIME_OCCUPATION_CODES",
    "ORG_2024_DONOR_CONTENT_SHA256",
    "ORG_2024_DONOR_FILENAME",
    "ORG_PREDICTORS",
    "US_ORG_WAGES_NONCONSTANT_PERSON_COLUMNS",
    "US_ORG_WAGES_OUTPUT_COLUMNS",
    "US_ORG_WAGES_REQUIRED_SOURCE_COLUMNS",
    "US_ORG_WAGES_STAGE_NAME",
    "derive_flsa_overtime_premium",
    "derive_us_org_occupation_inputs",
    "fetch_org_2024_donor",
    "impute_us_org_wages",
    "load_org_2024_donor",
    "us_org_wages_signal_gate",
    "us_org_wages_stage_spec",
    "us_org_wages_summary",
    "with_us_org_wages_inputs",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL",
    "OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL",
    "US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES",
    "US_OTHER_HEALTH_INSURANCE_NONCONSTANT_PERSON_COLUMNS",
    "US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS",
    "US_OTHER_HEALTH_INSURANCE_REQUIRED_SOURCE_COLUMNS",
    "US_OTHER_HEALTH_INSURANCE_STAGE_NAME",
    "derive_us_other_health_insurance_from_asec",
    "derive_us_other_health_insurance_from_manifest",
    "impute_us_other_health_insurance_to_puf_support_from_manifest",
    "us_other_health_insurance_signal_gate",
    "us_other_health_insurance_stage_spec",
    "us_other_health_insurance_summary",
    "with_us_other_health_insurance_inputs",
    "derive_us_immigration_status_from_manifest",
    "us_immigration_composition_gate",
    "us_immigration_composition_summary",
    "us_immigration_stage_spec",
    "with_us_immigration_inputs",
    "US_TAKE_UP_SHARE_BAND",
    "SeededTakeUpResult",
    "US_MEDICAID_ENROLLMENT_SUBSTITUTIONS",
    "US_MEDICAID_ENROLLMENT_TARGET_ROLE",
    "US_MEDICAID_ENROLLMENT_TARGET_TABLE",
    "US_MEDICAID_ENROLLMENT_TOLERANCE",
    "US_MEDICAID_TAKE_UP_ANCHOR",
    "US_MEDICAID_TAKE_UP_STAGE",
    "US_MEDICAID_TAKE_UP_VARIABLE",
    "MedicaidEnrollmentSubstitution",
    "apply_us_medicaid_enrollment_substitutions",
    "count_calibrated_take_up_programs",
    "us_medicaid_take_up_diagnostics",
    "us_medicaid_take_up_gate",
    "us_take_up_participation_diagnostics",
    "us_take_up_signal_gate",
    "us_take_up_summary",
    "with_us_medicaid_take_up",
    "with_us_take_up_inputs",
    "write_us_medicaid_take_up_diagnostics",
    "write_us_take_up_participation_diagnostics",
    "TakeUpContract",
    "TakeUpProgram",
    "assert_take_up_contract_current",
    "assert_take_up_treatments_consistent",
    "load_take_up_contract",
    "seeded_take_up_programs",
    "US_NONNEGATIVE_SOURCE_OUTPUTS",
    "US_SOURCE_COVERAGE",
    "US_SOI_FISCAL_TARGET_SPECS",
    "US_SOI_FISCAL_TARGET_REFERENCES",
    "US_SOURCE_MANIFEST",
    "US_SUPPORT_SPINE_MANIFEST",
    "US_SUPPORT_SPINE_SPEC",
    "US_SOURCE_STAGE_SPECS",
    "US_STAGE_NAMES",
    "PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS",
    "PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS",
    "PUF_TAX_DETAIL_SUPPORT_CHANNEL",
    "US_PUF_SUPPORT_FIT_NAME",
    "US_PUF_SUPPORT_STAGE_NAME",
    "US_STATE_INCOME_TAX_TARGET_SPECS",
    "US_STATE_INCOME_TAX_TARGET_REFERENCES",
    "compile_us_fiscal_target_registry",
    "assign_congressional_districts_to_households",
    "build_pooled_asec_unit_frame",
    "clone_us_frame_for_puf_support",
    "congressional_district_assignment_summary",
    "congressional_district_distribution_from_ledger_facts",
    "derive_us_cps_carried_inputs",
    "disaggregate_us_puf_aggregate_records_from_manifest",
    "hard_target_package_aliases",
    "impute_us_puf_tax_detail_support",
    "in_sample_reform_specs",
    "load_default_reform_specs",
    "load_congressional_district_vintage_crosswalk",
    "load_asec_h5_tables",
    "out_of_sample_reform_specs",
    "puf_tax_unit_donor_from_arrays",
    "pool_asec_sources",
    "reform_validation_payload",
    "source_gap_family_ids",
    "ECPS_PARITY_KNOWN_GAPS_RESOURCE",
    "ECPS_PARITY_REFERENCE_RESOURCE",
    "EcpsParityReference",
    "EcpsParitySource",
    "ParityKnownGap",
    "load_ecps_parity_known_gaps",
    "load_ecps_parity_reference",
    "nonzero_share",
    "us_input_mass_totals",
    "us_nonzero_shares",
    "us_plan",
    "us_source_operation_handlers",
    "write_reform_validation",
    "us_source_coverage_diagnostics",
    "us_source_coverage_gate",
    "us_source_stage_outputs",
    "us_validation_input_coverage_gate",
    "US_VALIDATION_PROVISION_INPUT_LEAVES",
    "ValidationInputLeaf",
    "assert_validation_leaf_registry_current",
    "SSI_COUNTABLE_RESOURCE_ASSETS",
    "POST_REFERENCE_ECPS_REQUIRED_INPUTS",
    "US_RELEASE_INPUT_COVERAGE_RESOURCE",
    "ReformCoverageProbe",
    "ReleaseInputColumn",
    "ReleaseInputCoverageManifest",
    "assert_release_input_coverage_manifest_current",
    "load_release_input_coverage_manifest",
    "us_release_input_coverage_gate",
    "us_release_input_coverage_required_columns",
    "us_release_input_coverage_reviewed_exclusions",
    "us_release_reform_coverage_probes",
    "us_reform_coverage_smoke_gate",
    "us_register_consistency_gate",
    "us_register_contradictions",
    "write_us_source_coverage_diagnostics",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
    "validation_only_family_ids",
    "translate_congressional_district_facts_to_current_vintage",
    "with_household_congressional_districts",
]


@dataclass(frozen=True)
class BuildConfig:
    """The declared knobs of a US build — everything a manifest must record.

    Attributes:
        year: The dataset's time period.
        seed: The build-wide imputation seed.
        max_weight_ratio: The hard calibration bound (part of the dataset's
            provenance; recorded by the calibration's options too).
        calibration_epochs: Solver epochs.
        calibration_learning_rate: Solver learning rate.
        mass: Calibration mass policy (``"free"`` or ``"conserve"``).
        registry_path: Path to the versioned target-registry artifact the
            calibration compiles (see
            :mod:`populace.calibrate.registry`).
        extra: Free-form recorded settings (donor file paths, vintages).
    """

    year: int
    seed: int = 0
    max_weight_ratio: float = 50.0
    calibration_epochs: int = 3000
    calibration_learning_rate: float = 0.15
    mass: str = "free"
    registry_path: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.year < 1990:
            raise ValueError(f"year must be a survey year, got {self.year!r}.")
        if not (self.max_weight_ratio > 0):
            raise ValueError(
                f"max_weight_ratio must be positive, got {self.max_weight_ratio!r}."
            )
        if self.mass not in ("free", "conserve"):
            raise ValueError(f"mass must be 'free' or 'conserve', got {self.mass!r}.")

    def to_manifest(self) -> dict[str, object]:
        """A JSON-ready record for the release manifest."""
        return {
            "year": self.year,
            "seed": self.seed,
            "max_weight_ratio": self.max_weight_ratio,
            "calibration_epochs": self.calibration_epochs,
            "calibration_learning_rate": self.calibration_learning_rate,
            "mass": self.mass,
            "registry_path": self.registry_path,
            "extra": dict(self.extra),
        }


#: The US donor graph: every imputation stage's primary survey, with
#: citations. This is the single place the build's sources are declared —
#: the observatory's sources diagram and the dataset card derive from it.
US_DONORS: Mapping[str, DonorSpec] = {
    "scf_wealth": DonorSpec(
        survey="Fed SCF 2022",
        source="https://www.federalreserve.gov/econres/scfindex.htm",
        notes=(
            "Wealth components and debts; household-grain auto loans use the "
            "full SCF while liquid assets are head-carried to person."
        ),
    ),
    "sipp_tips": DonorSpec(
        survey="Census SIPP",
        source="https://www.census.gov/programs-surveys/sipp.html",
        notes="Tip income for tipped occupations.",
    ),
    "org_wages": DonorSpec(
        survey="CPS ORG",
        source=("https://www2.census.gov/programs-surveys/cps/datasets/2024/basic/"),
        notes=(
            "Hourly-wage labor-market inputs. Donor load failures abort the "
            "build — the silent zero-fallback this stage once had is "
            "structurally impossible under StagePlan."
        ),
    ),
    "meps_esi_premiums": DonorSpec(
        survey="MEPS-IC",
        source="https://meps.ahrq.gov/mepsweb/survey_comp/Insurance.jsp",
        notes="Employer-sponsored insurance premium parameters.",
    ),
    "aca_marketplace_inputs": DonorSpec(
        survey="CPS ASEC + CMS Marketplace Open Enrollment PUFs",
        source="https://www.cms.gov/marketplace/resources/data/public-use-files",
        notes=(
            "Marketplace take-up and selected-plan inputs: CPS reported "
            "Marketplace coverage and premium reports anchor the records; "
            "CMS OEP enrollment, APTC, and metal-level tables provide the "
            "calibration targets."
        ),
    ),
    "medicaid_take_up": DonorSpec(
        survey="CPS ASEC reported coverage + CMS Medicaid monthly enrollment snapshot",
        source="https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360",
        notes=(
            "Medicaid take-up by anchored count-calibration (contract "
            "treatment count_calibrated, populace #331): CPS-reported "
            "Medicaid coverage at interview anchors the flag; the fill is "
            "calibrated to CMS December 2024 state enrollment snapshots. "
            "Point-in-time semantics per #332; heals the #170 "
            "enrollment==eligibility degeneracy."
        ),
    ),
    US_OTHER_HEALTH_INSURANCE_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "Measured PHIP_VAL is reduced by PolicyEngine-calculated CHIP, "
            "Marketplace, and Medicaid premiums after take-up stages; an "
            "ASEC-trained weighted QRF replaces both premium leaves on the "
            "PUF support half."
        ),
    ),
    "prior_year_income": DonorSpec(
        survey="CPS ASEC (prior year)",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes="PERIDNUM longitudinal join for prior-year earnings.",
    ),
    US_IMMIGRATION_STAGE_NAME: DonorSpec(
        survey="CPS ASEC + published unauthorized-population estimates",
        source=(
            "https://www.pewresearch.org/short-reads/2024/07/22/"
            "what-we-know-about-unauthorized-immigrants-living-in-the-us/"
        ),
        notes=(
            "SSN card type and immigration status from ASEC citizenship, "
            "entry-year, nativity, and program-participation fields via the "
            "ASEC-UA residual method (SSRN 4662801), targeted to published "
            "undocumented population/worker/student control totals."
        ),
    ),
    US_HOURS_WORKED_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "Hours-worked inputs mapped directly from measured ASEC person "
            "variables (HRSWK, A_HRS1, WKSWORK) — nothing imputed. Without "
            "them the engine defaults every person to 40 weekly hours and "
            "hours-conditioned rules (SNAP work requirements) become no-ops."
        ),
    ),
    US_SNAP_TAKE_UP_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC + USDA FNS participation rate estimates",
        source="https://www.fns.usda.gov/snap/participation-rates",
        notes=(
            "SNAP take-up: reported recipients (SPM_SNAPSUB) always take "
            "up; non-reporting units drawn to the cited FNS participation "
            "rate. Without it the engine defaults every eligible unit to "
            "100% take-up."
        ),
    ),
    US_ELIGIBILITY_INPUTS_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "SNAP eligibility/exemption inputs mapped directly from "
            "measured ASEC person variables (PEDIS*, A_HSCOL/A_FTPT, "
            "PEPAR1/PEPAR2, VET_VAL, SSI_VAL) — nothing imputed. Without "
            "them disability, student, parent/child, and veteran "
            "exemption channels default to False/0."
        ),
    ),
    US_PREGNANCY_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC + CDC natality-derived national pregnancy rate",
        source="https://www.cdc.gov/nchs/nvss/births.htm",
        notes=(
            "Pregnancy seeded among women 15-44 at the national "
            "point-in-time rate (births x 39/52 over female 15-44 "
            "population), matching the retired pipeline's national "
            "fallback; state-level rates are follow-up work (#351). "
            "The ASEC does not measure pregnancy."
        ),
    ),
    US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC + statutory exemption cap (7 U.S.C. 2015(o)(6))",
        source="https://www.law.cornell.edu/uscode/text/7/2015#o_6",
        notes=(
            "ABAWD discretionary exemptions seeded at the statutory cap "
            "(8% from FY2024) across potentially covered adults 18-64; "
            "the engine intersects with modeled coverage. Assumes full "
            "state usage of the cap (#323)."
        ),
    ),
    US_CHILDCARE_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "Measured replicated SPM_CHILDCAREXPNS is validated and carried "
            "to the SPM-unit childcare leaf. After support expansion, an "
            "ASEC-trained weighted QRF replaces only the PUF half and the "
            "archived first-person reduction places predictions on SPM units."
        ),
    ),
    US_CHILD_SUPPORT_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "Measured annual CSP_VAL receipts and positive CHSP_VAL expenses "
            "are carried directly. After PUF tax-detail imputation, one "
            "ASEC-trained weighted QRF jointly replaces both person leaves "
            "on the PUF support half using the archived predictor subset."
        ),
    ),
    US_DISABILITY_BENEFITS_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "Measured annual DIS_VAL1/DIS_VAL2 benefits are retained only "
            "when their source code is not workers' compensation. After PUF "
            "tax-detail imputation, an ASEC-trained weighted QRF replaces the "
            "CPS-only person leaf on the PUF support half."
        ),
    ),
    "puf_tax_detail": DonorSpec(
        survey="IRS PUF 2015 (uprated)",
        source="https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files",
        notes=(
            "Itemized-deduction detail, versioned processed-PUF Section 199A "
            "simulation leaves (carried without redrawing), partnership SE, "
            "mortgage-interest split, direct E00800/E03500 alimony, direct "
            "E20500 casualty loss, and the E20400 miscellaneous-expense proxy; IRS "
            "disclosure aggregate rows are "
            "disaggregated from raw PUF totals before uprating, with Forbes "
            "top-tail synthesis disabled; support clipped to the PUF's own "
            "realized ranges."
        ),
    ),
    US_EDUCATION_INPUTS_STAGE_NAME: DonorSpec(
        survey="IRS PUF 2015 (uprated) + Census CPS ASEC",
        source="https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files",
        notes=(
            "Qualified tuition comes from the PUF E03230/E87530 maximum; "
            "the published retired path drops the reported AOTC output and its "
            "five affirmative factual inputs therefore follow positive tuition; "
            "educational assistance carries directly from ASEC ED_VAL."
        ),
    ),
    US_RETIREMENT_CONTRIBUTION_STAGE_NAME: DonorSpec(
        survey="Census CPS ASEC + published retirement-contribution shares",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes=(
            "Measured ASEC RETCB_VAL is allocated across five desired "
            "retirement-contribution leaves using archived IRS/BEA/Vanguard/"
            "PSCA shares, then CPS-trained QRF predictions replace the PUF "
            "support half. PolicyEngine-US owns all statutory caps."
        ),
    ),
    "capital_gain_distributions": DonorSpec(
        survey="IRS SOI Sales of Capital Assets (TY2015) + Pub 1304 Table 1.4",
        source=(
            "https://www.irs.gov/statistics/"
            "soi-tax-stats-sales-of-capital-assets-reported-on-individual-tax-returns"
        ),
        notes=(
            "Schedule D line 13 capital gain distributions split out of "
            "long-term gains as a memo component at the national SOCA-derived "
            "share; the direct-1040 route is already a PUF-stage output and "
            "the two routes are mutually exclusive on a real return."
        ),
    ),
    "acs_rent": DonorSpec(
        survey="Census ACS 2022",
        source="https://www.census.gov/programs-surveys/acs",
        notes="Rent for renter households.",
    ),
    "vehicle_assets": DonorSpec(
        survey="Census SIPP",
        source="https://www.census.gov/programs-surveys/sipp.html",
        notes=(
            "Household vehicle count and value from the pinned full 2023 "
            "public-use donor. They remain independent policy inputs until "
            "the full mixed-source net-worth reconciliation is restored."
        ),
    ),
}

#: Stage order of the US build. Derivation stages (no donor) interleave with
#: the donor imputations; the export/calibration stages close the plan.
US_STAGE_NAMES: tuple[str, ...] = (
    "asec_load",
    "unit_assignment",
    "derive_cps_carried",
    US_IMMIGRATION_STAGE_NAME,
    US_HOURS_WORKED_STAGE_NAME,
    US_SNAP_TAKE_UP_STAGE_NAME,
    US_ELIGIBILITY_INPUTS_STAGE_NAME,
    US_PREGNANCY_STAGE_NAME,
    US_SNAP_DISCRETIONARY_EXEMPTION_STAGE_NAME,
    US_RETIREMENT_CONTRIBUTION_STAGE_NAME,
    US_CHILDCARE_STAGE_NAME,
    US_PUF_SUPPORT_STAGE_NAME,
    "puf_tax_detail",
    US_CHILD_SUPPORT_STAGE_NAME,
    US_DISABILITY_BENEFITS_STAGE_NAME,
    US_EDUCATION_INPUTS_STAGE_NAME,
    "capital_gain_distributions",
    "scf_wealth",
    "sipp_tips",
    "org_wages",
    "meps_esi_premiums",
    "prior_year_income",
    "mortgage_conversion",
    "acs_rent",
    "vehicle_assets",
    "entity_placement",
    "aca_marketplace_inputs",
    "medicaid_take_up",
    US_OTHER_HEALTH_INSURANCE_STAGE_NAME,
    "export",
)


def _load_us_source_manifest() -> SourceManifest:
    return load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )


def _load_us_support_spine_manifest() -> SupportSpineManifest:
    return load_support_spine_manifest(
        files("populace.build.us").joinpath("support_spine.json")
    )


US_SOURCE_MANIFEST = _load_us_source_manifest()
US_SUPPORT_SPINE_MANIFEST = _load_us_support_spine_manifest()
US_SUPPORT_SPINE_SPEC: SupportSpineSpec = US_SUPPORT_SPINE_MANIFEST.support_spine
US_SOURCE_STAGE_SPECS: tuple[SourceStageSpec, ...] = US_SOURCE_MANIFEST.stages
US_NONNEGATIVE_SOURCE_OUTPUTS: frozenset[str] = frozenset(
    output for stage in US_SOURCE_STAGE_SPECS for output in stage.nonnegative_outputs
)


def us_plan(
    implementations: Mapping[str, Callable[[Frame], Frame]],
) -> StagePlan:
    """Assemble the US build plan from stage implementations.

    Args:
        implementations: One ``transform(frame) -> Frame`` per stage in
            :data:`US_STAGE_NAMES`. ALL stages must be provided — a missing
            stage refuses to assemble (there are no default or stub
            implementations), and an unknown name is refused too (a typo
            must not silently drop a stage).

    Returns:
        The validated :class:`~populace.build.plan.StagePlan`, with each
        imputation stage carrying its :data:`US_DONORS` citation.

    Raises:
        ValueError: If any declared stage lacks an implementation, or an
            implementation is supplied for an undeclared stage.
    """
    missing = [name for name in US_STAGE_NAMES if name not in implementations]
    if missing:
        raise ValueError(
            f"us_plan needs an implementation for every declared stage; "
            f"missing {missing}. There are no stubs or fallbacks by design."
        )
    unknown = sorted(set(implementations) - set(US_STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown stage implementation(s) {unknown}; declared stages "
            f"are {list(US_STAGE_NAMES)}."
        )
    return StagePlan(
        Stage(
            name=name,
            transform=implementations[name],
            donor=US_DONORS.get(name),
        )
        for name in US_STAGE_NAMES
    )
