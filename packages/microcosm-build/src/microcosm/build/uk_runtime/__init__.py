"""UK build helpers, resolved from their defining modules on explicit access.

Direct helper imports do not initialize unrelated calibration or source loaders.
The public export roster is unchanged; requesting every export with ``import *``
still imports every defining module. Use ``dir`` or ``__all__`` for discovery:
``vars`` contains only exports already resolved in this process.
"""

from importlib import import_module as _import_module
from types import MappingProxyType as _MappingProxyType
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

if _TYPE_CHECKING:
    from microcosm.build.uk_runtime.age_tail import (
        UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE,
        UK_AGE_TAIL_BANDS,
        UK_AGE_TAIL_DECLARED_SEEDS,
        UK_AGE_TOP_CODE,
        UKAgeTailStageTransform,
        disaggregate_uk_age_top_code,
        load_uk_age_tail_band_populations,
    )
    from microcosm.build.uk_runtime.battery_bindings import (
        UK_GATE_REGISTRY,
        UKGateBinding,
    )
    from microcosm.build.uk_runtime.calibration_run import (
        load_bound_spine_sidecar,
        runtime_provenance,
        spine_provenance_from_sidecar,
    )
    from microcosm.build.uk_runtime.cgt_calibration import (
        UK_CGT_ANNUAL_EXEMPT_AMOUNTS,
        UK_CGT_GAINS_AMOUNT_COLUMN,
        UK_CGT_SOURCE_COLUMN,
        UK_CGT_TAXPAYER_COUNT_COLUMN,
        UKCGTTargetMaterialization,
        materialize_uk_cgt_calibration_frame,
        uk_cgt_annual_exempt_amount,
    )
    from microcosm.build.uk_runtime.content_identity import (
        uk_frame_content_identity,
    )
    from microcosm.build.uk_runtime.diagnostics import (
        UK_DIAGNOSTICS_SCHEMA_VERSION,
        UK_TARGET_GEOGRAPHY_LEVELS,
        uk_calibration_diagnostics_payload,
        uk_fit_by_family,
        uk_support_limited_misses,
        uk_weakest_areas_by_fit,
        uk_weakest_families,
        uk_weight_summary,
        uk_zero_weight_strata,
        write_uk_calibration_diagnostics,
    )
    from microcosm.build.uk_runtime.firm_generation import (
        EMPLOYMENT_BANDS,
        HMRC_BAND_COLUMNS,
        INPUT_FILES,
        VAT_LIABILITY_BANDS,
        VINTAGES,
        AxiomVATRuleEvaluator,
        UKFirmCalibrationResult,
        UKFirmGenerationConfig,
        UKFirmGenerationResult,
        UKFirmSourceData,
        UKFirmTargetLayout,
        UKFirmValidationReport,
        UKFirmVATRuleEvaluator,
        assign_employment,
        assign_vat_flags,
        build_firm_target_matrix,
        employment_band_name,
        generate_base_firms,
        generate_input_values,
        generate_uk_firm_population,
        hmrc_band_name,
        map_to_hmrc_band_indices,
        optimize_firm_weights,
        read_uk_firm_source_data,
        solve_firm_weights,
        target_diagnostics,
        uk_firm_source_data_from_frames,
        uk_firm_source_data_from_ledger_facts,
        validate_uk_firm_population,
        write_uk_firm_population,
    )
    from microcosm.build.uk_runtime.fiscal_targets import (
        UK_CGT_REQUIRED_COLUMNS,
        UK_CGT_TARGET_COVERAGE_REQUIREMENTS,
        UK_CGT_TARGET_SPECS,
        UK_FISCAL_TARGET_REGISTRY,
    )
    from microcosm.build.uk_runtime.frs_council_tax import (
        FRS_COUNCIL_TAX_OUTPUT_COLUMNS,
        UKFRSCouncilTaxStageTransform,
        add_frs_council_tax,
        derive_council_tax,
    )
    from microcosm.build.uk_runtime.frs_disability import (
        FRS_DISABILITY_OUTPUT_COLUMNS,
        UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
        UKDWPDisabilityCategoryRates,
        UKDWPDisabilityFlagRates,
        UKFRSDisabilityStageTransform,
        add_frs_disability,
        derive_frs_disability,
        uk_dwp_disability_category_rates,
        uk_dwp_disability_flag_rates,
    )
    from microcosm.build.uk_runtime.frs_education import (
        EDUCQUAL_MAP,
        FRS_EDUCATION_OUTPUT_COLUMNS,
        UKFRSEducationStageTransform,
        add_frs_education,
        derive_current_education,
        derive_frs_education,
    )
    from microcosm.build.uk_runtime.frs_education_grants import (
        FRS_EDUCATION_GRANT_OUTPUT_COLUMNS,
        FRS_EDUCATION_GRANT_REWRITES,
        UK_EDUCATION_GRANT_CAPACITY_PREDICTORS,
        UKDSAPolicy,
        UKFRSEducationGrantSplitStageTransform,
        add_frs_education_grant_split,
        allocate_reported_education_grants,
        disabled_students_allowance_capacity,
        uk_dsa_policy,
    )
    from microcosm.build.uk_runtime.frs_employment import (
        FRS_EMPLOYMENT_OUTPUT_COLUMNS,
        UKFRSEmploymentStageTransform,
        add_frs_employment,
        derive_frs_employment,
    )
    from microcosm.build.uk_runtime.frs_hmrc_leaves import (
        FRS_HMRC_INCPBEN_COLUMN,
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
        FRS_HMRC_PAY_COLUMN,
        FRS_HMRC_RETAINED_LEAF_COLUMNS,
        FRS_HMRC_RETAINED_LEAVES_STAGE_NAME,
        FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
        FRS_HMRC_UBISJA_COLUMN,
        UKFRSHMRCRetainedLeavesResult,
        UKFRSHMRCRetainedLeavesStageTransform,
        retain_uk_frs_hmrc_leaves,
    )
    from microcosm.build.uk_runtime.frs_legacy_proxies import (
        FRS_LEGACY_PROXY_OUTPUT_COLUMNS,
        UK_LEGACY_PROXY_PREDICTORS,
        UKFRSLegacyProxiesStageTransform,
        UKLegacyJSAPolicy,
        derive_frs_legacy_proxies,
        uk_legacy_jsa_policy,
    )
    from microcosm.build.uk_runtime.frs_release import (
        UK_YEAR_RULES,
        UKFRSRelease,
        load_uk_frs_release,
        resolve_uk_year_rule,
    )
    from microcosm.build.uk_runtime.geography_ladder import (
        GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR,
        GEOGRAPHY_LADDER_VINTAGES_ATTR,
        UK_ENGLAND_WALES_REGION_CODES,
        UK_GEOGRAPHY_LADDER_COLUMNS,
        UK_LONDON_REGION_CODE,
        UK_OA_LADDER_DERIVED_LAYERS,
        UK_OA_LADDER_KIND,
        UK_OA_LADDER_SCHEMA_VERSION,
        UkOaLadder,
        assign_uk_geography_ladder,
        expected_uk_ladder_area_support,
        load_uk_oa_ladder,
        uk_geography_ladder_assignment_summary,
        uk_geography_ladder_gate,
        uk_region_mix,
    )
    from microcosm.build.uk_runtime.geography_sources import (
        ENGLAND_LAD_REGION_URL,
        ENGLAND_WALES_OA2021_COUNT,
        EW_OA_CONSTITUENCY_URL,
        EW_OA_HIERARCHY_URL,
        EW_OA_HOUSEHOLDS_URL,
        EW_OA_LAD23_URL,
        EW_OA_POPULATION_URL,
        EW_OA_WARD_URL,
        LAD23_ITL_URL,
        NI_DZ2021_COUNT,
        NI_DZ_GEOJSON_ZIP_URL,
        NI_DZ_HOUSEHOLDS_CSV_URL,
        NI_DZ_LOOKUP_SHEET,
        NI_DZ_PARLCON24_LOOKUP_XLSX_URL,
        NI_DZ_POPULATION_CSV_URL,
        NI_PARLCON24_COUNT,
        SCOTLAND_CENSUS_INDEX_ZIP_URL,
        SCOTLAND_OA2022_COUNT,
        SCOTLAND_OA_CONSTITUENCY_URL,
        SCOTLAND_OA_DZ_IZ_URL,
        SCOTLAND_OA_LAU_ITL_URL,
        SCOTLAND_OA_POPULATION_URL,
        build_complete_uk_geography_crosswalk,
        build_england_wales_crosswalk,
        build_great_britain_crosswalk,
        build_northern_ireland_crosswalk,
        build_official_uk_geography_crosswalk,
        build_scotland_crosswalk,
        load_england_lad_region_lookup,
        load_england_wales_oa_constituencies,
        load_england_wales_oa_hierarchy,
        load_england_wales_oa_households,
        load_england_wales_oa_population,
        load_england_wales_oa_ward_lookup,
        load_ew_oa_lad23_lookup,
        load_lad_itl_lookup,
        load_ni_dz_hierarchy,
        load_ni_dz_households,
        load_ni_dz_parlcon24_lookup,
        load_ni_dz_population,
        load_ni_dz_ward_lookup,
        load_scotland_oa_constituencies,
        load_scotland_oa_dz_iz_lookup,
        load_scotland_oa_households,
        load_scotland_oa_lau_lookup,
        load_scotland_oa_population,
        load_scotland_oa_ward_lookup,
        update_england_wales_lad_codes,
        write_geography_crosswalk,
    )
    from microcosm.build.uk_runtime.hmrc_calibration import (
        DEFAULT_HMRC_CALIBRATION_EPOCHS,
        DEFAULT_HMRC_CALIBRATION_LEARNING_RATE,
        DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR,
        DEFAULT_HMRC_MAX_WEIGHT_RATIO,
        HMRC_ASSESSABLE_INCOME_COLUMN,
        HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN,
        HMRC_TAXPAYER_COLUMN,
        UKHMRCIncomeCalibration,
        UKHMRCTargetMaterialization,
        calibrate_uk_hmrc_income,
        materialize_uk_hmrc_calibration_frame,
    )
    from microcosm.build.uk_runtime.hmrc_income import (
        HMRC_SPI_BUILD_PERIOD,
        HMRC_SPI_COLLATED_ODS_URL,
        HMRC_SPI_INCOME_COMPONENTS,
        HMRC_SPI_PUBLICATION_URL,
        HMRC_SPI_SOURCE_VINTAGE,
        HMRC_SPI_TARGET_RECORD_COUNT,
        HMRCIncomeBandTargetRecord,
        HMRCIncomeSourceProvenance,
        HMRCIncomeTargetSet,
        materialize_hmrc_spi_income_band_targets,
        verify_hmrc_spi_collated_ods,
    )
    from microcosm.build.uk_runtime.hmrc_replay import (
        CANONICAL_HMRC_FACT_FENCES,
        FULL_FRS_TI_BAND_FENCE_ID,
        HMRCFactFence,
        HMRCReplayDiagnosticAggregate,
        HMRCReplayFact,
        HMRCReplayReport,
        build_conservative_hmrc_replay_report,
        classify_hmrc_replay_targets,
        write_hmrc_replay_report,
    )
    from microcosm.build.uk_runtime.hmrc_source_contract import (
        HMRC_DISTRIBUTIONAL_INPUTS,
        UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE,
        assert_uk_hmrc_income_source_contract_current,
    )
    from microcosm.build.uk_runtime.ladder_targets import (
        constituency_household_targets,
        ladder_target_provenance,
        ladder_vs_chronicle_household_dispersion,
        local_authority_household_targets,
    )
    from microcosm.build.uk_runtime.ledger_targets import (
        UK_CENSUS_HOUSEHOLDS_TARGET_ID,
        UK_CROSS_GRAIN_BRIDGES,
        UK_CROSS_GRAIN_GRAIN_PRECEDENCE,
        UK_CROSS_GRAIN_RULE,
        UKFrameTargetAdapter,
        UKLedgerTargetCompilation,
        apply_uk_cross_grain_reconciliation,
        compile_uk_local_target_registry,
        compile_uk_target_registry,
        load_uk_local_area_crosswalk,
        materialize_uk_ledger_targets,
        uk_census_household_uprating,
        uk_ledger_households_total,
        uk_local_target_surface,
    )
    from microcosm.build.uk_runtime.local_doctrine import (
        UK_LOCAL_CLONE_COUNT,
        UK_LOCAL_MAX_WEIGHT_RATIO,
        UK_LOCAL_SOLVE_DOCTRINE,
        UK_LOCAL_SOLVE_EPOCHS,
        UK_LOCAL_TARGET_LOSS_CAP,
        UK_LOCAL_TARGET_WEIGHT_RULE,
        UKLocalSolveDoctrine,
        uk_local_doctrine_with_overrides,
        uk_local_target_loss_weights,
    )
    from microcosm.build.uk_runtime.local_geography import (
        align_area_targets,
    )
    from microcosm.build.uk_runtime.local_rowwise import (
        UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE,
        UK_LOCAL_HOLDOUT_FOLDS,
        UK_LOCAL_HOLDOUT_SEED,
        UKRowwiseDoctrineSolve,
        UKRowwiseLocalMatrix,
        UKRowwiseNationalRows,
        build_uk_rowwise_local_matrix,
        build_uk_rowwise_local_surface_matrix,
        past_cap_census,
        require_adjudicated_uk_local_binding,
        rotated_uk_local_holdout,
        rowwise_area_support_summary,
        rowwise_calibration_mass_reason,
        solve_uk_rowwise_weights_under_doctrine,
        uk_area_support_summary,
        uk_ladder_area_support_summary,
    )
    from microcosm.build.uk_runtime.local_target_census import (
        CENSUS_KIND,
        CENSUS_RESOURCE,
        CENSUS_SCHEMA_VERSION,
        METRIC_STATUS_BOUND_IN_CODE,
        SOURCE_STATUS_DOCUMENTED_UNPINNED,
        assert_uk_local_target_census_current,
        build_uk_local_target_census,
        committed_uk_local_target_census_path,
        load_uk_local_target_census,
        write_uk_local_target_census,
    )
    from microcosm.build.uk_runtime.local_targets import (
        AGE_BANDS,
        AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL,
        AREA_TYPES,
        COUNTRY_TO_REGION,
        INCOME_VARIABLES,
        LA_EXTRA_METRICS,
        area_groups_from_codes,
        compute_household_metrics,
        metric_names,
        metric_names_from_target_profile,
        metric_tables_by_area_group,
    )
    from microcosm.build.uk_runtime.national_calibration import (
        CalibrationFrameAdapter,
        drop_injected_measure_inputs,
        inject_measure_inputs,
        prepare_uk_target_frame,
    )
    from microcosm.build.uk_runtime.national_doctrine import (
        UK_NATIONAL_L0_LAMBDA,
        UK_NATIONAL_LEARNING_RATE,
        UK_NATIONAL_MASS_RULE,
        UK_NATIONAL_MAX_WEIGHT_RATIO,
        UK_NATIONAL_SEED,
        UK_NATIONAL_SOLVE_DOCTRINE,
        UK_NATIONAL_SOLVE_EPOCHS,
        UK_NATIONAL_TARGET_LOSS_CAP,
        UK_NATIONAL_TARGET_WEIGHT_RULE,
        UKNationalSolveDoctrine,
        uk_doctrine_with_overrides,
        uk_national_target_loss_weights,
    )
    from microcosm.build.uk_runtime.national_frame import (
        UK_NATIONAL_SCHEMA,
        UKNationalStage,
        UKStagingProvenance,
        load_uk_national_frame,
        uk_household_weight_kind,
        uk_national_frame,
        uk_time_period,
        validate_uk_national_frame,
        write_uk_national_frame,
    )
    from microcosm.build.uk_runtime.national_sampling import (
        sample_uk_spine_frame,
        uk_spine_source_family_units,
    )
    from microcosm.build.uk_runtime.oa_ladder_sources import (
        LADDER_OA_COLUMNS,
        assemble_uk_oa_ladder,
        concat_uk_ladder_frames,
        join_uk_oa_ladder_layers,
    )
    from microcosm.build.uk_runtime.parity_reference import (
        EFRS_PARITY_KNOWN_GAPS_RESOURCE,
        EFRS_PARITY_REFERENCE_RESOURCE,
        EfrsParityKnownGap,
        EfrsParityReference,
        EfrsParitySource,
        load_efrs_parity_known_gaps,
        load_efrs_parity_reference,
    )
    from microcosm.build.uk_runtime.release_identity import (
        UK_DENSE_RELEASE_ID,
        UK_RELEASE_TIER_CPS_TRANSFER,
        UK_RELEASE_TIER_FRS,
        UK_RELEASE_TIERS,
        UKReleaseIdentity,
        apply_uk_release_identity,
        format_uk_release_id,
        validate_uk_release_tier,
    )
    from microcosm.build.uk_runtime.release_input_coverage import (
        RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS,
        UK_LOADER_INPUT_ALIASES,
        UK_RELEASE_INPUT_COVERAGE_RESOURCE,
        PolicyEngineUKCoverageEngine,
        UKEffectiveMassCoveragePolicy,
        UKReleaseInputColumn,
        UKReleaseInputCoverageManifest,
        assert_uk_release_input_coverage_build_stages,
        assert_uk_release_input_coverage_manifest_current,
        load_uk_release_input_coverage_manifest,
        uk_release_input_coverage_gate,
        uk_release_input_coverage_required_columns,
        uk_release_input_coverage_reviewed_exclusions,
    )
    from microcosm.build.uk_runtime.rowwise_dataset import (
        ARTIFACT_CLONE_INDEX_COLUMN,
        BENUNIT_ID_COLUMNS,
        HOUSEHOLD_ID_COLUMNS,
        MASS_CONSERVATION_RELATIVE_TOLERANCE,
        PERSON_ID_COLUMNS,
        POOL_SOURCE_LINEAGE_COLUMN,
        UK_SINGLE_YEAR_TABLES,
        UKLadderRowwiseDatasetResult,
        UKRowwiseDatasetResult,
        apply_uk_source_lineage_modulus,
        clone_uk_dataset_tables_with_ladder_geography,
        clone_uk_dataset_tables_with_rowwise_geography,
        clone_uk_dataset_with_ladder_geography,
        clone_uk_dataset_with_rowwise_geography,
        ladder_clone_index_column,
        load_uk_rowwise_dataset,
        read_uk_single_year_weight_metadata,
        validate_uk_ladder_rowwise_dataset_tables,
        validate_uk_rowwise_dataset_tables,
        write_uk_rowwise_dataset,
    )
    from microcosm.build.uk_runtime.rowwise_geography import (
        AREA_TYPE_TO_CROSSWALK_COLUMN,
        CROSSWALK_COLUMNS,
        FRS_REGION_TO_COUNTRY,
        FRS_REGION_TO_REGION_CODE,
        ROWWISE_GEOGRAPHY_COLUMNS,
        RowwiseGeographyAssignment,
        assign_household_geography,
        clone_entity_frame,
        expected_uk_rowwise_area_support,
        geography_coverage_summary,
        id_multiplier_for_values,
        prepare_geography_crosswalk,
        validate_geography_coverage,
    )
    from microcosm.build.uk_runtime.size_evaluation import (
        PRE_REGISTERED_OUTCOMES_V1,
        RowwiseRun,
        area_support_tables,
        dense_reference_deltas,
        fit_tables,
        footprint,
        frozen_vs_recomputed,
        gate_table,
        load_run,
        paired_targets,
        run_acceptance,
        summarize,
        weight_tables,
    )
    from microcosm.build.uk_runtime.spi_income import (
        SPI_DONOR_DOI,
        SPI_DONOR_FILENAME,
        SPI_DONOR_SHA256,
        SPI_DONOR_SIZE_BYTES,
        SPI_DONOR_UKDS_STUDY,
        SPI_DONOR_VINTAGE,
        SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
        UKSPIIncomeImputationResult,
        assert_frs_hmrc_auxiliary_crosswalk_available,
        derive_hmrc_income_auxiliaries,
        impute_uk_spi_income_support,
        verify_spi_donor_identity,
    )
    from microcosm.build.uk_runtime.spi_spine import (
        EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN,
        UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS,
        UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME,
        UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME,
        UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND,
        UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS,
        UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS,
        UK_SPI_INCOME_SPINE_REWRITE_COLUMNS,
        UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS,
        UKFRSHMRCSpineLeavesResult,
        UKFRSHMRCSpineLeavesStageTransform,
        UKSPIIncomeSpineResult,
        UKSPIIncomeSpineStageTransform,
        UKSPISupportChannelStageTransform,
    )
    from microcosm.build.uk_runtime.spi_support import (
        BASE_FRS_SUPPORT_CHANNEL,
        DEFAULT_SPI_PRIOR_MASS_SHARE,
        DEFAULT_SPI_SUPPORT_HOUSEHOLDS,
        FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
        FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
        FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
        HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        SPI_INCOME_COMPONENT_COLUMNS,
        SPI_INCOME_IMPUTATION_COLUMNS,
        SPI_PRIOR_MASS_CHANGE_REASON,
        SPI_REPLACEMENT_STRATA_COLUMNS,
        SPI_SYNTHETIC_SUPPORT_CHANNEL,
        UK_SPI_SUPPORT_STAGE_NAME,
        UKSPISupportResult,
        build_uk_spi_support_channel,
        create_uk_spi_support_tables,
        fill_support_channel_from_source,
        replace_uk_spi_support_tables,
        support_channel_column,
        support_clone_index_column,
        support_source_id_column,
    )
    from microcosm.build.uk_runtime.stage_checkpoints import (
        UK_FRAME_METADATA_KEY,
        load_uk_stage_checkpoint,
        load_uk_stage_predecessor,
        uk_stage_metadata,
    )
    from microcosm.build.uk_runtime.terminal_gates import (
        UK_DEFAULT_ZERO_WEIGHT_STRATA,
        UK_MAX_TARGET_ABS_RELATIVE_ERROR,
        UKZeroWeightStratumDeclaration,
        uk_degenerate_release_surface_gate,
        uk_export_surface_gate,
        uk_target_fit_gate,
        uk_target_surface_gate,
        uk_weight_ess_gate,
        uk_weight_ratio_gate,
        uk_zero_weight_strata_gate,
    )
    from microcosm.build.uk_runtime.weighted_integrity import (
        UKInputMassParityPolicy,
        UKInputMassReference,
        UKQRFTailConcentrationPolicy,
        load_uk_input_mass_reference,
        load_uk_local_area_support_exclusion_register,
        load_uk_reviewed_exclusion_register,
        uk_input_mass_parity_gate,
        uk_input_mass_totals,
        uk_qrf_tail_concentration_columns,
        uk_qrf_tail_concentration_gate,
    )

__all__ = [
    "PRE_REGISTERED_OUTCOMES_V1",
    "RowwiseRun",
    "UK_CENSUS_HOUSEHOLDS_TARGET_ID",
    "UK_AGE_TAIL_BANDS",
    "UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE",
    "UK_AGE_TAIL_DECLARED_SEEDS",
    "UK_AGE_TOP_CODE",
    "UKAgeTailStageTransform",
    "disaggregate_uk_age_top_code",
    "load_uk_age_tail_band_populations",
    "ARTIFACT_CLONE_INDEX_COLUMN",
    "CalibrationFrameAdapter",
    "UKGateBinding",
    "UKRowwiseDoctrineSolve",
    "UK_CGT_ANNUAL_EXEMPT_AMOUNTS",
    "UK_GATE_REGISTRY",
    "UK_FRAME_METADATA_KEY",
    "UK_LOCAL_CLONE_COUNT",
    "UK_LOCAL_MAX_WEIGHT_RATIO",
    "UK_LOCAL_SOLVE_EPOCHS",
    "UK_LOCAL_TARGET_WEIGHT_RULE",
    "UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE",
    "UK_LOCAL_HOLDOUT_FOLDS",
    "UK_LOCAL_HOLDOUT_SEED",
    "UK_LOCAL_SOLVE_DOCTRINE",
    "UK_LOCAL_TARGET_LOSS_CAP",
    "UK_CGT_GAINS_AMOUNT_COLUMN",
    "UK_CGT_SOURCE_COLUMN",
    "UK_CGT_TAXPAYER_COUNT_COLUMN",
    "UKCGTTargetMaterialization",
    "ladder_clone_index_column",
    "load_uk_stage_checkpoint",
    "load_uk_stage_predecessor",
    "materialize_uk_cgt_calibration_frame",
    "require_adjudicated_uk_local_binding",
    "rowwise_calibration_mass_reason",
    "runtime_provenance",
    "rotated_uk_local_holdout",
    "sample_uk_spine_frame",
    "uk_cgt_annual_exempt_amount",
    "UK_CGT_REQUIRED_COLUMNS",
    "UK_CGT_TARGET_COVERAGE_REQUIREMENTS",
    "UK_CGT_TARGET_SPECS",
    "UK_DIAGNOSTICS_SCHEMA_VERSION",
    "UK_FISCAL_TARGET_REGISTRY",
    "UKFrameTargetAdapter",
    "UKLedgerTargetCompilation",
    "UK_CROSS_GRAIN_BRIDGES",
    "UK_CROSS_GRAIN_GRAIN_PRECEDENCE",
    "UK_CROSS_GRAIN_RULE",
    "apply_uk_cross_grain_reconciliation",
    "compile_uk_target_registry",
    "materialize_uk_ledger_targets",
    "AGE_BANDS",
    "AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL",
    "AREA_TYPES",
    "AREA_TYPE_TO_CROSSWALK_COLUMN",
    "AxiomVATRuleEvaluator",
    "BASE_FRS_SUPPORT_CHANNEL",
    "BENUNIT_ID_COLUMNS",
    "COUNTRY_TO_REGION",
    "CENSUS_KIND",
    "CENSUS_RESOURCE",
    "CENSUS_SCHEMA_VERSION",
    "CROSSWALK_COLUMNS",
    "DEFAULT_SPI_SUPPORT_HOUSEHOLDS",
    "DEFAULT_SPI_PRIOR_MASS_SHARE",
    "DEFAULT_HMRC_CALIBRATION_EPOCHS",
    "DEFAULT_HMRC_CALIBRATION_LEARNING_RATE",
    "DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR",
    "DEFAULT_HMRC_MAX_WEIGHT_RATIO",
    "EMPLOYMENT_BANDS",
    "METRIC_STATUS_BOUND_IN_CODE",
    "SOURCE_STATUS_DOCUMENTED_UNPINNED",
    "EFRS_PARITY_KNOWN_GAPS_RESOURCE",
    "EFRS_PARITY_REFERENCE_RESOURCE",
    "ENGLAND_LAD_REGION_URL",
    "ENGLAND_WALES_OA2021_COUNT",
    "EW_OA_CONSTITUENCY_URL",
    "EW_OA_HIERARCHY_URL",
    "EW_OA_HOUSEHOLDS_URL",
    "EW_OA_LAD23_URL",
    "EW_OA_POPULATION_URL",
    "EW_OA_WARD_URL",
    "EfrsParityReference",
    "EfrsParitySource",
    "EfrsParityKnownGap",
    "GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR",
    "GEOGRAPHY_LADDER_VINTAGES_ATTR",
    "FRS_HMRC_INCPBEN_COLUMN",
    "FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN",
    "FRS_HMRC_PAY_COLUMN",
    "FRS_HMRC_RETAINED_LEAF_COLUMNS",
    "FRS_HMRC_RETAINED_LEAVES_STAGE_NAME",
    "FRS_HMRC_SRP_REGULAR_CODE5_COLUMN",
    "FRS_HMRC_UBISJA_COLUMN",
    "FRS_REGION_TO_COUNTRY",
    "FRS_REGION_TO_REGION_CODE",
    "FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS",
    "FRS_ONLY_SPI_FILL_PERSON_COLUMNS",
    "FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS",
    "HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN",
    "HOUSEHOLD_ID_COLUMNS",
    "HMRC_BAND_COLUMNS",
    "HMRC_ASSESSABLE_INCOME_COLUMN",
    "HMRC_DISTRIBUTIONAL_INPUTS",
    "HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN",
    "HMRC_SPI_BUILD_PERIOD",
    "HMRC_SPI_COLLATED_ODS_URL",
    "HMRC_SPI_INCOME_COMPONENTS",
    "HMRC_SPI_PUBLICATION_URL",
    "HMRC_SPI_SOURCE_VINTAGE",
    "HMRC_SPI_TARGET_RECORD_COUNT",
    "HMRC_TAXPAYER_COLUMN",
    "HMRCIncomeBandTargetRecord",
    "HMRCIncomeSourceProvenance",
    "HMRCIncomeTargetSet",
    "HMRCFactFence",
    "HMRCReplayDiagnosticAggregate",
    "HMRCReplayFact",
    "HMRCReplayReport",
    "CANONICAL_HMRC_FACT_FENCES",
    "FULL_FRS_TI_BAND_FENCE_ID",
    "INPUT_FILES",
    "INCOME_VARIABLES",
    "LA_EXTRA_METRICS",
    "LADDER_OA_COLUMNS",
    "LAD23_ITL_URL",
    "MASS_CONSERVATION_RELATIVE_TOLERANCE",
    "NI_DZ2021_COUNT",
    "NI_DZ_GEOJSON_ZIP_URL",
    "NI_DZ_HOUSEHOLDS_CSV_URL",
    "NI_DZ_LOOKUP_SHEET",
    "NI_DZ_PARLCON24_LOOKUP_XLSX_URL",
    "NI_DZ_POPULATION_CSV_URL",
    "NI_PARLCON24_COUNT",
    "PERSON_ID_COLUMNS",
    "POOL_SOURCE_LINEAGE_COLUMN",
    "PolicyEngineUKCoverageEngine",
    "ROWWISE_GEOGRAPHY_COLUMNS",
    "RowwiseGeographyAssignment",
    "SCOTLAND_CENSUS_INDEX_ZIP_URL",
    "SCOTLAND_OA2022_COUNT",
    "SCOTLAND_OA_CONSTITUENCY_URL",
    "SCOTLAND_OA_DZ_IZ_URL",
    "SCOTLAND_OA_LAU_ITL_URL",
    "SCOTLAND_OA_POPULATION_URL",
    "SPI_INCOME_COMPONENT_COLUMNS",
    "SPI_INCOME_IMPUTATION_COLUMNS",
    "SPI_DONOR_DOI",
    "SPI_DONOR_FILENAME",
    "SPI_DONOR_SHA256",
    "SPI_DONOR_SIZE_BYTES",
    "SPI_DONOR_UKDS_STUDY",
    "SPI_DONOR_VINTAGE",
    "SPI_PRIOR_MASS_CHANGE_REASON",
    "SPI_REPLACEMENT_STRATA_COLUMNS",
    "SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS",
    "SPI_SYNTHETIC_SUPPORT_CHANNEL",
    "UK_ENGLAND_WALES_REGION_CODES",
    "UK_GEOGRAPHY_LADDER_COLUMNS",
    "UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE",
    "UK_LONDON_REGION_CODE",
    "UK_LOADER_INPUT_ALIASES",
    "UK_OA_LADDER_DERIVED_LAYERS",
    "UK_OA_LADDER_KIND",
    "UK_OA_LADDER_SCHEMA_VERSION",
    "UK_RELEASE_INPUT_COVERAGE_RESOURCE",
    "UK_DENSE_RELEASE_ID",
    "UK_RELEASE_TIERS",
    "UK_RELEASE_TIER_CPS_TRANSFER",
    "UK_RELEASE_TIER_FRS",
    "UkOaLadder",
    "UKFirmCalibrationResult",
    "UKFirmGenerationConfig",
    "UKFirmGenerationResult",
    "UKFirmSourceData",
    "UKFirmTargetLayout",
    "UKFirmVATRuleEvaluator",
    "UKFirmValidationReport",
    "UKFRSHMRCRetainedLeavesResult",
    "UKFRSHMRCRetainedLeavesStageTransform",
    "UKLadderRowwiseDatasetResult",
    "UKLocalSolveDoctrine",
    "UKRowwiseLocalMatrix",
    "UKRowwiseNationalRows",
    "RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS",
    "UKHMRCIncomeCalibration",
    "UKHMRCTargetMaterialization",
    "UKNationalSolveDoctrine",
    "UKStagingProvenance",
    "UK_NATIONAL_L0_LAMBDA",
    "UK_NATIONAL_LEARNING_RATE",
    "UK_NATIONAL_MASS_RULE",
    "UK_NATIONAL_MAX_WEIGHT_RATIO",
    "UK_NATIONAL_SCHEMA",
    "UK_NATIONAL_SEED",
    "UK_NATIONAL_SOLVE_DOCTRINE",
    "UK_NATIONAL_SOLVE_EPOCHS",
    "UK_NATIONAL_TARGET_LOSS_CAP",
    "UK_NATIONAL_TARGET_WEIGHT_RULE",
    "uk_doctrine_with_overrides",
    "uk_national_target_loss_weights",
    "UKNationalStage",
    "UKRowwiseDatasetResult",
    "UKReleaseInputColumn",
    "UKReleaseInputCoverageManifest",
    "UKReleaseIdentity",
    "UKEffectiveMassCoveragePolicy",
    "UKSPISupportResult",
    "UKSPIIncomeImputationResult",
    "UKSPIIncomeSpineResult",
    "UK_SINGLE_YEAR_TABLES",
    "UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME",
    "UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME",
    "UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND",
    "UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS",
    "UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS",
    "UK_SPI_INCOME_SPINE_REWRITE_COLUMNS",
    "UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS",
    "UK_SPI_SUPPORT_STAGE_NAME",
    "UK_TARGET_GEOGRAPHY_LEVELS",
    "VAT_LIABILITY_BANDS",
    "VINTAGES",
    "align_area_targets",
    "apply_uk_release_identity",
    "apply_uk_source_lineage_modulus",
    "area_groups_from_codes",
    "area_support_tables",
    "assemble_uk_oa_ladder",
    "concat_uk_ladder_frames",
    "assert_uk_hmrc_income_source_contract_current",
    "assert_uk_local_target_census_current",
    "assert_uk_release_input_coverage_build_stages",
    "assert_uk_release_input_coverage_manifest_current",
    "assign_employment",
    "assign_household_geography",
    "assert_frs_hmrc_auxiliary_crosswalk_available",
    "derive_hmrc_income_auxiliaries",
    "dense_reference_deltas",
    "assign_uk_geography_ladder",
    "assign_vat_flags",
    "build_firm_target_matrix",
    "build_conservative_hmrc_replay_report",
    "build_uk_spi_support_channel",
    "build_uk_local_target_census",
    "calibrate_uk_hmrc_income",
    "build_complete_uk_geography_crosswalk",
    "build_england_wales_crosswalk",
    "build_great_britain_crosswalk",
    "build_northern_ireland_crosswalk",
    "build_official_uk_geography_crosswalk",
    "build_scotland_crosswalk",
    "build_uk_rowwise_local_matrix",
    "build_uk_rowwise_local_surface_matrix",
    "clone_entity_frame",
    "clone_uk_dataset_tables_with_ladder_geography",
    "expected_uk_ladder_area_support",
    "clone_uk_dataset_tables_with_rowwise_geography",
    "clone_uk_dataset_with_ladder_geography",
    "clone_uk_dataset_with_rowwise_geography",
    "expected_uk_rowwise_area_support",
    "classify_hmrc_replay_targets",
    "compute_household_metrics",
    "constituency_household_targets",
    "ladder_vs_chronicle_household_dispersion",
    "compile_uk_local_target_registry",
    "load_uk_local_area_crosswalk",
    "create_uk_spi_support_tables",
    "derive_council_tax",
    "derive_current_education",
    "derive_frs_disability",
    "derive_frs_education",
    "derive_frs_employment",
    "derive_frs_legacy_proxies",
    "drop_injected_measure_inputs",
    "employment_band_name",
    "fill_support_channel_from_source",
    "fit_tables",
    "footprint",
    "format_uk_release_id",
    "frozen_vs_recomputed",
    "impute_uk_spi_income_support",
    "replace_uk_spi_support_tables",
    "retain_uk_frs_hmrc_leaves",
    "add_frs_council_tax",
    "add_frs_disability",
    "add_frs_education",
    "add_frs_education_grant_split",
    "add_frs_employment",
    "allocate_reported_education_grants",
    "disabled_students_allowance_capacity",
    "generate_base_firms",
    "generate_input_values",
    "generate_uk_firm_population",
    "geography_coverage_summary",
    "gate_table",
    "hmrc_band_name",
    "id_multiplier_for_values",
    "inject_measure_inputs",
    "join_uk_oa_ladder_layers",
    "load_england_lad_region_lookup",
    "load_england_wales_oa_constituencies",
    "load_england_wales_oa_hierarchy",
    "load_england_wales_oa_households",
    "load_england_wales_oa_population",
    "load_england_wales_oa_ward_lookup",
    "load_ew_oa_lad23_lookup",
    "load_efrs_parity_known_gaps",
    "load_efrs_parity_reference",
    "load_lad_itl_lookup",
    "load_run",
    "load_ni_dz_hierarchy",
    "load_ni_dz_households",
    "load_ni_dz_parlcon24_lookup",
    "load_ni_dz_population",
    "load_ni_dz_ward_lookup",
    "load_scotland_oa_constituencies",
    "load_scotland_oa_households",
    "load_scotland_oa_dz_iz_lookup",
    "load_scotland_oa_lau_lookup",
    "load_scotland_oa_population",
    "load_scotland_oa_ward_lookup",
    "load_uk_oa_ladder",
    "local_authority_household_targets",
    "load_uk_local_target_census",
    "ladder_target_provenance",
    "ladder_vs_chronicle_household_dispersion",
    "load_uk_national_frame",
    "load_uk_release_input_coverage_manifest",
    "committed_uk_local_target_census_path",
    "materialize_hmrc_spi_income_band_targets",
    "materialize_uk_hmrc_calibration_frame",
    "map_to_hmrc_band_indices",
    "metric_names",
    "metric_names_from_target_profile",
    "metric_tables_by_area_group",
    "optimize_firm_weights",
    "paired_targets",
    "prepare_uk_target_frame",
    "prepare_geography_crosswalk",
    "read_uk_single_year_weight_metadata",
    "read_uk_firm_source_data",
    "solve_firm_weights",
    "past_cap_census",
    "rowwise_area_support_summary",
    "run_acceptance",
    "solve_uk_rowwise_weights_under_doctrine",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
    "summarize",
    "target_diagnostics",
    "uk_firm_source_data_from_frames",
    "uk_firm_source_data_from_ledger_facts",
    "uk_calibration_diagnostics_payload",
    "uk_fit_by_family",
    "uk_support_limited_misses",
    "uk_spine_source_family_units",
    "uk_frame_content_identity",
    "uk_dsa_policy",
    "uk_dwp_disability_category_rates",
    "uk_dwp_disability_flag_rates",
    "uk_legacy_jsa_policy",
    "uk_geography_ladder_assignment_summary",
    "uk_geography_ladder_gate",
    "uk_region_mix",
    "uk_area_support_summary",
    "uk_ladder_area_support_summary",
    "uk_census_household_uprating",
    "uk_ledger_households_total",
    "uk_local_target_surface",
    "uk_local_doctrine_with_overrides",
    "uk_local_target_loss_weights",
    "uk_release_input_coverage_gate",
    "uk_release_input_coverage_required_columns",
    "uk_release_input_coverage_reviewed_exclusions",
    "uk_stage_metadata",
    "uk_weight_summary",
    "uk_weakest_areas_by_fit",
    "uk_weakest_families",
    "uk_zero_weight_strata",
    "update_england_wales_lad_codes",
    "validate_uk_firm_population",
    "uk_household_weight_kind",
    "uk_national_frame",
    "uk_time_period",
    "validate_uk_national_frame",
    "validate_uk_ladder_rowwise_dataset_tables",
    "validate_uk_rowwise_dataset_tables",
    "validate_uk_release_tier",
    "verify_hmrc_spi_collated_ods",
    "verify_spi_donor_identity",
    "write_uk_local_target_census",
    "write_uk_national_frame",
    "validate_geography_coverage",
    "write_geography_crosswalk",
    "write_uk_firm_population",
    "write_uk_calibration_diagnostics",
    "load_bound_spine_sidecar",
    "spine_provenance_from_sidecar",
    "load_uk_rowwise_dataset",
    "write_uk_rowwise_dataset",
    "weight_tables",
    "write_hmrc_replay_report",
    "UK_DEFAULT_ZERO_WEIGHT_STRATA",
    "EDUCQUAL_MAP",
    "FRS_COUNCIL_TAX_OUTPUT_COLUMNS",
    "FRS_DISABILITY_OUTPUT_COLUMNS",
    "FRS_EDUCATION_GRANT_OUTPUT_COLUMNS",
    "FRS_EDUCATION_GRANT_REWRITES",
    "FRS_EDUCATION_OUTPUT_COLUMNS",
    "FRS_EMPLOYMENT_OUTPUT_COLUMNS",
    "FRS_LEGACY_PROXY_OUTPUT_COLUMNS",
    "EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN",
    "UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS",
    "UK_MAX_TARGET_ABS_RELATIVE_ERROR",
    "UKInputMassParityPolicy",
    "UKDSAPolicy",
    "UKDWPDisabilityCategoryRates",
    "UKDWPDisabilityFlagRates",
    "UKFRSCouncilTaxStageTransform",
    "UKFRSDisabilityStageTransform",
    "UKFRSEducationGrantSplitStageTransform",
    "UKFRSEducationStageTransform",
    "UKFRSEmploymentStageTransform",
    "UKFRSHMRCSpineLeavesResult",
    "UKFRSHMRCSpineLeavesStageTransform",
    "UKFRSLegacyProxiesStageTransform",
    "UKFRSRelease",
    "UKSPIIncomeSpineStageTransform",
    "UKSPISupportChannelStageTransform",
    "UKInputMassReference",
    "UKLegacyJSAPolicy",
    "UK_INTERNAL_DISABILITY_REPORTED_COLUMNS",
    "UK_EDUCATION_GRANT_CAPACITY_PREDICTORS",
    "UK_LEGACY_PROXY_PREDICTORS",
    "UK_YEAR_RULES",
    "UKQRFTailConcentrationPolicy",
    "UKZeroWeightStratumDeclaration",
    "load_uk_frs_release",
    "load_uk_input_mass_reference",
    "load_uk_local_area_support_exclusion_register",
    "load_uk_reviewed_exclusion_register",
    "uk_degenerate_release_surface_gate",
    "uk_export_surface_gate",
    "uk_input_mass_parity_gate",
    "uk_input_mass_totals",
    "uk_qrf_tail_concentration_columns",
    "uk_qrf_tail_concentration_gate",
    "uk_target_fit_gate",
    "uk_target_surface_gate",
    "uk_weight_ess_gate",
    "resolve_uk_year_rule",
    "uk_weight_ratio_gate",
    "uk_zero_weight_strata_gate",
]


_EXPORTS = _MappingProxyType(
    {
        "UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE": (
            "microcosm.build.uk_runtime.age_tail",
            "UK_AGE_TAIL_BAND_POPULATIONS_RESOURCE",
        ),
        "UK_AGE_TAIL_BANDS": (
            "microcosm.build.uk_runtime.age_tail",
            "UK_AGE_TAIL_BANDS",
        ),
        "UK_AGE_TAIL_DECLARED_SEEDS": (
            "microcosm.build.uk_runtime.age_tail",
            "UK_AGE_TAIL_DECLARED_SEEDS",
        ),
        "UK_AGE_TOP_CODE": ("microcosm.build.uk_runtime.age_tail", "UK_AGE_TOP_CODE"),
        "UKAgeTailStageTransform": (
            "microcosm.build.uk_runtime.age_tail",
            "UKAgeTailStageTransform",
        ),
        "disaggregate_uk_age_top_code": (
            "microcosm.build.uk_runtime.age_tail",
            "disaggregate_uk_age_top_code",
        ),
        "load_uk_age_tail_band_populations": (
            "microcosm.build.uk_runtime.age_tail",
            "load_uk_age_tail_band_populations",
        ),
        "UK_GATE_REGISTRY": (
            "microcosm.build.uk_runtime.battery_bindings",
            "UK_GATE_REGISTRY",
        ),
        "UKGateBinding": (
            "microcosm.build.uk_runtime.battery_bindings",
            "UKGateBinding",
        ),
        "load_bound_spine_sidecar": (
            "microcosm.build.uk_runtime.calibration_run",
            "load_bound_spine_sidecar",
        ),
        "runtime_provenance": (
            "microcosm.build.uk_runtime.calibration_run",
            "runtime_provenance",
        ),
        "spine_provenance_from_sidecar": (
            "microcosm.build.uk_runtime.calibration_run",
            "spine_provenance_from_sidecar",
        ),
        "UK_CGT_ANNUAL_EXEMPT_AMOUNTS": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "UK_CGT_ANNUAL_EXEMPT_AMOUNTS",
        ),
        "UK_CGT_GAINS_AMOUNT_COLUMN": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "UK_CGT_GAINS_AMOUNT_COLUMN",
        ),
        "UK_CGT_SOURCE_COLUMN": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "UK_CGT_SOURCE_COLUMN",
        ),
        "UK_CGT_TAXPAYER_COUNT_COLUMN": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "UK_CGT_TAXPAYER_COUNT_COLUMN",
        ),
        "UKCGTTargetMaterialization": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "UKCGTTargetMaterialization",
        ),
        "materialize_uk_cgt_calibration_frame": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "materialize_uk_cgt_calibration_frame",
        ),
        "uk_cgt_annual_exempt_amount": (
            "microcosm.build.uk_runtime.cgt_calibration",
            "uk_cgt_annual_exempt_amount",
        ),
        "uk_frame_content_identity": (
            "microcosm.build.uk_runtime.content_identity",
            "uk_frame_content_identity",
        ),
        "UK_DIAGNOSTICS_SCHEMA_VERSION": (
            "microcosm.build.uk_runtime.diagnostics",
            "UK_DIAGNOSTICS_SCHEMA_VERSION",
        ),
        "UK_TARGET_GEOGRAPHY_LEVELS": (
            "microcosm.build.uk_runtime.diagnostics",
            "UK_TARGET_GEOGRAPHY_LEVELS",
        ),
        "uk_calibration_diagnostics_payload": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_calibration_diagnostics_payload",
        ),
        "uk_fit_by_family": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_fit_by_family",
        ),
        "uk_support_limited_misses": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_support_limited_misses",
        ),
        "uk_weakest_areas_by_fit": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_weakest_areas_by_fit",
        ),
        "uk_weakest_families": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_weakest_families",
        ),
        "uk_weight_summary": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_weight_summary",
        ),
        "uk_zero_weight_strata": (
            "microcosm.build.uk_runtime.diagnostics",
            "uk_zero_weight_strata",
        ),
        "write_uk_calibration_diagnostics": (
            "microcosm.build.uk_runtime.diagnostics",
            "write_uk_calibration_diagnostics",
        ),
        "EMPLOYMENT_BANDS": (
            "microcosm.build.uk_runtime.firm_generation",
            "EMPLOYMENT_BANDS",
        ),
        "HMRC_BAND_COLUMNS": (
            "microcosm.build.uk_runtime.firm_generation",
            "HMRC_BAND_COLUMNS",
        ),
        "INPUT_FILES": ("microcosm.build.uk_runtime.firm_generation", "INPUT_FILES"),
        "VAT_LIABILITY_BANDS": (
            "microcosm.build.uk_runtime.firm_generation",
            "VAT_LIABILITY_BANDS",
        ),
        "VINTAGES": ("microcosm.build.uk_runtime.firm_generation", "VINTAGES"),
        "AxiomVATRuleEvaluator": (
            "microcosm.build.uk_runtime.firm_generation",
            "AxiomVATRuleEvaluator",
        ),
        "UKFirmCalibrationResult": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmCalibrationResult",
        ),
        "UKFirmGenerationConfig": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmGenerationConfig",
        ),
        "UKFirmGenerationResult": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmGenerationResult",
        ),
        "UKFirmSourceData": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmSourceData",
        ),
        "UKFirmTargetLayout": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmTargetLayout",
        ),
        "UKFirmValidationReport": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmValidationReport",
        ),
        "UKFirmVATRuleEvaluator": (
            "microcosm.build.uk_runtime.firm_generation",
            "UKFirmVATRuleEvaluator",
        ),
        "assign_employment": (
            "microcosm.build.uk_runtime.firm_generation",
            "assign_employment",
        ),
        "assign_vat_flags": (
            "microcosm.build.uk_runtime.firm_generation",
            "assign_vat_flags",
        ),
        "build_firm_target_matrix": (
            "microcosm.build.uk_runtime.firm_generation",
            "build_firm_target_matrix",
        ),
        "employment_band_name": (
            "microcosm.build.uk_runtime.firm_generation",
            "employment_band_name",
        ),
        "generate_base_firms": (
            "microcosm.build.uk_runtime.firm_generation",
            "generate_base_firms",
        ),
        "generate_input_values": (
            "microcosm.build.uk_runtime.firm_generation",
            "generate_input_values",
        ),
        "generate_uk_firm_population": (
            "microcosm.build.uk_runtime.firm_generation",
            "generate_uk_firm_population",
        ),
        "hmrc_band_name": (
            "microcosm.build.uk_runtime.firm_generation",
            "hmrc_band_name",
        ),
        "map_to_hmrc_band_indices": (
            "microcosm.build.uk_runtime.firm_generation",
            "map_to_hmrc_band_indices",
        ),
        "optimize_firm_weights": (
            "microcosm.build.uk_runtime.firm_generation",
            "optimize_firm_weights",
        ),
        "read_uk_firm_source_data": (
            "microcosm.build.uk_runtime.firm_generation",
            "read_uk_firm_source_data",
        ),
        "solve_firm_weights": (
            "microcosm.build.uk_runtime.firm_generation",
            "solve_firm_weights",
        ),
        "target_diagnostics": (
            "microcosm.build.uk_runtime.firm_generation",
            "target_diagnostics",
        ),
        "uk_firm_source_data_from_frames": (
            "microcosm.build.uk_runtime.firm_generation",
            "uk_firm_source_data_from_frames",
        ),
        "uk_firm_source_data_from_ledger_facts": (
            "microcosm.build.uk_runtime.firm_generation",
            "uk_firm_source_data_from_ledger_facts",
        ),
        "validate_uk_firm_population": (
            "microcosm.build.uk_runtime.firm_generation",
            "validate_uk_firm_population",
        ),
        "write_uk_firm_population": (
            "microcosm.build.uk_runtime.firm_generation",
            "write_uk_firm_population",
        ),
        "UK_CGT_REQUIRED_COLUMNS": (
            "microcosm.build.uk_runtime.fiscal_targets",
            "UK_CGT_REQUIRED_COLUMNS",
        ),
        "UK_CGT_TARGET_COVERAGE_REQUIREMENTS": (
            "microcosm.build.uk_runtime.fiscal_targets",
            "UK_CGT_TARGET_COVERAGE_REQUIREMENTS",
        ),
        "UK_CGT_TARGET_SPECS": (
            "microcosm.build.uk_runtime.fiscal_targets",
            "UK_CGT_TARGET_SPECS",
        ),
        "UK_FISCAL_TARGET_REGISTRY": (
            "microcosm.build.uk_runtime.fiscal_targets",
            "UK_FISCAL_TARGET_REGISTRY",
        ),
        "FRS_COUNCIL_TAX_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.frs_council_tax",
            "FRS_COUNCIL_TAX_OUTPUT_COLUMNS",
        ),
        "UKFRSCouncilTaxStageTransform": (
            "microcosm.build.uk_runtime.frs_council_tax",
            "UKFRSCouncilTaxStageTransform",
        ),
        "add_frs_council_tax": (
            "microcosm.build.uk_runtime.frs_council_tax",
            "add_frs_council_tax",
        ),
        "derive_council_tax": (
            "microcosm.build.uk_runtime.frs_council_tax",
            "derive_council_tax",
        ),
        "FRS_DISABILITY_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.frs_disability",
            "FRS_DISABILITY_OUTPUT_COLUMNS",
        ),
        "UK_INTERNAL_DISABILITY_REPORTED_COLUMNS": (
            "microcosm.build.uk_runtime.frs_disability",
            "UK_INTERNAL_DISABILITY_REPORTED_COLUMNS",
        ),
        "UKDWPDisabilityCategoryRates": (
            "microcosm.build.uk_runtime.frs_disability",
            "UKDWPDisabilityCategoryRates",
        ),
        "UKDWPDisabilityFlagRates": (
            "microcosm.build.uk_runtime.frs_disability",
            "UKDWPDisabilityFlagRates",
        ),
        "UKFRSDisabilityStageTransform": (
            "microcosm.build.uk_runtime.frs_disability",
            "UKFRSDisabilityStageTransform",
        ),
        "add_frs_disability": (
            "microcosm.build.uk_runtime.frs_disability",
            "add_frs_disability",
        ),
        "derive_frs_disability": (
            "microcosm.build.uk_runtime.frs_disability",
            "derive_frs_disability",
        ),
        "uk_dwp_disability_category_rates": (
            "microcosm.build.uk_runtime.frs_disability",
            "uk_dwp_disability_category_rates",
        ),
        "uk_dwp_disability_flag_rates": (
            "microcosm.build.uk_runtime.frs_disability",
            "uk_dwp_disability_flag_rates",
        ),
        "EDUCQUAL_MAP": ("microcosm.build.uk_runtime.frs_education", "EDUCQUAL_MAP"),
        "FRS_EDUCATION_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.frs_education",
            "FRS_EDUCATION_OUTPUT_COLUMNS",
        ),
        "UKFRSEducationStageTransform": (
            "microcosm.build.uk_runtime.frs_education",
            "UKFRSEducationStageTransform",
        ),
        "add_frs_education": (
            "microcosm.build.uk_runtime.frs_education",
            "add_frs_education",
        ),
        "derive_current_education": (
            "microcosm.build.uk_runtime.frs_education",
            "derive_current_education",
        ),
        "derive_frs_education": (
            "microcosm.build.uk_runtime.frs_education",
            "derive_frs_education",
        ),
        "FRS_EDUCATION_GRANT_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "FRS_EDUCATION_GRANT_OUTPUT_COLUMNS",
        ),
        "FRS_EDUCATION_GRANT_REWRITES": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "FRS_EDUCATION_GRANT_REWRITES",
        ),
        "UK_EDUCATION_GRANT_CAPACITY_PREDICTORS": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "UK_EDUCATION_GRANT_CAPACITY_PREDICTORS",
        ),
        "UKDSAPolicy": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "UKDSAPolicy",
        ),
        "UKFRSEducationGrantSplitStageTransform": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "UKFRSEducationGrantSplitStageTransform",
        ),
        "add_frs_education_grant_split": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "add_frs_education_grant_split",
        ),
        "allocate_reported_education_grants": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "allocate_reported_education_grants",
        ),
        "disabled_students_allowance_capacity": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "disabled_students_allowance_capacity",
        ),
        "uk_dsa_policy": (
            "microcosm.build.uk_runtime.frs_education_grants",
            "uk_dsa_policy",
        ),
        "FRS_EMPLOYMENT_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.frs_employment",
            "FRS_EMPLOYMENT_OUTPUT_COLUMNS",
        ),
        "UKFRSEmploymentStageTransform": (
            "microcosm.build.uk_runtime.frs_employment",
            "UKFRSEmploymentStageTransform",
        ),
        "add_frs_employment": (
            "microcosm.build.uk_runtime.frs_employment",
            "add_frs_employment",
        ),
        "derive_frs_employment": (
            "microcosm.build.uk_runtime.frs_employment",
            "derive_frs_employment",
        ),
        "FRS_HMRC_INCPBEN_COLUMN": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_INCPBEN_COLUMN",
        ),
        "FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN",
        ),
        "FRS_HMRC_PAY_COLUMN": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_PAY_COLUMN",
        ),
        "FRS_HMRC_RETAINED_LEAF_COLUMNS": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_RETAINED_LEAF_COLUMNS",
        ),
        "FRS_HMRC_RETAINED_LEAVES_STAGE_NAME": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_RETAINED_LEAVES_STAGE_NAME",
        ),
        "FRS_HMRC_SRP_REGULAR_CODE5_COLUMN": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_SRP_REGULAR_CODE5_COLUMN",
        ),
        "FRS_HMRC_UBISJA_COLUMN": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "FRS_HMRC_UBISJA_COLUMN",
        ),
        "UKFRSHMRCRetainedLeavesResult": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "UKFRSHMRCRetainedLeavesResult",
        ),
        "UKFRSHMRCRetainedLeavesStageTransform": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "UKFRSHMRCRetainedLeavesStageTransform",
        ),
        "retain_uk_frs_hmrc_leaves": (
            "microcosm.build.uk_runtime.frs_hmrc_leaves",
            "retain_uk_frs_hmrc_leaves",
        ),
        "FRS_LEGACY_PROXY_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.frs_legacy_proxies",
            "FRS_LEGACY_PROXY_OUTPUT_COLUMNS",
        ),
        "UK_LEGACY_PROXY_PREDICTORS": (
            "microcosm.build.uk_runtime.frs_legacy_proxies",
            "UK_LEGACY_PROXY_PREDICTORS",
        ),
        "UKFRSLegacyProxiesStageTransform": (
            "microcosm.build.uk_runtime.frs_legacy_proxies",
            "UKFRSLegacyProxiesStageTransform",
        ),
        "UKLegacyJSAPolicy": (
            "microcosm.build.uk_runtime.frs_legacy_proxies",
            "UKLegacyJSAPolicy",
        ),
        "derive_frs_legacy_proxies": (
            "microcosm.build.uk_runtime.frs_legacy_proxies",
            "derive_frs_legacy_proxies",
        ),
        "uk_legacy_jsa_policy": (
            "microcosm.build.uk_runtime.frs_legacy_proxies",
            "uk_legacy_jsa_policy",
        ),
        "UK_YEAR_RULES": ("microcosm.build.uk_runtime.frs_release", "UK_YEAR_RULES"),
        "UKFRSRelease": ("microcosm.build.uk_runtime.frs_release", "UKFRSRelease"),
        "load_uk_frs_release": (
            "microcosm.build.uk_runtime.frs_release",
            "load_uk_frs_release",
        ),
        "resolve_uk_year_rule": (
            "microcosm.build.uk_runtime.frs_release",
            "resolve_uk_year_rule",
        ),
        "GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR": (
            "microcosm.build.uk_runtime.geography_ladder",
            "GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR",
        ),
        "GEOGRAPHY_LADDER_VINTAGES_ATTR": (
            "microcosm.build.uk_runtime.geography_ladder",
            "GEOGRAPHY_LADDER_VINTAGES_ATTR",
        ),
        "UK_ENGLAND_WALES_REGION_CODES": (
            "microcosm.build.uk_runtime.geography_ladder",
            "UK_ENGLAND_WALES_REGION_CODES",
        ),
        "UK_GEOGRAPHY_LADDER_COLUMNS": (
            "microcosm.build.uk_runtime.geography_ladder",
            "UK_GEOGRAPHY_LADDER_COLUMNS",
        ),
        "UK_LONDON_REGION_CODE": (
            "microcosm.build.uk_runtime.geography_ladder",
            "UK_LONDON_REGION_CODE",
        ),
        "UK_OA_LADDER_DERIVED_LAYERS": (
            "microcosm.build.uk_runtime.geography_ladder",
            "UK_OA_LADDER_DERIVED_LAYERS",
        ),
        "UK_OA_LADDER_KIND": (
            "microcosm.build.uk_runtime.geography_ladder",
            "UK_OA_LADDER_KIND",
        ),
        "UK_OA_LADDER_SCHEMA_VERSION": (
            "microcosm.build.uk_runtime.geography_ladder",
            "UK_OA_LADDER_SCHEMA_VERSION",
        ),
        "UkOaLadder": ("microcosm.build.uk_runtime.geography_ladder", "UkOaLadder"),
        "assign_uk_geography_ladder": (
            "microcosm.build.uk_runtime.geography_ladder",
            "assign_uk_geography_ladder",
        ),
        "expected_uk_ladder_area_support": (
            "microcosm.build.uk_runtime.geography_ladder",
            "expected_uk_ladder_area_support",
        ),
        "load_uk_oa_ladder": (
            "microcosm.build.uk_runtime.geography_ladder",
            "load_uk_oa_ladder",
        ),
        "uk_geography_ladder_assignment_summary": (
            "microcosm.build.uk_runtime.geography_ladder",
            "uk_geography_ladder_assignment_summary",
        ),
        "uk_geography_ladder_gate": (
            "microcosm.build.uk_runtime.geography_ladder",
            "uk_geography_ladder_gate",
        ),
        "uk_region_mix": (
            "microcosm.build.uk_runtime.geography_ladder",
            "uk_region_mix",
        ),
        "ENGLAND_LAD_REGION_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "ENGLAND_LAD_REGION_URL",
        ),
        "ENGLAND_WALES_OA2021_COUNT": (
            "microcosm.build.uk_runtime.geography_sources",
            "ENGLAND_WALES_OA2021_COUNT",
        ),
        "EW_OA_CONSTITUENCY_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "EW_OA_CONSTITUENCY_URL",
        ),
        "EW_OA_HIERARCHY_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "EW_OA_HIERARCHY_URL",
        ),
        "EW_OA_HOUSEHOLDS_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "EW_OA_HOUSEHOLDS_URL",
        ),
        "EW_OA_LAD23_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "EW_OA_LAD23_URL",
        ),
        "EW_OA_POPULATION_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "EW_OA_POPULATION_URL",
        ),
        "EW_OA_WARD_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "EW_OA_WARD_URL",
        ),
        "LAD23_ITL_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "LAD23_ITL_URL",
        ),
        "NI_DZ2021_COUNT": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_DZ2021_COUNT",
        ),
        "NI_DZ_GEOJSON_ZIP_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_DZ_GEOJSON_ZIP_URL",
        ),
        "NI_DZ_HOUSEHOLDS_CSV_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_DZ_HOUSEHOLDS_CSV_URL",
        ),
        "NI_DZ_LOOKUP_SHEET": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_DZ_LOOKUP_SHEET",
        ),
        "NI_DZ_PARLCON24_LOOKUP_XLSX_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_DZ_PARLCON24_LOOKUP_XLSX_URL",
        ),
        "NI_DZ_POPULATION_CSV_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_DZ_POPULATION_CSV_URL",
        ),
        "NI_PARLCON24_COUNT": (
            "microcosm.build.uk_runtime.geography_sources",
            "NI_PARLCON24_COUNT",
        ),
        "SCOTLAND_CENSUS_INDEX_ZIP_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "SCOTLAND_CENSUS_INDEX_ZIP_URL",
        ),
        "SCOTLAND_OA2022_COUNT": (
            "microcosm.build.uk_runtime.geography_sources",
            "SCOTLAND_OA2022_COUNT",
        ),
        "SCOTLAND_OA_CONSTITUENCY_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "SCOTLAND_OA_CONSTITUENCY_URL",
        ),
        "SCOTLAND_OA_DZ_IZ_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "SCOTLAND_OA_DZ_IZ_URL",
        ),
        "SCOTLAND_OA_LAU_ITL_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "SCOTLAND_OA_LAU_ITL_URL",
        ),
        "SCOTLAND_OA_POPULATION_URL": (
            "microcosm.build.uk_runtime.geography_sources",
            "SCOTLAND_OA_POPULATION_URL",
        ),
        "build_complete_uk_geography_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "build_complete_uk_geography_crosswalk",
        ),
        "build_england_wales_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "build_england_wales_crosswalk",
        ),
        "build_great_britain_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "build_great_britain_crosswalk",
        ),
        "build_northern_ireland_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "build_northern_ireland_crosswalk",
        ),
        "build_official_uk_geography_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "build_official_uk_geography_crosswalk",
        ),
        "build_scotland_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "build_scotland_crosswalk",
        ),
        "load_england_lad_region_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_england_lad_region_lookup",
        ),
        "load_england_wales_oa_constituencies": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_england_wales_oa_constituencies",
        ),
        "load_england_wales_oa_hierarchy": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_england_wales_oa_hierarchy",
        ),
        "load_england_wales_oa_households": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_england_wales_oa_households",
        ),
        "load_england_wales_oa_population": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_england_wales_oa_population",
        ),
        "load_england_wales_oa_ward_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_england_wales_oa_ward_lookup",
        ),
        "load_ew_oa_lad23_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_ew_oa_lad23_lookup",
        ),
        "load_lad_itl_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_lad_itl_lookup",
        ),
        "load_ni_dz_hierarchy": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_ni_dz_hierarchy",
        ),
        "load_ni_dz_households": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_ni_dz_households",
        ),
        "load_ni_dz_parlcon24_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_ni_dz_parlcon24_lookup",
        ),
        "load_ni_dz_population": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_ni_dz_population",
        ),
        "load_ni_dz_ward_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_ni_dz_ward_lookup",
        ),
        "load_scotland_oa_constituencies": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_scotland_oa_constituencies",
        ),
        "load_scotland_oa_dz_iz_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_scotland_oa_dz_iz_lookup",
        ),
        "load_scotland_oa_households": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_scotland_oa_households",
        ),
        "load_scotland_oa_lau_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_scotland_oa_lau_lookup",
        ),
        "load_scotland_oa_population": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_scotland_oa_population",
        ),
        "load_scotland_oa_ward_lookup": (
            "microcosm.build.uk_runtime.geography_sources",
            "load_scotland_oa_ward_lookup",
        ),
        "update_england_wales_lad_codes": (
            "microcosm.build.uk_runtime.geography_sources",
            "update_england_wales_lad_codes",
        ),
        "write_geography_crosswalk": (
            "microcosm.build.uk_runtime.geography_sources",
            "write_geography_crosswalk",
        ),
        "DEFAULT_HMRC_CALIBRATION_EPOCHS": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "DEFAULT_HMRC_CALIBRATION_EPOCHS",
        ),
        "DEFAULT_HMRC_CALIBRATION_LEARNING_RATE": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "DEFAULT_HMRC_CALIBRATION_LEARNING_RATE",
        ),
        "DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR",
        ),
        "DEFAULT_HMRC_MAX_WEIGHT_RATIO": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "DEFAULT_HMRC_MAX_WEIGHT_RATIO",
        ),
        "HMRC_ASSESSABLE_INCOME_COLUMN": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "HMRC_ASSESSABLE_INCOME_COLUMN",
        ),
        "HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN",
        ),
        "HMRC_TAXPAYER_COLUMN": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "HMRC_TAXPAYER_COLUMN",
        ),
        "UKHMRCIncomeCalibration": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "UKHMRCIncomeCalibration",
        ),
        "UKHMRCTargetMaterialization": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "UKHMRCTargetMaterialization",
        ),
        "calibrate_uk_hmrc_income": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "calibrate_uk_hmrc_income",
        ),
        "materialize_uk_hmrc_calibration_frame": (
            "microcosm.build.uk_runtime.hmrc_calibration",
            "materialize_uk_hmrc_calibration_frame",
        ),
        "HMRC_SPI_BUILD_PERIOD": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRC_SPI_BUILD_PERIOD",
        ),
        "HMRC_SPI_COLLATED_ODS_URL": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRC_SPI_COLLATED_ODS_URL",
        ),
        "HMRC_SPI_INCOME_COMPONENTS": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRC_SPI_INCOME_COMPONENTS",
        ),
        "HMRC_SPI_PUBLICATION_URL": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRC_SPI_PUBLICATION_URL",
        ),
        "HMRC_SPI_SOURCE_VINTAGE": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRC_SPI_SOURCE_VINTAGE",
        ),
        "HMRC_SPI_TARGET_RECORD_COUNT": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRC_SPI_TARGET_RECORD_COUNT",
        ),
        "HMRCIncomeBandTargetRecord": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRCIncomeBandTargetRecord",
        ),
        "HMRCIncomeSourceProvenance": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRCIncomeSourceProvenance",
        ),
        "HMRCIncomeTargetSet": (
            "microcosm.build.uk_runtime.hmrc_income",
            "HMRCIncomeTargetSet",
        ),
        "materialize_hmrc_spi_income_band_targets": (
            "microcosm.build.uk_runtime.hmrc_income",
            "materialize_hmrc_spi_income_band_targets",
        ),
        "verify_hmrc_spi_collated_ods": (
            "microcosm.build.uk_runtime.hmrc_income",
            "verify_hmrc_spi_collated_ods",
        ),
        "CANONICAL_HMRC_FACT_FENCES": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "CANONICAL_HMRC_FACT_FENCES",
        ),
        "FULL_FRS_TI_BAND_FENCE_ID": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "FULL_FRS_TI_BAND_FENCE_ID",
        ),
        "HMRCFactFence": ("microcosm.build.uk_runtime.hmrc_replay", "HMRCFactFence"),
        "HMRCReplayDiagnosticAggregate": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "HMRCReplayDiagnosticAggregate",
        ),
        "HMRCReplayFact": ("microcosm.build.uk_runtime.hmrc_replay", "HMRCReplayFact"),
        "HMRCReplayReport": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "HMRCReplayReport",
        ),
        "build_conservative_hmrc_replay_report": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "build_conservative_hmrc_replay_report",
        ),
        "classify_hmrc_replay_targets": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "classify_hmrc_replay_targets",
        ),
        "write_hmrc_replay_report": (
            "microcosm.build.uk_runtime.hmrc_replay",
            "write_hmrc_replay_report",
        ),
        "HMRC_DISTRIBUTIONAL_INPUTS": (
            "microcosm.build.uk_runtime.hmrc_source_contract",
            "HMRC_DISTRIBUTIONAL_INPUTS",
        ),
        "UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE": (
            "microcosm.build.uk_runtime.hmrc_source_contract",
            "UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE",
        ),
        "assert_uk_hmrc_income_source_contract_current": (
            "microcosm.build.uk_runtime.hmrc_source_contract",
            "assert_uk_hmrc_income_source_contract_current",
        ),
        "constituency_household_targets": (
            "microcosm.build.uk_runtime.ladder_targets",
            "constituency_household_targets",
        ),
        "ladder_target_provenance": (
            "microcosm.build.uk_runtime.ladder_targets",
            "ladder_target_provenance",
        ),
        "ladder_vs_chronicle_household_dispersion": (
            "microcosm.build.uk_runtime.ladder_targets",
            "ladder_vs_chronicle_household_dispersion",
        ),
        "local_authority_household_targets": (
            "microcosm.build.uk_runtime.ladder_targets",
            "local_authority_household_targets",
        ),
        "UK_CENSUS_HOUSEHOLDS_TARGET_ID": (
            "microcosm.build.uk_runtime.ledger_targets",
            "UK_CENSUS_HOUSEHOLDS_TARGET_ID",
        ),
        "UK_CROSS_GRAIN_BRIDGES": (
            "microcosm.build.uk_runtime.ledger_targets",
            "UK_CROSS_GRAIN_BRIDGES",
        ),
        "UK_CROSS_GRAIN_GRAIN_PRECEDENCE": (
            "microcosm.build.uk_runtime.ledger_targets",
            "UK_CROSS_GRAIN_GRAIN_PRECEDENCE",
        ),
        "UK_CROSS_GRAIN_RULE": (
            "microcosm.build.uk_runtime.ledger_targets",
            "UK_CROSS_GRAIN_RULE",
        ),
        "UKFrameTargetAdapter": (
            "microcosm.build.uk_runtime.ledger_targets",
            "UKFrameTargetAdapter",
        ),
        "UKLedgerTargetCompilation": (
            "microcosm.build.uk_runtime.ledger_targets",
            "UKLedgerTargetCompilation",
        ),
        "apply_uk_cross_grain_reconciliation": (
            "microcosm.build.uk_runtime.ledger_targets",
            "apply_uk_cross_grain_reconciliation",
        ),
        "compile_uk_local_target_registry": (
            "microcosm.build.uk_runtime.ledger_targets",
            "compile_uk_local_target_registry",
        ),
        "compile_uk_target_registry": (
            "microcosm.build.uk_runtime.ledger_targets",
            "compile_uk_target_registry",
        ),
        "load_uk_local_area_crosswalk": (
            "microcosm.build.uk_runtime.ledger_targets",
            "load_uk_local_area_crosswalk",
        ),
        "materialize_uk_ledger_targets": (
            "microcosm.build.uk_runtime.ledger_targets",
            "materialize_uk_ledger_targets",
        ),
        "uk_census_household_uprating": (
            "microcosm.build.uk_runtime.ledger_targets",
            "uk_census_household_uprating",
        ),
        "uk_ledger_households_total": (
            "microcosm.build.uk_runtime.ledger_targets",
            "uk_ledger_households_total",
        ),
        "uk_local_target_surface": (
            "microcosm.build.uk_runtime.ledger_targets",
            "uk_local_target_surface",
        ),
        "UK_LOCAL_CLONE_COUNT": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UK_LOCAL_CLONE_COUNT",
        ),
        "UK_LOCAL_MAX_WEIGHT_RATIO": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UK_LOCAL_MAX_WEIGHT_RATIO",
        ),
        "UK_LOCAL_SOLVE_DOCTRINE": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UK_LOCAL_SOLVE_DOCTRINE",
        ),
        "UK_LOCAL_SOLVE_EPOCHS": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UK_LOCAL_SOLVE_EPOCHS",
        ),
        "UK_LOCAL_TARGET_LOSS_CAP": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UK_LOCAL_TARGET_LOSS_CAP",
        ),
        "UK_LOCAL_TARGET_WEIGHT_RULE": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UK_LOCAL_TARGET_WEIGHT_RULE",
        ),
        "UKLocalSolveDoctrine": (
            "microcosm.build.uk_runtime.local_doctrine",
            "UKLocalSolveDoctrine",
        ),
        "uk_local_doctrine_with_overrides": (
            "microcosm.build.uk_runtime.local_doctrine",
            "uk_local_doctrine_with_overrides",
        ),
        "uk_local_target_loss_weights": (
            "microcosm.build.uk_runtime.local_doctrine",
            "uk_local_target_loss_weights",
        ),
        "align_area_targets": (
            "microcosm.build.uk_runtime.local_geography",
            "align_area_targets",
        ),
        "UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE": (
            "microcosm.build.uk_runtime.local_rowwise",
            "UK_LOCAL_BINDING_ADJUDICATION_REGISTER_RESOURCE",
        ),
        "UK_LOCAL_HOLDOUT_FOLDS": (
            "microcosm.build.uk_runtime.local_rowwise",
            "UK_LOCAL_HOLDOUT_FOLDS",
        ),
        "UK_LOCAL_HOLDOUT_SEED": (
            "microcosm.build.uk_runtime.local_rowwise",
            "UK_LOCAL_HOLDOUT_SEED",
        ),
        "UKRowwiseDoctrineSolve": (
            "microcosm.build.uk_runtime.local_rowwise",
            "UKRowwiseDoctrineSolve",
        ),
        "UKRowwiseLocalMatrix": (
            "microcosm.build.uk_runtime.local_rowwise",
            "UKRowwiseLocalMatrix",
        ),
        "UKRowwiseNationalRows": (
            "microcosm.build.uk_runtime.local_rowwise",
            "UKRowwiseNationalRows",
        ),
        "build_uk_rowwise_local_matrix": (
            "microcosm.build.uk_runtime.local_rowwise",
            "build_uk_rowwise_local_matrix",
        ),
        "build_uk_rowwise_local_surface_matrix": (
            "microcosm.build.uk_runtime.local_rowwise",
            "build_uk_rowwise_local_surface_matrix",
        ),
        "past_cap_census": (
            "microcosm.build.uk_runtime.local_rowwise",
            "past_cap_census",
        ),
        "require_adjudicated_uk_local_binding": (
            "microcosm.build.uk_runtime.local_rowwise",
            "require_adjudicated_uk_local_binding",
        ),
        "rotated_uk_local_holdout": (
            "microcosm.build.uk_runtime.local_rowwise",
            "rotated_uk_local_holdout",
        ),
        "rowwise_area_support_summary": (
            "microcosm.build.uk_runtime.local_rowwise",
            "rowwise_area_support_summary",
        ),
        "rowwise_calibration_mass_reason": (
            "microcosm.build.uk_runtime.local_rowwise",
            "rowwise_calibration_mass_reason",
        ),
        "solve_uk_rowwise_weights_under_doctrine": (
            "microcosm.build.uk_runtime.local_rowwise",
            "solve_uk_rowwise_weights_under_doctrine",
        ),
        "uk_area_support_summary": (
            "microcosm.build.uk_runtime.local_rowwise",
            "uk_area_support_summary",
        ),
        "uk_ladder_area_support_summary": (
            "microcosm.build.uk_runtime.local_rowwise",
            "uk_ladder_area_support_summary",
        ),
        "CENSUS_KIND": (
            "microcosm.build.uk_runtime.local_target_census",
            "CENSUS_KIND",
        ),
        "CENSUS_RESOURCE": (
            "microcosm.build.uk_runtime.local_target_census",
            "CENSUS_RESOURCE",
        ),
        "CENSUS_SCHEMA_VERSION": (
            "microcosm.build.uk_runtime.local_target_census",
            "CENSUS_SCHEMA_VERSION",
        ),
        "METRIC_STATUS_BOUND_IN_CODE": (
            "microcosm.build.uk_runtime.local_target_census",
            "METRIC_STATUS_BOUND_IN_CODE",
        ),
        "SOURCE_STATUS_DOCUMENTED_UNPINNED": (
            "microcosm.build.uk_runtime.local_target_census",
            "SOURCE_STATUS_DOCUMENTED_UNPINNED",
        ),
        "assert_uk_local_target_census_current": (
            "microcosm.build.uk_runtime.local_target_census",
            "assert_uk_local_target_census_current",
        ),
        "build_uk_local_target_census": (
            "microcosm.build.uk_runtime.local_target_census",
            "build_uk_local_target_census",
        ),
        "committed_uk_local_target_census_path": (
            "microcosm.build.uk_runtime.local_target_census",
            "committed_uk_local_target_census_path",
        ),
        "load_uk_local_target_census": (
            "microcosm.build.uk_runtime.local_target_census",
            "load_uk_local_target_census",
        ),
        "write_uk_local_target_census": (
            "microcosm.build.uk_runtime.local_target_census",
            "write_uk_local_target_census",
        ),
        "AGE_BANDS": ("microcosm.build.uk_runtime.local_targets", "AGE_BANDS"),
        "AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL": (
            "microcosm.build.uk_runtime.local_targets",
            "AREA_TYPE_TO_LEDGER_GEOGRAPHY_LEVEL",
        ),
        "AREA_TYPES": ("microcosm.build.uk_runtime.local_targets", "AREA_TYPES"),
        "COUNTRY_TO_REGION": (
            "microcosm.build.uk_runtime.local_targets",
            "COUNTRY_TO_REGION",
        ),
        "INCOME_VARIABLES": (
            "microcosm.build.uk_runtime.local_targets",
            "INCOME_VARIABLES",
        ),
        "LA_EXTRA_METRICS": (
            "microcosm.build.uk_runtime.local_targets",
            "LA_EXTRA_METRICS",
        ),
        "area_groups_from_codes": (
            "microcosm.build.uk_runtime.local_targets",
            "area_groups_from_codes",
        ),
        "compute_household_metrics": (
            "microcosm.build.uk_runtime.local_targets",
            "compute_household_metrics",
        ),
        "metric_names": ("microcosm.build.uk_runtime.local_targets", "metric_names"),
        "metric_names_from_target_profile": (
            "microcosm.build.uk_runtime.local_targets",
            "metric_names_from_target_profile",
        ),
        "metric_tables_by_area_group": (
            "microcosm.build.uk_runtime.local_targets",
            "metric_tables_by_area_group",
        ),
        "CalibrationFrameAdapter": (
            "microcosm.build.uk_runtime.national_calibration",
            "CalibrationFrameAdapter",
        ),
        "drop_injected_measure_inputs": (
            "microcosm.build.uk_runtime.national_calibration",
            "drop_injected_measure_inputs",
        ),
        "inject_measure_inputs": (
            "microcosm.build.uk_runtime.national_calibration",
            "inject_measure_inputs",
        ),
        "prepare_uk_target_frame": (
            "microcosm.build.uk_runtime.national_calibration",
            "prepare_uk_target_frame",
        ),
        "UK_NATIONAL_L0_LAMBDA": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_L0_LAMBDA",
        ),
        "UK_NATIONAL_LEARNING_RATE": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_LEARNING_RATE",
        ),
        "UK_NATIONAL_MASS_RULE": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_MASS_RULE",
        ),
        "UK_NATIONAL_MAX_WEIGHT_RATIO": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_MAX_WEIGHT_RATIO",
        ),
        "UK_NATIONAL_SEED": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_SEED",
        ),
        "UK_NATIONAL_SOLVE_DOCTRINE": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_SOLVE_DOCTRINE",
        ),
        "UK_NATIONAL_SOLVE_EPOCHS": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_SOLVE_EPOCHS",
        ),
        "UK_NATIONAL_TARGET_LOSS_CAP": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_TARGET_LOSS_CAP",
        ),
        "UK_NATIONAL_TARGET_WEIGHT_RULE": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UK_NATIONAL_TARGET_WEIGHT_RULE",
        ),
        "UKNationalSolveDoctrine": (
            "microcosm.build.uk_runtime.national_doctrine",
            "UKNationalSolveDoctrine",
        ),
        "uk_doctrine_with_overrides": (
            "microcosm.build.uk_runtime.national_doctrine",
            "uk_doctrine_with_overrides",
        ),
        "uk_national_target_loss_weights": (
            "microcosm.build.uk_runtime.national_doctrine",
            "uk_national_target_loss_weights",
        ),
        "UK_NATIONAL_SCHEMA": (
            "microcosm.build.uk_runtime.national_frame",
            "UK_NATIONAL_SCHEMA",
        ),
        "UKNationalStage": (
            "microcosm.build.uk_runtime.national_frame",
            "UKNationalStage",
        ),
        "UKStagingProvenance": (
            "microcosm.build.uk_runtime.national_frame",
            "UKStagingProvenance",
        ),
        "load_uk_national_frame": (
            "microcosm.build.uk_runtime.national_frame",
            "load_uk_national_frame",
        ),
        "uk_household_weight_kind": (
            "microcosm.build.uk_runtime.national_frame",
            "uk_household_weight_kind",
        ),
        "uk_national_frame": (
            "microcosm.build.uk_runtime.national_frame",
            "uk_national_frame",
        ),
        "uk_time_period": (
            "microcosm.build.uk_runtime.national_frame",
            "uk_time_period",
        ),
        "validate_uk_national_frame": (
            "microcosm.build.uk_runtime.national_frame",
            "validate_uk_national_frame",
        ),
        "write_uk_national_frame": (
            "microcosm.build.uk_runtime.national_frame",
            "write_uk_national_frame",
        ),
        "sample_uk_spine_frame": (
            "microcosm.build.uk_runtime.national_sampling",
            "sample_uk_spine_frame",
        ),
        "uk_spine_source_family_units": (
            "microcosm.build.uk_runtime.national_sampling",
            "uk_spine_source_family_units",
        ),
        "LADDER_OA_COLUMNS": (
            "microcosm.build.uk_runtime.oa_ladder_sources",
            "LADDER_OA_COLUMNS",
        ),
        "assemble_uk_oa_ladder": (
            "microcosm.build.uk_runtime.oa_ladder_sources",
            "assemble_uk_oa_ladder",
        ),
        "concat_uk_ladder_frames": (
            "microcosm.build.uk_runtime.oa_ladder_sources",
            "concat_uk_ladder_frames",
        ),
        "join_uk_oa_ladder_layers": (
            "microcosm.build.uk_runtime.oa_ladder_sources",
            "join_uk_oa_ladder_layers",
        ),
        "EFRS_PARITY_KNOWN_GAPS_RESOURCE": (
            "microcosm.build.uk_runtime.parity_reference",
            "EFRS_PARITY_KNOWN_GAPS_RESOURCE",
        ),
        "EFRS_PARITY_REFERENCE_RESOURCE": (
            "microcosm.build.uk_runtime.parity_reference",
            "EFRS_PARITY_REFERENCE_RESOURCE",
        ),
        "EfrsParityKnownGap": (
            "microcosm.build.uk_runtime.parity_reference",
            "EfrsParityKnownGap",
        ),
        "EfrsParityReference": (
            "microcosm.build.uk_runtime.parity_reference",
            "EfrsParityReference",
        ),
        "EfrsParitySource": (
            "microcosm.build.uk_runtime.parity_reference",
            "EfrsParitySource",
        ),
        "load_efrs_parity_known_gaps": (
            "microcosm.build.uk_runtime.parity_reference",
            "load_efrs_parity_known_gaps",
        ),
        "load_efrs_parity_reference": (
            "microcosm.build.uk_runtime.parity_reference",
            "load_efrs_parity_reference",
        ),
        "UK_DENSE_RELEASE_ID": (
            "microcosm.build.uk_runtime.release_identity",
            "UK_DENSE_RELEASE_ID",
        ),
        "UK_RELEASE_TIER_CPS_TRANSFER": (
            "microcosm.build.uk_runtime.release_identity",
            "UK_RELEASE_TIER_CPS_TRANSFER",
        ),
        "UK_RELEASE_TIER_FRS": (
            "microcosm.build.uk_runtime.release_identity",
            "UK_RELEASE_TIER_FRS",
        ),
        "UK_RELEASE_TIERS": (
            "microcosm.build.uk_runtime.release_identity",
            "UK_RELEASE_TIERS",
        ),
        "UKReleaseIdentity": (
            "microcosm.build.uk_runtime.release_identity",
            "UKReleaseIdentity",
        ),
        "apply_uk_release_identity": (
            "microcosm.build.uk_runtime.release_identity",
            "apply_uk_release_identity",
        ),
        "format_uk_release_id": (
            "microcosm.build.uk_runtime.release_identity",
            "format_uk_release_id",
        ),
        "validate_uk_release_tier": (
            "microcosm.build.uk_runtime.release_identity",
            "validate_uk_release_tier",
        ),
        "RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS",
        ),
        "UK_LOADER_INPUT_ALIASES": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "UK_LOADER_INPUT_ALIASES",
        ),
        "UK_RELEASE_INPUT_COVERAGE_RESOURCE": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "UK_RELEASE_INPUT_COVERAGE_RESOURCE",
        ),
        "PolicyEngineUKCoverageEngine": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "PolicyEngineUKCoverageEngine",
        ),
        "UKEffectiveMassCoveragePolicy": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "UKEffectiveMassCoveragePolicy",
        ),
        "UKReleaseInputColumn": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "UKReleaseInputColumn",
        ),
        "UKReleaseInputCoverageManifest": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "UKReleaseInputCoverageManifest",
        ),
        "assert_uk_release_input_coverage_build_stages": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "assert_uk_release_input_coverage_build_stages",
        ),
        "assert_uk_release_input_coverage_manifest_current": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "assert_uk_release_input_coverage_manifest_current",
        ),
        "load_uk_release_input_coverage_manifest": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "load_uk_release_input_coverage_manifest",
        ),
        "uk_release_input_coverage_gate": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "uk_release_input_coverage_gate",
        ),
        "uk_release_input_coverage_required_columns": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "uk_release_input_coverage_required_columns",
        ),
        "uk_release_input_coverage_reviewed_exclusions": (
            "microcosm.build.uk_runtime.release_input_coverage",
            "uk_release_input_coverage_reviewed_exclusions",
        ),
        "ARTIFACT_CLONE_INDEX_COLUMN": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "ARTIFACT_CLONE_INDEX_COLUMN",
        ),
        "BENUNIT_ID_COLUMNS": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "BENUNIT_ID_COLUMNS",
        ),
        "HOUSEHOLD_ID_COLUMNS": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "HOUSEHOLD_ID_COLUMNS",
        ),
        "MASS_CONSERVATION_RELATIVE_TOLERANCE": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "MASS_CONSERVATION_RELATIVE_TOLERANCE",
        ),
        "PERSON_ID_COLUMNS": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "PERSON_ID_COLUMNS",
        ),
        "POOL_SOURCE_LINEAGE_COLUMN": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "POOL_SOURCE_LINEAGE_COLUMN",
        ),
        "UK_SINGLE_YEAR_TABLES": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "UK_SINGLE_YEAR_TABLES",
        ),
        "UKLadderRowwiseDatasetResult": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "UKLadderRowwiseDatasetResult",
        ),
        "UKRowwiseDatasetResult": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "UKRowwiseDatasetResult",
        ),
        "apply_uk_source_lineage_modulus": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "apply_uk_source_lineage_modulus",
        ),
        "clone_uk_dataset_tables_with_ladder_geography": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "clone_uk_dataset_tables_with_ladder_geography",
        ),
        "clone_uk_dataset_tables_with_rowwise_geography": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "clone_uk_dataset_tables_with_rowwise_geography",
        ),
        "clone_uk_dataset_with_ladder_geography": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "clone_uk_dataset_with_ladder_geography",
        ),
        "clone_uk_dataset_with_rowwise_geography": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "clone_uk_dataset_with_rowwise_geography",
        ),
        "ladder_clone_index_column": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "ladder_clone_index_column",
        ),
        "load_uk_rowwise_dataset": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "load_uk_rowwise_dataset",
        ),
        "read_uk_single_year_weight_metadata": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "read_uk_single_year_weight_metadata",
        ),
        "validate_uk_ladder_rowwise_dataset_tables": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "validate_uk_ladder_rowwise_dataset_tables",
        ),
        "validate_uk_rowwise_dataset_tables": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "validate_uk_rowwise_dataset_tables",
        ),
        "write_uk_rowwise_dataset": (
            "microcosm.build.uk_runtime.rowwise_dataset",
            "write_uk_rowwise_dataset",
        ),
        "AREA_TYPE_TO_CROSSWALK_COLUMN": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "AREA_TYPE_TO_CROSSWALK_COLUMN",
        ),
        "CROSSWALK_COLUMNS": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "CROSSWALK_COLUMNS",
        ),
        "FRS_REGION_TO_COUNTRY": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "FRS_REGION_TO_COUNTRY",
        ),
        "FRS_REGION_TO_REGION_CODE": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "FRS_REGION_TO_REGION_CODE",
        ),
        "ROWWISE_GEOGRAPHY_COLUMNS": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "ROWWISE_GEOGRAPHY_COLUMNS",
        ),
        "RowwiseGeographyAssignment": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "RowwiseGeographyAssignment",
        ),
        "assign_household_geography": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "assign_household_geography",
        ),
        "clone_entity_frame": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "clone_entity_frame",
        ),
        "expected_uk_rowwise_area_support": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "expected_uk_rowwise_area_support",
        ),
        "geography_coverage_summary": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "geography_coverage_summary",
        ),
        "id_multiplier_for_values": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "id_multiplier_for_values",
        ),
        "prepare_geography_crosswalk": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "prepare_geography_crosswalk",
        ),
        "validate_geography_coverage": (
            "microcosm.build.uk_runtime.rowwise_geography",
            "validate_geography_coverage",
        ),
        "PRE_REGISTERED_OUTCOMES_V1": (
            "microcosm.build.uk_runtime.size_evaluation",
            "PRE_REGISTERED_OUTCOMES_V1",
        ),
        "RowwiseRun": ("microcosm.build.uk_runtime.size_evaluation", "RowwiseRun"),
        "area_support_tables": (
            "microcosm.build.uk_runtime.size_evaluation",
            "area_support_tables",
        ),
        "dense_reference_deltas": (
            "microcosm.build.uk_runtime.size_evaluation",
            "dense_reference_deltas",
        ),
        "fit_tables": ("microcosm.build.uk_runtime.size_evaluation", "fit_tables"),
        "footprint": ("microcosm.build.uk_runtime.size_evaluation", "footprint"),
        "frozen_vs_recomputed": (
            "microcosm.build.uk_runtime.size_evaluation",
            "frozen_vs_recomputed",
        ),
        "gate_table": ("microcosm.build.uk_runtime.size_evaluation", "gate_table"),
        "load_run": ("microcosm.build.uk_runtime.size_evaluation", "load_run"),
        "paired_targets": (
            "microcosm.build.uk_runtime.size_evaluation",
            "paired_targets",
        ),
        "run_acceptance": (
            "microcosm.build.uk_runtime.size_evaluation",
            "run_acceptance",
        ),
        "summarize": ("microcosm.build.uk_runtime.size_evaluation", "summarize"),
        "weight_tables": (
            "microcosm.build.uk_runtime.size_evaluation",
            "weight_tables",
        ),
        "SPI_DONOR_DOI": ("microcosm.build.uk_runtime.spi_income", "SPI_DONOR_DOI"),
        "SPI_DONOR_FILENAME": (
            "microcosm.build.uk_runtime.spi_income",
            "SPI_DONOR_FILENAME",
        ),
        "SPI_DONOR_SHA256": (
            "microcosm.build.uk_runtime.spi_income",
            "SPI_DONOR_SHA256",
        ),
        "SPI_DONOR_SIZE_BYTES": (
            "microcosm.build.uk_runtime.spi_income",
            "SPI_DONOR_SIZE_BYTES",
        ),
        "SPI_DONOR_UKDS_STUDY": (
            "microcosm.build.uk_runtime.spi_income",
            "SPI_DONOR_UKDS_STUDY",
        ),
        "SPI_DONOR_VINTAGE": (
            "microcosm.build.uk_runtime.spi_income",
            "SPI_DONOR_VINTAGE",
        ),
        "SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS": (
            "microcosm.build.uk_runtime.spi_income",
            "SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS",
        ),
        "UKSPIIncomeImputationResult": (
            "microcosm.build.uk_runtime.spi_income",
            "UKSPIIncomeImputationResult",
        ),
        "assert_frs_hmrc_auxiliary_crosswalk_available": (
            "microcosm.build.uk_runtime.spi_income",
            "assert_frs_hmrc_auxiliary_crosswalk_available",
        ),
        "derive_hmrc_income_auxiliaries": (
            "microcosm.build.uk_runtime.spi_income",
            "derive_hmrc_income_auxiliaries",
        ),
        "impute_uk_spi_income_support": (
            "microcosm.build.uk_runtime.spi_income",
            "impute_uk_spi_income_support",
        ),
        "verify_spi_donor_identity": (
            "microcosm.build.uk_runtime.spi_income",
            "verify_spi_donor_identity",
        ),
        "EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN": (
            "microcosm.build.uk_runtime.spi_spine",
            "EMPLOYER_PENSION_CONTRIBUTIONS_COLUMN",
        ),
        "UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS",
        ),
        "UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_FRS_HMRC_SPINE_LEAVES_STAGE_NAME",
        ),
        "UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_HMRC_SPI_INCOME_SPINE_STAGE_NAME",
        ),
        "UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_HMRC_SPI_SPINE_REPLAY_REPORT_KIND",
        ),
        "UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS",
        ),
        "UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS",
        ),
        "UK_SPI_INCOME_SPINE_REWRITE_COLUMNS": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_SPI_INCOME_SPINE_REWRITE_COLUMNS",
        ),
        "UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS": (
            "microcosm.build.uk_runtime.spi_spine",
            "UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS",
        ),
        "UKFRSHMRCSpineLeavesResult": (
            "microcosm.build.uk_runtime.spi_spine",
            "UKFRSHMRCSpineLeavesResult",
        ),
        "UKFRSHMRCSpineLeavesStageTransform": (
            "microcosm.build.uk_runtime.spi_spine",
            "UKFRSHMRCSpineLeavesStageTransform",
        ),
        "UKSPIIncomeSpineResult": (
            "microcosm.build.uk_runtime.spi_spine",
            "UKSPIIncomeSpineResult",
        ),
        "UKSPIIncomeSpineStageTransform": (
            "microcosm.build.uk_runtime.spi_spine",
            "UKSPIIncomeSpineStageTransform",
        ),
        "UKSPISupportChannelStageTransform": (
            "microcosm.build.uk_runtime.spi_spine",
            "UKSPISupportChannelStageTransform",
        ),
        "BASE_FRS_SUPPORT_CHANNEL": (
            "microcosm.build.uk_runtime.spi_support",
            "BASE_FRS_SUPPORT_CHANNEL",
        ),
        "DEFAULT_SPI_PRIOR_MASS_SHARE": (
            "microcosm.build.uk_runtime.spi_support",
            "DEFAULT_SPI_PRIOR_MASS_SHARE",
        ),
        "DEFAULT_SPI_SUPPORT_HOUSEHOLDS": (
            "microcosm.build.uk_runtime.spi_support",
            "DEFAULT_SPI_SUPPORT_HOUSEHOLDS",
        ),
        "FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS": (
            "microcosm.build.uk_runtime.spi_support",
            "FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS",
        ),
        "FRS_ONLY_SPI_FILL_PERSON_COLUMNS": (
            "microcosm.build.uk_runtime.spi_support",
            "FRS_ONLY_SPI_FILL_PERSON_COLUMNS",
        ),
        "FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS": (
            "microcosm.build.uk_runtime.spi_support",
            "FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS",
        ),
        "HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN": (
            "microcosm.build.uk_runtime.spi_support",
            "HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN",
        ),
        "SPI_INCOME_COMPONENT_COLUMNS": (
            "microcosm.build.uk_runtime.spi_support",
            "SPI_INCOME_COMPONENT_COLUMNS",
        ),
        "SPI_INCOME_IMPUTATION_COLUMNS": (
            "microcosm.build.uk_runtime.spi_support",
            "SPI_INCOME_IMPUTATION_COLUMNS",
        ),
        "SPI_PRIOR_MASS_CHANGE_REASON": (
            "microcosm.build.uk_runtime.spi_support",
            "SPI_PRIOR_MASS_CHANGE_REASON",
        ),
        "SPI_REPLACEMENT_STRATA_COLUMNS": (
            "microcosm.build.uk_runtime.spi_support",
            "SPI_REPLACEMENT_STRATA_COLUMNS",
        ),
        "SPI_SYNTHETIC_SUPPORT_CHANNEL": (
            "microcosm.build.uk_runtime.spi_support",
            "SPI_SYNTHETIC_SUPPORT_CHANNEL",
        ),
        "UK_SPI_SUPPORT_STAGE_NAME": (
            "microcosm.build.uk_runtime.spi_support",
            "UK_SPI_SUPPORT_STAGE_NAME",
        ),
        "UKSPISupportResult": (
            "microcosm.build.uk_runtime.spi_support",
            "UKSPISupportResult",
        ),
        "build_uk_spi_support_channel": (
            "microcosm.build.uk_runtime.spi_support",
            "build_uk_spi_support_channel",
        ),
        "create_uk_spi_support_tables": (
            "microcosm.build.uk_runtime.spi_support",
            "create_uk_spi_support_tables",
        ),
        "fill_support_channel_from_source": (
            "microcosm.build.uk_runtime.spi_support",
            "fill_support_channel_from_source",
        ),
        "replace_uk_spi_support_tables": (
            "microcosm.build.uk_runtime.spi_support",
            "replace_uk_spi_support_tables",
        ),
        "support_channel_column": (
            "microcosm.build.uk_runtime.spi_support",
            "support_channel_column",
        ),
        "support_clone_index_column": (
            "microcosm.build.uk_runtime.spi_support",
            "support_clone_index_column",
        ),
        "support_source_id_column": (
            "microcosm.build.uk_runtime.spi_support",
            "support_source_id_column",
        ),
        "UK_FRAME_METADATA_KEY": (
            "microcosm.build.uk_runtime.stage_checkpoints",
            "UK_FRAME_METADATA_KEY",
        ),
        "load_uk_stage_checkpoint": (
            "microcosm.build.uk_runtime.stage_checkpoints",
            "load_uk_stage_checkpoint",
        ),
        "load_uk_stage_predecessor": (
            "microcosm.build.uk_runtime.stage_checkpoints",
            "load_uk_stage_predecessor",
        ),
        "uk_stage_metadata": (
            "microcosm.build.uk_runtime.stage_checkpoints",
            "uk_stage_metadata",
        ),
        "UK_DEFAULT_ZERO_WEIGHT_STRATA": (
            "microcosm.build.uk_runtime.terminal_gates",
            "UK_DEFAULT_ZERO_WEIGHT_STRATA",
        ),
        "UK_MAX_TARGET_ABS_RELATIVE_ERROR": (
            "microcosm.build.uk_runtime.terminal_gates",
            "UK_MAX_TARGET_ABS_RELATIVE_ERROR",
        ),
        "UKZeroWeightStratumDeclaration": (
            "microcosm.build.uk_runtime.terminal_gates",
            "UKZeroWeightStratumDeclaration",
        ),
        "uk_degenerate_release_surface_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_degenerate_release_surface_gate",
        ),
        "uk_export_surface_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_export_surface_gate",
        ),
        "uk_target_fit_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_target_fit_gate",
        ),
        "uk_target_surface_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_target_surface_gate",
        ),
        "uk_weight_ess_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_weight_ess_gate",
        ),
        "uk_weight_ratio_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_weight_ratio_gate",
        ),
        "uk_zero_weight_strata_gate": (
            "microcosm.build.uk_runtime.terminal_gates",
            "uk_zero_weight_strata_gate",
        ),
        "UKInputMassParityPolicy": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "UKInputMassParityPolicy",
        ),
        "UKInputMassReference": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "UKInputMassReference",
        ),
        "UKQRFTailConcentrationPolicy": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "UKQRFTailConcentrationPolicy",
        ),
        "load_uk_input_mass_reference": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "load_uk_input_mass_reference",
        ),
        "load_uk_local_area_support_exclusion_register": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "load_uk_local_area_support_exclusion_register",
        ),
        "load_uk_reviewed_exclusion_register": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "load_uk_reviewed_exclusion_register",
        ),
        "uk_input_mass_parity_gate": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "uk_input_mass_parity_gate",
        ),
        "uk_input_mass_totals": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "uk_input_mass_totals",
        ),
        "uk_qrf_tail_concentration_columns": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "uk_qrf_tail_concentration_columns",
        ),
        "uk_qrf_tail_concentration_gate": (
            "microcosm.build.uk_runtime.weighted_integrity",
            "uk_qrf_tail_concentration_gate",
        ),
    }
)

# Reload clears public aliases, retaining the real defining modules and submodules.
for _export_name in __all__:
    globals().pop(_export_name, None)
del _export_name


def __getattr__(name: str) -> _Any:
    """Resolve and cache the exact public object, preserving import failures."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Discover declared exports without importing their defining modules."""
    return sorted(set(globals()) | set(__all__))
