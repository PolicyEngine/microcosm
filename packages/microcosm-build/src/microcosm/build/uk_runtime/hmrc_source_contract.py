"""Executable anti-drift contract for the UK HMRC/SPI source family."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    FRS_HMRC_RETAINED_LEAF_COLUMNS,
    FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_COLLATED_ODS_MIME_TYPE,
    HMRC_SPI_COLLATED_ODS_SHA256,
    HMRC_SPI_COLLATED_ODS_SIZE_BYTES,
    HMRC_SPI_COLLATED_ODS_URL,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_PUBLICATION_URL,
    HMRC_SPI_SOURCE_VINTAGE,
    HMRC_SPI_TARGET_RECORD_COUNT,
    hmrc_spi_component_source_columns,
)
from microcosm.build.uk_runtime.hmrc_replay import (
    CANONICAL_HMRC_FACT_FENCES,
    FULL_FRS_TI_BAND_FENCE_ID,
)
from microcosm.build.uk_runtime.spi_income import (
    DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS,
    SPI_DONOR_DOCUMENTATION_URL,
    SPI_DONOR_DOI,
    SPI_DONOR_FILENAME,
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_DONOR_SHA256,
    SPI_DONOR_SIZE_BYTES,
    SPI_DONOR_UKDS_STUDY,
    SPI_DONOR_VINTAGE,
    SPI_POLICYENGINE_EMPLOYMENT_FORMULA,
    SPI_QRF_SOURCE_COLUMNS,
    SPI_SOURCE_COMPOSITE_INDICATOR,
    SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP,
    SPI_SOURCE_TEI_FORMULA,
    SPI_SOURCE_TI_FORMULA,
    SPI_SOURCE_TII_FORMULA,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    SPI_TI_IDENTITY_ABS_TOLERANCE_GBP,
)
from microcosm.build.uk_runtime.spi_support import (
    DEFAULT_SPI_PRIOR_MASS_SHARE,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
    SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_PRIOR_MASS_CHANGE_REASON,
    SPI_REPLACEMENT_STRATA_COLUMNS,
)

__all__ = [
    "HMRC_DISTRIBUTIONAL_INPUTS",
    "UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE",
    "assert_uk_hmrc_income_source_contract_current",
    "uk_hmrc_weighted_qrf_output_columns",
]

UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE = "hmrc_income_source_stages.json"
HMRC_DISTRIBUTIONAL_INPUTS = (
    "gift_aid",
    "charitable_investment_gifts",
)
_STAGE2_SOURCE_FAITHFUL_INCOME_PREDICTORS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
)
_STAGE2_SOURCE_FAITHFUL_PREDICTORS = (
    "age",
    "gender",
    "region",
    *_STAGE2_SOURCE_FAITHFUL_INCOME_PREDICTORS,
)
_STAGE2_REVIEWED_ABSENT_PREDICTORS = {
    "other_investment_income": (
        "This remains a stage-1 SPI draw and an official HMRC fact component, "
        "but it is not an FRS-only stage-2 predictor: policyengine-uk-data "
        "frs_only.py defines exactly six income predictors and the certified "
        "Microcosm UK base candidate has no other_investment_income column."
    )
}

_EXPECTED_OPERATION_KINDS = (
    "verify_certified_candidate",
    "retain_adjudicated_frs_hmrc_leaves",
    "verify_pinned_hmrc_source_pair",
    "replace_zero_weight_spi_support",
    "strict_read_private_table",
    "fit_weighted_qrf_stage1",
    "fit_weighted_qrf_stage2",
    "materialize_hmrc_income_bands_fail_closed",
    "classify_hmrc_income_facts_with_reviewed_fences",
    "gate_distributional_effective_mass",
)


def assert_uk_hmrc_income_source_contract_current(
    resource: Any | None = None,
) -> None:
    """Fail if source-stage declarations and the executable runtime diverge."""

    payload = _load_payload(resource)
    failures: list[str] = []
    _expect(failures, "country", payload.get("country"), "uk")
    _expect(failures, "version", payload.get("version"), 1)

    stages = _mapping_sequence(payload.get("stages"), "stages", failures)
    if len(stages) != 1:
        failures.append(f"stages: expected exactly one stage, got {len(stages)}")
        _raise_failures(failures)
    stage = stages[0]
    _expect(failures, "stage.stage", stage.get("stage"), "hmrc_spi_income")
    _expect(failures, "stage.grain", stage.get("grain"), "person")

    # Imported lazily to keep the candidate identity constants in their runtime
    # owner without introducing an import cycle at module import time.
    from microcosm.build.uk_runtime import hmrc_restoration

    base = _mapping(stage.get("base_candidate"), "base_candidate", failures)
    _expect(
        failures,
        "base_candidate.filename",
        base.get("filename"),
        hmrc_restoration.CERTIFIED_UK_CANDIDATE_FILENAME,
    )
    _expect(
        failures,
        "base_candidate.tier",
        base.get("tier"),
        hmrc_restoration.CERTIFIED_UK_CANDIDATE_TIER,
    )
    _expect(
        failures,
        "base_candidate.revision",
        base.get("revision"),
        hmrc_restoration.CERTIFIED_UK_CANDIDATE_REVISION,
    )
    _expect(
        failures,
        "base_candidate.sha256",
        base.get("sha256"),
        hmrc_restoration.CERTIFIED_UK_CANDIDATE_SHA256,
    )
    _expect(
        failures,
        "base_candidate.size_bytes",
        base.get("size_bytes"),
        hmrc_restoration.CERTIFIED_UK_CANDIDATE_SIZE_BYTES,
    )
    _expect(
        failures,
        "base_candidate.runtime_sha256_required",
        base.get("runtime_sha256_required"),
        True,
    )

    artifacts = _keyed_items(
        stage.get("artifacts"),
        key="role",
        label="artifacts",
        failures=failures,
    )
    _expect(
        failures,
        "artifact roles",
        tuple(sorted(artifacts)),
        ("published_fact_surface", "qrf_donor"),
    )
    donor = artifacts.get("qrf_donor", {})
    _expect(failures, "qrf_donor.filename", donor.get("filename"), SPI_DONOR_FILENAME)
    _expect(failures, "qrf_donor.vintage", donor.get("vintage"), SPI_DONOR_VINTAGE)
    _expect(
        failures,
        "qrf_donor.ukds_study_number",
        donor.get("ukds_study_number"),
        SPI_DONOR_UKDS_STUDY,
    )
    _expect(failures, "qrf_donor.doi", donor.get("doi"), SPI_DONOR_DOI)
    _expect(
        failures,
        "qrf_donor.sha256",
        donor.get("sha256"),
        SPI_DONOR_SHA256,
    )
    _expect(
        failures,
        "qrf_donor.size_bytes",
        donor.get("size_bytes"),
        SPI_DONOR_SIZE_BYTES,
    )
    _expect(
        failures,
        "qrf_donor.runtime_sha256_required",
        donor.get("runtime_sha256_required"),
        True,
    )
    surface = artifacts.get("published_fact_surface", {})
    _expect(
        failures,
        "published_fact_surface.locator",
        surface.get("locator"),
        HMRC_SPI_COLLATED_ODS_URL,
    )
    _expect(
        failures,
        "published_fact_surface.publication",
        surface.get("publication"),
        HMRC_SPI_PUBLICATION_URL,
    )
    _expect(
        failures,
        "published_fact_surface.vintage",
        surface.get("vintage"),
        HMRC_SPI_SOURCE_VINTAGE,
    )
    _expect(
        failures,
        "published_fact_surface.mapped_build_period",
        str(surface.get("mapped_build_period")),
        HMRC_SPI_BUILD_PERIOD,
    )
    _expect(
        failures,
        "published_fact_surface.period_mapping",
        surface.get("period_mapping"),
        "tax_year_start",
    )
    _expect(
        failures,
        "published_fact_surface.sheets",
        tuple(surface.get("sheets", ())),
        ("Table_3_6", "Table_3_7"),
    )
    _expect(
        failures,
        "published_fact_surface.runtime_sha256_required",
        surface.get("runtime_sha256_required"),
        True,
    )
    _expect(
        failures,
        "published_fact_surface.sha256",
        surface.get("sha256"),
        HMRC_SPI_COLLATED_ODS_SHA256,
    )
    _expect(
        failures,
        "published_fact_surface.size_bytes",
        surface.get("size_bytes"),
        HMRC_SPI_COLLATED_ODS_SIZE_BYTES,
    )
    _expect(
        failures,
        "published_fact_surface.mime_type",
        surface.get("mime_type"),
        HMRC_SPI_COLLATED_ODS_MIME_TYPE,
    )

    operations = _keyed_items(
        stage.get("operations"),
        key="kind",
        label="operations",
        failures=failures,
    )
    actual_operation_kinds = tuple(
        str(operation.get("kind", ""))
        for operation in _mapping_sequence(
            stage.get("operations"), "operations", failures
        )
    )
    _expect(
        failures,
        "operation order",
        actual_operation_kinds,
        _EXPECTED_OPERATION_KINDS,
    )

    verify = operations.get("verify_certified_candidate", {})
    _expect(failures, "verify.artifact", verify.get("artifact"), "base_candidate")
    _expect(failures, "verify.fail_on_mismatch", verify.get("fail_on_mismatch"), True)

    frs_leaves = operations.get("retain_adjudicated_frs_hmrc_leaves", {})
    _expect(
        failures,
        "frs_leaves.source_vintage",
        frs_leaves.get("source_vintage"),
        "2023-24",
    )
    _expect(
        failures,
        "frs_leaves.mapped_build_period",
        str(frs_leaves.get("mapped_build_period")),
        HMRC_SPI_BUILD_PERIOD,
    )
    _expect(
        failures,
        "frs_leaves.status",
        frs_leaves.get("status"),
        "adjudicated_partial_replay",
    )
    full = _mapping(
        frs_leaves.get("retained_full_constituents"),
        "frs_leaves.retained_full_constituents",
        failures,
    )
    subsets = _mapping(
        frs_leaves.get("retained_named_subsets"),
        "frs_leaves.retained_named_subsets",
        failures,
    )
    expected_full_columns = FRS_HMRC_RETAINED_LEAF_COLUMNS[:3]
    expected_subset_columns = FRS_HMRC_RETAINED_LEAF_COLUMNS[3:]
    _expect(
        failures,
        "frs_leaves.retained_full_constituents.columns",
        tuple(full),
        expected_full_columns,
    )
    _expect(
        failures,
        "frs_leaves.retained_named_subsets.columns",
        tuple(subsets),
        expected_subset_columns,
    )
    for column in FRS_HMRC_RETAINED_LEAF_COLUMNS:
        actual = dict((full | subsets).get(column, {}))
        observed_support = actual.pop("observed_support", None)
        _expect(
            failures,
            f"frs_leaves.{column}.source_evidence",
            actual,
            FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE[column],
        )
        if column == "hmrc_spi_incapacity_benefit_income":
            _expect(
                failures,
                f"frs_leaves.{column}.observed_support",
                observed_support,
                "structural zero in the audited 2023-24 FRS; retained so future vintages flow",
            )
        elif observed_support is not None:
            failures.append(
                f"frs_leaves.{column}.observed_support: unexpected declaration"
            )
    _expect(
        failures,
        "frs_leaves.source_absent_full_constituents",
        tuple(frs_leaves.get("source_absent_full_constituents", ())),
        ("EPB", "EXPS", "TAXTERM", "MOTHINC", "OTHERINC"),
    )
    _expect(
        failures,
        "frs_leaves.full_concepts_forbidden_on_frs",
        tuple(frs_leaves.get("full_concepts_forbidden_on_frs", ())),
        (
            "hmrc_spi_employment_benefits",
            "hmrc_spi_employment_expenses",
            "hmrc_spi_taxable_termination_pay",
            "hmrc_spi_miscellaneous_employment_income",
            "hmrc_spi_other_income",
            "hmrc_spi_other_social_security_income",
            "hmrc_spi_state_pension_income",
        ),
    )
    _expect(
        failures,
        "frs_leaves.forbid_proxy_substitution",
        tuple(frs_leaves.get("forbid_proxy_substitution", ())),
        ("employment_income", "miscellaneous_income"),
    )
    for flag in (
        "fail_on_missing_retained_constituent",
        "fail_on_full_concept_alias",
    ):
        _expect(failures, f"frs_leaves.{flag}", frs_leaves.get(flag), True)

    source_pair = operations.get("verify_pinned_hmrc_source_pair", {})
    _expect(
        failures,
        "source_pair.artifact_roles",
        tuple(source_pair.get("artifact_roles", ())),
        ("qrf_donor", "published_fact_surface"),
    )
    for flag in (
        "require_before_source_read",
        "runtime_sha256_required",
        "fail_on_mismatch",
    ):
        _expect(failures, f"source_pair.{flag}", source_pair.get(flag), True)

    prior = operations.get("replace_zero_weight_spi_support", {})
    _expect(failures, "prior.existing_channel", prior.get("existing_channel"), "spi")
    _expect(
        failures,
        "prior.require_existing_weight",
        prior.get("require_existing_weight"),
        0,
    )
    _expect(
        failures,
        "prior.replacement_strata",
        tuple(prior.get("replacement_strata", ())),
        SPI_REPLACEMENT_STRATA_COLUMNS,
    )
    _expect(
        failures,
        "prior.mass_share",
        prior.get("spi_prior_national_household_mass_share"),
        DEFAULT_SPI_PRIOR_MASS_SHARE,
    )
    _expect(
        failures,
        "prior.output_weight_kind",
        prior.get("output_weight_kind"),
        "importance",
    )
    _expect(
        failures,
        "prior.preserve_total_household_mass",
        prior.get("preserve_total_household_mass"),
        True,
    )
    _expect(
        failures,
        "prior.require_mass_change_record",
        prior.get("require_mass_change_record"),
        True,
    )
    _expect(
        failures,
        "prior.mass_change_reason",
        prior.get("mass_change_reason"),
        SPI_PRIOR_MASS_CHANGE_REASON,
    )
    _expect(
        failures,
        "prior.fail_on_live_existing_spi_mass",
        prior.get("fail_on_live_existing_spi_mass"),
        True,
    )

    strict = operations.get("strict_read_private_table", {})
    _expect(failures, "strict.filename", strict.get("filename"), SPI_DONOR_FILENAME)
    _expect(failures, "strict.delimiter", strict.get("delimiter"), "\t")
    _expect(failures, "strict.weight", strict.get("weight"), "FACT")
    _expect(
        failures,
        "strict.required_columns",
        tuple(sorted(strict.get("required_columns", ()))),
        tuple(sorted(SPI_DONOR_REQUIRED_COLUMNS)),
    )
    for flag in (
        "runtime_sha256_required",
        "fail_on_missing_file",
        "fail_on_missing_columns",
        "fail_on_invalid_weight",
    ):
        _expect(failures, f"strict.{flag}", strict.get(flag), True)

    stage1 = operations.get("fit_weighted_qrf_stage1", {})
    _expect(
        failures,
        "stage1.predictors",
        tuple(stage1.get("predictors", ())),
        ("age", "gender", "region"),
    )
    _expect(
        failures,
        "stage1.categorical_predictors",
        tuple(stage1.get("categorical_predictors", ())),
        ("gender", "region"),
    )
    _expect(
        failures,
        "stage1.source_sampling_weight",
        stage1.get("source_sampling_weight"),
        "FACT",
    )
    _expect(
        failures,
        "stage1.sample_size",
        stage1.get("sample_size"),
        DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    )
    _expect(
        failures,
        "stage1.sample_with_replacement",
        stage1.get("sample_with_replacement"),
        True,
    )
    _expect(
        failures,
        "stage1.post_sample_fit_weight",
        stage1.get("post_sample_fit_weight"),
        "uniform",
    )
    _expect(failures, "stage1.fit_weight_kind", stage1.get("fit_weight_kind"), "design")
    _expect(
        failures,
        "stage1.double_apply_source_weight",
        stage1.get("double_apply_source_weight"),
        False,
    )
    stage1_sources = {
        str(output): tuple(columns)
        for output, columns in _mapping(
            stage1.get("source_columns"), "stage1.source_columns", failures
        ).items()
    }
    _expect(failures, "stage1.source_columns", stage1_sources, SPI_QRF_SOURCE_COLUMNS)
    derived_policyengine = _mapping(
        stage1.get("derived_policyengine_outputs"),
        "stage1.derived_policyengine_outputs",
        failures,
    )
    employment_derivation = _mapping(
        derived_policyengine.get("employment_income"),
        "stage1.derived_policyengine_outputs.employment_income",
        failures,
    )
    _expect(
        failures,
        "stage1.derived_policyengine_outputs.employment_income.source_columns",
        tuple(employment_derivation.get("source_columns", ())),
        SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS["employment_income"],
    )
    _expect(
        failures,
        "stage1.derived_policyengine_outputs.employment_income.formula",
        employment_derivation.get("formula"),
        SPI_POLICYENGINE_EMPLOYMENT_FORMULA,
    )
    _expect(
        failures,
        "stage1.derived_policyengine_outputs.employment_income.derive_after_draw",
        employment_derivation.get("derive_after_draw"),
        True,
    )
    _expect(
        failures,
        "stage1.outputs",
        tuple(stage1.get("outputs", ())),
        SPI_INCOME_QRF_OUTPUT_COLUMNS,
    )
    _expect(
        failures,
        "stage1.ti_identity_absolute_tolerance_gbp",
        stage1.get("ti_identity_absolute_tolerance_gbp"),
        SPI_TI_IDENTITY_ABS_TOLERANCE_GBP,
    )
    _expect(
        failures,
        "stage1.source_ti_identity_fields",
        tuple(stage1.get("source_ti_identity_fields", ())),
        ("TI", "TEI", "TII"),
    )
    reconciliation = _mapping(
        stage1.get("source_leaf_reconciliation"),
        "stage1.source_leaf_reconciliation",
        failures,
    )
    _expect(
        failures,
        "stage1.source_leaf_reconciliation.documentation_url",
        reconciliation.get("documentation_url"),
        SPI_DONOR_DOCUMENTATION_URL,
    )
    _expect(
        failures,
        "stage1.source_leaf_reconciliation.composite_indicator",
        reconciliation.get("composite_indicator"),
        SPI_SOURCE_COMPOSITE_INDICATOR,
    )
    _expect(
        failures,
        "stage1.source_leaf_reconciliation.formulas",
        dict(reconciliation.get("formulas", {})),
        {
            "TEI": SPI_SOURCE_TEI_FORMULA,
            "TII": SPI_SOURCE_TII_FORMULA,
            "TI": SPI_SOURCE_TI_FORMULA,
        },
    )
    _expect(
        failures,
        "stage1.source_leaf_reconciliation.maximum_absolute_difference_gbp",
        dict(reconciliation.get("maximum_absolute_difference_gbp", {})),
        SPI_SOURCE_LEAF_RECONCILIATION_ABS_TOLERANCE_GBP,
    )
    _expect(
        failures,
        "stage1.stochastic_aggregates_forbidden",
        tuple(stage1.get("stochastic_aggregates_forbidden", ())),
        SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
    )
    for flag in ("joint_draw", "require_all_predictors", "require_all_outputs"):
        _expect(failures, f"stage1.{flag}", stage1.get(flag), True)

    stage2 = operations.get("fit_weighted_qrf_stage2", {})
    _expect(
        failures,
        "stage2.predictors",
        tuple(stage2.get("predictors", ())),
        _STAGE2_SOURCE_FAITHFUL_PREDICTORS,
    )
    _expect(
        failures,
        "runtime.stage2.predictors",
        FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
        _STAGE2_SOURCE_FAITHFUL_PREDICTORS,
    )
    _expect(
        failures,
        "stage2.reviewed_absent_predictors",
        dict(stage2.get("reviewed_absent_predictors", {})),
        _STAGE2_REVIEWED_ABSENT_PREDICTORS,
    )
    expected_stage2_outputs = tuple(
        column
        for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
        if column not in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS
    )
    _expect(
        failures,
        "stage2.outputs",
        tuple(stage2.get("outputs", ())),
        expected_stage2_outputs,
    )
    _expect(
        failures,
        "stage2.reviewed_absent_outputs",
        dict(stage2.get("reviewed_absent_outputs", {})),
        SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    )
    _expect(failures, "stage2.weight", stage2.get("weight"), "household_weight")
    _expect(
        failures,
        "stage2.weight_mapping",
        stage2.get("weight_mapping"),
        "household_to_person",
    )
    _expect(failures, "stage2.joint_draw", stage2.get("joint_draw"), True)
    _expect(
        failures,
        "stage2.require_all_predictors",
        stage2.get("require_all_predictors"),
        True,
    )
    _expect(
        failures,
        "stage2.require_all_materializable_outputs",
        stage2.get("require_all_materializable_outputs"),
        True,
    )
    _expect(
        failures, "stage2.require_all_outputs", stage2.get("require_all_outputs"), False
    )

    materialize = operations.get("materialize_hmrc_income_bands_fail_closed", {})
    _expect(
        failures,
        "materialize.artifact_role",
        materialize.get("artifact_role"),
        "published_fact_surface",
    )
    actual_layout = {
        str(component): (
            str(spec.get("sheet")),
            spec.get("count_column_index"),
            spec.get("amount_column_index"),
        )
        for component, spec in _mapping(
            materialize.get("component_columns"),
            "materialize.component_columns",
            failures,
        ).items()
        if isinstance(spec, Mapping)
    }
    _expect(
        failures,
        "materialize.component_columns",
        actual_layout,
        hmrc_spi_component_source_columns(),
    )
    _expect(
        failures,
        "materialize.mapped_build_period",
        str(materialize.get("mapped_build_period")),
        HMRC_SPI_BUILD_PERIOD,
    )
    _expect(
        failures,
        "materialize.period_mapping",
        materialize.get("period_mapping"),
        "tax_year_start",
    )
    _expect(
        failures,
        "materialize.data_row_start_index",
        materialize.get("data_row_start_index"),
        5,
    )
    _expect(
        failures, "materialize.stop_label", materialize.get("stop_label"), "All ranges"
    )
    _expect(
        failures,
        "materialize.count_unit_multiplier",
        materialize.get("count_unit_multiplier"),
        1_000,
    )
    _expect(
        failures,
        "materialize.amount_unit_multiplier",
        materialize.get("amount_unit_multiplier"),
        1_000_000,
    )
    _expect(
        failures,
        "materialize.required_band_lower_bounds_gbp",
        tuple(materialize.get("required_band_lower_bounds_gbp", ())),
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    )
    _expect(
        failures,
        "materialize.required_measures",
        tuple(materialize.get("required_measures", ())),
        ("count", "amount"),
    )
    for flag in (
        "fail_on_missing_sheet",
        "fail_on_missing_component",
        "fail_on_missing_band",
        "fail_on_non_numeric_value",
    ):
        _expect(failures, f"materialize.{flag}", materialize.get(flag), True)

    classification = operations.get(
        "classify_hmrc_income_facts_with_reviewed_fences", {}
    )
    _expect(
        failures,
        "classification.components",
        tuple(classification.get("components", ())),
        HMRC_SPI_INCOME_COMPONENTS,
    )
    _expect(
        failures,
        "classification.breakdown_dependency",
        classification.get("breakdown_dependency"),
        HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    )
    _expect(
        failures,
        "classification.frs_breakdown_status",
        classification.get("frs_breakdown_status"),
        "unavailable_full_measure",
    )
    _expect(
        failures,
        "classification.input_weight_kind",
        classification.get("input_weight_kind"),
        "importance",
    )
    _expect(
        failures,
        "classification.output_weight_kind",
        classification.get("output_weight_kind"),
        "importance",
    )
    _expect(
        failures,
        "classification.calibration_permitted",
        classification.get("calibration_permitted"),
        False,
    )
    _expect(
        failures,
        "classification.required_fact_count",
        classification.get("required_fact_count"),
        HMRC_SPI_TARGET_RECORD_COUNT,
    )
    _expect(
        failures,
        "classification.outcome_counts",
        dict(classification.get("outcome_counts", {})),
        {
            "exact_pass": 0,
            "exact_fail": 0,
            "directional_pass": 0,
            "directional_fail": 0,
            "excluded_with_fence": HMRC_SPI_TARGET_RECORD_COUNT,
        },
    )
    _expect(
        failures,
        "classification.fact_fence_id",
        classification.get("fact_fence_id"),
        FULL_FRS_TI_BAND_FENCE_ID,
    )
    _expect(
        failures,
        "classification.blocked_dependency",
        classification.get("blocked_dependency"),
        HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    )
    actual_fences = _mapping_sequence(
        classification.get("reviewed_fences"),
        "classification.reviewed_fences",
        failures,
    )
    _expect(
        failures,
        "classification.reviewed_fences",
        tuple(dict(fence) for fence in actual_fences),
        tuple(
            {"fence_id": fence.fence_id, **fence.to_payload()}
            for fence in CANONICAL_HMRC_FACT_FENCES
        ),
    )
    for flag in (
        "fail_on_unfenced_exclusion",
        "fail_on_fact_count_mismatch",
        "forbid_biased_estimate_or_delta",
    ):
        _expect(
            failures,
            f"classification.{flag}",
            classification.get(flag),
            True,
        )

    effective = operations.get("gate_distributional_effective_mass", {})
    _expect(
        failures,
        "effective.columns",
        tuple(effective.get("columns", ())),
        HMRC_DISTRIBUTIONAL_INPUTS,
    )
    _expect(failures, "effective.weight", effective.get("weight"), "household_weight")
    _expect(
        failures,
        "effective.weight_mapping",
        effective.get("weight_mapping"),
        "household_to_person",
    )
    _expect(
        failures,
        "effective.support_channel_column",
        effective.get("support_channel_column"),
        "person_support_channel",
    )
    _expect(
        failures,
        "effective.required_support_channel",
        effective.get("required_support_channel"),
        "spi",
    )
    _expect(
        failures,
        "effective.mass_share_denominator",
        effective.get("mass_share_denominator"),
        "all_person_effective_mass",
    )
    _expect(
        failures,
        "effective.minimum_nondefault_mass_share",
        effective.get("minimum_nondefault_mass_share"),
        1e-6,
    )
    _expect(
        failures, "effective.fail_below_floor", effective.get("fail_below_floor"), True
    )

    _expect(
        failures,
        "stage.official_table_components",
        tuple(stage.get("official_table_components", ())),
        HMRC_SPI_INCOME_COMPONENTS,
    )
    _expect(
        failures,
        "stage.donor_relief_outputs",
        tuple(stage.get("donor_relief_outputs", ())),
        HMRC_DISTRIBUTIONAL_INPUTS,
    )
    _expect(
        failures,
        "stage.outputs",
        tuple(stage.get("outputs", ())),
        (
            *HMRC_SPI_INCOME_COMPONENTS,
            *HMRC_DISTRIBUTIONAL_INPUTS,
            *SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
        ),
    )

    _raise_failures(failures)


def uk_hmrc_weighted_qrf_output_columns(
    resource: Any | None = None,
) -> tuple[str, ...]:
    """Columns produced by declared ``fit_weighted_qrf_stage*`` operations.

    This is the UK tail-concentration gate's column surface (#609). It is
    derived from the declarative source manifest rather than a hand list —
    mirroring the US derivation from ``us/source_stages.json`` — so a new
    weighted-QRF output is covered by the gate the day the manifest declares
    it. Declaration order is preserved (stage 1 before stage 2).
    """

    payload = _load_payload(resource)
    stages = payload.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise ValueError("UK HMRC source manifest must declare a stages list.")
    outputs: dict[str, None] = {}
    weighted_qrf_operations = 0
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("UK HMRC source manifest stages must be objects.")
        operations = stage.get("operations", ())
        if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
            raise ValueError("UK HMRC source manifest stage operations must be a list.")
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise ValueError("UK HMRC source manifest operations must be objects.")
            kind = operation.get("kind")
            if not isinstance(kind, str) or not kind.startswith("fit_weighted_qrf"):
                continue
            weighted_qrf_operations += 1
            declared = operation.get("outputs")
            if (
                not isinstance(declared, Sequence)
                or isinstance(declared, (str, bytes))
                or not declared
            ):
                raise ValueError(
                    f"UK HMRC source manifest operation {kind!r} must declare a "
                    "non-empty outputs list."
                )
            for output in declared:
                if not isinstance(output, str) or not output:
                    raise ValueError(
                        f"UK HMRC source manifest operation {kind!r} declares a "
                        "non-string or empty output."
                    )
                outputs[output] = None
    if not weighted_qrf_operations:
        raise ValueError(
            "UK HMRC source manifest declares no fit_weighted_qrf operations; "
            "an empty tail-concentration surface would make the gate vacuous."
        )
    return tuple(outputs)


def _load_payload(resource: Any | None) -> Mapping[str, Any]:
    target = (
        files("microcosm.build.uk").joinpath(UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE)
        if resource is None
        else resource
    )
    if hasattr(target, "read_text"):
        raw = target.read_text(encoding="utf-8")
    else:
        raw = Path(target).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("UK HMRC source manifest root must be a JSON object.")
    return payload


def _mapping(value: object, label: str, failures: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        failures.append(f"{label}: expected an object")
        return {}
    return value


def _mapping_sequence(
    value: object,
    label: str,
    failures: list[str],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        failures.append(f"{label}: expected a list of objects")
        return ()
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            failures.append(f"{label}[{index}]: expected an object")
            continue
        result.append(item)
    return tuple(result)


def _keyed_items(
    value: object,
    *,
    key: str,
    label: str,
    failures: list[str],
) -> dict[str, Mapping[str, Any]]:
    items = _mapping_sequence(value, label, failures)
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items):
        raw_name = item.get(key)
        if not isinstance(raw_name, str) or not raw_name:
            failures.append(f"{label}[{index}].{key}: expected a non-empty string")
            continue
        if raw_name in result:
            failures.append(f"{label}: duplicate {key} {raw_name!r}")
            continue
        result[raw_name] = item
    return result


def _expect(
    failures: list[str],
    label: str,
    actual: object,
    expected: object,
) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


def _raise_failures(failures: list[str]) -> None:
    if failures:
        raise ValueError(
            "UK HMRC/SPI source manifest has drifted from the executable "
            "runtime:\n" + "\n".join(f"  - {failure}" for failure in failures)
        )
