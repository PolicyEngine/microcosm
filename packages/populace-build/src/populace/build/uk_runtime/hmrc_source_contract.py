"""Executable anti-drift contract for the UK HMRC/SPI source family."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.uk_runtime.hmrc_calibration import (
    DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR,
    DEFAULT_HMRC_MAX_WEIGHT_RATIO,
    HMRC_ASSESSABLE_INCOME_COLUMN,
    HMRC_TAXPAYER_COLUMN,
)
from populace.build.uk_runtime.hmrc_income import (
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
from populace.build.uk_runtime.spi_income import (
    DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    SPI_DERIVED_POLICYENGINE_SOURCE_COLUMNS,
    SPI_DONOR_DOI,
    SPI_DONOR_FILENAME,
    SPI_DONOR_REQUIRED_COLUMNS,
    SPI_DONOR_SHA256,
    SPI_DONOR_SIZE_BYTES,
    SPI_DONOR_UKDS_STUDY,
    SPI_DONOR_VINTAGE,
    SPI_HMRC_EMPLOYED_INCOME_FORMULA,
    SPI_QRF_SOURCE_COLUMNS,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    SPI_TI_IDENTITY_ABS_TOLERANCE_GBP,
)
from populace.build.uk_runtime.spi_support import (
    DEFAULT_SPI_PRIOR_MASS_SHARE,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
    SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
    SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_PRIOR_MASS_CHANGE_REASON,
    SPI_REPLACEMENT_STRATA_COLUMNS,
)

__all__ = [
    "HMRC_DISTRIBUTIONAL_INPUTS",
    "UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE",
    "assert_uk_hmrc_income_source_contract_current",
]

UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE = "hmrc_income_source_stages.json"
HMRC_DISTRIBUTIONAL_INPUTS = (
    "gift_aid",
    "charitable_investment_gifts",
)

_EXPECTED_OPERATION_KINDS = (
    "verify_certified_candidate",
    "require_frs_hmrc_employment_crosswalk",
    "replace_zero_weight_spi_support",
    "strict_read_private_table",
    "fit_weighted_qrf_stage1",
    "fit_weighted_qrf_stage2",
    "materialize_hmrc_income_bands_fail_closed",
    "derive_taxpayer_mask",
    "calibrate_weighted_income_bands",
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
    from populace.build.uk_runtime import hmrc_restoration

    base = _mapping(stage.get("base_candidate"), "base_candidate", failures)
    _expect(
        failures,
        "base_candidate.filename",
        base.get("filename"),
        hmrc_restoration.CERTIFIED_UK_CANDIDATE_FILENAME,
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
        ("calibration_surface", "qrf_donor"),
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
    surface = artifacts.get("calibration_surface", {})
    _expect(
        failures,
        "calibration_surface.locator",
        surface.get("locator"),
        HMRC_SPI_COLLATED_ODS_URL,
    )
    _expect(
        failures,
        "calibration_surface.publication",
        surface.get("publication"),
        HMRC_SPI_PUBLICATION_URL,
    )
    _expect(
        failures,
        "calibration_surface.vintage",
        surface.get("vintage"),
        HMRC_SPI_SOURCE_VINTAGE,
    )
    _expect(
        failures,
        "calibration_surface.mapped_build_period",
        str(surface.get("mapped_build_period")),
        HMRC_SPI_BUILD_PERIOD,
    )
    _expect(
        failures,
        "calibration_surface.period_mapping",
        surface.get("period_mapping"),
        "tax_year_start",
    )
    _expect(
        failures,
        "calibration_surface.sheets",
        tuple(surface.get("sheets", ())),
        ("Table_3_6", "Table_3_7"),
    )
    _expect(
        failures,
        "calibration_surface.runtime_sha256_required",
        surface.get("runtime_sha256_required"),
        True,
    )
    _expect(
        failures,
        "calibration_surface.sha256",
        surface.get("sha256"),
        HMRC_SPI_COLLATED_ODS_SHA256,
    )
    _expect(
        failures,
        "calibration_surface.size_bytes",
        surface.get("size_bytes"),
        HMRC_SPI_COLLATED_ODS_SIZE_BYTES,
    )
    _expect(
        failures,
        "calibration_surface.mime_type",
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

    frs_crosswalk = operations.get("require_frs_hmrc_employment_crosswalk", {})
    normalized_constituents = (
        *SPI_HMRC_EMPLOYED_INCOME_LEAF_COLUMNS,
        SPI_HMRC_OTHER_INCOME_COLUMN,
        SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    )
    _expect(
        failures,
        "frs_crosswalk.normalized_constituents",
        tuple(frs_crosswalk.get("normalized_constituents", ())),
        normalized_constituents,
    )
    _expect(
        failures,
        "frs_crosswalk.status",
        frs_crosswalk.get("status"),
        "blocked_pending_reviewed_frs_decomposition",
    )
    _expect(
        failures,
        "frs_crosswalk.employed_income_formula",
        frs_crosswalk.get("employed_income_formula"),
        SPI_HMRC_EMPLOYED_INCOME_FORMULA,
    )
    _expect(
        failures,
        "frs_crosswalk.current_candidate_missing_all_normalized_constituents",
        frs_crosswalk.get("current_candidate_missing_all_normalized_constituents"),
        True,
    )
    _expect(
        failures,
        "frs_crosswalk.forbid_proxy_substitution",
        tuple(frs_crosswalk.get("forbid_proxy_substitution", ())),
        ("employment_income", "miscellaneous_income"),
    )
    _expect(
        failures,
        "frs_crosswalk.fail_on_missing_constituent",
        frs_crosswalk.get("fail_on_missing_constituent"),
        True,
    )

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
        FRS_ONLY_SPI_FILL_PREDICTOR_COLUMNS,
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

    taxpayer = operations.get("derive_taxpayer_mask", {})
    _expect(failures, "taxpayer.output", taxpayer.get("output"), HMRC_TAXPAYER_COLUMN)
    _expect(failures, "taxpayer.variable", taxpayer.get("variable"), "income_tax")
    _expect(
        failures,
        "taxpayer.comparison",
        taxpayer.get("comparison"),
        "strictly_greater_than",
    )
    _expect(failures, "taxpayer.threshold", taxpayer.get("threshold"), 0)
    _expect(
        failures,
        "taxpayer.mapped_build_period",
        str(taxpayer.get("mapped_build_period")),
        HMRC_SPI_BUILD_PERIOD,
    )
    _expect(
        failures,
        "taxpayer.fail_on_missing_variable",
        taxpayer.get("fail_on_missing_variable"),
        True,
    )

    calibration = operations.get("calibrate_weighted_income_bands", {})
    _expect(
        failures,
        "calibration.components",
        tuple(calibration.get("components", ())),
        HMRC_SPI_INCOME_COMPONENTS,
    )
    _expect(
        failures,
        "calibration.breakdown_variable",
        calibration.get("breakdown_variable"),
        HMRC_ASSESSABLE_INCOME_COLUMN,
    )
    _expect(
        failures,
        "calibration.employment_measure",
        calibration.get("employment_measure"),
        SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    )
    _expect(
        failures,
        "calibration.require_identical_crosswalk_all_channels",
        calibration.get("require_identical_crosswalk_all_channels"),
        True,
    )
    _expect(
        failures,
        "calibration.savings_interest_measure",
        calibration.get("savings_interest_measure"),
        "savings_interest_income - tax_free_savings_income; fail if tax-free "
        "exceeds gross",
    )
    _expect(
        failures,
        "calibration.taxpayer_mask",
        calibration.get("taxpayer_mask"),
        HMRC_TAXPAYER_COLUMN,
    )
    _expect(
        failures,
        "calibration.input_weight_kind",
        calibration.get("input_weight_kind"),
        "importance",
    )
    _expect(
        failures,
        "calibration.output_weight_kind",
        calibration.get("output_weight_kind"),
        "calibrated",
    )
    _expect(
        failures,
        "calibration.max_weight_ratio",
        calibration.get("max_weight_ratio"),
        DEFAULT_HMRC_MAX_WEIGHT_RATIO,
    )
    _expect(
        failures,
        "calibration.maximum_abs_relative_error",
        calibration.get("maximum_abs_relative_error"),
        DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR,
    )
    _expect(
        failures,
        "calibration.required_target_count",
        calibration.get("required_target_count"),
        HMRC_SPI_TARGET_RECORD_COUNT,
    )
    for flag in (
        "require_strictly_positive_prior",
        "preserve_total_household_mass",
        "fail_on_unmaterialized_target",
        "fail_on_zero_support",
    ):
        _expect(failures, f"calibration.{flag}", calibration.get(flag), True)

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


def _load_payload(resource: Any | None) -> Mapping[str, Any]:
    target = (
        files("populace.build.uk").joinpath(UK_HMRC_INCOME_SOURCE_STAGES_RESOURCE)
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
