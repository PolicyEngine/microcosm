"""Contract tests for the raw UK HMRC/SPI income source manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from populace.build.uk_runtime.hmrc_source_contract import (
    assert_uk_hmrc_income_source_contract_current,
)

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "populace"
    / "build"
    / "uk"
    / "hmrc_income_source_stages.json"
)
_COLLATED_ODS_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "69f1f12d2fae53a03709682f/Collated_Tables_3_1_to_3_11_2324.ods"
)
_COLLATED_ODS_SHA256 = (
    "ad063b06b2bdeef8600dbbb09d48153337a4966f8c7eea50df7a2e0304ebd73e"
)
_COLLATED_ODS_SIZE_BYTES = 166_693
_SPI_DONOR_SHA256 = "5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66"
_SPI_DONOR_SIZE_BYTES = 141_323_762
_OFFICIAL_COMPONENTS = [
    "employment_income",
    "self_employment_income",
    "state_pension",
    "private_pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "other_investment_income",
]
_STAGE1_OUTPUTS = [
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "private_pension_income",
    "property_income",
    "other_investment_income",
    "gift_aid",
    "charitable_investment_gifts",
    "hmrc_spi_pay",
    "hmrc_spi_employment_benefits",
    "hmrc_spi_employment_expenses",
    "hmrc_spi_incapacity_benefit_income",
    "hmrc_spi_other_social_security_income",
    "hmrc_spi_taxable_termination_pay",
    "hmrc_spi_unemployment_benefit_income",
    "hmrc_spi_miscellaneous_employment_income",
    "hmrc_spi_other_income",
    "hmrc_spi_state_pension_income",
]
_FRS_HMRC_NORMALIZED_CONSTITUENTS = [
    "hmrc_spi_pay",
    "hmrc_spi_employment_benefits",
    "hmrc_spi_employment_expenses",
    "hmrc_spi_incapacity_benefit_income",
    "hmrc_spi_other_social_security_income",
    "hmrc_spi_taxable_termination_pay",
    "hmrc_spi_unemployment_benefit_income",
    "hmrc_spi_miscellaneous_employment_income",
    "hmrc_spi_other_income",
    "hmrc_spi_state_pension_income",
]
_DERIVED_AUXILIARIES = [
    "hmrc_spi_employed_income",
    "hmrc_spi_total_earned_income",
    "hmrc_spi_total_investment_income",
    "hmrc_spi_assessable_income",
]
_COMPONENT_COLUMNS = {
    "employment_income": ("Table_3_6", 4, 5),
    "self_employment_income": ("Table_3_6", 1, 2),
    "state_pension": ("Table_3_6", 7, 8),
    "private_pension_income": ("Table_3_6", 10, 11),
    "property_income": ("Table_3_7", 1, 2),
    "savings_interest_income": ("Table_3_7", 4, 5),
    "dividend_income": ("Table_3_7", 7, 8),
    "other_investment_income": ("Table_3_7", 10, 11),
}


def _manifest() -> dict[str, object]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _stage(payload: dict[str, object]) -> dict[str, object]:
    stages = payload["stages"]
    assert isinstance(stages, list)
    assert len(stages) == 1
    stage = stages[0]
    assert isinstance(stage, dict)
    return stage


def _by_role(stage: dict[str, object]) -> dict[str, dict[str, object]]:
    artifacts = stage["artifacts"]
    assert isinstance(artifacts, list)
    assert all(isinstance(artifact, dict) for artifact in artifacts)
    return {artifact["role"]: artifact for artifact in artifacts}


def _by_kind(stage: dict[str, object]) -> dict[str, dict[str, object]]:
    operations = stage["operations"]
    assert isinstance(operations, list)
    assert all(isinstance(operation, dict) for operation in operations)
    return {operation["kind"]: operation for operation in operations}


def test__given_uk_hmrc_manifest__then_one_scoped_stage_is_declared() -> None:
    payload = _manifest()
    stage = _stage(payload)

    assert payload["version"] == 1
    assert payload["country"] == "uk"
    assert stage["stage"] == "hmrc_spi_income"
    assert stage["grain"] == "person"
    assert stage["official_table_components"] == _OFFICIAL_COMPONENTS
    assert stage["donor_relief_outputs"] == [
        "gift_aid",
        "charitable_investment_gifts",
    ]
    assert stage["outputs"] == [
        *_OFFICIAL_COMPONENTS,
        "gift_aid",
        "charitable_investment_gifts",
        *_DERIVED_AUXILIARIES,
    ]
    assert [operation["kind"] for operation in stage["operations"]] == [
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
    ]


def test__given_hmrc_source_artifacts__then_vintages_and_runtime_hashes_are_exact() -> (
    None
):
    stage = _stage(_manifest())
    artifacts = _by_role(stage)

    assert len(stage["artifacts"]) == 2
    assert set(artifacts) == {"qrf_donor", "calibration_surface"}
    donor = artifacts["qrf_donor"]
    assert donor["survey"] == "Survey of Personal Incomes Public Use Tape 2022-23"
    assert donor["vintage"] == "2022-23"
    assert donor["ukds_study_number"] == "SN 9422"
    assert donor["doi"] == "10.5255/UKDA-SN-9422-1"
    assert donor["filename"] == "put2223uk.tab"
    assert donor["sha256"] == _SPI_DONOR_SHA256
    assert donor["size_bytes"] == _SPI_DONOR_SIZE_BYTES
    assert donor["access"] == "private_local_input"
    assert donor["locator"] == "caller-supplied local input"
    assert donor["runtime_sha256_required"] is True

    surface = artifacts["calibration_surface"]
    assert surface["vintage"] == "2023-24"
    assert surface["locator"] == _COLLATED_ODS_URL
    assert surface["sha256"] == _COLLATED_ODS_SHA256
    assert surface["size_bytes"] == _COLLATED_ODS_SIZE_BYTES
    assert surface["mime_type"] == ("application/vnd.oasis.opendocument.spreadsheet")
    assert surface["sheets"] == ["Table_3_6", "Table_3_7"]
    assert surface["mapped_build_period"] == 2023
    assert surface["period_mapping"] == "tax_year_start"
    assert surface["runtime_sha256_required"] is True
    assert "tax-year-2023-to-2024" in surface["publication"]

    base = stage["base_candidate"]
    assert base["sha256"] == (
        "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
    )
    assert base["size_bytes"] == 1_315_880_118
    assert base["runtime_sha256_required"] is True


def test__given_frs_channel__then_hmrc_crosswalk_fails_closed_without_leaves() -> None:
    operations = _by_kind(_stage(_manifest()))
    crosswalk = operations["require_frs_hmrc_employment_crosswalk"]

    assert crosswalk["normalized_constituents"] == _FRS_HMRC_NORMALIZED_CONSTITUENTS
    assert crosswalk["status"] == "blocked_pending_reviewed_frs_decomposition"
    assert crosswalk["current_candidate_missing_all_normalized_constituents"] is True
    assert crosswalk["forbid_proxy_substitution"] == [
        "employment_income",
        "miscellaneous_income",
    ]
    assert crosswalk["fail_on_missing_constituent"] is True
    assert crosswalk["employed_income_formula"] == (
        "max(0, hmrc_spi_pay + hmrc_spi_employment_benefits - "
        "hmrc_spi_employment_expenses) + hmrc_spi_incapacity_benefit_income + "
        "hmrc_spi_other_social_security_income + "
        "hmrc_spi_taxable_termination_pay + "
        "hmrc_spi_unemployment_benefit_income + "
        "hmrc_spi_miscellaneous_employment_income"
    )


def test__given_spi_donor__then_both_qrf_stages_are_weighted_and_strict() -> None:
    operations = _by_kind(_stage(_manifest()))

    strict_read = operations["strict_read_private_table"]
    assert strict_read["artifact_role"] == "qrf_donor"
    assert strict_read["filename"] == "put2223uk.tab"
    assert strict_read["weight"] == "FACT"
    assert strict_read["runtime_sha256_required"] is True
    assert strict_read["fail_on_missing_file"] is True
    assert strict_read["fail_on_missing_columns"] is True
    assert strict_read["fail_on_invalid_weight"] is True
    assert {
        "EXPS",
        "INCPBEN",
        "OSSBEN",
        "UBISJA",
        "MOTHINC",
        "OTHERINC",
        "CAPALL",
        "LOSSBF",
        "SRP",
        "TI",
    } <= set(strict_read["required_columns"])

    stage1 = operations["fit_weighted_qrf_stage1"]
    assert stage1["source_sampling_weight"] == "FACT"
    assert stage1["sample_size"] == 100_000
    assert stage1["sample_with_replacement"] is True
    assert stage1["post_sample_fit_weight"] == "uniform"
    assert stage1["fit_weight_kind"] == "design"
    assert stage1["double_apply_source_weight"] is False
    assert stage1["outputs"] == _STAGE1_OUTPUTS
    employment_derivation = stage1["derived_policyengine_outputs"]["employment_income"]
    assert employment_derivation["source_columns"] == [
        "PAY",
        "EPB",
        "TAXTERM",
    ]
    assert employment_derivation["formula"] == (
        "hmrc_spi_pay + hmrc_spi_employment_benefits + hmrc_spi_taxable_termination_pay"
    )
    assert employment_derivation["derive_after_draw"] is True
    assert "employment_income" not in stage1["source_columns"]
    assert "employment_income" not in stage1["outputs"]
    assert stage1["source_columns"]["other_investment_income"] == ["OTHERINV"]
    assert stage1["source_columns"]["hmrc_spi_other_income"] == ["OTHERINC"]
    assert stage1["source_columns"]["hmrc_spi_state_pension_income"] == ["SRP"]
    assert stage1["ti_identity_absolute_tolerance_gbp"] == 5
    assert stage1["source_ti_identity_fields"] == ["TI", "TEI", "TII"]
    assert stage1["stochastic_aggregates_forbidden"] == _DERIVED_AUXILIARIES
    assert all(
        column not in stage1["source_columns"] for column in _DERIVED_AUXILIARIES
    )
    assert all(column not in stage1["outputs"] for column in _DERIVED_AUXILIARIES)
    assert "TEI + TII" in stage1["assessable_income_source_semantics"]
    assert "deterministic post-draw" in stage1["assessable_income_source_semantics"]
    assert stage1["source_columns"]["gift_aid"] == ["GIFTAID"]
    assert stage1["source_columns"]["charitable_investment_gifts"] == ["GIFTINV"]
    assert "state_pension" not in stage1["outputs"]
    assert stage1["joint_draw"] is True
    assert stage1["require_all_predictors"] is True
    assert stage1["require_all_outputs"] is True

    stage2 = operations["fit_weighted_qrf_stage2"]
    assert stage2["weight"] == "household_weight"
    assert stage2["weight_mapping"] == "household_to_person"
    assert "other_investment_income" in stage2["predictors"]
    assert "state_pension_reported" in stage2["outputs"]
    assert "universal_credit_reported" in stage2["outputs"]
    assert "employee_pension_contributions" in stage2["outputs"]
    assert "incapacity_benefit_reported" not in stage2["outputs"]
    assert "maternity_allowance_reported" not in stage2["outputs"]
    assert set(stage2["reviewed_absent_outputs"]) == {
        "incapacity_benefit_reported",
        "maternity_allowance_reported",
    }
    assert stage2["require_all_predictors"] is True
    assert stage2["require_all_materializable_outputs"] is True
    assert stage2["require_all_outputs"] is False
    assert stage2["postprocess"]["gross_savings_interest_income"] == (
        "stage1 INCBBS draw + stage2 tax_free_savings_income"
    )
    assert "pip_dl_category" in stage2["postprocess"]["refresh_disability_categories"]
    assert (
        "is_disabled_for_benefits" in stage2["postprocess"]["refresh_disability_flags"]
    )


def test__given_official_tables__then_all_eight_components_fail_closed() -> None:
    operations = _by_kind(_stage(_manifest()))
    materializer = operations["materialize_hmrc_income_bands_fail_closed"]

    actual_columns = {
        component: (
            spec["sheet"],
            spec["count_column_index"],
            spec["amount_column_index"],
        )
        for component, spec in materializer["component_columns"].items()
    }
    assert actual_columns == _COMPONENT_COLUMNS
    assert materializer["mapped_build_period"] == 2023
    assert materializer["period_mapping"] == "tax_year_start"
    assert materializer["required_measures"] == ["count", "amount"]
    assert materializer["required_band_lower_bounds_gbp"] == [
        12_570,
        15_000,
        20_000,
        30_000,
        40_000,
        50_000,
        70_000,
        100_000,
        150_000,
        200_000,
        300_000,
        500_000,
        1_000_000,
    ]
    assert materializer["fail_on_missing_sheet"] is True
    assert materializer["fail_on_missing_component"] is True
    assert materializer["fail_on_missing_band"] is True
    assert materializer["fail_on_non_numeric_value"] is True


def test__given_materialized_hmrc_targets__then_taxpayer_calibration_is_guarded() -> (
    None
):
    operations = _by_kind(_stage(_manifest()))

    mask = operations["derive_taxpayer_mask"]
    assert mask == {
        "kind": "derive_taxpayer_mask",
        "output": "hmrc_spi_taxpayer",
        "entity": "person",
        "variable": "income_tax",
        "comparison": "strictly_greater_than",
        "threshold": 0,
        "mapped_build_period": 2023,
        "fail_on_missing_variable": True,
    }

    calibration = operations["calibrate_weighted_income_bands"]
    assert calibration["components"] == _OFFICIAL_COMPONENTS
    assert calibration["taxpayer_mask"] == "hmrc_spi_taxpayer"
    assert calibration["count_condition"] == "component > 0"
    assert calibration["measures"] == ["count", "amount"]
    assert calibration["weight"] == "household_weight"
    assert calibration["weight_mapping"] == "household_to_person"
    assert calibration["input_weight_kind"] == "importance"
    assert calibration["output_weight_kind"] == "calibrated"
    assert calibration["breakdown_variable"] == "hmrc_spi_assessable_income"
    assert calibration["required_target_count"] == 208
    assert calibration["max_weight_ratio"] == 5
    assert calibration["maximum_abs_relative_error"] == 0.05
    assert calibration["require_strictly_positive_prior"] is True
    assert calibration["preserve_total_household_mass"] is True
    assert calibration["fail_on_unmaterialized_target"] is True
    assert calibration["fail_on_zero_support"] is True

    prior = operations["replace_zero_weight_spi_support"]
    assert prior["require_existing_weight"] == 0
    assert prior["spi_prior_national_household_mass_share"] == 0.5
    assert prior["output_weight_kind"] == "importance"
    assert prior["preserve_total_household_mass"] is True
    assert prior["require_mass_change_record"] is True

    effective = operations["gate_distributional_effective_mass"]
    assert effective["columns"] == [
        "gift_aid",
        "charitable_investment_gifts",
    ]
    assert effective["minimum_nondefault_mass_share"] == 0.000001
    assert effective["fail_below_floor"] is True


def test__given_standalone_contract__then_sources_are_explicit_artifacts() -> None:
    artifacts = _by_role(_stage(_manifest()))

    assert artifacts["qrf_donor"]["access"] == "private_local_input"
    assert artifacts["qrf_donor"]["locator"] == "caller-supplied local input"
    assert artifacts["calibration_surface"]["locator"] == _COLLATED_ODS_URL
    assert all(artifact["runtime_sha256_required"] for artifact in artifacts.values())


def test_runtime_source_contract_matches_committed_manifest() -> None:
    assert_uk_hmrc_income_source_contract_current()


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    [
        (
            ("stages", 0, "base_candidate", "sha256"),
            "0" * 64,
            "base_candidate.sha256",
        ),
        (
            ("stages", 0, "artifacts", 1, "vintage"),
            "2022-23",
            "calibration_surface.vintage",
        ),
        (
            ("stages", 0, "artifacts", 0, "sha256"),
            "0" * 64,
            "qrf_donor.sha256",
        ),
        (
            ("stages", 0, "artifacts", 1, "size_bytes"),
            1,
            "calibration_surface.size_bytes",
        ),
        (
            ("stages", 0, "operations", 1, "status"),
            "ready",
            "frs_crosswalk.status",
        ),
        (
            (
                "stages",
                0,
                "operations",
                2,
                "spi_prior_national_household_mass_share",
            ),
            0.25,
            "prior.mass_share",
        ),
        (
            ("stages", 0, "operations", 4, "sample_size"),
            50_000,
            "stage1.sample_size",
        ),
        (
            ("stages", 0, "operations", 4, "outputs"),
            ["employment_income"],
            "stage1.outputs",
        ),
        (
            ("stages", 0, "operations", 4, "source_ti_identity_fields"),
            ["TI"],
            "stage1.source_ti_identity_fields",
        ),
        (
            ("stages", 0, "operations", 5, "reviewed_absent_outputs"),
            {"maternity_allowance_reported": "changed"},
            "stage2.reviewed_absent_outputs",
        ),
        (
            (
                "stages",
                0,
                "operations",
                6,
                "component_columns",
                "savings_interest_income",
                "amount_column_index",
            ),
            6,
            "materialize.component_columns",
        ),
        (
            ("stages", 0, "operations", 8, "max_weight_ratio"),
            10,
            "calibration.max_weight_ratio",
        ),
        (
            ("stages", 0, "operations", 8, "maximum_abs_relative_error"),
            0.10,
            "calibration.maximum_abs_relative_error",
        ),
        (
            ("stages", 0, "operations", 9, "required_support_channel"),
            "frs",
            "effective.required_support_channel",
        ),
        (
            ("stages", 0, "operations", 9, "minimum_nondefault_mass_share"),
            0.01,
            "effective.minimum_nondefault_mass_share",
        ),
    ],
)
def test_runtime_source_contract_rejects_manifest_drift(
    tmp_path,
    path,
    replacement,
    match,
) -> None:
    payload = _manifest()
    cursor = payload
    for segment in path[:-1]:
        cursor = cursor[segment]
    cursor[path[-1]] = replacement
    tampered = tmp_path / "hmrc_income_source_stages.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        assert_uk_hmrc_income_source_contract_current(tampered)


@pytest.mark.parametrize(
    ("collection", "key"), [("artifacts", "role"), ("operations", "kind")]
)
def test_runtime_source_contract_rejects_duplicate_keys(
    tmp_path,
    collection,
    key,
) -> None:
    payload = _manifest()
    values = payload["stages"][0][collection]
    values.append(dict(values[0]))
    tampered = tmp_path / f"duplicate_{key}.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=f"duplicate {key}"):
        assert_uk_hmrc_income_source_contract_current(tampered)
