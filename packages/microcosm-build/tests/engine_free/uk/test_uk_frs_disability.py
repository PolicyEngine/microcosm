"""Tests split from packages/microcosm-build/tests/test_uk_frs_disability.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_disability import *


def test_disability_category_threshold_and_overwrite() -> None:
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [
                (20 - 1) * WEEKS_IN_YEAR,
                (20 - 1.01) * WEEKS_IN_YEAR,
            ],
            "dla_sc_reported": [0, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["aa_category"].tolist() == ["HIGHER", "LOWER"]


def test_disability_flag_operator_asymmetry_and_afcs() -> None:
    boundary = (30 - 1) * WEEKS_IN_YEAR
    person = pd.DataFrame(
        {
            "attendance_allowance_reported": [0, 0],
            "dla_sc_reported": [boundary, 0],
            "dla_m_reported": [0, 0],
            "pip_m_reported": [0, 0],
            "pip_dl_reported": [0, 0],
            "sda_reported": [0, 0],
            "incapacity_benefit_reported": [0, 0],
            "iidb_reported": [0, 0],
            "afcs_reported": [0, 1],
            "esa_contrib_reported": [0, 0],
            "esa_income_reported": [0, 0],
        }
    )

    result = derive_frs_disability(
        person, category_rates=_category_rates(), flag_rates=_flags()
    )

    assert result["is_enhanced_disabled_for_benefits"].tolist() == [False, False]
    assert result["is_severely_disabled_for_benefits"].tolist() == [True, True]


@pytest.mark.parametrize("operation_index", [0, 1])
def test_disability_stage_rejects_year_rule_drift(operation_index: int) -> None:
    stage = load_country_spec("uk").sources.stage_map()["frs_disability"]
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        operation.kind,
        {**operation.parameters, "year_rule": "calibration_year"},
    )

    with pytest.raises(ValueError, match="declaration drifted"):
        _assert_frs_disability_stage_parameters(
            replace(stage, operations=tuple(operations))
        )


def test_disability_stage_resolves_the_survey_year_not_the_frame_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Survey year and frame period are both 2024 on the current release, so a
    # frame stamped 2023 is the only way to see which one picks the rates.
    from microcosm.build.uk_runtime import frs_disability
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_time_period,
    )

    stage = load_country_spec("uk").sources.stage_map()["frs_disability"]
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1],
                "person_benunit_id": [1],
                "person_household_id": [1],
                "attendance_allowance_reported": [20 * WEEKS_IN_YEAR],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1]}),
        time_period="2023",
        household_weights=[1.0],
    )
    years: list[int] = []

    def _category_reader(year: int) -> UKDWPDisabilityCategoryRates:
        years.append(year)
        return _category_rates()

    def _flag_reader(year: int) -> UKDWPDisabilityFlagRates:
        years.append(year)
        return _flags()

    monkeypatch.setattr(
        frs_disability, "uk_dwp_disability_category_rates", _category_reader
    )
    monkeypatch.setattr(frs_disability, "uk_dwp_disability_flag_rates", _flag_reader)

    result = frs_disability.UKFRSDisabilityStageTransform(stage=stage)(frame)

    assert years == [2024, 2024]
    assert uk_time_period(result) == "2023"
    assert result.table("person")["aa_category"].tolist() == ["HIGHER"]
