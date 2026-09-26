from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _acs_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 1],
            "person_tax_unit_id": [1, 1],
            "person_spm_unit_id": [1, 1],
            "person_family_id": [1, 1],
            "person_marital_unit_id": [1, 2],
            "AGEP": [40, 12],
            "SEX": [2, 1],
            "RELSHIPP": [20, 25],
            "ADJINC": [1_100_000, 1_100_000],
            "WAGP": [50_000.0, np.nan],
            "SEMP": [-500.0, np.nan],
            "SSP": [1_000.0, np.nan],
            "SSIP": [200.0, np.nan],
            "RETP": [300.0, np.nan],
            "INTP": [-100.0, np.nan],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": [1],
                "ADJHSG": [1_250_000],
                "TEN": [3],
                "RNTP": [1_000.0],
                "GRNTP": [1_200.0],
                "TAXAMT": [np.nan],
            }
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": [1], "filing_status_input": ["HEAD_OF_HOUSEHOLD"]}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [1]}),
        "family": pd.DataFrame({"family_id": [1]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([10.0]), WeightKind.DESIGN)},
        pd.Series(["acs_2024_1yr", "acs_2024_1yr"], name="stratum"),
    )


def test_acs_demographic_and_relationship_mapping_is_native() -> None:
    result = map_acs_native_inputs(_acs_frame())
    person = result.frame.table("person")

    assert person["age"].tolist() == [40.0, 12.0]
    assert person["is_female"].tolist() == [True, False]
    assert person["is_household_head"].tolist() == [True, False]
    assert result.native_inputs["age"]["source_columns"] == ["AGEP"]
    assert result.native_inputs["is_household_head"]["source_columns"] == ["RELSHIPP"]


def _hours_frame(**columns) -> Frame:
    before = _acs_frame()
    tables = {entity: before.table(entity).copy() for entity in before.entities}
    for column, values in columns.items():
        tables["person"][column] = values
    return Frame(
        tables,
        before.schema,
        {"household": before.weights_for("household")},
        before.strata,
    )


def test_acs_usual_hours_preserve_forty_and_allocation_despite_current_unemployment():
    before = _hours_frame(WKHP=[40, np.nan], WKL=[1, np.nan], FWKHP=[1, 0], ESR=[3, 0])
    result = map_acs_native_inputs(before)
    assert result.frame.person["weekly_hours_worked_before_lsr"].iloc[0] == 40
    assert pd.isna(result.frame.person["weekly_hours_worked_before_lsr"].iloc[1])
    assert "hours_worked_last_week" not in result.frame.person
    pd.testing.assert_series_equal(result.frame.person["WKHP"], before.person["WKHP"])
    pd.testing.assert_series_equal(result.frame.person["FWKHP"], before.person["FWKHP"])
    receipt = result.native_inputs["weekly_hours_worked_before_lsr"]
    assert receipt["source_value_rows"] == 1
    assert receipt["structural_zero_rows"] == 0
    assert receipt["source_universe_unavailable_rows"] == 1
    assert receipt["allocated_value_rows"] == 1
    assert receipt["allocation_unknown_value_rows"] == 0
    assert receipt["missing_rows"] == 1
    assert receipt["observed_rows"] == 1


@pytest.mark.parametrize("wkl", [2, 3])
def test_acs_blank_usual_hours_are_zero_for_confirmed_past_year_nonworkers(wkl):
    result = map_acs_native_inputs(_hours_frame(WKHP=[" ", np.nan], WKL=[wkl, np.nan]))
    assert result.frame.person["weekly_hours_worked_before_lsr"].iloc[0] == 0
    assert pd.isna(result.frame.person["weekly_hours_worked_before_lsr"].iloc[1])
    receipt = result.native_inputs["weekly_hours_worked_before_lsr"]
    assert receipt["structural_zero_rows"] == 1
    assert receipt["source_universe_unavailable_rows"] == 1


@pytest.mark.parametrize("wkl", [1, np.nan])
def test_acs_eligible_or_unknown_hours_blank_stays_unresolved(wkl):
    result = map_acs_native_inputs(
        _hours_frame(WKHP=[np.nan, np.nan], WKL=[wkl, np.nan])
    )
    assert pd.isna(result.frame.person["weekly_hours_worked_before_lsr"].iloc[0])
    assert result.native_inputs["weekly_hours_worked_before_lsr"]["missing_rows"] == 2


@pytest.mark.parametrize("age", [0, 12, 15])
def test_acs_under_sixteen_niu_is_unavailable_not_observed_nonwork(age):
    result = map_acs_native_inputs(
        _hours_frame(AGEP=[40, age], WKHP=[40, np.nan], WKL=[1, np.nan])
    )
    assert pd.isna(result.frame.person["weekly_hours_worked_before_lsr"].iloc[1])
    receipt = result.native_inputs["weekly_hours_worked_before_lsr"]
    assert receipt["source_universe_unavailable_rows"] == 1
    assert receipt["observed_rows"] == 1
    assert receipt["structural_zero_rows"] == 0


def test_acs_absent_hours_source_does_not_become_a_structural_zero():
    result = map_acs_native_inputs(_hours_frame(WKL=[2, np.nan]))
    assert "weekly_hours_worked_before_lsr" not in result.frame.person
    assert "weekly_hours_worked_before_lsr" not in result.native_inputs


def test_acs_usual_hours_topcode_and_unknown_allocation_are_explicit():
    result = map_acs_native_inputs(_hours_frame(WKHP=[99, np.nan], WKL=[1, np.nan]))
    assert result.frame.person["weekly_hours_worked_before_lsr"].iloc[0] == 99
    assert pd.isna(result.frame.person["weekly_hours_worked_before_lsr"].iloc[1])
    assert (
        result.native_inputs["weekly_hours_worked_before_lsr"][
            "allocation_unknown_value_rows"
        ]
        == 1
    )


@pytest.mark.parametrize("invalid", [0, -1, 100, 40.5, "unknown", np.inf])
def test_acs_ftp_hours_reject_invalid_codes_including_api_zero(invalid):
    with pytest.raises(ValueError, match="WKHP requires blank or integer"):
        map_acs_native_inputs(_hours_frame(WKHP=[invalid, np.nan]))


@pytest.mark.parametrize(
    "columns",
    [
        {"WKHP": [40, 10]},
        {"WKHP": [40, np.nan], "WKL": [2, np.nan]},
        {"WKHP": [40, np.nan], "WKL": [1, 2]},
        {"WKHP": [40, np.nan], "FWKHP": [2, 0]},
    ],
)
def test_acs_usual_hours_refuse_conflicting_universe_or_invalid_allocation(columns):
    with pytest.raises(ValueError):
        map_acs_native_inputs(_hours_frame(**columns))


def test_acs_income_mapping_adjusts_native_dollars_without_splitting_aggregates() -> (
    None
):
    result = map_acs_native_inputs(_acs_frame())
    person = result.frame.table("person")

    assert person.loc[0, "employment_income_before_lsr"] == pytest.approx(55_000.0)
    assert person.loc[0, "self_employment_income_before_lsr"] == pytest.approx(-550.0)
    assert person.loc[0, "ssi_reported"] == pytest.approx(220.0)
    assert person.loc[0, "acs_social_security_income"] == pytest.approx(1_100.0)
    assert person.loc[0, "acs_retirement_income"] == pytest.approx(330.0)
    assert person.loc[0, "acs_interest_dividend_rental_income"] == pytest.approx(-110.0)
    assert pd.isna(person.loc[1, "employment_income_before_lsr"])
    assert pd.isna(person.loc[1, "acs_retirement_income"])
    assert "social_security" not in person
    assert "taxable_private_pension_income" not in person
    assert "interest_income" not in person
    assert result.native_inputs["acs_social_security_income"]["transformation"] == (
        "SSP * ADJINC / 1_000_000"
    )
    assert result.native_inputs["employment_income_before_lsr"]["observed_rows"] == 1
    assert result.native_inputs["employment_income_before_lsr"]["missing_rows"] == 1


def test_acs_housing_mapping_preserves_rent_without_synthesizing_model_rent() -> None:
    result = map_acs_native_inputs(_acs_frame())
    person = result.frame.table("person")
    household = result.frame.table("household")
    spm_unit = result.frame.table("spm_unit")

    assert household["tenure_type"].tolist() == ["RENTED"]
    assert spm_unit["spm_unit_tenure_type"].tolist() == ["RENTER"]
    assert household["acs_monthly_contract_rent"].tolist() == [1_250.0]
    assert household["acs_monthly_gross_rent"].tolist() == [1_500.0]
    assert household["acs_annual_property_tax"].isna().all()
    assert "pre_subsidy_rent" not in person
    assert person["real_estate_taxes"].isna().all()
    assert result.native_inputs["acs_monthly_contract_rent"]["source_columns"] == [
        "RNTP",
        "ADJHSG",
    ]


def test_acs_owner_property_tax_mapping_places_observed_amount_on_head() -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"]
    household["TEN"] = 1
    household["RNTP"] = np.nan
    household["GRNTP"] = np.nan
    household["TAXAMT"] = 2_400.0
    owner = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    result = map_acs_native_inputs(owner)

    assert result.frame.table("household")["tenure_type"].tolist() == [
        "OWNED_WITH_MORTGAGE"
    ]
    assert result.frame.table("person")["real_estate_taxes"].tolist() == [
        3_000.0,
        0.0,
    ]


def test_acs_group_quarters_blank_tenure_is_not_synthesized() -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"]
    household["TEN"] = np.nan
    household["TYPEHUGQ"] = 3
    household[["RNTP", "GRNTP", "TAXAMT"]] = np.nan
    group_quarters = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    result = map_acs_native_inputs(group_quarters)

    assert result.frame.table("household")["tenure_type"].isna().all()
    assert result.frame.table("spm_unit")["spm_unit_tenure_type"].isna().all()
    assert result.native_inputs["tenure_type"]["observed_rows"] == 0
    assert result.native_inputs["tenure_type"]["missing_rows"] == 1


@pytest.mark.parametrize(
    ("factor", "invalid"),
    [("ADJINC", np.nan), ("ADJINC", np.inf), ("ADJHSG", np.nan)],
)
def test_acs_observed_dollars_require_finite_positive_adjustment(
    factor: str,
    invalid: float,
) -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    entity = "person" if factor == "ADJINC" else "household"
    tables[entity][factor] = tables[entity][factor].astype(float)
    tables[entity].loc[0, factor] = invalid
    invalid_frame = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    with pytest.raises(ValueError, match=f"{factor} must be finite and positive"):
        map_acs_native_inputs(invalid_frame)


def test_acs_native_mapping_refuses_existing_output_collision() -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["age"] = [999.0, 999.0]
    collision = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    with pytest.raises(ValueError, match="refuses to overwrite.*age"):
        map_acs_native_inputs(collision)


def test_acs_unmapped_missing_source_columns_stay_absent() -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=["RETP", "INTP"])
    sparse = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    result = map_acs_native_inputs(sparse)

    assert "acs_retirement_income" not in result.frame.table("person")
    assert "acs_interest_dividend_rental_income" not in result.frame.table("person")
    assert "acs_retirement_income" not in result.native_inputs
    assert "acs_interest_dividend_rental_income" not in result.native_inputs
