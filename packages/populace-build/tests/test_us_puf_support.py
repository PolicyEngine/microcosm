"""US PUF support-channel expansion tests."""

import numpy as np
import pandas as pd
import pytest

from populace.build.us import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1000, 1000, 2000], dtype="int64"),
            "person_marital_unit_id": np.asarray([10000, 10000, 20000], dtype="int64"),
            "employment_income": np.asarray([50_000, 20_000, 125_000], dtype="int64"),
            "partnership_income": [1_000.0, 2_000.0, 3_000.0],
            "s_corp_income": [4_000.0, 5_000.0, 6_000.0],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([100, 200])}),
        "family": pd.DataFrame({"family_id": np.asarray([1000, 2000])}),
        "marital_unit": pd.DataFrame({"marital_unit_id": np.asarray([10000, 20000])}),
    }
    strata = pd.Series(
        ["asec_2024", "asec_2024", "asec_2023"],
        name="stratum",
    )
    weights = {
        "household": Weights(
            values=np.asarray([100.0, 300.0]),
            kind=WeightKind.DESIGN,
        )
    }
    return Frame(tables, US_SCHEMA, weights, strata)


def test_puf_support_channel_doubles_rows_without_doubling_mass() -> None:
    frame = _minimal_us_frame()

    expanded = clone_us_frame_for_puf_support(frame)

    for entity in frame.entities:
        assert expanded.n(entity) == 2 * frame.n(entity)
    assert expanded.weights_for("household").kind == WeightKind.DESIGN
    assert (
        expanded.weights_for("household").total == frame.weights_for("household").total
    )
    assert expanded.weights_for("household").values.tolist() == [
        50.0,
        150.0,
        50.0,
        150.0,
    ]
    assert expanded.strata.tolist() == frame.strata.tolist() + frame.strata.tolist()


def test_puf_support_channel_preserves_provenance_and_remaps_linked_ids() -> None:
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())

    person = expanded.table("person")
    tax_unit = expanded.table("tax_unit")
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    puf_tax_units = tax_unit[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]

    assert person[support_channel_column("person")].tolist() == [
        BASE_ASEC_SUPPORT_CHANNEL,
        BASE_ASEC_SUPPORT_CHANNEL,
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
    assert person[support_clone_index_column("person")].tolist() == [0, 0, 0, 1, 1, 1]
    assert person[support_source_id_column("person")].tolist() == [1, 2, 3, 1, 2, 3]
    assert set(puf_people["person_tax_unit_id"]).issubset(
        set(puf_tax_units["tax_unit_id"])
    )
    assert set(puf_people["person_tax_unit_id"]).isdisjoint(
        set(
            tax_unit.loc[
                tax_unit[support_channel_column("tax_unit")] == "asec", "tax_unit_id"
            ]
        )
    )
    assert puf_people["employment_income"].tolist() == [50_000.0, 20_000.0, 125_000.0]
    assert puf_tax_units[support_source_id_column("tax_unit")].tolist() == [10, 20]


def test_puf_support_channel_refuses_duplicate_or_missing_puf_channel() -> None:
    frame = _minimal_us_frame()

    with pytest.raises(ValueError, match="must be unique"):
        clone_us_frame_for_puf_support(frame, channels=("asec", "asec"))

    with pytest.raises(ValueError, match="must start with 'asec'"):
        clone_us_frame_for_puf_support(frame, channels=("puf_tax_detail", "tail"))

    with pytest.raises(ValueError, match="must include 'puf_tax_detail'"):
        clone_us_frame_for_puf_support(frame, channels=("asec", "tail"))


def test_puf_support_channel_refuses_to_run_twice() -> None:
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())

    with pytest.raises(ValueError, match="should run exactly once"):
        clone_us_frame_for_puf_support(expanded)


def test_puf_tax_unit_donor_from_arrays_aggregates_person_values() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 10, 20],
            "employment_income": [5.0, 7.0, 11.0],
            "non_qualified_dividend_income": [1.0, 2.0, 3.0],
            "qualified_dividend_income": [4.0, 5.0, 6.0],
            "home_mortgage_interest": [10.0, 20.0, 30.0],
            "taxable_unemployment_compensation": [13.0, 17.0, 19.0],
            "state_and_local_sales_or_income_tax": [40.0, 50.0],
        },
        person_outputs=(
            "employment_income",
            "dividend_income",
            "home_mortgage_interest",
            "unemployment_compensation",
        ),
        tax_unit_outputs=("interest_deduction", "state_withheld_income_tax"),
    )

    assert donor["employment_income"].tolist() == [12.0, 11.0]
    assert donor["dividend_income"].tolist() == [12.0, 9.0]
    assert donor["unemployment_compensation"].tolist() == [30.0, 19.0]
    assert donor["interest_deduction"].tolist() == [30.0, 30.0]
    assert donor["state_withheld_income_tax"].tolist() == [40.0, 50.0]
    assert donor["puf_predictor_employment_income"].tolist() == [12.0, 11.0]
    assert donor["puf_predictor_filing_status_code"].tolist() == [1.0, 2.0]


def test_puf_tax_unit_donor_derives_partnership_and_s_corp_split_from_raw_fields() -> (
    None
):
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20],
            "E25980": [50.0, 70.0],
            "E25960": [5.0, 10.0],
            "E26190": [80.0, 110.0],
            "E26180": [20.0, 30.0],
        },
        person_outputs=("partnership_income", "s_corp_income"),
        tax_unit_outputs=("tax_unit_partnership_s_corp_income",),
    )

    assert donor["partnership_income"].tolist() == [45.0, 60.0]
    assert donor["s_corp_income"].tolist() == [60.0, 80.0]
    assert donor["tax_unit_partnership_s_corp_income"].tolist() == [105.0, 140.0]


def test_puf_tax_detail_imputation_writes_only_puf_channel() -> None:
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "filing_status_code": [1.0, 2.0, 4.0, 1.0],
            "tax_unit_person_count": [1.0, 2.0, 1.0, 2.0],
            "employment_income": [1_000.0, 1_000.0, 1_000.0, 1_000.0],
            "state_withheld_income_tax": [100.0, 100.0, 100.0, 100.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=("employment_income",),
        tax_unit_outputs=("state_withheld_income_tax",),
        n_estimators=4,
        seed=0,
    )

    person = imputed.table("person")
    tax_unit = imputed.table("tax_unit")
    asec_people = person[
        person[support_channel_column("person")] == BASE_ASEC_SUPPORT_CHANNEL
    ]
    puf_people = person[
        person[support_channel_column("person")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]
    puf_tax_units = tax_unit[
        tax_unit[support_channel_column("tax_unit")] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    ]

    assert asec_people["employment_income"].tolist() == [50_000.0, 20_000.0, 125_000.0]
    np.testing.assert_allclose(
        puf_people.groupby("person_tax_unit_id")["employment_income"].sum().to_numpy(),
        [1_000.0, 1_000.0],
    )
    assert tax_unit.loc[
        tax_unit[support_channel_column("tax_unit")] == BASE_ASEC_SUPPORT_CHANNEL,
        "state_withheld_income_tax",
    ].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(
        puf_tax_units["state_withheld_income_tax"].to_numpy(),
        [100.0, 100.0],
    )
    np.testing.assert_allclose(
        puf_tax_units["tax_unit_partnership_s_corp_income"].to_numpy(),
        [12_000.0, 9_000.0],
    )
