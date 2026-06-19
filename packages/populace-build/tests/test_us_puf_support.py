"""US PUF support-channel expansion tests."""

import numpy as np
import pandas as pd
import pytest

from populace.build.us import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
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
            "employment_income": [50_000.0, 20_000.0, 125_000.0],
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
