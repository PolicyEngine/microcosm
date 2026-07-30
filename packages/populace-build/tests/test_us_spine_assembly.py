from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from populace.build.us_runtime.puf_support import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _asec_frame() -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1, 2], dtype=np.int64),
                    "person_household_id": np.asarray([10, 20], dtype=np.int64),
                    "person_tax_unit_id": np.asarray([100, 200], dtype=np.int64),
                    "person_spm_unit_id": np.asarray([1_000, 2_000], dtype=np.int64),
                    "person_family_id": np.asarray([10_000, 20_000], dtype=np.int64),
                    "person_marital_unit_id": np.asarray(
                        [100_000, 200_000], dtype=np.int64
                    ),
                    "age": np.asarray([30, 40], dtype=np.int64),
                    "asec_measured_income": np.asarray([10.0, 20.0]),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([10, 20], dtype=np.int64),
                    "state_fips": np.asarray([6, 36], dtype=np.int64),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([100, 200], dtype=np.int64)}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([1_000, 2_000], dtype=np.int64)}
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([10_000, 20_000], dtype=np.int64)}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([100_000, 200_000], dtype=np.int64)}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([100.0, 300.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["asec_2024", "asec_2023"], dtype=object),
    )


def _acs_frame(*, age_dtype: str = "int64") -> Frame:
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.asarray([1], dtype=np.int64),
                    "person_household_id": np.asarray([10], dtype=np.int64),
                    "person_tax_unit_id": np.asarray([100], dtype=np.int64),
                    "person_spm_unit_id": np.asarray([1_000], dtype=np.int64),
                    "person_family_id": np.asarray([10_000], dtype=np.int64),
                    "person_marital_unit_id": np.asarray([100_000], dtype=np.int64),
                    "age": np.asarray([50], dtype=age_dtype),
                    "acs_measured_income": np.asarray([30.0]),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.asarray([10], dtype=np.int64),
                    "state_fips": np.asarray([12], dtype=np.int64),
                    "puma": pd.Series(["1200101"], dtype=object),
                }
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": np.asarray([100], dtype=np.int64)}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([1_000], dtype=np.int64)}
            ),
            "family": pd.DataFrame({"family_id": np.asarray([10_000], dtype=np.int64)}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([100_000], dtype=np.int64)}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([50.0]),
                WeightKind.CALIBRATED,
            )
        },
        pd.Series(["acs_2024_1yr"], dtype=object),
    )


def _snapshot(frame: Frame) -> tuple[dict[str, pd.DataFrame], pd.Series, np.ndarray]:
    return (
        {entity: frame.table(entity).copy(deep=True) for entity in frame.entities},
        frame.strata.copy(deep=True),
        frame.weights_for("household").values.copy(),
    )


def test_assemble_spines__combines_raw_sources_before_operators() -> None:
    asec = _asec_frame()
    acs = _acs_frame()
    asec_before = _snapshot(asec)
    acs_before = _snapshot(acs)

    result = assemble_spines(
        {"acs": acs, "asec": asec},
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )

    assert result.n("person") == 3
    assert result.weights_for("household").kind is WeightKind.IMPORTANCE
    np.testing.assert_allclose(
        result.weights_for("household").values,
        np.asarray([75.0, 225.0, 100.0]),
    )
    assert result.weights_for("household").total == pytest.approx(400.0)
    assert result.strata.tolist() == ["asec_2024", "asec_2023", "acs_2024_1yr"]

    person = result.table("person")
    assert person["person_id"].tolist() == [1, 2, 3]
    assert person["person_household_id"].tolist() == [10, 20, 21]
    assert person["age"].tolist() == [30, 40, 50]
    assert person["asec_measured_income"].tolist()[:2] == [10.0, 20.0]
    assert np.isnan(person["asec_measured_income"].iloc[2])
    assert np.isnan(person["acs_measured_income"].iloc[0])
    assert np.isnan(person["acs_measured_income"].iloc[1])
    assert person["acs_measured_income"].iloc[2] == 30.0
    assert person["asec_measured_income"].dtype == np.dtype(np.float64)
    assert person["acs_measured_income"].dtype == np.dtype(np.float64)

    for entity in US_SCHEMA.entities:
        table = result.table(entity)
        channel = support_channel_column(entity)
        source_id = support_source_id_column(entity)
        clone_index = support_clone_index_column(entity)
        assert set(table[channel]) == {"asec", "acs"}
        assert table[clone_index].eq(0).all()
        acs_rows = table[channel].eq("acs")
        assert table.loc[acs_rows, source_id].tolist() == [
            _acs_frame().table(entity)[US_SCHEMA.entity_id_column(entity)].iloc[0]
        ]

    for frame, snapshot in ((asec, asec_before), (acs, acs_before)):
        tables, strata, weights = snapshot
        for entity in frame.entities:
            assert_frame_equal(frame.table(entity), tables[entity])
        assert_series_equal(frame.strata, strata)
        np.testing.assert_array_equal(frame.weights_for("household").values, weights)

    assert len(result.mass_log) == 2
    assert [record.new_total for record in result.mass_log] == pytest.approx(
        [300.0, 100.0]
    )
    assert all(
        "pre-operator spine assembly" in record.reason for record in result.mass_log
    )


def test_assemble_spines__accepts_a_future_source_channel() -> None:
    result = assemble_spines(
        {"asec": _asec_frame(), "future_survey": _acs_frame()},
        household_mass_shares={"asec": 0.5, "future_survey": 0.5},
    )

    assert set(result.table("household")["household_support_channel"]) == {
        "asec",
        "future_survey",
    }


@pytest.mark.parametrize(
    ("spines", "shares", "match"),
    [
        (
            {"asec": _asec_frame()},
            {"asec": 1.0},
            "at least two peer",
        ),
        (
            {"asec": _asec_frame(), "puf_tax_detail": _acs_frame()},
            {"asec": 0.5, "puf_tax_detail": 0.5},
            "clone operator channel",
        ),
        (
            {"asec": _asec_frame(), "acs": _acs_frame()},
            {"asec": 1.0},
            "keys must exactly match",
        ),
        (
            {"asec": _asec_frame(), "acs": _acs_frame()},
            {"asec": 0.6, "acs": 0.5},
            "sum to one",
        ),
    ],
)
def test_assemble_spines__rejects_invalid_contracts(
    spines: dict[str, Frame],
    shares: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        assemble_spines(spines, household_mass_shares=shares)


def test_assemble_spines__rejects_shared_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="identical dtypes.*age"):
        assemble_spines(
            {"asec": _asec_frame(), "acs": _acs_frame(age_dtype="float64")},
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )


def test_assemble_spines__owns_support_provenance() -> None:
    asec = _asec_frame()
    tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    tables["person"]["person_source_id"] = tables["person"]["person_id"]
    pretagged = Frame(
        tables,
        US_SCHEMA,
        {"household": asec.weights_for("household")},
        asec.strata,
    )

    with pytest.raises(ValueError, match="provenance owner"):
        assemble_spines(
            {"asec": pretagged, "acs": _acs_frame()},
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )
