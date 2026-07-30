from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from populace.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.support_provenance import (
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
    validate_assembly_provenance,
)
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


def _with_structural_ids(frame: Frame, ids: list[int]) -> Frame:
    values = np.asarray(ids, dtype=np.int64)
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    person = tables[US_SCHEMA.person_entity]
    person[US_SCHEMA.person_id_column] = values
    for entity in US_SCHEMA.group_entities:
        tables[entity][US_SCHEMA.entity_id_column(entity)] = values
        person[US_SCHEMA.membership_column(entity)] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
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
        spine_source_id = spine_source_id_column(entity)
        clone_index = support_clone_index_column(entity)
        assert set(table[channel]) == {"asec", "acs"}
        assert table[clone_index].eq(0).all()
        assert table[source_id].equals(table[US_SCHEMA.entity_id_column(entity)])
        acs_rows = table[channel].eq("acs")
        assert table.loc[acs_rows, spine_source_id].tolist() == [
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


def test_assembly_manifest_is_immutable_and_detects_channel_forgery() -> None:
    assembled = assemble_spines(
        {"asec": _asec_frame(), "acs": _acs_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    manifest = assembled.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    with pytest.raises(TypeError):
        manifest["channels"] = ("forged_source",)

    assembled.table("person").loc[0, "person_support_channel"] = "forged_source"
    with pytest.raises(ValueError, match="assembly manifest.*unknown channel"):
        validate_assembly_provenance(
            assembled,
            boundary="test assembly output",
        )


def test_assembly_manifest_detects_cross_grain_channel_disagreement() -> None:
    assembled = assemble_spines(
        {"asec": _asec_frame(), "acs": _acs_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    person = assembled.table("person")
    # Preserve every per-channel count while assigning two people to the
    # opposite source from their linked households.
    person.loc[[0, 2], "person_support_channel"] = ["acs", "asec"]

    with pytest.raises(ValueError, match="cross-grain.*person/household"):
        validate_assembly_provenance(
            assembled,
            boundary="test assembly output",
        )


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


def test_assemble_spines__rejects_negative_source_ids_before_clone() -> None:
    negative = _with_structural_ids(_asec_frame(), [-5, 95])
    positive = _with_structural_ids(_asec_frame(), [1, 2])

    with pytest.raises(
        ValueError,
        match=r"Spine 'asec'.*negative source IDs.*-5",
    ):
        assemble_spines(
            {"asec": negative, "acs": positive},
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )


def test_assemble_then_clone_composes_for_adversarial_nonnegative_ids() -> None:
    wide = _with_structural_ids(_asec_frame(), [5, 95])
    low = _with_structural_ids(_asec_frame(), [1, 2])
    assembled = assemble_spines(
        {"asec": wide, "acs": low},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )

    cloned = clone_us_frame_for_puf_support(assembled)

    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        assert cloned.n(entity) == 2 * assembled.n(entity)
        assert not cloned.table(entity)[id_column].duplicated().any()


def test_clone_rejects_fractional_assembled_source_ids_without_truncation() -> None:
    assembled = assemble_spines(
        {"asec": _asec_frame(), "acs": _acs_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    person = assembled.table("person")
    person["person_source_id"] = person["person_source_id"].astype(np.float64)
    person.loc[0, "person_source_id"] = 1.5

    with pytest.raises(
        ValueError,
        match=r"Preassembled source IDs in 'person_source_id' must be integral",
    ):
        clone_us_frame_for_puf_support(assembled)


def test_three_spine_output_is_invariant_across_all_input_permutations() -> None:
    frames = {
        "asec": _asec_frame(),
        "acs": _acs_frame(),
        "future_survey": _acs_frame(),
    }
    shares = {"asec": 0.5, "acs": 0.25, "future_survey": 0.25}
    baseline: Frame | None = None
    seen_orders: set[tuple[str, ...]] = set()

    for order in permutations(frames):
        seen_orders.add(order)
        result = assemble_spines(
            {channel: frames[channel] for channel in order},
            household_mass_shares={channel: shares[channel] for channel in order},
        )
        if baseline is None:
            baseline = result
            continue
        for entity in US_SCHEMA.entities:
            assert_frame_equal(result.table(entity), baseline.table(entity))
        assert_series_equal(result.strata, baseline.strata)
        np.testing.assert_array_equal(
            result.weights_for("household").values,
            baseline.weights_for("household").values,
        )
        assert result.mass_log == baseline.mass_log

    assert len(seen_orders) == 6
