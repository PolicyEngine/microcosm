from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from microcosm.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    clone_us_frame_for_puf_support,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.support_provenance import (
    PUF_TAX_DETAIL_CLONE_INDEX,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_provenance_counts,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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


def test_assembly_manifest_detects_declared_channel_count_drift() -> None:
    assembled = assemble_spines(
        {"asec": _asec_frame(), "acs": _acs_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    person = assembled.table("person")
    person.loc[0, "person_support_channel"] = "acs"

    with pytest.raises(
        ValueError,
        match="native row counts drifted from the assembly manifest",
    ):
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


def _cloned_assembly() -> Frame:
    return clone_us_frame_for_puf_support(
        assemble_spines(
            {"asec": _asec_frame(), "acs": _acs_frame()},
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )
    )


_ASSEMBLY_PROVENANCE_READERS = pytest.mark.parametrize(
    "reader",
    [validate_assembly_provenance, spine_provenance_counts],
    ids=["validate", "counts"],
)


@_ASSEMBLY_PROVENANCE_READERS
@pytest.mark.parametrize(
    "bad_index",
    [np.inf, -np.inf, np.nan, 1.5, -1.0, 2.0**63],
    ids=[
        "inf",
        "negative_inf",
        "nan",
        "non_integer",
        "negative_float",
        "float_past_int64",
    ],
)
def test_assembly_provenance_rejects_malformed_clone_indices_before_casting(
    reader: Callable[..., object],
    bad_index: float,
) -> None:
    # Corrupt a PUF-detail copy rather than a native row, so the frozen
    # native-row counts still match: only clone-index validation can refuse.
    # Before #992's gate fix, +inf here passed validation and the receipt
    # counter then died in an opaque pandas cast.
    cloned = _cloned_assembly()
    person = cloned.table("person")
    column = support_clone_index_column("person")
    copy_row = person.index[person[column].eq(PUF_TAX_DETAIL_CLONE_INDEX)][0]
    person[column] = person[column].astype(np.float64)
    person.loc[copy_row, column] = bad_index

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            ValueError,
            match=(
                r"test assembly output: assembly manifest provenance column "
                r"'person_support_clone_index' must contain nonnegative integers "
                r"\(finite and representable as int64\)"
            ),
        ):
            reader(cloned, boundary="test assembly output")


def test_assembly_provenance_readers_accept_integral_float_clone_indices() -> None:
    cloned = _cloned_assembly()
    baseline = spine_provenance_counts(cloned, boundary="test assembly output")
    person_counts = baseline["person"]["by_clone_index"]
    assert person_counts == {"0": 3, "1": 3}

    for entity in cloned.entities:
        table = cloned.table(entity)
        column = support_clone_index_column(entity)
        table[column] = table[column].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert validate_assembly_provenance(
            cloned,
            boundary="test assembly output",
        )
        assert (
            spine_provenance_counts(cloned, boundary="test assembly output") == baseline
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


def test_assemble_spines__string_storage_mismatch_is_named_in_the_error() -> None:
    """Divergent pandas string storages must be visible in the refusal.

    Both python- and pyarrow-backed 'str' dtypes print identically under
    str(); the first full-scale run failed with an error whose two sides both
    read 'str'. The refusal must render repr so the storages are legible.
    """

    pytest.importorskip("pyarrow")

    def _with_probe(frame: Frame, storage: str, value: str) -> Frame:
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        person = tables["person"]
        person["stacked_probe_label"] = pd.Series(
            [value] * len(person),
            index=person.index,
        ).astype(pd.StringDtype(storage=storage, na_value=np.nan))
        return Frame(
            tables,
            US_SCHEMA,
            {"household": frame.weights_for("household")},
            frame.strata,
        )

    with pytest.raises(
        ValueError, match="identical dtypes.*stacked_probe_label"
    ) as excinfo:
        assemble_spines(
            {
                "asec": _with_probe(_asec_frame(), "python", "x"),
                "acs": _with_probe(_acs_frame(), "pyarrow", "y"),
            },
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )
    message = str(excinfo.value)
    # The python-backed side names its storage explicitly; pandas omits the
    # storage kwarg from the repr of the environment-default (pyarrow) side.
    # Both reprs must appear and must differ — under str() both sides
    # rendered as 'str' and the mismatch was unreadable.
    asec_repr = repr(pd.StringDtype(storage="python", na_value=np.nan))
    acs_repr = repr(pd.StringDtype(storage="pyarrow", na_value=np.nan))
    assert asec_repr != acs_repr
    assert asec_repr in message
    assert acs_repr in message


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


def test_clone_safe_bound_is_the_documented_literal() -> None:
    """The bound is part of the assembly/clone composition contract:
    10**15 - 1 caps the decimal multiplier at 10**16, leaving clone
    indices up to 921 inside int64. A drifted constant must fail HERE —
    the payload-based tests below derive from the constant and cannot
    see it move (sol round-3 finding)."""

    assert PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID == 10**15 - 1


def test_remap_ids_raises_governed_error_instead_of_overflowing() -> None:
    """Direct _remap_ids regression: overflow-shaped inputs must raise
    the governed ValueError naming the bound, never OverflowError."""

    from microcosm.build.us_runtime.puf_support import _remap_ids

    with pytest.raises(ValueError, match=r"overflow int64.*999999999999999"):
        _remap_ids(
            np.array([10**18], dtype="int64"),
            clone_index=9,
            id_multiplier=10**19,
        )


def test_assemble_spines__rejects_ids_above_the_clone_safe_bound() -> None:
    """Round-2 blocker: int64-valid IDs above the shared bound assembled
    fine, then overflowed the clone stage's decimal remap (OverflowError:
    'Python int too large to convert to C long'). Assembly now rejects
    them at the door, naming the shared constant's value."""

    oversized = _with_structural_ids(
        _asec_frame(),
        [1, PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID + 1],
    )
    low = _with_structural_ids(_asec_frame(), [1, 2])

    with pytest.raises(ValueError, match=r"clone-safe bound"):
        assemble_spines(
            {"asec": oversized, "acs": low},
            household_mass_shares={"asec": 0.5, "acs": 0.5},
        )


def test_assemble_then_clone_composes_at_the_clone_safe_bound() -> None:
    """The bound itself must compose: assembly accepts it and the clone
    remap stays inside int64 (multiplier 10**16, clone index 1)."""

    at_bound = _with_structural_ids(
        _asec_frame(), [1, PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID]
    )
    low = _with_structural_ids(_asec_frame(), [2, 3])
    assembled = assemble_spines(
        {"asec": at_bound, "acs": low},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )

    cloned = clone_us_frame_for_puf_support(assembled)

    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        assert not cloned.table(entity)[id_column].duplicated().any()
        assert (cloned.table(entity)[id_column] >= 0).all()


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


def _with_columns(frame: Frame, entity: str, columns: dict[str, object]) -> Frame:
    tables = {name: frame.table(name).copy(deep=True) for name in frame.entities}
    for column, values in columns.items():
        tables[entity][column] = values
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _asec_with_cps_only_integer_carries() -> Frame:
    # CPS-only integer carries: household tenure (H_TENURE today; HPUBLIC and
    # HLORENT ride the same path), raw person codes, and the unsigned and
    # pandas-nullable integer storages a source could use.
    asec = _with_columns(
        _asec_frame(),
        "household",
        {
            "H_TENURE": np.asarray([1, 3], dtype=np.int64),
            "HPUBLIC": np.asarray([0, 2], dtype=np.int64),
        },
    )
    return _with_columns(
        asec,
        "person",
        {
            "A_HGA": np.asarray([39, 43], dtype=np.int64),
            "A_FLAG": np.asarray([0, 255], dtype=np.uint8),
            "NULLABLE_CODE": pd.array([7, None], dtype="Int64"),
            "NULLABLE_UNSIGNED_CODE": pd.array([3, None], dtype="UInt8"),
            "ALL_MISSING_CODE": pd.array([None, None], dtype="Int64"),
        },
    )


def _acs_with_typehugq() -> Frame:
    # An ACS-only integer carry, padded on CPS rows, with no unfilled boolean:
    # an object boolean plus None is table-writable only once a CPS producer
    # fills it (see the boolean test below).
    return _with_columns(
        _acs_frame(),
        "household",
        {"TYPEHUGQ": np.asarray([1], dtype=np.int64)},
    )


def _acs_with_acs_only_columns() -> Frame:
    return _with_columns(
        _acs_with_typehugq(),
        "person",
        {"is_female": np.asarray([True], dtype=bool)},
    )


def test_assemble_spines__pads_source_only_integer_columns_as_float64() -> None:
    result = assemble_spines(
        {
            "asec": _asec_with_cps_only_integer_carries(),
            "acs": _acs_with_acs_only_columns(),
        },
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )

    household = result.table("household")
    acs_households = household[support_channel_column("household")].eq("acs")
    for column, expected in {"H_TENURE": [1.0, 3.0], "HPUBLIC": [0.0, 2.0]}.items():
        assert household[column].dtype == np.dtype(np.float64), column
        assert household.loc[~acs_households, column].tolist() == expected
        assert household.loc[acs_households, column].isna().all()
    assert household["TYPEHUGQ"].dtype == np.dtype(np.float64)
    assert household.loc[acs_households, "TYPEHUGQ"].tolist() == [1.0]
    assert household.loc[~acs_households, "TYPEHUGQ"].isna().all()
    # A column every spine carries keeps its integer dtype and values.
    assert household["state_fips"].dtype == np.dtype(np.int64)
    assert household["state_fips"].tolist() == [6, 36, 12]

    person = result.table("person")
    acs_people = person[support_channel_column("person")].eq("acs")
    for column, expected in {
        "A_HGA": [39.0, 43.0],
        "A_FLAG": [0.0, 255.0],
    }.items():
        assert person[column].dtype == np.dtype(np.float64), column
        assert person.loc[~acs_people, column].tolist() == expected
        assert person.loc[acs_people, column].isna().all()
    for column, expected in {
        "NULLABLE_CODE": 7.0,
        "NULLABLE_UNSIGNED_CODE": 3.0,
    }.items():
        # pandas nullable integers become float64, not Float64: a plain
        # concat with NaN padding would give Float64, which PyTables refuses.
        assert person[column].dtype == np.dtype(np.float64), column
        assert person.loc[~acs_people, column].iloc[0] == expected
        assert person[column].isna().tolist() == [False, True, True]
    assert person["ALL_MISSING_CODE"].dtype == np.dtype(np.float64)
    assert person["ALL_MISSING_CODE"].isna().all()
    assert person["age"].dtype == np.dtype(np.int64)
    # Booleans are not widened: the source-operator merge reads padded booleans
    # as object booleans plus None (multispine_pool._merge_source_operator_outputs).
    assert person["is_female"].dtype == np.dtype(object)
    assert person["is_female"].tolist() == [None, None, True]


def test_assemble_spines__padding_leaves_columns_every_spine_carries_unchanged() -> (
    None
):
    asec = _asec_frame()
    acs = _acs_frame()
    result = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )
    padded = assemble_spines(
        {"asec": _asec_with_cps_only_integer_carries(), "acs": acs},
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )

    for entity in US_SCHEMA.entities:
        shared = result.table(entity)
        assert_frame_equal(padded.table(entity).loc[:, shared.columns], shared)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1, 2**53 + 1], dtype=np.int64),
        np.asarray([-(2**53) - 1, 0], dtype=np.int64),
        np.asarray([1, 2**63], dtype=np.uint64),
    ],
    ids=["above_bound", "below_bound", "uint64_above_bound"],
)
def test_assemble_spines__refuses_integers_float64_cannot_hold_exactly(
    values: np.ndarray,
) -> None:
    asec = _with_columns(_asec_frame(), "person", {"WIDE_CODE": values})

    with pytest.raises(
        ValueError,
        match=r"'person' column 'WIDE_CODE' is absent from spine\(s\) \['acs'\]",
    ):
        assemble_spines(
            {"asec": asec, "acs": _acs_frame()},
            household_mass_shares={"asec": 0.75, "acs": 0.25},
        )


def test_assemble_spines__keeps_integers_at_the_float64_exact_bound() -> None:
    at_bound = _with_columns(
        _asec_frame(),
        "person",
        {
            "WIDE_CODE": np.asarray([-(2**53), 2**53], dtype=np.int64),
            "WIDE_UNSIGNED_CODE": np.asarray([0, 2**53], dtype=np.uint64),
        },
    )
    result = assemble_spines(
        {"asec": at_bound, "acs": _acs_frame()},
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )
    person = result.table("person")
    values = person["WIDE_CODE"].to_numpy()
    assert values[:2].tolist() == [float(-(2**53)), float(2**53)]
    assert int(values[1]) == 2**53
    assert int(person["WIDE_UNSIGNED_CODE"].to_numpy()[1]) == 2**53


def _three_spines_with_tenure(future_tenure: int) -> dict[str, Frame]:
    # H_TENURE on two of three spines: the anchor and a later carrier.
    return {
        "asec": _with_columns(
            _asec_frame(),
            "household",
            {"H_TENURE": np.asarray([1, 3], dtype=np.int64)},
        ),
        "acs": _acs_frame(),
        "future_survey": _with_columns(
            _acs_frame(),
            "household",
            {"H_TENURE": np.asarray([future_tenure], dtype=np.int64)},
        ),
    }


def test_assemble_spines__checks_exactness_on_every_carrying_spine() -> None:
    with pytest.raises(
        ValueError,
        match=r"Spine 'future_survey' 'household' column 'H_TENURE' is absent "
        r"from spine\(s\) \['acs'\]",
    ):
        assemble_spines(
            _three_spines_with_tenure(2**53 + 1),
            household_mass_shares={"asec": 0.5, "acs": 0.25, "future_survey": 0.25},
        )


def test_assemble_spines__integer_carry_on_two_of_three_spines_is_order_free() -> None:
    frames = _three_spines_with_tenure(2)
    shares = {"asec": 0.5, "acs": 0.25, "future_survey": 0.25}
    baseline: Frame | None = None

    for order in permutations(frames):
        result = assemble_spines(
            {channel: frames[channel] for channel in order},
            household_mass_shares={channel: shares[channel] for channel in order},
        )
        household = result.table("household")
        channel = household[support_channel_column("household")]
        assert household["H_TENURE"].dtype == np.dtype(np.float64)
        assert household.loc[channel.eq("asec"), "H_TENURE"].tolist() == [1.0, 3.0]
        assert household.loc[channel.eq("future_survey"), "H_TENURE"].tolist() == [2.0]
        assert household.loc[channel.eq("acs"), "H_TENURE"].isna().all()
        if baseline is None:
            baseline = result
            continue
        for entity in US_SCHEMA.entities:
            assert_frame_equal(result.table(entity), baseline.table(entity))


def test_assembled_integer_carries_round_trip_through_table_format_hdf(
    tmp_path,
) -> None:
    """Assembly, then the PUF clone, then the national release writer's
    put_frame_table(format="table") path, with integer carries padded both
    ways (CPS-only codes on ACS rows, ACS-only TYPEHUGQ on CPS rows)."""

    from microcosm.frame import put_frame_table, read_frame_table

    assembled = assemble_spines(
        {"asec": _asec_with_cps_only_integer_carries(), "acs": _acs_with_typehugq()},
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )
    result = clone_us_frame_for_puf_support(assembled)
    path = tmp_path / "cloned.h5"
    with pd.HDFStore(path, mode="w") as store:
        for entity in US_SCHEMA.entities:
            put_frame_table(
                store,
                entity,
                result.table(entity),
                preferred_format="table",
                data_columns=True,
            )
    with pd.HDFStore(path, mode="r") as store:
        for entity in US_SCHEMA.entities:
            assert store.get_storer(entity).is_table, entity
            restored = read_frame_table(store, entity)
            original = result.table(entity)
            assert list(restored.columns) == list(original.columns)
            for column in original.columns:
                if pd.api.types.is_numeric_dtype(original[column].dtype):
                    assert restored[column].dtype == original[column].dtype, column
                    np.testing.assert_array_equal(
                        restored[column].to_numpy(),
                        original[column].to_numpy(),
                    )
    household = result.table("household")
    acs_households = household[support_channel_column("household")].eq("acs")
    assert household.index.size == 2 * assembled.table("household").index.size
    for column, carrier in {
        "H_TENURE": ~acs_households,
        "TYPEHUGQ": acs_households,
    }.items():
        assert household[column].dtype == np.dtype(np.float64), column
        assert household.loc[carrier, column].notna().all(), column
        assert household.loc[~carrier, column].isna().all(), column


def test_padded_boolean_columns_write_as_table_format_after_the_cps_fill(
    tmp_path,
) -> None:
    """A padded ACS-only boolean is table-writable once its CPS producer runs.

    ``is_female`` is ACS-native (acs_inputs.map_acs_native_inputs) and a
    declared output of the CPS-scope ``derive_us_cps_carried_inputs``; the
    source chain drops the union's all-null column before the CPS kernel and
    merges the kernel's booleans back here.
    """

    from microcosm.build.us_runtime.multispine_pool import (
        _merge_source_operator_outputs,
    )
    from microcosm.frame import put_frame_table, read_frame_table

    assembled = assemble_spines(
        {"asec": _asec_frame(), "acs": _acs_with_acs_only_columns()},
        household_mass_shares={"asec": 0.75, "acs": 0.25},
    )
    person = assembled.table("person")
    cps_rows = person[support_channel_column("person")].eq("asec").to_numpy()
    projection = assembled.select(cps_rows)
    operated = _with_columns(
        projection,
        "person",
        {"is_female": np.asarray([False, True], dtype=bool)},
    )
    merged, rows = _merge_source_operator_outputs(
        assembled,
        operated,
        {"person": frozenset({"is_female"})},
        operator_name="derive_us_cps_carried_inputs",
    )

    assert rows == {"person": 2}
    merged_person = merged.table("person")
    assert isinstance(merged_person["is_female"].dtype, pd.BooleanDtype)
    assert merged_person["is_female"].tolist() == [False, True, True]
    path = tmp_path / "merged.h5"
    with pd.HDFStore(path, mode="w") as store:
        put_frame_table(
            store,
            "person",
            merged_person,
            preferred_format="table",
            data_columns=True,
        )
    with pd.HDFStore(path, mode="r") as store:
        assert store.get_storer("person").is_table
        restored = read_frame_table(store, "person")
    assert restored["is_female"].dtype == np.dtype(bool)
    assert restored["is_female"].tolist() == [False, True, True]


# Person references through the PUF clone (review r2 P1). Assembly pads a
# parent pointer one spine lacks as float64; the clone renumbers person ids,
# and once an assembled id reaches 10**15 the multiplier is 10**16, so the
# PUF-arm ids pass 2**53. Odd targets matter: float64 rounds 10**16 + 1 to
# 10**16 but holds 10**16 + 2.
_PARENT_COLUMNS = ("parent_1_id", "parent_2_id")


def _large_id_spines() -> tuple[Frame, Frame]:
    # ACS ids [1, 2] collide with the ASEC anchor's, so assembly offsets them
    # to [10**15, 10**15 + 1]; every source id stays within the clone-safe bound.
    asec = _with_structural_ids(
        _asec_frame(), [1, PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID]
    )
    acs = _with_structural_ids(_asec_frame(), [1, 2])
    return asec, acs


def _references(table: pd.DataFrame, column: str) -> list[object]:
    # Python objects compare int against float exactly, so 1e16 != 10**16 + 1
    # here, whereas a NumPy comparison would cast the int side and agree.
    return [None if pd.isna(value) else value for value in table[column].tolist()]


def _assert_cloned_references_name_their_parents(
    assembled: Frame, cloned: Frame
) -> None:
    """Each cloned pointer names the cloned parent in its own arm, exactly."""

    source = assembled.table("person")
    person = cloned.table("person")
    rows = len(source)
    assert len(person) == 2 * rows
    source_ids = [int(value) for value in source["person_id"]]
    cloned_ids = [int(value) for value in person["person_id"]]
    for column in _PARENT_COLUMNS:
        expected: list[int] = [0] * (2 * rows)
        for row, parent in enumerate(source[column].tolist()):
            if pd.isna(parent) or int(parent) == 0:
                continue  # unknown parent: the 0 sentinel on both arms
            target = source_ids.index(int(parent))
            expected[row] = cloned_ids[target]
            expected[rows + row] = cloned_ids[rows + target]
        assert _references(person, column) == expected, column
        assert person[column].dtype == np.dtype(np.int64), column


@pytest.mark.parametrize("storage", ["int64", "Int64"])
def test_padded_parent_references_name_the_intended_person_after_the_clone(
    storage: str,
) -> None:
    """Review r2's orientation: ASEC carries the pointers, ACS lacks them.

    Assembly pads the ACS rows with NaN; a nullable carrier adds a missing
    value on a carrying row. The clone writes both as the 0 sentinel and
    keeps every PUF-arm pointer exact past 2**53.
    """

    asec, acs = _large_id_spines()
    if storage == "int64":
        parent_2 = np.asarray([PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID, 0], dtype=np.int64)
    else:
        parent_2 = pd.array([PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID, None], dtype="Int64")
    asec = _with_columns(
        asec,
        "person",
        {
            "parent_1_id": pd.array([0, 1], dtype=storage),
            "parent_2_id": parent_2,
        },
    )
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    source = assembled.table("person")
    for column in _PARENT_COLUMNS:
        assert source[column].dtype == np.dtype(np.float64), column
    assert source["parent_1_id"].isna().tolist() == [False, False, True, True]
    assert source["parent_2_id"].isna().tolist() == [
        False,
        storage == "Int64",
        True,
        True,
    ]

    cloned = clone_us_frame_for_puf_support(assembled)

    _assert_cloned_references_name_their_parents(assembled, cloned)
    person = cloned.table("person")
    # The PUF arm's parent of ASEC person 10**15 - 1 is person 1 shifted by
    # the 10**16 multiplier, an odd id float64 cannot hold.
    assert int(person["person_id"].iloc[4]) == 10**16 + 1
    assert person["parent_1_id"].tolist() == [0, 1, 0, 0, 0, 10**16 + 1, 0, 0]
    assert person["parent_2_id"].tolist() == [
        PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
        0,
        0,
        0,
        10**16 + PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
        0,
        0,
        0,
    ]


def test_parent_references_filled_by_the_cps_merge_stay_exact_through_the_clone() -> (
    None
):
    """The production orientation: ACS maps every pointer to 0 and ASEC lacks
    them, so assembly pads the ASEC rows; the pre-clone eligibility operator
    then fills them through the source-operator merge as float64 values."""

    from microcosm.build.us_runtime.multispine_pool import (
        _merge_source_operator_outputs,
    )

    asec, acs = _large_id_spines()
    acs = _with_columns(
        acs,
        "person",
        {column: np.zeros(2, dtype=np.int64) for column in _PARENT_COLUMNS},
    )
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    person = assembled.table("person")
    cps_rows = person[support_channel_column("person")].eq("asec").to_numpy()
    operated = _with_columns(
        assembled.select(cps_rows),
        "person",
        {
            "parent_1_id": np.asarray([0, 1], dtype=np.int64),
            "parent_2_id": np.asarray(
                [PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID, 0], dtype=np.int64
            ),
        },
    )
    merged, rows = _merge_source_operator_outputs(
        assembled,
        operated,
        {"person": frozenset(_PARENT_COLUMNS)},
        operator_name="with_us_eligibility_inputs",
    )
    assert rows == {"person": 2}
    assert merged.table("person")["parent_1_id"].notna().all()

    cloned = clone_us_frame_for_puf_support(merged)

    _assert_cloned_references_name_their_parents(merged, cloned)
    assert cloned.table("person")["parent_1_id"].tolist() == [
        0,
        1,
        0,
        0,
        0,
        10**16 + 1,
        0,
        0,
    ]
