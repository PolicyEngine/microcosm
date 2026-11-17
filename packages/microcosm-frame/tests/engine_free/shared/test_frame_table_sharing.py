"""Frames that share unchanged tables: ``_adopt``, ``replace_tables``, ``with_columns``.

The default :class:`Frame` constructor deep-copies every table it is given.
``Frame._adopt`` instead stores the tables it is given without copying them
when pandas copy-on-write is on, so a frame built by ``replace_tables`` or
``with_columns`` shares the memory of every unchanged column with its source
frame. Copy-on-write keeps a write made through pandas on one frame from
reaching the other. When copy-on-write is off (the pandas 2.x default), a
shallow copy shares data even for ordinary pandas writes, so ``_adopt``
deep-copies instead. These tests replace the copy-on-write check to take that
path, because pandas 3 always has copy-on-write on.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    EntitySchema,
    Frame,
    LinkSpec,
    MassChange,
    WeightKind,
    Weights,
)
from microcosm.frame import bundle as bundle_module

_SCHEMA = EntitySchema(
    group_entities=("household", "firm"),
    links=(LinkSpec(name="jobs", left_entity="person", right_entity="firm"),),
)

_SOURCE_INCOME = [50_000.0, 0.0, 30_000.0, 20_000.0]
_SOURCE_RENT = [900.0, 1_200.0]


def _source_tables() -> dict[str, pd.DataFrame]:
    """Four persons in two households and two firms, plus a jobs link table."""

    return {
        "person": pd.DataFrame(
            {
                "person_id": np.array([0, 1, 2, 3], dtype="int64"),
                "person_household_id": np.array([1, 1, 2, 2], dtype="int64"),
                "person_firm_id": np.array([10, 10, 20, 20], dtype="int64"),
                "age": np.array([40, 8, 30, 28], dtype="int64"),
                "income": np.array(_SOURCE_INCOME),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.array([1, 2], dtype="int64"),
                "rent": np.array(_SOURCE_RENT),
                "state": ["CA", "NY"],
            }
        ),
        "firm": pd.DataFrame(
            {
                "firm_id": np.array([10, 20], dtype="int64"),
                "payroll": np.array([1.0e6, 2.0e6]),
            }
        ),
        "jobs": pd.DataFrame(
            {
                "person_id": np.array([0, 2, 3], dtype="int64"),
                "firm_id": np.array([10, 20, 20], dtype="int64"),
            }
        ),
    }


def _source_weights() -> dict[str, Weights]:
    return {
        "household": Weights(values=np.array([100.0, 200.0]), kind=WeightKind.DESIGN)
    }


def _source_strata() -> pd.Series:
    return pd.Series(["cps", "cps", "puf", "puf"], index=pd.RangeIndex(4))


@pytest.fixture
def source_frame() -> Frame:
    """A frame with a link table, strata, metadata and one mass-log entry."""

    frame = Frame(
        _source_tables(),
        _SCHEMA,
        _source_weights(),
        _source_strata(),
        metadata={"stage": "source"},
    )
    return frame.with_weights(
        "household",
        Weights(values=np.array([150.0, 300.0]), kind=WeightKind.CALIBRATED),
        mass=MassChange(factor=1.5, reason="test calibration"),
    )


@pytest.fixture
def copy_on_write_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Frame._adopt`` see pandas copy-on-write as off."""

    monkeypatch.setattr(bundle_module, "_pandas_copy_on_write", lambda: False)


def _tables_of(frame: Frame) -> dict[str, pd.DataFrame]:
    """Every entity table and link table the frame stores, by name."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    return tables


def _numeric_columns_sharing_memory(
    left: dict[str, pd.DataFrame], right: dict[str, pd.DataFrame]
) -> dict[tuple[str, str], bool]:
    """For each numeric column both table sets carry, whether its data is shared."""

    sharing: dict[tuple[str, str], bool] = {}
    for table_name, left_table in left.items():
        right_table = right.get(table_name)
        if right_table is None:
            continue
        for column in left_table.select_dtypes("number").columns:
            if column not in right_table.columns:
                continue
            sharing[(table_name, column)] = np.shares_memory(
                left_table[column].to_numpy(), right_table[column].to_numpy()
            )
    return sharing


class TestDefaultConstructor:
    def test_constructor_deep_copies_every_table(self) -> None:
        tables = _source_tables()
        frame = Frame(tables, _SCHEMA, _source_weights(), _source_strata())

        for name, table in _tables_of(frame).items():
            assert table is not tables[name]
        sharing = _numeric_columns_sharing_memory(_tables_of(frame), tables)
        assert sharing, "the comparison found no numeric columns"
        assert not any(sharing.values()), sharing


class TestAdopt:
    def test_adopt_stores_the_given_tables_without_copying(self) -> None:
        tables = _source_tables()
        frame = Frame._adopt(tables, _SCHEMA, _source_weights(), _source_strata())

        for name, table in _tables_of(frame).items():
            assert table is tables[name]
        assert frame.links == ("jobs",)

    def test_adopt_deep_copies_when_copy_on_write_is_off(
        self, copy_on_write_off: None
    ) -> None:
        tables = _source_tables()
        strata = _source_strata()
        frame = Frame._adopt(tables, _SCHEMA, _source_weights(), strata)

        for name, table in _tables_of(frame).items():
            assert table is not tables[name]
        sharing = _numeric_columns_sharing_memory(_tables_of(frame), tables)
        assert sharing, "the comparison found no numeric columns"
        assert not any(sharing.values()), sharing
        assert not np.shares_memory(frame.strata.to_numpy(), strata.to_numpy())

    def test_adopt_names_the_strata_without_renaming_the_callers_series(
        self,
    ) -> None:
        strata = _source_strata().rename("provenance")
        frame = Frame._adopt(_source_tables(), _SCHEMA, _source_weights(), strata)

        assert frame.strata.name == "stratum"
        assert strata.name == "provenance"
        assert frame.strata.tolist() == ["cps", "cps", "puf", "puf"]

    def test_adopt_carries_the_mass_log_and_metadata(self, source_frame) -> None:
        frame = Frame._adopt(
            _source_tables(),
            _SCHEMA,
            _source_weights(),
            mass_log=source_frame.mass_log,
            metadata={"stage": "adopted", "inputs": ["cps", "puf"]},
        )

        assert frame.mass_log == source_frame.mass_log
        assert dict(frame.metadata) == {"stage": "adopted", "inputs": ("cps", "puf")}


def _drop_table(name: str) -> Callable[[dict[str, Any]], None]:
    def change(arguments: dict[str, Any]) -> None:
        del arguments["tables"][name]

    return change


def _set_table(name: str, make: Callable[[pd.DataFrame], Any]) -> Callable:
    def change(arguments: dict[str, Any]) -> None:
        tables = arguments["tables"]
        tables[name] = make(tables.get(name))

    return change


def _set_cell(table: str, row: int, column: str, value: Any) -> Callable:
    def change(arguments: dict[str, Any]) -> None:
        arguments["tables"][table].loc[row, column] = value

    return change


def _set_argument(name: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(arguments: dict[str, Any]) -> None:
        arguments[name] = value() if callable(value) else value

    return change


#: Each case breaks one constructor invariant; the ids name the invariant.
_INVALID_ARGUMENTS = {
    "missing_entity_table": _drop_table("firm"),
    "unknown_entity_table": _set_table(
        "spaceship", lambda _table: pd.DataFrame({"spaceship_id": [1]})
    ),
    "missing_person_id_column": _set_table(
        "person", lambda table: table.drop(columns=["person_id"])
    ),
    "duplicate_person_id": _set_cell("person", 1, "person_id", 0),
    "missing_membership_column": _set_table(
        "person", lambda table: table.drop(columns=["person_firm_id"])
    ),
    "unsorted_group_ids": _set_table(
        "household", lambda table: table.iloc[::-1].reset_index(drop=True)
    ),
    "orphaned_group_id": _set_table(
        "firm",
        lambda _table: pd.DataFrame(
            {"firm_id": [10, 20, 30], "payroll": [1.0, 2.0, 3.0]}
        ),
    ),
    "global_column_collision": _set_table(
        "firm", lambda table: table.assign(rent=[1.0, 2.0])
    ),
    "link_table_missing_id_column": _set_table(
        "jobs", lambda table: table.drop(columns=["firm_id"])
    ),
    "link_table_dangling_id": _set_cell("jobs", 0, "firm_id", 99),
    "reserved_weight_column": _set_table(
        "person", lambda table: table.assign(person_weight=1.0)
    ),
    "no_weights": _set_argument("weights", {}),
    "weights_length_mismatch": _set_argument(
        "weights",
        {
            "household": Weights(
                values=np.array([1.0, 2.0, 3.0]), kind=WeightKind.DESIGN
            )
        },
    ),
    "weights_not_typed": _set_argument("weights", {"household": np.array([1.0, 2.0])}),
    "weights_for_unknown_entity": _set_argument(
        "weights",
        {
            **_source_weights(),
            "galaxy": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN),
        },
    ),
    "strata_misaligned": _set_argument(
        "strata", lambda: pd.Series(["cps"] * 4, index=[5, 6, 7, 8])
    ),
    "strata_not_a_series": _set_argument("strata", ["cps"] * 4),
    "strata_missing_label": _set_argument(
        "strata", lambda: pd.Series(["cps", None, "puf", "puf"])
    ),
    "metadata_not_a_mapping": _set_argument("metadata", ["stage"]),
    "metadata_unsupported_value": _set_argument("metadata", {"stage": object()}),
}


def _constructor_arguments(change: Callable[[dict[str, Any]], None]) -> dict:
    arguments: dict[str, Any] = {
        "tables": _source_tables(),
        "schema": _SCHEMA,
        "weights": _source_weights(),
        "strata": _source_strata(),
        "metadata": None,
    }
    change(arguments)
    return arguments


@pytest.mark.parametrize("copy_on_write", [True, False], ids=["cow_on", "cow_off"])
@pytest.mark.parametrize(
    "change", list(_INVALID_ARGUMENTS.values()), ids=list(_INVALID_ARGUMENTS)
)
def test_adopt_raises_the_constructors_validation_errors(
    monkeypatch: pytest.MonkeyPatch, change, copy_on_write: bool
) -> None:
    monkeypatch.setattr(bundle_module, "_pandas_copy_on_write", lambda: copy_on_write)
    with pytest.raises((TypeError, ValueError)) as constructor_error:
        Frame(**_constructor_arguments(change))
    with pytest.raises(type(constructor_error.value)) as adopt_error:
        Frame._adopt(**_constructor_arguments(change))

    assert str(adopt_error.value) == str(constructor_error.value)


class TestReplaceTables:
    def test_unchanged_tables_share_memory_with_the_source(self, source_frame) -> None:
        # A shallow copy with one column replaced keeps sharing the others.
        household = source_frame.table("household").copy(deep=False)
        household["rent"] = [1_000.0, 1_300.0]
        replaced = source_frame.replace_tables(household=household)

        assert replaced.table("household")["rent"].tolist() == [1_000.0, 1_300.0]
        assert source_frame.table("household")["rent"].tolist() == _SOURCE_RENT
        sharing = _numeric_columns_sharing_memory(
            _tables_of(replaced), _tables_of(source_frame)
        )
        unchanged = {
            key: shared for key, shared in sharing.items() if key[0] != "household"
        }
        assert unchanged and all(unchanged.values()), unchanged
        assert not sharing[("household", "rent")]
        assert sharing[("household", "household_id")]

    def test_every_table_is_a_distinct_object(self, source_frame) -> None:
        household = source_frame.table("household").copy()
        replaced = source_frame.replace_tables(household=household)

        source_tables = _tables_of(source_frame)
        for name, table in _tables_of(replaced).items():
            assert table is not source_tables[name]
        assert replaced.table("household") is not household
        assert np.shares_memory(
            replaced.table("household")["rent"].to_numpy(),
            household["rent"].to_numpy(),
        )

    def test_weights_strata_mass_log_and_metadata_are_carried(
        self, source_frame
    ) -> None:
        replaced = source_frame.replace_tables(
            firm=source_frame.table("firm").assign(payroll=[5.0, 6.0])
        )

        assert type(replaced) is Frame
        assert replaced.weights_for("household") is source_frame.weights_for(
            "household"
        )
        assert replaced.strata.equals(source_frame.strata)
        assert replaced.mass_log == source_frame.mass_log
        assert len(replaced.mass_log) == 1
        assert replaced.metadata == source_frame.metadata
        assert replaced.links == ("jobs",)

    def test_a_link_table_can_be_replaced(self, source_frame) -> None:
        jobs = pd.DataFrame({"person_id": [1, 3], "firm_id": [10, 20]})
        replaced = source_frame.replace_tables(jobs=jobs)

        assert replaced.link("jobs")["person_id"].tolist() == [1, 3]
        assert source_frame.link("jobs")["person_id"].tolist() == [0, 2, 3]

    def test_a_later_write_to_the_passed_table_does_not_reach_the_frame(
        self, source_frame
    ) -> None:
        household = source_frame.table("household").copy()
        replaced = source_frame.replace_tables(household=household)

        household.loc[0, "rent"] = -1.0

        assert replaced.table("household")["rent"].tolist() == _SOURCE_RENT

    def test_replace_tables_deep_copies_when_copy_on_write_is_off(
        self, source_frame, copy_on_write_off: None
    ) -> None:
        household = source_frame.table("household").copy()
        replaced = source_frame.replace_tables(household=household)

        against_source = _numeric_columns_sharing_memory(
            _tables_of(replaced), _tables_of(source_frame)
        )
        against_passed = _numeric_columns_sharing_memory(
            _tables_of(replaced), {"household": household}
        )
        assert against_source and not any(against_source.values()), against_source
        assert against_passed and not any(against_passed.values()), against_passed

    def test_the_result_is_fully_validated(self, source_frame) -> None:
        reversed_households = (
            source_frame.table("household").iloc[::-1].reset_index(drop=True)
        )
        with pytest.raises(ValueError, match="sorted ascending"):
            source_frame.replace_tables(household=reversed_households)
        with pytest.raises(
            ValueError, match=r"Unknown entity table\(s\): \['spaceship'\]"
        ):
            source_frame.replace_tables(spaceship=pd.DataFrame({"spaceship_id": [1]}))

    def test_a_table_that_is_not_a_dataframe_is_refused(self, source_frame) -> None:
        with pytest.raises(TypeError, match="'household'.*DataFrame.*dict"):
            source_frame.replace_tables(household={"household_id": [1, 2]})


class TestWithColumns:
    def test_columns_are_replaced_and_added(self, source_frame) -> None:
        income = np.array([1.0, 2.0, 3.0, 4.0])
        updated = source_frame.with_columns("person", income=income, bonus=np.zeros(4))

        assert updated.person["income"].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert updated.person["bonus"].tolist() == [0.0] * 4
        assert source_frame.person["income"].tolist() == _SOURCE_INCOME
        assert "bonus" not in source_frame.person.columns
        assert not np.shares_memory(updated.person["income"].to_numpy(), income)

    def test_unchanged_columns_share_memory_with_the_source(self, source_frame) -> None:
        updated = source_frame.with_columns("person", income=np.ones(4))

        sharing = _numeric_columns_sharing_memory(
            _tables_of(updated), _tables_of(source_frame)
        )
        unchanged = {
            key: shared
            for key, shared in sharing.items()
            if key != ("person", "income")
        }
        assert unchanged and all(unchanged.values()), unchanged
        assert not sharing[("person", "income")]
        assert updated.person is not source_frame.person

    def test_weights_strata_mass_log_and_metadata_are_carried(
        self, source_frame
    ) -> None:
        updated = source_frame.with_columns("household", rent=[1.0, 2.0])

        assert updated.weights_for("household") is source_frame.weights_for("household")
        assert updated.strata.equals(source_frame.strata)
        assert updated.mass_log == source_frame.mass_log
        assert updated.metadata == source_frame.metadata
        assert updated.links == ("jobs",)

    def test_a_column_may_be_named_entity(self, source_frame) -> None:
        updated = source_frame.with_columns("person", entity=np.arange(4))

        assert updated.person["entity"].tolist() == [0, 1, 2, 3]

    def test_an_unknown_entity_is_named(self, source_frame) -> None:
        with pytest.raises(ValueError, match="Unknown entity 'spaceship'"):
            source_frame.with_columns("spaceship", fuel=[1.0])
        with pytest.raises(ValueError, match="Unknown entity 'jobs'"):
            source_frame.with_columns("jobs", hours=[1.0, 2.0, 3.0])

    def test_the_result_is_fully_validated(self, source_frame) -> None:
        with pytest.raises(ValueError, match="globally unique"):
            source_frame.with_columns("firm", rent=[1.0, 2.0])
        with pytest.raises(ValueError, match="sorted ascending"):
            source_frame.with_columns("household", household_id=[2, 1])
        with pytest.raises(ValueError, match="reserved weight-column"):
            source_frame.with_columns("person", person_weight=np.ones(4))

    def test_with_columns_deep_copies_when_copy_on_write_is_off(
        self, source_frame, copy_on_write_off: None
    ) -> None:
        updated = source_frame.with_columns("person", bonus=np.zeros(4))

        sharing = _numeric_columns_sharing_memory(
            _tables_of(updated), _tables_of(source_frame)
        )
        assert sharing and not any(sharing.values()), sharing


def _frames_built_from(source_frame: Frame) -> dict[str, Frame]:
    return {
        "with_columns": source_frame.with_columns("person", bonus=np.zeros(4)),
        "replace_tables": source_frame.replace_tables(
            firm=source_frame.table("firm").assign(payroll=[5.0, 6.0])
        ),
    }


@pytest.mark.parametrize("method", ["with_columns", "replace_tables"])
def test_a_pandas_write_to_the_new_frame_leaves_the_source_unchanged(
    source_frame, method: str
) -> None:
    derived = _frames_built_from(source_frame)[method]
    assert np.shares_memory(
        derived.person["income"].to_numpy(), source_frame.person["income"].to_numpy()
    )

    derived.person.loc[0, "income"] = -1.0
    derived.table("household").loc[1, "rent"] = -1.0
    derived.link("jobs").loc[0, "firm_id"] = 20

    assert derived.person["income"].tolist()[0] == -1.0
    assert source_frame.person["income"].tolist() == _SOURCE_INCOME
    assert source_frame.table("household")["rent"].tolist() == _SOURCE_RENT
    assert source_frame.link("jobs")["firm_id"].tolist() == [10, 20, 20]
    source_frame.revalidate()


@pytest.mark.parametrize("method", ["with_columns", "replace_tables"])
def test_a_pandas_write_to_the_source_leaves_the_new_frame_unchanged(
    source_frame, method: str
) -> None:
    derived = _frames_built_from(source_frame)[method]

    source_frame.person.loc[0, "income"] = -1.0
    source_frame.table("household").loc[1, "rent"] = -1.0

    assert derived.person["income"].tolist() == _SOURCE_INCOME
    assert derived.table("household")["rent"].tolist() == _SOURCE_RENT


class TestCopyOnWriteCheck:
    def test_pandas_3_has_copy_on_write_without_reading_the_option(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse_option_reads(name: str) -> object:
            raise AssertionError(f"pandas 3 must not read option {name!r}")

        monkeypatch.setattr(bundle_module.pd, "__version__", "3.0.3")
        monkeypatch.setattr(bundle_module.pd, "get_option", refuse_option_reads)

        assert bundle_module._pandas_copy_on_write() is True

    @pytest.mark.parametrize(
        ("option_value", "expected"),
        [(True, True), (False, False), ("warn", False)],
        ids=["enabled", "disabled", "warn_mode"],
    )
    def test_pandas_2_reads_the_copy_on_write_option(
        self, monkeypatch: pytest.MonkeyPatch, option_value: object, expected: bool
    ) -> None:
        options = {"mode.copy_on_write": option_value}
        monkeypatch.setattr(bundle_module.pd, "__version__", "2.3.3")
        monkeypatch.setattr(bundle_module.pd, "get_option", options.__getitem__)

        assert bundle_module._pandas_copy_on_write() is expected
