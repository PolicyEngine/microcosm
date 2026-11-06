"""The shared readers of copy lineage, on invented tables (microcosm#931)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_lineage import (
    CopyBranch,
    CopyLineage,
    collapse_replicas,
    copy_paths,
    record_sources,
    replica_keys,
)
from microcosm.frame import EntitySchema

SCHEMA = EntitySchema(group_entities=("unit", "household"))
LINEAGE = CopyLineage(
    entity="household",
    branches=(
        CopyBranch(
            "support", index="support_index", marker="channel", copy="b", original="a"
        ),
        CopyBranch("clone", marker="is_clone"),
        CopyBranch("place", index="place_index"),
    ),
)


def _tables(*, with_source: bool = True) -> dict[str, pd.DataFrame]:
    """Two survey households; the first has a support copy and a placed clone."""

    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "channel": ["a", "a", "b", "a"],
            "support_index": [0, 0, 1, 0],
            "is_clone": [False, False, False, False],
            "place_index": [0, 0, 0, 1],
        }
    )
    if with_source:
        household["source_household_id"] = [1, 2, 1, 1]
    person = pd.DataFrame(
        {
            "person_id": [10, 11, 20, 30, 31, 40, 41],
            "person_unit_id": [100, 101, 200, 300, 301, 400, 401],
            "person_household_id": [1, 1, 2, 3, 3, 4, 4],
        }
    )
    unit = pd.DataFrame({"unit_id": [100, 101, 200, 300, 301, 400, 401]})
    return {"person": person, "unit": unit, "household": household}


def test_every_entity_reads_its_households_true_source() -> None:
    tables = _tables()
    assert record_sources(tables, SCHEMA, "household", LINEAGE).tolist() == [
        1,
        2,
        1,
        1,
    ]
    assert record_sources(tables, SCHEMA, "person", LINEAGE).tolist() == [
        1,
        1,
        2,
        1,
        1,
        1,
        1,
    ]
    assert record_sources(tables, SCHEMA, "unit", LINEAGE).tolist() == [
        1,
        1,
        2,
        1,
        1,
        1,
        1,
    ]


def test_a_frame_without_the_source_is_refused_unless_it_holds_no_copies() -> None:
    tables = _tables(with_source=False)
    with pytest.raises(ValueError, match="copy lineage"):
        record_sources(tables, SCHEMA, "person", LINEAGE, before_copies=True)
    before = {
        **tables,
        "household": tables["household"][["household_id"]].iloc[:2],
        "person": tables["person"].iloc[:3],
        "unit": tables["unit"].iloc[:3],
    }
    with pytest.raises(ValueError, match="lacks 'source_household_id'"):
        record_sources(before, SCHEMA, "person", LINEAGE)
    # Before the first copying step each household is its own source.
    assert record_sources(
        before, SCHEMA, "person", LINEAGE, before_copies=True
    ).tolist() == [1, 1, 2]


@pytest.mark.parametrize("defect", ["null_source", "orphan_person", "split_unit"])
def test_a_record_without_one_true_source_is_refused(defect: str) -> None:
    tables = _tables()
    entity = "person"
    if defect == "null_source":
        tables["household"]["source_household_id"] = pd.array(
            [1, None, 1, 1], dtype="Int64"
        )
    elif defect == "orphan_person":
        tables["person"].loc[0, "person_household_id"] = 9
    else:
        tables["person"].loc[1, "person_unit_id"] = 200
        entity = "unit"
    with pytest.raises(ValueError):
        record_sources(tables, SCHEMA, entity, LINEAGE)


@pytest.mark.parametrize(
    "branches",
    [
        (CopyBranch("a", index="x"), CopyBranch("a", index="y")),
        (CopyBranch("a", index="x"), CopyBranch("b", marker="x")),
        (CopyBranch("a", index="source_household_id"),),
    ],
)
def test_a_lineage_needs_distinct_branches_and_columns(branches) -> None:
    with pytest.raises(ValueError):
        CopyLineage(entity="household", branches=branches)


@pytest.mark.parametrize(
    "kwargs", [{}, {"index": "x", "copy": 1, "original": 1}, {"name": ""}]
)
def test_a_branch_declares_a_column_and_tells_copies_apart(kwargs) -> None:
    with pytest.raises(ValueError):
        CopyBranch(**{"name": "a", **kwargs})


def test_record_sources_returns_the_tables_row_order() -> None:
    tables = _tables()
    order = [3, 0, 2, 1]
    shuffled = {**tables, "household": tables["household"].iloc[order]}
    assert np.array_equal(
        record_sources(shuffled, SCHEMA, "household", LINEAGE),
        np.asarray([1, 1, 1, 2]),
    )


def test_copy_paths_read_each_step_in_declared_order() -> None:
    tables = _tables()
    tables["household"]["is_clone"] = [False, False, True, False]
    sources, paths = copy_paths(tables["household"], LINEAGE)
    assert sources.tolist() == [1, 2, 1, 1]
    assert paths == [(), (), (("support", 1), ("clone", 1)), (("place", 1),)]


def test_a_step_that_has_not_run_adds_nothing_unless_required() -> None:
    household = _tables()["household"].drop(columns=["place_index"])
    _, paths = copy_paths(household, LINEAGE)
    assert paths == [(), (), (("support", 1),), ()]
    with pytest.raises(ValueError, match="lacks the 'place'"):
        copy_paths(household, LINEAGE, require=("place",))
    with pytest.raises(ValueError, match="No copy branch"):
        copy_paths(household, LINEAGE, require=("unknown",))


@pytest.mark.parametrize(
    "defect",
    [
        "partial_step",
        "disagreeing_marker",
        "negative_ordinal",
        "float_ordinal",
        "unknown_marker",
        "integer_flag",
        "null_marker",
        "no_source",
    ],
)
def test_copy_paths_refuse_lineage_they_cannot_read(defect: str) -> None:
    household = _tables()["household"]
    if defect == "partial_step":
        household = household.drop(columns=["support_index"])
    elif defect == "disagreeing_marker":
        household.loc[0, "channel"] = "b"
    elif defect == "negative_ordinal":
        household.loc[3, "place_index"] = -1
    elif defect == "float_ordinal":
        household["place_index"] = household["place_index"].astype(float)
    elif defect == "unknown_marker":
        household.loc[0, "channel"] = "c"
    elif defect == "integer_flag":
        household["is_clone"] = household["is_clone"].astype(int)
    elif defect == "null_marker":
        household["channel"] = pd.array(["a", None, "b", "a"], dtype="string")
    else:
        household = household.drop(columns=["source_household_id"])
    with pytest.raises(ValueError):
        copy_paths(household, LINEAGE)


def _placed_tables() -> dict[str, pd.DataFrame]:
    """The tables placed twice: clone 1 of every row at id + 1000."""

    tables = _tables()
    placed = {}
    for entity, table in tables.items():
        clone = table.copy()
        for column in clone.columns:
            if column.endswith("_id") and column != "source_household_id":
                clone[column] = clone[column] + 1000
        if entity == "household":
            clone["place_index"] = clone["place_index"] + 1
        placed[entity] = pd.concat([table, clone], ignore_index=True)
    return placed


def test_replica_keys_are_shared_by_placed_clones_alone() -> None:
    tables = _placed_tables()
    households = replica_keys(tables, SCHEMA, "household", LINEAGE, merge=("place",))
    # Households 1 and 4 are one record placed twice, and so are their
    # clones; the support copy (3) stays apart from its source (1).
    first = dict(zip(tables["household"]["household_id"], households, strict=True))
    assert first[1] == first[4] == first[1001] == first[1004]
    assert len({first[1], first[2], first[3]}) == 3
    assert first[2] == first[1002] and first[3] == first[1003]
    people = replica_keys(tables, SCHEMA, "person", LINEAGE, merge=("place",))
    by_person = dict(zip(tables["person"]["person_id"], people, strict=True))
    # Members keep their rank: a household's first member is the clone's.
    assert by_person[10] == by_person[1010] == by_person[40]
    assert by_person[11] == by_person[41] and by_person[10] != by_person[11]
    units = replica_keys(tables, SCHEMA, "unit", LINEAGE, merge=("place",))
    by_unit = dict(zip(tables["unit"]["unit_id"], units, strict=True))
    assert by_unit[100] == by_unit[400] == by_unit[1100]
    with pytest.raises(ValueError, match="No copy branch"):
        replica_keys(tables, SCHEMA, "unit", LINEAGE, merge=("elsewhere",))


def test_collapse_keeps_each_records_first_row_with_its_summed_weight() -> None:
    keys = np.asarray(["a", "b", "a", "c", "b", "a"], dtype=object)
    rows = np.asarray([True, True, True, False, True, True])
    weights = np.asarray([0.1, 0.2, 0.2, 9.0, 0.3, 0.7])
    kept, summed = collapse_replicas(keys, rows, weights)
    assert kept.tolist() == [True, True, False, False, False, False]
    # Exact sums, whatever the order: 0.1 + 0.2 + 0.7 rounds once.
    assert summed[0] == 1.0 and summed[1] == 0.5
    assert summed[3] == 9.0
    order = np.asarray([5, 4, 3, 2, 1, 0])
    reversed_kept, reversed_summed = collapse_replicas(
        keys[order], rows[order], weights[order]
    )
    assert reversed_summed[reversed_kept].tolist() == [1.0, 0.5]


def test_a_selection_without_replicas_keeps_its_rows_and_weights() -> None:
    keys = np.asarray(["a", "b", "a"], dtype=object)
    rows = np.asarray([True, True, False])
    weights = np.asarray([0.1, 0.2, 0.3])
    kept, summed = collapse_replicas(keys, rows, weights)
    assert kept.tolist() == rows.tolist()
    assert summed.tobytes() == weights.tobytes()
    with pytest.raises(ValueError, match="align"):
        collapse_replicas(keys, rows[:2], weights)


def test_replicas_fold_only_where_they_agree_on_the_inputs() -> None:
    """Each distinct variant keeps its first record and its replicas' weight
    (microcosm#931)."""

    keys = np.asarray(["a", "a", "a", "a", "b", "b"], dtype=object)
    rows = np.ones(6, dtype=bool)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.5])
    inputs = pd.DataFrame(
        {
            "income": [3.0, 0.0, 3.0, np.nan, 7.0, 7.0],
            "band": pd.Categorical(["x", "x", "x", "x", "y", "y"]),
        }
    )
    kept, summed = collapse_replicas(keys, rows, weights, inputs=inputs)
    # a holds three variants (3.0 twice, 0.0, missing); b agrees and folds.
    assert kept.tolist() == [True, True, False, True, True, False]
    assert summed[kept].tolist() == [
        pytest.approx(0.4),
        0.2,
        0.4,
        1.0,
    ]
    # The record's weight is kept whole across its variants.
    assert np.isclose(summed[kept][:3].sum(), 1.0)
    # A missing value agrees only with a missing value.
    both_missing = inputs.assign(income=[np.nan, np.nan, 3.0, 3.0, 7.0, 7.0])
    kept, _ = collapse_replicas(keys, rows, weights, inputs=both_missing)
    assert kept.tolist() == [True, False, True, False, True, False]


def test_agreeing_inputs_fold_as_the_keys_alone_do() -> None:
    keys = np.asarray(["a", "b", "a", "c", "b", "a"], dtype=object)
    rows = np.asarray([True, True, True, False, True, True])
    weights = np.asarray([0.1, 0.2, 0.2, 9.0, 0.3, 0.7])
    alone = collapse_replicas(keys, rows, weights)
    agreeing = collapse_replicas(
        keys, rows, weights, inputs=pd.DataFrame({"value": np.ones(6)})
    )
    assert alone[0].tolist() == agreeing[0].tolist()
    assert alone[1].tobytes() == agreeing[1].tobytes()


def test_inputs_must_align_and_name_each_column_once() -> None:
    keys = np.asarray(["a", "a"], dtype=object)
    rows = np.ones(2, dtype=bool)
    weights = np.ones(2)
    with pytest.raises(ValueError, match="one row per record"):
        collapse_replicas(keys, rows, weights, inputs=pd.DataFrame({"x": [1.0]}))
    duplicated = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], columns=["x", "x"])
    with pytest.raises(ValueError, match="each column once"):
        collapse_replicas(keys, rows, weights, inputs=duplicated)
