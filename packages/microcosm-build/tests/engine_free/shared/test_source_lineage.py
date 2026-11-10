"""The shared readers of copy lineage, on invented tables (microcosm#931)."""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_lineage import (
    CopyBranch,
    CopyLineage,
    ReplicaFold,
    collapse_replicas,
    copy_paths,
    fold_replicas,
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


# --- The fold's representatives, variety and key groups (microcosm#931) ----


def _fold_by_hand(
    keys: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    inputs: pd.DataFrame | None,
    order: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """The fold's contract, record by record in plain Python.

    A variant is a key with one set of input values, a missing value
    matching only a missing value. Its representative is its selected
    record with the least order, then the earliest row (the earliest row
    alone without an order), and carries the variant's weights summed
    exactly. A selected record varies when its key holds two variants or
    more, and its key group numbers its key by first appearance.
    """

    def cell(value: object) -> object:
        return None if isinstance(value, float) and math.isnan(value) else value

    def variant(row: int) -> tuple[object, ...]:
        values = () if inputs is None else tuple(map(cell, inputs.iloc[row]))
        return (keys[row], *values)

    def rank(row: int) -> tuple[int, int]:
        return (0 if order is None else int(order[row]), row)

    selected = np.flatnonzero(rows).tolist()
    members: dict[tuple[object, ...], list[int]] = {}
    codes: dict[object, int] = {}
    for row in selected:
        members.setdefault(variant(row), []).append(row)
        codes.setdefault(keys[row], len(codes))
    variants = {key: 0 for key in codes}
    for key, *_ in members:
        variants[key] += 1
    kept = np.zeros(len(keys), dtype=bool)
    summed = np.asarray(weights, dtype=np.float64).copy()
    representative = np.full(len(keys), -1, dtype=np.int64)
    for group in members.values():
        first = min(group, key=rank)
        kept[first] = True
        summed[first] = math.fsum(float(weights[row]) for row in group)
        representative[group] = first
    varied = np.zeros(len(keys), dtype=bool)
    key_group = np.full(len(keys), -1, dtype=np.int64)
    for row in selected:
        varied[row] = variants[keys[row]] > 1
        key_group[row] = codes[keys[row]]
    return {
        "kept": kept,
        "weights": summed,
        "representative": representative,
        "varied": varied,
        "key_group": key_group,
    }


_FOLD_FIELDS = ("kept", "weights", "representative", "varied", "key_group")


def _assert_fold(fold: ReplicaFold, expected: dict[str, np.ndarray]) -> None:
    """Each named field of ``fold`` equals ``expected``'s, dtype and bytes."""

    for name, values in expected.items():
        found = getattr(fold, name)
        assert found.dtype == values.dtype, name
        assert found.tobytes() == values.tobytes(), name


def test_each_record_trains_on_its_least_order_clone_in_any_row_order() -> None:
    """Two records placed three times, clone k at id + 1000 k: whatever the
    rows' order, each record trains on clone 0 with all three clones'
    weight, and every clone points at it."""

    keys = np.asarray(["r1", "r2"] * 3, dtype=object)
    ids = np.asarray([11, 12, 1011, 1012, 2011, 2012], dtype=np.int64)
    weights = np.asarray([0.1, 0.25, 0.2, 0.25, 0.7, 0.5])
    agreeing = pd.DataFrame({"value": np.zeros(6)})
    clone_zero = {"r1": 11, "r2": 12}
    for permutation in permutations(range(6)):
        shuffled = np.asarray(permutation)
        for inputs in (None, agreeing):
            fold = fold_replicas(
                keys[shuffled],
                np.ones(6, dtype=bool),
                weights[shuffled],
                inputs=inputs,
                order=ids[shuffled],
            )
            assert sorted(ids[shuffled][fold.kept].tolist()) == [11, 12]
            # 0.1 + 0.2 + 0.7 rounds once to 1.0, as does 0.25 + 0.25 + 0.5.
            assert fold.weights[fold.kept].tolist() == [1.0, 1.0]
            assert ids[shuffled][fold.representative].tolist() == [
                clone_zero[key] for key in keys[shuffled]
            ]
            assert not fold.varied.any()


def test_each_variant_trains_on_its_own_least_order_replica() -> None:
    keys = np.asarray(["a", "a", "a", "a", "b", "b"], dtype=object)
    rows = np.ones(6, dtype=bool)
    weights = np.asarray([0.5, 0.25, 0.125, 1.0, 2.0, 4.0])
    inputs = pd.DataFrame({"band": ["x", "y", "x", "y", "z", "z"]})
    order = np.asarray([30, 10, 20, 40, 50, 5], dtype=np.int64)
    fold = fold_replicas(keys, rows, weights, inputs=inputs, order=order)
    # a's variants x (rows 0, 2) and y (rows 1, 3) each keep their own
    # least order; b's one variant keeps its later row, which orders first.
    assert fold.kept.tolist() == [False, True, True, False, False, True]
    assert fold.representative.tolist() == [2, 1, 2, 1, 5, 5]
    assert fold.weights.tolist() == [0.5, 1.25, 0.625, 1.0, 2.0, 6.0]
    assert fold.varied.tolist() == [True, True, True, True, False, False]
    assert fold.key_group.tolist() == [0, 0, 0, 0, 1, 1]
    # Replicas tied on the order go to the earlier row.
    tied = fold_replicas(
        keys, rows, weights, inputs=inputs, order=np.zeros(6, dtype=np.int64)
    )
    assert tied.representative.tolist() == [0, 1, 0, 1, 4, 4]


def test_a_fold_without_replicas_is_the_identity() -> None:
    keys = np.asarray(["a", "b", "a"], dtype=object)
    rows = np.asarray([True, True, False])
    weights = np.asarray([0.1, 0.2, 0.3])
    for inputs in (None, pd.DataFrame({"value": [1.0, 2.0, 3.0]})):
        for order in (None, np.asarray([9, 8, 7], dtype=np.int64)):
            fold = fold_replicas(keys, rows, weights, inputs=inputs, order=order)
            _assert_fold(
                fold,
                {
                    "kept": rows,
                    "weights": weights,
                    "representative": np.asarray([0, 1, -1], dtype=np.int64),
                    "varied": np.zeros(3, dtype=bool),
                    "key_group": np.asarray([0, 1, -1], dtype=np.int64),
                },
            )


def test_the_fold_matches_hand_built_maps() -> None:
    keys = np.asarray(["a", "a", "a", "a", "b", "b", "c", "d"], dtype=object)
    rows = np.asarray([True] * 7 + [False])
    weights = np.asarray([0.5, 0.25, 0.125, 1.0, 2.0, 4.0, 8.0, 9.0])
    inputs = pd.DataFrame({"income": [3.0, 0.0, 3.0, np.nan, 7.0, 7.0, 1.0, 2.0]})
    order = np.asarray([40, 30, 20, 10, 60, 50, 70, 80], dtype=np.int64)
    fold = fold_replicas(keys, rows, weights, inputs=inputs, order=order)
    # a holds three variants (3.0 on rows 0 and 2, 0.0, missing); b agrees
    # and folds onto its lower order; c is alone; d is not selected.
    _assert_fold(
        fold,
        {
            "kept": np.asarray([False, True, True, True, False, True, True, False]),
            "weights": np.asarray([0.5, 0.25, 0.625, 1.0, 2.0, 6.0, 8.0, 9.0]),
            "representative": np.asarray([2, 1, 2, 3, 5, 5, 6, -1], dtype=np.int64),
            "varied": np.asarray([True] * 4 + [False] * 4),
            "key_group": np.asarray([0, 0, 0, 0, 1, 1, 2, -1], dtype=np.int64),
        },
    )
    _assert_fold(fold, _fold_by_hand(keys, rows, weights, inputs, order))
    kept, summed = collapse_replicas(keys, rows, weights, inputs=inputs, order=order)
    assert kept.tobytes() == fold.kept.tobytes()
    assert summed.tobytes() == fold.weights.tobytes()


_TODAYS_FIXTURE_INPUTS = pd.DataFrame(
    {
        "income": [3.0, 0.0, 3.0, np.nan, 7.0, 7.0],
        "band": pd.Categorical(["x", "x", "x", "x", "y", "y"]),
    }
)


@pytest.mark.parametrize(
    ("keys", "rows", "weights", "inputs", "kept", "summed", "representative"),
    [
        (
            ["a", "b", "a", "c", "b", "a"],
            [True, True, True, False, True, True],
            [0.1, 0.2, 0.2, 9.0, 0.3, 0.7],
            None,
            [True, True, False, False, False, False],
            [1.0, 0.5, 0.2, 9.0, 0.3, 0.7],
            [0, 1, 0, -1, 1, 0],
        ),
        (
            ["a", "b", "a", "c", "b", "a"],
            [True, True, True, False, True, True],
            [0.1, 0.2, 0.2, 9.0, 0.3, 0.7],
            pd.DataFrame({"value": np.ones(6)}),
            [True, True, False, False, False, False],
            [1.0, 0.5, 0.2, 9.0, 0.3, 0.7],
            [0, 1, 0, -1, 1, 0],
        ),
        (
            ["a", "a", "a", "a", "b", "b"],
            [True] * 6,
            [0.5, 0.25, 0.25, 1.0, 2.0, 2.0],
            _TODAYS_FIXTURE_INPUTS,
            [True, True, False, True, True, False],
            [0.75, 0.25, 0.25, 1.0, 4.0, 2.0],
            [0, 1, 0, 3, 4, 4],
        ),
    ],
    ids=["keys_alone", "agreeing_inputs", "variants"],
)
def test_without_an_order_each_variant_keeps_its_first_row(
    keys, rows, weights, inputs, kept, summed, representative
) -> None:
    """``order=None`` is the fold as it was, on the keys and inputs of the
    fixtures above: each variant keeps its first row, as the row positions
    taken as the order give too."""

    keys = np.asarray(keys, dtype=object)
    rows, weights = np.asarray(rows), np.asarray(weights)
    fold = fold_replicas(keys, rows, weights, inputs=inputs)
    assert fold.kept.tolist() == kept
    assert fold.weights.tolist() == summed
    assert fold.representative.tolist() == representative
    by_position = fold_replicas(
        keys, rows, weights, inputs=inputs, order=np.arange(len(keys))
    )
    _assert_fold(fold, {name: getattr(by_position, name) for name in _FOLD_FIELDS})
    collapsed = collapse_replicas(keys, rows, weights, inputs=inputs)
    assert collapsed[0].tobytes() == fold.kept.tobytes()
    assert collapsed[1].tobytes() == fold.weights.tobytes()


def test_clones_that_all_differ_fold_nothing_yet_vary() -> None:
    """Two clones of each record differ on a continuous input: nothing
    folds, as before, and every record varies. A fold that decided its
    early return on the key and the inputs together would report no
    variety exactly here."""

    keys = np.asarray(["a", "b", "a", "b"], dtype=object)
    rows = np.ones(4, dtype=bool)
    weights = np.asarray([0.5, 0.25, 0.5, 0.25])
    inputs = pd.DataFrame({"income": [1.5, 2.5, 3.5, 4.5]})
    order = np.asarray([1, 2, 1001, 1002], dtype=np.int64)
    fold = fold_replicas(keys, rows, weights, inputs=inputs, order=order)
    _assert_fold(
        fold,
        {
            "kept": rows,
            "weights": weights,
            "representative": np.arange(4, dtype=np.int64),
            "varied": np.ones(4, dtype=bool),
            "key_group": np.asarray([0, 1, 0, 1], dtype=np.int64),
        },
    )
    kept, summed = collapse_replicas(keys, rows, weights, inputs=inputs)
    assert kept.tolist() == rows.tolist()
    assert summed.tobytes() == weights.tobytes()


def test_the_fold_keeps_its_contract_on_random_frames() -> None:
    rng = np.random.default_rng(17)
    letters = np.asarray(["a", "b", "c", "d"], dtype=object)
    for trial in range(240):
        n = int(rng.integers(0, 24))
        keys = rng.choice(letters, n)
        rows = rng.random(n) < 0.8
        weights = rng.uniform(0.1, 3.0, n)
        inputs = None
        if trial % 3:
            inputs = pd.DataFrame(
                {
                    "value": rng.choice([0.0, 1.0, np.nan], n),
                    "band": rng.choice(np.asarray(["x", "y"], dtype=object), n),
                }
            )
        # A narrow range of orders makes ties, which go to the earlier row.
        order = None if trial % 2 else rng.integers(0, 4, n)
        fold = fold_replicas(keys, rows, weights, inputs=inputs, order=order)
        _assert_fold(fold, _fold_by_hand(keys, rows, weights, inputs, order))


@pytest.mark.parametrize(
    "order",
    [
        np.asarray([7357.5, 7358.5, 7359.5]),
        np.asarray([7357, 7358], dtype=np.int64),
        np.asarray([[7357, 7358, 7359]], dtype=np.int64),
        np.asarray([True, False, True]),
        np.asarray(["7357", "7358", "7359"], dtype=object),
        np.asarray([7357, 7358, 7359], dtype="m8[ns]"),
    ],
    ids=["float", "short", "two_dimensional", "boolean", "text", "timedelta"],
)
def test_the_order_is_one_integer_per_record_refused_by_count(order) -> None:
    for keys in (["a", "b", "c"], ["a", "a", "b"]):
        with pytest.raises(ValueError, match="one integer per record") as refusal:
            fold_replicas(
                np.asarray(keys, dtype=object),
                np.ones(3, dtype=bool),
                np.ones(3),
                order=order,
            )
        # The refusal counts values and records; it prints no order value.
        assert "735" not in str(refusal.value)


def test_a_selected_record_without_a_key_is_refused() -> None:
    # A null key names no record, so its record would fold with others.
    keys = np.asarray(["a", None, "a", None], dtype=object)
    with pytest.raises(ValueError, match="must not be null"):
        fold_replicas(keys, np.ones(4, dtype=bool), np.ones(4))
    # Off the selection a key is never read.
    fold = fold_replicas(keys, np.asarray([True, False, True, False]), np.ones(4))
    assert fold.kept.tolist() == [True, False, False, False]
    assert fold.key_group.tolist() == [0, -1, 0, -1]
