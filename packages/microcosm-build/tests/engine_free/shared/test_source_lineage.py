"""The shared readers of copy lineage, on invented tables (microcosm#931)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_lineage import CopyBranch, CopyLineage, record_sources
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
