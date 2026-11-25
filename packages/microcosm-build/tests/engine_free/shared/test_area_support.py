"""Area support: the table, the floors and one gate for any country (#1729).

Every weight, code and count below is made up for the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build import gates
from microcosm.build.area_support import (
    area_support_floor_table,
    area_support_gate,
    area_support_table,
    effective_sample_size_by_group,
)
from microcosm.calibrate.solve import effective_sample_size
from test_support.microcosm_build import area_support_gate_reference as before

LEVEL = "district"


def _legacy_ess_by_group(weights: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    """``ess_by_group`` of ``tools/us_acs_local_cd_surface.py`` at ac7ab99ae.

    A verbatim copy, kept to hold the shared function to the bytes the ACS
    local release's weight-origin summaries recorded before the move.
    """

    frame = pd.DataFrame(
        {"code": np.asarray(codes).astype(str), "w": np.asarray(weights, float)}
    )
    frame["w2"] = frame["w"] ** 2
    sums = frame.groupby("code", sort=True)[["w", "w2"]].sum()
    return {
        str(code): float(row.w**2 / row.w2) if row.w2 > 0 else 0.0
        for code, row in sums.iterrows()
    }


_WEIGHTED_GROUPS = st.lists(
    st.tuples(
        st.floats(0.0, 1e9, allow_nan=False),
        st.sampled_from(["0601", "0602", "3601", "2401", "7"]),
    ),
    min_size=1,
    max_size=60,
)


# ---------------------------------------------------------------------------
# Measuring support
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(records=_WEIGHTED_GROUPS)
def test_grouped_ess_is_the_moved_function_bit_for_bit(records) -> None:
    weights = np.asarray([weight for weight, _code in records])
    codes = np.asarray([code for _weight, code in records])

    moved = effective_sample_size_by_group(weights, codes)
    legacy = _legacy_ess_by_group(weights, codes)

    assert list(moved) == list(legacy)
    assert [value.hex() for value in moved.values()] == [
        value.hex() for value in legacy.values()
    ]
    # And it is the Kish ESS of each group's records.
    for code, ess in moved.items():
        assert ess == pytest.approx(effective_sample_size(weights[codes == code]))


def test_grouped_ess_refuses_weights_the_solver_refuses() -> None:
    codes = np.asarray(["a", "a", "b"])
    for bad in ([1.0, -1.0, 1.0], [1.0, np.nan, 1.0], [1.0, np.inf, 1.0]):
        with pytest.raises(ValueError, match="finite and non-negative"):
            effective_sample_size_by_group(np.asarray(bad), codes)
    with pytest.raises(ValueError, match="do not align"):
        effective_sample_size_by_group(np.ones(2), codes)
    assert effective_sample_size_by_group(np.zeros(3), codes) == {"a": 0.0, "b": 0.0}


def test_the_support_table_counts_positive_rows_and_distinct_sources() -> None:
    weights = np.asarray([2.0, 2.0, 0.0, 5.0, 1.0, 0.0])
    areas = np.asarray(["A", "A", "A", "B", "B", "C"])
    # Two copies of one source household in A; the zero-weight row is a third.
    sources = np.asarray(["s1", "s1", "s9", "s2", "s3", "s4"])
    starting = np.ones(6)

    table = area_support_table(
        weights,
        areas,
        geography_level=LEVEL,
        source_ids=sources,
        starting_weights=starting,
    )

    assert table.to_dict("list") == {
        "geography_level": [LEVEL, LEVEL, LEVEL],
        "area_code": ["A", "B", "C"],
        "nonzero_households": [2, 2, 0],
        "nonzero_source_households": [1, 2, 0],
        "effective_sample_size": [2.0, 36.0 / 26.0, 0.0],
        "starting_effective_sample_size": [3.0, 2.0, 1.0],
    }
    # Without sources or starting weights the table carries neither column.
    assert list(area_support_table(weights, areas, geography_level=LEVEL)) == [
        "geography_level",
        "area_code",
        "nonzero_households",
        "effective_sample_size",
    ]


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _table(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "area_code",
            "nonzero_households",
            "nonzero_source_households",
            "effective_sample_size",
            "starting_effective_sample_size",
        ],
    ).assign(geography_level=LEVEL)


_AREAS = st.lists(
    st.tuples(
        st.integers(0, 120),
        st.integers(0, 120),
        st.floats(0.0, 120.0, allow_nan=False),
        st.floats(0.0, 400.0, allow_nan=False),
    ),
    min_size=1,
    max_size=12,
)
_EXCLUDED = st.lists(st.integers(0, 14), max_size=4, unique=True)


def test_there_is_one_gate() -> None:
    assert area_support_gate is gates.area_support_gate


@settings(max_examples=300, deadline=None)
@given(
    areas=_AREAS,
    floors=st.tuples(st.integers(1, 80), st.floats(0.5, 80.0), st.integers(1, 80)),
    excluded=_EXCLUDED,
)
def test_the_absolute_floors_give_the_gate_of_before(areas, floors, excluded) -> None:
    """With the three absolute floors the gate is the gate it always was.

    The UK local build calls it this way. The result, the per-area flag and
    the wording are held to a verbatim copy of the code before the floors
    became optional, reviewed exclusions included.
    """

    table = _table([(f"a{index:02d}", *area) for index, area in enumerate(areas)])
    declared = dict(
        minimum_rows=floors[0],
        minimum_effective_sample_size=floors[1],
        minimum_distinct_sources=floors[2],
    )
    arguments = dict(
        area_roster={LEVEL: list(table["area_code"])},
        geography_levels=(LEVEL,),
        # Some name areas of the table, some an area it does not have.
        reviewed_exclusions={
            f"{LEVEL}/a{index:02d}": {"reviewer": "test"} for index in excluded
        },
        **declared,
    )

    assert area_support_gate(table, **arguments) == before.area_support_gate(
        table, **arguments
    )
    assert gates.area_support_below_floor(table, **declared).equals(
        before.area_support_below_floor(table, **declared)
    )
    for row in table.itertuples(index=False):
        measured = (
            row.nonzero_households,
            row.effective_sample_size,
            row.nonzero_source_households,
        )
        assert gates.area_support_shortfalls(*measured, **declared) == (
            before.area_support_shortfalls(*measured, **declared)
        )


@pytest.mark.parametrize(
    ("floor", "value", "match"),
    [
        ("minimum_rows", 0, "minimum_rows must be a positive integer"),
        ("minimum_distinct_sources", True, "minimum_distinct_sources must be"),
        ("minimum_effective_sample_size", 0.0, "must be positive and finite"),
    ],
)
def test_an_invalid_absolute_floor_keeps_its_refusal(floor, value, match) -> None:
    table = _table([("a", 60, 60, 60.0, 60.0)])
    arguments = dict(
        area_roster={LEVEL: ["a"]},
        geography_levels=(LEVEL,),
        minimum_rows=50,
        minimum_effective_sample_size=50.0,
        minimum_distinct_sources=50,
    )
    arguments[floor] = value
    for gate in (area_support_gate, before.area_support_gate):
        with pytest.raises(ValueError, match=match):
            gate(table, **arguments)


def test_the_flag_needs_the_columns_of_its_floors() -> None:
    table = _table([("a", 60, 60, 60.0, 60.0)]).drop(columns="nonzero_households")
    with pytest.raises(ValueError, match="nonzero_households"):
        gates.area_support_below_floor(
            table,
            minimum_rows=50,
            minimum_effective_sample_size=50.0,
            minimum_distinct_sources=50,
        )
    # A floor that is not declared reads no column.
    flags = gates.area_support_below_floor(
        table,
        minimum_rows=None,
        minimum_effective_sample_size=70.0,
        minimum_distinct_sources=None,
        minimum_effective_sample_size_ratio=0.5,
    )
    assert flags.tolist() == [True]


def test_the_relative_floor_fails_an_area_below_a_share_of_its_starting_ess() -> None:
    table = _table(
        [
            ("held", 0, 0, 30.0, 100.0),  # 0.30 of its start
            ("lost", 0, 0, 24.0, 100.0),  # 0.24: below a quarter
            ("thin", 0, 0, 9.0, 12.0),  # 0.75, but below the absolute floor
            ("edge", 0, 0, 25.0, 100.0),  # exactly a quarter: not below
            ("empty", 0, 0, 0.0, 0.0),  # no starting ESS: never below the ratio
        ]
    )
    result = area_support_gate(
        table,
        area_roster={LEVEL: ["held", "lost", "thin", "edge", "empty"]},
        geography_levels=(LEVEL,),
        minimum_effective_sample_size=15.0,
        minimum_effective_sample_size_ratio=0.25,
    )

    assert not result.passed
    assert result.failures == (
        f"{LEVEL}/empty: ESS 0 < 15",
        f"{LEVEL}/lost: ESS 24 < 0.25 x starting ESS 100",
        f"{LEVEL}/thin: ESS 9 < 15",
    )
    assert result.details["floors"] == {
        "minimum_rows": None,
        "minimum_effective_sample_size": 15.0,
        "minimum_distinct_sources": None,
        "minimum_effective_sample_size_ratio": 0.25,
    }
    level = result.details["by_geography_level"][LEVEL]
    assert level["areas_checked"] == 5
    assert level["areas_below_floor"] == 3
    assert level["areas_below_effective_sample_size_ratio"] == 1
    assert level["minimum_effective_sample_size_ratio"] == pytest.approx(0.24)
    assert level["minimum_effective_sample_size"] == 0.0


def test_a_gate_needs_only_the_columns_of_its_floors() -> None:
    table = pd.DataFrame(
        {
            "geography_level": LEVEL,
            "area_code": ["a", "b"],
            "effective_sample_size": [40.0, 8.0],
            "starting_effective_sample_size": [80.0, 80.0],
        }
    )
    roster = dict(area_roster={LEVEL: ["a", "b"]}, geography_levels=(LEVEL,))

    relative = area_support_gate(
        table, **roster, minimum_effective_sample_size_ratio=0.25
    )
    assert relative.failures == (f"{LEVEL}/b: ESS 8 < 0.25 x starting ESS 80",)
    assert "minimum_rows" not in relative.details["by_geography_level"][LEVEL]

    # No floor declared: the gate checks the roster and reports the minima.
    undeclared = area_support_gate(table, **roster)
    assert undeclared.passed
    assert undeclared.details["by_geography_level"][LEVEL] == {
        "areas_checked": 2,
        "areas_below_floor": 0,
        "minimum_effective_sample_size": 8.0,
    }

    with pytest.raises(ValueError, match="nonzero_households"):
        area_support_gate(table, **roster, minimum_rows=50)
    with pytest.raises(ValueError, match="starting_effective_sample_size"):
        area_support_gate(
            table.drop(columns="starting_effective_sample_size"),
            **roster,
            minimum_effective_sample_size_ratio=0.25,
        )


def test_the_roster_is_load_bearing_under_any_floors() -> None:
    table = pd.DataFrame(
        {
            "geography_level": LEVEL,
            "area_code": ["a"],
            "effective_sample_size": [40.0],
            "starting_effective_sample_size": [80.0],
        }
    )
    with pytest.raises(ValueError, match="exactly cover the declared roster"):
        area_support_gate(
            table,
            area_roster={LEVEL: ["a", "b"]},
            geography_levels=(LEVEL,),
            minimum_effective_sample_size_ratio=0.25,
        )
    with pytest.raises(ValueError, match="duplicate area rows"):
        area_support_gate(
            pd.concat([table, table]),
            area_roster={LEVEL: ["a"]},
            geography_levels=(LEVEL,),
            minimum_effective_sample_size_ratio=0.25,
        )


@pytest.mark.parametrize(
    ("floor", "value", "match"),
    [
        ("minimum_effective_sample_size_ratio", 0.0, r"in \(0, 1\]"),
        ("minimum_effective_sample_size_ratio", 1.5, r"in \(0, 1\]"),
        ("minimum_effective_sample_size_ratio", float("nan"), r"in \(0, 1\]"),
        ("minimum_effective_sample_size", 0.0, "positive and finite"),
        ("minimum_rows", 0, "positive integer"),
        ("minimum_distinct_sources", 2.5, "positive integer"),
    ],
)
def test_an_invalid_floor_is_refused(floor, value, match) -> None:
    table = _table([("a", 60, 60, 60.0, 60.0)])
    with pytest.raises(ValueError, match=match):
        area_support_floor_table(table, **{floor: value})
    with pytest.raises(ValueError, match=match):
        # A relative floor too, so the general rule checks the floors itself.
        area_support_gate(
            table,
            area_roster={LEVEL: ["a"]},
            geography_levels=(LEVEL,),
            **{"minimum_effective_sample_size_ratio": 0.5, floor: value},
        )


_FLOORS = st.fixed_dictionaries(
    {
        "minimum_rows": st.none() | st.integers(1, 80),
        "minimum_effective_sample_size": st.none() | st.floats(0.5, 80.0),
        "minimum_distinct_sources": st.none() | st.integers(1, 80),
        "minimum_effective_sample_size_ratio": st.none() | st.floats(0.01, 1.0),
    }
)


@settings(max_examples=200, deadline=None)
@given(areas=_AREAS, floors=_FLOORS)
def test_the_floor_table_and_the_gate_give_one_verdict(areas, floors) -> None:
    table = _table([(f"a{index:02d}", *area) for index, area in enumerate(areas)])

    checked = area_support_floor_table(table, **floors)
    result = area_support_gate(
        table,
        area_roster={LEVEL: list(table["area_code"])},
        geography_levels=(LEVEL,),
        **floors,
    )

    below = sorted(checked.loc[checked["below_any_floor"], "area_code"])
    assert sorted(line.split(":")[0] for line in result.failures) == [
        f"{LEVEL}/{code}" for code in below
    ]
    assert result.passed == (not below)
    assert result.details["by_geography_level"][LEVEL]["areas_below_floor"] == (
        len(below)
    )
    # One flag column for each declared floor, and no other.
    assert {column for column in checked.columns if column.startswith("below_")} == {
        f"below_{floor}" for floor, value in floors.items() if value is not None
    } | {"below_any_floor"}
    # The ratio is the ESS over the starting ESS, undefined at a zero start.
    starting = table["starting_effective_sample_size"].to_numpy()
    assert np.isnan(
        checked["effective_sample_size_ratio"].to_numpy()[starting == 0]
    ).all()
