"""The country-neutral columns a support carries for a conditioned draw, and
the fallback level each household draws at."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.atomic_geography import (
    assign_atomic,
    decode_atomic_support,
    encode_atomic_support,
    validate_assignment_spec,
)
from microcosm.build.atomic_support_conditioning import (
    FallbackLevel,
    apportion_weights,
    choose_fallback_levels,
    extend_atomic_support,
    modal_area_codes,
    zero_mass_cells,
)

DOMAIN = ("1", "2", "3")


def _u(*values: str) -> np.ndarray:
    return np.asarray(values, dtype="U")


def _support() -> bytes:
    """Four areas in two regions, invented codes and counts."""
    code = {"kind": "code", "source": "invented", "vintage": "v1", "relation": "exact"}
    return encode_atomic_support(
        {
            "version": 1,
            "system": "toy",
            "level": "area",
            "code_system": "toy",
            "vintage": "v1",
            "columns": {
                "area": code,
                "region": code,
                "households": {
                    "kind": "weight",
                    "source": "invented",
                    "basis": "count",
                },
            },
        },
        {
            "area": _u("a1", "a2", "b1", "b2"),
            "region": _u("north", "north", "south", "south"),
            "households": np.asarray([10, 30, 20, 0], dtype=np.int64),
        },
    )


def test_modal_codes_take_the_plurality_then_the_parent_then_the_order():
    areas = _u("a1", "a2", "a3", "b1")
    parents = _u("A", "A", "A", "B")
    records = {
        "a1": ["2", "2", "1"],  # own plurality
        "a2": ["2", "3"],  # tie broken by the parent's records (A: 3 twos)
        # a3 has no record: it takes its parent's plurality
        "b1": ["1", "2"],  # tie the parent cannot break: domain order
    }
    record_areas = _u(*[area for area, codes in records.items() for _ in codes])
    record_codes = _u(*[code for codes in records.values() for code in codes])
    modal = modal_area_codes(areas, parents, record_areas, record_codes, domain=DOMAIN)
    assert modal.codes.tolist() == ["2", "2", "2", "1"]
    assert modal.rule.tolist() == [
        "area",
        "area_tie_parent",
        "parent",
        "area_tie_order",
    ]
    assert modal.spanning.tolist() == [True, True, False, True]
    assert modal.receipt() == {
        "areas": 4,
        "records": 7,
        "areas_by_rule": {
            "area": 1,
            "area_tie_order": 1,
            "area_tie_parent": 1,
            "parent": 1,
        },
        "areas_whose_records_span_codes": 3,
    }


@pytest.mark.parametrize(
    ("record_areas", "record_codes", "message"),
    [
        (("a1", "zz"), ("1", "1"), "outside the support"),
        (("a1", "a2"), ("1", "9"), "outside the domain"),
        (("a2",), ("1",), "parent have no record"),
    ],
)
def test_modal_codes_refuse_unknown_records_and_unsupported_areas(
    record_areas, record_codes, message
):
    with pytest.raises(ValueError, match=message):
        modal_area_codes(
            _u("a1", "a2"),
            _u("A", "B"),
            _u(*record_areas),
            _u(*record_codes),
            domain=DOMAIN,
        )


def test_apportioned_weights_are_exact_and_round_half_to_even():
    counts = np.asarray([3, 1, 0], dtype=np.int64)
    numerators = {
        "x": np.asarray([1, 1, 0], dtype=np.int64),
        "y": np.asarray([1, 1, 0], dtype=np.int64),
    }
    denominators = np.asarray([2, 2, 0], dtype=np.int64)
    weights, receipt = apportion_weights(counts, numerators, denominators, scale=1)
    # 3 × 1/2 = 1.5 rounds to 2 (even) in both columns; 1 × 1/2 = 0.5 to 0.
    assert weights["x"].tolist() == [2, 0, 0]
    assert weights["y"].tolist() == [2, 0, 0]
    assert receipt == {
        "rounding": "round_half_even",
        "scale": 1,
        "areas_with_rounding": 2,
        "largest_area_deviation": 1,
        "total_deviation": 0,
        "column_totals": {"x": 2, "y": 2},
    }
    exact, receipt = apportion_weights(counts, numerators, denominators, scale=10)
    assert exact["x"].tolist() == [15, 5, 0]
    assert receipt["areas_with_rounding"] == 0


@pytest.mark.parametrize(
    ("numerators", "denominators", "message"),
    [
        ({"x": [1], "y": [2]}, [2], "sum to a positive denominator"),
        ({"x": [-1], "y": [3]}, [2], "nonnegative"),
        ({"x": [1], "y": [1]}, [2, 2], "one denominator per area"),
    ],
)
def test_apportioned_weights_refuse_shares_that_do_not_partition(
    numerators, denominators, message
):
    with pytest.raises(ValueError, match=message):
        apportion_weights(
            np.asarray([4], dtype=np.int64),
            {k: np.asarray(v, dtype=np.int64) for k, v in numerators.items()},
            np.asarray(denominators, dtype=np.int64),
            scale=1,
        )


def test_apportioned_weights_refuse_totals_beyond_exact_sampling():
    with pytest.raises(ValueError, match="exact sampling range"):
        apportion_weights(
            np.asarray([2**40], dtype=np.int64),
            {"x": np.asarray([1], dtype=np.int64)},
            np.asarray([1], dtype=np.int64),
            scale=2**20,
        )


def test_extended_support_keeps_its_columns_and_decodes():
    base = _support()
    extended = extend_atomic_support(
        base,
        arrays={
            "decile": _u("1", "2", "1", "3"),
            "households_x": np.asarray([5, 0, 20, 0], dtype=np.int64),
        },
        columns={
            "decile": {
                "kind": "code",
                "source": "invented",
                "vintage": "v2019",
                "relation": "inferred_modal",
            },
            "households_x": {"kind": "weight", "source": "invented", "basis": "x"},
        },
    )
    before, after = decode_atomic_support(base), decode_atomic_support(extended)
    assert set(after.arrays) == {*before.arrays, "decile", "households_x"}
    for name, array in before.arrays.items():
        assert np.array_equal(after.arrays[name], array)
    with pytest.raises(ValueError, match="already present"):
        extend_atomic_support(
            base,
            arrays={"households": np.zeros(4, dtype=np.int64)},
            columns={"households": {"kind": "weight", "source": "s", "basis": "b"}},
        )


def test_zero_mass_cells_list_empty_and_absent_cells_at_each_level():
    extended = decode_atomic_support(
        extend_atomic_support(
            _support(),
            arrays={
                "decile": _u("1", "2", "1", "3"),
                "households_x": np.asarray([5, 0, 20, 0], dtype=np.int64),
            },
            columns={
                "decile": {
                    "kind": "code",
                    "source": "invented",
                    "vintage": "v",
                    "relation": "exact",
                },
                "households_x": {"kind": "weight", "source": "s", "basis": "b"},
            },
        )
    )
    table = zero_mass_cells(
        extended,
        domains={"region": ("north", "south"), "decile": DOMAIN},
        levels=(("region", "decile"), ("region",)),
        weights=("households", "households_x"),
    )
    assert table == [
        {
            "constraints": ["region", "decile"],
            "weight": "households",
            # b2 (south, 3) has no households; (north, 3) and (south, 2) are absent.
            "cells": [["north", "3"], ["south", "2"], ["south", "3"]],
        },
        {
            "constraints": ["region", "decile"],
            "weight": "households_x",
            "cells": [["north", "2"], ["north", "3"], ["south", "2"], ["south", "3"]],
        },
        {"constraints": ["region"], "weight": "households", "cells": []},
        {"constraints": ["region"], "weight": "households_x", "cells": []},
    ]


def test_the_shared_operator_draws_on_the_added_columns():
    """An added code column is an exact-match constraint and an added weight
    column is a weight_by choice: the operator needs nothing else."""
    import pandas as pd

    extended = extend_atomic_support(
        _support(),
        arrays={
            "decile": _u("1", "2", "1", "2"),
            "households_x": np.asarray([0, 7, 5, 0], dtype=np.int64),
        },
        columns={
            "decile": {
                "kind": "code",
                "source": "s",
                "vintage": "v",
                "relation": "exact",
            },
            "households_x": {"kind": "weight", "source": "s", "basis": "b"},
        },
    )
    spec = validate_assignment_spec(
        {
            "version": 1,
            "identity": ["key"],
            "stream": ["sha256-u53-v1", "toy", 0, 1],
            "outputs": {
                "area": "area_code",
                "system": "area_system",
                "basis": "area_basis",
            },
            "systems": [
                {
                    "id": "toy",
                    "level": "area",
                    "code_system": "toy",
                    "vintage": "v1",
                    "source": "toy_support",
                    "selector": {"region": ["north", "south"]},
                    "constraints": [
                        {"input": "region", "support": "region", "required": True},
                        {"input": "decile", "support": "decile", "required": False},
                    ],
                    "observed_area": None,
                    "stages": [
                        {
                            "level": "area",
                            "weight_by": {
                                "input": "band",
                                "map": {"x": "households_x"},
                                "default": "households",
                            },
                        }
                    ],
                    "layers": [],
                }
            ],
        }
    )
    households = pd.DataFrame(
        {
            "key": ["h1", "h2", "h3"],
            "region": ["north", "south", "north"],
            "decile": ["2", "1", pd.NA],
            "band": ["x", "x", pd.NA],
        }
    )
    assigned = assign_atomic(households, spec, {"toy": decode_atomic_support(extended)})
    # h1 can only be a2 (north, decile 2, band x); h2 only b1; h3 any northern
    # area with households.
    assert assigned["area_code"].tolist()[:2] == ["a2", "b1"]
    assert assigned["area_code"].tolist()[2] in {"a1", "a2"}


# The fallback-level chooser. Every support and household below is invented.

#: A ladder in a conditioned draw's fallback order: the band's weight before
#: households, then the decile with the region before the region alone.
LADDER = (
    FallbackLevel("L0", ("region", "draw_decile"), weight_by=True),
    FallbackLevel("L1", ("region", "draw_decile"), weight_by=False),
    FallbackLevel("L2", ("region",), weight_by=True),
    FallbackLevel("L3", ("region",), weight_by=False),
)
#: Each household input to the support column it must match.
CONSTRAINTS = {"region": "region", "draw_decile": "decile"}
RULE = {"input": "band", "map": {"x": "households_x", "y": "households_y"}}
NO_LEVEL = (
    "Atomic support conditioning: a household has no fallback level whose "
    "inputs are present and whose cell has sampling mass"
)


def _conditioned(region, decile, **weights):
    """Invented areas, each with a region, a decile and the given weights."""
    code = {"kind": "code", "source": "invented", "vintage": "v1", "relation": "exact"}
    return decode_atomic_support(
        encode_atomic_support(
            {
                "version": 1,
                "system": "toy",
                "level": "area",
                "code_system": "toy",
                "vintage": "v1",
                "columns": {
                    "area": code,
                    "region": code,
                    "decile": code,
                    **{
                        name: {"kind": "weight", "source": "invented", "basis": name}
                        for name in weights
                    },
                },
            },
            {
                "area": _u(*(f"a{index}" for index in range(len(region)))),
                "region": _u(*region),
                "decile": _u(*decile),
                **{
                    name: np.asarray(values, dtype=np.int64)
                    for name, values in weights.items()
                },
            },
        )
    )


#: Five areas: band x has mass only in a0 (north, 1) and band y only in a2
#: (south, 1); a1 (north, 2) has households but no banded mass; a4, the fen's
#: one area, has no mass in any column.
LADDER_SUPPORT = _conditioned(
    region=("north", "north", "south", "south", "fen"),
    decile=("1", "2", "1", "2", "1"),
    households=(10, 30, 20, 0, 0),
    households_x=(5, 0, 0, 0, 0),
    households_y=(0, 0, 7, 0, 0),
)


def _households(*rows) -> pd.DataFrame:
    """Households as (region, decile, band); None is a missing input."""
    return pd.DataFrame(
        list(rows), columns=["region", "draw_decile", "band"], dtype=object
    )


def _choose(households, support=LADDER_SUPPORT, **overrides) -> list[str]:
    arguments = {
        "constraint_columns": CONSTRAINTS,
        "weight_by": RULE,
        "default_weight": "households",
        "levels": LADDER,
        **overrides,
    }
    return choose_fallback_levels(support, households, **arguments).tolist()


def _level_weight(household, level, weight_by, default_weight="households"):
    """The weight column a household samples by at ``level``, or None where
    an input the level reads is missing or, for its own weight, unmapped."""
    if any(pd.isna(household[column]) for column in level.constraints):
        return None
    if not level.weight_by:
        return default_weight
    code = household[weight_by["input"]]
    return None if pd.isna(code) else weight_by["map"].get(code)


def _per_row(support, households, *, weight_by, levels, constraint_columns=CONSTRAINTS):
    """The chooser's rule one household and one level at a time, on boolean
    masks of the support's arrays: each household's first level whose inputs
    are present and whose cell has mass, or None where some household has
    none."""
    chosen = []
    for _, household in households.iterrows():
        for level in levels:
            weight = _level_weight(household, level, weight_by)
            if weight is None:
                continue
            cell = np.ones(len(support.arrays["area"]), dtype=bool)
            for column in level.constraints:
                cell &= support.arrays[constraint_columns[column]] == household[column]
            if int(support.arrays[weight][cell].sum()) > 0:
                chosen.append(level.name)
                break
        else:
            return None
    return chosen


def test_a_household_whose_band_has_mass_in_its_cell_takes_level_zero():
    rows = _households(("north", "1", "x"), ("south", "1", "y"))
    assert _choose(rows) == ["L0", "L0"]


def test_a_household_whose_cell_has_only_households_takes_level_one():
    # a1 (north, 2) has no band-x mass; band z is a code the rule does not map,
    # so its household skips both band-weighted levels.
    rows = _households(("north", "2", "x"), ("north", "1", "z"))
    assert _choose(rows) == ["L1", "L1"]


def test_a_missing_decile_falls_to_the_region_levels():
    rows = _households(
        ("north", None, "x"),  # band x has mass in the north: L2
        ("south", None, "x"),  # but none in the south: L3
        ("north", None, None),  # no band either: L3
        ("north", "1", None),  # a decile and no band: L1
    )
    assert _choose(rows) == ["L2", "L3", "L3", "L1"]


@pytest.mark.parametrize(
    "row",
    [
        ("fen", "1", "x"),  # the fen's area has no mass in any column
        ("moor", "1", "x"),  # no area in the moor
        (None, "1", "x"),  # every level keeps the region
    ],
)
def test_a_household_no_level_takes_is_refused_naming_no_count_or_value(row):
    with pytest.raises(ValueError) as refused:
        _choose(_households(("north", "1", "x"), row, ("south", None, None)))
    assert str(refused.value) == NO_LEVEL


def test_without_households_the_chooser_returns_no_level():
    assert _choose(_households()) == []


def test_the_vectorised_choice_equals_a_per_row_loop():
    rng = np.random.default_rng(931)
    areas, rows = 60, 3000
    counts, band_x, band_y = (rng.integers(0, 4, areas) for _ in range(3))
    # Many areas have no banded mass; every region keeps households, and every
    # weight column has some mass.
    band_x *= rng.integers(0, 2, areas)
    band_y *= rng.integers(0, 2, areas)
    counts[:3] += 1
    band_x[0] += 1
    band_y[1] += 1
    support = _conditioned(
        region=np.tile(["r1", "r2", "r3"], areas // 3),
        decile=rng.choice(["1", "2", "3", "4"], areas),
        households=counts,
        households_x=band_x,
        households_y=band_y,
    )
    frame = pd.DataFrame(
        {
            "region": rng.choice(["r1", "r2", "r3"], rows),
            # Decile 5 and band z are absent from the support and the rule.
            "draw_decile": rng.choice(["1", "2", "3", "4", "5", None], rows),
            "band": rng.choice(["x", "y", "z", None], rows),
        },
        dtype=object,
    )
    expected = _per_row(support, frame, weight_by=RULE, levels=LADDER)
    assert set(expected) == {"L0", "L1", "L2", "L3"}
    assert _choose(frame, support) == expected
    # The answer is per household: it does not depend on the row order.
    order = rng.permutation(rows)
    assert _choose(frame.iloc[order], support) == [expected[i] for i in order]


#: Every level the invented ladders can hold: each set of the two constraints,
#: sampled by the household's own weight or by households.
_RUNGS = tuple(
    (constraints, weighted)
    for constraints in ((), ("region",), ("draw_decile",), ("region", "draw_decile"))
    for weighted in (True, False)
)


@st.composite
def _invented(draw):
    """An invented support, households and ladder over code domains so small
    that empty, absent and missing cells are common: region r9 and decile 9
    are in no support, band z is in no rule, and an input may be missing."""
    areas = draw(st.integers(1, 6))

    def per_area(elements):
        return draw(st.lists(elements, min_size=areas, max_size=areas))

    weights = {}
    for name in ("households", "households_x", "households_y"):
        values = per_area(st.integers(0, 2))
        if not any(values):
            values[0] = 1  # the decoder requires mass in every weight column
        weights[name] = values
    support = _conditioned(
        region=per_area(st.sampled_from(("r1", "r2"))),
        decile=per_area(st.sampled_from(("1", "2"))),
        **weights,
    )
    rows = draw(st.integers(0, 8))

    def per_household(codes, stray):
        # A code a support or rule holds is twice as likely as a stray one or
        # a missing input, so most households find some level.
        choices = (*codes, *codes, stray, None)
        return draw(st.lists(st.sampled_from(choices), min_size=rows, max_size=rows))

    households = pd.DataFrame(
        {
            "key": [f"h{row}" for row in range(rows)],
            "lane": ["toy"] * rows,
            "region": per_household(("r1", "r2"), "r9"),
            "draw_decile": per_household(("1", "2"), "9"),
            "band": per_household(("x", "y"), "z"),
        },
        dtype=object,
    )
    rule = draw(
        st.sampled_from((None, RULE, {"input": "band", "map": {"x": "households_y"}}))
    )
    rungs = draw(
        st.lists(
            st.sampled_from([r for r in _RUNGS if rule is not None or not r[1]]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    levels = tuple(
        FallbackLevel(f"L{index}", constraints, weight_by=weighted)
        for index, (constraints, weighted) in enumerate(rungs)
    )
    return support, households, rule, levels


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_invented())
def test_on_invented_supports_the_chooser_equals_the_per_row_loop(invented):
    support, households, rule, levels = invented
    expected = _per_row(support, households, weight_by=rule, levels=levels)
    if expected is None:
        with pytest.raises(ValueError) as refused:
            _choose(households, support, weight_by=rule, levels=levels)
        assert str(refused.value) == NO_LEVEL
    else:
        assert _choose(households, support, weight_by=rule, levels=levels) == expected


def _draw_spec(rule) -> dict:
    """The shared draw over an invented support: a constant lane routes every
    household, both constraints are optional and one stage samples areas by
    ``rule`` (households where it is None or a code is missing)."""
    stage = (
        {"level": "area", "weight": "households"}
        if rule is None
        else {"level": "area", "weight_by": {**rule, "default": "households"}}
    )
    return validate_assignment_spec(
        {
            "version": 1,
            "identity": ["key"],
            "stream": ["sha256-u53-v1", "toy", 0, 1],
            "outputs": {
                "area": "area_code",
                "system": "area_system",
                "basis": "area_basis",
            },
            "systems": [
                {
                    "id": "toy",
                    "level": "area",
                    "code_system": "toy",
                    "vintage": "v1",
                    "source": "toy_support",
                    "selector": {"lane": ["toy"]},
                    "constraints": [
                        {"input": column, "support": matched, "required": False}
                        for column, matched in CONSTRAINTS.items()
                    ],
                    "observed_area": None,
                    "stages": [stage],
                    "layers": [],
                }
            ],
        }
    )


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_invented())
def test_on_invented_supports_the_draw_meets_mass_at_the_chosen_level_only(invented):
    """The chosen level is the first at which the shared draw itself finds
    mass: drawing there, with every input the level drops missing, lands in
    an area that matches the kept inputs and carries the household's weight,
    and drawing at any earlier level whose inputs are present is refused for
    an empty or absent cell. A household the chooser refuses is refused by
    the draw at every level its inputs allow."""
    support, households, rule, levels = invented
    spec = _draw_spec(rule)
    for row in range(len(households)):
        household = households.iloc[[row]]
        try:
            (chosen,) = _choose(household, support, weight_by=rule, levels=levels)
        except ValueError:
            chosen = None
        for level in levels:
            weight = _level_weight(household.iloc[0], level, rule)
            if weight is None:
                continue
            at_level = household.copy()
            for column in CONSTRAINTS:
                if column not in level.constraints:
                    at_level[column] = None
            if rule is not None and not level.weight_by:
                at_level[rule["input"]] = None
            if level.name != chosen:
                with pytest.raises(
                    ValueError, match="no common support|zero sampling mass"
                ):
                    assign_atomic(at_level, spec, {"toy": support})
                continue
            area = assign_atomic(at_level, spec, {"toy": support})["area_code"].iloc[0]
            (index,) = np.flatnonzero(support.arrays["area"] == area)
            for column in level.constraints:
                assert (
                    support.arrays[CONSTRAINTS[column]][index]
                    == household[column].iloc[0]
                )
            assert support.arrays[weight][index] > 0
            break
        else:
            # A chosen level is one whose inputs the household has.
            assert chosen is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weight_by": None}, "no weight_by is given"),
        (
            {"levels": (FallbackLevel("L0", ("tenure",), weight_by=False),)},
            "does not map",
        ),
        ({"constraint_columns": {"region": "households"}}, "support code columns"),
        ({"constraint_columns": {"region": "area"}}, "support code columns"),
        (
            {"constraint_columns": {"region": "region", "draw_decile": "region"}},
            "support code columns",
        ),
        ({"default_weight": "decile"}, "default_weight must be"),
        ({"weight_by": {"input": "band", "map": {"x": "decile"}}}, "weight_by must"),
        ({"weight_by": {**RULE, "default": "households"}}, "weight_by must"),
        ({"levels": ()}, "nonempty sequence"),
        (
            {
                "levels": (
                    FallbackLevel("L0", ("region",), weight_by=False),
                    FallbackLevel("L0", (), weight_by=False),
                )
            },
            "names must be distinct",
        ),
    ],
)
def test_the_chooser_refuses_declarations_it_cannot_read(overrides, message):
    with pytest.raises(ValueError, match=message):
        _choose(_households(("north", "1", "x")), **overrides)


@pytest.mark.parametrize(
    ("households", "message"),
    [
        (_households(("north", "1", "x")).drop(columns="band"), "lack input columns"),
        (_households(("north", 1, "x")), "nonempty strings"),
        (_households(("north", " 1", "x")), "nonempty strings"),
        (_households(("north", "1", "")), "nonempty strings"),
    ],
)
def test_the_chooser_refuses_missing_inputs_and_malformed_codes(households, message):
    with pytest.raises(ValueError, match=message):
        _choose(households)


@pytest.mark.parametrize(
    ("name", "constraints", "weight_by", "message"),
    [
        ("", ("region",), True, "needs a name"),
        ("L0", ["region"], True, "tuple of distinct"),
        ("L0", ("region", "region"), True, "tuple of distinct"),
        ("L0", ("region",), 1, "is a bool"),
    ],
)
def test_a_fallback_level_refuses_a_malformed_declaration(
    name, constraints, weight_by, message
):
    with pytest.raises(ValueError, match=message):
        FallbackLevel(name, constraints, weight_by=weight_by)
