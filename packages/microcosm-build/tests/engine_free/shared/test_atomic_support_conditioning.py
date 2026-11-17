"""The country-neutral columns a support carries for a conditioned draw."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.atomic_geography import (
    assign_atomic,
    decode_atomic_support,
    encode_atomic_support,
    validate_assignment_spec,
)
from microcosm.build.atomic_support_conditioning import (
    apportion_weights,
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
