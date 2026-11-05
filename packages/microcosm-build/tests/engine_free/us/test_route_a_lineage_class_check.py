"""The stage-001 lineage check accepts only the rounding class Max ruled on (d1712, d2274, d2336)."""

from __future__ import annotations

import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_modal_stage_plan_tool import ROOT


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location(
        "route_a_lineage_class_check", ROOT / "tools/route_a/lineage_class_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tables(rent, *, esi=None, other=(1.0, 2.0), ages=(30, 40, 50), index=(1, 2, 3)):
    esi = np.zeros(len(rent)) if esi is None else np.asarray(esi, dtype=float)
    return {
        "person": pd.DataFrame(
            {
                "pre_subsidy_rent": np.asarray(rent, dtype=float),
                "employer_sponsored_insurance_premiums": esi,
                "age": list(ages),
            },
            index=pd.Index(list(index), name="person_id"),
        ),
        "household": pd.DataFrame(
            {"other": np.asarray(other, dtype=float)},
            index=pd.Index([1, 2], name="household_id"),
        ),
    }


def _run(check, monkeypatch, tmp_path, local, modal):
    paths = {tmp_path / "local.h5": local, tmp_path / "modal.h5": modal}
    for path in paths:
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(check, "tables", lambda path: paths[Path(path)])
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["check", *(str(p) for p in paths), str(out)])
    return check.main(), json.loads(out.read_text())


def test_identical_frames_pass_with_no_differences(check, monkeypatch, tmp_path):
    rc, verdict = _run(
        check,
        monkeypatch,
        tmp_path,
        _tables([100.0, 0.0, 250.5]),
        _tables([100.0, 0.0, 250.5]),
    )
    assert rc == 0
    assert verdict["inside_accepted_class"] is True
    assert verdict["differences"] == []


def test_ulp_difference_in_the_9_30_column_is_inside_the_class(
    check, monkeypatch, tmp_path
):
    modal = [100.0, 0.0, 250.5 * (1 + 4e-16)]
    rc, verdict = _run(
        check, monkeypatch, tmp_path, _tables([100.0, 0.0, 250.5]), _tables(modal)
    )
    assert rc == 0
    [difference] = verdict["differences"]
    assert (difference["table"], difference["column"]) == ("person", "pre_subsidy_rent")
    assert difference["rows_differ"] == 1
    assert difference["max_rel"] <= check.REL_TOL


def test_ulp_difference_in_the_esi_premium_column_is_inside_the_class(
    check, monkeypatch, tmp_path
):
    # d2274: on the 2026-11 base every differing ESI value sat one unit in the
    # last place above the Mac value.
    esi = (0.0, 5_000.0, 7_250.25)
    modal_esi = (0.0, float(np.nextafter(5_000.0, np.inf)), 7_250.25)
    rc, verdict = _run(
        check,
        monkeypatch,
        tmp_path,
        _tables([100.0, 0.0, 250.5], esi=esi),
        _tables([100.0, 0.0, 250.5], esi=modal_esi),
    )
    assert rc == 0
    [difference] = verdict["differences"]
    assert (difference["table"], difference["column"]) == (
        "person",
        "employer_sponsored_insurance_premiums",
    )
    assert difference["rows_differ"] == 1
    assert difference["max_rel"] <= check.REL_TOL


def test_the_accepted_columns_are_exactly_the_ruled_ones(check):
    # Widening this set changes a stop rule Max set; it needs his ruling.
    assert check.ACCEPTED_COLUMNS == {
        ("person", "pre_subsidy_rent"),
        ("person", "employer_sponsored_insurance_premiums"),
    }


@pytest.mark.parametrize(
    ("local", "modal", "problem"),
    [
        # Within tolerance, but not an accepted column: stops for Max.
        (
            _tables([1.0, 2.0, 3.0]),
            _tables([1.0, 2.0, 3.0], other=(1.0, 2.0 * (1 + 4e-16))),
            "not an accepted column",
        ),
        # Above 1e-12 relative in the ESI column accepted under d2274 and d2336.
        (
            _tables([1.0, 2.0, 3.0], esi=(0.0, 5_000.0, 7_250.25)),
            _tables([1.0, 2.0, 3.0], esi=(0.0, 5_000.0, 7_251.0)),
            "difference above",
        ),
        # Above 1e-12 relative in the 9/30 column.
        (_tables([1.0, 2.0, 3.0]), _tables([1.0, 2.0, 3.001]), "difference above"),
        # A non-float column that differs.
        (
            _tables([1.0, 2.0, 3.0]),
            _tables([1.0, 2.0, 3.0], ages=(30, 40, 51)),
            "non-float column differs",
        ),
        # NaN positions that differ.
        (_tables([1.0, np.nan, 3.0]), _tables([1.0, 2.0, 3.0]), "NaN positions differ"),
        # A shape change.
        (
            _tables([1.0, 2.0, 3.0]),
            _tables([1.0, 2.0], ages=(30, 40), index=(1, 2)),
            "shape",
        ),
        # The same shape on a different index.
        (
            _tables([1.0, 2.0, 3.0]),
            _tables([1.0, 2.0, 3.0], index=(1, 2, 4)),
            "index differs",
        ),
    ],
)
def test_anything_outside_the_class_fails(
    check, monkeypatch, tmp_path, local, modal, problem
):
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 1
    assert verdict["inside_accepted_class"] is False
    assert any(problem in entry for entry in verdict["problems"])


def test_each_value_within_tolerance_but_the_total_beyond_it_fails(
    check, monkeypatch, tmp_path
):
    # Every value moves by at most 5e-13 relative, but signed values cancel,
    # so the column total moves by about 5e-7 relative.
    local = _tables([1e6, -999999.0, 0.0])
    modal = _tables([1e6 * (1 + 5e-13), -999999.0, 0.0])
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 1
    [difference] = verdict["differences"]
    assert difference["max_rel"] <= check.REL_TOL < difference["sum_rel_diff"]
    assert any("difference above" in entry for entry in verdict["problems"])


def test_a_total_that_leaves_zero_fails(check, monkeypatch, tmp_path):
    # Each value moves by about 1e-13 relative, but the local total is exactly
    # zero, so the total moves by 100%. A floor on the divisor hid this.
    local = _tables([1.0, 2.0, 3.0], esi=(1e-307, -1e-307, 0.0))
    modal = _tables([1.0, 2.0, 3.0], esi=(1e-307 + 1e-320, -1e-307, 0.0))
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 1
    [difference] = verdict["differences"]
    assert difference["sum_local"] == 0.0 != difference["sum_modal"]
    assert difference["max_rel"] <= check.REL_TOL < difference["sum_rel_diff"]
    assert any("difference above" in entry for entry in verdict["problems"])


def test_differing_values_with_equal_zero_totals_pass(check, monkeypatch, tmp_path):
    # Both totals are exactly zero, so they agree; each value sits one ulp off.
    up = float(np.nextafter(2.0, np.inf))
    local = _tables([1.0, 2.0, 3.0], esi=(2.0, -2.0, 0.0))
    modal = _tables([1.0, 2.0, 3.0], esi=(up, -up, 0.0))
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 0
    [difference] = verdict["differences"]
    assert difference["sum_local"] == difference["sum_modal"] == 0.0
    assert difference["sum_rel_diff"] == 0.0


def _esi_tables(esi):
    rows = len(esi)
    return _tables(
        [1.0] * rows,
        esi=esi,
        ages=tuple(range(30, 30 + rows)),
        index=tuple(range(1, rows + 1)),
    )


_UP_1 = float(np.nextafter(1.0, np.inf))


@pytest.mark.parametrize(
    ("local_esi", "modal_esi"),
    [
        # The same infinite value on both sides: the finite part of the total
        # leaves zero, but both float totals read inf.
        ((math.inf, 1.0, -1.0), (math.inf, _UP_1, -1.0)),
        ((-math.inf, 1.0, -1.0), (-math.inf, _UP_1, -1.0)),
        # Every value finite, but the running total passes the float range.
        (
            (1e308, 1e308, -1e308, -1e308),
            (float(np.nextafter(1e308, np.inf)), 1e308, -1e308, -1e308),
        ),
    ],
)
def test_a_total_the_check_cannot_compute_fails(
    check, monkeypatch, tmp_path, local_esi, modal_esi
):
    rc, verdict = _run(
        check, monkeypatch, tmp_path, _esi_tables(local_esi), _esi_tables(modal_esi)
    )
    assert rc == 1
    [difference] = verdict["differences"]
    assert difference["max_rel"] <= check.REL_TOL
    assert difference["sum_rel_diff"] is None
    assert any("column total not finite" in entry for entry in verdict["problems"])


def test_a_total_move_that_a_large_pair_absorbs_fails(check, monkeypatch, tmp_path):
    # np.nansum gives -1.0 for both columns: 1e17 swallows the 1.0 and its
    # one-ulp move. The exact totals are 0 and 2.2e-16.
    local = _esi_tables((1e17, 1.0, -1e17, -1.0))
    modal = _esi_tables((1e17, _UP_1, -1e17, -1.0))
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 1
    [difference] = verdict["differences"]
    assert difference["sum_local"] == 0.0 != difference["sum_modal"]
    assert difference["max_rel"] <= check.REL_TOL < difference["sum_rel_diff"]


def test_a_zero_total_against_infinite_values_records_its_stop(
    check, monkeypatch, tmp_path
):
    # max(0.0, nan) is 0.0 in Python, so a NaN total once divided by zero and
    # raised before the check wrote its verdict.
    local = _esi_tables((0.0, 0.0))
    modal = _esi_tables((math.inf, -math.inf))
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 1
    assert verdict["inside_accepted_class"] is False
    assert any("employer_sponsored" in entry for entry in verdict["problems"])


def _two_level(
    tables, people, names=("person_id", "household_id"), households=(1, 1, 2)
):
    tables["person"].index = pd.MultiIndex.from_arrays(
        [list(people), list(households)], names=list(names)
    )
    return tables


@pytest.mark.parametrize(
    ("local", "modal"),
    [
        # Index.equals compares values only, so it misses each of these.
        (_tables([1.0, 2.0, 3.0]), _tables([1.0, 2.0, 3.0], index=(1.0, 2.0, 3.0))),
        (
            _tables([1.0, 2.0, 3.0]),
            {
                **_tables([1.0, 2.0, 3.0]),
                "person": _tables([1.0, 2.0, 3.0])["person"].rename_axis(
                    "household_id"
                ),
            },
        ),
        # The same changes in either level of a two-level index.
        (
            _two_level(_tables([1.0, 2.0, 3.0]), (1, 2, 3)),
            _two_level(_tables([1.0, 2.0, 3.0]), (1.0, 2.0, 3.0)),
        ),
        (
            _two_level(_tables([1.0, 2.0, 3.0]), (1, 2, 3)),
            _two_level(_tables([1.0, 2.0, 3.0]), (1, 2, 3), ("person_id", "unit_id")),
        ),
        (
            _two_level(_tables([1.0, 2.0, 3.0]), (1, 2, 3)),
            _two_level(
                _tables([1.0, 2.0, 3.0]), (1, 2, 3), ("unit_id", "household_id")
            ),
        ),
        (
            _two_level(_tables([1.0, 2.0, 3.0]), (1, 2, 3)),
            _two_level(_tables([1.0, 2.0, 3.0]), (1, 2, 3), households=(1.0, 1.0, 2.0)),
        ),
    ],
)
def test_an_index_dtype_or_name_change_fails(
    check, monkeypatch, tmp_path, local, modal
):
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 1
    assert any("index names or dtypes differ" in entry for entry in verdict["problems"])


def test_a_dtype_change_fails(check, monkeypatch, tmp_path):
    modal = _tables([1.0, 2.0, 3.0])
    modal["person"]["age"] = modal["person"]["age"].astype(float)
    rc, verdict = _run(check, monkeypatch, tmp_path, _tables([1.0, 2.0, 3.0]), modal)
    assert rc == 1
    assert any("dtype" in entry for entry in verdict["problems"])


def test_a_renamed_column_fails(check, monkeypatch, tmp_path):
    # Same shape, different names (an extra column is caught as a shape change).
    modal = _tables([1.0, 2.0, 3.0])
    modal["person"] = modal["person"].rename(columns={"age": "age_years"})
    rc, verdict = _run(check, monkeypatch, tmp_path, _tables([1.0, 2.0, 3.0]), modal)
    assert rc == 1
    assert any("column names or order" in entry for entry in verdict["problems"])


def test_nans_at_the_same_positions_with_an_ulp_difference_pass(
    check, monkeypatch, tmp_path
):
    local = _tables([1.0, np.nan, 3.0])
    modal = _tables([1.0, np.nan, 3.0 * (1 + 4e-16)])
    rc, verdict = _run(check, monkeypatch, tmp_path, local, modal)
    assert rc == 0
    [difference] = verdict["differences"]
    assert difference["rows_differ"] == 1


def test_a_missing_table_fails(check, monkeypatch, tmp_path):
    modal = _tables([1.0, 2.0, 3.0])
    del modal["household"]
    rc, verdict = _run(check, monkeypatch, tmp_path, _tables([1.0, 2.0, 3.0]), modal)
    assert rc == 1
    assert any("table sets differ" in entry for entry in verdict["problems"])


_RENTS = st.lists(
    st.floats(min_value=0.0, max_value=1e7, allow_nan=False), min_size=1, max_size=40
)


@settings(max_examples=60, deadline=None)
@given(rents=_RENTS, data=st.data())
def test_property_nonnegative_rents_within_1e13_are_inside(check, rents, data):
    # Rent is non-negative, so per-value perturbations of at most 1e-13
    # relative keep the total within 1e-12 relative too.
    eps = data.draw(
        st.lists(
            st.floats(min_value=-1e-13, max_value=1e-13),
            min_size=len(rents),
            max_size=len(rents),
        )
    )
    local = pd.Series(rents, dtype=float)
    modal = pd.Series(
        [r * (1 + e) for r, e in zip(rents, eps, strict=True)], dtype=float
    )
    entry = check.compare_column("person", "pre_subsidy_rent", local, modal)
    assert entry is None or "problem" not in entry


@settings(max_examples=60, deadline=None)
@given(rents=_RENTS, data=st.data())
def test_property_any_value_moved_by_1e9_is_outside(check, rents, data):
    position = data.draw(st.integers(min_value=0, max_value=len(rents) - 1))
    local = pd.Series(rents, dtype=float)
    moved = list(rents)
    moved[position] = rents[position] * (1 + 1e-9) + 1.0
    entry = check.compare_column(
        "person", "pre_subsidy_rent", local, pd.Series(moved, dtype=float)
    )
    assert entry is not None and "problem" in entry


_SIGNED = st.builds(
    lambda magnitude, negative: -magnitude if negative else magnitude,
    st.one_of(
        st.just(0.0),
        st.floats(min_value=1e-310, max_value=1e-300),
        st.floats(min_value=1e-3, max_value=1e3),
        st.floats(min_value=1e15, max_value=1e20),
    ),
    st.booleans(),
)


def _nudge(value, steps):
    for _ in range(abs(steps)):
        value = float(np.nextafter(value, math.copysign(math.inf, steps)))
    return value


def _exact_rel(a, b):
    a, b = Fraction(a), Fraction(b)
    return Fraction(0) if a == b else abs(a - b) / max(abs(a), abs(b))


@settings(max_examples=1000, deadline=None)
@given(
    base=st.lists(_SIGNED, min_size=1, max_size=12),
    mirrored=st.integers(min_value=0, max_value=12),
    steps=st.lists(st.integers(-3, 3), min_size=24, max_size=24),
    bumped=st.one_of(st.none(), st.integers(min_value=0, max_value=23)),
)
# A total that leaves exact zero by one subnormal ulp: a floor on the
# divisor hides the move.
@example(base=[1e-310], mirrored=1, steps=[1] + [0] * 23, bumped=None)
# A one-ulp move beside a large pair: a running float sum absorbs it.
@example(base=[1e17, 1.0], mirrored=2, steps=[0, 1] + [0] * 22, bumped=None)
def test_property_the_check_agrees_with_exact_arithmetic(
    check, base, mirrored, steps, bumped
):
    # Mirrored values cancel exactly, so totals sit at or near zero while
    # large values absorb small moves in float sums. Each value moves up to
    # three ulps, and sometimes one moves by 1e-9 relative.
    local = base + [-value for value in base[:mirrored]]
    modal = [
        _nudge(value, k) for value, k in zip(local, steps[: len(local)], strict=True)
    ]
    if bumped is not None:
        modal[bumped % len(modal)] *= 1 + 1e-9
    entry = check.compare_column(
        "person",
        "employer_sponsored_insurance_premiums",
        pd.Series(local, dtype=float),
        pd.Series(modal, dtype=float),
    )
    if entry is None:
        assert local == modal
        return
    tol = Fraction(check.REL_TOL)
    value_rel = max(_exact_rel(a, b) for a, b in zip(local, modal, strict=True))
    total_rel = _exact_rel(
        sum(map(Fraction, local), Fraction(0)), sum(map(Fraction, modal), Fraction(0))
    )
    if "problem" in entry:
        # No false stop: a move well inside the tolerance passes.
        assert value_rel > tol / 2 or total_rel > tol / 2
    else:
        # A pass never hides a value or total past the tolerance, allowing
        # for the one rounding of each float total.
        assert value_rel <= tol * Fraction(1001, 1000)
        assert total_rel <= tol * Fraction(1001, 1000)


@settings(max_examples=200, deadline=None)
@given(
    base=st.lists(_SIGNED, min_size=1, max_size=12),
    steps=st.lists(st.integers(-3, 3), min_size=12, max_size=12),
    infinite=st.sampled_from([math.inf, -math.inf]),
)
# The same infinite value on both sides makes the float totals equal.
@example(base=[1.0, -1.0], steps=[1] + [0] * 11, infinite=math.inf)
def test_property_an_infinite_value_stops_any_difference(check, base, steps, infinite):
    # An infinite value hides how far the total moved, so the check stops
    # whenever any value in its column differs.
    local = [*base, infinite]
    modal = [
        *(_nudge(value, k) for value, k in zip(base, steps[: len(base)], strict=True)),
        infinite,
    ]
    entry = check.compare_column(
        "person",
        "employer_sponsored_insurance_premiums",
        pd.Series(local, dtype=float),
        pd.Series(modal, dtype=float),
    )
    if entry is None:
        assert local == modal
    else:
        assert "problem" in entry


_PREMIUMS = st.lists(
    st.floats(min_value=1.0, max_value=1e6, allow_nan=False), min_size=3, max_size=3
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(premiums=_PREMIUMS, nudged=st.lists(st.booleans(), min_size=3, max_size=3))
def test_property_one_ulp_up_in_esi_premiums_is_inside(
    check, monkeypatch, tmp_path, premiums, nudged
):
    # d2274's class: each differing premium sits one ulp above the local value.
    modal = [
        float(np.nextafter(p, np.inf)) if up else p
        for p, up in zip(premiums, nudged, strict=True)
    ]
    rc, verdict = _run(
        check,
        monkeypatch,
        tmp_path,
        _tables([1.0, 2.0, 3.0], esi=premiums),
        _tables([1.0, 2.0, 3.0], esi=modal),
    )
    assert rc == 0
    assert verdict["inside_accepted_class"] is True
