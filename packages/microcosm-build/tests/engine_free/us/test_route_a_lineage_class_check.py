"""The d1712 stage-001 lineage check accepts only the 9/30 float-rounding class."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
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


def _tables(rent, *, other=(1.0, 2.0), ages=(30, 40, 50), index=(1, 2, 3)):
    return {
        "person": pd.DataFrame(
            {"pre_subsidy_rent": np.asarray(rent, dtype=float), "age": list(ages)},
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


@pytest.mark.parametrize(
    ("local", "modal", "problem"),
    [
        # Within tolerance, but not the 9/30 column: stops for Max.
        (
            _tables([1.0, 2.0, 3.0]),
            _tables([1.0, 2.0, 3.0], other=(1.0, 2.0 * (1 + 4e-16))),
            "not the 9/30 column",
        ),
        # Above 1e-12 relative in the accepted column.
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
