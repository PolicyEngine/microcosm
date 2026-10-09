"""The ACS local release's district ESS gate (finalize stage).

Invariants, each for every input Hypothesis draws:

1. The collapsed districts are nested in the relative floor, and the districts
   below the floor in the absolute floor, so both counts are monotone in their
   threshold.
2. A design-weight solve (calibrated weights equal to the design weights, or a
   uniform rescaling of them) has no collapsed district at any relative floor
   in [0, 1].
3. Both counts are at most the number of districts, which is the number of
   distinct district codes.
4. ``blocking`` changes only ``passed``, ``blocking`` and ``report_only``: a
   report-only entry always passes, and a blocking one passes exactly when no
   district collapses or falls below the floor.
5. Differential: the collapsed districts the gate reads from
   ``weight_origin_summary`` (the calibration summary's evidence) are the ones
   an independent bincount computation finds from the raw weights, away from
   floating-point ties.
"""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest

from test_support.paths import paths_for

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_TEST_PATHS = paths_for("microcosm-build")
_SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_district_ess_gate", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool_module()
_CD = _TOOL.cd_surface
_GATE = _TOOL.district_ess_collapse_gate
_LIMITATION = _TOOL.district_ess_collapse_limitation


def _origin(design: dict, calibrated: dict) -> dict:
    """A calibration summary's ``weight_origin`` with per-district ESS only."""

    return {
        "design": {"effective_sample_size_by_district": design},
        "calibrated": {"effective_sample_size_by_district": calibrated},
    }


def _districts(rows: list[dict]) -> list[str]:
    return [row["district"] for row in rows]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_the_floors_default_to_the_measured_separation() -> None:
    assert _TOOL.DISTRICT_ESS_RELATIVE_FLOOR == 0.25
    assert _TOOL.DISTRICT_ESS_FLOOR == 15.0
    assert _TOOL.DISTRICT_ESS_GATE == "district_ess_collapse"


def test_the_gate_counts_collapsed_and_below_floor_districts() -> None:
    gate = _GATE(
        _origin(
            {"0101": 100.0, "0102": 100.0, "0103": 40.0, "0104": 80.0, "0105": 30.0},
            {"0101": 20.0, "0102": 25.0, "0103": 12.0, "0104": 60.0, "0105": 15.0},
        )
    )
    # 0101 keeps a fifth of its design ESS; 0102 exactly a quarter, which is
    # not below it; 0103 keeps 30% but ends below 15; 0105 sits on the floor.
    assert gate["n_districts"] == 5
    assert _districts(gate["collapsed"]) == ["0101"]
    assert _districts(gate["below_floor"]) == ["0103"]
    assert (gate["n_collapsed"], gate["n_below_floor"]) == (1, 1)
    assert gate["collapsed"][0] == {
        "district": "0101",
        "calibrated_ess": 20.0,
        "design_ess": 100.0,
        "ratio": 0.2,
    }
    assert gate["criteria_met"] is False
    assert (gate["min_calibrated_ess"], gate["min_calibrated_ess_district"]) == (
        12.0,
        "0103",
    )
    assert (gate["min_ess_ratio"], gate["min_ess_ratio_district"]) == (0.2, "0101")
    assert gate["failures"] == []
    # Report-only by default: the entry passes and says what it measured.
    assert gate["passed"] is True
    assert (gate["blocking"], gate["report_only"]) == (False, True)
    assert (gate["relative_floor"], gate["absolute_floor"]) == (0.25, 15.0)


def test_collapsed_districts_are_listed_worst_first() -> None:
    gate = _GATE(
        _origin(
            {"a": 100.0, "b": 100.0, "c": 100.0},
            {"a": 20.0, "b": 10.0, "c": 20.0},
        ),
        absolute_floor=0.0,
    )
    assert _districts(gate["collapsed"]) == ["b", "a", "c"]


def test_a_blocking_gate_passes_only_when_no_district_collapses() -> None:
    collapsed = _origin({"1": 100.0, "2": 50.0}, {"1": 10.0, "2": 40.0})
    clean = _origin({"1": 100.0, "2": 50.0}, {"1": 90.0, "2": 40.0})
    below_floor = _origin({"1": 100.0, "2": 12.0}, {"1": 90.0, "2": 11.0})

    failing = _GATE(collapsed, blocking=True)
    assert (failing["passed"], failing["criteria_met"]) == (False, False)
    assert (failing["blocking"], failing["report_only"]) == (True, False)
    assert _GATE(below_floor, blocking=True)["passed"] is False
    passing = _GATE(clean, blocking=True)
    assert (passing["passed"], passing["criteria_met"]) == (True, True)


def test_a_zero_floor_turns_its_check_off() -> None:
    origin = _origin({"1": 100.0, "2": 10.0}, {"1": 1.0, "2": 9.0})

    assert (_GATE(origin)["n_collapsed"], _GATE(origin)["n_below_floor"]) == (1, 2)
    relative_off = _GATE(origin, relative_floor=0.0)
    assert (relative_off["n_collapsed"], relative_off["n_below_floor"]) == (0, 2)
    absolute_off = _GATE(origin, absolute_floor=0.0)
    assert (absolute_off["n_collapsed"], absolute_off["n_below_floor"]) == (1, 0)
    both_off = _GATE(origin, relative_floor=0.0, absolute_floor=0.0, blocking=True)
    assert (both_off["criteria_met"], both_off["passed"]) == (True, True)


def test_a_district_with_no_design_ess_cannot_collapse() -> None:
    gate = _GATE(_origin({"1": 0.0, "2": 50.0}, {"1": 0.0, "2": 40.0}))

    assert gate["collapsed"] == []
    assert _districts(gate["below_floor"]) == ["1"]
    assert gate["below_floor"][0]["ratio"] is None
    # The smallest ratio is taken over districts that have one.
    assert (gate["min_ess_ratio"], gate["min_ess_ratio_district"]) == (0.8, "2")


def test_integer_ess_values_from_json_are_read_as_numbers() -> None:
    gate = _GATE(_origin({"1": 100, "2": 40}, {"1": 20, "2": 40}))

    assert (gate["n_collapsed"], gate["n_below_floor"]) == (1, 0)
    assert gate["collapsed"][0]["calibrated_ess"] == 20.0


@pytest.mark.parametrize(
    "origin",
    [
        None,
        {},
        "not a mapping",
        # A summary written before weight_origin recorded districts.
        {"design": {"rows": 3}, "calibrated": {"rows": 3}},
        {"design": {"effective_sample_size_by_district": {"1": 10.0}}},
        _origin({}, {}),
        _origin({"1": 10.0}, {"2": 10.0}),
        _origin({"1": 10.0}, {"1": math.nan}),
        _origin({"1": math.inf}, {"1": 10.0}),
        _origin({"1": 10.0}, {"1": -1.0}),
        _origin({"1": 10.0}, {"1": "10"}),
        _origin({"1": 10.0}, {"1": True}),
        _origin({"1": 10.0}, {"1": None}),
        _origin([("1", 10.0)], {"1": 10.0}),
        {"design": [10.0], "calibrated": "10"},
    ],
    ids=[
        "none",
        "empty",
        "not-a-mapping",
        "legacy-summary",
        "no-calibrated",
        "no-districts",
        "different-districts",
        "nan",
        "infinite",
        "negative",
        "string",
        "bool",
        "null",
        "list",
        "blocks-not-mappings",
    ],
)
def test_missing_or_malformed_evidence_fails_only_a_blocking_gate(origin) -> None:
    report_only = _GATE(origin)
    assert report_only["criteria_met"] is None
    assert report_only["failures"]
    assert report_only["passed"] is True
    assert "n_collapsed" not in report_only

    blocking = _GATE(origin, blocking=True)
    assert blocking["criteria_met"] is None
    assert blocking["passed"] is False


@pytest.mark.parametrize(
    "floors",
    [
        {"relative_floor": -0.1},
        {"relative_floor": 1.5},
        {"relative_floor": math.nan},
        {"absolute_floor": -1.0},
        {"absolute_floor": math.inf},
        {"absolute_floor": math.nan},
    ],
)
def test_floors_out_of_range_are_refused(floors) -> None:
    with pytest.raises(ValueError, match="floor must be"):
        _GATE(_origin({"1": 10.0}, {"1": 10.0}), **floors)


def test_the_limitation_states_the_result_and_whether_it_blocks() -> None:
    collapsed = _origin({"0601": 100.0, "0602": 80.0}, {"0601": 20.0, "0602": 60.0})
    clean = _origin({"0601": 100.0, "0602": 80.0}, {"0601": 90.0, "0602": 60.0})

    report_only = _LIMITATION(_GATE(collapsed))
    assert report_only["id"] == "district_effective_sample_size_gate"
    assert report_only["status"] == "recorded_concentration"
    assert report_only["gate"] == _TOOL.DISTRICT_ESS_GATE
    assert report_only["reason"] == (
        "Congressional districts with a calibrated Kish ESS below 0.25 x their "
        "design-weight Kish ESS: 1 of 2; below 15: 0 of 2. Smallest calibrated "
        "district ESS: 20.0 (0601); smallest ratio to design ESS: 0.200 (0601). "
        "The gate is report-only for this build; --district-ess-gate-blocking "
        "makes it a hard failure."
    )
    assert (report_only["n_collapsed"], report_only["n_below_floor"]) == (1, 0)
    assert report_only["criteria_met"] is False
    assert report_only["blocking"] is False
    assert report_only["calibration_blocker"] is False

    blocked = _LIMITATION(_GATE(collapsed, blocking=True))
    assert blocked["reason"].endswith(
        "The gate is blocking for this build and fails, so the build is not "
        "simulation-ready."
    )
    assert blocked["calibration_blocker"] is True

    passed = _LIMITATION(_GATE(clean, blocking=True))
    assert passed["reason"].endswith("The gate is blocking for this build and passed.")
    assert passed["calibration_blocker"] is False

    unavailable = _LIMITATION(_GATE(None))
    assert unavailable["reason"].startswith(
        "The district ESS gate could not be evaluated: calibration_summary.json "
        "weight_origin records no"
    )
    assert unavailable["criteria_met"] is None
    assert unavailable["n_collapsed"] is None
    assert unavailable["calibration_blocker"] is False
    assert _LIMITATION(_GATE(None, blocking=True))["calibration_blocker"] is True

    unchecked = _LIMITATION(_GATE(collapsed, relative_floor=0.0, absolute_floor=0.0))
    assert unchecked["reason"].startswith(
        "No district ESS check is on for the 2 districts. Smallest calibrated "
        "district ESS: 20.0 (0601)"
    )
    relative_only = _LIMITATION(_GATE(collapsed, absolute_floor=0.0))
    assert relative_only["reason"].startswith(
        "Congressional districts with a calibrated Kish ESS below 0.25 x their "
        "design-weight Kish ESS: 1 of 2. Smallest"
    )


def test_the_release_contract_publishes_a_report_only_gate_whatever_it_measured():
    """Every local-area gate must pass to publish, so report-only must pass."""

    from microcosm.data import contract

    collapsed = _origin({"1": 100.0}, {"1": 10.0})
    for blocking, expected in ((False, 0), (True, 1)):
        failures: list[str] = []
        gate = _GATE(collapsed, blocking=blocking)
        contract._check_local_area_gates(
            {"gates": {_TOOL.DISTRICT_ESS_GATE: gate}}, failures
        )
        assert len(failures) == expected, failures


def _parse(tmp_path, *extra: str):
    return _TOOL._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
            *extra,
        ]
    )


def test_the_gate_is_report_only_unless_blocking_is_asked_for(tmp_path) -> None:
    args = _parse(tmp_path)
    assert args.district_ess_relative_floor == 0.25
    assert args.district_ess_floor == 15.0
    assert args.district_ess_gate_blocking is False

    args = _parse(
        tmp_path,
        "--district-ess-relative-floor",
        "0.5",
        "--district-ess-floor",
        "0",
        "--district-ess-gate-blocking",
    )
    assert (args.district_ess_relative_floor, args.district_ess_floor) == (0.5, 0.0)
    assert args.district_ess_gate_blocking is True


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--district-ess-relative-floor", "-0.1"),
        ("--district-ess-relative-floor", "1.5"),
        ("--district-ess-relative-floor", "nan"),
        ("--district-ess-floor", "-1"),
        ("--district-ess-floor", "inf"),
        ("--district-ess-floor", "nan"),
    ],
)
def test_floors_out_of_range_are_refused_at_parse_time(
    tmp_path, capsys, flag, value
) -> None:
    with pytest.raises(SystemExit):
        _parse(tmp_path, flag, value)
    assert flag in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@st.composite
def _solves(draw):
    """Design and calibrated household weights over a few districts.

    Calibrated weights are the design weights times a per-household factor in
    [0, 5] (the release's weight cap), so a whole district can go to zero.
    """

    n = draw(st.integers(min_value=1, max_value=80))
    n_districts = draw(st.integers(min_value=1, max_value=6))
    codes = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_districts - 1),
            min_size=n,
            max_size=n,
        )
    )
    design = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=1_000.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    factors = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    design = np.asarray(design, dtype=np.float64)
    calibrated = design * np.asarray(factors, dtype=np.float64)
    return design, calibrated, np.asarray([f"{code:04d}" for code in codes])


def _weight_origin(design, calibrated, codes) -> dict:
    """``weight_origin`` exactly as ``calibration_evidence`` records it."""

    return {
        "design": _CD.weight_origin_summary(
            design, spine=None, source_id=None, district=codes
        ),
        "calibrated": _CD.weight_origin_summary(
            calibrated, spine=None, source_id=None, district=codes
        ),
    }


_FLOORS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_ABSOLUTE_FLOORS = st.floats(min_value=0.0, max_value=200.0, allow_nan=False)


@_SETTINGS
@given(solve=_solves(), floors=st.tuples(_FLOORS, _FLOORS))
def test_collapsed_districts_are_monotone_in_the_relative_floor(solve, floors):
    low, high = sorted(floors)
    origin = _weight_origin(*solve)
    at_low = _GATE(origin, relative_floor=low)
    at_high = _GATE(origin, relative_floor=high)
    assert set(_districts(at_low["collapsed"])) <= set(_districts(at_high["collapsed"]))
    assert at_low["n_collapsed"] <= at_high["n_collapsed"]


@_SETTINGS
@given(solve=_solves(), floors=st.tuples(_ABSOLUTE_FLOORS, _ABSOLUTE_FLOORS))
def test_districts_below_the_floor_are_monotone_in_the_floor(solve, floors):
    low, high = sorted(floors)
    origin = _weight_origin(*solve)
    at_low = _GATE(origin, absolute_floor=low)
    at_high = _GATE(origin, absolute_floor=high)
    assert set(_districts(at_low["below_floor"])) <= set(
        _districts(at_high["below_floor"])
    )
    assert at_low["n_below_floor"] <= at_high["n_below_floor"]


@_SETTINGS
@given(solve=_solves(), relative_floor=_FLOORS, floor=_ABSOLUTE_FLOORS)
def test_a_design_weight_solve_collapses_no_district(solve, relative_floor, floor):
    design, _, codes = solve
    gate = _GATE(
        _weight_origin(design, design, codes),
        relative_floor=relative_floor,
        absolute_floor=floor,
    )
    assert gate["n_collapsed"] == 0
    # The absolute floor is not relative: it flags the design's own thin
    # districts, and only those.
    design_ess = _CD.ess_by_group(design, codes)
    assert set(_districts(gate["below_floor"])) == {
        district for district, ess in design_ess.items() if ess < floor
    }


@_SETTINGS
@given(
    solve=_solves(),
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    relative_floor=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
)
def test_a_uniform_rescaling_of_the_design_weights_collapses_no_district(
    solve, scale, relative_floor
):
    design, _, codes = solve
    gate = _GATE(
        _weight_origin(design, design * scale, codes), relative_floor=relative_floor
    )
    assert gate["n_collapsed"] == 0


@_SETTINGS
@given(solve=_solves(), relative_floor=_FLOORS, floor=_ABSOLUTE_FLOORS)
def test_counts_are_bounded_by_the_districts(solve, relative_floor, floor):
    _, _, codes = solve
    gate = _GATE(
        _weight_origin(*solve), relative_floor=relative_floor, absolute_floor=floor
    )
    assert gate["n_districts"] == len(set(codes.tolist()))
    assert 0 <= gate["n_collapsed"] <= gate["n_districts"]
    assert 0 <= gate["n_below_floor"] <= gate["n_districts"]
    assert gate["n_collapsed"] == len(gate["collapsed"])
    assert gate["n_below_floor"] == len(gate["below_floor"])
    assert gate["criteria_met"] is (
        gate["n_collapsed"] == 0 and gate["n_below_floor"] == 0
    )
    # The entry is what gate_summary.json stores, unchanged by a round trip.
    assert json.loads(json.dumps(gate)) == gate


@_SETTINGS
@given(solve=_solves(), relative_floor=_FLOORS, floor=_ABSOLUTE_FLOORS)
def test_blocking_changes_only_whether_the_gate_passes(solve, relative_floor, floor):
    origin = _weight_origin(*solve)
    report_only = _GATE(origin, relative_floor=relative_floor, absolute_floor=floor)
    blocking = _GATE(
        origin, relative_floor=relative_floor, absolute_floor=floor, blocking=True
    )
    switched = {"passed", "blocking", "report_only"}
    assert {k: v for k, v in report_only.items() if k not in switched} == {
        k: v for k, v in blocking.items() if k not in switched
    }
    assert report_only["passed"] is True
    assert blocking["passed"] is report_only["criteria_met"]
    assert _LIMITATION(blocking)["calibration_blocker"] is not blocking["passed"]


def _reference_ess(weights: np.ndarray, index: np.ndarray, groups: int) -> np.ndarray:
    total = np.bincount(index, weights=weights, minlength=groups)
    square = np.bincount(index, weights=weights * weights, minlength=groups)
    ess = np.zeros(groups)
    positive = square > 0
    ess[positive] = total[positive] ** 2 / square[positive]
    return ess


@_SETTINGS
@given(solve=_solves(), relative_floor=_FLOORS)
def test_the_gate_agrees_with_a_reference_from_the_raw_weights(solve, relative_floor):
    design, calibrated, codes = solve
    labels, index = np.unique(codes, return_inverse=True)
    design_ess = _reference_ess(design, index, len(labels))
    calibrated_ess = _reference_ess(calibrated, index, len(labels))
    threshold = relative_floor * design_ess
    expected = {
        str(label)
        for label, c, t in zip(labels, calibrated_ess, threshold, strict=True)
        if c < t
    }
    # Districts within rounding of the threshold may fall either way: the two
    # computations sum in a different order.
    ties = {
        str(label)
        for label, c, t in zip(labels, calibrated_ess, threshold, strict=True)
        if abs(c - t) <= 1e-9 * (1.0 + t)
    }
    gate = _GATE(
        _weight_origin(design, calibrated, codes), relative_floor=relative_floor
    )
    assert set(_districts(gate["collapsed"])) - ties == expected - ties
