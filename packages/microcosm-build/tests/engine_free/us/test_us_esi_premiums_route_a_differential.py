"""Route A's ESI premiums and gate verdicts stay main's (d2283 PR 0, d3062).

Route A reaches ``esi_premiums.py`` in two places, both with the default
``allow_transferred_rows=False``:

* the base builder (``tools/build_us_puf_support_base.py``) stages the
  premium, runs the signal gate, runs the PUF support clone and runs the
  signal gate again; and
* the release (``tools/build_us_fiscal_refresh_release.py``) runs the signal
  gate on the base, then the signal and anchor gates on the calibrated export.

On Route A-shaped frames, in one process, this module checks that the
premium column is byte-equal to main's stage expression and that every gate's
verdict is main's. Main's gates are the vendored oracle
(``test_support/microcosm_build/us_esi_premiums_main_oracle.py``). Where main
grades a frame without a structural error, the failures and every summary and
``details`` value are identical too, bit for bit. Where main refuses a frame,
only the verdict is compared: the refusal text may differ. A null raw field
is still refused with main's diagnosis (the column, its row count and the
remediation), which one test checks.

One part of the summary's all-CPS guard is out of this module's reach: on a
frame of CPS rows the summary does not subset by ``source``. A subset by an
all-True mask holds identical values in the same order, so no in-process test
can tell the two apart. That part holds by construction, as the code reads,
not by these tests.

Fixtures, by the branch the row rule takes:

* F1: no ``PERIDNUM`` column, every row complete.
* F2: ``PERIDNUM`` on every row, through the production PUF clone.
* F3: red on both: premiums off the anchor, a raw column missing, a row with
  some raw fields null.
* F4: ``PERIDNUM`` blank on a complete row. Main never reads it.
* F5: a row with neither the raw fields nor ``PERIDNUM``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime.esi_premiums import (
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    us_esi_premiums_summary,
    with_us_esi_premium_inputs,
)
from microcosm.frame import Frame
from test_support.microcosm_build import us_esi_premiums_main_oracle as main
from test_support.microcosm_build.us_esi_premiums_frames import (
    CPS_ID,
    EMPLOYER,
    RAW,
    YEAR,
    asec_arm,
    calibration_factors,
    puf_cloned,
    reweighted,
    staged,
    with_person_cells,
    with_scaled_premiums,
    without_person_columns,
)

_BLANK_IDS = [None, "", "   "]


def _identical(actual: object, expected: object, path: str) -> None:
    """Equal values of equal types; floats equal bit for bit (NaN included)."""

    assert type(actual) is type(expected), f"{path}: {type(actual)} != {type(expected)}"
    if isinstance(expected, Mapping):
        assert list(actual) == list(expected), f"{path}: keys differ"
        for key in expected:
            _identical(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: lengths differ"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _identical(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        assert np.float64(actual).tobytes() == np.float64(expected).tobytes(), (
            f"{path}: {actual!r} != {expected!r}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _mains_verdicts(frame: Frame, *, anchor: bool = True) -> list[bool]:
    """Grade ``frame`` with the live gates and main's; return main's verdicts.

    Signal gate first, then the anchor gate, as the release runs them. The
    live gates take their default, as every Route A caller does.
    """

    try:
        expected_summary = main.us_esi_premiums_summary(frame)
    except SourceRuntimeError:
        with pytest.raises(SourceRuntimeError):
            us_esi_premiums_summary(frame)
    else:
        _identical(us_esi_premiums_summary(frame), expected_summary, "summary")
    pairs = [
        (us_esi_premiums_signal_gate(frame), main.us_esi_premiums_signal_gate(frame))
    ]
    if anchor:
        pairs.append(
            (
                us_esi_premiums_anchor_gate(frame, time_period=YEAR),
                main.us_esi_premiums_anchor_gate(frame, time_period=YEAR),
            )
        )
    verdicts = []
    for live, expected in pairs:
        assert live.name == expected.name
        assert live.passed == expected.passed, (live.failures, expected.failures)
        if "structural_error" not in expected.details:
            assert live.failures == expected.failures
            _identical(dict(live.details), dict(expected.details), "details")
        verdicts.append(expected.passed)
    return verdicts


def _mains_stage_premiums(frame: Frame) -> np.ndarray:
    """Main's stage expression on the table ``with_us_esi_premium_inputs`` builds.

    ``derive_us_employer_esi_premiums_from_manifest`` at main 5499382772:
    ``_anchor(year) / float(weights @ raw)`` times each raw share in the
    employed-policyholder universe, zero elsewhere. Recomputed in this process,
    so both sides make the same BLAS call.
    """

    person = esi._person_with_state(frame)
    person[esi._PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    codes, raw, _premium, _contribution = esi._person_raw_shares(person)
    weights = esi._person_weights(person)
    scale = esi._anchor(YEAR) / float(weights @ raw)
    return np.where(codes.universe, raw * scale, 0.0)


@pytest.fixture(
    scope="module",
    params=[False, True],
    ids=["F1_no_id_column", "F2_id_on_every_row"],
)
def route_a(request) -> dict[str, object]:
    raw = asec_arm(record_ids=request.param)
    pre_clone = staged(raw)
    base = puf_cloned(pre_clone)
    return {
        "record_ids": request.param,
        "raw": raw,
        "pre_clone": pre_clone,
        "base": base,
        "export": reweighted(base, calibration_factors(base)),
    }


@pytest.fixture(scope="module")
def f2() -> dict[str, Frame]:
    pre_clone = staged(asec_arm())
    base = puf_cloned(pre_clone)
    return {
        "pre_clone": pre_clone,
        "base": base,
        "export": reweighted(base, calibration_factors(base)),
    }


def test_the_fixtures_have_route_a_shape(route_a) -> None:
    for name in ("pre_clone", "base", "export"):
        person = route_a[name].table("person")
        assert person[list(RAW)].notna().all().all()
        if route_a["record_ids"]:
            # The production PUF clone copies the CPS record id to every row.
            assert person[CPS_ID].notna().all()
        else:
            assert CPS_ID not in person.columns
    assert len(route_a["base"].table("person")) == 2 * len(
        route_a["pre_clone"].table("person")
    )


def test_the_stage_writes_mains_premium_bytes(route_a) -> None:
    live = route_a["pre_clone"].table("person")[EMPLOYER].to_numpy()
    expected = _mains_stage_premiums(route_a["raw"])

    assert live.dtype == expected.dtype == np.float64
    assert np.array_equal(live, expected)
    assert live.tobytes() == expected.tobytes()
    # The PUF clone carries each person's premium to both support rows.
    cloned = route_a["base"].table("person")[EMPLOYER].to_numpy()
    assert cloned.tobytes() == np.concatenate([expected, expected]).tobytes()


def test_a_staged_frame_is_returned_unchanged(route_a) -> None:
    # The idempotent branch grades the frame with the default signal gate.
    for name in ("pre_clone", "base"):
        frame = route_a[name]
        assert with_us_esi_premium_inputs(frame, seed=0, time_period=YEAR) is frame


def test_the_base_builder_order_gets_mains_verdicts(route_a) -> None:
    # Stage, signal gate, PUF support clone, signal gate.
    assert _mains_verdicts(route_a["pre_clone"], anchor=False) == [True]
    assert _mains_verdicts(route_a["base"], anchor=False) == [True]


def test_the_release_order_gets_mains_verdicts(route_a) -> None:
    # Signal gate on the base, then signal and anchor gates on the export.
    assert _mains_verdicts(route_a["base"], anchor=False) == [True]
    assert _mains_verdicts(route_a["export"]) == [True, True]


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda frame: with_scaled_premiums(frame, 1.2), [True, False]),
        (lambda frame: without_person_columns(frame, ["NOEMP"]), [False, False]),
        (
            lambda frame: with_person_cells(frame, [5], [RAW[0], RAW[3]], np.nan),
            [False, False],
        ),
    ],
    ids=["F3_premiums_off_the_anchor", "F3_raw_column_missing", "F3_partial_row"],
)
def test_frames_main_refuses_are_refused(route_a, edit, expected) -> None:
    assert _mains_verdicts(edit(route_a["base"]), anchor=False) == expected[:1]
    assert _mains_verdicts(edit(route_a["export"])) == expected


_NOT_RESTORED = (
    "the Census column was not restored for every vintage "
    "(asec_census_person_columns) and is never defaulted"
)


def test_a_null_raw_field_is_refused_with_mains_diagnosis(route_a) -> None:
    # The base builder prints this text in its SystemExit and the release
    # writes it to esi_premiums.json, so it keeps main's diagnosis: the column,
    # its row count and the remediation.
    for name, anchor in (("pre_clone", False), ("base", False), ("export", True)):
        frame = with_person_cells(route_a[name], [7], "NOW_HIPAID", np.nan)
        with pytest.raises(SourceRuntimeError) as refused:
            us_esi_premiums_summary(frame)
        gates = [us_esi_premiums_signal_gate(frame)]
        if anchor:
            gates.append(us_esi_premiums_anchor_gate(frame, time_period=YEAR))
        for gate in gates:
            assert not gate.passed
            assert gate.failures[0] == gate.details["structural_error"]
        for text in (str(refused.value), *(gate.failures[0] for gate in gates)):
            assert "1 row(s) lack some of the raw ASEC fields" in text
            assert "NOW_HIPAID has 1 missing row(s)" in text
            # Only the incomplete column is named.
            assert not any(column in text for column in RAW if column != "NOW_HIPAID")
            assert _NOT_RESTORED in text
            # The keyword is the ACS local lane's, not a Route A remedy.
            assert (
                "Only a frame that holds transferred ACS rows needs "
                "allow_transferred_rows=True; on any other frame, restore the "
                "fields." in text
            )
        # Main says the same of the same row.
        assert _NOT_RESTORED in main.us_esi_premiums_signal_gate(frame).failures[0]
    # Several columns: each is named with its own count, and the lead
    # count is rows, not cells (row 7 lacks both fields).
    frame = with_person_cells(
        with_person_cells(route_a["export"], [7], "NOW_HIPAID", np.nan),
        [7, 9],
        "PEMLR",
        np.nan,
    )
    text = us_esi_premiums_signal_gate(frame).failures[0]
    assert "2 row(s) lack some of the raw ASEC fields" in text
    assert "NOW_HIPAID has 1 missing row(s), PEMLR has 2 missing row(s);" in text
    assert (
        "the Census columns were not restored for every vintage "
        "(asec_census_person_columns) and are never defaulted" in text
    )


@pytest.mark.parametrize("blank", _BLANK_IDS)
def test_f4_a_blank_record_id_on_a_complete_row_keeps_mains_verdict(f2, blank) -> None:
    for name, anchor in (("pre_clone", False), ("base", False), ("export", True)):
        frame = with_person_cells(f2[name], [7], CPS_ID, blank)
        assert _mains_verdicts(frame, anchor=anchor) == [True] * (1 + anchor)
    # Grading transferred rows reads the id, and a row with the raw fields
    # but no id is neither kind.
    frame = with_person_cells(f2["export"], [7], CPS_ID, blank)
    with pytest.raises(SourceRuntimeError, match="without a CPS record id"):
        us_esi_premiums_summary(frame, allow_transferred_rows=True)


def test_f5_a_row_with_neither_raw_fields_nor_record_id_keeps_mains_verdict(
    f2,
) -> None:
    for name, anchor in (("pre_clone", False), ("base", False), ("export", True)):
        frame = with_person_cells(f2[name], [7], [*RAW, CPS_ID], None)
        assert _mains_verdicts(frame, anchor=anchor) == [False] * (1 + anchor)
        assert "structural_error" in us_esi_premiums_signal_gate(frame).details
    # Only a caller that grades transferred rows reads it as one.
    frame = with_person_cells(f2["export"], [7], [*RAW, CPS_ID], None)
    summary = us_esi_premiums_summary(frame, allow_transferred_rows=True)
    assert summary["transferred_rows"] == 1


def test_the_lane_keyword_grades_an_all_cps_frame_as_main_does(route_a) -> None:
    # Every row a CPS row: the keyword changes nothing, bit for bit.
    for name, anchor in (("pre_clone", False), ("base", False), ("export", True)):
        frame = route_a[name]
        _identical(
            us_esi_premiums_summary(frame, allow_transferred_rows=True),
            main.us_esi_premiums_summary(frame),
            "summary",
        )
        live = [us_esi_premiums_signal_gate(frame, allow_transferred_rows=True)]
        expected = [main.us_esi_premiums_signal_gate(frame)]
        if anchor:
            live.append(
                us_esi_premiums_anchor_gate(
                    frame, time_period=YEAR, allow_transferred_rows=True
                )
            )
            expected.append(main.us_esi_premiums_anchor_gate(frame, time_period=YEAR))
        for left, right in zip(live, expected, strict=True):
            assert left.passed and right.passed
            _identical(dict(left.details), dict(right.details), "details")


# --- any mix of complete, incomplete and blank-id rows ----------------------


@cache
def _small_export(record_ids: bool) -> Frame:
    base = puf_cloned(staged(asec_arm(60, record_ids=record_ids)))
    return reweighted(base, calibration_factors(base))


_ROW_EDITS = ("blank_id", "some_fields", "all_fields", "all_fields_and_id")


@st.composite
def _edited_frames(draw) -> Frame:
    record_ids = draw(st.booleans())
    frame = _small_export(record_ids)
    rows = len(frame.table("person"))
    edits = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=rows - 1),
                st.sampled_from(_ROW_EDITS),
                st.lists(st.sampled_from(RAW), min_size=1, max_size=len(RAW) - 1),
                st.sampled_from(_BLANK_IDS),
            ),
            max_size=4,
        )
    )
    for row, edit, fields, blank in edits:
        if edit == "blank_id":
            if record_ids:
                frame = with_person_cells(frame, [row], CPS_ID, blank)
        elif edit == "some_fields":
            frame = with_person_cells(frame, [row], sorted(set(fields)), np.nan)
        elif edit == "all_fields":
            frame = with_person_cells(frame, [row], list(RAW), np.nan)
        else:
            columns = [*RAW, CPS_ID] if record_ids else list(RAW)
            frame = with_person_cells(frame, [row], columns, None)
    return frame


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_edited_frames())
def test_any_mix_of_rows_gets_mains_verdict(frame: Frame) -> None:
    """Any mix of complete, partial, field-less and blank-id rows: main's verdict.

    It compares statistics too wherever main grades without a structural
    error, but it is not the evidence that they are bit-equal: a mutant that
    swapped main's ``other_policyholder_employer_total`` expression for
    #1199's passed it. The named F1/F2 tests (the base builder and release
    orders, and the lane keyword on an all-CPS frame) pin the bit-equal
    statistics.
    """

    _mains_verdicts(frame)
