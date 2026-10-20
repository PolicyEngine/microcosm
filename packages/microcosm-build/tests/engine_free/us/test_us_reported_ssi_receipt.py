"""Reported SSI receipt (``receives_ssi``) from CPS ASEC ``SSI_VAL``/``RESNSSI``.

Invariants under test, for every input:

* a person is marked only if they report a positive ``SSI_VAL``;
* a person whose only reported reasons are "on behalf of a disabled (blind)
  child" is never marked;
* every other positive reporter is marked (own reason or no reason recorded);
* adding an own reason never unmarks a person;
* the derivation is idempotent and leaves an existing column untouched; and
* the release consistency check fails exactly when a marked person lacks a
  positive observed ``SSI_VAL`` or the SSI take-up anchor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.cps_carried import (
    CPS_CARRIED_PERSON_INPUTS,
    CPS_REPORTED_SSI_AMOUNT_RAW_COLUMN,
    CPS_REPORTED_SSI_ON_BEHALF_OF_CHILD_REASON_CODES,
    CPS_REPORTED_SSI_REASON_RAW_COLUMNS,
    derive_us_cps_carried_inputs,
    reported_ssi_receipt,
)
from microcosm.build.us_runtime.release_input_coverage import (
    us_reported_ssi_receipt_consistency_failures,
)
from microcosm.build.us_runtime.ssi_take_up import US_SSI_TAKE_UP_ANCHOR
from microcosm.frame import Frame
from test_support.microcosm_build.us_puf_support import _raw_asec_predictor_frame

_REASON_CODES = st.integers(min_value=0, max_value=5)
_AMOUNTS = st.one_of(
    st.just(0.0),
    st.just(float("nan")),
    st.floats(min_value=-500.0, max_value=60_000.0),
)


def _person(amounts: list[float], first: list[int], second: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            US_SSI_TAKE_UP_ANCHOR: amounts,
            CPS_REPORTED_SSI_REASON_RAW_COLUMNS[0]: first,
            CPS_REPORTED_SSI_REASON_RAW_COLUMNS[1]: second,
        }
    )


def _expected(amounts: list[float], first: list[int], second: list[int]) -> list[bool]:
    # Independent oracle: a payee is someone whose recorded (nonzero) reasons
    # are all child-payee codes.
    payee = CPS_REPORTED_SSI_ON_BEHALF_OF_CHILD_REASON_CODES
    expected = []
    for amount, one, two in zip(amounts, first, second, strict=True):
        recorded = {reason for reason in (one, two) if reason != 0}
        payee_only = bool(recorded) and recorded <= payee
        expected.append(bool(np.nan_to_num(amount) > 0) and not payee_only)
    return expected


def test_reason_codes_follow_the_asec_data_dictionary() -> None:
    assert CPS_REPORTED_SSI_REASON_RAW_COLUMNS == ("RESNSSI1", "RESNSSI2")
    assert frozenset({3, 4}) == CPS_REPORTED_SSI_ON_BEHALF_OF_CHILD_REASON_CODES
    # The receipt reads the take-up stage's anchor column, so every marked
    # person is an anchored claimant.
    assert CPS_REPORTED_SSI_AMOUNT_RAW_COLUMN == US_SSI_TAKE_UP_ANCHOR == "SSI_VAL"
    assert "receives_ssi" in CPS_CARRIED_PERSON_INPUTS


@pytest.mark.parametrize(
    ("amount", "first", "second", "marked"),
    [
        (0.0, 0, 0, False),  # no SSI reported
        (9_600.0, 1, 0, True),  # disabled
        (9_600.0, 2, 0, True),  # blind
        (9_600.0, 5, 0, True),  # other (typically aged)
        (9_600.0, 0, 0, True),  # positive amount, reason not recorded
        (9_600.0, 3, 0, False),  # only on behalf of a disabled child
        (9_600.0, 4, 0, False),  # only on behalf of a blind child
        (9_600.0, 3, 4, False),  # both child-payee reasons
        (9_600.0, 0, 3, False),  # child-payee reason recorded second only
        (9_600.0, 0, 1, True),  # own reason recorded second only
        (9_600.0, 3, 1, True),  # payee who also receives SSI in own right
        (9_600.0, 1, 3, True),
        (-50.0, 1, 0, False),  # nonpositive amounts never mark
    ],
)
def test_reported_ssi_receipt_examples(
    amount: float, first: int, second: int, marked: bool
) -> None:
    assert reported_ssi_receipt(_person([amount], [first], [second])).tolist() == [
        marked
    ]


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(_AMOUNTS, _REASON_CODES, _REASON_CODES), max_size=40))
def test_reported_ssi_receipt_matches_its_definition(rows) -> None:
    amounts = [row[0] for row in rows]
    first = [row[1] for row in rows]
    second = [row[2] for row in rows]
    result = reported_ssi_receipt(_person(amounts, first, second))

    assert result.dtype == bool
    assert result.tolist() == _expected(amounts, first, second)
    # A marked person always reports a positive amount.
    assert not np.any(result & ~(np.nan_to_num(np.asarray(amounts, float)) > 0))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(_AMOUNTS, _REASON_CODES, _REASON_CODES), min_size=1, max_size=20
    ),
    st.sampled_from([1, 2, 5]),
)
def test_adding_an_own_reason_never_unmarks(rows, own_reason: int) -> None:
    amounts = [row[0] for row in rows]
    first = [row[1] for row in rows]
    second = [row[2] for row in rows]
    before = reported_ssi_receipt(_person(amounts, first, second))
    after = reported_ssi_receipt(_person(amounts, first, [own_reason] * len(rows)))

    assert not np.any(before & ~after)
    assert after.tolist() == [bool(np.nan_to_num(amount) > 0) for amount in amounts]


def test_missing_reasons_refuse_only_when_an_amount_is_positive() -> None:
    reported = pd.DataFrame({US_SSI_TAKE_UP_ANCHOR: [0.0, 1_200.0]})
    with pytest.raises(ValueError, match="RESNSSI1"):
        reported_ssi_receipt(reported)

    nobody = pd.DataFrame({US_SSI_TAKE_UP_ANCHOR: [0.0, 0.0]})
    assert reported_ssi_receipt(nobody).tolist() == [False, False]
    assert reported_ssi_receipt(pd.DataFrame(index=range(2))).tolist() == [
        False,
        False,
    ]


def _with_person_columns(frame: Frame, columns: dict[str, list]) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for name, values in columns.items():
        tables["person"][name] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def test_cps_carried_derives_receives_ssi_on_person() -> None:
    frame = _with_person_columns(
        _raw_asec_predictor_frame(),
        {
            "SSI_VAL": [0.0, 9_600.0, 7_200.0],
            "RESNSSI1": [0, 1, 3],
            "RESNSSI2": [0, 0, 0],
        },
    )
    person = derive_us_cps_carried_inputs(frame).table("person")

    assert person["receives_ssi"].tolist() == [False, True, False]
    assert pd.api.types.is_bool_dtype(person["receives_ssi"])


def test_cps_carried_receives_ssi_is_idempotent() -> None:
    frame = _with_person_columns(
        _raw_asec_predictor_frame(),
        {
            "SSI_VAL": [0.0, 9_600.0, 7_200.0],
            "RESNSSI1": [0, 1, 3],
            "RESNSSI2": [0, 0, 0],
        },
    )
    once = derive_us_cps_carried_inputs(frame)
    twice = derive_us_cps_carried_inputs(once)
    pd.testing.assert_series_equal(
        once.table("person")["receives_ssi"], twice.table("person")["receives_ssi"]
    )

    preset = _with_person_columns(frame, {"receives_ssi": [True, False, False]})
    kept = derive_us_cps_carried_inputs(preset).table("person")["receives_ssi"]
    assert kept.tolist() == [True, False, False]


def test_cps_carried_fixture_without_ssi_columns_marks_nobody() -> None:
    person = derive_us_cps_carried_inputs(_raw_asec_predictor_frame()).table("person")
    assert person["receives_ssi"].tolist() == [False, False, False]


def _consistency_person(receipt, ssi_val, take_up) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "receives_ssi": np.asarray(receipt, dtype=bool),
            "SSI_VAL": np.asarray(ssi_val, dtype=np.float64),
            "takes_up_ssi_if_eligible": np.asarray(take_up, dtype=bool),
        }
    )


def test_consistency_check_skips_a_table_without_receipt() -> None:
    failures, details = us_reported_ssi_receipt_consistency_failures(
        pd.DataFrame({"SSI_VAL": [1.0]})
    )
    assert failures == ()
    assert details == {"evaluated": False}


def test_consistency_check_judges_acs_rows_by_their_native_amount() -> None:
    # A stacked ACS row carries ssi_reported instead of SSI_VAL. Its amount
    # arm uses ssi_reported; it sits outside the CPS take-up anchor, so its
    # take-up is counted but not judged.
    person = pd.DataFrame(
        {
            "receives_ssi": [True, True, True],
            "SSI_VAL": [9_600.0, np.nan, np.nan],
            "ssi_reported": [np.nan, 8_400.0, 0.0],
            "takes_up_ssi_if_eligible": [True, False, True],
        }
    )
    failures, details = us_reported_ssi_receipt_consistency_failures(person)
    assert details["receipt_rows_without_positive_reported_amount"] == 1
    assert details["receipt_rows_without_take_up"] == 0
    assert details["receipt_rows_outside_take_up_anchor"] == 2
    assert details["receipt_rows_outside_take_up_anchor_without_take_up"] == 1
    assert len(failures) == 1
    assert "without a positive reported SSI amount" in failures[0]

    consistent = person.assign(ssi_reported=[np.nan, 8_400.0, 1.0])
    assert us_reported_ssi_receipt_consistency_failures(consistent)[0] == ()


def test_consistency_check_fails_receipt_without_any_amount_column() -> None:
    failures, details = us_reported_ssi_receipt_consistency_failures(
        pd.DataFrame(
            {"receives_ssi": [True, False], "takes_up_ssi_if_eligible": [True, True]}
        )
    )
    assert details["receipt_rows_without_positive_reported_amount"] == 1
    assert len(failures) == 1 and "neither SSI_VAL nor ACS ssi_reported" in failures[0]

    nobody = us_reported_ssi_receipt_consistency_failures(
        pd.DataFrame({"receives_ssi": [False, False]})
    )
    assert nobody[0] == ()


def test_consistency_check_fails_a_marked_row_with_no_observed_amount() -> None:
    failures, details = us_reported_ssi_receipt_consistency_failures(
        _consistency_person([True], [np.nan], [True])
    )
    assert details["receipt_rows_without_positive_reported_amount"] == 1
    assert failures != ()


def test_consistency_check_rejects_incomplete_or_nonboolean_receipt() -> None:
    with pytest.raises(ValueError, match="complete"):
        us_reported_ssi_receipt_consistency_failures(
            pd.DataFrame({"receives_ssi": [True, None]})
        )
    with pytest.raises(ValueError, match="boolean"):
        us_reported_ssi_receipt_consistency_failures(
            pd.DataFrame({"receives_ssi": [2, 0]})
        )


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.booleans(),
            st.one_of(st.just(np.nan), st.floats(-10.0, 20_000.0)),
            st.booleans(),
        ),
        max_size=30,
    )
)
def test_consistency_check_fails_exactly_on_violations(rows) -> None:
    receipt = np.asarray([row[0] for row in rows], dtype=bool)
    ssi_val = np.asarray([row[1] for row in rows], dtype=np.float64)
    take_up = np.asarray([row[2] for row in rows], dtype=bool)
    failures, details = us_reported_ssi_receipt_consistency_failures(
        _consistency_person(receipt, ssi_val, take_up)
    )

    observed = ~np.isnan(ssi_val)
    without_amount = int(np.sum(receipt & ~(np.nan_to_num(ssi_val) > 0)))
    without_take_up = int(np.sum(receipt & observed & ~take_up))
    assert details["receipt_rows"] == int(receipt.sum())
    assert details["receipt_rows_without_positive_reported_amount"] == without_amount
    assert details["receipt_rows_without_take_up"] == without_take_up
    assert details["receipt_rows_outside_take_up_anchor"] == int(
        np.sum(receipt & ~observed)
    )
    assert details["receipt_rows_outside_take_up_anchor_without_take_up"] == int(
        np.sum(receipt & ~observed & ~take_up)
    )
    assert (failures == ()) == (without_amount == 0 and without_take_up == 0)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(_AMOUNTS, _REASON_CODES, _REASON_CODES), max_size=30),
)
def test_derived_receipt_always_passes_the_consistency_check(rows) -> None:
    # Round trip: receipt derived from SSI_VAL, with the take-up stage's anchor
    # (every positive SSI_VAL reporter takes up), is always consistent.
    amounts = [row[0] for row in rows]
    person = _person(amounts, [row[1] for row in rows], [row[2] for row in rows])
    person["receives_ssi"] = reported_ssi_receipt(person)
    person["takes_up_ssi_if_eligible"] = np.nan_to_num(np.asarray(amounts, float)) > 0
    failures, _ = us_reported_ssi_receipt_consistency_failures(person)
    assert failures == ()
