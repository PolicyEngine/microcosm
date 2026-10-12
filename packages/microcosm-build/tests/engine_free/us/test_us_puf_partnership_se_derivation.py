"""The raw-PUF derivation of ``partnership_self_employment_net_earnings``.

``puf_support._person_source_values`` derives this leaf from raw PUF columns
only when the donor arrays carry neither the leaf nor the processed
``partnership_se_income`` column. The derivation ports the archived
processed-PUF pipeline (PolicyEngine/policyengine-us-data at
42ed5d45c56df80d754fbe24cce21cfeb8d05cbe, ``datasets/puf/puf.py`` lines
723-746), copied below as ``_archived_partnership_se``.

Invariants, each for every input Hypothesis draws:

1. It equals the archived formula bit for bit, on float columns with missing
   values and on the integer columns a raw PUF extract carries.
2. A record with no partnership activity (its net partnership income
   ``E25940 + E25980 - E25920 - E25960`` is exactly zero) gets exactly zero,
   whatever its Schedule SE, C and F amounts.
3. On a record with partnership activity, the result plus the Schedule C and
   Schedule F amounts, times 0.9235, returns the Schedule SE amounts.
4. No S corporation column changes it.
5. Rows are derived independently: permuting the rows permutes the result.
6. On integer-dollar amounts it is homogeneous of degree one under
   power-of-two rescaling, bit for bit.
7. It fails closed: with any of the eight raw columns absent it derives
   nothing, whatever other raw columns are present.

Plus worked examples, the source precedence, and the donor-builder seam.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.puf_support as puf_support_module
from microcosm.build.us_runtime import puf_tax_unit_donor_from_arrays

_LEAF = "partnership_self_employment_net_earnings"
_RAW_COLUMNS = (
    "E30400",
    "E30500",
    "E00900",
    "E02100",
    "E25920",
    "E25940",
    "E25960",
    "E25980",
)
_S_CORP_COLUMNS = ("E26180", "E26190")
_ARCHIVED_SE_DEDUCTION_FACTOR = 0.9235
_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
_MAX_ROWS = 12
_INTEGER_DOLLARS = st.integers(min_value=-(10**9), max_value=10**9)
# Mostly zero, like a raw PUF extract, with integer dollars, arbitrary finite
# floats and missing values mixed in.
_FLOAT_CELL = st.one_of(
    st.just(0.0),
    _INTEGER_DOLLARS.map(float),
    st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, width=64),
    st.just(float("nan")),
)


def _archived_partnership_se(puf: pd.DataFrame) -> np.ndarray:
    """Lines 734-745 of the archived ``puf.py``.

    Verbatim apart from the factor's name, which the archive binds as a local
    ``SE_DEDUCTION_FACTOR``.
    """

    taxable_se = puf["E30400"].fillna(0) + puf["E30500"].fillna(0)
    gross_se = taxable_se / _ARCHIVED_SE_DEDUCTION_FACTOR
    schedule_c_f_income = puf["E00900"].fillna(0) + puf["E02100"].fillna(0)
    # Only compute when there's partnership activity (net partnership income != 0)
    has_partnership = (
        puf["E25940"].fillna(0)
        + puf["E25980"].fillna(0)
        - puf["E25920"].fillna(0)
        - puf["E25960"].fillna(0)
    ) != 0
    partnership_se = np.where(has_partnership, gross_se - schedule_c_f_income, 0)
    return partnership_se


def _derive(arrays: dict[str, np.ndarray]) -> np.ndarray:
    values = puf_support_module._person_source_values(arrays, _LEAF)
    assert values is not None
    return values


def _assert_bit_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.float64
    assert expected.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@st.composite
def _float_arrays(draw: st.DrawFn) -> dict[str, np.ndarray]:
    rows = draw(st.integers(min_value=1, max_value=_MAX_ROWS))
    return {
        column: np.asarray(
            draw(st.lists(_FLOAT_CELL, min_size=rows, max_size=rows)),
            dtype=np.float64,
        )
        for column in _RAW_COLUMNS
    }


@st.composite
def _integer_arrays(draw: st.DrawFn) -> dict[str, np.ndarray]:
    rows = draw(st.integers(min_value=1, max_value=_MAX_ROWS))
    return {
        column: np.asarray(
            draw(st.lists(_INTEGER_DOLLARS, min_size=rows, max_size=rows)),
            dtype=np.int64,
        )
        for column in _RAW_COLUMNS
    }


@st.composite
def _arrays_without_partnership_activity(draw: st.DrawFn) -> dict[str, np.ndarray]:
    """Integer-dollar records whose net partnership income cancels exactly."""

    arrays = draw(_integer_arrays())
    arrays["E25960"] = arrays["E25940"] + arrays["E25980"] - arrays["E25920"]
    return arrays


@_SETTINGS
@given(arrays=_float_arrays())
def test_derivation_equals_the_archived_formula_on_float_columns(
    arrays: dict[str, np.ndarray],
) -> None:
    _assert_bit_identical(
        _derive(arrays), _archived_partnership_se(pd.DataFrame(arrays))
    )


@_SETTINGS
@given(arrays=_integer_arrays())
def test_derivation_equals_the_archived_formula_on_integer_columns(
    arrays: dict[str, np.ndarray],
) -> None:
    _assert_bit_identical(
        _derive(arrays), _archived_partnership_se(pd.DataFrame(arrays))
    )


@_SETTINGS
@given(arrays=_arrays_without_partnership_activity())
def test_no_partnership_activity_gives_exactly_zero(
    arrays: dict[str, np.ndarray],
) -> None:
    derived = _derive(arrays)

    assert derived.tolist() == [0.0] * len(derived)
    assert not np.signbit(derived).any()


@_SETTINGS
@given(arrays=_integer_arrays())
def test_active_records_return_the_schedule_se_amounts(
    arrays: dict[str, np.ndarray],
) -> None:
    derived = _derive(arrays)
    active = (
        arrays["E25940"] + arrays["E25980"] - arrays["E25920"] - arrays["E25960"]
    ) != 0

    assert (derived[~active] == 0.0).all()
    np.testing.assert_allclose(
        (derived + arrays["E00900"] + arrays["E02100"])[active]
        * _ARCHIVED_SE_DEDUCTION_FACTOR,
        (arrays["E30400"] + arrays["E30500"])[active],
        rtol=1e-12,
        atol=1e-5,
    )


@_SETTINGS
@given(arrays=_float_arrays(), data=st.data())
def test_s_corporation_columns_never_change_the_result(
    arrays: dict[str, np.ndarray],
    data: st.DataObject,
) -> None:
    rows = len(arrays["E30400"])
    with_s_corp = dict(arrays)
    for column in _S_CORP_COLUMNS:
        with_s_corp[column] = np.asarray(
            data.draw(st.lists(_FLOAT_CELL, min_size=rows, max_size=rows)),
            dtype=np.float64,
        )

    _assert_bit_identical(_derive(with_s_corp), _derive(arrays))


@_SETTINGS
@given(arrays=_float_arrays(), data=st.data())
def test_rows_are_derived_independently(
    arrays: dict[str, np.ndarray],
    data: st.DataObject,
) -> None:
    rows = len(arrays["E30400"])
    order = np.asarray(data.draw(st.permutations(range(rows))), dtype=np.intp)
    permuted = {column: values[order] for column, values in arrays.items()}

    _assert_bit_identical(_derive(permuted), _derive(arrays)[order])


@_SETTINGS
@given(arrays=_integer_arrays(), exponent=st.integers(min_value=-8, max_value=8))
def test_derivation_is_homogeneous_under_power_of_two_rescaling(
    arrays: dict[str, np.ndarray],
    exponent: int,
) -> None:
    # Integer dollars keep every intermediate in the normal float range, where
    # a power-of-two rescaling commutes exactly with each rounding.
    scale = 2.0**exponent
    scaled = {column: values * scale for column, values in arrays.items()}

    _assert_bit_identical(_derive(scaled), _derive(arrays) * scale)


@_SETTINGS
@given(
    arrays=_float_arrays(),
    present=st.sets(st.sampled_from(_RAW_COLUMNS), max_size=len(_RAW_COLUMNS) - 1),
    s_corp=st.sets(st.sampled_from(_S_CORP_COLUMNS)),
)
def test_incomplete_raw_columns_derive_nothing(
    arrays: dict[str, np.ndarray],
    present: set[str],
    s_corp: set[str],
) -> None:
    rows = len(arrays["E30400"])
    partial = {column: arrays[column] for column in present}
    partial.update({column: np.ones(rows) for column in s_corp})

    assert puf_support_module._person_source_values(partial, _LEAF) is None


def test_derivation_does_not_mutate_its_inputs_and_is_deterministic() -> None:
    rng = np.random.default_rng(20261011)
    arrays = {
        column: rng.integers(-5_000, 5_000, size=50).astype(np.float64)
        for column in _RAW_COLUMNS
    }
    before = {column: values.copy() for column, values in arrays.items()}

    first = _derive(arrays)
    second = _derive(arrays)

    _assert_bit_identical(first, second)
    for column, values in arrays.items():
        np.testing.assert_array_equal(values, before[column])


def test_worked_examples_match_hand_arithmetic() -> None:
    arrays = {
        # (E30400 + E30500) / 0.9235 = 1,000; 2,000; 1,000; 1,000; 0; 500.
        "E30400": [923.5, 0.0, 923.5, 923.5, 0.0, 461.75],
        "E30500": [0.0, 1_847.0, 0.0, 0.0, 0.0, 0.0],
        # Schedule C plus Schedule F = 400; 300; 1,500; 400; 250; 0.
        "E00900": [300.0, 500.0, 1_500.0, 300.0, 250.0, 0.0],
        "E02100": [100.0, -200.0, 0.0, 100.0, 0.0, 0.0],
        # Net partnership income = 45; 60; -30; 0 (cancelling); 80; 0 (absent).
        "E25940": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        "E25980": [50.0, 70.0, 0.0, 50.0, 80.0, 0.0],
        "E25920": [0.0, 0.0, 30.0, 55.0, 0.0, 0.0],
        "E25960": [5.0, 10.0, 0.0, 5.0, 0.0, 0.0],
    }

    derived = _derive(arrays)

    np.testing.assert_allclose(
        derived,
        [
            600.0,  # 1,000 - 400
            1_700.0,  # 2,000 - 300
            -500.0,  # 1,000 - 1,500: Schedule C exceeds Schedule SE earnings
            0.0,  # income and losses cancel: no partnership activity
            -250.0,  # 0 - 250: the residual goes negative the same way
            0.0,  # no partnership columns at all, so not the residual 500
        ],
        rtol=0.0,
        atol=1e-9,
    )
    assert derived[3] == 0.0
    assert derived[5] == 0.0


def test_derivation_is_not_the_sum_of_partnership_and_s_corporation_losses() -> None:
    """The replaced fallback returned ``E25960 + E26180``."""

    arrays = {column: np.zeros(2) for column in _RAW_COLUMNS}
    arrays["E25960"] = np.asarray([5.0, 10.0])
    arrays["E26180"] = np.asarray([20.0, 30.0])

    derived = _derive(arrays)

    # A partnership loss with no Schedule SE earnings is no SE earnings, and an
    # S corporation loss never enters.
    assert derived.tolist() == [0.0, 0.0]


def test_missing_values_count_as_zero_like_the_archived_fillna() -> None:
    arrays = {column: np.zeros(1) for column in _RAW_COLUMNS}
    arrays["E30400"] = np.asarray([np.nan])
    arrays["E30500"] = np.asarray([923.5])
    arrays["E00900"] = np.asarray([np.nan])
    arrays["E25980"] = np.asarray([40.0])
    arrays["E25920"] = np.asarray([np.nan])

    np.testing.assert_allclose(_derive(arrays), [1_000.0], rtol=0.0, atol=1e-9)


def test_stored_leaf_and_processed_column_take_precedence_over_raw_columns() -> None:
    raw = {column: np.asarray([1.0, 2.0]) for column in _RAW_COLUMNS}

    processed = puf_support_module._person_source_values(
        {**raw, "partnership_se_income": [100.0, -200.0]}, _LEAF
    )
    stored = puf_support_module._person_source_values(
        {**raw, "partnership_se_income": [100.0, -200.0], _LEAF: [7.0, 8.0]},
        _LEAF,
    )

    assert processed.tolist() == [100.0, -200.0]
    assert stored.tolist() == [7.0, 8.0]


def _donor_base() -> dict[str, object]:
    return {
        "tax_unit_id": [10, 20, 30],
        "household_weight": [100.0, 200.0, 300.0],
        "filing_status": [b"SINGLE", b"JOINT", b"SINGLE"],
        "person_tax_unit_id": [10, 20, 30],
    }


def _donor_raw_columns() -> dict[str, list[float]]:
    return {
        "E30400": [923.5, 0.0, 923.5],
        "E30500": [0.0, 1_847.0, 0.0],
        "E00900": [300.0, 500.0, 300.0],
        "E02100": [100.0, -200.0, 100.0],
        "E25920": [0.0, 0.0, 0.0],
        "E25940": [0.0, 0.0, 0.0],
        "E25960": [5.0, 10.0, 0.0],
        "E25980": [50.0, 70.0, 0.0],
    }


def test_donor_builder_carries_the_archived_derivation_from_raw_columns() -> None:
    raw = _donor_raw_columns()

    donor = puf_tax_unit_donor_from_arrays(
        {**_donor_base(), **raw},
        person_outputs=(_LEAF,),
        tax_unit_outputs=(),
    )

    np.testing.assert_allclose(
        donor[_LEAF].to_numpy(), [600.0, 1_700.0, 0.0], rtol=0.0, atol=1e-9
    )
    _assert_bit_identical(
        donor[_LEAF].to_numpy(), _archived_partnership_se(pd.DataFrame(raw))
    )


@pytest.mark.parametrize("missing", _RAW_COLUMNS)
def test_donor_builder_fails_closed_when_a_raw_column_is_absent(missing: str) -> None:
    raw = _donor_raw_columns()
    del raw[missing]

    with pytest.raises(ValueError, match="cannot derive requested output") as error:
        puf_tax_unit_donor_from_arrays(
            {**_donor_base(), **raw},
            person_outputs=(_LEAF,),
            tax_unit_outputs=(),
        )

    assert _LEAF in str(error.value)


def test_donor_builder_refuses_the_replaced_loss_column_pair() -> None:
    with pytest.raises(ValueError, match="cannot derive requested output") as error:
        puf_tax_unit_donor_from_arrays(
            {
                **_donor_base(),
                "E25960": [5.0, 10.0, 0.0],
                "E26180": [20.0, 30.0, 0.0],
            },
            person_outputs=(_LEAF,),
            tax_unit_outputs=(),
        )

    assert _LEAF in str(error.value)


def test_no_pair_of_raw_columns_derives_the_leaf() -> None:
    columns = (*_RAW_COLUMNS, *_S_CORP_COLUMNS)

    for pair in combinations(columns, 2):
        arrays = {column: np.ones(2) for column in pair}
        assert puf_support_module._person_source_values(arrays, _LEAF) is None


def test_raw_column_contract_names_the_archived_inputs() -> None:
    assert set(puf_support_module._PUF_PARTNERSHIP_SE_RAW_COLUMNS) == set(_RAW_COLUMNS)
    assert not set(_S_CORP_COLUMNS) & set(
        puf_support_module._PUF_PARTNERSHIP_SE_RAW_COLUMNS
    )
    assert (
        puf_support_module._PUF_SE_NET_EARNINGS_FACTOR == _ARCHIVED_SE_DEDUCTION_FACTOR
    )
