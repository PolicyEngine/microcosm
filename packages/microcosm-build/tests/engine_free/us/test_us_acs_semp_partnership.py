"""Property and example tests for the SEMP-covered partnership rule (#1387).

ACS SEMP includes own-partnership income and excludes incorporated-business
profit. On the ACS local lane, a donor ``partnership_income`` draw that SEMP
already accounts for is kept at zero; ``s_corp_income`` is never touched.

Invariants of :func:`keep_semp_covered_partnership_income`, for every person
table and imputation mask, checked against an oracle written here:

* **Exact cells.** A ``partnership_income`` cell becomes 0.0 exactly when the
  transfer filled it, did not draw the row's self-employment total, and the
  row reports a finite nonzero SEMP, is a HEAD or SPOUSE of a tax unit whose
  HEAD or SPOUSE reports one, or has blank SEMP.
* **Nothing else moves.** Every other cell and column, ``s_corp_income``
  included, keeps its value.
* **Idempotent** and **sign-symmetric**: negating SEMP changes nothing but
  the positive and negative receipt counts.
* **Receipt.** The counts equal the cells and nonzero draws in each set, and
  the dollar sums equal the discarded draws' positive and negative parts.
* **Roles.** A role outside HEAD/SPOUSE/DEPENDENT (after decoding bytes and
  folding case), or a missing role on a row with a tax unit, is refused.

Invariants of the extended :func:`acs_self_employment_identity_gate`:

* Its line-32 counts equal the oracle's, and it passes only when they are 0.
* Unselected rows never change the verdict, even in a shared tax unit and
  whether they come before or after the selected rows (the pooled staging
  frame puts ACS rows after the base's).

The composition tests run the real transfer: the lane's output passes the
gate, the same transfer off the lane fails it on exactly the covered rows,
the two differ only in ``partnership_income`` and the passive subset and
fit the same models with the same seeds, and a step moved after the passive
reconciliation fails the gate's passive clause.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.acs_multispine as acs_multispine_module
import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime.acs_inputs import acs_self_employment_identity_gate
from microcosm.build.us_runtime.acs_transfer import (
    ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS,
    ACS_SEMP_COVERED_LINE_32_INPUTS,
    acs_transfer_execution_contract_identity,
    declared_acs_transfer_target_families,
    keep_semp_covered_partnership_income,
    semp_covered_partnership_cells,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

_P = "partnership_income"
_S = "s_corp_income"
_PASSIVE = "passive_partnership_s_corp_income"
_TOTAL = "self_employment_income_before_lsr"
_UNIT = "person_tax_unit_id"
_ROLE = "tax_unit_role_input"
_SETS = ("own_semp", "tax_unit", "blank_semp")


def _decode(value: object) -> str:
    text = value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
    return text.upper()


def _oracle_sets(person: pd.DataFrame) -> dict[str, np.ndarray]:
    """The three covered row sets, row by row, without the module's code."""

    semp = pd.to_numeric(person["SEMP"], errors="coerce").to_numpy(dtype=float)
    roles = [_decode(value) for value in person[_ROLE]]
    units = person[_UNIT].tolist()
    head_or_spouse = np.asarray([role in ("HEAD", "SPOUSE") for role in roles])
    own = np.isfinite(semp) & (semp != 0.0)
    blank = np.isnan(semp)
    reporting_units = {
        unit
        for unit, reports, member in zip(units, own, head_or_spouse, strict=True)
        if reports and member and not pd.isna(unit)
    }
    in_reporting_unit = np.asarray(
        [not pd.isna(unit) and unit in reporting_units for unit in units]
    )
    return {
        "own_semp": own,
        "tax_unit": head_or_spouse & in_reporting_unit & ~own & ~blank,
        "blank_semp": blank,
    }


_semp = st.one_of(
    st.just(np.nan),
    st.just(0.0),
    st.integers(min_value=-30_000, max_value=-1).map(float),
    st.integers(min_value=1, max_value=200_000).map(float),
)
_amount = st.one_of(
    st.just(0.0),
    st.floats(min_value=-1e6, max_value=1e7, allow_nan=False, width=64),
)
_role = st.sampled_from(
    ["HEAD", "SPOUSE", "DEPENDENT", b"HEAD", b"SPOUSE", "spouse", b"dependent"]
)
# Mostly small ids so rows share tax units; NaN is a row without one.
_unit = st.one_of(st.integers(min_value=1, max_value=4).map(float), st.just(np.nan))
# Whole dollars, so a tax unit's line-32 sum is exact in any order and the
# oracle's zero test agrees with the gate's. Zeros, a cancelling pair and NaN
# are drawn often, so the passive clause (person and tax-unit line 32) and the
# NaN-as-zero reading are reached.
_dollars = st.one_of(
    st.just(0.0),
    st.just(np.nan),
    st.sampled_from([-900.0, 900.0]),
    st.integers(min_value=-1_000_000, max_value=10_000_000).map(float),
)


@st.composite
def _drawn_tables(draw):
    rows = draw(st.integers(min_value=1, max_value=10))
    person = pd.DataFrame(
        {
            "SEMP": draw(st.lists(_semp, min_size=rows, max_size=rows)),
            _UNIT: draw(st.lists(_unit, min_size=rows, max_size=rows)),
            _ROLE: draw(st.lists(_role, min_size=rows, max_size=rows)),
            _P: draw(st.lists(_amount, min_size=rows, max_size=rows)),
            _S: draw(st.lists(_amount, min_size=rows, max_size=rows)),
            "other": draw(st.lists(_amount, min_size=rows, max_size=rows)),
        }
    )
    # A recipient's index need not start at zero or run in order; the masks
    # are positional either way.
    start = draw(st.integers(min_value=0, max_value=50))
    index = np.arange(start, start + rows)
    person.index = index[::-1] if draw(st.booleans()) else index
    masks: dict[tuple[str, str], np.ndarray] = {
        ("person", _P): np.asarray(
            draw(st.lists(st.booleans(), min_size=rows, max_size=rows)), dtype=bool
        ),
    }
    if draw(st.booleans()):
        masks[("person", _TOTAL)] = np.asarray(
            draw(st.lists(st.booleans(), min_size=rows, max_size=rows)), dtype=bool
        )
    return person, masks


@given(_drawn_tables())
@settings(max_examples=400, deadline=None)
def test_the_step_zeroes_exactly_the_covered_drawn_cells(case) -> None:
    before, masks = case
    after = before.copy(deep=True)
    counts = keep_semp_covered_partnership_income(after, imputed_masks=masks)

    rows = len(before)
    drawn = masks[("person", _P)]
    if not drawn.any():
        assert counts is None
        pd.testing.assert_frame_equal(after, before)
        return
    total_drawn = masks.get(("person", _TOTAL), np.zeros(rows, dtype=bool))
    sets = _oracle_sets(before)
    mutable = drawn & ~total_drawn
    covered = mutable & (sets["own_semp"] | sets["tax_unit"] | sets["blank_semp"])

    assert (after.loc[covered, _P] == 0.0).all()
    pd.testing.assert_series_equal(after.loc[~covered, _P], before.loc[~covered, _P])
    pd.testing.assert_frame_equal(after.drop(columns=[_P]), before.drop(columns=[_P]))

    nonzero = before[_P].to_numpy() != 0.0
    assert counts is not None
    assert counts["cells_kept_at_zero"] == int(covered.sum())
    assert counts["nonzero_draws_discarded"] == int((covered & nonzero).sum())
    for name in _SETS:
        cleared = mutable & sets[name]
        assert counts[f"{name}_cells"] == int(cleared.sum())
        assert counts[f"{name}_nonzero_draws_discarded"] == int(
            (cleared & nonzero).sum()
        )
    semp = before["SEMP"].to_numpy(dtype=float)
    own = mutable & sets["own_semp"]
    assert counts["own_semp_positive_cells"] == int((own & (semp > 0.0)).sum())
    assert counts["own_semp_negative_cells"] == int((own & (semp < 0.0)).sum())
    amounts = before[_P].tolist()
    for prefix, rows_in in [("", covered)] + [
        (f"{name}_", mutable & sets[name]) for name in _SETS
    ]:
        kept = [value for value, row in zip(amounts, rows_in, strict=True) if row]
        assert (
            abs(
                counts[f"{prefix}discarded_positive_dollars"]
                - sum(value for value in kept if value > 0)
            )
            <= 1
        )
        assert (
            abs(
                counts[f"{prefix}discarded_negative_dollars"]
                - sum(value for value in kept if value < 0)
            )
            <= 1
        )
    assert not any("weighted" in key for key in counts)

    again = after.copy(deep=True)
    keep_semp_covered_partnership_income(again, imputed_masks=masks)
    pd.testing.assert_frame_equal(again, after)


@given(_drawn_tables())
@settings(max_examples=200, deadline=None)
def test_the_step_is_symmetric_in_the_sign_of_semp(case) -> None:
    before, masks = case
    flipped = before.copy(deep=True)
    flipped["SEMP"] = -flipped["SEMP"]
    kept = before.copy(deep=True)

    counts = keep_semp_covered_partnership_income(kept, imputed_masks=masks)
    flipped_counts = keep_semp_covered_partnership_income(flipped, imputed_masks=masks)

    pd.testing.assert_series_equal(flipped[_P], kept[_P])
    if counts is None:
        assert flipped_counts is None
        return
    assert flipped_counts is not None
    swap = {
        "own_semp_positive_cells": "own_semp_negative_cells",
        "own_semp_negative_cells": "own_semp_positive_cells",
    }
    assert {swap.get(key, key): value for key, value in counts.items()} == (
        flipped_counts
    )


@given(_drawn_tables())
@settings(max_examples=200, deadline=None)
def test_the_covered_sets_are_disjoint_and_match_the_oracle(case) -> None:
    person, _masks = case

    sets = semp_covered_partnership_cells(person)
    expected = _oracle_sets(person)

    for name in _SETS:
        np.testing.assert_array_equal(sets[name], expected[name])
    for left, right in itertools.combinations(_SETS, 2):
        assert not (sets[left] & sets[right]).any()


def test_a_dependent_reporter_extends_to_no_one_and_a_unitless_row_to_no_one() -> None:
    # Rows 2 and 3 have no tax unit; row 2 reports SEMP, so grouping unitless
    # rows together would wrongly extend it to row 3.
    person = pd.DataFrame(
        {
            "SEMP": [0.0, 3_000.0, 3_000.0, 0.0],
            _UNIT: [40.0, 40.0, np.nan, np.nan],
            _ROLE: ["HEAD", "DEPENDENT", "HEAD", "SPOUSE"],
        }
    )

    sets = semp_covered_partnership_cells(person)

    assert sets["own_semp"].tolist() == [False, True, True, False]
    assert sets["tax_unit"].tolist() == [False, False, False, False]


def test_case_and_bytes_fold_but_an_unknown_role_is_refused() -> None:
    person = pd.DataFrame(
        {"SEMP": [5_000.0, 0.0], _UNIT: [1, 1], _ROLE: [b"head", "Spouse"]}
    )
    assert semp_covered_partnership_cells(person)["tax_unit"].tolist() == [
        False,
        True,
    ]

    for bad in ("SPOUSE ", "OTHER", 1, None):
        malformed = person.astype({_ROLE: object})
        malformed.loc[1, _ROLE] = bad
        with pytest.raises(ValueError, match="neither head nor spouse"):
            semp_covered_partnership_cells(malformed)


def test_a_missing_role_is_allowed_only_without_a_tax_unit() -> None:
    person = pd.DataFrame(
        {"SEMP": [5_000.0, 0.0], _UNIT: [1.0, np.nan], _ROLE: ["HEAD", None]}
    )

    assert semp_covered_partnership_cells(person)["tax_unit"].tolist() == [
        False,
        False,
    ]


def test_the_step_does_nothing_when_no_partnership_income_was_drawn() -> None:
    person = pd.DataFrame(
        {"SEMP": [500.0], _UNIT: [1], _ROLE: ["HEAD"], _P: [700.0], _S: [9.0]}
    )

    assert keep_semp_covered_partnership_income(person, imputed_masks={}) is None
    # The state the transfer makes: a mask for the planned target, all False.
    # Nothing was drawn, so even a recipient without SEMP is not refused.
    nothing = {("person", _P): np.asarray([False])}
    assert keep_semp_covered_partnership_income(person, imputed_masks=nothing) is None
    assert (
        keep_semp_covered_partnership_income(
            person.drop(columns=["SEMP"]), imputed_masks=nothing
        )
        is None
    )
    assert person[_P].tolist() == [700.0]


@pytest.mark.parametrize(
    "masks",
    (
        {("person", _P): np.asarray([True, False])},
        {("person", _P): np.asarray([True]), ("person", _TOTAL): np.ones(2, bool)},
    ),
)
def test_the_step_refuses_masks_not_aligned_to_person(masks) -> None:
    person = pd.DataFrame({"SEMP": [500.0], _UNIT: [1], _ROLE: ["HEAD"], _P: [7.0]})

    with pytest.raises(ValueError, match="aligned to person"):
        keep_semp_covered_partnership_income(person, imputed_masks=masks)


def test_the_receipt_weights_the_discarded_draws() -> None:
    person = pd.DataFrame(
        {
            "SEMP": [500.0, 0.0, np.nan],
            _UNIT: [1, 1, 2],
            _ROLE: ["HEAD", "SPOUSE", "HEAD"],
            _P: [700.0, -300.0, 50.0],
        }
    )

    counts = keep_semp_covered_partnership_income(
        person,
        imputed_masks={("person", _P): np.ones(3, dtype=bool)},
        person_weights=np.asarray([2.0, 2.0, 10.0]),
    )

    assert counts is not None
    assert counts["own_semp_discarded_positive_dollars"] == 700
    assert counts["tax_unit_discarded_negative_dollars"] == -300
    assert counts["blank_semp_discarded_positive_dollars"] == 50
    assert counts["discarded_positive_dollars"] == 750
    assert counts["discarded_negative_dollars"] == -300
    assert counts["household_weighted_discarded_positive_dollars"] == 1_900
    assert counts["household_weighted_discarded_negative_dollars"] == -600
    assert counts["tax_unit_household_weighted_discarded_negative_dollars"] == -600


@pytest.mark.parametrize("column", ("SEMP", _UNIT, _ROLE))
def test_the_step_refuses_a_recipient_missing_a_column_it_reads(column: str) -> None:
    person = pd.DataFrame(
        {"SEMP": [500.0], _UNIT: [1], _ROLE: ["HEAD"], _P: [700.0]}
    ).drop(columns=[column])

    with pytest.raises(ValueError, match="refusing to treat them as zero"):
        keep_semp_covered_partnership_income(
            person, imputed_masks={("person", _P): np.asarray([True])}
        )


# --- the gate ---------------------------------------------------------------


@st.composite
def _gate_tables(draw):
    rows = draw(st.integers(min_value=1, max_value=10))
    semp = np.asarray(draw(st.lists(_semp, min_size=rows, max_size=rows)))
    adjinc = np.full(rows, 1_000_000.0)
    person = pd.DataFrame(
        {
            "SEMP": semp,
            "ADJINC": adjinc,
            _TOTAL: np.where(np.isnan(semp), np.nan, semp),
            **{
                column: np.zeros(rows)
                for column in ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS
            },
            _UNIT: draw(st.lists(_unit, min_size=rows, max_size=rows)),
            _ROLE: draw(st.lists(_role, min_size=rows, max_size=rows)),
            _P: draw(st.lists(_dollars, min_size=rows, max_size=rows)),
            _S: draw(st.lists(_dollars, min_size=rows, max_size=rows)),
            _PASSIVE: draw(st.lists(_dollars, min_size=rows, max_size=rows)),
        }
    )
    start = draw(st.integers(min_value=0, max_value=50))
    index = np.arange(start, start + rows)
    person.index = index[::-1] if draw(st.booleans()) else index
    return person


def _oracle_gate_counts(person: pd.DataFrame) -> dict[str, int]:
    sets = _oracle_sets(person)
    partnership = person[_P].to_numpy(dtype=float)
    not_zero = ~(np.isfinite(partnership) & (partnership == 0.0))
    line_32 = (person[_P].fillna(0.0) + person[_S].fillna(0.0)).to_numpy()
    units = person[_UNIT].tolist()
    unit_line_32 = {
        unit: sum(
            value for other, value in zip(units, line_32, strict=True) if other == unit
        )
        for unit in units
        if not pd.isna(unit)
    }
    passive = person[_PASSIVE].fillna(0.0).to_numpy()
    no_line_32 = np.asarray(
        [
            own == 0.0 or (not pd.isna(unit) and unit_line_32[unit] == 0.0)
            for own, unit in zip(line_32, units, strict=True)
        ]
    )
    return {
        "partnership_on_semp_rows": int((sets["own_semp"] & not_zero).sum()),
        "partnership_on_spouse_rows": int((sets["tax_unit"] & not_zero).sum()),
        "partnership_on_blank_semp_rows": int((sets["blank_semp"] & not_zero).sum()),
        "passive_without_line_32_rows": int((no_line_32 & (passive != 0.0)).sum()),
    }


@given(_gate_tables())
@settings(max_examples=400, deadline=None)
def test_the_gate_counts_exactly_the_line_32_violations(person) -> None:
    gate = acs_self_employment_identity_gate(
        person, rows=np.ones(len(person), dtype=bool)
    )
    expected = _oracle_gate_counts(person)

    for key, value in expected.items():
        assert gate.details[key] == value, key
    assert gate.passed is (sum(expected.values()) == 0)


@given(_gate_tables(), _gate_tables(), st.booleans())
@settings(max_examples=300, deadline=None)
def test_unselected_rows_never_change_the_gate(
    selected, ignored, ignored_first
) -> None:
    alone = acs_self_employment_identity_gate(
        selected, rows=np.ones(len(selected), dtype=bool)
    )
    # The ignored rows share tax-unit ids with the selected ones. The pooled
    # staging frame puts the base rows first, so the selected ACS rows then
    # carry an index that does not start at zero.
    blocks = [ignored, selected] if ignored_first else [selected, ignored]
    combined = pd.concat(blocks, ignore_index=True)
    flags = [np.zeros(len(ignored), bool), np.ones(len(selected), bool)]
    rows = np.concatenate(flags if ignored_first else flags[::-1])

    together = acs_self_employment_identity_gate(combined, rows=rows)

    assert together.passed is alone.passed
    assert together.failures == alone.failures
    assert together.details == alone.details


def _gate_row(**values) -> pd.DataFrame:
    rows = len(next(iter(values.values())))
    base = {
        "SEMP": [0.0] * rows,
        "ADJINC": [1_000_000.0] * rows,
        _TOTAL: [0.0] * rows,
        **{column: [0.0] * rows for column in ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS},
        _UNIT: list(range(1, rows + 1)),
        _ROLE: ["HEAD"] * rows,
        _P: [0.0] * rows,
        _S: [0.0] * rows,
        _PASSIVE: [0.0] * rows,
    }
    base.update(values)
    if "SEMP" in values:
        base[_TOTAL] = [np.nan if pd.isna(value) else value for value in values["SEMP"]]
    return pd.DataFrame(base)


@pytest.mark.parametrize("index", (None, [10, 11], [11, 10]))
def test_the_gate_extends_a_reporting_head_to_the_spouse_at_any_index(index) -> None:
    person = _gate_row(
        **{
            "SEMP": [5_000.0, 0.0],
            _UNIT: [3, 3],
            _ROLE: ["HEAD", "SPOUSE"],
            _P: [0.0, 7_000.0],
        }
    )
    if index is not None:
        person.index = index

    gate = acs_self_employment_identity_gate(person, rows=np.ones(2, dtype=bool))

    assert not gate.passed
    assert gate.details["partnership_on_spouse_rows"] == 1


@pytest.mark.parametrize("index", (None, [10, 11], [11, 10]))
def test_the_gate_reads_a_tax_unit_whose_line_32_cancels_as_zero(index) -> None:
    # Each person has line 32, but the unit's total is zero (refuter A7). The
    # pooled staging frame's ACS rows carry an index that does not start at 0.
    person = _gate_row(**{_UNIT: [7, 7], _P: [900.0, -900.0], _PASSIVE: [300.0, 0.0]})
    if index is not None:
        person.index = index

    gate = acs_self_employment_identity_gate(person, rows=np.ones(2, dtype=bool))

    assert not gate.passed
    assert gate.details["passive_without_line_32_rows"] == 1


@pytest.mark.parametrize(
    ("values", "violations"),
    (
        # NaN partnership income on a SEMP row is not a finite zero.
        ({"SEMP": [5_000.0], _P: [np.nan]}, {"partnership_on_semp_rows": 1}),
        # NaN line-32 leaves read as zero under a passive draw.
        ({_P: [np.nan], _PASSIVE: [300.0]}, {"passive_without_line_32_rows": 1}),
        # A NaN passive draw is not a passive amount.
        ({_PASSIVE: [np.nan]}, {}),
    ),
)
def test_the_gate_reads_nan_line_32_leaves(values, violations) -> None:
    gate = acs_self_employment_identity_gate(
        _gate_row(**values), rows=np.ones(1, dtype=bool)
    )

    for key in ("partnership_on_semp_rows", "passive_without_line_32_rows"):
        assert gate.details[key] == violations.get(key, 0), key
    assert gate.passed is (not violations)


@pytest.mark.parametrize("bad", ("SPOUSE ", "OTHER", 2, None))
def test_the_gate_fails_a_malformed_role_instead_of_exempting_its_unit(bad) -> None:
    person = _gate_row(
        **{"SEMP": [5_000.0, 0.0], _UNIT: [1, 1], _ROLE: ["HEAD", bad], _P: [0.0, 9.0]}
    )

    gate = acs_self_employment_identity_gate(person, rows=np.ones(2, dtype=bool))

    assert not gate.passed
    assert gate.details["malformed_tax_unit_role_rows"] == 1


def test_the_gate_leaves_s_corp_income_and_partnership_on_unreported_rows_alone() -> (
    None
):
    person = pd.DataFrame(
        {
            "SEMP": [0.0, 5_000.0, 0.0],
            "ADJINC": [1_000_000.0] * 3,
            _TOTAL: [0.0, 5_000.0, 0.0],
            **{column: [0.0] * 3 for column in ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS},
            _UNIT: [1, 2, 3],
            _ROLE: ["HEAD", "HEAD", "HEAD"],
            _P: [8_000.0, 0.0, -2_000.0],
            _S: [0.0, 40_000.0, 0.0],
            _PASSIVE: [500.0, 300.0, 0.0],
        }
    )

    gate = acs_self_employment_identity_gate(person, rows=np.ones(3, dtype=bool))

    assert gate.passed, gate.failures


@pytest.mark.parametrize("column", (_P, _S, _PASSIVE))
def test_the_gate_refuses_an_infinite_line_32_value(column: str) -> None:
    person = _gate_row(**{_S: [10.0], column: [np.inf]})

    gate = acs_self_employment_identity_gate(person, rows=np.ones(1, dtype=bool))

    assert not gate.passed
    assert gate.details["infinite_line_32_rows"] == 1


# --- the real transfer -------------------------------------------------------

_SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))
_PASS_THROUGH = (_P, _S, _PASSIVE)


def _frame(person: pd.DataFrame, weights: np.ndarray, kind: WeightKind) -> Frame:
    households = np.unique(person["person_household_id"])
    units = np.unique(person[_UNIT])
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": households, "state_fips": [6] * len(households)}
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": units}),
        },
        _SCHEMA,
        {"household": Weights(weights, kind)},
    )


def _donor(s_corp: float) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 5, dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20, 20], dtype=np.int64),
            _UNIT: np.asarray([1, 1, 2, 2], dtype=np.int64),
            "age": [45.0, 43.0, 30.0, 52.0],
            "is_female": [False, True, True, False],
            _TOTAL: [0.0, 20_000.0, 0.0, 5_000.0],
            _P: [5_000.0, 0.0, 3_000.0, 0.0],
            _S: [s_corp, 0.0, 0.0, 0.0],
            _PASSIVE: [400.0, 0.0, 0.0, 0.0],
        }
    )
    return _frame(person, np.asarray([1.0, 2.0]), WeightKind.CALIBRATED)


# Seven ACS persons, one per covered case:
#   0 single head, SEMP 0          -> keeps its draw
#   1 head, SEMP 12,000            -> own SEMP (positive)
#   2 spouse of 1, SEMP 0          -> tax-unit extension
#   3 head, SEMP -800              -> own SEMP (negative)
#   4 dependent of 3, blank SEMP   -> blank SEMP
#   5 head, SEMP 0                 -> keeps: only a dependent reports
#   6 dependent of 5, SEMP 3,000   -> own SEMP (positive)
_SEMP = np.asarray([0.0, 12_000.0, 0.0, -800.0, np.nan, 0.0, 3_000.0])
_COVERED = np.asarray([False, True, True, True, True, False, True])


def _recipient() -> Frame:
    rows = len(_SEMP)
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 101 + rows, dtype=np.int64),
            "person_household_id": np.asarray(
                [110, 120, 120, 130, 130, 140, 140], dtype=np.int64
            ),
            _UNIT: np.asarray([11, 12, 12, 13, 13, 14, 14], dtype=np.int64),
            _ROLE: pd.array(
                ["HEAD", "HEAD", "SPOUSE", "HEAD", "DEPENDENT", "HEAD", "DEPENDENT"],
                dtype=CANONICAL_STRING_DTYPE,
            ),
            "age": [50.0, 44.0, 42.0, 61.0, 9.0, 47.0, 17.0],
            "is_female": [True, False, True, False, True, False, True],
            "SEMP": _SEMP,
            "ADJINC": np.where(np.isnan(_SEMP), np.nan, 1_000_000.0),
            _TOTAL: _SEMP,
            **{
                column: np.zeros(rows)
                for column in ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS
            },
            _P: np.full(rows, np.nan),
            _S: np.full(rows, np.nan),
            _PASSIVE: np.full(rows, np.nan),
        }
    )
    return _frame(person, np.asarray([5.0, 7.0, 3.0, 4.0]), WeightKind.DESIGN)


class _DonorMeanQRF:
    def __init__(self, *, n_estimators: int, seed: int) -> None:
        del n_estimators, seed

    def fit(self, frame, predictors, targets, *, weights):
        del predictors
        table = frame.table(frame.column_entity(targets[0]))
        assert weights == frame.resolve_weights("person").kind.value
        return _DonorMeanFitted(
            {target: float(table[target].mean()) for target in targets}, weights
        )


class _DonorMeanFitted:
    def __init__(self, means: dict[str, float], weight_kind: str) -> None:
        self.means = means
        self.weight_kind = weight_kind

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {target: np.full(len(frame), mean) for target, mean in self.means.items()},
            index=frame.index,
        )


def _transfer(monkeypatch, *, lane: bool, s_corp: float = 0.0, targets=_PASS_THROUGH):
    monkeypatch.setattr(acs_transfer_module, "QRF", _DonorMeanQRF)
    return acs_transfer_module.transfer_acs_inputs(
        _recipient(),
        _donor(s_corp),
        target_families={"person": {"pass_through": targets}},
        donor_channel=None,
        n_estimators=1,
        semp_covers_partnership_income=lane,
    )


def _gate(person: pd.DataFrame):
    return acs_self_employment_identity_gate(
        person, rows=np.ones(len(person), dtype=bool)
    )


def test_the_lane_keeps_semp_covered_draws_at_zero_and_passes_the_gate(
    monkeypatch,
) -> None:
    result = _transfer(monkeypatch, lane=True)
    person = result.frame.table("person")

    assert (person.loc[_COVERED, _P] == 0.0).all()
    assert np.allclose(person.loc[~_COVERED, _P], 2_000.0)
    assert (person[_S] == 0.0).all()
    # The passive reconciliation ran after the step: with no S-corp income,
    # line 32 is zero exactly where the step cleared the partnership draw.
    assert (person.loc[_COVERED, _PASSIVE] == 0.0).all()
    assert np.allclose(person.loc[~_COVERED, _PASSIVE], 100.0)
    gate = _gate(person)
    assert gate.passed, gate.failures

    # Each covered cell held the donor mean, 2,000. Household design weights:
    # 7 (rows 1-2), 3 (rows 3-4), 4 (rows 5-6).
    record = next(item for item in result.imputed_inputs if item.column == _P)
    assert record.reconciliation == {
        "own_semp_cells": 3,
        "own_semp_nonzero_draws_discarded": 3,
        "own_semp_discarded_positive_dollars": 6_000,
        "own_semp_discarded_negative_dollars": 0,
        "own_semp_household_weighted_discarded_positive_dollars": 28_000,
        "own_semp_household_weighted_discarded_negative_dollars": 0,
        "tax_unit_cells": 1,
        "tax_unit_nonzero_draws_discarded": 1,
        "tax_unit_discarded_positive_dollars": 2_000,
        "tax_unit_discarded_negative_dollars": 0,
        "tax_unit_household_weighted_discarded_positive_dollars": 14_000,
        "tax_unit_household_weighted_discarded_negative_dollars": 0,
        "blank_semp_cells": 1,
        "blank_semp_nonzero_draws_discarded": 1,
        "blank_semp_discarded_positive_dollars": 2_000,
        "blank_semp_discarded_negative_dollars": 0,
        "blank_semp_household_weighted_discarded_positive_dollars": 6_000,
        "blank_semp_household_weighted_discarded_negative_dollars": 0,
        "own_semp_positive_cells": 2,
        "own_semp_negative_cells": 1,
        "cells_kept_at_zero": 5,
        "nonzero_draws_discarded": 5,
        "discarded_positive_dollars": 10_000,
        "discarded_negative_dollars": 0,
        "household_weighted_discarded_positive_dollars": 48_000,
        "household_weighted_discarded_negative_dollars": 0,
    }


def test_the_lane_keeps_s_corp_income_on_every_row(monkeypatch) -> None:
    person = _transfer(monkeypatch, lane=True, s_corp=1_000.0).frame.table("person")

    assert np.allclose(person[_S], 250.0)
    assert (person.loc[_COVERED, _P] == 0.0).all()
    # Line 32 stays nonzero on every row, so every passive draw stands.
    assert np.allclose(person[_PASSIVE], 100.0)
    assert _gate(person).passed


def test_off_the_lane_the_gate_fails_on_exactly_the_covered_rows(monkeypatch) -> None:
    person = _transfer(monkeypatch, lane=False).frame.table("person")

    gate = _gate(person)

    assert not gate.passed
    assert gate.details["partnership_on_semp_rows"] == 3
    assert gate.details["partnership_on_positive_semp_rows"] == 2
    assert gate.details["partnership_on_negative_semp_rows"] == 1
    assert gate.details["partnership_on_spouse_rows"] == 1
    assert gate.details["partnership_on_blank_semp_rows"] == 1
    assert gate.details["passive_without_line_32_rows"] == 0


class _RecordingQRF(_DonorMeanQRF):
    calls: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        super().__init__(n_estimators=n_estimators, seed=seed)
        self.seed = seed

    def fit(self, frame, predictors, targets, *, weights):
        type(self).calls.append((self.seed, tuple(predictors), tuple(targets)))
        return super().fit(frame, predictors, targets, weights=weights)


def test_the_lane_changes_only_partnership_income_and_the_passive_subset(
    monkeypatch,
) -> None:
    runs = {}
    for lane in (True, False):
        _RecordingQRF.calls = []
        monkeypatch.setattr(acs_transfer_module, "QRF", _RecordingQRF)
        result = acs_transfer_module.transfer_acs_inputs(
            _recipient(),
            _donor(0.0),
            target_families={"person": {"pass_through": _PASS_THROUGH}},
            donor_channel=None,
            n_estimators=1,
            semp_covers_partnership_income=lane,
        )
        runs[lane] = (result, list(_RecordingQRF.calls))
    (on, on_calls), (off, off_calls) = runs[True], runs[False]
    on_person, off_person = on.frame.table("person"), off.frame.table("person")

    pd.testing.assert_frame_equal(
        on_person.drop(columns=[_P, _PASSIVE]), off_person.drop(columns=[_P, _PASSIVE])
    )
    assert (on_person[_P] != off_person[_P]).to_numpy().tolist() == _COVERED.tolist()
    # Same models, seeds, predictors and target order; only the partnership
    # record's receipt and the passive record's counts differ.
    assert on_calls == off_calls and on_calls
    assert on.fit_records == off.fit_records
    assert [
        dataclasses.replace(item, reconciliation=None) for item in on.imputed_inputs
    ] == [dataclasses.replace(item, reconciliation=None) for item in off.imputed_inputs]


def test_a_step_run_after_the_passive_reconciliation_fails_the_gate(
    monkeypatch,
) -> None:
    """The order matters: zeroing after the passive block strands passive."""

    monkeypatch.setattr(
        acs_transfer_module,
        "keep_semp_covered_partnership_income",
        lambda person, **_masks_and_weights: None,
    )
    person = _transfer(monkeypatch, lane=True).frame.table("person")
    person.loc[_COVERED, _P] = 0.0

    gate = _gate(person)

    assert not gate.passed
    assert gate.details["partnership_on_semp_rows"] == 0
    assert gate.details["passive_without_line_32_rows"] == int(_COVERED.sum())


def test_the_lane_refuses_a_plan_without_s_corp_income(monkeypatch) -> None:
    with pytest.raises(ValueError, match="without s_corp_income"):
        _transfer(monkeypatch, lane=True, targets=(_P, _PASSIVE))


class _NoFitQRF(_DonorMeanQRF):
    def fit(self, frame, predictors, targets, *, weights):
        raise AssertionError("the refusal must come before any fit")


@pytest.mark.parametrize("column", ("SEMP", _ROLE))
def test_the_lane_refuses_a_recipient_without_a_column_the_rule_reads(
    monkeypatch, column: str
) -> None:
    recipient = _recipient()
    tables = {name: recipient.table(name) for name in recipient.entities}
    tables["person"] = tables["person"].drop(columns=[column])
    monkeypatch.setattr(acs_transfer_module, "QRF", _NoFitQRF)

    with pytest.raises(ValueError, match="refusing to treat them as zero"):
        acs_transfer_module.transfer_acs_inputs(
            Frame(
                tables,
                recipient.schema,
                {"household": recipient.weights_for("household")},
            ),
            _donor(0.0),
            target_families={"person": {"pass_through": _PASS_THROUGH}},
            donor_channel=None,
            n_estimators=1,
            semp_covers_partnership_income=True,
        )


# --- the contract stays opt-in -----------------------------------------------

_DECLARED_TARGETS = tuple(
    sorted(
        {
            target
            for families in declared_acs_transfer_target_families().values()
            for targets in families.values()
            for target in targets
        }
    )
)


@given(st.sets(st.sampled_from(_DECLARED_TARGETS)))
@settings(max_examples=150, deadline=None)
def test_the_contract_is_unchanged_off_the_lane(targets) -> None:
    assert acs_transfer_execution_contract_identity(
        targets=sorted(targets), semp_covers_partnership_income=False
    ) == acs_transfer_execution_contract_identity(targets=sorted(targets))


def test_the_lane_binds_its_rule_into_the_contract() -> None:
    off = acs_transfer_execution_contract_identity(targets=_PASS_THROUGH)
    on = acs_transfer_execution_contract_identity(
        targets=_PASS_THROUGH, semp_covers_partnership_income=True
    )
    without_partnership = acs_transfer_execution_contract_identity(
        targets=(_S,), semp_covers_partnership_income=True
    )

    assert "semp_covered_partnership_income" not in off["post_transfer_structure"]
    rule = on["post_transfer_structure"]["semp_covered_partnership_income"]
    assert rule["enabled"] is True
    assert rule["covered"] == list(ACS_SEMP_COVERED_LINE_32_INPUTS) == [_P]
    assert rule["order"] == "before_passive_partnership_s_corp"
    assert on["sha256"] != off["sha256"]
    assert (
        without_partnership["post_transfer_structure"][
            "semp_covered_partnership_income"
        ]["enabled"]
        is False
    )


@pytest.mark.parametrize(
    "function",
    (
        acs_transfer_module.transfer_acs_inputs,
        acs_transfer_execution_contract_identity,
        acs_multispine_module.build_optional_acs_multispine,
    ),
)
def test_every_entry_point_defaults_the_lane_off(function) -> None:
    parameter = inspect.signature(function).parameters["semp_covers_partnership_income"]

    assert parameter.default is False
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
