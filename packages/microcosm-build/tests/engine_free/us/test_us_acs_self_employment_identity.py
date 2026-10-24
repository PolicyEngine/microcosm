"""Property and example tests for the ACS self-employment identity (#1380).

Invariants of :func:`keep_measured_self_employment_totals`, for every person
table and imputation mask:

* **Total untouched.** ``self_employment_income_before_lsr`` never changes.
* **Kept at zero.** A component cell the transfer filled is zero unless the
  transfer also drew that row's total.
* **No total, no change.** A table with no total column is left exactly as
  it was: there is no measured total for a component to add to.
* **Joint draws survive.** A component filled on a row whose total was drawn
  alongside it keeps its drawn value.
* **Observed cells untouched.** A component cell the transfer did not fill
  keeps its value, blank or not.
* **Idempotent.** Applying the step twice equals applying it once.
* **Receipt.** Each touched component's counts equal the cells kept at zero
  and the nonzero draws among them.

Invariants of :func:`acs_self_employment_identity_gate`:

* **Splits pass.** Any division of ``SEMP * ADJINC / 1e6`` across the base
  leaf and the three components passes.
* **Additions fail.** Moving one reporting row's leaves off that total by more
  than the tolerance fails, and the gate counts exactly that row.
* **Selection.** Rows outside ``rows`` never affect the verdict.
* **Finite only.** An infinite SEMP, leaf or ADJINC never passes, even when
  opposing infinities would cancel.

The composition test runs the real transfer and requires its output to pass
the gate, and the same transfer without the post-transfer step to fail it:
the gate detects exactly the stacking the step removes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.build.us_runtime.acs_inputs import (
    ACS_SELF_EMPLOYMENT_IDENTITY_TOLERANCE,
    acs_self_employment_identity_gate,
)
from microcosm.build.us_runtime.acs_transfer import (
    ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS,
    keep_measured_self_employment_totals,
)

_TOTAL = "self_employment_income_before_lsr"
_LEAVES = (_TOTAL, *ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS)
_TOLERANCE = ACS_SELF_EMPLOYMENT_IDENTITY_TOLERANCE

_amounts = st.one_of(
    st.just(0.0),
    st.just(np.nan),
    st.floats(min_value=-1e6, max_value=1e7, allow_nan=False, width=64),
)


@st.composite
def _imputed_tables(draw):
    rows = draw(st.integers(min_value=1, max_value=12))
    table = pd.DataFrame(
        {
            column: draw(st.lists(_amounts, min_size=rows, max_size=rows))
            for column in _LEAVES
        }
    )
    masks: dict[tuple[str, str], np.ndarray] = {}
    for column in _LEAVES:
        if draw(st.booleans()):
            masks[("person", column)] = np.asarray(
                draw(st.lists(st.booleans(), min_size=rows, max_size=rows)),
                dtype=bool,
            )
    # A filled cell is never blank after a transfer.
    for (_entity, column), mask in masks.items():
        filled = table[column].to_numpy(copy=True)
        filled[mask & np.isnan(filled)] = 0.0
        table[column] = filled
    # A recipient may carry no total at all; then the transfer drew none.
    if draw(st.booleans()):
        table = table.drop(columns=[_TOTAL])
        masks.pop(("person", _TOTAL), None)
    return table, masks


@given(_imputed_tables())
@settings(max_examples=300, deadline=None)
def test_post_transfer_step_keeps_measured_totals(case) -> None:
    before, masks = case
    after = before.copy(deep=True)
    counts = keep_measured_self_employment_totals(after, imputed_masks=masks)

    if _TOTAL not in before.columns:
        assert counts == {}
        pd.testing.assert_frame_equal(after, before)
        return
    rows = len(before)
    total_drawn = masks.get(("person", _TOTAL), np.zeros(rows, dtype=bool))
    pd.testing.assert_series_equal(after[_TOTAL], before[_TOTAL])
    for column in ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS:
        drawn = masks.get(("person", column), np.zeros(rows, dtype=bool))
        kept = drawn & ~total_drawn
        assert (after.loc[kept, column] == 0.0).all()
        unchanged = ~kept
        pd.testing.assert_series_equal(
            after.loc[unchanged, column], before.loc[unchanged, column]
        )
        if kept.any():
            values = before.loc[kept, column].to_numpy()
            assert counts[column] == {
                "cells_kept_at_zero": int(kept.sum()),
                "nonzero_draws_discarded": int((values != 0.0).sum()),
            }
        else:
            assert column not in counts

    again = after.copy(deep=True)
    keep_measured_self_employment_totals(again, imputed_masks=masks)
    pd.testing.assert_frame_equal(again, after)


def test_post_transfer_step_ignores_absent_components() -> None:
    table = pd.DataFrame({_TOTAL: [10.0, 20.0]})
    masks = {("person", "farm_operations_income"): np.asarray([True, True])}

    assert keep_measured_self_employment_totals(table, imputed_masks=masks) == {}
    assert table[_TOTAL].tolist() == [10.0, 20.0]


def _person(**columns) -> pd.DataFrame:
    rows = len(next(iter(columns.values())))
    defaults = {
        "ADJINC": [1_000_000.0] * rows,
        **{column: [0.0] * rows for column in _LEAVES},
    }
    return pd.DataFrame({**defaults, **columns})


def _all(person: pd.DataFrame) -> np.ndarray:
    return np.ones(len(person), dtype=bool)


def test_gate_passes_native_semp_and_splits_of_it() -> None:
    person = _person(
        SEMP=[1_000.0, 2_000.0, 0.0, -500.0, np.nan],
        ADJINC=[1_010_000.0] * 5,
        self_employment_income_before_lsr=[1_010.0, 0.0, 0.0, -505.0, np.nan],
        sstb_self_employment_income_before_lsr=[0.0, 2_020.0, 0.0, 0.0, 0.0],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert gate.passed, gate.failures
    assert gate.details["semp_reported_rows"] == 4
    assert gate.details["blank_semp_rows"] == 1
    assert gate.details["mismatched_rows"] == 0


@pytest.mark.parametrize("component", ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS)
def test_gate_refuses_a_component_drawn_on_top_of_semp(component: str) -> None:
    person = _person(
        SEMP=[0.0, 3_000.0],
        self_employment_income_before_lsr=[0.0, 3_000.0],
        **{component: [4_000.0, 0.0]},
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["mismatched_rows"] == 1
    assert gate.details["excess_over_semp"] == 4_000.0
    assert "counts that income twice" in gate.failures[0]


def test_gate_refuses_self_employment_outside_the_semp_universe() -> None:
    person = _person(
        SEMP=[np.nan, np.nan],
        self_employment_income_before_lsr=[np.nan, np.nan],
        sstb_self_employment_income_before_lsr=[0.0, 750.0],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["blank_semp_nonzero_rows"] == 1
    assert "blank SEMP carry nonzero self-employment" in gate.failures[0]


def test_gate_refuses_rows_it_cannot_check() -> None:
    person = _person(
        SEMP=[100.0, 100.0],
        ADJINC=[np.nan, 1_000_000.0],
        self_employment_income_before_lsr=[100.0, np.nan],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["nonfinite_adjinc_rows"] == 1
    assert gate.details["nonfinite_leaf_rows"] == 1


def test_gate_refuses_opposing_infinite_leaves() -> None:
    person = _person(
        SEMP=[100.0],
        self_employment_income_before_lsr=[100.0],
        farm_operations_income=[np.inf],
        partnership_self_employment_net_earnings=[-np.inf],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["nonfinite_leaf_rows"] == 1
    assert "blank or infinite self-employment leaf" in gate.failures[0]


def test_gate_refuses_infinite_semp_adjinc_and_leaves_outside_the_universe() -> (
    None
):
    person = _person(
        SEMP=[np.inf, 100.0, np.nan],
        ADJINC=[1_000_000.0, np.inf, 1_000_000.0],
        self_employment_income_before_lsr=[0.0, 100.0, np.nan],
        sstb_self_employment_income_before_lsr=[0.0, 0.0, -np.inf],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["infinite_semp_rows"] == 1
    assert gate.details["blank_semp_rows"] == 1
    assert gate.details["nonfinite_adjinc_rows"] == 1
    assert gate.details["blank_semp_nonzero_rows"] == 1
    assert gate.details["semp_reported_rows"] == 1


def test_gate_refuses_a_gap_that_overflows() -> None:
    person = _person(
        SEMP=[1e308],
        ADJINC=[1e7],
        self_employment_income_before_lsr=[1e308],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["mismatched_rows"] == 1


@pytest.mark.parametrize("column", ("SEMP", "ADJINC", *_LEAVES))
def test_gate_fails_closed_on_a_missing_column(column: str) -> None:
    person = _person(SEMP=[100.0], self_employment_income_before_lsr=[100.0])

    gate = acs_self_employment_identity_gate(
        person.drop(columns=[column]), rows=np.ones(1, dtype=bool)
    )

    assert not gate.passed
    assert gate.details == {"missing_columns": [column]}


def test_gate_tolerance_is_one_dollar_inclusive() -> None:
    person = _person(
        SEMP=[100.0, 100.0],
        self_employment_income_before_lsr=[101.0, 101.01],
    )

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert gate.details["mismatched_rows"] == 1


def test_gate_reads_only_selected_rows() -> None:
    person = _person(
        SEMP=[100.0, np.nan],
        self_employment_income_before_lsr=[100.0, 9_999.0],
        sstb_self_employment_income_before_lsr=[0.0, 9_999.0],
    )

    gate = acs_self_employment_identity_gate(person, rows=np.asarray([True, False]))

    assert gate.passed, gate.failures
    assert gate.details["acs_person_rows"] == 1


def test_gate_refuses_an_empty_selection_and_a_misaligned_mask() -> None:
    person = _person(SEMP=[100.0], self_employment_income_before_lsr=[100.0])

    empty = acs_self_employment_identity_gate(person, rows=np.zeros(1, dtype=bool))
    assert not empty.passed
    assert empty.failures == ("No ACS person rows were selected for the check.",)
    with pytest.raises(ValueError, match="aligned to person"):
        acs_self_employment_identity_gate(person, rows=np.ones(2, dtype=bool))


@st.composite
def _split_rows(draw):
    rows = draw(st.integers(min_value=1, max_value=10))
    semp = draw(
        st.lists(
            st.integers(min_value=-20_000, max_value=1_500_000),
            min_size=rows,
            max_size=rows,
        )
    )
    adjinc = draw(
        st.lists(
            st.integers(min_value=990_000, max_value=1_050_000),
            min_size=rows,
            max_size=rows,
        )
    )
    shares = draw(
        st.lists(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=3,
                max_size=3,
            ),
            min_size=rows,
            max_size=rows,
        )
    )
    total = np.asarray(semp, dtype=np.float64) * np.asarray(adjinc) / 1_000_000
    components = np.asarray(shares) / 3.0 * total[:, None]
    person = pd.DataFrame(
        {
            "SEMP": np.asarray(semp, dtype=np.float64),
            "ADJINC": np.asarray(adjinc, dtype=np.float64),
            _TOTAL: total - components.sum(axis=1),
        }
    )
    for index, column in enumerate(ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS):
        person[column] = components[:, index]
    return person


@given(_split_rows())
@settings(max_examples=300, deadline=None)
def test_any_split_of_semp_passes(person: pd.DataFrame) -> None:
    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert gate.passed, gate.failures


@given(
    _split_rows(),
    st.data(),
    st.floats(min_value=_TOLERANCE + 0.01, max_value=1e6, allow_nan=False),
    st.booleans(),
)
@settings(max_examples=300, deadline=None)
def test_moving_one_row_off_semp_fails_on_exactly_that_row(
    person: pd.DataFrame, data, delta: float, negative: bool
) -> None:
    row = data.draw(st.integers(min_value=0, max_value=len(person) - 1))
    leaf = data.draw(st.sampled_from(_LEAVES))
    person.loc[row, leaf] += -delta if negative else delta

    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert not gate.passed
    assert gate.details["mismatched_rows"] == 1


def _transfer_case():
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    schema = EntitySchema(group_entities=("household",))
    donor_person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20, 20, 30, 30]),
            "age": [45.0, 43.0, 30.0, 52.0, 61.0, 38.0],
            "is_female": [False, True, True, False, False, True],
            _TOTAL: [0.0, 20_000.0, 0.0, 0.0, 5_000.0, 0.0],
            "farm_operations_income": [0.0, 0.0, -3_000.0, 0.0, 0.0, 0.0],
            "partnership_self_employment_net_earnings": [0.0] * 5 + [9_000.0],
            "sstb_self_employment_income_before_lsr": [60_000.0] + [0.0] * 5,
        }
    )
    donor = Frame(
        {
            "person": donor_person,
            "household": pd.DataFrame(
                {"household_id": np.asarray([10, 20, 30]), "state_fips": [6, 6, 36]}
            ),
        },
        schema,
        {"household": Weights(np.asarray([1.0, 2.0, 3.0]), WeightKind.CALIBRATED)},
    )
    semp = np.asarray([0.0, 12_000.0, np.nan, 800.0])
    adjinc = np.full(4, 1_021_000.0)
    recipient_person = pd.DataFrame(
        {
            "person_id": np.arange(101, 105, dtype=np.int64),
            "person_household_id": np.asarray([110, 110, 120, 120]),
            "age": [40.0, 44.0, 9.0, 66.0],
            "is_female": [True, False, False, True],
            "SEMP": semp,
            "ADJINC": adjinc,
            _TOTAL: semp * adjinc / 1_000_000,
        }
    )
    recipient = Frame(
        {
            "person": recipient_person,
            "household": pd.DataFrame(
                {"household_id": np.asarray([110, 120]), "state_fips": [6, 36]}
            ),
        },
        schema,
        {"household": Weights(np.asarray([5.0, 7.0]), WeightKind.DESIGN)},
    )
    return recipient, donor


class _DonorMeanQRF:
    def __init__(self, *, n_estimators: int, seed: int) -> None:
        del n_estimators, seed

    def fit(self, frame, predictors, targets, *, weights):
        del predictors
        entity = frame.column_entity(targets[0])
        table = frame.table(entity)
        assert weights == frame.resolve_weights(entity).kind.value
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


def _transfer(monkeypatch, recipient, donor):
    monkeypatch.setattr(acs_transfer_module, "QRF", _DonorMeanQRF)
    return acs_transfer_module.transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {"self_employment": ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS}
        },
        donor_channel=None,
        n_estimators=1,
    )


def test_component_only_transfer_without_a_total_keeps_its_draws(
    monkeypatch,
) -> None:
    """A recipient with no self-employment total has none to protect.

    Its component draws are the only self-employment it gets, so the step
    leaves them as drawn and records nothing.
    """

    recipient, donor = _transfer_case()
    from microcosm.frame import Frame

    recipient = Frame(
        {
            "person": recipient.table("person").drop(columns=[_TOTAL]),
            "household": recipient.table("household"),
        },
        recipient.schema,
        {"household": recipient.weights_for("household")},
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _DonorMeanQRF)
    result = acs_transfer_module.transfer_acs_inputs(
        recipient,
        donor,
        target_families={"person": {"farm": ("farm_operations_income",)}},
        donor_channel=None,
        n_estimators=1,
    )
    person = result.frame.table("person")

    assert _TOTAL not in person.columns
    assert np.allclose(person["farm_operations_income"], -500.0)
    record = next(
        item
        for item in result.imputed_inputs
        if item.column == "farm_operations_income"
    )
    assert record.reconciliation is None


def test_transfer_output_passes_the_gate_only_with_the_step(monkeypatch) -> None:
    recipient, donor = _transfer_case()

    result = _transfer(monkeypatch, recipient, donor)
    person = result.frame.table("person")
    gate = acs_self_employment_identity_gate(person, rows=_all(person))

    assert gate.passed, gate.failures
    for column in ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS:
        assert (person[column] == 0.0).all()
        record = next(item for item in result.imputed_inputs if item.column == column)
        assert record.reconciliation == {
            "cells_kept_at_zero": 4,
            "nonzero_draws_discarded": 4,
        }
    pd.testing.assert_series_equal(person[_TOTAL], recipient.table("person")[_TOTAL])

    monkeypatch.setattr(
        acs_transfer_module,
        "keep_measured_self_employment_totals",
        lambda person, *, imputed_masks: {},
    )
    stacked = _transfer(monkeypatch, recipient, donor).frame.table("person")
    refused = acs_self_employment_identity_gate(stacked, rows=_all(stacked))

    assert not refused.passed
    assert refused.details["mismatched_rows"] == 3
    assert refused.details["blank_semp_nonzero_rows"] == 1
