"""Reusing a household-batched pass for households whose inputs are unchanged.

``microcosm.build.household_pass_reuse`` keeps a pass's per-person values and
a digest of each household's rows (entity IDs, membership, row order and the
pass's input columns). A later pass over a changed frame takes the earlier
value for every household whose digest matches, runs the engine on the other
households plus a few unchanged ones as a check, and runs every household
again if that check disagrees.

These tests stand in for the engine with a function of each household's rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.household_pass_reuse import (
    HouseholdPassInputs,
    PassInputColumn,
    record_household_pass,
    reuse_household_pass,
)
from microcosm.frame import EntitySchema, Frame, MassChange, WeightKind, Weights

_SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))

_INPUTS = HouseholdPassInputs(
    variable="benefit",
    columns=(
        PassInputColumn("income", "person", default=0.0, absent_reads_default=True),
        PassInputColumn(
            "is_disabled", "person", default=False, absent_reads_default=True
        ),
        PassInputColumn(
            "claims_credit", "tax_unit", default=False, absent_reads_default=True
        ),
        PassInputColumn(
            "region", "household", default="NONE", absent_reads_default=True
        ),
        # A column the engine computes when it is absent: a stored value
        # replaces the formula, so its presence matters.
        PassInputColumn(
            "countable_income", "person", default=0.0, absent_reads_default=False
        ),
    ),
)


def _frame(
    *,
    income=(100.0, 0.0, 250.0, 0.0, 40.0, 0.0, 0.0),
    tax_units=(10, 10, 20, 30, 30, 40, 50),
    person_ids=(1, 2, 3, 4, 5, 6, 7),
    households=(1, 1, 2, 3, 3, 4, 5),
) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray(person_ids, dtype="int64"),
            "person_household_id": np.asarray(households, dtype="int64"),
            "person_tax_unit_id": np.asarray(tax_units, dtype="int64"),
            "income": np.asarray(income, dtype=np.float64),
            "unrelated_note": np.arange(len(person_ids), dtype=np.float64),
        }
    )
    tax_unit_ids = np.unique(person["person_tax_unit_id"].to_numpy())
    household_ids = np.unique(person["person_household_id"].to_numpy())
    tables = {
        "person": person,
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": tax_unit_ids,
                "claims_credit": tax_unit_ids % 20 == 0,
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "region": np.asarray(
                    [
                        "NORTH" if household % 2 else "SOUTH"
                        for household in household_ids
                    ],
                    dtype=object,
                ),
            }
        ),
    }
    return Frame(
        tables,
        _SCHEMA,
        {
            "household": Weights(
                np.arange(1.0, len(household_ids) + 1.0), WeightKind.DESIGN
            )
        },
    )


def _with_tables(frame: Frame, **tables: pd.DataFrame) -> Frame:
    return Frame(
        {entity: tables.get(entity, frame.table(entity)) for entity in frame.entities},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )


class _StandInEngine:
    """A per-household function of the rows, recording the households it saw."""

    def __init__(self, *, offset: float = 0.0) -> None:
        self.offset = offset
        self.calls: list[list[int]] = []

    def __call__(self, frame: Frame) -> np.ndarray:
        person = frame.table("person")
        tax_unit = frame.table("tax_unit").set_index("tax_unit_id")
        household = frame.table("household").set_index("household_id")
        self.calls.append(sorted(household.index.tolist()))
        income = person["income"].to_numpy(dtype=np.float64)
        if "countable_income" in person:
            income = person["countable_income"].to_numpy(dtype=np.float64)
        unit_total = person.groupby("person_tax_unit_id")["income"].transform("sum")
        credit = tax_unit["claims_credit"].reindex(person["person_tax_unit_id"])
        region = household["region"].reindex(person["person_household_id"])
        disabled = (
            person["is_disabled"].to_numpy(dtype=bool)
            if "is_disabled" in person
            else np.zeros(len(person), dtype=bool)
        )
        # Each person's place among the people of their household, so the
        # order of rows inside a household matters as it can to an engine.
        place_in_household = person.groupby("person_household_id").cumcount()
        return (
            income
            + 0.5 * unit_total.to_numpy()
            + 1_000.0 * credit.to_numpy(dtype=bool)
            + 10.0 * (region.to_numpy() == "NORTH")
            + 7.0 * disabled
            + 0.25 * place_in_household.to_numpy()
            + self.offset
        )


def _record(frame: Frame, engine: _StandInEngine):
    return record_household_pass(frame, engine(frame), _INPUTS)


def _reuse(record, frame: Frame, engine: _StandInEngine, *, checked: int = 1):
    return reuse_household_pass(
        record, frame, recompute=engine, checked_household_count=checked
    )


def test_unchanged_households_take_their_recorded_values() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)

    result = _reuse(record, frame, engine, checked=2)

    assert result.values.tobytes() == engine(frame).tobytes()
    assert (result.reused_households, result.recomputed_households) == (5, 0)
    assert result.checked_households == 2
    assert result.check_failed is False
    # One run over the two checked households only.
    assert engine.calls[1] == [1, 2]


def test_a_changed_input_reruns_only_its_household() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    person = frame.table("person").copy()
    person.loc[person["person_id"] == 4, "income"] = 75.0
    changed = _with_tables(frame, person=person)

    result = _reuse(record, changed, engine, checked=0)

    assert result.values.tobytes() == engine(changed).tobytes()
    assert (result.reused_households, result.recomputed_households) == (4, 1)
    assert engine.calls[1] == [3]


def test_columns_outside_the_inputs_are_not_changes() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    person = frame.table("person").copy()
    person["unrelated_note"] = -1.0
    person["another_column"] = 5
    household = frame.table("household").copy()
    household["household_weight_copy"] = 3.0

    result = _reuse(
        record,
        _with_tables(frame, person=person, household=household),
        engine,
        checked=0,
    )

    assert (result.reused_households, result.recomputed_households) == (5, 0)
    assert len(engine.calls) == 1


def test_an_absent_input_reads_as_its_default() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)

    person = frame.table("person").copy()
    person["is_disabled"] = False
    defaults_written = _with_tables(frame, person=person)
    result = _reuse(record, defaults_written, engine, checked=0)
    assert (result.reused_households, result.recomputed_households) == (5, 0)

    person.loc[person["person_id"] == 6, "is_disabled"] = True
    one_changed = _with_tables(frame, person=person)
    result = _reuse(record, one_changed, engine, checked=0)
    assert (result.reused_households, result.recomputed_households) == (4, 1)
    assert result.values.tobytes() == engine(one_changed).tobytes()


def test_a_stored_computed_column_reruns_every_household() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    person = frame.table("person").copy()
    # Zero everywhere, so it differs from the formula only where income is
    # positive, but any stored value replaces the formula.
    person["countable_income"] = 0.0
    stored = _with_tables(frame, person=person)

    result = _reuse(record, stored, engine, checked=0)

    assert (result.reused_households, result.recomputed_households) == (0, 5)
    assert result.values.tobytes() == engine(stored).tobytes()


def test_group_and_household_inputs_count_for_their_household() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    tax_unit = frame.table("tax_unit").copy()
    tax_unit.loc[tax_unit["tax_unit_id"] == 30, "claims_credit"] = True
    household = frame.table("household").copy()
    household.loc[household["household_id"] == 5, "region"] = "SOUTH"
    changed = _with_tables(frame, tax_unit=tax_unit, household=household)

    result = _reuse(record, changed, engine, checked=0)

    assert (result.reused_households, result.recomputed_households) == (3, 2)
    assert engine.calls[1] == [3, 5]
    assert result.values.tobytes() == engine(changed).tobytes()


def test_new_weights_and_dropped_households_reuse_values_by_person_id() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    kept = frame.table("person")["person_household_id"].isin([1, 3, 5]).to_numpy()
    selected = frame.select(kept)
    reweighted = selected.with_weights(
        "household",
        Weights(np.asarray([9.0, 0.5, 2.0]), WeightKind.CALIBRATED),
        mass=MassChange(factor=11.5 / 9.0, reason="test calibration"),
    )

    result = _reuse(record, reweighted, engine, checked=0)

    expected = pd.Series(engine(frame), index=frame.table("person")["person_id"])
    person_ids = reweighted.table("person")["person_id"].to_numpy()
    assert result.values.tobytes() == expected.reindex(person_ids).to_numpy().tobytes()
    assert (result.reused_households, result.recomputed_households) == (3, 0)


@pytest.mark.parametrize(
    "change",
    [
        # Two people in household 1 swap incomes.
        dict(income=(0.0, 100.0, 250.0, 0.0, 40.0, 0.0, 0.0)),
        # Person 5 moves to a new tax unit inside household 3.
        dict(tax_units=(10, 10, 20, 30, 31, 40, 50)),
        # Household 3's two people change places in the table.
        dict(
            person_ids=(1, 2, 3, 5, 4, 6, 7),
            income=(100.0, 0.0, 250.0, 40.0, 0.0, 0.0, 0.0),
            tax_units=(10, 10, 20, 30, 30, 40, 50),
        ),
        # Income of -0.0 instead of 0.0: the same number, different bits.
        dict(income=(100.0, 0.0, 250.0, -0.0, 40.0, 0.0, 0.0)),
    ],
    ids=["swapped_values", "moved_tax_unit", "reordered_rows", "negative_zero"],
)
def test_changes_inside_a_household_rerun_it(change) -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    changed = _frame(**change)

    result = _reuse(record, changed, engine, checked=0)

    assert result.recomputed_households == 1
    assert result.values.tobytes() == engine(changed).tobytes()


def test_wider_storage_of_the_same_values_is_not_a_change() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    person = frame.table("person").copy()
    person["income"] = person["income"].astype(np.float32)
    tax_unit = frame.table("tax_unit").copy()
    tax_unit["claims_credit"] = tax_unit["claims_credit"].astype(np.int8)

    result = _reuse(
        record, _with_tables(frame, person=person, tax_unit=tax_unit), engine, checked=0
    )

    assert (result.reused_households, result.recomputed_households) == (5, 0)


def test_households_missing_from_the_record_are_rerun() -> None:
    frame = _frame()
    engine = _StandInEngine()
    record = _record(frame, engine)
    grown = _frame(
        income=(100.0, 0.0, 250.0, 0.0, 40.0, 0.0, 0.0, 30.0),
        tax_units=(10, 10, 20, 30, 30, 40, 50, 60),
        person_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        households=(1, 1, 2, 3, 3, 4, 5, 6),
    )

    result = _reuse(record, grown, engine, checked=0)

    assert (result.reused_households, result.recomputed_households) == (5, 1)
    assert engine.calls[1] == [6]


def test_a_failed_check_reruns_every_household() -> None:
    """The recorded values disagree with a fresh run on a checked household,
    as they would if the pass read a column missing from its input list."""

    frame = _frame()
    record = _record(frame, _StandInEngine())
    engine = _StandInEngine(offset=1.0)

    result = _reuse(record, frame, engine, checked=1)

    assert result.check_failed is True
    # The check ran on household 1, then every household ran again.
    assert engine.calls == [[1], [1, 2, 3, 4, 5]]
    assert result.values.tobytes() == engine(frame).tobytes()


def test_record_and_reuse_refuse_misaligned_values() -> None:
    frame = _frame()
    engine = _StandInEngine()
    with pytest.raises(ValueError, match="one value per person"):
        record_household_pass(frame, engine(frame)[:-1], _INPUTS)

    record = _record(frame, engine)
    person = frame.table("person").copy()
    person.loc[person["person_id"] == 4, "income"] = 75.0
    with pytest.raises(ValueError, match="one value per person"):
        reuse_household_pass(
            record,
            _with_tables(frame, person=person),
            recompute=lambda run_frame: np.zeros(1, dtype=np.float64),
            checked_household_count=0,
        )
    with pytest.raises(ValueError, match="dtype"):
        reuse_household_pass(
            record,
            _with_tables(frame, person=person),
            recompute=lambda run_frame: np.zeros(run_frame.n("person"), dtype=bool),
            checked_household_count=0,
        )
