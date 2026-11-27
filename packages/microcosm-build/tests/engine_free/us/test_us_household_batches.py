"""Household batches shared by the US release's policyengine-us passes.

``microcosm.build.us_runtime.household_batches`` splits a frame's households
into contiguous batches, runs the frame checks that batching needs, and runs
each batch engine inside :func:`checked_batch_engine`, which refuses a
population-aggregate variable before the engine is released. These tests use
stand-in engines; ``test_us_input_batch_loops.py`` runs the real engine.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.household_batches as household_batches
from microcosm.build.us_runtime.household_batches import (
    INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
    assert_group_entities_nest_in_households,
    assert_medicaid_claiming_tax_units_local,
    automatic_gc_suspended,
    checked_batch_engine,
    household_batch_plan,
    household_position_batches,
    refuse_batch_population_aggregates,
    select_households_by_position,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

AGGREGATE = "medicaid_slcsp_state_denominator"
WORK = "The probe pass"


def _frame(
    *,
    person_household_ids=(1, 1, 2, 3, 3),
    person_spm_unit_ids=(10, 10, 20, 30, 31),
    medicaid_claiming_tax_unit_ids=None,
) -> Frame:
    person_household_ids = np.asarray(person_household_ids, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(person_household_ids) + 1, dtype="int64"),
            "person_household_id": person_household_ids,
            "person_tax_unit_id": person_household_ids * 100,
            "person_spm_unit_id": np.asarray(person_spm_unit_ids, dtype="int64"),
            "person_family_id": person_household_ids * 1_000,
            "person_marital_unit_id": person_household_ids * 10_000,
        }
    )
    if medicaid_claiming_tax_unit_ids is not None:
        person["medicaid_claiming_tax_unit_id"] = np.asarray(
            medicaid_claiming_tax_unit_ids, dtype="int64"
        )
    tables = {"person": person}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(person[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    household_ids = np.unique(person_household_ids)
    tables["household"] = pd.DataFrame({"household_id": household_ids})
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(household_ids), dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


class _StandInEngine:
    """Holds known periods per variable and creates stand-in branches."""

    def __init__(self, known_periods=None) -> None:
        self.known_periods = dict(known_periods or {})
        self.branches: dict[str, _StandInEngine] = {}

    def get_holder(self, variable):
        return SimpleNamespace(
            get_known_periods=lambda: list(self.known_periods.get(variable, []))
        )

    def get_branch(self, name):
        branch = _StandInEngine()
        self.branches[name] = branch
        return branch


@pytest.fixture
def released(monkeypatch):
    """Record each engine ``checked_batch_engine`` releases, in order."""

    engines: list[object] = []
    monkeypatch.setattr(household_batches, "release_engine_simulation", engines.append)
    return engines


@pytest.mark.parametrize(
    ("n_households", "batch_size", "expected"),
    [
        (5, 2, [[0, 1], [2, 3], [4]]),
        (5, None, [[0, 1, 2, 3, 4]]),
        (5, 0, [[0, 1, 2, 3, 4]]),
        (5, -3, [[0, 1, 2, 3, 4]]),
        (5, 5, [[0, 1, 2, 3, 4]]),
        (5, 9, [[0, 1, 2, 3, 4]]),
        (0, 2, []),
    ],
)
def test_household_position_batches_are_contiguous_ranges(
    n_households, batch_size, expected
) -> None:
    batches = list(household_position_batches(n_households, batch_size))

    assert [batch.tolist() for batch in batches] == expected
    assert all(batch.dtype == np.int64 for batch in batches)


def test_selected_households_keep_their_people_and_groups() -> None:
    selected = select_households_by_position(_frame(), np.asarray([0, 2]))

    assert selected.table("household")["household_id"].tolist() == [1, 3]
    assert selected.table("person")["person_id"].tolist() == [1, 2, 4, 5]
    assert selected.table("spm_unit")["spm_unit_id"].tolist() == [10, 30, 31]


@pytest.mark.parametrize(
    ("batch_size", "refuse_population_aggregates", "batches", "refusing"),
    [
        pytest.param(2, None, 2, True, id="more-than-one-batch"),
        pytest.param(None, None, 1, False, id="one-batch"),
        pytest.param(None, True, 1, True, id="one-batch-switched-on"),
        pytest.param(2, False, 2, False, id="more-than-one-batch-switched-off"),
    ],
)
def test_batch_plan_switches_the_refusal_on_for_more_than_one_batch(
    batch_size, refuse_population_aggregates, batches, refusing
) -> None:
    plan = household_batch_plan(
        _frame(),
        batch_size,
        refuse_population_aggregates=refuse_population_aggregates,
        work=WORK,
    )

    assert len(plan.batches) == batches
    assert plan.refuse_population_aggregates is refusing


@pytest.mark.parametrize(
    ("batch_size", "refuse_population_aggregates", "refused"),
    [
        pytest.param(2, None, True, id="more-than-one-batch"),
        pytest.param(2, False, True, id="split-even-with-refusal-off"),
        pytest.param(None, True, True, id="one-batch-refusal-on"),
        pytest.param(None, None, False, id="one-batch-holds-the-pool"),
    ],
)
def test_batch_plan_refuses_a_group_that_spans_households(
    batch_size, refuse_population_aggregates, refused
) -> None:
    # SPM unit 20 holds a person from household 2 and one from household 3.
    frame = _frame(person_spm_unit_ids=(10, 10, 20, 20, 31))

    def plan():
        return household_batch_plan(
            frame,
            batch_size,
            refuse_population_aggregates=refuse_population_aggregates,
            work=WORK,
        )

    if refused:
        with pytest.raises(ValueError, match="spm_unit units must be nested"):
            plan()
    else:
        plan()


@pytest.mark.parametrize(
    ("batch_size", "refuse_population_aggregates", "refused"),
    [
        pytest.param(2, None, True, id="more-than-one-batch"),
        pytest.param(None, True, True, id="one-batch-refusal-on"),
        pytest.param(2, False, False, id="refusal-off"),
        pytest.param(None, None, False, id="one-batch-holds-the-pool"),
    ],
)
def test_batch_plan_refuses_a_medicaid_claim_outside_the_household(
    batch_size, refuse_population_aggregates, refused
) -> None:
    # Person 4 (household 3) is claimed by tax unit 100 (household 1).
    frame = _frame(medicaid_claiming_tax_unit_ids=(0, 0, 0, 100, 0))

    def plan():
        return household_batch_plan(
            frame,
            batch_size,
            refuse_population_aggregates=refuse_population_aggregates,
            work=WORK,
        )

    if refused:
        with pytest.raises(
            ValueError,
            match=(
                f"^{WORK} is not batch-invariant: a positive "
                "medicaid_claiming_tax_unit_id does not resolve"
            ),
        ):
            plan()
    else:
        plan()


def test_a_medicaid_claim_within_the_household_passes() -> None:
    frame = _frame(medicaid_claiming_tax_unit_ids=(100, 100, 0, 300, 0))

    assert_medicaid_claiming_tax_units_local(frame, work=WORK)
    assert_group_entities_nest_in_households(frame)


def test_refusal_names_the_pass_the_batch_and_the_aggregate() -> None:
    engine = _StandInEngine({AGGREGATE: ["2024"]})

    with pytest.raises(ValueError) as refusal:
        refuse_batch_population_aggregates(
            engine,
            batch=2,
            batches=3,
            work=WORK,
            remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
        )

    message = str(refusal.value)
    assert message.startswith(f"{WORK} is not batch-invariant: the engine for ")
    assert f"household batch 2/3 computed {AGGREGATE}@2024." in message
    assert message.endswith(INPUT_PASS_POPULATION_AGGREGATE_REMEDY)


def test_checked_engine_refuses_an_aggregate_from_a_dropped_branch(
    released,
) -> None:
    engine = _StandInEngine()

    with pytest.raises(ValueError, match=rf"batch 1/2 computed {AGGREGATE}@2024"):
        with checked_batch_engine(
            engine,
            refuse_population_aggregates=True,
            batch=1,
            batches=2,
            work=WORK,
            remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
        ):
            branch = engine.get_branch("probe")
            branch.known_periods[AGGREGATE] = ["2024"]
            del engine.branches["probe"]

    # The branch the batch dropped was kept for the check, then released,
    # and the batch engine itself was released after the check.
    assert released == [branch, engine]
    assert "get_branch" not in vars(engine)


def test_checked_engine_passes_a_household_local_batch_and_releases_it(
    released,
) -> None:
    engine = _StandInEngine({"is_medicaid_eligible": ["2024"]})

    with checked_batch_engine(
        engine,
        refuse_population_aggregates=True,
        batch=1,
        batches=2,
        work=WORK,
        remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
    ) as checked:
        assert checked is engine

    assert released == [engine]


def test_checked_engine_with_the_refusal_off_keeps_an_aggregate(released) -> None:
    engine = _StandInEngine({AGGREGATE: ["2024"]})

    with checked_batch_engine(
        engine,
        refuse_population_aggregates=False,
        batch=1,
        batches=1,
        work=WORK,
        remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
    ):
        assert "get_branch" not in vars(engine)

    assert released == [engine]


@pytest.mark.parametrize("refusing", [True, False])
def test_checked_engine_releases_a_batch_that_raised(released, refusing) -> None:
    # The batch raised before the check, so the aggregate is not reported.
    engine = _StandInEngine({AGGREGATE: ["2024"]})

    with pytest.raises(RuntimeError, match="batch failed"):
        with checked_batch_engine(
            engine,
            refuse_population_aggregates=refusing,
            batch=1,
            batches=2,
            work=WORK,
            remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
        ):
            raise RuntimeError("batch failed")

    assert released == [engine]


@pytest.mark.parametrize("enabled", [True, False])
def test_automatic_gc_is_suspended_inside_a_batch_and_restored(enabled) -> None:
    was_enabled = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        with pytest.raises(RuntimeError):
            with automatic_gc_suspended():
                assert not gc.isenabled()
                raise RuntimeError("batch failed")
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if was_enabled else gc.disable)()
