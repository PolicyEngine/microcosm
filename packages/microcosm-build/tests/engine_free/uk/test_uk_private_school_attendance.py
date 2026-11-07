"""Private-school attendance stored at the spine's weights (uk-data#568, microcosm#1095).

The stage restates policyengine-uk's formula over the frame from the engine's
own values and parameters (microcosm#931). The rank in it is microdf's, the
rank the formula calls, which the engine test holds to the formula bit for bit
(tests/engine/uk/test_uk_private_school_pooled_rank.py). Here a stand-in rank
records what the stage hands it, so the stage's arithmetic, both its routes,
its parity check and its floor run without the engine.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime import private_school_attendance
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.private_school_attendance import (
    PRIVATE_SCHOOL_ATTENDANCE_DRAW,
    PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE,
    PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS,
    PRIVATE_SCHOOL_ATTENDANCE_OUTPUT,
    PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS,
    PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE,
    UKPrivateSchoolAttendanceStageTransform,
    store_private_school_attendance,
)

INPUTS = (
    *PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS,
    *PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS,
)
#: Household weights whose people (each household's weight once per member)
#: weigh above the formula's one million in all, and weights far under it.
WEIGHTS = (400_000.0, 300_000.0, 200_000.0)
LIGHT_WEIGHTS = (100.0, 50.0, 30.0)
#: The stand-in rank's percentiles for the four people. With the made-up rates
#: (a percentile's rate is the percentile over 100), the factor (0.5), the
#: under-18s (people 2 and 3) and the draws, only person 2 attends.
PERCENTILES = (80, 80, 30, 50)
ATTENDS = [False, True, False, False]


def _parameters(factor=0.5):
    vat = SimpleNamespace(
        private_school_attendance_rate={str(p): p / 100 for p in range(0, 101, 5)},
        private_school_factor=factor,
    )
    return SimpleNamespace(
        gov=SimpleNamespace(simulation=SimpleNamespace(private_school_vat=vat))
    )


def _engine_values(weights):
    """What the engine returns for the restatement: people in the frame's
    order, households by id, in the engine's own types."""

    return {
        "income_tax": np.array([1_000.0, 0.0, 0.0, 200.0], dtype=np.float32),
        "national_insurance": np.array([500.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "age_under_18": np.array([False, True, True, False]),
        PRIVATE_SCHOOL_ATTENDANCE_DRAW: np.array(
            [0.5, 0.01, 0.2, 0.9], dtype=np.float32
        ),
        "household_market_income": np.array(
            [30_000.0, 10_000.0, 15_000.0], dtype=np.float32
        ),
        "household_benefits": np.array([2_000.0, 5_000.0, 1_000.0], dtype=np.float32),
        "household_weight": np.array(weights, dtype=np.float32),
    }


class _FakeEngine:
    """An engine that computes the frame in one simulation, with no
    ``simulation_batches``: its attendance and the restatement's inputs,
    recorded per call, and its parameters."""

    country = "uk"

    def __init__(self, attends=ATTENDS, *, weights=WEIGHTS):
        self.attends = np.asarray(attends, dtype=bool)
        self.values = _engine_values(weights)
        self.calls = []
        self.parameter_periods = []

    def materialize(self, frame, variables, period, *, dataset_period=None):
        self.calls.append(
            (
                tuple(variables),
                period,
                dataset_period,
                frame.weights_for("household").values.tolist(),
            )
        )
        values = {**self.values, PRIVATE_SCHOOL_ATTENDANCE_OUTPUT: self.attends}
        return {name: values[name] for name in variables}

    def parameters(self, period):
        self.parameter_periods.append(period)
        return _parameters()


class _BatchingEngine(_FakeEngine):
    """An engine that says how many simulations it computes the frame in."""

    def __init__(self, batches, attends=ATTENDS, **kwargs):
        super().__init__(attends, **kwargs)
        self.batches = batches

    def simulation_batches(self, frame):
        return self.batches

    def materialize(self, frame, variables, period, *, dataset_period=None):
        if self.batches > 1 and PRIVATE_SCHOOL_ATTENDANCE_OUTPUT in variables:
            raise AssertionError("a call clone by clone refuses attendance")
        return super().materialize(
            frame, variables, period, dataset_period=dataset_period
        )


@pytest.fixture
def ranks(monkeypatch):
    """A stand-in for microdf's rank that records what the stage hands it."""

    seen = []

    def stand_in(net_income, weights):
        seen.append((net_income, weights))
        return np.asarray(PERCENTILES, dtype=np.int64)

    monkeypatch.setattr(private_school_attendance, "_net_income_percentiles", stand_in)
    return seen


def _frame(weights=WEIGHTS, **person_columns):
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [1, 1, 2, 3],
            "age": [40, 9, 12, 70],
            PRIVATE_SCHOOL_ATTENDANCE_DRAW: [0.5, 0.01, 0.2, 0.9],
            **person_columns,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": [10, 20, 30]}),
        household=pd.DataFrame({"household_id": [1, 2, 3], "region": ["LONDON"] * 3}),
        household_weights=np.array(weights),
        time_period="2024",
    )


def _stage():
    return load_country_spec("uk").sources.stage_map()["private_school_attendance"]


def _stored(result):
    return result.frame.table("person")[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT]


def test_one_simulation_stores_the_engines_attendance_once_restated(ranks):
    engine = _FakeEngine()
    result = store_private_school_attendance(
        _frame(), engine=engine, calibration_year=2025
    )

    assert _stored(result).tolist() == ATTENDS
    assert _stored(result).dtype == bool
    # One materialization at the calibration year, the frame's own period as
    # the dataset year, at the spine's design weights: the engine's attendance
    # beside what the restatement reads, and the engine's parameters there.
    ((variables, period, dataset_period, weights),) = engine.calls
    assert variables == (PRIVATE_SCHOOL_ATTENDANCE_OUTPUT, *INPUTS)
    assert (period, dataset_period) == (2025, "2024")
    assert weights == list(WEIGHTS)
    assert engine.parameter_periods == [2025]
    # The rank reads each person's household net income (market income plus
    # benefits less the members' income tax and National Insurance) and the
    # household's weight once per member, in the engine's types.
    ((net_income, person_weights),) = ranks
    assert net_income.dtype == np.float64
    assert net_income.tolist() == [30_500.0, 30_500.0, 15_000.0, 15_800.0]
    assert person_weights.dtype == np.float32
    assert person_weights.tolist() == [400_000.0, 400_000.0, 300_000.0, 200_000.0]
    # The weights pass through and the receipt counts weighted pupils.
    assert result.frame.weights_for("household").values.tolist() == weights
    assert result.weighted_pupils == 400_000.0
    evidence = result.evidence()
    assert evidence["weights"] == "spine_design_weights"
    assert evidence["rank_route"] == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
    assert evidence["engine_parity_checked"] is True


def test_flags_the_restatement_does_not_reproduce_are_refused(ranks):
    engine = _FakeEngine([False, True, True, False])

    with pytest.raises(ValueError, match="differs from the engine's flags") as error:
        store_private_school_attendance(_frame(), engine=engine, calibration_year=2025)
    # It says which formula, never how many people differ.
    assert not any(character.isdigit() for character in str(error.value))


def test_a_frame_computed_clone_by_clone_stores_the_restatement(ranks):
    engine = _BatchingEngine(3, attends=[True] * 4)
    result = store_private_school_attendance(
        _frame(), engine=engine, calibration_year=2025
    )

    assert _stored(result).tolist() == ATTENDS
    ((variables, period, dataset_period, _weights),) = engine.calls
    assert variables == INPUTS
    assert (period, dataset_period) == (2025, "2024")
    evidence = result.evidence()
    assert evidence["rank_route"] == PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE
    assert evidence["engine_parity_checked"] is False
    assert result.weighted_pupils == 400_000.0


def test_one_batch_is_one_simulation(ranks):
    engine = _BatchingEngine(1)
    result = store_private_school_attendance(
        _frame(), engine=engine, calibration_year=2025
    )

    ((variables, *_),) = engine.calls
    assert variables == (PRIVATE_SCHOOL_ATTENDANCE_OUTPUT, *INPUTS)
    assert result.evidence()["rank_route"] == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
    for batches in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive whole number"):
            store_private_school_attendance(
                _frame(), engine=_BatchingEngine(batches), calibration_year=2025
            )


def test_under_one_million_people_of_weight_nobody_attends_on_either_route(ranks):
    for engine in (
        _FakeEngine([False] * 4, weights=LIGHT_WEIGHTS),
        _BatchingEngine(3, attends=[True] * 4, weights=LIGHT_WEIGHTS),
    ):
        result = store_private_school_attendance(
            _frame(LIGHT_WEIGHTS), engine=engine, calibration_year=2025
        )
        assert not _stored(result).any()
    # The floor comes before the rank, which never runs.
    assert ranks == []
    with pytest.raises(ValueError, match="differs from the engine's flags"):
        store_private_school_attendance(
            _frame(LIGHT_WEIGHTS),
            engine=_FakeEngine(ATTENDS, weights=LIGHT_WEIGHTS),
            calibration_year=2025,
        )


def test_the_stage_needs_the_engines_parameters_and_its_inputs(ranks):
    class _NoParameters(_FakeEngine):
        parameters = None

    with pytest.raises(TypeError, match="parameters"):
        store_private_school_attendance(
            _frame(), engine=_NoParameters(), calibration_year=2025
        )

    class _Partial(_FakeEngine):
        def materialize(self, frame, variables, period, *, dataset_period=None):
            values = super().materialize(
                frame, variables, period, dataset_period=dataset_period
            )
            values.pop("household_benefits")
            return values

    with pytest.raises(ValueError, match="household_benefits"):
        store_private_school_attendance(
            _frame(), engine=_Partial(), calibration_year=2025
        )


def test_the_stage_needs_the_stored_draw_and_writes_attendance_once(ranks):
    engine = _FakeEngine()
    frame = _frame()
    no_draw = uk_national_frame(
        person=frame.table("person").drop(columns=[PRIVATE_SCHOOL_ATTENDANCE_DRAW]),
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values,
        time_period="2024",
    )
    with pytest.raises(ValueError, match="random_draw"):
        store_private_school_attendance(no_draw, engine=engine, calibration_year=2025)
    stored = _frame(**{PRIVATE_SCHOOL_ATTENDANCE_OUTPUT: [False] * 4})
    with pytest.raises(ValueError, match="only writer"):
        store_private_school_attendance(stored, engine=engine, calibration_year=2025)
    assert engine.calls == []


def test_only_the_isa_subscriptions_and_receipts_follow_the_stage():
    """No later stage copies or re-keys a person, so the stored flags reach
    the release as the engine set them. Only the ISA subscriptions and the
    self-employed's gross receipts follow it, each writing person columns on
    the rows they are given (microcosm#1415, microcosm#1428)."""

    stages = load_country_spec("uk").sources.stages
    names = [stage.stage for stage in stages]
    later = stages[names.index("private_school_attendance") + 1 :]
    assert [stage.stage for stage in later] == [
        "isa_subscriptions",
        "trading_allowance_receipts",
    ]
    for stage in later:
        assert PRIVATE_SCHOOL_ATTENDANCE_OUTPUT not in {
            *stage.outputs,
            *stage.rewrites,
        }


def test_the_stage_refuses_drifted_parameters(ranks):
    stage = _stage()
    operation, receipt = stage.operations
    drifted = replace(
        stage,
        operations=(
            SourceOperationSpec(
                operation.kind, {**operation.parameters, "year_rule": "survey_year"}
            ),
            receipt,
        ),
    )
    transform = UKPrivateSchoolAttendanceStageTransform(
        stage=drifted, engine=_FakeEngine(), calibration_year=2025
    )
    with pytest.raises(ValueError, match="drifted"):
        transform(_frame())
    transform = UKPrivateSchoolAttendanceStageTransform(
        stage=stage, engine=_FakeEngine(), calibration_year=2025
    )
    stored = transform(_frame())
    assert stored.table("person")[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT].tolist() == ATTENDS
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["rank_route"] == PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
