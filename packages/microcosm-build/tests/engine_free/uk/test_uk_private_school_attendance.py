"""Private-school attendance stored at the spine's weights (uk-data#568, microcosm#1095)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.private_school_attendance import (
    PRIVATE_SCHOOL_ATTENDANCE_DRAW,
    PRIVATE_SCHOOL_ATTENDANCE_OUTPUT,
    UKPrivateSchoolAttendanceStageTransform,
    store_private_school_attendance,
)


class _FakeEngine:
    """The engine's attendance at the frame's weights, recorded per call."""

    country = "uk"

    def __init__(self, attends):
        self.attends = np.asarray(attends, dtype=bool)
        self.calls = []

    def materialize(self, frame, variables, period, *, dataset_period=None):
        self.calls.append(
            (
                tuple(variables),
                period,
                dataset_period,
                frame.weights_for("household").values.tolist(),
            )
        )
        return {PRIVATE_SCHOOL_ATTENDANCE_OUTPUT: self.attends}


def _frame(**person_columns):
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
        household_weights=np.array([100.0, 50.0, 30.0]),
        time_period="2024",
    )


def _stage():
    return load_country_spec("uk").sources.stage_map()["private_school_attendance"]


def test_the_stage_stores_the_engines_attendance_at_the_spines_weights():
    engine = _FakeEngine([False, True, True, False])
    result = store_private_school_attendance(
        _frame(), engine=engine, calibration_year=2025
    )

    person = result.frame.table("person")
    assert person[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert person[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT].dtype == bool
    # One materialization at the calibration year, the frame's own period as
    # the dataset year, at the spine's design weights.
    ((variables, period, dataset_period, weights),) = engine.calls
    assert variables == (PRIVATE_SCHOOL_ATTENDANCE_OUTPUT,)
    assert (period, dataset_period) == (2025, "2024")
    assert weights == [100.0, 50.0, 30.0]
    # The weights pass through and the receipt counts weighted pupils.
    assert result.frame.weights_for("household").values.tolist() == weights
    assert result.weighted_pupils == 150.0
    assert result.evidence()["weights"] == "spine_design_weights"


def test_the_stage_needs_the_stored_draw_and_writes_attendance_once():
    engine = _FakeEngine([False] * 4)
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


def test_only_the_isa_subscriptions_follow_the_stage():
    """No later stage copies or re-keys a person, so the stored flags reach
    the release as the engine set them. Only the ISA subscriptions follow it,
    writing person ISA columns on the rows they are given (microcosm#1415)."""

    stages = load_country_spec("uk").sources.stages
    names = [stage.stage for stage in stages]
    later = stages[names.index("private_school_attendance") + 1 :]
    assert [stage.stage for stage in later] == ["isa_subscriptions"]
    for stage in later:
        assert PRIVATE_SCHOOL_ATTENDANCE_OUTPUT not in {
            *stage.outputs,
            *stage.rewrites,
        }


def test_the_stage_refuses_drifted_parameters():
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
        stage=drifted, engine=_FakeEngine([False] * 4), calibration_year=2025
    )
    with pytest.raises(ValueError, match="drifted"):
        transform(_frame())
    stored = UKPrivateSchoolAttendanceStageTransform(
        stage=stage,
        engine=_FakeEngine([True, False, False, False]),
        calibration_year=2025,
    )(_frame())
    assert stored.table("person")[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT].tolist() == [
        True,
        False,
        False,
        False,
    ]
