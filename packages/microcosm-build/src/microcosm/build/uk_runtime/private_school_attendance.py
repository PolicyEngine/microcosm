"""Private-school attendance stored at the spine's weights (uk-data#568, microcosm#1095).

policyengine-uk imputes ``attends_private_school`` with a formula: it ranks
households by net income at their weights, reads an attendance rate for each
income percentile, scales it to the national pupil count and compares it with
the stored ``attends_private_school_random_draw``. Left to the formula, the
attendance is re-imputed in every simulation:

* a reform to income tax, National Insurance or benefits re-ranks households,
  so pupils switch in or out and the fee VAT moves with them (uk-data#568);
* a population under one million people of weight, such as a constituency or
  a local authority filtered from the national file, imputes no attendance at
  all (pe-uk#1905);
* the calibration's engine run and the release's read different flags once the
  weights change.

This stage, the last of the spine, materializes the engine's attendance once,
at the calibration year and the spine's design weights, and stores it, as the
root stage stores ``months_since_last_birthday``. The solve and the release
then read the same flags, and no stage after it copies or re-keys a person.
Stored attendance no longer responds to income changes or to reforms of the
attendance parameters; a reform to the fee VAT does not move pupils either way
in the pinned engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame, MassChangeRecord
from microcosm.frame.rules import assert_rules_engine_country

PRIVATE_SCHOOL_ATTENDANCE_STAGE_NAME = "private_school_attendance"
PRIVATE_SCHOOL_ATTENDANCE_OUTPUT = "attends_private_school"
PRIVATE_SCHOOL_ATTENDANCE_DRAW = "attends_private_school_random_draw"
PRIVATE_SCHOOL_ATTENDANCE_YEAR_RULE = "calibration_year"
PRIVATE_SCHOOL_ATTENDANCE_WEIGHTS = "spine_design_weights"
PRIVATE_SCHOOL_ATTENDANCE_MASS_CHANGE_REASON = (
    "Private-school attendance stored on the source spine: household weights "
    "pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)


@dataclass(frozen=True)
class UKPrivateSchoolAttendanceResult:
    """The frame with the stored flags, and the stage's receipt."""

    frame: Frame
    year: int
    weighted_pupils: float

    def evidence(self) -> dict[str, object]:
        return {
            "stage": PRIVATE_SCHOOL_ATTENDANCE_STAGE_NAME,
            "year_rule": PRIVATE_SCHOOL_ATTENDANCE_YEAR_RULE,
            "year": self.year,
            "weights": PRIVATE_SCHOOL_ATTENDANCE_WEIGHTS,
            # A national weighted total over every pupil the engine imputes.
            "weighted_pupils": self.weighted_pupils,
        }


@dataclass(frozen=True)
class UKPrivateSchoolAttendanceStageTransform:
    """Whole-stage callable for the stored private-school attendance."""

    stage: SourceStageSpec
    engine: object
    calibration_year: int | None = None
    last_result: UKPrivateSchoolAttendanceResult | None = field(
        default=None, init=False
    )

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = store_private_school_attendance(
            frame, engine=self.engine, calibration_year=self.calibration_year
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (PRIVATE_SCHOOL_ATTENDANCE_OUTPUT,)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def store_private_school_attendance(
    frame: Frame, *, engine: object, calibration_year: int | None = None
) -> UKPrivateSchoolAttendanceResult:
    """Store the engine's attendance at the calibration year and these weights."""

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    if PRIVATE_SCHOOL_ATTENDANCE_DRAW not in person.columns:
        raise ValueError(
            "Private-school attendance needs the stored "
            f"{PRIVATE_SCHOOL_ATTENDANCE_DRAW!r}; the engine would draw its own."
        )
    if PRIVATE_SCHOOL_ATTENDANCE_OUTPUT in person.columns:
        raise ValueError(
            f"{PRIVATE_SCHOOL_ATTENDANCE_OUTPUT!r} is already stored: this stage "
            "is its only writer."
        )
    period = uk_time_period(frame)
    year = (
        calibration_year
        if calibration_year is not None
        else resolve_uk_year_rule(PRIVATE_SCHOOL_ATTENDANCE_YEAR_RULE)
    )
    materialized = engine.materialize(
        frame, [PRIVATE_SCHOOL_ATTENDANCE_OUTPUT], year, dataset_period=period
    )
    if PRIVATE_SCHOOL_ATTENDANCE_OUTPUT not in materialized:
        raise ValueError(
            "Private-school attendance engine output missing: "
            f"{[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT]}."
        )
    values = np.asarray(materialized[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT])
    if values.shape != (len(person),):
        raise ValueError(
            f"{PRIVATE_SCHOOL_ATTENDANCE_OUTPUT} has {values.shape[0]} values for "
            f"{len(person)} people."
        )
    attends = values.astype(bool)
    person[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT] = attends
    household_weights = frame.weights_for("household")
    weight_by_household = dict(
        zip(
            frame.table("household")["household_id"].to_numpy(),
            np.asarray(household_weights.values, dtype=float),
            strict=True,
        )
    )
    person_weights = (
        person["person_household_id"].map(weight_by_household).to_numpy(dtype=float)
    )
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=PRIVATE_SCHOOL_ATTENDANCE_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=frame.table("household").copy(),
        time_period=period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=household_weights.values,
        mass_log=(*frame.mass_log, mass),
    )
    validate_uk_national_frame(result)
    return UKPrivateSchoolAttendanceResult(
        frame=result,
        year=int(year),
        weighted_pupils=float(person_weights[attends].sum()),
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage's two operations closed-world."""

    expected: tuple[tuple[str, Mapping[str, object]], ...] = (
        (
            "store_rules_engine_input",
            {
                "output": PRIVATE_SCHOOL_ATTENDANCE_OUTPUT,
                "draw": PRIVATE_SCHOOL_ATTENDANCE_DRAW,
                "year_rule": PRIVATE_SCHOOL_ATTENDANCE_YEAR_RULE,
                "weights": PRIVATE_SCHOOL_ATTENDANCE_WEIGHTS,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": PRIVATE_SCHOOL_ATTENDANCE_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )
    operations = tuple(stage.operations)
    if tuple(operation.kind for operation in operations) != tuple(
        kind for kind, _ in expected
    ):
        raise ValueError(
            "private_school_attendance operation order drifted: "
            f"{[operation.kind for operation in operations]!r}."
        )
    for operation, (kind, parameters) in zip(operations, expected, strict=True):
        if dict(operation.parameters) != dict(parameters):
            raise ValueError(
                f"private_school_attendance {kind} parameters drifted: "
                f"{dict(operation.parameters)!r} != {dict(parameters)!r}."
            )
