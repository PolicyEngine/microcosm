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

This stage, the last of the spine, stores attendance once, at the calibration
year and the spine's design weights, as the root stage stores
``months_since_last_birthday``. The solve and the release then read the same
flags, and no stage after it copies or re-keys a person.
Stored attendance no longer responds to income changes or to reforms of the
attendance parameters; a reform to the fee VAT does not move pupils either way
in the pinned engine.

The rank is over the people a simulation holds, so a simulation over part of
the frame ranks that part alone. Once a dense spine's households carry more
than one geographic clone, its engine calls run clone by clone
(microcosm#931), so the stage restates the formula's last step itself, over
every person in the frame (:func:`restate_private_school_attendance`): from
engine values that each household's own records decide, the engine's own
parameters, and microdf's weighted percentile rank, the rank the formula
calls. Where the engine computes the frame in one simulation (the national
line, and a spine before its clones exist) the stage stores the engine's
flags once the restatement equals them bit for bit, so every such build
re-checks the restatement on its own data; where it computes the frame clone
by clone the stage stores the restatement. The receipt names the route and
whether the engine's flags were checked, never a count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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
#: How the stored flags were reached: the engine's own, once the restatement
#: equalled them bit for bit (one simulation over the frame), or the
#: restatement over the whole frame (the engine computes it clone by clone).
PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE = "engine_with_parity"
PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE = "pooled_restatement"
#: What policyengine-uk's formula reads (2.174.1,
#: variables/contrib/labour/attends_private_school.py, a source the UK engine
#: declaration pins), each decided by a household's own records, so the
#: engine returns it alike over the frame and clone by clone.
PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS = (
    "income_tax",
    "national_insurance",
    "age_under_18",
    PRIVATE_SCHOOL_ATTENDANCE_DRAW,
)
PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS = (
    "household_market_income",
    "household_benefits",
    "household_weight",
)
#: The engine parameters the formula reads, the only ones the restatement
#: reads: the attendance rate at every fifth percentile, and the population
#: factor.
PRIVATE_SCHOOL_ATTENDANCE_PARAMETERS = (
    "gov.simulation.private_school_vat.private_school_attendance_rate",
    "gov.simulation.private_school_vat.private_school_factor",
)
# The formula's own constants: the percentiles its attendance rates are given
# at, the percentiles it looks rates up for, and the total weight of people
# (a household's weight once per member) below which it imputes no attendance
# (pe-uk#1905). The pinned source and the parity check hold these copies to it.
_RATE_PERCENTILES = tuple(range(0, 101, 5))
_LOOKUP_PERCENTILES = 101
_IMPUTED_POPULATION_FLOOR = 1e6


@dataclass(frozen=True)
class UKPrivateSchoolAttendanceResult:
    """The frame with the stored flags, and the stage's receipt."""

    frame: Frame
    year: int
    weighted_pupils: float
    rank_route: str
    engine_parity_checked: bool

    def evidence(self) -> dict[str, object]:
        return {
            "stage": PRIVATE_SCHOOL_ATTENDANCE_STAGE_NAME,
            "year_rule": PRIVATE_SCHOOL_ATTENDANCE_YEAR_RULE,
            "year": self.year,
            "weights": PRIVATE_SCHOOL_ATTENDANCE_WEIGHTS,
            # A national weighted total over every pupil the stage stores.
            "weighted_pupils": self.weighted_pupils,
            # The route's name, and whether the engine's own flags were checked
            # against the restatement: True over one simulation; False where
            # the engine computes the frame clone by clone and no simulation
            # holds the pool's flags. A check that fails raises, so the field
            # never records a mismatch.
            "rank_route": self.rank_route,
            "engine_parity_checked": self.engine_parity_checked,
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
    """Store policyengine-uk's attendance at the calibration year and these weights.

    ``engine`` computes the frame's values (``materialize``) and gives its own
    parameters (``parameters(period)``). Where it computes the frame in one
    simulation, the stage stores the engine's flags once the restatement over
    the frame equals them bit for bit, and raises otherwise; where it computes
    the frame in more (``simulation_batches``: one per clone; an engine
    without the method runs one), the stage stores the restatement.
    """

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
    parameters = _engine_parameters(engine, year)
    batched = _simulation_batches(engine, frame) > 1
    inputs = (
        *PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS,
        *PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS,
    )
    # A call that runs clone by clone refuses the engine's own attendance, which
    # would rank each clone alone; one simulation computes it beside the inputs.
    requested = inputs if batched else (PRIVATE_SCHOOL_ATTENDANCE_OUTPUT, *inputs)
    materialized = engine.materialize(
        frame, list(requested), year, dataset_period=period
    )
    missing = [name for name in requested if name not in materialized]
    if missing:
        raise ValueError(f"Private-school attendance engine output missing: {missing}.")
    restated = restate_private_school_attendance(frame, materialized, parameters)
    if batched:
        attends = restated
        rank_route = PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE
    else:
        values = np.asarray(materialized[PRIVATE_SCHOOL_ATTENDANCE_OUTPUT])
        if values.shape != (len(person),):
            raise ValueError(
                f"{PRIVATE_SCHOOL_ATTENDANCE_OUTPUT} has {values.shape[0]} values "
                f"for {len(person)} people."
            )
        attends = values.astype(bool)
        if not np.array_equal(attends, restated):
            raise ValueError(
                "policyengine-uk's attends_private_school, restated from the "
                "engine's own values and parameters, differs from the engine's "
                "flags over the same frame. The restatement no longer follows "
                "the engine's formula, so a spine whose engine calls run clone by "
                "clone would store other flags than one simulation over its pool."
            )
        rank_route = PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE
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
        rank_route=rank_route,
        engine_parity_checked=not batched,
    )


def restate_private_school_attendance(
    frame: Frame, values: Mapping[str, Any], parameters: Any
) -> np.ndarray:
    """policyengine-uk's ``attends_private_school`` over every person in ``frame``.

    ``values`` holds the engine's own arrays at one period for
    :data:`PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS` and
    :data:`PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS`: people in the frame's
    order and households by ascending id, the frame's own order, as the
    adapter returns them whether its call ran in one simulation or clone by
    clone. ``parameters`` is the engine's parameter tree at that period, read
    at :data:`PRIVATE_SCHOOL_ATTENDANCE_PARAMETERS` only. The steps, their
    order and their types are the formula's (policyengine-uk 2.174.1,
    variables/contrib/labour/attends_private_school.py), not its code:

    1. each household's members' income tax plus National Insurance, summed
       over the members in the people's order (policyengine-core's group sum,
       a ``bincount``) and given to each member;
    2. net income: the household's market income plus its benefits, at each
       member, less those taxes;
    3. each person carries the household's weight, so a household counts once
       per member; while those weights sum below one million, nobody attends;
    4. each person's percentile of net income among the people of positive
       weight, by microdf's weighted percentile rank, the rank the formula
       calls; people of no weight keep percentile 0;
    5. the rate at that percentile, interpolated between the engine's rates at
       every fifth percentile, times the engine's population factor, for
       under-18s only;
    6. a person attends when the stored draw is below that rate.

    The adapter builds every simulation from data, so the formula's branch
    for a situation, which imputes nobody, never applies.
    """

    person = frame.table("person")
    household_ids = frame.table("household")["household_id"].to_numpy()
    member_of = pd.Index(household_ids).get_indexer(
        person["person_household_id"].to_numpy()
    )
    if (member_of < 0).any():
        raise ValueError("A person's household is not in the frame's household table.")
    arrays: dict[str, np.ndarray] = {}
    for names, rows, entity in (
        (PRIVATE_SCHOOL_ATTENDANCE_PERSON_INPUTS, len(person), "person"),
        (PRIVATE_SCHOOL_ATTENDANCE_HOUSEHOLD_INPUTS, len(household_ids), "household"),
    ):
        for name in names:
            if name not in values:
                raise ValueError(f"The restatement needs the engine's {name!r}.")
            array = np.asarray(values[name])
            if array.shape != (rows,):
                raise ValueError(
                    f"The engine's {name!r} is not one value per {entity} of the frame."
                )
            arrays[name] = array
    taxes = np.bincount(
        member_of,
        weights=arrays["income_tax"] + arrays["national_insurance"],
        minlength=len(household_ids),
    )[member_of]
    net_income = (
        arrays["household_market_income"][member_of]
        + arrays["household_benefits"][member_of]
        - taxes
    )
    weights = arrays["household_weight"][member_of]
    if weights.sum() < _IMPUTED_POPULATION_FLOOR:
        return np.zeros(len(person), dtype=bool)
    percentile = _net_income_percentiles(net_income, weights)
    rate_path, factor_path = PRIVATE_SCHOOL_ATTENDANCE_PARAMETERS
    rate = _parameter(parameters, rate_path)
    rates = np.array([float(rate[str(p)]) for p in _RATE_PERCENTILES])
    rate_by_percentile = np.interp(
        np.arange(_LOOKUP_PERCENTILES), list(_RATE_PERCENTILES), rates
    )
    probability = (
        rate_by_percentile[percentile]
        * _parameter(parameters, factor_path)
        * arrays["age_under_18"]
    )
    return np.asarray(arrays[PRIVATE_SCHOOL_ATTENDANCE_DRAW] < probability, dtype=bool)


def _parameter(parameters: Any, path: str) -> Any:
    """The node or value at a dotted ``path`` in a parameter tree."""

    node = parameters
    for name in path.split("."):
        node = getattr(node, name)
    return node


def _net_income_percentiles(net_income: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each person's percentile of net income among the people of positive
    weight, in the frame's order: microdf's weighted percentile rank, the
    rank policyengine-uk's formula calls, clipped to 0-100. People of no
    weight keep 0, as the formula leaves them. microdf comes with
    policyengine-uk, so it is imported only here."""

    try:
        from microdf import MicroSeries
    except ImportError as exc:
        raise ImportError(
            "Restating private-school attendance ranks with microdf, which "
            "policyengine-uk installs: install microcosm-build's 'uk' extra."
        ) from exc
    income = MicroSeries(net_income, weights=weights)
    percentile = np.zeros(len(net_income), dtype=np.int64)
    weighted = weights > 0
    percentile[weighted] = np.asarray(
        income[weighted].percentile_rank().clip(0, 100)
    ).astype(np.int64)
    return percentile


def _engine_parameters(engine: object, year: int | str) -> Any:
    """The engine's own parameter tree at ``year``, which the restatement
    reads its rates and factor from."""

    parameters = getattr(engine, "parameters", None)
    if not callable(parameters):
        raise TypeError(
            "Private-school attendance restates policyengine-uk's formula from "
            "the engine's own parameters, so its engine must give them "
            "(parameters(period))."
        )
    return parameters(year)


def _simulation_batches(engine: object, frame: Frame) -> int:
    """How many simulations the engine computes ``frame`` in: one, or one per
    clone (the UK adapter's ``simulation_batches``). An engine without the
    method computes it in one."""

    batches = getattr(engine, "simulation_batches", None)
    if batches is None:
        return 1
    count = batches(frame)
    if isinstance(count, bool) or not isinstance(count, int | np.integer) or count < 1:
        raise ValueError(
            "The engine's simulation_batches must be a positive whole number."
        )
    return int(count)


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
