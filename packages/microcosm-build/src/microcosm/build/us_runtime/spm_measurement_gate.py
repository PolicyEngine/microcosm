"""Release gate: every SPM unit the runtime measures has a classified adult.

spm-calculator 1.0.0 counts an SPM unit's adults as
``age >= 18 OR (age >= 15 AND is_spm_independent_minor_role)``
(:data:`~microcosm.build.us_runtime.spm_role_source.SPM_ADULT_RULE`) and
refuses the whole simulation's SPM measurement
(``SPM_COMPOSITION_REQUIRED``) when a single unit it measures has none.
Which units it measures depends on the engine:

- An engine that does not scope SPM by measurement universe (policyengine-us
  2.2.1 and every release before policyengine-us#9462) measures every unit.
- An engine that defines
  :data:`~microcosm.build.us_runtime.spm_universe_source.UNIVERSE_INPUT`
  measures only the units the dataset declares ``INCLUDED``, and refuses a
  dataset that declares nothing (``SPM_UNIVERSE_REQUIRED``).

:func:`check_release_spm_measurement` grades a release's final frame against
the engine it is certified with. It fails when any unit that engine measures
has no classified adult, so a release can never be certified while one of its
SPM measurements would crash. It also fails when the role is not a complete
stored Boolean (the engine's head/spouse fallback is not a release remedy, and
``policyengine_us.spm.DATASET_SOURCE_INPUTS`` forbids defaulting it), when the
source universe cannot be declared, and when a stored declaration differs from
the source's. The measured set always contains every unit the source places
inside the SPM universe, so a housing-unit SPM unit with no classified adult
fails under every engine.

The universe split comes from the source producer
(:func:`~microcosm.build.us_runtime.spm_universe_source.classify_spm_universe`).
Under an engine that does not scope, it is reported only; it never excuses a
unit the engine measures. The gate records aggregate counts only, never a
unit id or a member's age.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.frame import Frame

from .spm_composition import _spm_composition
from .spm_role_source import NATIVE_SPM_ROLE, SPM_ADULT_RULE
from .spm_universe_source import (
    INCLUDED,
    OUTSIDE,
    SPM_UNIVERSE_PROFILE_NAME,
    SPM_UNIVERSE_RULE,
    UNIVERSE_INPUT,
    classify_spm_universe,
)

__all__ = [
    "SPM_MEASUREMENT_GATE_NAME",
    "check_release_spm_measurement",
    "engine_scopes_spm_universe",
]

SPM_MEASUREMENT_GATE_NAME = "spm_measurement_composition"

#: What a measured scope is called in the gate's details.
_ALL_UNITS = "all_units"
_INCLUDED_UNITS = "included_units"


def engine_scopes_spm_universe(engine_variables: Collection[str]) -> bool:
    """Does an engine with these variables measure only ``INCLUDED`` units?

    policyengine-us#9462 introduces :data:`UNIVERSE_INPUT` together with the
    scoped threshold formulas that pass only ``INCLUDED`` units to the
    calculator. An engine that does not define the variable cannot read a
    declaration, so it measures every unit.
    """

    return UNIVERSE_INPUT in engine_variables


def _role_failures(person: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    """Why the stored role is not a complete Boolean source column."""

    if NATIVE_SPM_ROLE not in person.columns:
        return (
            [
                f"The release stores no {NATIVE_SPM_ROLE!r} column. It is the "
                "engine's one declared dataset source input; without it the "
                "engine falls back to household head/spouse structure, which "
                "is not a release remedy. Rebuild the staging, which derives "
                "the role for every spine."
            ],
            {"role_column_present": False},
        )
    column = person[NATIVE_SPM_ROLE]
    nulls = int(column.isna().sum())
    values = column.dropna()
    if pd.api.types.is_bool_dtype(values.dtype):
        non_boolean = 0
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        non_boolean = int((~numeric.isin([0, 1])).sum())
    failures = []
    if nulls:
        failures.append(
            f"{nulls} person row(s) store a null {NATIVE_SPM_ROLE!r}; a declared "
            "dataset source input may not be defaulted, so the release must "
            "store it for every person."
        )
    if non_boolean:
        failures.append(
            f"{non_boolean} person row(s) store a {NATIVE_SPM_ROLE!r} value that "
            "is not Boolean."
        )
    return failures, {
        "role_column_present": True,
        "role_null_values": nulls,
        "role_non_boolean_values": non_boolean,
        "role_true_persons": int(pd.to_numeric(values, errors="coerce").eq(1).sum()),
    }


def _oldest_member_bands(
    age: np.ndarray, membership: np.ndarray, unit_ids: np.ndarray, units: np.ndarray
) -> dict[str, int]:
    """How many of ``units`` have their oldest member in each rule band."""

    oldest = (
        pd.Series(age, index=membership)
        .groupby(level=0)
        .max()
        .reindex(unit_ids)
        .to_numpy(dtype=np.float64)
    )[units]
    with np.errstate(invalid="ignore"):
        return {
            "under_15": int((oldest < 15).sum()),
            "15_to_17": int(((oldest >= 15) & (oldest < 18)).sum()),
            "unknown": int(np.isnan(oldest).sum()),
        }


def check_release_spm_measurement(
    frame: Frame,
    *,
    engine_scopes_universe: bool,
    unit_entity: str = "spm_unit",
) -> GateResult:
    """Grade a release frame's SPM measurement composition against its engine.

    Args:
        frame: The release's final frame: the calibrated artifact as the
            package stage loads it.
        engine_scopes_universe: Whether the certified engine measures only
            ``INCLUDED`` units (:func:`engine_scopes_spm_universe`).
        unit_entity: The SPM unit entity.

    Returns:
        The gate verdict. It passes only when the role is a complete stored
        Boolean, the source universe is declarable, any stored declaration
        equals the source's, and every unit the engine measures has a
        classified adult.
    """

    schema = frame.schema
    person = frame.table(schema.person_entity)
    failures, role_details = _role_failures(person)
    details: dict[str, Any] = {
        "rule": SPM_ADULT_RULE,
        "role_column": NATIVE_SPM_ROLE,
        **role_details,
        "engine_scopes_spm_universe": bool(engine_scopes_universe),
        "universe_profile": SPM_UNIVERSE_PROFILE_NAME,
        "universe_rule": SPM_UNIVERSE_RULE,
        "n_persons": int(len(person)),
    }
    if failures:
        return GateResult(
            name=SPM_MEASUREMENT_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    try:
        composition = _spm_composition(frame, unit_entity=unit_entity)
    except ValueError as error:
        return GateResult(
            name=SPM_MEASUREMENT_GATE_NAME,
            passed=False,
            failures=(f"The SPM composition cannot be evaluated: {error}",),
            details=details,
        )
    unit_ids = composition.unit_ids
    unclassified = composition.offending

    try:
        universe = classify_spm_universe(
            household=frame.table("household"),
            person=person,
            unit_ids=unit_ids,
            household_id_column=schema.id_column("household"),
            household_membership_column=schema.membership_column("household"),
            unit_membership_column=schema.membership_column(unit_entity),
        )
    except ValueError as error:
        return GateResult(
            name=SPM_MEASUREMENT_GATE_NAME,
            passed=False,
            failures=(
                "The source SPM measurement universe cannot be declared for "
                f"this release: {error}",
            ),
            details={**details, "n_units": int(len(unit_ids))},
        )
    status = np.asarray(universe.status, dtype=object)
    included = status == INCLUDED

    unit_table = frame.table(unit_entity)
    declared = UNIVERSE_INPUT in unit_table.columns
    if declared:
        stored = unit_table.set_index(schema.id_column(unit_entity))[UNIVERSE_INPUT]
        # H5 strings can read back as bytes; the declaration is the member name.
        stored = np.asarray(
            [
                value.decode() if isinstance(value, bytes) else value
                for value in stored.reindex(unit_ids).to_numpy()
            ],
            dtype=object,
        )
        drift = int((stored != status).sum())
        if drift:
            failures.append(
                f"{drift} SPM unit(s) store a {UNIVERSE_INPUT!r} that differs "
                "from the source declaration (ACS TYPEHUGQ and the ASEC frame "
                "ruling); the release must store the source's declaration."
            )
    if engine_scopes_universe:
        if not declared:
            failures.append(
                f"The engine scopes SPM measurement by {UNIVERSE_INPUT!r}, but "
                "the release declares no universe, so every SPM output would "
                "raise SPM_UNIVERSE_REQUIRED."
            )
        measured = included
        scope = _INCLUDED_UNITS
    else:
        measured = np.ones(len(unit_ids), dtype=bool)
        scope = _ALL_UNITS

    measured_unclassified = unclassified & measured
    n_measured_unclassified = int(measured_unclassified.sum())
    by_universe = {
        INCLUDED: int((unclassified & included).sum()),
        OUTSIDE: int((unclassified & ~included).sum()),
    }
    details.update(
        {
            "universe_declared_in_release": declared,
            "measured_scope": scope,
            "n_units": int(len(unit_ids)),
            "n_units_by_universe": {
                INCLUDED: int(included.sum()),
                OUTSIDE: int((~included).sum()),
            },
            "n_measured_units": int(measured.sum()),
            "n_measured_units_without_classified_adult": n_measured_unclassified,
            "n_units_without_classified_adult": int(unclassified.sum()),
            "n_units_without_classified_adult_by_universe": by_universe,
            "n_units_without_member_aged_18_or_over": (
                composition.n_units_without_member_18_plus
            ),
            "unclassified_units_by_oldest_member": _oldest_member_bands(
                composition.age, composition.membership, unit_ids, unclassified
            ),
        }
    )
    if n_measured_unclassified:
        failures.append(
            f"{n_measured_unclassified} of {int(measured.sum())} SPM units the "
            f"certified engine measures ({scope}) have no classified adult "
            f"under {SPM_ADULT_RULE}: {by_universe[INCLUDED]} inside the source "
            f"SPM universe and {by_universe[OUTSIDE]} outside it. Every SPM "
            "measurement on this release would raise SPM_COMPOSITION_REQUIRED."
        )
    return GateResult(
        name=SPM_MEASUREMENT_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
