"""ESI employer premium frames: Route A's shape and a mixed frame (microcosm #454).

Route A writes the premium on CPS ASEC rows in its base builder (the stage,
the signal gate, the PUF support clone, the signal gate again), and its
release grades the base and the calibrated export. :func:`asec_arm` is that
CPS side, with or without the CPS record id, and :func:`staged`,
:func:`puf_cloned` and :func:`reweighted` take it through those steps.

:func:`mixed_frame` adds rows a cross-source transfer filled, which carry none
of the raw ASEC fields and no CPS record id: the frame
``allow_transferred_rows=True`` grades (d2283 PR 0).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import esi_premiums as esi
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_CPS_RECORD_COLUMN,
    US_ESI_PREMIUMS_WAGE_COLUMN,
    with_us_esi_premium_inputs,
    with_us_esi_premium_pool_anchor,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_stacked_spine import _source_frame

EMPLOYER = US_ESI_EMPLOYER_PREMIUM_COLUMN
WAGES = US_ESI_PREMIUMS_WAGE_COLUMN
CPS_ID = US_ESI_PREMIUMS_CPS_RECORD_COLUMN
RAW = esi._RAW_EVIDENCE_COLUMNS
RAW_WAGE = esi._RAW_ACS_WAGE_COLUMN
YEAR = 2024
#: BEA NIPA 7.8 line 17 for :data:`YEAR`: the stage's anchor.
ANCHOR = float(esi.EMPLOYER_PREMIUM_ANCHOR["values"][str(YEAR)])
STATES = (6, 36, 48, 12, 17, 39, 53, 13)
_CHANNEL = "person_support_channel"
_CLONE_INDEX = "person_support_clone_index"
_SOURCE_ID = "person_source_id"


def rebuilt(
    frame: Frame,
    *,
    person: pd.DataFrame | None = None,
    household: pd.DataFrame | None = None,
    household_weights: Weights | None = None,
) -> Frame:
    """``frame`` with some tables or the household weights replaced."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    if person is not None:
        tables["person"] = person
    if household is not None:
        tables["household"] = household
    tables.update({name: frame.link(name) for name in frame.links})
    return Frame(
        tables,
        frame.schema,
        {
            "household": household_weights
            if household_weights is not None
            else frame.weights_for("household")
        },
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def asec_arm(
    households: int = 320,
    *,
    first_household_id: int = 1,
    record_ids: bool = True,
) -> Frame:
    """One-person CPS ASEC households: a mix inside the signal-gate bands.

    Of every 20 people, 7 are employed policyholders, 2 are policyholders
    without a job (retirees), 5 are employed without a policy and 6 are
    neither. ``record_ids=False`` leaves out the CPS record id column.
    """

    frame = _source_frame(
        household_ids=list(range(first_household_id, first_household_id + households)),
        weights=[900.0 + 37.0 * (index % 11) for index in range(households)],
        stratum="asec_2024",
    )
    index = np.arange(households)
    kind = index % 20
    holder = kind < 9
    employed = (kind < 7) | ((kind >= 9) & (kind < 14))
    tier = np.where(holder, np.asarray([1, 2, 3])[index % 3], 0)
    wages = np.where(employed, 28_000.0 + 850.0 * (index % 97), 0.0)
    person = frame.table("person").copy()
    if record_ids:
        person[CPS_ID] = [f"asec-{position:05d}" for position in index]
    person["source_year"] = YEAR
    person["source_household_id"] = person["person_household_id"].to_numpy()
    person["source_person_id"] = 1
    person["age"] = np.where(employed, 24.0 + index % 40, 66.0 + index % 20)
    person["is_female"] = index % 2 == 0
    person[WAGES] = wages
    person["NOW_OWNGRP"] = np.where(holder, 1, 2)
    person["NOW_HIPAID"] = np.where(holder, np.asarray([1, 2, 2, 3])[index % 4], 0)
    person["NOW_GRPFTYP2"] = tier
    person["NOW_GRPFTYP"] = np.asarray([0, 1, 1, 2])[tier]
    person["PEMLR"] = np.where(employed, 1, 5)
    person["NOEMP"] = np.where(employed, np.asarray([1, 3, 6, 0])[index % 4], 0)
    person["PEIO1COW"] = np.where(
        employed, np.asarray([4, 4, 5, 2, 3, 1])[index % 6], 0
    )
    household = frame.table("household").copy()
    household["state_fips"] = np.asarray(STATES, dtype=np.int64)[index % len(STATES)]
    return rebuilt(frame, person=person, household=household)


def staged(frame: Frame) -> Frame:
    """The base builder's stage call (``tools/build_us_puf_support_base.py``)."""

    return with_us_esi_premium_inputs(frame, seed=0, time_period=YEAR)


def puf_cloned(frame: Frame) -> Frame:
    """The production PUF support clone the base builder runs after the stage."""

    return clone_us_frame_for_puf_support(frame)


def calibration_factors(frame: Frame, *, spread: float = 0.03) -> np.ndarray:
    """Deterministic household weight factors within ``1 +/- spread``."""

    count = len(frame.table("household"))
    return 1.0 + spread * np.cos(np.arange(count, dtype=np.float64))


def reweighted(frame: Frame, factors: np.ndarray | float) -> Frame:
    """The same people at household weights times ``factors``: an export."""

    values = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    return rebuilt(
        frame,
        household_weights=Weights(values * factors, WeightKind.CALIBRATED),
    )


def with_person_cells(
    frame: Frame, rows: Sequence[int] | np.ndarray, columns, value: object
) -> Frame:
    """``frame`` with ``value`` written into ``columns`` at person positions ``rows``.

    An integer column takes a value that is not an integer as float64 first,
    so a null lands as NaN.
    """

    person = frame.table("person").copy()
    for column in [columns] if isinstance(columns, str) else list(columns):
        if pd.api.types.is_integer_dtype(person[column]) and not isinstance(
            value, (int, np.integer)
        ):
            person[column] = person[column].astype(np.float64)
    person.loc[person.index[np.asarray(rows, dtype=np.int64)], columns] = value
    return rebuilt(frame, person=person)


def without_person_columns(frame: Frame, columns: Sequence[str]) -> Frame:
    return rebuilt(frame, person=frame.table("person").drop(columns=list(columns)))


def with_scaled_premiums(
    frame: Frame, factor: float, rows: np.ndarray | None = None
) -> Frame:
    """``frame`` with the employer premium times ``factor`` on ``rows`` (all)."""

    person = frame.table("person").copy()
    rows = np.ones(len(person), dtype=bool) if rows is None else np.asarray(rows)
    values = person[EMPLOYER].to_numpy(dtype=np.float64).copy()
    values[rows] *= factor
    person[EMPLOYER] = values
    return rebuilt(frame, person=person)


def cps_rows(frame: Frame) -> np.ndarray:
    """Rows that carry the raw ASEC fields."""

    return frame.table("person")[RAW[0]].notna().to_numpy()


def households_of(frame: Frame, rows: np.ndarray) -> np.ndarray:
    """Household-table flags for the households of person ``rows``."""

    person = frame.table("person")
    household = frame.table("household")
    return (
        household["household_id"]
        .isin(person.loc[np.asarray(rows), "person_household_id"])
        .to_numpy()
    )


def mixed_frame(
    asec_households: int = 320,
    acs_households: int = 280,
    *,
    acs_clones: bool = False,
) -> Frame:
    """CPS rows beside rows a cross-source transfer filled, on the stage's scale.

    The CPS side is :func:`asec_arm` staged on its own weights, then given
    half of them: half the household mass of one population. The transfer
    gives each of ``acs_households`` one-person households the age, sex, wages
    and premium of the CPS person at the same position, with none of the raw
    ASEC fields and no CPS record id, and their household weights carry the
    other half of the mass. ``acs_clones`` gives every transferred person one
    support clone, and splits its household weight between the two.
    :func:`with_us_esi_premium_pool_anchor` then holds the transferred
    premiums to the CPS rows' premium per unit of household mass, so the
    frame's anchor-universe total is the anchor.
    """

    if not 0 < acs_households <= asec_households:
        raise ValueError("mixed_frame copies each transferred row from a CPS row.")
    donor = staged(asec_arm(asec_households))
    donor_person = donor.table("person")
    donor_household = donor.table("household")
    donor_weights = np.asarray(donor.weights_for("household").values, np.float64)
    pattern = 40.0 + 3.0 * (np.arange(acs_households) % 13)
    acs_weights = pattern * (donor_weights.sum() / 2.0 / pattern.sum())
    acs_ids = np.arange(10_001, 10_001 + acs_households)
    household_ids = [*range(1, asec_households + 1), *acs_ids.tolist()]
    weights = [*(donor_weights / 2.0)]
    # Each person row's donor position, and its transferred origin (-1: CPS).
    donor_position = [*range(asec_households), *range(acs_households)]
    origin = [-1] * asec_households + list(range(acs_households))
    if acs_clones:
        household_ids += list(range(20_001, 20_001 + acs_households))
        weights += [*(acs_weights / 2.0), *(acs_weights / 2.0)]
        donor_position += list(range(acs_households))
        origin += list(range(acs_households))
    else:
        weights += list(acs_weights)
    frame = _source_frame(
        household_ids=household_ids, weights=weights, stratum="asec_acs_2024"
    )
    position = np.asarray(donor_position)
    origin_array = np.asarray(origin)
    cps = origin_array < 0
    clone = np.zeros(len(position), dtype=bool)
    clone[asec_households + acs_households :] = True
    person = frame.table("person").copy()
    for column in ("age", "is_female", WAGES, EMPLOYER):
        person[column] = donor_person[column].to_numpy()[position]
    for column in RAW:
        values = donor_person[column].to_numpy(dtype=np.float64)[position]
        person[column] = np.where(cps, values, np.nan)
    ids = donor_person[CPS_ID].to_numpy(dtype=object)[position]
    person[CPS_ID] = np.where(cps, ids, None)
    person[RAW_WAGE] = np.where(cps, np.nan, person[WAGES].to_numpy())
    person["source_year"] = YEAR
    source_household = donor_person["source_household_id"].to_numpy()[position]
    source_household[~cps] = acs_ids[origin_array[~cps]]
    person["source_household_id"] = source_household
    person["source_person_id"] = 1
    person[_CHANNEL] = np.where(cps, "asec", "acs")
    person[_CLONE_INDEX] = clone.astype(np.int64)
    person[_SOURCE_ID] = [
        f"cps-{index}" if from_cps else f"acs-{source}"
        for index, (from_cps, source) in enumerate(zip(cps, origin_array, strict=True))
    ]
    household = frame.table("household").copy()
    donor_states = donor_household["state_fips"].to_numpy(dtype=np.int64)
    acs_states = np.asarray(STATES, dtype=np.int64)[
        (np.arange(acs_households) + 3) % len(STATES)
    ]
    household["state_fips"] = np.concatenate(
        [donor_states, acs_states, *([acs_states] if acs_clones else [])]
    )
    filled = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "person": person,
            "household": household,
        },
        US_SCHEMA,
        {"household": Weights(np.asarray(weights, np.float64), WeightKind.DESIGN)},
        frame.strata,
    )
    anchored, _receipt = with_us_esi_premium_pool_anchor(filled)
    return anchored
