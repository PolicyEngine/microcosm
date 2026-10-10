"""Employer premiums on the ACS spine of the retained ACS local lane (#454).

The lane's staging frame pools a dense ASEC-by-PUF donor release with the ACS
spine (:func:`~.base_pool.with_optional_acs_spine`). The donor carries
``employer_sponsored_insurance_premiums`` from the ``meps_esi_premiums`` stage.
ACS has no policyholder flag, payment status, coverage tier or firm size, so
the stage cannot run on its rows, and the declared ACS transfer plan does not
name the column. Three steps put it on every row, on the stage's scale:

1. **Qualify the donor** (:func:`prepare_acs_local_esi_premium_donor`). The
   donor must carry the column and the raw ASEC fields the stage read, and
   pass the stage's own signal gate. The transfer then fits on its ASEC
   observation role only (support clone 0), whose wages are the measured CPS
   wages. The lane's other transfers fit on the PUF-detail role, whose wages
   are PUF-imputed while its premium is a copy of the source person's: on
   those rows the premium no longer follows the wage predictor.
2. **Transfer** the column onto the ACS rows with the lane's own one-family
   plan (:func:`acs_local_esi_premium_transfer_target_families`). The declared
   ACS plan and the stacked pool's plan are untouched.
3. **Anchor** the pooled frame (:func:`with_acs_local_esi_premium_anchor`)
   with the stacked pool's kernel,
   :func:`~.esi_premiums.with_us_esi_premium_pool_anchor`: it clears the
   premium where the ACS person reports zero wages and scales the rest so both
   spines carry the same weighted premium per unit of household mass. The
   donor rows are never rewritten.

The kernel tells the two kinds of row apart by the CPS record id
(``PERIDNUM``) and the raw ASEC fields, which the donor rows carry and the ACS
rows do not. It refuses a transferred row whose wage is missing. ACS PUMS
leaves ``WAGP`` blank below age 15, outside its earnings universe, and the
lane keeps that blank in the staging frame. The anchor therefore reads those
people's wages as zero under the named universe rule of
:mod:`.acs_income_universe`, in its view of the frame only. A blank wage at
age 15 or older stays missing and is refused.

Invariant: the lane-wide weighted column equals the donor rows' weighted
column divided by their share of household mass. Pooling multiplies every
donor household weight by that share, so the lane-wide column is the donor
release's own.

The kernel and both gates read a handful of columns, but the kernel copies
every table it is given and each gate copies the person table. This module
hands them a narrow view of the pooled frame: the staging frame holds two full
spines, and the lane's memory budget has no room for further copies of them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.acs_income_universe import (
    ACS_PUMS_EARNINGS_MINIMUM_AGE,
    ACS_PUMS_EARNINGS_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.acs_transfer import (
    TargetFamilies,
    resolve_acs_donor_channel,
)
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    US_ESI_PREMIUMS_POOL_ANCHOR_PERSON_INPUTS,
    US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS,
    US_ESI_PREMIUMS_WAGE_COLUMN,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    with_us_esi_premium_pool_anchor,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_ESI_PREMIUM_FAMILY",
    "acs_local_esi_premium_gate_payload",
    "acs_local_esi_premium_gates",
    "acs_local_esi_premium_transfer_target_families",
    "prepare_acs_local_esi_premium_donor",
    "with_acs_local_esi_premium_anchor",
]

#: The stacked pool files the same column under the same family name
#: (``multispine_pool.pool_transfer_target_families``).
ACS_LOCAL_ESI_PREMIUM_FAMILY = "source_operator_esi_premiums"

_PERSON = US_SCHEMA.person_entity
_STATE_COLUMN = "state_fips"
#: Raw ASEC person fields whose presence marks a row the stage derived itself.
_RAW_EVIDENCE_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS
    if column != _STATE_COLUMN
)
#: The CPS record id. The kernel requires it beside the raw fields on every
#: row the stage derived, and blank on every transferred row.
_CPS_RECORD_ID_COLUMN = "PERIDNUM"
_CLONE_INDEX_COLUMN = support_clone_index_column(_PERSON)
_SOURCE_ID_COLUMN = support_source_id_column(_PERSON)
_AGE_COLUMN = "age"
#: The raw PUMS field the ACS wage is mapped from, and the named rule that
#: makes its blank a zero: the field applies to people aged 15 and older.
_ACS_WAGE_SOURCE_COLUMN = ACS_PUMS_EARNINGS_SOURCE_COLUMNS[US_ESI_PREMIUMS_WAGE_COLUMN]
_ACS_WAGE_UNIVERSE_RULE = (
    f"acs_2024_pums_{_ACS_WAGE_SOURCE_COLUMN.lower()}"
    f"_age_{ACS_PUMS_EARNINGS_MINIMUM_AGE}_plus"
)
#: Person columns the pool anchor kernel reads, without the two support
#: provenance columns. The kernel looks a clone's wages up by
#: ``person_source_id`` across the whole frame. A stacked pool makes that id
#: unique at assembly; this lane does not: ACS rows keep their pre-remap ids
#: as source ids, which can repeat a donor's. Every ACS row here is its own
#: source record, so the view omits the lookup's columns and the kernel reads
#: each row's own wages.
_ANCHOR_PERSON_COLUMNS: tuple[str, ...] = (
    *US_ESI_PREMIUMS_OUTPUT_COLUMNS,
    *(
        column
        for column in US_ESI_PREMIUMS_POOL_ANCHOR_PERSON_INPUTS
        if column not in (_CLONE_INDEX_COLUMN, _SOURCE_ID_COLUMN)
    ),
)
#: Person columns the two ESI gates read: the output, the raw fields and
#: State the cells are recomputed from, the record id that settles each
#: row's kind, wages and support provenance for the transferred-row summary,
#: and the stable source identity the donor's clones are compared by.
_GATE_PERSON_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *US_ESI_PREMIUMS_OUTPUT_COLUMNS,
            *US_ESI_PREMIUMS_REQUIRED_SOURCE_COLUMNS,
            *US_ESI_PREMIUMS_POOL_ANCHOR_PERSON_INPUTS,
            "source_year",
            "source_household_id",
            "source_person_id",
        )
    )
)
_GATE_HOUSEHOLD_COLUMNS: tuple[str, ...] = (_STATE_COLUMN,)


def acs_local_esi_premium_transfer_target_families() -> TargetFamilies:
    """The lane's separate ASEC-only pass for the employer premium."""

    return {_PERSON: {ACS_LOCAL_ESI_PREMIUM_FAMILY: US_ESI_PREMIUMS_OUTPUT_COLUMNS}}


def _narrow_view(
    frame: Frame,
    *,
    person_columns: tuple[str, ...],
    household_columns: tuple[str, ...] = (),
    person_values: Mapping[str, np.ndarray] | None = None,
) -> Frame:
    """The frame's structure and weights with only the named value columns.

    ``person_values`` replaces a person column in the view; the frame's own
    column is not touched.
    """

    schema = frame.schema
    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        if entity == schema.person_entity:
            keep = [
                schema.entity_id_column(entity),
                *(schema.membership_column(group) for group in schema.group_entities),
                *person_columns,
            ]
        else:
            keep = [schema.entity_id_column(entity)]
            if entity == "household":
                keep.extend(household_columns)
        present = list(dict.fromkeys(column for column in keep if column in table))
        tables[entity] = table.loc[:, present]
    if person_values:
        person = tables[schema.person_entity].copy(deep=False)
        for column, values in person_values.items():
            person[column] = values
        tables[schema.person_entity] = person
    return Frame(
        tables,
        schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def prepare_acs_local_esi_premium_donor(
    frame: Frame,
) -> tuple[Frame, dict[str, object]]:
    """Qualify the dense donor and select the rows the lane's transfer fits on.

    Refuses a donor that lacks the stage's output or the raw ASEC fields it
    read, that fails the stage's signal gate, or whose support roles cannot be
    told apart. Returns the ASEC observation role (support clone 0) and a
    receipt. The donor frame itself is not changed.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("Local ESI premium donor requires the US schema.")
    person = frame.table(_PERSON)
    missing = [
        column
        for column in (*US_ESI_PREMIUMS_OUTPUT_COLUMNS, *_RAW_EVIDENCE_COLUMNS)
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"Local ESI premium donor lacks {missing}: the donor release was "
            "built without the meps_esi_premiums stage (microcosm#454). The "
            "lane transfers the employer premium from a donor that carries it "
            "and never defaults it."
        )
    if _CPS_RECORD_ID_COLUMN not in person:
        raise ValueError(
            f"Local ESI premium donor lacks {_CPS_RECORD_ID_COLUMN!r}, the CPS "
            "record id that marks its rows as the ones the stage derived. "
            "Without it the pooled frame cannot tell them from the ACS rows."
        )
    if US_ESI_PREMIUMS_WAGE_COLUMN not in person:
        raise ValueError(
            f"Local ESI premium donor lacks {US_ESI_PREMIUMS_WAGE_COLUMN!r}, the "
            "wage the transfer conditions the premium on."
        )
    if not has_support_role_metadata(
        person, entity=_PERSON
    ) or has_assembled_support_metadata(person, entity=_PERSON):
        raise ValueError(
            "Local ESI premium donor requires explicit legacy ASEC observation "
            "roles; unclassified or multispine records need their own source "
            "lineage."
        )
    gate = us_esi_premiums_signal_gate(
        _narrow_view(
            frame,
            person_columns=_GATE_PERSON_COLUMNS,
            household_columns=_GATE_HOUSEHOLD_COLUMNS,
        )
    )
    if not gate.passed:
        raise ValueError(
            "Local ESI premium donor fails the meps_esi_premiums signal gate: "
            + "; ".join(gate.failures)
        )
    donor, role = resolve_acs_donor_channel(frame, BASE_ASEC_SUPPORT_CHANNEL)
    donor_person = donor.table(_PERSON)
    employer = donor_person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
    wages = pd.to_numeric(
        donor_person[US_ESI_PREMIUMS_WAGE_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(wages).all():
        raise ValueError(
            "Local ESI premium donor has ASEC observations with no "
            f"{US_ESI_PREMIUMS_WAGE_COLUMN!r}; the transfer never treats a "
            "missing wage as zero."
        )
    weights = np.asarray(frame.resolve_weights(_PERSON).values, dtype=np.float64)
    donor_weights = np.asarray(donor.resolve_weights(_PERSON).values, dtype=np.float64)
    positive = employer > 0
    return donor, {
        "producer": "meps_esi_premiums",
        "donor_channel": role,
        "donor_wage_column": US_ESI_PREMIUMS_WAGE_COLUMN,
        "donor_wage_basis": "measured CPS ASEC wages of the observation role",
        "donor_rows": int(frame.n(_PERSON)),
        "asec_observation_rows": int(donor.n(_PERSON)),
        "donor_employer_premium_total": float(
            weights @ person[US_ESI_EMPLOYER_PREMIUM_COLUMN].to_numpy(dtype=np.float64)
        ),
        "asec_observation_positive_share": (
            float(donor_weights[positive].sum() / donor_weights.sum())
            if donor_weights.sum()
            else 0.0
        ),
        # The structural zero the transfer has to learn: how much of the
        # observation role's premium mass sits on people with no measured
        # wages (employed at interview, no wages in the income year).
        "asec_observation_premium_share_without_wages": (
            float(
                donor_weights[positive & ~(wages > 0)]
                @ employer[positive & ~(wages > 0)]
            )
            / float(donor_weights @ employer)
            if float(donor_weights @ employer)
            else 0.0
        ),
        "donor_signal_gate": {"passed": True, "failures": []},
    }


def _wages_under_the_acs_universe(
    person: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Wages for the anchor's view, with ACS universe blanks read as zero.

    ACS PUMS leaves ``WAGP`` blank for people below age 15: the field does
    not apply to them. The lane keeps that blank in the staging frame, and the
    anchor refuses a missing wage. Only a transferred row below the age floor
    whose mapped wage is null, and whose raw field is blank where the frame
    carries it, takes a zero here. Any other missing wage stays missing.
    """

    wages = pd.to_numeric(
        person[US_ESI_PREMIUMS_WAGE_COLUMN], errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    transferred = person[list(_RAW_EVIDENCE_COLUMNS)].isna().all(axis=1).to_numpy()
    blank = transferred & np.isnan(wages)
    outside = np.zeros(len(person), dtype=bool)
    if blank.any() and _AGE_COLUMN in person:
        age = pd.to_numeric(person[_AGE_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        outside = blank & (age < ACS_PUMS_EARNINGS_MINIMUM_AGE)
        if _ACS_WAGE_SOURCE_COLUMN in person:
            outside &= person[_ACS_WAGE_SOURCE_COLUMN].isna().to_numpy()
    wages = np.where(outside, 0.0, wages)
    return wages, {
        "rule_id": _ACS_WAGE_UNIVERSE_RULE,
        "minimum_age": ACS_PUMS_EARNINGS_MINIMUM_AGE,
        "raw_source_column": _ACS_WAGE_SOURCE_COLUMN,
        "transferred_rows_with_blank_wage": int(blank.sum()),
        "transferred_rows_outside_universe_read_as_zero": int(outside.sum()),
        # The staging frame keeps the source's blanks; the zeros exist in the
        # anchor's view only.
        "frame_cells_written": 0,
    }


def with_acs_local_esi_premium_anchor(frame: Frame) -> tuple[Frame, dict[str, object]]:
    """Hold the pooled lane frame's transferred premiums to the donor's scale.

    Runs :func:`~.esi_premiums.with_us_esi_premium_pool_anchor` on a narrow
    view of ``frame`` and writes the anchored column back. Rows with the raw
    ASEC fields (the donor spine) are never rewritten. On the other rows the
    premium is cleared where the person reports zero wages and the rest is
    scaled by one factor, so both spines carry the same weighted premium per
    unit of household mass. A blank ACS wage below age 15 is a zero wage
    (:func:`_wages_under_the_acs_universe`); any other missing wage is
    refused. A frame already on that scale, or with no transferred row, is
    returned unchanged. Returns the frame and the kernel's receipt.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("Local ESI premium anchor requires the US schema.")
    person = frame.table(_PERSON)
    absent = [column for column in _ANCHOR_PERSON_COLUMNS if column not in person]
    if absent:
        raise SourceRuntimeError(
            f"Local ESI premium anchor: the person table lacks {absent}; it "
            "needs the transferred premium on every row, the raw ASEC fields "
            "that mark the donor rows and the wages that place the structural "
            "zeros."
        )
    if _CLONE_INDEX_COLUMN in person:
        transferred = person[list(_RAW_EVIDENCE_COLUMNS)].isna().all(axis=1)
        clone_index = pd.to_numeric(person[_CLONE_INDEX_COLUMN], errors="coerce")
        clones = transferred & ~clone_index.eq(0)
        if clones.any():
            raise SourceRuntimeError(
                f"Local ESI premium anchor: {int(clones.sum())} row(s) without "
                "raw ASEC fields are not clone-0 source records. The lane reads "
                "each transferred person's own wages; support clones of "
                "transferred rows need the stacked pool's source-record lookup."
            )
    wages, wage_universe = _wages_under_the_acs_universe(person)
    view = _narrow_view(
        frame,
        person_columns=_ANCHOR_PERSON_COLUMNS,
        person_values={US_ESI_PREMIUMS_WAGE_COLUMN: wages},
    )
    anchored, receipt = with_us_esi_premium_pool_anchor(view)
    receipt = dict(receipt) | {
        "kernel": "with_us_esi_premium_pool_anchor",
        "transferred_source_record": "each transferred row is its own source record",
        "wage_universe": wage_universe,
    }
    if anchored is view:
        return frame, receipt
    id_column = US_SCHEMA.entity_id_column(_PERSON)
    anchored_person = anchored.table(_PERSON)
    if not np.array_equal(
        anchored_person[id_column].to_numpy(), person[id_column].to_numpy()
    ):
        raise SourceRuntimeError(
            "Local ESI premium anchor: the kernel reordered the person table."
        )
    tables = {
        entity: (person.copy(deep=False) if entity == _PERSON else frame.table(entity))
        for entity in frame.entities
    }
    tables.update({name: frame.link(name) for name in frame.links})
    for column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
        tables[_PERSON][column] = anchored_person[column].to_numpy(dtype=np.float64)
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return result, receipt


def acs_local_esi_premium_gates(
    frame: Frame, *, time_period: int
) -> tuple[GateResult, GateResult]:
    """Both ESI premium gates on a lane frame: ``(signal, anchor)``.

    The gates grade a pooled frame by row kind: the donor rows, which carry
    the raw ASEC fields, get the full MEPS-IC recomputation, and the ACS rows
    are compared with them. They run on a narrow view; the verdicts are those
    of the full frame.
    """

    view = _narrow_view(
        frame,
        person_columns=_GATE_PERSON_COLUMNS,
        household_columns=_GATE_HOUSEHOLD_COLUMNS,
    )
    return (
        us_esi_premiums_signal_gate(view),
        us_esi_premiums_anchor_gate(view, time_period=time_period),
    )


def _strict_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def acs_local_esi_premium_gate_payload(gate: GateResult) -> dict[str, object]:
    """One gate verdict for a lane summary or gate report.

    Failures are kept verbatim, so a diagnostic build that waived them still
    records what was red. Non-finite detail values become ``null``: the
    staging summary is written as strict JSON.
    """

    return {
        "name": gate.name,
        "passed": bool(gate.passed),
        "failures": list(gate.failures),
        "details": _strict_json(dict(gate.details)),
    }
