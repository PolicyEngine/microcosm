"""Runtime-owned take-up and default-fill inputs for the ACS rows of the
retained ACS local lane.

The ACS transfer leaves take-up draws to the runtime, and the local lane runs
neither ``snap_take_up`` nor ``snap_state_take_up``. Every ACS SPM unit
therefore reached the engine pass with ``takes_up_snap_if_eligible`` and
``takes_up_tanf_if_eligible`` missing, and the reviewed-null fill gave them
the engine default, ``True``: every eligible ACS unit took both programs up
(microcosm#1019, the take-up counterpart of the #765 hours defect).

This is a fresh-build repair for that lane. It fills only missing cells on
ACS-spine rows; donor-spine values and any stored ACS value are kept.

- SNAP follows :mod:`~microcosm.build.us_runtime.snap_take_up`: a unit with
  reported ``receives_snap`` always takes up, and non-reporters draw at the
  rate that puts the weighted ACS take-up share on the national FNS
  participation rate of the ``snap_take_up`` manifest stage. On ACS rows the
  reporters are native household ``FS`` (every SPM unit of an FS == 1
  housing unit), which staging writes over the QRF transfer
  (:mod:`~microcosm.build.us_runtime.acs_local_receipt_anchors`,
  microcosm#1022).
- TANF uses the take-up contract's seeded Bernoulli draw at its
  administrative rate, with no receipt anchor, exactly as the contract seeds
  it elsewhere.

Three more inputs reached the engine pass missing on every ACS row and took
an engine default that biases SNAP (microcosm#1022). None needs the engine:

- ``is_snap_abawd_discretionary_exempt`` (person) mirrors the donor's
  :mod:`~microcosm.build.us_runtime.snap_discretionary_exemption` seeding:
  a person aged 18-64 is flagged when a stable draw falls below the
  ``snap_abawd_discretionary_exemption`` manifest rate. ACS persons carry no
  complete source identity, so the draws key on
  ``acs_2024_1yr:SERIALNO:SPORDER`` (``SPORDER`` alone repeats in every
  household). Like the donor, the flag is seeded across all adults 18-64 and
  the engine intersects it with modeled ABAWD coverage, which makes it an
  upper bound on actual state usage.
- ``receives_housing_assistance`` (SPM unit) copies the transferred
  ``takes_up_housing_assistance_if_eligible``; on the donor both come from
  ``SPM_CAPHOUSESUB > 0`` and are equal by construction
  (:mod:`~microcosm.build.us_runtime.housing_inputs`). Units in an ACS
  group-quarters household (``TYPEHUGQ`` 2/3) have no housing unit and get
  ``False``. Their transferred take-up flag is left alone: forcing it is the
  transfer-side fix of microcosm#975.
- ``takes_up_medicare_if_eligible`` (person) is native ACS ``HINS3 == 1``
  (Medicare coverage at interview), the ACS counterpart of the donor's
  measured ``MCARE == 1``
  (:mod:`~microcosm.build.us_runtime.medicare_take_up`): coverage is the
  take-up signal and the engine applies eligibility. ``HINS3`` covers every
  person; a blank reads as not covered.

Two more shipped at an engine default although neither moves SNAP in
practice (microcosm#1022); the helpers of
:mod:`~microcosm.build.us_runtime.acs_local_vehicles_head_start` fill them
here too:

- ``household_vehicles_owned`` (household) is native ACS ``VEH``, the
  vehicles kept at home (0-6, 6 meaning six or more), and 0 in group
  quarters. ``household_vehicles_value`` is not filled: the ACS has no value
  item, and it stays at the reviewed engine default, 0.
- ``takes_up_head_start_if_eligible`` (person): persons aged 3-5 draw a
  stable uniform keyed on ``acs_2024_1yr:SERIALNO:SPORDER`` and take up below
  the donor spine's weighted share there, the output of the donor's SIPP
  model; everyone else is ``False``.

SNAP draws come from :func:`~microcosm.build.us_runtime.take_up._stable_unit_draws`,
which keys a unit without complete source identity on its own ids, so the
assignment depends only on ``seed``, the frame and its weights. The
per-state recalibration of the ASEC lane is not applied; the release's state
SNAP household targets reweight instead. Other ``takes_up_*`` flags on ACS
rows are not this stage's: SSI and Medicaid take-up are assigned after an
engine pre-pass by
:mod:`~microcosm.build.us_runtime.acs_local_ssi_medicaid_take_up`, and EITC,
Early Head Start and the rest still ship at the engine default
(microcosm#1022); the gate here does not grade them.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_local_vehicles_head_start import (
    ACS_VEHICLES_AVAILABLE,
    HEAD_START_AGES,
    US_HEAD_START_TAKE_UP_COLUMN,
    US_VEHICLES_OWNED_COLUMN,
    donor_head_start_share,
    grade_head_start,
    grade_vehicles_owned,
    head_start_fill,
    vehicles_owned_fill,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.medicare_take_up import (
    US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.snap_discretionary_exemption import (
    _COVERED_AGE_RANGE,
    US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN,
    _exemption_rate,
    _stable_person_draws,
    us_snap_discretionary_exemption_stage_spec,
)
from microcosm.build.us_runtime.snap_take_up import (
    _TAKE_UP_SHARE_BAND as _SNAP_TAKE_UP_SHARE_BAND,
)
from microcosm.build.us_runtime.snap_take_up import (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    _take_up_rate,
    us_snap_take_up_stage_spec,
)
from microcosm.build.us_runtime.take_up import (
    US_TAKE_UP_SHARE_BAND,
    _seed_program,
    _stable_unit_draws,
    _units_with_source_identity,
)
from microcosm.build.us_runtime.take_up_contract import (
    TakeUpProgram,
    seeded_take_up_programs,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE",
    "ACS_LOCAL_ENGINE_FREE_FILL_COLUMNS",
    "ACS_LOCAL_TAKE_UP_COLUMNS",
    "ACS_LOCAL_TAKE_UP_GATE_NAME",
    "US_HOUSING_ASSISTANCE_RECEIPT_COLUMN",
    "US_TANF_TAKE_UP_OUTPUT_COLUMN",
    "acs_local_take_up_signal_gate",
    "with_acs_local_take_up_inputs",
]

US_TANF_TAKE_UP_OUTPUT_COLUMN = "takes_up_tanf_if_eligible"
US_HOUSING_ASSISTANCE_RECEIPT_COLUMN = "receives_housing_assistance"

#: The SPM-unit take-up flags this stage owns on ACS rows.
ACS_LOCAL_TAKE_UP_COLUMNS: tuple[str, ...] = (
    US_SNAP_TAKE_UP_OUTPUT_COLUMN,
    US_TANF_TAKE_UP_OUTPUT_COLUMN,
)
ACS_LOCAL_TAKE_UP_GATE_NAME = "acs_local_take_up_signal"

ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE = "microcosm#1022"
_DISCRETIONARY = US_SNAP_DISCRETIONARY_EXEMPTION_OUTPUT_COLUMN
_HOUSING = US_HOUSING_ASSISTANCE_RECEIPT_COLUMN
(_MEDICARE,) = US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS
_VEHICLES_OWNED = US_VEHICLES_OWNED_COLUMN
_HEAD_START = US_HEAD_START_TAKE_UP_COLUMN
#: The ``(entity, column)`` inputs this stage fills on ACS rows without the
#: engine (microcosm#1022).
ACS_LOCAL_ENGINE_FREE_FILL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("person", _DISCRETIONARY),
    ("spm_unit", _HOUSING),
    ("person", _MEDICARE),
    ("household", _VEHICLES_OWNED),
    ("person", _HEAD_START),
)

_ID_COLUMN = "spm_unit_id"
#: Reported SNAP receipt: the survey anchor (on ACS rows, native household
#: FS from staging, microcosm#1022; the QRF transfer's before it).
_REPORTED_SNAP_COLUMN = "receives_snap"
_SNAP_OPERATION = "derive_snap_take_up"
_SHARE_BANDS: dict[str, tuple[float, float]] = {
    US_SNAP_TAKE_UP_OUTPUT_COLUMN: _SNAP_TAKE_UP_SHARE_BAND,
    US_TANF_TAKE_UP_OUTPUT_COLUMN: US_TAKE_UP_SHARE_BAND[US_TANF_TAKE_UP_OUTPUT_COLUMN],
}

_AGE_COLUMN = "age"
_DISCRETIONARY_OPERATION = "derive_snap_abawd_discretionary_exemption"
#: The weighted exempt share of ACS persons aged 18-64 must stay within these
#: multiples of the manifest rate ([0.04, 0.12] at the 8% cap).
_DISCRETIONARY_SHARE_FACTORS = (0.5, 1.5)
_DRAW_KEY_FORMAT = f"{ACS_2024_1YR_SPINE}:SERIALNO:SPORDER"

#: The transferred SPM-unit flag the housing receipt copies.
_HOUSING_TAKE_UP = "takes_up_housing_assistance_if_eligible"
_HOUSEHOLD_KIND = "TYPEHUGQ"
_HOUSEHOLD_KINDS = (1, 2, 3)
_HOUSING_UNIT_KIND = 1
#: ACS group quarters (institutional, noninstitutional): no housing unit.
_GROUP_QUARTERS_KINDS = (2, 3)

#: ACS Medicare coverage item (1 yes, 2 no), asked of every person.
_MEDICARE_COVERAGE = "HINS3"
_MEDICARE_COVERED = 1
_MEDICARE_AGE = 65
#: Weighted Medicare take-up among ACS persons 65+ must reach this floor; ACS
#: coverage runs well above 90% there, so a lower share means a miscoded item.
_MEDICARE_AGED_SHARE_FLOOR = 0.8

#: What the engine default would do to each column, for gate failures.
_ENGINE_DEFAULT_EFFECT = {
    _DISCRETIONARY: "no one is discretionarily exempt",
    _HOUSING: "no one receives housing assistance",
    _MEDICARE: "every eligible person enrolls in Medicare",
    _HEAD_START: "every Head Start-eligible child aged 3-5 takes Head Start up",
}


def _flags(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Boolean values with missing cells as ``False``, and the present mask."""
    present = values.notna().to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[present] = values[present].astype(bool).to_numpy(dtype=bool)
    return flags, present


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _snap_rate() -> tuple[float, str]:
    """The national rate and citation of the ``snap_take_up`` manifest stage."""
    stage = us_snap_take_up_stage_spec()
    operations = [op for op in stage.operations if op.kind == _SNAP_OPERATION]
    if len(operations) != 1:
        raise ValueError(
            f"The {stage.stage!r} stage must declare exactly one {_SNAP_OPERATION!r} "
            f"operation; found {len(operations)}."
        )
    rate = _take_up_rate(operations[0])
    return rate, str(operations[0].parameters["take_up_rate"]["source"])


def _discretionary_rate() -> tuple[float, str]:
    """The exemption rate and citation of the donor's manifest stage."""
    stage = us_snap_discretionary_exemption_stage_spec()
    operations = [op for op in stage.operations if op.kind == _DISCRETIONARY_OPERATION]
    if len(operations) != 1:
        raise ValueError(
            f"The {stage.stage!r} stage must declare exactly one "
            f"{_DISCRETIONARY_OPERATION!r} operation; found {len(operations)}."
        )
    rate = _exemption_rate(operations[0])
    return rate, str(operations[0].parameters["exemption_rate"]["source"])


def _discretionary_share_band(rate: float) -> tuple[float, float]:
    low, high = _DISCRETIONARY_SHARE_FACTORS
    return rate * low, rate * high


def _covered_age(age: np.ndarray) -> np.ndarray:
    """The donor seeding's potentially ABAWD-covered band; a blank age is not."""
    low, high = _COVERED_AGE_RANGE
    return (age >= low) & (age <= high)


def _tanf_program() -> TakeUpProgram:
    for program in seeded_take_up_programs():
        if program.variable == US_TANF_TAKE_UP_OUTPUT_COLUMN:
            return program
    raise ValueError(
        f"The take-up contract does not seed {US_TANF_TAKE_UP_OUTPUT_COLUMN!r}."
    )


def _filled(values: pd.Series, missing: np.ndarray, assigned: np.ndarray) -> pd.Series:
    if not missing.any():
        return values
    filled = values.astype(object)
    filled.loc[missing] = assigned[missing]
    return filled.astype(bool) if filled.notna().all() else filled


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _acs_person_draw_keys(
    frame: Frame, rows: np.ndarray, *, purpose: str = "discretionary-exemption"
) -> pd.Series:
    """``acs_2024_1yr:SERIALNO:SPORDER`` for the selected persons; unique."""
    household = frame.table("household")
    person = frame.table("person")
    if "SERIALNO" not in household or "SPORDER" not in person:
        raise ValueError(
            f"ACS local take-up keys the {purpose} draws on household "
            "SERIALNO and person SPORDER; the frame lacks one of them."
        )
    selected = person.loc[rows]
    serial = selected["person_household_id"].map(
        household.set_index("household_id")["SERIALNO"]
    )
    order = pd.to_numeric(selected["SPORDER"], errors="coerce")
    if serial.isna().any() or order.isna().any():
        raise ValueError(
            f"Every ACS person in the {purpose} draws needs its household "
            "SERIALNO and its SPORDER."
        )
    keys = (
        f"{ACS_2024_1YR_SPINE}:"
        + serial.astype(str)
        + ":"
        + order.astype(np.int64).astype(str)
    ).reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError(
            f"{int(keys.duplicated().sum())} ACS person draw key(s) repeat; "
            f"{_DRAW_KEY_FORMAT} must identify one person."
        )
    return keys


def _unit_household_kinds(frame: Frame) -> np.ndarray:
    """``TYPEHUGQ`` of each SPM unit's household, aligned to the spm_unit table.

    NaN where the household table lacks the column, the unit has no member,
    or the household carries no code (donor households).
    """
    unit_ids = frame.table("spm_unit")[_ID_COLUMN]
    household = frame.table("household")
    if _HOUSEHOLD_KIND not in household:
        return np.full(len(unit_ids), np.nan)
    person = frame.table("person")
    unit_household = person.drop_duplicates("person_spm_unit_id").set_index(
        "person_spm_unit_id"
    )["person_household_id"]
    kinds = pd.to_numeric(
        household.set_index("household_id")[_HOUSEHOLD_KIND], errors="coerce"
    )
    return pd.to_numeric(
        unit_ids.map(unit_household).map(kinds), errors="coerce"
    ).to_numpy(dtype=np.float64)


def _discretionary_fill(
    frame: Frame, acs_persons: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Missing ACS cells, their seeded flags, and the receipt entry."""
    person = frame.table("person")
    _, present = _flags(person[_DISCRETIONARY])
    missing = acs_persons & ~present
    rate, source = _discretionary_rate()
    covered = _covered_age(_numeric(person[_AGE_COLUMN]))
    assigned = np.zeros(len(person), dtype=bool)
    drawn = missing & covered
    if missing.any():
        # Keys for every filled person, so a repeated key is refused even when
        # the person is outside the covered ages; only the covered draw.
        keys = _acs_person_draw_keys(frame, missing)
        draws = _stable_person_draws(
            pd.DataFrame({"person_id": keys[covered[missing]].to_numpy()}),
            seed=int(seed),
        )
        assigned[drawn] = draws < rate
    return (
        missing,
        assigned,
        {
            "source": (
                "the snap_abawd_discretionary_exemption stage's seeding: age "
                f"{_COVERED_AGE_RANGE[0]}-{_COVERED_AGE_RANGE[1]} and a stable "
                "draw below the manifest rate"
            ),
            "rate": rate,
            "rate_source": source,
            "covered_ages": list(_COVERED_AGE_RANGE),
            "draw_key": _DRAW_KEY_FORMAT,
            "filled_rows": int(missing.sum()),
            "preserved_rows": int((acs_persons & present).sum()),
            "covered_filled_rows": int(drawn.sum()),
            "exempt_filled_rows": int(assigned.sum()),
            "share_band": list(_discretionary_share_band(rate)),
        },
    )


def _housing_fill(
    frame: Frame, acs_units: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Missing ACS receipt cells, their copied flags, and the receipt entry."""
    spm_unit = frame.table("spm_unit")
    _, present = _flags(spm_unit[_HOUSING])
    missing = acs_units & ~present
    take_up, take_up_present = _flags(spm_unit[_HOUSING_TAKE_UP])
    kinds = _unit_household_kinds(frame)
    unknown = missing & ~np.isin(kinds, _HOUSEHOLD_KINDS)
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} ACS SPM unit(s) have no household {_HOUSEHOLD_KIND} "
            "1/2/3; the housing-assistance receipt cannot tell a housing unit from "
            "group quarters."
        )
    group_quarters = np.isin(kinds, _GROUP_QUARTERS_KINDS)
    uncopied = missing & ~group_quarters & ~take_up_present
    if uncopied.any():
        raise ValueError(
            f"{int(uncopied.sum())} ACS housing-unit SPM unit(s) have no transferred "
            f"{_HOUSING_TAKE_UP} to copy into {_HOUSING}."
        )
    assigned = take_up & ~group_quarters
    return (
        missing,
        assigned,
        {
            "source": (
                f"copy of the transferred {_HOUSING_TAKE_UP} (equal to "
                f"{_HOUSING} on the donor by construction); False in "
                f"{_HOUSEHOLD_KIND} 2/3 group quarters"
            ),
            "filled_rows": int(missing.sum()),
            "preserved_rows": int((acs_units & present).sum()),
            "group_quarters_filled_false": int((missing & group_quarters).sum()),
            # Group-quarters units whose transferred take-up flag is True: the
            # transfer-side defect microcosm#975 owns; this receipt stays False.
            "group_quarters_take_up_true": int(
                (acs_units & group_quarters & take_up_present & take_up).sum()
            ),
        },
    )


def _medicare_fill(
    frame: Frame, acs_persons: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Missing ACS Medicare cells, their native flags, and the receipt entry."""
    person = frame.table("person")
    _, present = _flags(person[_MEDICARE])
    missing = acs_persons & ~present
    coverage = _numeric(person[_MEDICARE_COVERAGE])
    return (
        missing,
        coverage == _MEDICARE_COVERED,
        {
            "source": (
                f"ACS {_MEDICARE_COVERAGE} == {_MEDICARE_COVERED} (Medicare "
                "coverage at interview), as the donor maps ASEC MCARE == 1; "
                "coverage is the take-up signal and the engine applies "
                "eligibility"
            ),
            "filled_rows": int(missing.sum()),
            "preserved_rows": int((acs_persons & present).sum()),
            "blank_coverage_rows_as_false": int((missing & np.isnan(coverage)).sum()),
            "aged_share_floor": _MEDICARE_AGED_SHARE_FLOOR,
        },
    )


def _filled_counts(
    values: pd.Series, missing: np.ndarray, assigned: np.ndarray
) -> pd.Series:
    """``values`` with the missing cells set; whole numbers once complete."""
    if not missing.any():
        return values
    filled = pd.to_numeric(values, errors="coerce").astype(np.float64)
    filled.loc[missing] = assigned[missing]
    return filled.astype(np.int64) if filled.notna().all() else filled


def _assignment_sha256(
    spm_unit: pd.DataFrame,
    units: np.ndarray,
    person: pd.DataFrame,
    persons: np.ndarray,
    household: pd.DataFrame,
    households: np.ndarray,
) -> str:
    """Digest of the ACS rows' ids and owned cells, to prove two passes agree."""
    selected = (
        pd.DataFrame(
            {
                _ID_COLUMN: spm_unit.loc[units, _ID_COLUMN].to_numpy(),
                **{
                    column: _flags(spm_unit.loc[units, column])[0]
                    for column in (*ACS_LOCAL_TAKE_UP_COLUMNS, _HOUSING)
                },
            }
        ),
        pd.DataFrame(
            {
                "person_id": person.loc[persons, "person_id"].to_numpy(),
                **{
                    column: _flags(person.loc[persons, column])[0]
                    for column in (_DISCRETIONARY, _MEDICARE, _HEAD_START)
                },
            }
        ),
        pd.DataFrame(
            {
                "household_id": household.loc[households, "household_id"].to_numpy(),
                _VEHICLES_OWNED: _numeric(household.loc[households, _VEHICLES_OWNED]),
            }
        ),
    )
    digest = hashlib.sha256()
    for table in selected:
        hashed = pd.util.hash_pandas_object(table, index=False).to_numpy()
        digest.update(hashed.tobytes())
    return digest.hexdigest()


def with_acs_local_take_up_inputs(
    frame: Frame, *, seed: int
) -> tuple[Frame, dict[str, object]]:
    """Fill missing ACS-row take-up and default-fill inputs; keep every other cell.

    Args:
        frame: The local lane's multispine US frame. Its spm_unit and person
            tables must carry complete origin tags; spm_unit both take-up
            columns, ``receives_snap``, ``receives_housing_assistance`` and
            the transferred ``takes_up_housing_assistance_if_eligible``;
            person ``is_snap_abawd_discretionary_exempt``,
            ``takes_up_medicare_if_eligible``,
            ``takes_up_head_start_if_eligible``, ``age`` and ACS ``HINS3``;
            household ``household_vehicles_owned``, ACS ``VEH`` and
            ``TYPEHUGQ``, and complete household origin tags. Filling a
            discretionary or Head Start cell also needs household
            ``SERIALNO`` and person ``SPORDER``.
        seed: The build seed for the draws.

    Returns:
        The frame with filled ACS cells (unchanged if none were missing) and
        a JSON-ready receipt, including a digest of the ACS assignment.

    Raises:
        ValueError: If the frame is not US-schema, origin tags are missing,
            a required column is absent, or an ACS row lacks what its fill
            reads (a unique draw key, a group-quarters code, a transferred
            housing take-up flag, a ``VEH`` code), or the donor spine's Head
            Start share is unreadable.
    """
    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local take-up inputs require the US schema.")
    spm_unit = frame.table("spm_unit")
    person = frame.table("person")
    household = frame.table("household")
    tag = spine_column("spm_unit")
    person_tag = spine_column("person")
    household_tag = spine_column("household")
    for table, column in (
        (spm_unit, tag),
        (person, person_tag),
        (household, household_tag),
    ):
        if column not in table or table[column].isna().any():
            raise ValueError(
                f"ACS local take-up requires complete origin tags: {column}."
            )
    absent = [
        column
        for column in (*ACS_LOCAL_TAKE_UP_COLUMNS, _REPORTED_SNAP_COLUMN)
        if column not in spm_unit
    ]
    if absent:
        raise ValueError(
            f"ACS local take-up requires spm_unit column(s) {absent}; a missing "
            "donor column would reach the engine as universal take-up."
        )
    absent = [
        f"{entity}.{column}"
        for entity, columns in (
            ("spm_unit", (_HOUSING, _HOUSING_TAKE_UP)),
            (
                "person",
                (
                    _DISCRETIONARY,
                    _MEDICARE,
                    _HEAD_START,
                    _AGE_COLUMN,
                    _MEDICARE_COVERAGE,
                ),
            ),
            ("household", (_VEHICLES_OWNED, ACS_VEHICLES_AVAILABLE, _HOUSEHOLD_KIND)),
        )
        for column in columns
        if column not in frame.table(entity)
    ]
    if absent:
        staging = (
            f" ACS {ACS_VEHICLES_AVAILABLE} comes from staging: re-run "
            "tools/build_us_acs_multispine_base.py with the current builder."
            if ACS_VEHICLES_AVAILABLE not in household
            else ""
        )
        raise ValueError(
            f"ACS local take-up requires column(s) {absent} for the engine-free "
            f"fills ({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}); a missing column would "
            f"reach the engine at its default.{staging}"
        )
    acs = spm_unit[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    acs_persons = person[person_tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    acs_households = (
        household[household_tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    )
    donor_persons = person[person_tag].eq(ASEC_PUF_DONOR_SPINE).to_numpy(dtype=bool)
    units = _units_with_source_identity(frame, "spm_unit")
    if not np.array_equal(units[_ID_COLUMN].to_numpy(), spm_unit[_ID_COLUMN]):
        raise ValueError("SPM-unit source identity does not align with the table.")
    weights = units["_weight"].to_numpy(dtype=np.float64)
    acs_weight = float(weights[acs].sum())

    # SNAP: reporters take up; non-reporters draw to the national rate.
    snap_flags, snap_present = _flags(spm_unit[US_SNAP_TAKE_UP_OUTPUT_COLUMN])
    snap_missing = acs & ~snap_present
    reported, reported_present = _flags(spm_unit[_REPORTED_SNAP_COLUMN])
    anchored = snap_missing & reported
    open_units = snap_missing & ~reported
    rate, rate_source = _snap_rate()
    # Weight the non-reporter draws must add for the ACS share to reach the
    # rate; when reporters and stored cells already exceed it, no one draws.
    shortfall = (
        rate * acs_weight
        - float(weights[acs & snap_present & snap_flags].sum())
        - float(weights[anchored].sum())
    )
    open_weight = float(weights[open_units].sum())
    draw_rate = (
        min(1.0, max(0.0, shortfall) / open_weight) if open_weight > 0.0 else 0.0
    )
    snap_assigned = anchored.copy()
    if open_units.any():
        draws = _stable_unit_draws(
            units.loc[open_units],
            id_column=_ID_COLUMN,
            seed=int(seed),
            variable=US_SNAP_TAKE_UP_OUTPUT_COLUMN,
        )
        snap_assigned[open_units] = draws < draw_rate

    # TANF: the contract's seeded Bernoulli draw, no receipt anchor.
    program = _tanf_program()
    seeded, _ = _seed_program(frame, program, seed=int(seed))
    if not np.array_equal(seeded[_ID_COLUMN], spm_unit[_ID_COLUMN].to_numpy()):
        raise ValueError("Seeded TANF take-up does not align with the spm_unit table.")
    tanf_assigned = np.asarray(seeded[US_TANF_TAKE_UP_OUTPUT_COLUMN], dtype=bool)
    tanf_missing = acs & ~_flags(spm_unit[US_TANF_TAKE_UP_OUTPUT_COLUMN])[1]

    # microcosm#1022: the engine-free fills.
    discretionary_missing, discretionary_assigned, discretionary = _discretionary_fill(
        frame, acs_persons, seed=int(seed)
    )
    housing_missing, housing_assigned, housing = _housing_fill(frame, acs)
    medicare_missing, medicare_assigned, medicare = _medicare_fill(frame, acs_persons)
    vehicles_missing, vehicles_assigned, vehicles = vehicles_owned_fill(
        household, acs_households
    )
    head_start_missing, head_start_assigned, head_start = head_start_fill(
        person,
        acs_persons,
        donor_persons,
        np.asarray(frame.resolve_weights("person").values, dtype=np.float64),
        seed=int(seed),
        draw_keys=lambda rows: _acs_person_draw_keys(
            frame, rows, purpose="Head Start take-up"
        ),
    )

    result = frame
    replaced: dict[str, pd.DataFrame] = {}
    if snap_missing.any() or tanf_missing.any() or housing_missing.any():
        updated = spm_unit.copy()
        for column, missing, assigned in (
            (US_SNAP_TAKE_UP_OUTPUT_COLUMN, snap_missing, snap_assigned),
            (US_TANF_TAKE_UP_OUTPUT_COLUMN, tanf_missing, tanf_assigned),
            (_HOUSING, housing_missing, housing_assigned),
        ):
            updated[column] = _filled(spm_unit[column], missing, assigned)
        replaced["spm_unit"] = updated
    if (
        discretionary_missing.any()
        or medicare_missing.any()
        or head_start_missing.any()
    ):
        updated = person.copy()
        for column, missing, assigned in (
            (_DISCRETIONARY, discretionary_missing, discretionary_assigned),
            (_MEDICARE, medicare_missing, medicare_assigned),
            (_HEAD_START, head_start_missing, head_start_assigned),
        ):
            updated[column] = _filled(person[column], missing, assigned)
        replaced["person"] = updated
    if vehicles_missing.any():
        updated = household.copy()
        updated[_VEHICLES_OWNED] = _filled_counts(
            household[_VEHICLES_OWNED], vehicles_missing, vehicles_assigned
        )
        replaced["household"] = updated
    if replaced:
        result = Frame(
            {
                entity: replaced[entity] if entity in replaced else frame.table(entity)
                for entity in frame.entities
            },
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
    final = result.table("spm_unit")
    final_person = result.table("person")
    final_household = result.table("household")
    acs_weights = weights[acs]
    person_weights = np.asarray(
        result.resolve_weights("person").values, dtype=np.float64
    )
    household_weights = np.asarray(
        result.resolve_weights("household").values, dtype=np.float64
    )
    age = _numeric(final_person[_AGE_COLUMN])
    covered = acs_persons & _covered_age(age)
    aged = acs_persons & (age >= _MEDICARE_AGE)
    head_start_ages = (
        acs_persons & (age >= HEAD_START_AGES[0]) & (age <= HEAD_START_AGES[1])
    )
    housing_units = acs_households & (
        _numeric(final_household[_HOUSEHOLD_KIND]) == _HOUSING_UNIT_KIND
    )
    discretionary["weighted_covered_exempt_share"] = _share(
        person_weights[covered], _flags(final_person.loc[covered, _DISCRETIONARY])[0]
    )
    housing["weighted_receipt_share"] = _share(
        acs_weights, _flags(final.loc[acs, _HOUSING])[0]
    )
    medicare["weighted_aged_take_up_share"] = _share(
        person_weights[aged], _flags(final_person.loc[aged, _MEDICARE])[0]
    )
    vehicles["weighted_housing_unit_share_with_vehicle"] = _share(
        household_weights[housing_units],
        _numeric(final_household.loc[housing_units, _VEHICLES_OWNED]) > 0,
    )
    head_start["weighted_age_domain_take_up_share"] = _share(
        person_weights[head_start_ages],
        _flags(final_person.loc[head_start_ages, _HEAD_START])[0],
    )
    receipt: dict[str, object] = {
        "issue": "microcosm#1019",
        "seed": int(seed),
        "spine": ACS_2024_1YR_SPINE,
        "acs_spm_units": int(acs.sum()),
        "programs": {
            US_SNAP_TAKE_UP_OUTPUT_COLUMN: {
                "rate": rate,
                "rate_source": rate_source,
                "filled_rows": int(snap_missing.sum()),
                "preserved_rows": int((acs & snap_present).sum()),
                "reported_receipt_rows": int(anchored.sum()),
                "missing_receipt_rows_as_non_reporters": int(
                    (snap_missing & ~reported_present).sum()
                ),
                "non_reporter_draw_rate": draw_rate,
                "weighted_take_up_share": _share(
                    acs_weights,
                    _flags(final.loc[acs, US_SNAP_TAKE_UP_OUTPUT_COLUMN])[0],
                ),
            },
            US_TANF_TAKE_UP_OUTPUT_COLUMN: {
                "rate": program.rate.get("value"),
                "rate_source": program.rate.get("source"),
                "filled_rows": int(tanf_missing.sum()),
                "preserved_rows": int((acs & ~tanf_missing).sum()),
                "weighted_take_up_share": _share(
                    acs_weights,
                    _flags(final.loc[acs, US_TANF_TAKE_UP_OUTPUT_COLUMN])[0],
                ),
            },
        },
        "engine_free_fills": {
            "issue": ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE,
            "acs_persons": int(acs_persons.sum()),
            "acs_households": int(acs_households.sum()),
            "columns": {
                _DISCRETIONARY: discretionary,
                _HOUSING: housing,
                _MEDICARE: medicare,
                _VEHICLES_OWNED: vehicles,
                _HEAD_START: head_start,
            },
        },
        "assigned_sha256": _assignment_sha256(
            final, acs, final_person, acs_persons, final_household, acs_households
        ),
    }
    return result, receipt


def _flag_entry(
    values: pd.Series, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    flags, present = _flags(values)
    return (
        flags,
        present,
        {
            "missing_rows": int((~present).sum()),
            "unique_values": len(np.unique(flags[present])),
            "share": _share(weights, flags),
        },
    )


def _complete_flags(
    table: pd.DataFrame,
    column: str,
    weights: np.ndarray,
    *,
    entity: str,
    columns: dict[str, object],
    failures: list[str],
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Record one column's completeness and variation; ``None`` if absent."""
    if column not in table:
        failures.append(f"missing {column}.")
        return None
    flags, present, entry = _flag_entry(table[column], weights)
    entry["entity"] = entity
    columns[column] = entry
    if not present.all():
        failures.append(
            f"{column} has missing rows; an engine-default fill means "
            f"{_ENGINE_DEFAULT_EFFECT[column]}."
        )
    if entry["unique_values"] < 2:
        failures.append(
            f"{column} is constant; the engine default is the same landmine."
        )
    return flags, entry


def _engine_free_fill_grades(
    frame: Frame,
    *,
    units: np.ndarray,
    persons: np.ndarray,
    households: np.ndarray,
    unit_weights: np.ndarray,
    person_weights: np.ndarray,
    household_weights: np.ndarray,
    head_start_donor_share: float | None,
    graded: bool,
    columns: dict[str, object],
) -> list[str]:
    """Grade the #1022 columns on one spine's rows; record entries in ``columns``.

    Every spine must carry each column complete and non-constant. Only
    ``graded`` (ACS) rows are held to the fill's own contract: the exempt
    share band and age scope, the copied housing receipt, native ``HINS3``
    Medicare coverage, native ``VEH`` vehicle counts, and Head Start take-up
    within ages 3-5 around the donor spine's share. Donor rows are reported
    only.
    """
    person = frame.table("person").loc[persons]
    spm_unit = frame.table("spm_unit").loc[units]
    household = frame.table("household").loc[households]
    person_weights = person_weights[persons]
    unit_weights = unit_weights[units]
    household_weights = household_weights[households]
    failures: list[str] = []
    recorded = {"columns": columns, "failures": failures}
    graded_flags = _complete_flags(
        person, _DISCRETIONARY, person_weights, entity="person", **recorded
    )
    if graded_flags is not None:
        failures += _grade_discretionary(
            person, graded_flags[0], person_weights, graded_flags[1], graded=graded
        )
    graded_flags = _complete_flags(
        spm_unit, _HOUSING, unit_weights, entity="spm_unit", **recorded
    )
    if graded_flags is not None:
        failures += _grade_housing(
            spm_unit,
            graded_flags[0],
            _unit_household_kinds(frame)[units],
            graded_flags[1],
            graded=graded,
        )
    graded_flags = _complete_flags(
        person, _MEDICARE, person_weights, entity="person", **recorded
    )
    if graded_flags is not None:
        failures += _grade_medicare(
            person, graded_flags[0], person_weights, graded_flags[1], graded=graded
        )
    vehicles: dict[str, object] = {"entity": "household"}
    if _VEHICLES_OWNED not in household:
        failures.append(f"missing {_VEHICLES_OWNED}.")
    else:
        columns[_VEHICLES_OWNED] = vehicles
        failures += grade_vehicles_owned(
            household, household_weights, vehicles, graded=graded
        )
    graded_flags = _complete_flags(
        person, _HEAD_START, person_weights, entity="person", **recorded
    )
    if graded_flags is not None:
        failures += grade_head_start(
            person,
            graded_flags[0],
            person_weights,
            graded_flags[1],
            donor_share=head_start_donor_share,
            graded=graded,
        )
    return failures


def _grade_discretionary(
    person: pd.DataFrame,
    flags: np.ndarray,
    weights: np.ndarray,
    entry: dict[str, object],
    *,
    graded: bool,
) -> list[str]:
    if _AGE_COLUMN not in person:
        return [f"missing {_AGE_COLUMN}; the exempt share cannot be scoped."]
    rate, _ = _discretionary_rate()
    low, high = _discretionary_share_band(rate)
    covered = _covered_age(_numeric(person[_AGE_COLUMN]))
    share = _share(weights[covered], flags[covered])
    outside = int((flags & ~covered).sum())
    entry.update(
        covered_exempt_share=share,
        share_band=[low, high],
        exempt_outside_covered_ages=outside,
    )
    if not graded:
        return []
    failures = []
    if not (low <= share <= high):
        failures.append(
            f"{_DISCRETIONARY} share among ages {_COVERED_AGE_RANGE[0]}-"
            f"{_COVERED_AGE_RANGE[1]} is {share:.3f}, outside [{low:.3f}, "
            f"{high:.3f}] around the manifest rate {rate}."
        )
    if outside:
        failures.append(
            f"{outside} person(s) outside ages {_COVERED_AGE_RANGE[0]}-"
            f"{_COVERED_AGE_RANGE[1]} carry {_DISCRETIONARY}."
        )
    return failures


def _grade_housing(
    spm_unit: pd.DataFrame,
    flags: np.ndarray,
    kinds: np.ndarray,
    entry: dict[str, object],
    *,
    graded: bool,
) -> list[str]:
    if _HOUSING_TAKE_UP not in spm_unit:
        return [f"missing {_HOUSING_TAKE_UP}; the housing receipt cannot be checked."]
    take_up, _ = _flags(spm_unit[_HOUSING_TAKE_UP])
    group_quarters = np.isin(kinds, _GROUP_QUARTERS_KINDS)
    disagreeing = int((~group_quarters & (flags != take_up)).sum())
    gq_receiving = int((group_quarters & flags).sum())
    unknown = int((~np.isin(kinds, _HOUSEHOLD_KINDS)).sum())
    entry.update(
        disagrees_with_take_up=disagreeing,
        group_quarters_units=int(group_quarters.sum()),
        group_quarters_receiving=gq_receiving,
    )
    if not graded:
        return []
    failures = []
    if unknown:
        failures.append(
            f"{unknown} SPM unit(s) have no household {_HOUSEHOLD_KIND} 1/2/3."
        )
    if disagreeing:
        failures.append(
            f"{_HOUSING} differs from the transferred {_HOUSING_TAKE_UP} on "
            f"{disagreeing} housing-unit SPM unit(s)."
        )
    if gq_receiving:
        failures.append(
            f"{gq_receiving} group-quarters SPM unit(s) receive housing assistance."
        )
    return failures


def _grade_medicare(
    person: pd.DataFrame,
    flags: np.ndarray,
    weights: np.ndarray,
    entry: dict[str, object],
    *,
    graded: bool,
) -> list[str]:
    if not graded:
        return []
    failures = []
    if _MEDICARE_COVERAGE not in person:
        return [f"missing {_MEDICARE_COVERAGE}; Medicare take-up cannot be checked."]
    native = _numeric(person[_MEDICARE_COVERAGE]) == _MEDICARE_COVERED
    mismatched = int((flags != native).sum())
    entry["differs_from_native_coverage"] = mismatched
    if mismatched:
        failures.append(
            f"{_MEDICARE} differs from ACS {_MEDICARE_COVERAGE} == "
            f"{_MEDICARE_COVERED} on {mismatched} person(s)."
        )
    if _AGE_COLUMN not in person:
        return [*failures, f"missing {_AGE_COLUMN}; the aged share cannot be scoped."]
    aged = _numeric(person[_AGE_COLUMN]) >= _MEDICARE_AGE
    if aged.any():
        share = _share(weights[aged], flags[aged])
        entry["aged_take_up_share"] = share
        if share < _MEDICARE_AGED_SHARE_FLOOR:
            failures.append(
                f"{_MEDICARE} share among ages {_MEDICARE_AGE}+ is {share:.3f}, "
                f"below {_MEDICARE_AGED_SHARE_FLOOR}."
            )
    return failures


def acs_local_take_up_signal_gate(frame: Frame) -> GateResult:
    """Require complete, anchored, plausible ACS-lane runtime inputs per origin.

    For each spine, fails when a flag is missing, has missing cells, or is
    constant (the engine-default universal take-up this lane shipped). On the
    ACS spine, whose cells this stage assigns, it also fails when a share
    leaves its plausibility band or (SNAP) a ``receives_snap`` reporter does
    not take up. The #1022 columns are held to the same completeness and
    variation on both spines and, on the ACS spine, to their fills: an
    exempt share of ages 18-64 around the manifest rate and no one exempt
    outside them, a housing receipt equal to the transferred take-up flag
    (``False`` in group quarters), Medicare take-up equal to ``HINS3 == 1``
    with a high share at 65+, vehicle counts equal to ``VEH`` (0 in group
    quarters), and Head Start take-up only at ages 3-5, at a share around the
    donor spine's. Donor-spine shares and anchors are reported but not
    graded: those cells come from the donor release, whose own gates graded
    them. Other ``takes_up_*`` columns are not graded (microcosm#1022).
    """
    spm_unit = frame.table("spm_unit")
    person = frame.table("person")
    household = frame.table("household")
    tag = spine_column("spm_unit")
    person_tag = spine_column("person")
    household_tag = spine_column("household")
    failures: list[str] = []
    by_spine: dict[str, object] = {}
    if tag not in spm_unit or spm_unit[tag].isna().any():
        failures.append(f"Missing SPM-unit origin tags: {tag}.")
    elif person_tag not in person or person[person_tag].isna().any():
        failures.append(f"Missing person origin tags: {person_tag}.")
    elif household_tag not in household or household[household_tag].isna().any():
        failures.append(f"Missing household origin tags: {household_tag}.")
    else:
        weights = np.asarray(frame.resolve_weights("spm_unit").values, dtype=float)
        person_weights = np.asarray(frame.resolve_weights("person").values, dtype=float)
        household_weights = np.asarray(
            frame.resolve_weights("household").values, dtype=float
        )
        head_start_share: float | None = None
        if _HEAD_START in person and _AGE_COLUMN in person:
            try:
                head_start_share = donor_head_start_share(
                    person[_HEAD_START],
                    _numeric(person[_AGE_COLUMN]),
                    person[person_tag].eq(ASEC_PUF_DONOR_SPINE).to_numpy(dtype=bool),
                    person_weights,
                )
            except ValueError as exc:
                failures.append(f"{ACS_2024_1YR_SPINE}: {exc}")
        for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
            selected = spm_unit[tag].eq(spine).to_numpy(dtype=bool)
            # Only the ACS cells are this stage's output; grade their
            # plausibility and anchor. Donor cells are graded upstream.
            graded = spine == ACS_2024_1YR_SPINE
            columns: dict[str, object] = {}
            by_spine[spine] = {
                "rows": int(selected.sum()),
                "graded": graded,
                "columns": columns,
            }
            if not selected.any():
                failures.append(f"{spine}: no spm_unit rows.")
                continue
            for column, (low, high) in _SHARE_BANDS.items():
                if column not in spm_unit:
                    failures.append(f"{spine}: missing {column}.")
                    continue
                flags, present = _flags(spm_unit.loc[selected, column])
                share = _share(weights[selected], flags)
                unique = len(np.unique(flags[present]))
                entry: dict[str, object] = {
                    "missing_rows": int((~present).sum()),
                    "unique_values": unique,
                    "take_up_share": share,
                    "share_band": [low, high],
                }
                columns[column] = entry
                if not present.all():
                    failures.append(
                        f"{spine}: {column} has missing rows; an engine-default "
                        "fill is universal take-up."
                    )
                if unique < 2:
                    failures.append(
                        f"{spine}: {column} is constant; universal take-up is the "
                        "engine-default landmine."
                    )
                if graded and not (low <= share <= high):
                    failures.append(
                        f"{spine}: {column} take-up share {share:.3f} outside "
                        f"plausibility band [{low}, {high}]."
                    )
                if column != US_SNAP_TAKE_UP_OUTPUT_COLUMN:
                    continue
                if _REPORTED_SNAP_COLUMN not in spm_unit:
                    failures.append(
                        f"{spine}: missing {_REPORTED_SNAP_COLUMN}; the reported-"
                        "receipt anchor cannot be verified."
                    )
                    continue
                reported, _ = _flags(spm_unit.loc[selected, _REPORTED_SNAP_COLUMN])
                not_taking_up = int((reported & present & ~flags).sum())
                entry["reporters_not_taking_up"] = not_taking_up
                if graded and not_taking_up:
                    failures.append(
                        f"{spine}: {not_taking_up} SPM unit(s) report SNAP receipt "
                        "but carry no take-up; reported recipients must take up."
                    )
            failures += [
                f"{spine}: {failure}"
                for failure in _engine_free_fill_grades(
                    frame,
                    units=selected,
                    persons=person[person_tag].eq(spine).to_numpy(dtype=bool),
                    households=household[household_tag].eq(spine).to_numpy(dtype=bool),
                    unit_weights=weights,
                    person_weights=person_weights,
                    household_weights=household_weights,
                    head_start_donor_share=head_start_share,
                    graded=graded,
                    columns=columns,
                )
            ]
        known = {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}
        if (
            set(spm_unit[tag].unique()) - known
            or set(person[person_tag].unique()) - known
            or set(household[household_tag].unique()) - known
        ):
            failures.append("Local take-up origin tags contain an unsupported spine.")
    return GateResult(
        name=ACS_LOCAL_TAKE_UP_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={"per_spine": by_spine},
    )
