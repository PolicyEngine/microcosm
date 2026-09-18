"""The SPM measurement composition check, as a pure frame check.

spm-calculator 1.0.0 classifies an SPM unit's adults as
``unit.sum((age >= 18) | ((age >= 15) & role))`` with ``role`` the engine's
one declared dataset source input, ``is_spm_independent_minor_role``, and
refuses the **whole population's** SPM measurement
(``SPMInputError("SPM_COMPOSITION_REQUIRED")``) when a single unit has none.
:func:`check_spm_composition` reproduces that classification over frame
columns, in seconds and without an engine, naming the offending units and the
remedy.

This module is deliberately free of build-tool imports so that build stages
(``spm_independence_role``) and the primary-QRF worker's static import
closure can reach the check; the release-gate preflight runbook re-exports
everything here and adds the checks that need the release tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from microcosm.frame import Frame

__all__ = [
    "SPM_COMPOSITION_REMEDY",
    "CheckResult",
    "PreflightStatus",
    "check_spm_composition",
    "spm_independence_role",
]

#: A check's verdict.
PreflightStatus = Literal["PASS", "FAIL", "AT_RISK", "SKIPPED"]

#: The person columns the SPM measurement composition rule reads, named exactly
#: as ``spm-calculator`` names them on a microcosm Frame
#: (``spm_calculator/microcosm_adapter.py`` ``PERSON_COLUMNS`` maps these frame
#: column names onto the engine variables) and as policyengine-us registers them.
_SPM_AGE_COLUMN = "age"
_SPM_ROLE_COLUMN = "is_spm_independent_minor_role"
_SPM_HEAD_COLUMN = "is_household_head"
_SPM_SPOUSE_COLUMN = "is_household_spouse"

#: How many offending SPM units the composition check names in its failure line
#: and its rows. The failure line reports the full count either way.
_MAX_REPORTED_SPM_UNITS = 20

#: The single-sourced remedy for a zero-classified-adult SPM unit. The release
#: tool raises with this same text, so an operator who skipped preflight reads
#: the identical instruction.
SPM_COMPOSITION_REMEDY = (
    "Remedy: supply the SPM independence role. Either (a) carry "
    "is_spm_independent_minor_role for these persons — for a base built from "
    "raw sources, the spm_independence_role build stage "
    "(microcosm.build.us_runtime.spm_independence_role) restores it from the "
    "pinned Census ASEC person files; for a Build P descendant, "
    "tools/build_us_spm_role_enrichment.py derives it from the same raw ASEC "
    "role rule (microcosm.build.us_runtime.spm_role_source.SPM_ROLE_RULE) — or "
    "(b) carry is_household_head / is_household_spouse through to the export so "
    "the engine's fallback formula classifies a 15-to-17-year-old head or spouse. "
    "Re-grouping child-only SPM units into another unit is NOT a remedy: it "
    "changes the poverty measurement the 104 state SPM poverty levels exist to "
    "check. Inventing an adult is not a remedy either."
)


@dataclass(frozen=True)
class CheckResult:
    """One preflight check's verdict and the numbers behind it.

    Attributes:
        name: Stable check id.
        status: ``PASS`` / ``FAIL`` / ``AT_RISK`` / ``SKIPPED``.
        summary: One-line human summary with the headline number.
        failures: Hard-failure lines (drive exit 1).
        at_risks: Advisory lines (drive exit 2 when no failure).
        rows: Per-item measured numbers, JSON-ready, for the report table.
        details: Extra machine-readable context.
    """

    name: str
    status: PreflightStatus
    summary: str
    failures: tuple[str, ...] = ()
    at_risks: tuple[str, ...] = ()
    rows: tuple[Mapping[str, Any], ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "failures": list(self.failures),
            "at_risks": list(self.at_risks),
            "rows": [dict(row) for row in self.rows],
            "details": dict(self.details),
        }


def _boolean_column(series: pd.Series) -> np.ndarray:
    """A role column as plain bool; a missing value reads False, as an absent
    engine input does."""
    return series.fillna(False).to_numpy(dtype=bool)


def spm_independence_role(person: pd.DataFrame) -> tuple[np.ndarray, str, dict]:
    """Each person's SPM independence role, and the source the engine would use.

    Reproduces ``is_spm_independent_minor_role`` resolution exactly. The
    variable carries a *formula* (``is_household_head | is_household_spouse``,
    ``spm_calculator/policyengine_adapter.py``) but is also the one declared
    dataset source input (``policyengine_us.spm.DATASET_SOURCE_INPUTS``), so a
    supplied column wins and the formula runs only when none is supplied — the
    behaviour ``policyengine_us/tests/core/test_spm_source_input_contract.py``
    pins.

    With no column supplied, ``is_household_head`` and ``is_household_spouse``
    are themselves input-only bools with no formula of their own, so an absent
    one contributes False rather than a computed value. Absent role *and* absent
    structure therefore leave everyone unclassified and the rule collapses to
    ``age >= 18``.

    Returns the role vector, the source label
    (``source_column`` / ``household_structure_fallback`` / ``unclassified``),
    and the provenance detail for the report.
    """
    if _SPM_ROLE_COLUMN in person.columns:
        column = person[_SPM_ROLE_COLUMN]
        return (
            _boolean_column(column),
            "source_column",
            {
                "role_column": _SPM_ROLE_COLUMN,
                "role_column_present": True,
                "fallback_columns_present": [],
                "role_missing_values_read_as_false": int(column.isna().sum()),
            },
        )
    present = [
        name
        for name in (_SPM_HEAD_COLUMN, _SPM_SPOUSE_COLUMN)
        if name in person.columns
    ]
    role = np.zeros(len(person), dtype=bool)
    for name in present:
        role |= _boolean_column(person[name])
    return (
        role,
        "household_structure_fallback" if present else "unclassified",
        {
            "role_column": None,
            "role_column_present": False,
            "fallback_columns_present": present,
            "role_missing_values_read_as_false": 0,
        },
    )


def _person_ages(person: pd.DataFrame) -> np.ndarray:
    """The person table's ``age`` as float64; unreadable values become NaN.

    NaN never satisfies ``>= 18`` or ``>= 15``, which is how the engine's
    comparison treats a missing age too — such a person is simply never counted
    as an adult, so an all-missing-age unit surfaces here as offending.
    """
    if _SPM_AGE_COLUMN not in person.columns:
        raise ValueError(
            f"The person table carries no {_SPM_AGE_COLUMN!r} column, so the "
            "SPM measurement composition rule cannot be evaluated."
        )
    return pd.to_numeric(person[_SPM_AGE_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )


def _sum_by_unit(
    flags: np.ndarray, membership: np.ndarray, unit_ids: np.ndarray
) -> np.ndarray:
    """``unit.sum(flags)`` over the declared membership column.

    Reindexed onto the unit table, so a unit with no member rows reads as zero
    — which is what ``unit.sum`` over no members gives, and which the engine
    refuses just the same.
    """
    return (
        pd.Series(np.asarray(flags).astype(np.int64), index=membership)
        .groupby(level=0)
        .sum()
        .reindex(unit_ids, fill_value=0)
        .to_numpy(dtype=np.int64)
    )


@dataclass(frozen=True)
class _SPMComposition:
    """One frame's SPM measurement classification, as the engine would compute it."""

    unit_ids: np.ndarray
    adults: np.ndarray
    offending: np.ndarray
    membership: np.ndarray
    age: np.ndarray
    role: np.ndarray
    role_source: str
    role_details: dict[str, Any]
    n_units_without_member_18_plus: int


def _spm_composition(frame: Frame, *, unit_entity: str) -> _SPMComposition:
    """Classify one frame under ``adult = (age >= 18) | ((age >= 15) & role)``."""
    schema = frame.schema
    person = frame.table("person")
    unit_ids = frame.table(unit_entity)[schema.id_column(unit_entity)].to_numpy()
    membership = person[schema.membership_column(unit_entity)].to_numpy()

    age = _person_ages(person)
    role, role_source, role_details = spm_independence_role(person)
    with np.errstate(invalid="ignore"):
        adult = (age >= 18.0) | ((age >= 15.0) & role)
        age_18_plus = age >= 18.0

    adults = _sum_by_unit(adult, membership, unit_ids)
    return _SPMComposition(
        unit_ids=unit_ids,
        adults=adults,
        offending=adults < 1,
        membership=membership,
        age=age,
        role=role,
        role_source=role_source,
        role_details={
            **role_details,
            "role_source": role_source,
            "ages_unreadable_as_numbers": int(np.isnan(age).sum()),
        },
        n_units_without_member_18_plus=int(
            (_sum_by_unit(age_18_plus, membership, unit_ids) < 1).sum()
        ),
    )


def _offending_unit_rows(
    composition: _SPMComposition, *, unit_entity: str, max_reported: int
) -> tuple[dict[str, Any], ...]:
    """Up to ``max_reported`` offending units with their members' ages and roles."""
    rows: list[dict[str, Any]] = []
    for unit_id in composition.unit_ids[composition.offending][:max_reported]:
        members = composition.membership == unit_id
        ages = composition.age[members]
        rows.append(
            {
                f"{unit_entity}_id": unit_id.item()
                if hasattr(unit_id, "item")
                else unit_id,
                "n_members": int(members.sum()),
                "member_ages": [
                    None if np.isnan(value) else float(value) for value in ages
                ],
                "member_independence_roles": [
                    bool(value) for value in composition.role[members]
                ],
            }
        )
    return tuple(rows)


def check_spm_composition(
    base_frame: Frame,
    selected_frame: Frame | None = None,
    *,
    unit_entity: str = "spm_unit",
    max_reported: int = _MAX_REPORTED_SPM_UNITS,
) -> CheckResult:
    """Does every SPM unit have a classified adult (no solve, no engine)?

    The release's 104 state SPM poverty levels are measured on one whole-dataset
    ``Microsimulation`` (``reform_validation.default_simulate_factory``), and in
    ``spm-calculator`` 1.0.0 a *single* SPM unit with no classified adult raises
    ``SPMInputError("SPM_COMPOSITION_REQUIRED")`` for the **whole population's**
    measurement (``spm_calculator/policyengine_adapter.py`` ``policyengine_amount``:
    ``if np.any(adults < 1): raise``), naming neither the offending unit nor a
    remedy. The release tool reaches that call only after calibration, export and
    the NPZ write, so the traceback arrives hours into a build.

    This reproduces the engine's classification exactly —
    ``adult = (age >= 18) | ((age >= 15) & role)`` with ``role`` resolved by
    :func:`spm_independence_role` — over frame columns, in seconds.

    ``base_frame`` is the pool. ``selected_frame``, when given, is the population
    the release calibrates, exports and measures; the verdict is graded on it,
    because a pool unit the selection drops never reaches the engine. The pool
    count is still reported, since it is what a *future* selection must keep
    avoiding.
    """
    graded = selected_frame if selected_frame is not None else base_frame
    scope = "selected pool" if selected_frame is not None else "base pool"

    composition = _spm_composition(graded, unit_entity=unit_entity)
    n_offending = int(composition.offending.sum())
    n_units = int(len(composition.unit_ids))

    details: dict[str, Any] = {
        "graded_scope": scope,
        "unit_entity": unit_entity,
        "n_units": n_units,
        "n_units_without_classified_adult": n_offending,
        "n_units_without_member_aged_18_or_over": (
            composition.n_units_without_member_18_plus
        ),
        **composition.role_details,
    }
    if selected_frame is not None:
        pool = _spm_composition(base_frame, unit_entity=unit_entity)
        details["base_pool_n_units"] = int(len(pool.unit_ids))
        details["base_pool_n_units_without_classified_adult"] = int(
            pool.offending.sum()
        )
        details["base_pool_n_units_without_member_aged_18_or_over"] = (
            pool.n_units_without_member_18_plus
        )

    rows = _offending_unit_rows(
        composition, unit_entity=unit_entity, max_reported=max_reported
    )
    details["n_units_reported"] = len(rows)

    if n_offending == 0:
        return CheckResult(
            name="spm_composition",
            status="PASS",
            summary=(
                f"all {n_units:,} {unit_entity}s in the {scope} have a "
                f"classified adult (role source: {composition.role_source})"
            ),
            rows=rows,
            details=details,
        )

    named = ", ".join(str(row[f"{unit_entity}_id"]) for row in rows)
    elided = n_offending - len(rows)
    return CheckResult(
        name="spm_composition",
        status="FAIL",
        summary=(
            f"{n_offending:,} of {n_units:,} {unit_entity}s in the {scope} have "
            "no classified adult; the whole population's SPM measurement would "
            "raise SPM_COMPOSITION_REQUIRED"
        ),
        failures=(
            f"{n_offending} {unit_entity}(s) have no member aged 18 or over and "
            "no member aged 15-17 carrying an SPM independence role (role "
            f"source: {composition.role_source}). "
            f"{unit_entity}_id(s): {named}"
            + (f" (+{elided} more)" if elided > 0 else "")
            + ".",
            SPM_COMPOSITION_REMEDY,
        ),
        rows=rows,
        details=details,
    )
