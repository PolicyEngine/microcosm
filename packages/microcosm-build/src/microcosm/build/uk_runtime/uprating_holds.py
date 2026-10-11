"""Every UK target held at an earlier period carries a declared reason (#1123).

A compiled target whose resolved Chronicle fact is older than the calibration
period must either have been uprated by a declared index (the applier stamps
``uprating_factor``) or match a declaration in ``uk/uprating_holds.json``.
Before microcosm#1123 the compile only stamped the periods and moved on: 509
national rows and every local row were held with no declaration and nothing
refused them.

Hold kinds:

* ``matched_period``: the binding measures the fact's own period on purpose
  (the CGT rows are measured under the 2024-25 rules they were published for).
* ``in_year_snapshot``: a stock counted at a date inside the calibration year
  (council-tax dwellings at September or October 2025, the benefit cap in
  November 2025); the date is the period, not a lag.
* ``control_rescaled``: the cell's level is set by a calibration-year control
  in its cross-grain group, so only its share is held (the census household
  and tenure cells under the household controls).
* ``reviewed_no_index``: a public spending line with no index that would
  carry it forward, held to its latest outturn until the next edition.

A hold is a signed decision. An entry is in force only with an approver, an
approval date and the status ``approved``: the compile refuses the rows of
any other entry as it refuses undeclared ones, so an open decision cannot be
merged around.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
from datetime import date
from importlib import resources as importlib_resources
from typing import Any

from microcosm.build.ledger_targets import (
    MONTHLY_WINDOW_COUNT_X_MEAN,
    MONTHLY_WINDOW_OPERATIONS,
    ROLLED_FORWARD_BY_RATIO,
    _period_key_from_value,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

UK_UPRATING_HOLDS_RESOURCE = "uprating_holds.json"
#: Value operations that resolve a window of the calibration year's own
#: subperiods (a calendar-year mean, a declared calendar window, a monthly
#: Stat-Xplore window): the fact key names the window's last month, but the
#: value covers the period.
_WINDOW_VALUE_OPERATIONS = frozenset(
    (
        "calendar_year_average",
        "calendar_year_window",
        MONTHLY_WINDOW_COUNT_X_MEAN,
        *MONTHLY_WINDOW_OPERATIONS,
    )
)
UK_HOLD_KINDS = (
    "matched_period",
    "in_year_snapshot",
    "control_rescaled",
    "reviewed_no_index",
)
#: The one status that puts a signed register entry in force. Any other value
#: (``proposed``, ``rejected``, ``withdrawn``, a typo) leaves it out of force.
UK_SIGNED_STATUS = "approved"


def uk_register_entry_is_signed(entry: Mapping[str, Any]) -> bool:
    """An approver, an approval date and the status ``approved``."""

    return (
        bool(str(entry.get("approved_by") or "").strip())
        and bool(str(entry.get("approved_on") or "").strip())
        and str(entry.get("status") or "").strip().lower() == UK_SIGNED_STATUS
    )


@functools.lru_cache(maxsize=1)
def load_uk_uprating_holds() -> Mapping[str, Any]:
    return validated_uk_uprating_holds(
        json.loads(
            importlib_resources.files("microcosm.build.uk")
            .joinpath(UK_UPRATING_HOLDS_RESOURCE)
            .read_text(encoding="utf-8")
        )
    )


def validated_uk_uprating_holds(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Refuse a register with an unknown kind or a missing required field."""

    if payload.get("schema_version") != 1:
        raise ValueError(
            "UK uprating holds schema_version must be 1, got "
            f"{payload.get('schema_version')!r}."
        )
    for entry in payload["holds"]:
        if entry.get("kind") not in UK_HOLD_KINDS:
            raise ValueError(
                f"UK uprating hold for {entry.get('target_ids')!r} has unknown kind "
                f"{entry.get('kind')!r}; expected one of {UK_HOLD_KINDS}."
            )
        for field in ("reason", "tracking", "expires_on"):
            if not str(entry.get(field) or "").strip():
                raise ValueError(
                    f"UK uprating hold for {entry.get('target_ids')!r} has no "
                    f"{field}: a hold names why it stands, where it is tracked "
                    "and when it lapses."
                )
        date.fromisoformat(str(entry["expires_on"]))
    return payload


def _contract_target_id(spec: TargetSpec) -> str:
    return str(spec.metadata.get("contract_target_id", spec.name.split("@", 1)[0]))


def _resolved_period(spec: TargetSpec) -> str:
    return str(
        spec.metadata.get("ledger_fact_period")
        or spec.metadata.get("uprating_from_period")
        or ""
    )


def _moved_or_covers_the_period(spec: TargetSpec) -> bool:
    """An index moved the value, or the value covers the period by construction.

    An applier that stamps ``uprating_hold`` beside its factor says it moved
    nothing (a census tenure share with no dwelling series to drift it): that
    row stays a hold. A window of the calibration year's subperiods (the
    compiled ``ledger_value_operation``), a calendar-year mean
    (``fact_aggregation: time_mean``) and a declared source window cover the
    period whatever month the fact key names. A ``rolled_forward_by_ratio``
    value is carried to the target period by a ratio whose numerator is pinned
    there, so the older base is not held.
    """

    metadata = spec.metadata
    if "uprating_factor" in metadata and "uprating_hold" not in metadata:
        return True
    operation = metadata.get("ledger_value_operation")
    return (
        operation in _WINDOW_VALUE_OPERATIONS
        or operation == ROLLED_FORWARD_BY_RATIO
        or metadata.get("fact_aggregation") == "time_mean"
        or metadata.get("period_match_policy") == "source_window"
    )


def has_unresolved_period(spec: TargetSpec, calibration_period: int | str) -> bool:
    """No index moved the value and its fact period cannot be read.

    Such a row can be neither shown current nor declared held, so the holds
    check refuses it rather than passing it as not held.
    """

    if _moved_or_covers_the_period(spec):
        return False
    resolved = _resolved_period(spec)
    return not (
        resolved
        and _period_key_from_value(resolved)[0]
        and _period_key_from_value(calibration_period)[0]
    )


def is_uprating_hold(spec: TargetSpec, calibration_period: int | str) -> bool:
    """The spec's fact is older than the period and no index moved it."""

    if _moved_or_covers_the_period(spec) or has_unresolved_period(
        spec, calibration_period
    ):
        return False
    source = _period_key_from_value(_resolved_period(spec))
    target = _period_key_from_value(calibration_period)
    return source[1] < target[1]


def uk_hold_is_signed(entry: Mapping[str, Any]) -> bool:
    """An approver, an approval date and the status ``approved``.

    A proposed, rejected or withdrawn hold is not the decision: it stands for
    nothing, so the compile refuses the rows it names.
    """

    return uk_register_entry_is_signed(entry)


def _declaration_index(
    holds: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for entry in holds:
        for target_id in entry["target_ids"]:
            if target_id in index:
                raise ValueError(
                    f"UK uprating hold for {target_id!r} is declared twice."
                )
            index[str(target_id)] = entry
    return index


def assert_uk_uprating_holds_declared(
    registry: TargetRegistry,
    *,
    calibration_period: int | str,
    evaluated_on: date,
    scope: str,
    holds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse an undeclared, unsigned or expired hold; return the holds by kind.

    ``scope`` (``national`` or ``local``) names the register in the receipt.
    A declaration that matches no held target is stale and refused too, so
    the register cannot outlive the holds it explains. A declaration nobody
    has signed (no approver, no approval date, or a status other than
    ``approved``) is not in force, so its rows are refused like undeclared
    ones. A register passed in is validated as the committed one is.
    """

    declared = _declaration_index(
        (
            load_uk_uprating_holds()
            if holds is None
            else validated_uk_uprating_holds(holds)
        )["holds"]
    )
    unresolved = sorted(
        spec.name
        for spec in registry.specs
        if has_unresolved_period(spec, calibration_period)
    )
    if unresolved:
        raise ValueError(
            f"UK {scope} uprating holds refused: {len(unresolved)} row(s) carry "
            f"no readable fact period, so they can be neither shown current nor "
            f"declared held: {unresolved[:10]} (microcosm#1123)."
        )
    held: dict[str, list[str]] = {}
    for spec in registry.specs:
        if is_uprating_hold(spec, calibration_period):
            held.setdefault(_contract_target_id(spec), []).append(spec.name)
    undeclared = sorted(target_id for target_id in held if target_id not in declared)
    unsigned = sorted(
        target_id
        for target_id in held
        if target_id in declared and not uk_hold_is_signed(declared[target_id])
    )
    expired = sorted(
        target_id
        for target_id in held
        if target_id in declared
        and date.fromisoformat(str(declared[target_id]["expires_on"])) < evaluated_on
    )
    scoped_declarations = {
        target_id
        for target_id, entry in declared.items()
        if entry.get("scope", scope) == scope
    }
    stale = sorted(scoped_declarations - set(held))
    if undeclared or unsigned or expired or stale:
        raise ValueError(
            f"UK {scope} uprating holds refused: undeclared {undeclared}; unsigned "
            f"{unsigned}; expired {expired}; stale declarations {stale}. Declare "
            "an uprating_index on the contract target or a hold in "
            "uk/uprating_holds.json; a hold is in force only once it carries "
            "approved_by and approved_on and its status is approved "
            "(microcosm#1123)."
        )
    by_kind: dict[str, list[str]] = {}
    for target_id in sorted(held):
        by_kind.setdefault(str(declared[target_id]["kind"]), []).append(target_id)
    return {
        "scope": scope,
        "calibration_period": str(calibration_period),
        "held_targets": len(held),
        "held_cells": sum(len(names) for names in held.values()),
        "by_kind": by_kind,
    }
