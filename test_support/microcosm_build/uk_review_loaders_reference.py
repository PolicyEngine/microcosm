# Verbatim copies, at 2ead043a4 (before microcosm#1759 step D6), of the UK
# loaders that now check their signed entries with
# microcosm.build.declared_registers: the calibration measure exclusions'
# loader and applier (uk_runtime.measure_simulation), and the support floor's
# payload reader and window check (uk_runtime.support_floor;
# UKSupportFloor.assert_in_force as a function). Tests hold the new code to
# them; do not edit.
# ruff: noqa

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from datetime import date
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.measure_simulation import (
    _UK_MEASURE_EXCLUSION_FIELDS,
)
from microcosm.build.uk_runtime.support_floor import (
    _FIELDS,
    _TEXT_FIELDS,
    UK_SUPPORT_FLOOR_METRIC,
    UK_SUPPORT_FLOOR_VALUE,
    UKSupportFloor,
    _scopes_from_payload,
)
from microcosm.build.uk_runtime.weighted_integrity import exclusion_evaluation_date
from microcosm.calibrate import TargetRegistry


def load_uk_calibration_measure_exclusions(
    path: Path | None = None,
) -> tuple[dict[str, str], ...]:
    """Load and validate the reviewed UK calibration-measure exclusions."""

    if path is None:
        text = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath("calibration_measure_exclusions.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    # Schema 4 carries the signed zero-support floor (microcosm#1095 U3) in
    # scopes of one unit each, which
    # ``support_floor.load_uk_support_floor_declaration`` reads; schema 3's
    # single-scope floor is retired with it.
    schema_version = payload.get("schema_version")
    allowed_top = {"schema_version", "exclusions"}
    if schema_version == 4:
        allowed_top.add("support_floor")
    unknown = sorted(set(payload) - allowed_top)
    if unknown:
        raise ValueError(f"unknown top-level exclusion key(s): {unknown}")
    if schema_version not in (2, 4):
        raise ValueError(
            "UK calibration measure exclusions schema_version must be 2 or 4."
        )
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("UK calibration measure exclusions must contain a list.")
    seen: set[str] = set()
    loaded: list[dict[str, str]] = []
    for entry in exclusions:
        if not isinstance(entry, dict):
            raise ValueError(
                "UK calibration measure exclusion entries must be objects."
            )
        unknown_fields = sorted(set(entry) - set(_UK_MEASURE_EXCLUSION_FIELDS))
        if unknown_fields:
            raise ValueError(
                "unknown UK calibration measure exclusion field(s) "
                f"{unknown_fields} on {entry.get('name')!r}."
            )
        record = {
            field: str(entry.get(field, "")) for field in _UK_MEASURE_EXCLUSION_FIELDS
        }
        name = record["name"]
        if name in seen:
            raise ValueError(f"duplicate UK calibration measure exclusion {name!r}.")
        for field in _UK_MEASURE_EXCLUSION_FIELDS:
            if not record[field].strip():
                raise ValueError(
                    f"UK calibration measure exclusion {name!r} has empty {field}; "
                    "a narrowed target surface names why, where it is being "
                    "resolved, who approved it, and for how long."
                )
        for field in ("approved_on", "expires_on"):
            try:
                parsed = date.fromisoformat(record[field])
            except ValueError:
                parsed = None
            if parsed is None or parsed.isoformat() != record[field]:
                raise ValueError(
                    f"UK calibration measure exclusion {name!r} {field} must be "
                    f"canonical ISO (YYYY-MM-DD), got {record[field]!r}."
                )
        if record["expires_on"] <= record["approved_on"]:
            raise ValueError(
                f"UK calibration measure exclusion {name!r} expires_on must be "
                "after approved_on."
            )
        seen.add(name)
        loaded.append(record)
    return tuple(loaded)


def apply_uk_calibration_measure_exclusions(
    registry: TargetRegistry,
    exclusions: tuple[dict[str, str], ...],
    *,
    now: date | None = None,
) -> tuple[TargetRegistry, dict[str, dict[str, str]]]:
    """Remove reviewed excluded references from a UK target registry.

    The receipt carries every field the register declares, tracking included:
    an exclusion narrows the calibrated target surface, so the run evidence
    must say where each narrowing is being resolved, not only why. The window
    is enforced at apply time — outside ``approved_on``..``expires_on`` the
    run fails with a correct-or-renew message rather than the narrowing
    lapsing silently or living forever.
    """

    evaluated_on = exclusion_evaluation_date(now)
    for entry in exclusions:
        approved = date.fromisoformat(entry["approved_on"])
        expires = date.fromisoformat(entry["expires_on"])
        if not approved <= evaluated_on <= expires:
            raise ValueError(
                f"UK calibration measure exclusion {entry['name']!r} is outside "
                f"its reviewed window ({entry['approved_on']}..{entry['expires_on']}, "
                f"evaluated {evaluated_on.isoformat()}): correct the underlying "
                f"gap ({entry['tracking']}) or renew the adjudication with a new "
                "approval and expiry."
            )
        if (expires - evaluated_on).days <= 7:
            warnings.warn(
                f"UK calibration measure exclusion {entry['name']!r} expires "
                f"{entry['expires_on']} — within one week of the evaluation "
                f"date {evaluated_on.isoformat()}; correct the underlying gap "
                f"({entry['tracking']}) or renew the adjudication before the "
                "run fails on expiry.",
                UserWarning,
                stacklevel=2,
            )
    declared = {entry["name"]: entry for entry in exclusions}
    matched = {spec.name for spec in registry.specs if spec.name in declared}
    stale = sorted(set(declared) - matched)
    if stale:
        raise ValueError(
            "UK calibration measure exclusion matched zero registry specs: "
            + ", ".join(stale)
        )
    kept = [spec for spec in registry.specs if spec.name not in declared]
    receipt = {
        name: {
            field: declared[name][field]
            for field in _UK_MEASURE_EXCLUSION_FIELDS
            if field != "name"
        }
        for name in sorted(matched)
    }
    for record in receipt.values():
        record["evaluated_on"] = evaluated_on.isoformat()
    return TargetRegistry(kept, country=registry.country), receipt


def uk_support_floor_from_payload(block: Mapping[str, Any]) -> UKSupportFloor:
    """Validate the ``support_floor`` block closed-world."""

    if not isinstance(block, Mapping):
        raise ValueError("UK support floor must be an object.")
    unknown = sorted(set(block) - set(_FIELDS))
    missing = sorted(set(_FIELDS) - set(block))
    if unknown or missing:
        raise ValueError(
            f"UK support floor fields are closed-world: unknown {unknown}, "
            f"missing {missing}."
        )
    if block["metric"] != UK_SUPPORT_FLOOR_METRIC:
        raise ValueError(
            f"UK support floor metric must be {UK_SUPPORT_FLOOR_METRIC!r}, got "
            f"{block['metric']!r}."
        )
    floor = block["floor"]
    if (
        not isinstance(floor, int)
        or isinstance(floor, bool)
        or floor != UK_SUPPORT_FLOOR_VALUE
    ):
        raise ValueError(
            f"UK support floor must be {UK_SUPPORT_FLOOR_VALUE}: a row that any source "
            "household carries stays bound however few they are (María's rule of "
            "2026-10-21)."
        )
    scopes = _scopes_from_payload(block["scopes"])
    for field in _TEXT_FIELDS:
        if not isinstance(block[field], str) or not block[field].strip():
            raise ValueError(f"UK support floor {field} is empty.")
    dates = {}
    for field in ("approved_on", "expires_on"):
        try:
            parsed = date.fromisoformat(str(block[field]))
        except ValueError:
            parsed = None
        if parsed is None or parsed.isoformat() != block[field]:
            raise ValueError(
                f"UK support floor {field} must be canonical ISO (YYYY-MM-DD), "
                f"got {block[field]!r}."
            )
        dates[field] = parsed
    if dates["expires_on"] <= dates["approved_on"]:
        raise ValueError("UK support floor expires_on must be after approved_on.")
    return UKSupportFloor(
        scopes=scopes,
        floor=floor,
        reason=block["reason"],
        tracking=block["tracking"],
        approved_by=block["approved_by"],
        adjudication=block["adjudication"],
        approved_on=dates["approved_on"],
        expires_on=dates["expires_on"],
    )


def assert_in_force(self, evaluated_on: date) -> None:
    if not self.approved_on <= evaluated_on <= self.expires_on:
        raise ValueError(
            "UK support floor is outside its reviewed window "
            f"({self.approved_on.isoformat()}..{self.expires_on.isoformat()}, "
            f"evaluated {evaluated_on.isoformat()}): renew the signature "
            f"({self.tracking})."
        )
