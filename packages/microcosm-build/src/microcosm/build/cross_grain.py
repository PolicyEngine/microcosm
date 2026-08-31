"""Country-agnostic reconciliation of targets bound at multiple grains.

The operator in this module changes target values, never loss weights.  A
country supplies the measurement signature, explicit bridges for known
partition relationships, grain precedence, and geography-to-leg mappings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Closure of a rescaled leg onto its control is the property this pass exists
# to establish, so it is asserted rather than assumed.  The tolerance matches
# the frame kernel's mass-conservation bound (``_MASS_CONSERVE_RTOL``).
_CLOSURE_RTOL = 1e-9
# A leg whose members cancel to near-nothing would yield an arbitrarily large
# factor, so the vanishing-total refusal is relative to the control rather than
# an exact-zero test.
_MIN_LEG_SUM_RTOL = 1e-9
# Absent, null, and empty signature values are one canonical "unspecified", so
# two spellings of the same measurement cannot land in different groups.
_UNSPECIFIED = ("<unspecified>",)


@dataclass(frozen=True)
class CrossGrainBridge:
    """Declare one higher-control partition as the identity of a lower side."""

    bridge_id: str
    concept: str
    higher_target_ids: tuple[str, ...]
    lower_side: str


@dataclass(frozen=True)
class CrossGrainRule:
    """All country-owned declarations needed by the shared operator."""

    grain_precedence: tuple[str, ...]
    signature_fields: tuple[str, ...]
    bridges: tuple[CrossGrainBridge, ...]
    leg_of_area: Callable[[str], str]
    parent_geography_legs: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class CrossGrainInconsistency:
    """Detected rows for one winning-grain/lower-grain reconciliation."""

    inconsistency_id: str
    bridge_id: str | None
    signature: tuple[tuple[str, Any], ...]
    winning_grain: str
    lower_grain: str
    winning_positions: tuple[int, ...]
    lower_positions: tuple[int, ...]
    higher_target_ids: tuple[str, ...]


def detect_cross_grain_inconsistencies(
    local_frame: pd.DataFrame,
    bound_higher_targets: Iterable[str],
    contract_signatures: Mapping[str, Mapping[str, Any]],
    rule: CrossGrainRule,
) -> tuple[CrossGrainInconsistency, ...]:
    """Detect exact-signature and explicitly bridged cross-grain groups.

    ``local_frame`` is a long target surface.  It must expose ``value`` and
    ``target_id`` plus either the canonical ``grain``/``geography_id`` columns
    or the rowwise aliases ``area_type``/``area_code``.  Contract sides may be
    written as either ``<target id>`` or ``contract:<target id>``; external
    sides use their declared ``external:...`` bridge name.
    """

    columns = _surface_columns(local_frame)
    _validate_rule(rule)
    bound = tuple(str(target_id) for target_id in bound_higher_targets)
    _require_unique_nonblank(bound, label="bound_higher_targets")
    unknown = sorted(set(bound) - set(contract_signatures))
    if unknown:
        raise ValueError(
            "cross-grain bound higher target(s) are absent from the contract: "
            f"{unknown}."
        )

    bridge_by_side = _bridge_by_side(rule)
    bound_set = set(bound)
    for bridge in rule.bridges:
        selected = bound_set & set(bridge.higher_target_ids)
        if selected and selected != set(bridge.higher_target_ids):
            missing = sorted(set(bridge.higher_target_ids) - selected)
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} is partially bound; "
                f"missing higher target(s) {missing}."
            )

    grouped: dict[
        tuple[str, str, tuple[tuple[str, Any], ...]],
        list[tuple[int, str, str, str, float]],
    ] = {}
    seen_bound: set[str] = set()
    values = local_frame[columns["value"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("cross-grain target values must all be finite.")

    for position, row in enumerate(local_frame.to_dict(orient="records")):
        grain = str(row[columns["grain"]])
        geography_id = str(row[columns["geography_id"]])
        side = str(row[columns["target_id"]])
        if grain not in rule.grain_precedence:
            raise ValueError(
                f"cross-grain surface row {position} uses unknown grain "
                f"{grain!r}; expected one of {rule.grain_precedence}."
            )
        if not geography_id:
            raise ValueError(
                f"cross-grain surface row {position} has a blank geography id."
            )
        bridge = bridge_by_side.get(side)
        contract_id = _contract_target_id(side)
        if bridge is None and contract_id is not None:
            bridge = bridge_by_side.get(contract_id)
            if bridge is None:
                bridge = bridge_by_side.get(f"contract:{contract_id}")

        is_bound_higher = contract_id in bound_set
        if grain == rule.grain_precedence[0] and not is_bound_higher:
            continue
        if is_bound_higher:
            seen_bound.add(contract_id)

        keys: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
        if bridge is not None:
            keys.append(
                (
                    "bridge",
                    bridge.bridge_id,
                    (("concept", _freeze(bridge.concept)),),
                )
            )
        # A declared bridge is additive to exact-signature detection.  This
        # matters when one lower side has both a bridged control and a second,
        # exact-signature partition (the UK UC case): both controls must be
        # seen so incompatible same-grain values fail closed.
        if contract_id is not None and contract_id in contract_signatures:
            keys.append(
                (
                    "signature",
                    "",
                    _measurement_signature(
                        contract_signatures[contract_id], rule.signature_fields
                    ),
                )
            )
        for key in dict.fromkeys(keys):
            grouped.setdefault(key, []).append(
                (position, grain, geography_id, side, float(values[position]))
            )

    missing_rows = sorted(bound_set - seen_bound)
    if missing_rows:
        raise ValueError(
            "cross-grain bound higher target(s) have no value rows in the "
            f"surface: {missing_rows}."
        )

    inconsistencies: list[CrossGrainInconsistency] = []
    precedence = {grain: index for index, grain in enumerate(rule.grain_precedence)}
    for (kind, bridge_id, signature), rows in grouped.items():
        grains = sorted({row[1] for row in rows}, key=precedence.__getitem__)
        if len(grains) < 2:
            continue
        winning_grain = grains[0]
        winning = tuple(row[0] for row in rows if row[1] == winning_grain)
        higher_ids = tuple(
            sorted(
                {
                    target_id
                    for position in winning
                    if (
                        target_id := _contract_target_id(
                            str(local_frame.iloc[position][columns["target_id"]])
                        )
                    )
                    is not None
                }
            )
        )
        identity = _inconsistency_identity(kind, bridge_id, signature)
        for lower_grain in grains[1:]:
            lower = tuple(row[0] for row in rows if row[1] == lower_grain)
            inconsistencies.append(
                CrossGrainInconsistency(
                    inconsistency_id=(
                        f"{identity}:{winning_grain}_over_{lower_grain}"
                    ),
                    bridge_id=bridge_id or None,
                    signature=signature,
                    winning_grain=winning_grain,
                    lower_grain=lower_grain,
                    winning_positions=winning,
                    lower_positions=lower,
                    higher_target_ids=higher_ids,
                )
            )
    return tuple(
        sorted(inconsistencies, key=lambda group: group.inconsistency_id)
    )


def reconcile_cross_grain_surface(
    local_frame: pd.DataFrame,
    groups: Iterable[CrossGrainInconsistency],
    rule: CrossGrainRule,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return a copied surface with every detected lower grain rescaled."""

    columns = _surface_columns(local_frame)
    _validate_rule(rule)
    materialized_groups = tuple(groups)
    _assert_compatible_overlapping_groups(
        local_frame, materialized_groups, columns, rule
    )
    reconciled = local_frame.copy(deep=True)
    receipts: list[dict[str, Any]] = []
    for group in materialized_groups:
        controls = _winning_controls(reconciled, group, columns, rule)
        lower_by_leg: dict[str, list[int]] = {}
        for position in group.lower_positions:
            area = str(reconciled.iloc[position][columns["geography_id"]])
            leg = str(rule.leg_of_area(area))
            if not leg:
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"maps area {area!r} to a blank leg."
                )
            lower_by_leg.setdefault(leg, []).append(position)

        assigned_controls: dict[str, dict[str, Any]] = {}
        for control in controls:
            for leg in control["covered_legs"]:
                existing = assigned_controls.get(leg)
                if existing is not None:
                    if (
                        existing["covered_legs"] != control["covered_legs"]
                        or existing["value"] != control["value"]
                    ):
                        raise ValueError(
                            "cross-grain inconsistency "
                            f"{group.inconsistency_id!r} has two different "
                            f"control values at grain {group.winning_grain!r} "
                            f"for leg {leg!r}."
                        )
                    continue
                assigned_controls[leg] = control

        unparented = sorted(set(lower_by_leg) - set(assigned_controls))
        if unparented:
            raise ValueError(
                f"cross-grain inconsistency {group.inconsistency_id!r} has "
                f"unparented lower-grain leg(s) {unparented}."
            )

        leg_receipts: list[dict[str, Any]] = []
        used_controls: set[tuple[str, tuple[str, ...]]] = set()
        for control in controls:
            control_key = (
                str(control["parent_geography_id"]),
                tuple(control["covered_legs"]),
            )
            if control_key in used_controls:
                continue
            used_controls.add(control_key)
            positions = [
                position
                for leg in control["covered_legs"]
                for position in lower_by_leg.get(leg, ())
            ]
            if not positions:
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"has an empty leg for parent geography "
                    f"{control['parent_geography_id']!r}."
                )
            raw_values = reconciled.iloc[positions][columns["value"]].to_numpy(
                dtype=np.float64
            )
            raw_sum = float(raw_values.sum())
            parent_value = float(control["value"])
            if not np.isfinite(raw_sum) or not np.isfinite(parent_value):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    "contains a non-finite leg total."
                )
            if (raw_values > 0.0).any() and (raw_values < 0.0).any():
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    "cannot reconcile a mixed-sign lower leg for parent "
                    f"geography {control['parent_geography_id']!r}; the "
                    "members would cancel and rescale by an arbitrary factor."
                )
            if abs(raw_sum) < _MIN_LEG_SUM_RTOL * abs(parent_value):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"cannot scale a vanishing lower-leg total {raw_sum!r} to "
                    f"control {parent_value!r}."
                )
            factor = 1.0 if raw_sum == 0.0 else parent_value / raw_sum
            if not np.isfinite(factor):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    "produced a non-finite reconciliation factor."
                )
            if factor < 0.0:
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    "cannot reconcile opposite-signed lower and control targets."
                )
            reconciled.iloc[
                positions, reconciled.columns.get_loc(columns["value"])
            ] = raw_values * factor
            new_total = float(
                reconciled.iloc[positions][columns["value"]]
                .to_numpy(dtype=np.float64)
                .sum()
            )
            if not np.isfinite(new_total):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    "produced a non-finite reconciled total."
                )
            if not np.isclose(
                new_total, parent_value, rtol=_CLOSURE_RTOL, atol=0.0
            ):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"left leg {'+'.join(control['covered_legs'])!r} off its "
                    f"control: reconciled total {new_total!r} against control "
                    f"{parent_value!r}."
                )
            areas = {
                str(reconciled.iloc[position][columns["geography_id"]])
                for position in positions
            }
            leg_receipts.append(
                {
                    "leg": "+".join(control["covered_legs"]),
                    "parent_geography_id": str(control["parent_geography_id"]),
                    "higher_target_ids": list(control["higher_target_ids"]),
                    "n_areas": len(areas),
                    "old_total": raw_sum,
                    "new_total": new_total,
                    "relative_shift": (
                        0.0 if raw_sum == 0.0 else (new_total - raw_sum) / raw_sum
                    ),
                    "declared_factor": factor,
                    "reason": (
                        "standing cross-grain rule: "
                        f"{group.winning_grain} controls {group.lower_grain}"
                    ),
                }
            )
        receipts.append(
            {
                "inconsistency_id": group.inconsistency_id,
                "bridge_id": group.bridge_id,
                "winning_grain": group.winning_grain,
                "legs": leg_receipts,
            }
        )
    return reconciled, receipts


def apply_cross_grain_reconciliation(
    local_frame: pd.DataFrame,
    bound_higher_targets: Iterable[str],
    contract_signatures: Mapping[str, Mapping[str, Any]],
    rule: CrossGrainRule,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect, reconcile, and return the always-present pass receipt."""

    bound = tuple(str(target_id) for target_id in bound_higher_targets)
    groups = detect_cross_grain_inconsistencies(
        local_frame,
        bound,
        contract_signatures,
        rule,
    )
    reconciled, group_receipts = reconcile_cross_grain_surface(
        local_frame, groups, rule
    )
    receipt = {
        "bound_higher_targets": list(bound),
        "inconsistencies_in_force": [
            group.inconsistency_id for group in groups
        ],
        "groups": group_receipts,
        "absence": (
            None
            if groups
            else "No cross-grain inconsistencies are in force on this surface."
        ),
    }
    return reconciled, receipt


def _surface_columns(frame: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "grain": ("grain", "area_type"),
        "geography_id": ("geography_id", "area_code"),
        "target_id": ("target_id",),
        "value": ("value",),
    }
    resolved: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        present = [candidate for candidate in candidates if candidate in frame.columns]
        if not present:
            raise ValueError(
                f"cross-grain surface must expose {canonical!r}; accepted "
                f"column name(s): {list(candidates)}."
            )
        resolved[canonical] = present[0]
    return resolved


def _validate_rule(rule: CrossGrainRule) -> None:
    _require_unique_nonblank(rule.grain_precedence, label="grain_precedence")
    _require_unique_nonblank(rule.signature_fields, label="signature_fields")
    if not rule.grain_precedence:
        raise ValueError("cross-grain grain_precedence must not be empty.")
    bridge_ids = tuple(bridge.bridge_id for bridge in rule.bridges)
    _require_unique_nonblank(bridge_ids, label="bridge ids")
    _bridge_by_side(rule)
    for bridge in rule.bridges:
        if not bridge.concept or not bridge.lower_side:
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} has a blank declaration."
            )
        _require_unique_nonblank(
            bridge.higher_target_ids,
            label=f"bridge {bridge.bridge_id!r} higher_target_ids",
        )
        if not bridge.higher_target_ids:
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} has no higher targets."
            )
    for geography_id, legs in rule.parent_geography_legs.items():
        if not str(geography_id) or not legs or any(not str(leg) for leg in legs):
            raise ValueError(
                "cross-grain parent_geography_legs must map nonblank geography "
                "ids to nonempty, nonblank leg tuples."
            )
        if len(set(legs)) != len(legs):
            raise ValueError(
                f"cross-grain parent geography {geography_id!r} repeats a leg."
            )


def _bridge_by_side(rule: CrossGrainRule) -> dict[str, CrossGrainBridge]:
    result: dict[str, CrossGrainBridge] = {}
    for bridge in rule.bridges:
        sides = (*bridge.higher_target_ids, bridge.lower_side)
        for side in sides:
            aliases = {side}
            if side.startswith("contract:"):
                aliases.add(side.removeprefix("contract:"))
            elif not side.startswith("external:"):
                aliases.add(f"contract:{side}")
            for alias in aliases:
                existing = result.get(alias)
                if existing is not None and existing.bridge_id != bridge.bridge_id:
                    raise ValueError(
                        f"cross-grain target side {side!r} is matched by two "
                        f"bridges: {existing.bridge_id!r} and {bridge.bridge_id!r}."
                    )
                result[alias] = bridge
    return result


def _measurement_signature(
    contract: Mapping[str, Any], fields: tuple[str, ...]
) -> tuple[tuple[str, Any], ...]:
    """Canonicalize one contract entry's measurement into a grouping key.

    A signature field that is absent, null, or an empty container collapses to
    one canonical "unspecified" value: two spellings of the same measurement
    must never land in different groups, because a silent split lets the joint
    solve reconcile the pair implicitly.  Sequence-valued fields are compared
    as unordered collections, since a filter list is a conjunction.
    """

    measurement = contract.get("measurement")
    if measurement is None:
        raise ValueError(
            "cross-grain contract entry must carry a 'measurement' mapping; "
            "a top-level fallback would let two contract spellings of the same "
            f"measurement split silently. Got keys {sorted(contract)}."
        )
    if not isinstance(measurement, Mapping):
        raise ValueError("cross-grain contract measurement must be a mapping.")
    return tuple(
        (field, _canonical_signature_value(measurement.get(field)))
        for field in fields
    )


def _canonical_signature_value(value: Any) -> Any:
    frozen = _freeze(value)
    if frozen is None or frozen == () or frozen == "":
        return _UNSPECIFIED
    return frozen


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(member)) for key, member in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        members = [_freeze(member) for member in value]
        # Order-insensitive: a conjunction of filter conditions is the same
        # measurement however the contract happens to order it.
        return tuple(
            sorted(
                members,
                key=lambda member: json.dumps(
                    _json_safe(member), sort_keys=True, separators=(",", ":")
                ),
            )
        )
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 0 and 0.0 are the same signature value; keep one numeric spelling.
        return float(value)
    return value


def _contract_target_id(side: str) -> str | None:
    if side.startswith("external:"):
        return None
    return side.removeprefix("contract:")


def _inconsistency_identity(
    kind: str,
    bridge_id: str,
    signature: tuple[tuple[str, Any], ...],
) -> str:
    if kind == "bridge":
        return bridge_id
    payload = json.dumps(_json_safe(signature), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    concept = next((str(value) for field, value in signature if field == "concept"), "target")
    return f"exact:{concept}:{digest}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(member) for key, member in value.items()}
    return value


def _winning_controls(
    frame: pd.DataFrame,
    group: CrossGrainInconsistency,
    columns: Mapping[str, str],
    rule: CrossGrainRule,
) -> list[dict[str, Any]]:
    rows = [
        {
            "position": position,
            "geography_id": str(frame.iloc[position][columns["geography_id"]]),
            "target_id": str(frame.iloc[position][columns["target_id"]]),
            "value": float(frame.iloc[position][columns["value"]]),
        }
        for position in group.winning_positions
    ]
    if group.winning_grain == rule.grain_precedence[0]:
        by_geography: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_geography.setdefault(row["geography_id"], []).append(row)
        controls = []
        for geography_id, members in sorted(by_geography.items()):
            deduplicated: dict[str, float] = {}
            for member in members:
                target_id = _contract_target_id(member["target_id"])
                if target_id is None:
                    raise ValueError(
                        f"cross-grain winning row {member['target_id']!r} is not "
                        "a contract target."
                    )
                existing = deduplicated.get(target_id)
                if existing is not None and existing != member["value"]:
                    raise ValueError(
                        f"cross-grain target {target_id!r} has two different "
                        f"control values at {geography_id!r}."
                    )
                deduplicated[target_id] = member["value"]
            if group.bridge_id is not None:
                bridge = next(
                    bridge
                    for bridge in rule.bridges
                    if bridge.bridge_id == group.bridge_id
                )
                missing = sorted(
                    set(bridge.higher_target_ids) - set(deduplicated)
                )
                if missing:
                    raise ValueError(
                        f"cross-grain bridge {bridge.bridge_id!r} is partially "
                        f"valued at {geography_id!r}; missing {missing}."
                    )
            covered_legs = rule.parent_geography_legs.get(geography_id)
            if covered_legs is None:
                raise ValueError(
                    f"cross-grain parent geography {geography_id!r} declares no legs."
                )
            controls.append(
                {
                    "parent_geography_id": geography_id,
                    "covered_legs": tuple(covered_legs),
                    "higher_target_ids": tuple(sorted(deduplicated)),
                    "value": float(sum(deduplicated.values())),
                }
            )
        return controls

    by_leg: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        leg = str(rule.leg_of_area(row["geography_id"]))
        if not leg:
            raise ValueError(
                f"cross-grain winning area {row['geography_id']!r} maps to a "
                "blank leg."
            )
        by_leg.setdefault(leg, []).append(row)
    return [
        {
            "parent_geography_id": f"area-derived:{leg}",
            "covered_legs": (leg,),
            "higher_target_ids": tuple(
                sorted(
                    {
                        target_id
                        for member in members
                        if (
                            target_id := _contract_target_id(member["target_id"])
                        )
                        is not None
                    }
                )
            ),
            "value": float(sum(member["value"] for member in members)),
        }
        for leg, members in sorted(by_leg.items())
    ]


def _assert_compatible_overlapping_groups(
    frame: pd.DataFrame,
    groups: tuple[CrossGrainInconsistency, ...],
    columns: Mapping[str, str],
    rule: CrossGrainRule,
) -> None:
    """Refuse competing same-grain controls over any shared lower row."""

    for index, left in enumerate(groups):
        left_positions = set(left.lower_positions)
        left_controls = _winning_controls(frame, left, columns, rule)
        for right in groups[index + 1 :]:
            if left.winning_grain != right.winning_grain:
                continue
            if not (left_positions & set(right.lower_positions)):
                continue
            right_controls = _winning_controls(frame, right, columns, rule)
            left_by_coverage = {
                tuple(control["covered_legs"]): float(control["value"])
                for control in left_controls
            }
            right_by_coverage = {
                tuple(control["covered_legs"]): float(control["value"])
                for control in right_controls
            }
            for coverage in set(left_by_coverage) & set(right_by_coverage):
                if left_by_coverage[coverage] != right_by_coverage[coverage]:
                    raise ValueError(
                        "cross-grain groups "
                        f"{left.inconsistency_id!r} and "
                        f"{right.inconsistency_id!r} have two different "
                        f"control values at grain {left.winning_grain!r} "
                        f"for legs {coverage}."
                    )


def _require_unique_nonblank(values: Iterable[str], *, label: str) -> None:
    materialized = tuple(str(value) for value in values)
    blanks = [value for value in materialized if not value]
    if blanks:
        raise ValueError(f"cross-grain {label} must not contain blank values.")
    duplicates = sorted(
        {value for value in materialized if materialized.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            f"cross-grain {label} must be unique; duplicate(s) {duplicates}."
        )
