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
# Two controls over the same legs agree when they differ by summation order
# only: a control summed from member rows (a partition parent) and the same
# total read from one row can differ in the last bits.
_CONTROL_AGREEMENT_RTOL = 1e-12
# Absent, null, and empty signature values are one canonical "unspecified", so
# two spellings of the same measurement cannot land in different groups.
_UNSPECIFIED = ("<unspecified>",)


@dataclass(frozen=True)
class CrossGrainBridge:
    """Declare one higher-control partition as the identity of a lower side.

    By default the higher targets partition one control at each geography
    (the UK household composition), so every one must be valued there.
    ``per_geography`` declares instead that each higher target is the control
    of the geography it sits at (each nation's household estimate over its
    own legs, microcosm#1123): a geography then carries some of them only.
    ``max_factor_shift``, when set, bounds each lower grain's total under a
    control: the control over that grain's rows in all the legs it covers, as
    given, before any group rescales them (under the GB Universal Credit
    caseload, each grain's GB total). A factor further than that from one is
    refused: the two sides would no longer measure one quantity. A grain
    another group rescales first is still bounded, but the legs inside the
    total are not bounded one by one: a group with no bound (an exact
    signature group putting one leg's authorities on its constituencies)
    rescales them by whatever factor it needs. The bound must lie in (0, 1):
    at one or more it would accept every factor below one.

    Three more fields shape the bound, and each default is the bound above.
    ``min_factor_shift`` is the lowest signed shift a factor may take, where
    a gap is known to run one way; without it the bound is symmetric.
    ``shift_around`` names what the shift is measured from: ``"one"``, or
    ``"median"`` for the median factor across the legs of the bridge, which
    takes out a gap that every leg shares (two sources of two vintages).
    ``out_of_bound`` says what an out-of-bound leg does: ``"refuse"`` stops
    the pass; ``"unbind"`` leaves the leg's lower rows as they are and
    records them, so that the caller takes them off its surface.
    """

    bridge_id: str
    concept: str
    higher_target_ids: tuple[str, ...]
    lower_side: str
    per_geography: bool = False
    max_factor_shift: float | None = None
    min_factor_shift: float | None = None
    shift_around: str = "one"
    out_of_bound: str = "refuse"


SHIFT_REFERENCES = ("one", "median")
OUT_OF_BOUND_ACTIONS = ("refuse", "unbind")

#: The metadata that a reconciled target row keeps. Target metadata is text,
#: so the numbers are exact ``repr`` strings.
VALUE_BEFORE_KEY = "cross_grain_value_before"
FACTOR_KEY = "cross_grain_factor"
#: What reconciled the row: a bridge id, a partition id, or
#: :data:`EXACT_SIGNATURE_CONTROL`.
CONTROL_KEY = "cross_grain_control"
EXACT_SIGNATURE_CONTROL = "exact_signature"
#: The name of the control row, when the control is one row.
PARENT_KEY = "cross_grain_parent"
#: The geographies a row stands for, when it stands for more than its own.
MEMBERS_KEY = "cross_grain_members"


def cross_grain_metadata(
    *,
    value_before: float,
    factor: float,
    control: str,
    parent: str | None = None,
    members: Iterable[str] = (),
) -> dict[str, str]:
    """The metadata of one reconciled row, in a fixed key order."""

    metadata = {
        VALUE_BEFORE_KEY: repr(float(value_before)),
        FACTOR_KEY: repr(float(factor)),
        CONTROL_KEY: str(control),
    }
    if parent is not None:
        metadata[PARENT_KEY] = str(parent)
    members = tuple(str(member) for member in members)
    if members:
        metadata[MEMBERS_KEY] = ",".join(members)
    return metadata


@dataclass(frozen=True)
class CrossGrainBridgeSelector:
    """One declared entry that makes one bridge for each pair of targets.

    Two sources that publish the same measures at two levels need one bridge
    for each measure. A selector states the pairing once: ``higher`` and
    ``lower`` choose the targets of each side by their attributes (a value,
    or ``{"prefix": ...}``), and ``pair_on`` names the attributes on which a
    higher and a lower target must agree. ``paired_values`` maps a higher
    target's value of an attribute to the lower side's value, where the two
    sources name one thing differently. The bound fields are those of
    :class:`CrossGrainBridge`, and every bridge of the selector takes them.
    """

    bridge_id: str
    concept: str
    higher: Mapping[str, Any]
    lower: Mapping[str, Any]
    pair_on: tuple[str, ...]
    paired_values: Mapping[str, Mapping[str, str]] | None = None
    max_factor_shift: float | None = None
    min_factor_shift: float | None = None
    shift_around: str = "one"
    out_of_bound: str = "refuse"


def expand_bridge_selectors(
    selectors: Iterable[CrossGrainBridgeSelector],
    targets: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[CrossGrainBridge, ...], dict[str, tuple[str, ...]]]:
    """The bridges that the selectors make over ``targets``.

    ``targets`` maps a target id to its attributes. Returns the bridges, each
    named ``<selector id>:<lower target id>``, and for each selector the lower
    targets that no higher target pairs with: the caller refuses them or
    declares why they have no control. Two higher targets that pair with one
    lower target are refused, since a lower target has one control.
    """

    bridges: list[CrossGrainBridge] = []
    unpaired: dict[str, tuple[str, ...]] = {}
    for selector in selectors:
        if not selector.pair_on:
            raise ValueError(
                f"cross-grain bridge selector {selector.bridge_id!r} must name "
                "the attributes its two sides are paired on."
            )
        mapped = selector.paired_values or {}
        higher_by_key: dict[tuple[Any, ...], str] = {}
        for target_id, attributes in targets.items():
            if not _selects(selector.higher, attributes):
                continue
            key = tuple(
                mapped.get(name, {}).get(attributes.get(name), attributes.get(name))
                for name in selector.pair_on
            )
            if key in higher_by_key:
                raise ValueError(
                    f"cross-grain bridge selector {selector.bridge_id!r} pairs "
                    f"two higher targets on {key!r}: {higher_by_key[key]!r} and "
                    f"{target_id!r}."
                )
            higher_by_key[key] = target_id
        without_control: list[str] = []
        for target_id, attributes in targets.items():
            if not _selects(selector.lower, attributes):
                continue
            key = tuple(attributes.get(name) for name in selector.pair_on)
            higher = higher_by_key.get(key)
            if higher is None:
                without_control.append(target_id)
                continue
            bridges.append(
                CrossGrainBridge(
                    bridge_id=f"{selector.bridge_id}:{target_id}",
                    concept=selector.concept,
                    higher_target_ids=(higher,),
                    lower_side=target_id,
                    max_factor_shift=selector.max_factor_shift,
                    min_factor_shift=selector.min_factor_shift,
                    shift_around=selector.shift_around,
                    out_of_bound=selector.out_of_bound,
                )
            )
        unpaired[selector.bridge_id] = tuple(without_control)
    return tuple(bridges), unpaired


def _selects(side: Mapping[str, Any], attributes: Mapping[str, Any]) -> bool:
    """Whether a target's attributes meet every test of a selector side."""

    for name, wanted in side.items():
        value = attributes.get(name)
        if isinstance(wanted, Mapping):
            if not str(value).startswith(str(wanted["prefix"])):
                return False
        elif value != wanted:
            return False
    return True


@dataclass(frozen=True)
class CrossGrainRule:
    """All country-owned declarations needed by the shared operator."""

    grain_precedence: tuple[str, ...]
    signature_fields: tuple[str, ...]
    bridges: tuple[CrossGrainBridge, ...]
    leg_of_area: Callable[[str], str]
    parent_geography_legs: Mapping[str, tuple[str, ...]]
    #: Grains whose rows parent the grains below them. Empty means the top
    #: grain alone (the standing single-winner rule). Declaring a middle tier
    #: (a region between country and constituency) lets each lower leg take
    #: its nearest covering control, so a region row controls its own
    #: constituencies and authorities while a country row controls the legs no
    #: region row covers.
    control_grains: tuple[str, ...] = ()
    #: How one control's value is divided among its lower rows. ``None`` is
    #: the standing rule: each row times the control over the rows' numpy
    #: sum. An allocator takes the rows' values, in surface order, and the
    #: control's value, and returns ``(values, factor, raw_sum)``: the
    #: reconciled rows in the same order, and the factor and the lower-row
    #: total that the leg receipt records. A country whose sources need
    #: another summation or a residual rule declares one (microcosm#1760);
    #: the refusals and the closure check below apply to it unchanged.
    allocator: Callable[[np.ndarray, float], tuple[np.ndarray, float, float]] | None = (
        None
    )


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
    *,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
) -> tuple[CrossGrainInconsistency, ...]:
    """Detect exact-signature and explicitly bridged cross-grain groups.

    ``local_frame`` is a long target surface.  It must expose ``value`` and
    ``target_id`` plus either the canonical ``grain``/``geography_id`` columns
    or the rowwise aliases ``area_type``/``area_code``. Contract sides may be
    written as either ``<target id>`` or ``contract:<target id>``. External
    target sides are forbidden: every reconciled row belongs to the contract.

    A partially bound declared bridge is ignored only when every missing
    higher member has a record in ``reviewed_unbound_higher_targets``.
    """

    columns = _surface_columns(local_frame)
    _validate_rule(rule)
    _validate_licensed_empty_legs(licensed_empty_legs)
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
    unbound_bridge_ids = {
        record["bridge_id"]
        for record in _reviewed_unbound_bridges(
            bound_set,
            rule,
            reviewed_unbound_higher_targets,
        )
    }

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
        if bridge is not None and bridge.bridge_id in unbound_bridge_ids:
            bridge = None

        is_bound_higher = contract_id in bound_set
        if grain in _control_grains(rule) and not is_bound_higher:
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
    control_grains = _control_grains(rule)
    for (kind, bridge_id, signature), rows in grouped.items():
        grains = sorted({row[1] for row in rows}, key=precedence.__getitem__)
        if len(grains) < 2:
            continue
        identity = _inconsistency_identity(kind, bridge_id, signature)
        controls_present = [grain for grain in grains if grain in control_grains]
        # A group with no control-tier row keeps the standing single-winner
        # rule: its top grain parents the rest. With control rows present,
        # every control grain parents each grain below it, and reconciliation
        # assigns each lower leg to its nearest covering control.
        pairs = (
            [(grains[0], lower) for lower in grains[1:]]
            if not controls_present
            else [
                (higher, lower)
                for lower in grains
                for higher in controls_present
                if precedence[higher] < precedence[lower]
            ]
        )
        for winning_grain, lower_grain in pairs:
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
            lower = tuple(row[0] for row in rows if row[1] == lower_grain)
            inconsistencies.append(
                CrossGrainInconsistency(
                    inconsistency_id=(f"{identity}:{winning_grain}_over_{lower_grain}"),
                    bridge_id=bridge_id or None,
                    signature=signature,
                    winning_grain=winning_grain,
                    lower_grain=lower_grain,
                    winning_positions=winning,
                    lower_positions=lower,
                    higher_target_ids=higher_ids,
                )
            )
    # Lower grains reconcile top-down (a region row is rescaled to its country
    # before it controls its authorities), and within one lower grain the
    # nearest control grain claims first.
    return tuple(
        sorted(
            inconsistencies,
            key=lambda group: (
                precedence[group.lower_grain],
                -precedence[group.winning_grain],
                group.inconsistency_id,
            ),
        )
    )


def _reconcile_cross_grain_surface_with_control_receipts(
    local_frame: pd.DataFrame,
    groups: Iterable[CrossGrainInconsistency],
    rule: CrossGrainRule,
    *,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
    check_factor_bounds: bool = True,
    frame_only_positions: frozenset[int] = frozenset(),
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return a copied surface with every detected lower grain rescaled.

    ``frame_only_positions`` are rows that join the frame to take a share of
    a control and bind nothing (a small-cell rule's deferred cells): a leg
    holding only those, for targets licensed empty on it, needs no control.
    """

    columns = _surface_columns(local_frame)
    _validate_rule(rule)
    _validate_licensed_empty_legs(licensed_empty_legs)
    materialized_groups = tuple(groups)
    _assert_compatible_overlapping_groups(
        local_frame, materialized_groups, columns, rule
    )
    reconciled = local_frame.copy(deep=True)
    receipts: list[dict[str, Any]] = []
    empty_legs_licensed: list[dict[str, Any]] = []
    controls_without_lower_rows: list[dict[str, Any]] = []
    delegated_legs: list[dict[str, Any]] = []
    control_grains = _control_grains(rule)
    bridges_by_id = {bridge.bridge_id: bridge for bridge in rule.bridges}
    # Rows already rescaled by a nearer control, per group identity and lower
    # grain, so a farther control never rescales them again; and the rows no
    # control has covered yet, checked once every pair has had its turn.
    claimed: dict[tuple[str, str], set[int]] = {}
    unparented_rows: dict[tuple[str, str], dict[int, str]] = {}
    first_pair_id: dict[tuple[str, str], str] = {}
    absent_middle_tier_legs: list[dict[str, Any]] = []
    deferred_legs_without_control: list[dict[str, Any]] = []
    unbound_legs: list[dict[str, Any]] = []
    pairs_per_key: dict[tuple[str, str], int] = {}
    for group in materialized_groups:
        key = (group.inconsistency_id.rsplit(":", 1)[0], group.lower_grain)
        pairs_per_key[key] = pairs_per_key.get(key, 0) + 1
    pairs_seen: dict[tuple[str, str], int] = {}
    # Per group identity, the targets on its leaf grains: a leg a middle tier
    # leaves empty is genuinely empty only when every leaf target is licensed
    # empty there too.
    leaf_targets: dict[str, set[str]] = {}
    for group in materialized_groups:
        identity = group.inconsistency_id.rsplit(":", 1)[0]
        bucket = leaf_targets.setdefault(identity, set())
        if group.lower_grain not in control_grains:
            bucket.update(
                str(reconciled.iloc[position][columns["target_id"]])
                for position in group.lower_positions
            )
    for group in materialized_groups:
        controls = _winning_controls(reconciled, group, columns, rule)
        claim_key = (group.inconsistency_id.rsplit(":", 1)[0], group.lower_grain)
        first_pair_id.setdefault(claim_key, group.inconsistency_id)
        pairs_seen[claim_key] = pairs_seen.get(claim_key, 0) + 1
        last_pair_for_key = pairs_seen[claim_key] == pairs_per_key[claim_key]
        already = claimed.setdefault(claim_key, set())
        lower_by_leg_all: dict[str, list[int]] = {}
        lower_by_leg: dict[str, list[int]] = {}
        row_legs: dict[int, tuple[str, ...]] = {}
        for position in group.lower_positions:
            area = str(reconciled.iloc[position][columns["geography_id"]])
            legs = _row_legs(area, rule)
            if not legs:
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"maps area {area!r} to a blank leg."
                )
            row_legs[position] = legs
            for leg in legs:
                lower_by_leg_all.setdefault(leg, []).append(position)
                if position not in already:
                    lower_by_leg.setdefault(leg, []).append(position)

        lower_target_ids = tuple(
            sorted(
                {
                    str(reconciled.iloc[position][columns["target_id"]])
                    for position in group.lower_positions
                }
            )
        )
        assigned_controls: dict[str, dict[str, Any]] = {}
        for control in controls:
            for leg in control["covered_legs"]:
                existing = assigned_controls.get(leg)
                if existing is not None:
                    if existing["covered_legs"] != control[
                        "covered_legs"
                    ] or not _controls_agree(existing["value"], control["value"]):
                        raise ValueError(
                            "cross-grain inconsistency "
                            f"{group.inconsistency_id!r} has two different "
                            f"control values at grain {group.winning_grain!r} "
                            f"for leg {leg!r}."
                        )
                    continue
                assigned_controls[leg] = control

        pending = unparented_rows.setdefault(claim_key, {})
        for leg in set(lower_by_leg) - set(assigned_controls):
            for position in lower_by_leg[leg]:
                pending[position] = leg
        if last_pair_for_key and group.lower_grain not in control_grains:
            # No farther control can still claim these leaf rows: refuse now,
            # ahead of the empty-leg licence check, as the standing rule does.
            # A leg only licensed-empty targets hold carries deferred cells
            # alone and needs no control (_wholly_deferred_legs).
            open_rows = {
                position: leg
                for position, leg in pending.items()
                if position not in already and leg not in assigned_controls
            }
            unparented = sorted(
                set(open_rows.values())
                - set(
                    _wholly_deferred_legs(
                        open_rows,
                        reconciled,
                        columns,
                        licensed_empty_legs,
                        frame_only_positions,
                    )
                )
            )
            if unparented:
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} has "
                    f"unparented lower-grain leg(s) {unparented}."
                )

        populated_controls: list[dict[str, Any]] = []
        for control in controls:
            positions = _control_positions(
                control, lower_by_leg, row_legs, group.inconsistency_id
            )
            delegated = [
                leg
                for leg in control["covered_legs"]
                if leg not in lower_by_leg and leg in lower_by_leg_all
            ]
            empty = [
                leg for leg in control["covered_legs"] if leg not in lower_by_leg_all
            ]
            if positions and delegated:
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"control {control['parent_geography_id']!r} covers leg(s) "
                    f"{delegated} that a nearer control already reconciled "
                    "alongside legs it must still reconcile; a control cannot "
                    "be split across tiers."
                )
            if positions and empty and group.lower_grain in control_grains:
                # A middle tier present on some of the control's legs and on no
                # row at all on the others would take the whole control onto
                # the legs it covers. Only legs licensed empty for every leaf
                # target (no data at any grain) may be left out.
                identity = group.inconsistency_id.rsplit(":", 1)[0]
                required = sorted(leaf_targets.get(identity) or lower_target_ids)
                unlicensed_legs = sorted(
                    leg
                    for leg in empty
                    if any(
                        leg
                        not in _licensed_legs_for_target(target_id, licensed_empty_legs)
                        for target_id in required
                    )
                )
                if unlicensed_legs:
                    raise ValueError(
                        f"cross-grain inconsistency {group.inconsistency_id!r} "
                        f"control {control['parent_geography_id']!r} has rows at "
                        f"grain {group.lower_grain!r} on some legs and no row at "
                        f"that grain on {unlicensed_legs}, which are not licensed "
                        "empty; a partial tier would take the whole control onto "
                        "the legs it covers."
                    )
            if positions:
                populated_controls.append(control)
                continue
            if delegated:
                delegated_legs.append(
                    {
                        "inconsistency_id": group.inconsistency_id,
                        "parent_geography_id": str(control["parent_geography_id"]),
                        "legs": delegated,
                        "reason": (
                            "every lower row on these legs was reconciled by a "
                            "nearer control grain"
                        ),
                    }
                )
            if group.lower_grain in control_grains:
                # A middle tier need not exist under every control: a country
                # row with no region rows on its legs parents the leaf rows
                # there directly. Not a licence matter.
                if empty:
                    absent_middle_tier_legs.append(
                        {
                            "inconsistency_id": group.inconsistency_id,
                            "parent_geography_id": str(control["parent_geography_id"]),
                            "legs": empty,
                        }
                    )
                continue
            if not empty:
                continue
            for leg in empty:
                unlicensed = [
                    target_id
                    for target_id in lower_target_ids
                    if leg
                    not in _licensed_legs_for_target(
                        target_id,
                        licensed_empty_legs,
                    )
                ]
                if unlicensed:
                    raise ValueError(
                        f"cross-grain inconsistency {group.inconsistency_id!r} "
                        f"has an empty leg {leg!r} for parent geography "
                        f"{control['parent_geography_id']!r}; the leg lacks a "
                        f"licence for lower target(s) {unlicensed}."
                    )
                empty_legs_licensed.append(
                    {
                        "inconsistency_id": group.inconsistency_id,
                        "parent_geography_id": str(control["parent_geography_id"]),
                        "leg": leg,
                        "lower_target_ids": list(lower_target_ids),
                    }
                )
            controls_without_lower_rows.append(
                {
                    "inconsistency_id": group.inconsistency_id,
                    "parent_geography_id": str(control["parent_geography_id"]),
                    "covered_legs": list(control["covered_legs"]),
                    "higher_target_ids": list(control["higher_target_ids"]),
                    "lower_target_ids": list(lower_target_ids),
                }
            )
        controls = populated_controls

        bridge = bridges_by_id.get(group.bridge_id or "")
        bound = None if bridge is None else bridge.max_factor_shift
        general_bound = bound is not None and (
            bridge.shift_around != "one"
            or bridge.min_factor_shift is not None
            or bridge.out_of_bound != "refuse"
        )
        reference = 1.0
        if bound is not None and bridge.shift_around == "median":
            reference = _median_input_factor(
                local_frame,
                controls,
                lower_by_leg,
                row_legs,
                columns,
                rule,
                group.inconsistency_id,
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
            positions = _control_positions(
                control, lower_by_leg, row_legs, group.inconsistency_id
            )
            raw_values = reconciled.iloc[positions][columns["value"]].to_numpy(
                dtype=np.float64
            )
            parent_value = float(control["value"])
            if rule.allocator is None:
                allocation = None
                raw_sum = float(raw_values.sum())
            else:
                allocation = _allocation(
                    rule.allocator, raw_values, parent_value, group.inconsistency_id
                )
                raw_sum = allocation[2]
            if not np.isfinite(raw_sum) or not np.isfinite(parent_value):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    "contains a non-finite leg total."
                )
            # Offsetting members are legitimate for a net-valued concept, so
            # only the conditioning matters: a leg that cancels to nothing
            # against a nonzero control would rescale by an arbitrary factor.
            if abs(raw_sum) < _MIN_LEG_SUM_RTOL * abs(parent_value):
                raise ValueError(
                    f"cross-grain inconsistency {group.inconsistency_id!r} "
                    f"cannot scale a vanishing lower-leg total {raw_sum!r} to "
                    f"control {parent_value!r}."
                )
            # Live only when the control is itself zero (the floor above is
            # vacuous there): a zero-summing leg against a zero control is a
            # no-op, where `parent_value / raw_sum` would be 0/0.
            if allocation is None:
                factor = 1.0 if raw_sum == 0.0 else parent_value / raw_sum
            else:
                factor = allocation[1]
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
            input_factor = None
            if general_bound:
                input_factor, input_sum = _input_factor(
                    local_frame.iloc[positions][columns["value"]].to_numpy(
                        dtype=np.float64
                    ),
                    parent_value,
                    rule,
                    group.inconsistency_id,
                )
                # A leg that sums to zero under a zero control has nothing to
                # judge. Otherwise the shift is the factor over the reference.
                relative = (
                    input_factor / reference
                    if np.isfinite(input_factor) and reference
                    else float("nan")
                )
                lowest = (
                    -bound
                    if bridge.min_factor_shift is None
                    else bridge.min_factor_shift
                )
                in_bound = input_sum == 0.0 or bool(
                    np.isfinite(relative) and 1.0 + lowest <= relative <= 1.0 + bound
                )
                if check_factor_bounds and not in_bound:
                    if bridge.out_of_bound == "unbind":
                        already.update(positions)
                        unbound_legs.append(
                            {
                                "inconsistency_id": group.inconsistency_id,
                                "bridge_id": bridge.bridge_id,
                                "leg": "+".join(control["covered_legs"]),
                                "parent_geography_id": str(
                                    control["parent_geography_id"]
                                ),
                                "higher_target_ids": list(control["higher_target_ids"]),
                                "positions": [int(p) for p in positions],
                                "factor_on_input_totals": input_factor,
                                "factor_reference": reference,
                                "relative_factor": relative,
                                "max_factor_shift": bound,
                                "min_factor_shift": lowest,
                                "shift_around": bridge.shift_around,
                            }
                        )
                        continue
                    raise ValueError(
                        f"cross-grain bridge {bridge.bridge_id!r} would scale the "
                        f"{group.lower_grain} rows of leg "
                        f"{'+'.join(control['covered_legs'])!r} by {input_factor!r} "
                        f"on their input totals, {relative!r} of the "
                        f"{bridge.shift_around} reference {reference!r}, outside "
                        f"its declared shift from {lowest} to {bound}: the two "
                        "sides no longer measure one quantity."
                    )
            elif bound is not None:
                # The bound reads the lower rows as given, not as an earlier
                # group left them: an exact-signature group that rescales this
                # grain onto a nearer tier first would leave a factor of one
                # here whatever the input (microcosm#1302 review).
                input_sum = float(
                    local_frame.iloc[positions][columns["value"]]
                    .to_numpy(dtype=np.float64)
                    .sum()
                )
                if input_sum != 0.0:
                    input_factor = parent_value / input_sum
                else:
                    input_factor = 1.0 if parent_value == 0.0 else float("inf")
                if check_factor_bounds and not abs(input_factor - 1.0) <= bound:
                    raise ValueError(
                        f"cross-grain bridge {bridge.bridge_id!r} would scale the "
                        f"{group.lower_grain} rows of leg "
                        f"{'+'.join(control['covered_legs'])!r} by {input_factor!r} "
                        f"on their input totals, further than its declared "
                        f"max_factor_shift {bound} from one: the two sides no "
                        "longer measure one quantity."
                    )
            reconciled.iloc[positions, reconciled.columns.get_loc(columns["value"])] = (
                raw_values * factor if allocation is None else allocation[0]
            )
            already.update(positions)
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
            if not np.isclose(new_total, parent_value, rtol=_CLOSURE_RTOL, atol=0.0):
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
                    **(
                        {}
                        if rule.allocator is None
                        else {"allocator": _allocator_name(rule.allocator)}
                    ),
                    **(
                        {}
                        if bound is None
                        else {
                            "max_factor_shift": bound,
                            "factor_on_input_totals": input_factor,
                            "factor_bound_checked": check_factor_bounds,
                        }
                    ),
                    **(
                        {}
                        if not general_bound
                        else {
                            "min_factor_shift": lowest,
                            "shift_around": bridge.shift_around,
                            "factor_reference": reference,
                        }
                    ),
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
                "lower_grain": group.lower_grain,
                "legs": leg_receipts,
            }
        )
    for (identity, lower_grain), pending in unparented_rows.items():
        if lower_grain in control_grains:
            # A control-tier row with no control above it is the top of its
            # own leg, not an orphan.
            continue
        open_rows = {
            position: leg
            for position, leg in pending.items()
            if position not in claimed[(identity, lower_grain)]
        }
        deferred = _wholly_deferred_legs(
            open_rows, reconciled, columns, licensed_empty_legs, frame_only_positions
        )
        still = set(open_rows.values()) - set(deferred)
        if still:
            raise ValueError(
                f"cross-grain inconsistency "
                f"{first_pair_id[(identity, lower_grain)]!r} has "
                f"unparented lower-grain leg(s) {sorted(still)}."
            )
        deferred_legs_without_control.extend(
            {
                "inconsistency_id": first_pair_id[(identity, lower_grain)],
                "lower_grain": lower_grain,
                "leg": leg,
                "lower_target_ids": list(target_ids),
            }
            for leg, target_ids in sorted(deferred.items())
        )
    return (
        reconciled,
        receipts,
        empty_legs_licensed,
        controls_without_lower_rows,
        delegated_legs,
        absent_middle_tier_legs,
        deferred_legs_without_control,
        unbound_legs,
    )


def _wholly_deferred_legs(
    rows: Mapping[int, str],
    frame: pd.DataFrame,
    columns: Mapping[str, str],
    licensed_empty_legs: Mapping[str, frozenset[str]] | None,
    frame_only_positions: frozenset[int],
) -> dict[str, tuple[str, ...]]:
    """The legs of ``rows`` that hold deferred cells alone.

    A leg qualifies when every row on it is frame-only (it binds nothing and
    joins the frame only to take its share of a control) and every target on
    it is licensed empty there, so no cell of it binds on the leg. With no
    control above them such rows have nothing to reconcile, and the leg is
    no orphan (microcosm#1301: a region row signed out below the small-cell
    floor leaves its wholly deferred authorities without one). A bound row,
    or a row of a target the leg is not licensed for, keeps it an orphan.
    """

    targets_by_leg: dict[str, set[str]] = {}
    bound_legs: set[str] = set()
    for position, leg in rows.items():
        targets_by_leg.setdefault(leg, set()).add(
            str(frame.iloc[position][columns["target_id"]])
        )
        if position not in frame_only_positions:
            bound_legs.add(leg)
    return {
        leg: tuple(sorted(target_ids))
        for leg, target_ids in targets_by_leg.items()
        if leg not in bound_legs
        and all(
            leg in _licensed_legs_for_target(target_id, licensed_empty_legs)
            for target_id in target_ids
        )
    }


def reconcile_cross_grain_surface(
    local_frame: pd.DataFrame,
    groups: Iterable[CrossGrainInconsistency],
    rule: CrossGrainRule,
    *,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
    check_factor_bounds: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return a copied surface with every detected lower grain rescaled."""

    reconciled, receipts, *_ = _reconcile_cross_grain_surface_with_control_receipts(
        local_frame,
        groups,
        rule,
        licensed_empty_legs=licensed_empty_legs,
        check_factor_bounds=check_factor_bounds,
    )
    return reconciled, receipts


def apply_cross_grain_reconciliation(
    local_frame: pd.DataFrame,
    bound_higher_targets: Iterable[str],
    contract_signatures: Mapping[str, Mapping[str, Any]],
    rule: CrossGrainRule,
    *,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None = None,
    check_factor_bounds: bool = True,
    frame_only_positions: Iterable[int] = (),
    expected_members: Mapping[str, Mapping[str, frozenset[str]]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect, reconcile, and return the always-present pass receipt.

    ``check_factor_bounds=False`` records each bounded bridge's factor without
    refusing it, for a pass that is not the caller's last (a surface whose
    off-surface cells join a second pass is bounded on that one).
    ``frame_only_positions`` names the rows that bind nothing (deferred
    cells); a leg holding only those needs no control
    (``deferred_legs_without_control``).
    ``expected_members`` maps a lower target to the geographies that each of
    its legs must hold. A leg that holds other rows than those is refused:
    rescaling a set that is not complete would move the missing members'
    mass onto the present ones. A declared leg with no row is refused in the
    same way. A leg that declares no member holds no row, and the empty-leg
    licences decide it.
    """

    _assert_expected_members(local_frame, rule, expected_members)
    bound = tuple(str(target_id) for target_id in bound_higher_targets)
    groups = detect_cross_grain_inconsistencies(
        local_frame,
        bound,
        contract_signatures,
        rule,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        licensed_empty_legs=licensed_empty_legs,
    )
    unbound_bridges = _reviewed_unbound_bridges(
        set(bound),
        rule,
        reviewed_unbound_higher_targets,
    )
    (
        reconciled,
        group_receipts,
        empty_legs_licensed,
        controls_without_lower_rows,
        delegated_legs,
        absent_middle_tier_legs,
        deferred_legs_without_control,
        unbound_legs,
    ) = _reconcile_cross_grain_surface_with_control_receipts(
        local_frame,
        groups,
        rule,
        licensed_empty_legs=licensed_empty_legs,
        check_factor_bounds=check_factor_bounds,
        frame_only_positions=frozenset(int(p) for p in frame_only_positions),
    )
    receipt = {
        "bound_higher_targets": list(bound),
        "factor_bounds_checked": check_factor_bounds,
        "inconsistencies_in_force": [group.inconsistency_id for group in groups],
        "groups": group_receipts,
        "unbound_bridges": unbound_bridges,
        "empty_legs_licensed": empty_legs_licensed,
        "controls_without_lower_rows": controls_without_lower_rows,
        "delegated_legs": delegated_legs,
        "absent_middle_tier_legs": absent_middle_tier_legs,
        "deferred_legs_without_control": deferred_legs_without_control,
        # Only a rule with a bridge that unbinds carries the key, so the
        # receipt of every other rule keeps the keys it had.
        **(
            {"unbound_legs": unbound_legs}
            if any(bridge.out_of_bound == "unbind" for bridge in rule.bridges)
            else {}
        ),
        "absence": (
            None
            if groups
            else "No cross-grain inconsistencies are in force on this surface."
        ),
    }
    return reconciled, receipt


PARTITION_KINDS = ("exhaustive", "share_of_parent")


@dataclass(frozen=True)
class CrossGrainPartition:
    """Member targets that divide one parent target at the same geography.

    ``exhaustive``: the members sum to the parent, so they are rescaled jointly
    onto it (council-tax bands onto the authority's total). ``share_of_parent``:
    the members cover part of the parent (tenure categories the census also
    splits into shared ownership and rent-free), so they move by the parent's
    own reconciliation factor and keep their shares.
    """

    partition_id: str
    parent_target_id: str
    member_target_ids: tuple[str, ...]
    kind: str = "exhaustive"


def apply_cross_grain_partitions(
    raw_frame: pd.DataFrame,
    reconciled_frame: pd.DataFrame,
    partitions: Iterable[CrossGrainPartition],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rescale each partition's members onto their reconciled parent.

    Runs after :func:`apply_cross_grain_reconciliation`: ``raw_frame`` is its
    input and ``reconciled_frame`` its output, row for row, so a
    ``share_of_parent`` partition reads the parent's own factor. A geography
    with the parent and no members is recorded; one with some members but not
    all is refused, since rescaling a partial set onto the whole parent would
    move the missing members' mass onto the present ones.
    """

    columns = _surface_columns(reconciled_frame)
    if len(raw_frame) != len(reconciled_frame):
        raise ValueError("cross-grain partitions need the raw and reconciled surface.")
    partitions = tuple(partitions)
    _validate_partitions(partitions)
    out = reconciled_frame.copy(deep=True)
    value_column = out.columns.get_loc(columns["value"])
    raw_values = raw_frame[columns["value"]].to_numpy(dtype=np.float64)
    target_ids = [
        _contract_target_id(str(value)) for value in out[columns["target_id"]]
    ]
    cells: dict[tuple[str, str], dict[str, list[int]]] = {}
    for position, (grain, geography_id, target_id) in enumerate(
        zip(
            out[columns["grain"]], out[columns["geography_id"]], target_ids, strict=True
        )
    ):
        cells.setdefault((str(grain), str(geography_id)), {}).setdefault(
            target_id, []
        ).append(position)
    receipts: list[dict[str, Any]] = []
    parents_without_members: list[dict[str, Any]] = []
    for partition in partitions:
        legs: list[dict[str, Any]] = []
        for (grain, geography_id), by_target in sorted(cells.items()):
            parent_rows = by_target.get(partition.parent_target_id, [])
            if not parent_rows:
                continue
            if len(parent_rows) != 1:
                raise ValueError(
                    f"cross-grain partition {partition.partition_id!r} has "
                    f"{len(parent_rows)} parent rows at {geography_id!r}."
                )
            present = [
                member for member in partition.member_target_ids if member in by_target
            ]
            if not present:
                parents_without_members.append(
                    {
                        "partition_id": partition.partition_id,
                        "grain": grain,
                        "geography_id": geography_id,
                    }
                )
                continue
            if len(present) != len(partition.member_target_ids):
                missing = sorted(set(partition.member_target_ids) - set(present))
                raise ValueError(
                    f"cross-grain partition {partition.partition_id!r} at "
                    f"{geography_id!r} lacks member(s) {missing}."
                )
            positions = [
                position for member in present for position in by_target[member]
            ]
            parent = parent_rows[0]
            parent_value = float(out.iloc[parent][columns["value"]])
            member_values = out.iloc[positions][columns["value"]].to_numpy(
                dtype=np.float64
            )
            old_total = float(member_values.sum())
            if partition.kind == "exhaustive":
                if abs(old_total) < _MIN_LEG_SUM_RTOL * abs(parent_value):
                    raise ValueError(
                        f"cross-grain partition {partition.partition_id!r} at "
                        f"{geography_id!r} cannot scale a vanishing member total."
                    )
                factor = 1.0 if old_total == 0.0 else parent_value / old_total
                expected_total = parent_value
            else:
                raw_parent = float(raw_values[parent])
                if raw_parent == 0.0:
                    factor = 1.0
                else:
                    factor = parent_value / raw_parent
                expected_total = old_total * factor
            if not np.isfinite(factor) or factor < 0.0:
                raise ValueError(
                    f"cross-grain partition {partition.partition_id!r} at "
                    f"{geography_id!r} produced factor {factor!r}."
                )
            out.iloc[positions, value_column] = member_values * factor
            new_total = float(
                out.iloc[positions][columns["value"]].to_numpy(dtype=np.float64).sum()
            )
            if not np.isclose(new_total, expected_total, rtol=_CLOSURE_RTOL, atol=0.0):
                raise ValueError(
                    f"cross-grain partition {partition.partition_id!r} left "
                    f"{geography_id!r} at {new_total!r} against {expected_total!r}."
                )
            legs.append(
                {
                    "grain": grain,
                    "geography_id": geography_id,
                    "parent_value": parent_value,
                    "old_total": old_total,
                    "new_total": new_total,
                    "declared_factor": factor,
                }
            )
        receipts.append(
            {
                "partition_id": partition.partition_id,
                "kind": partition.kind,
                "parent_target_id": partition.parent_target_id,
                "member_target_ids": list(partition.member_target_ids),
                "cells": legs,
            }
        )
    return out, {
        "partitions": receipts,
        "parents_without_members": parents_without_members,
    }


def _validate_partitions(partitions: tuple[CrossGrainPartition, ...]) -> None:
    _require_unique_nonblank(
        tuple(partition.partition_id for partition in partitions),
        label="partition ids",
    )
    for partition in partitions:
        if partition.kind not in PARTITION_KINDS:
            raise ValueError(
                f"cross-grain partition {partition.partition_id!r} has unknown "
                f"kind {partition.kind!r}; expected one of {PARTITION_KINDS}."
            )
        if not partition.parent_target_id or not partition.member_target_ids:
            raise ValueError(
                f"cross-grain partition {partition.partition_id!r} has a blank "
                "parent or no members."
            )
        _require_unique_nonblank(
            partition.member_target_ids,
            label=f"partition {partition.partition_id!r} members",
        )
        if partition.parent_target_id in partition.member_target_ids:
            raise ValueError(
                f"cross-grain partition {partition.partition_id!r} lists its "
                "parent as a member."
            )


def _reviewed_unbound_bridges(
    bound_higher_targets: set[str],
    rule: CrossGrainRule,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None,
) -> list[dict[str, Any]]:
    reviewed = reviewed_unbound_higher_targets or {}
    unbound: list[dict[str, Any]] = []
    for bridge in rule.bridges:
        higher = set(bridge.higher_target_ids)
        selected = bound_higher_targets & higher
        if not selected or selected == higher:
            continue
        missing = sorted(higher - selected)
        unreviewed = [target_id for target_id in missing if target_id not in reviewed]
        if unreviewed:
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} is partially bound; "
                f"missing higher target(s) {missing}; missing target(s) that "
                f"lack a reviewed exclusion: {unreviewed}."
            )
        unbound.append(
            {
                "bridge_id": bridge.bridge_id,
                "missing": missing,
                "basis": "reviewed_exclusion",
                "records": {
                    target_id: dict(reviewed[target_id]) for target_id in missing
                },
            }
        )
    return unbound


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


def _control_grains(rule: CrossGrainRule) -> tuple[str, ...]:
    return tuple(rule.control_grains) or (rule.grain_precedence[0],)


def _validate_rule(rule: CrossGrainRule) -> None:
    _require_unique_nonblank(rule.grain_precedence, label="grain_precedence")
    if rule.control_grains:
        _require_unique_nonblank(rule.control_grains, label="control_grains")
        unknown = [g for g in rule.control_grains if g not in rule.grain_precedence]
        if unknown:
            raise ValueError(
                f"cross-grain control_grains {unknown} are not in grain_precedence."
            )
        if rule.grain_precedence[0] not in rule.control_grains:
            raise ValueError(
                "cross-grain control_grains must include the top grain "
                f"{rule.grain_precedence[0]!r}."
            )
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
        external_sides = [
            side
            for side in (*bridge.higher_target_ids, bridge.lower_side)
            if side.startswith("external:")
        ]
        if external_sides:
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} uses forbidden "
                f"external side(s) {external_sides}; all sides must be contract targets."
            )
        bound = bridge.max_factor_shift
        if bound is not None and (
            isinstance(bound, bool)
            or not isinstance(bound, int | float)
            or not 0.0 < bound < 1.0
        ):
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} declares max_factor_shift "
                f"{bound!r}; a bound must be a number in (0, 1): at one or more "
                "it would accept every factor below one."
            )
        floor = bridge.min_factor_shift
        if (
            bridge.shift_around not in SHIFT_REFERENCES
            or bridge.out_of_bound not in OUT_OF_BOUND_ACTIONS
        ):
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} declares shift_around "
                f"{bridge.shift_around!r} and out_of_bound {bridge.out_of_bound!r}; "
                f"expected one of {SHIFT_REFERENCES} and one of "
                f"{OUT_OF_BOUND_ACTIONS}."
            )
        if bound is None and (
            floor is not None
            or bridge.shift_around != "one"
            or bridge.out_of_bound != "refuse"
        ):
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} shapes a bound it does "
                "not declare; set max_factor_shift."
            )
        if floor is not None and (
            isinstance(floor, bool)
            or not isinstance(floor, int | float)
            or not -1.0 < floor < bound
        ):
            raise ValueError(
                f"cross-grain bridge {bridge.bridge_id!r} declares min_factor_shift "
                f"{floor!r}; it must be a number above -1 and below "
                f"max_factor_shift {bound!r}."
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
            canonical = side.removeprefix("contract:")
            aliases = {canonical, f"contract:{canonical}"}
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
        (field, _canonical_signature_value(measurement.get(field))) for field in fields
    )


def _canonical_signature_value(value: Any) -> Any:
    frozen = _freeze(value)
    if frozen is None or frozen == () or frozen == "":
        return _UNSPECIFIED
    if isinstance(value, (list, tuple, set, frozenset)):
        # Order-insensitive at this level only: a signature field holding a
        # sequence is a conjunction (the filter list), so its members are the
        # same measurement however the contract orders them.  Ordering *inside*
        # a member stays significant — a `between` payload's bounds are data,
        # not a conjunction — so `_freeze` below is order-preserving.
        return tuple(sorted(frozen, key=_sort_key))
    return frozen


def _sort_key(member: Any) -> str:
    return json.dumps(_json_safe(member), sort_keys=True, separators=(",", ":"))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _freeze(member)) for key, member in value.items())
        )
    if isinstance(value, (set, frozenset)):
        # A set has no order to preserve; sort for a deterministic signature.
        return tuple(sorted((_freeze(member) for member in value), key=_sort_key))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(member) for member in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 0 and 0.0 are the same signature value; keep one numeric spelling.
        return float(value)
    return value


def _contract_target_id(side: str) -> str:
    if side.startswith("external:"):
        raise ValueError(
            f"cross-grain target side {side!r} is external; every side must "
            "name a contract target."
        )
    return side.removeprefix("contract:")


def _validate_licensed_empty_legs(
    licensed_empty_legs: Mapping[str, frozenset[str]] | None,
) -> None:
    if licensed_empty_legs is None:
        return
    if not isinstance(licensed_empty_legs, Mapping):
        raise TypeError("licensed_empty_legs must be a mapping.")
    for target_id, legs in licensed_empty_legs.items():
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("licensed_empty_legs target ids must be nonblank strings.")
        if not isinstance(legs, frozenset):
            raise TypeError(f"licensed_empty_legs[{target_id!r}] must be a frozenset.")
        if any(not isinstance(leg, str) or not leg.strip() for leg in legs):
            raise ValueError(
                f"licensed_empty_legs[{target_id!r}] must contain nonblank strings."
            )


def _licensed_legs_for_target(
    target_id: str,
    licensed_empty_legs: Mapping[str, frozenset[str]] | None,
) -> frozenset[str]:
    if licensed_empty_legs is None:
        return frozenset()
    canonical = _contract_target_id(target_id)
    aliases = {target_id}
    if canonical is not None:
        aliases.update((canonical, f"contract:{canonical}"))
    return frozenset(
        leg for alias in aliases for leg in licensed_empty_legs.get(alias, frozenset())
    )


def _inconsistency_identity(
    kind: str,
    bridge_id: str,
    signature: tuple[tuple[str, Any], ...],
) -> str:
    if kind == "bridge":
        return bridge_id
    payload = json.dumps(_json_safe(signature), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    concept = next(
        (str(value) for field, value in signature if field == "concept"), "target"
    )
    return f"exact:{concept}:{digest}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(member) for member in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(member) for key, member in value.items()}
    return value


def _row_legs(geography_id: str, rule: CrossGrainRule) -> tuple[str, ...]:
    """The legs one lower row covers.

    A row on a declared parent geography covers that geography's legs, so a
    nation row (England) sits under a country control across all of its
    regions; any other row covers the one leg its area maps to.
    """

    declared = rule.parent_geography_legs.get(geography_id)
    if declared is not None:
        return tuple(str(leg) for leg in declared)
    leg = str(rule.leg_of_area(geography_id))
    return (leg,) if leg else ()


def _control_positions(
    control: Mapping[str, Any],
    lower_by_leg: Mapping[str, list[int]],
    row_legs: Mapping[int, tuple[str, ...]],
    inconsistency_id: str,
) -> list[int]:
    """The unclaimed lower rows on a control's legs, each once.

    A row whose legs are not all inside the control's would be rescaled by a
    factor computed for a different total, so it is refused.
    """

    covered = tuple(control["covered_legs"])
    positions = list(
        dict.fromkeys(
            position for leg in covered for position in lower_by_leg.get(leg, ())
        )
    )
    for position in positions:
        outside = sorted(set(row_legs[position]) - set(covered))
        if outside:
            raise ValueError(
                f"cross-grain inconsistency {inconsistency_id!r}: a lower row "
                f"spans leg(s) {outside} outside control "
                f"{control['parent_geography_id']!r}; a row must sit inside one "
                "control."
            )
    return positions


def _allocation(
    allocator: Callable[[np.ndarray, float], tuple[np.ndarray, float, float]],
    raw_values: np.ndarray,
    parent_value: float,
    inconsistency_id: str,
) -> tuple[np.ndarray, float, float]:
    """One control's lower rows as the rule's declared allocator divides them.

    Returns ``(values, factor, raw_sum)``. A result that is not one value for
    each row is refused here; the caller's refusals (a non-finite or vanishing
    total, a negative factor, a total off its control) then read ``factor``
    and ``raw_sum`` as they read the standing rule's own.
    """

    values, factor, raw_sum = allocator(raw_values, parent_value)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != raw_values.shape:
        raise ValueError(
            f"cross-grain inconsistency {inconsistency_id!r}: allocator "
            f"{_allocator_name(allocator)!r} returned {values.shape} value(s) "
            f"for {raw_values.shape} lower row(s)."
        )
    return values, float(factor), float(raw_sum)


def _allocator_name(allocator: Callable[..., Any]) -> str:
    return str(getattr(allocator, "__name__", type(allocator).__name__))


def _input_factor(
    input_values: np.ndarray,
    parent_value: float,
    rule: CrossGrainRule,
    inconsistency_id: str,
) -> tuple[float, float]:
    """``(factor, lower total)`` of a leg on its rows as given.

    The factor and the total are those of the rule's own division: the
    allocator's when it declares one, else the control over the numpy sum.
    """

    if rule.allocator is not None:
        _values, factor, input_sum = _allocation(
            rule.allocator, input_values, parent_value, inconsistency_id
        )
        return factor, input_sum
    input_sum = float(input_values.sum())
    if input_sum != 0.0:
        return parent_value / input_sum, input_sum
    return (1.0 if parent_value == 0.0 else float("inf")), input_sum


def _median_input_factor(
    local_frame: pd.DataFrame,
    controls: Iterable[Mapping[str, Any]],
    lower_by_leg: Mapping[str, list[int]],
    row_legs: Mapping[int, tuple[str, ...]],
    columns: Mapping[str, str],
    rule: CrossGrainRule,
    inconsistency_id: str,
) -> float:
    """The median factor across a bridge's legs, on their rows as given.

    Only a leg with a nonzero total and a positive finite factor counts, so
    a leg that cannot be scaled does not move the reference. With no such
    leg the reference is ``NaN``, and every judged leg is out of bound.
    """

    factors: list[float] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for control in controls:
        key = (str(control["parent_geography_id"]), tuple(control["covered_legs"]))
        if key in seen:
            continue
        seen.add(key)
        positions = _control_positions(
            control, lower_by_leg, row_legs, inconsistency_id
        )
        factor, input_sum = _input_factor(
            local_frame.iloc[positions][columns["value"]].to_numpy(dtype=np.float64),
            float(control["value"]),
            rule,
            inconsistency_id,
        )
        if input_sum != 0.0 and np.isfinite(factor) and factor > 0:
            factors.append(factor)
    return float(np.median(factors)) if factors else float("nan")


def _assert_expected_members(
    local_frame: pd.DataFrame,
    rule: CrossGrainRule,
    expected_members: Mapping[str, Mapping[str, frozenset[str]]] | None,
) -> None:
    """Refuse a leg of a target that does not hold exactly its members.

    The check reads every declared leg, with or without rows, so a leg whose
    members are all absent is refused like a leg that lacks one member.
    """

    if not expected_members:
        return
    columns = _surface_columns(local_frame)
    expected = {
        _contract_target_id(str(target_id)): by_leg
        for target_id, by_leg in expected_members.items()
    }
    held: dict[tuple[str, str], set[str]] = {}
    for side, area in zip(
        local_frame[columns["target_id"]].tolist(),
        local_frame[columns["geography_id"]].tolist(),
        strict=True,
    ):
        target_id = _contract_target_id(str(side))
        if target_id not in expected:
            continue
        for leg in _row_legs(str(area), rule):
            held.setdefault((target_id, leg), set()).add(str(area))
    declared = {
        (target_id, str(leg))
        for target_id, by_leg in expected.items()
        for leg in by_leg
    }
    for target_id, leg in sorted(declared | held.keys()):
        present = held.get((target_id, leg), set())
        members = expected[target_id].get(leg)
        if members is None:
            raise ValueError(
                f"cross-grain target {target_id!r} has rows on leg {leg!r}, "
                "which declares no members for it."
            )
        missing = sorted(set(members) - present)
        unexpected = sorted(present - set(members))
        if missing or unexpected:
            raise ValueError(
                f"cross-grain target {target_id!r} does not hold the declared "
                f"members of leg {leg!r}: missing {missing}, unexpected "
                f"{unexpected}. Rescaling a set that is not complete would "
                "misstate every member."
            )


def _controls_agree(left: float, right: float) -> bool:
    return bool(
        np.isclose(float(left), float(right), rtol=_CONTROL_AGREEMENT_RTOL, atol=0.0)
    )


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
    # A winning row whose geography declares its legs is a parent control in
    # its own right, whatever grain it sits at: the top grain always is, and a
    # middle tier (a region between country and constituency) is when the
    # country declares a leg map for every one of its codes. Rows with no
    # declared legs stay on the area-derived path below.
    declared_parents = group.winning_grain == rule.grain_precedence[0] or all(
        row["geography_id"] in rule.parent_geography_legs for row in rows
    )
    if declared_parents:
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
                missing = sorted(set(bridge.higher_target_ids) - set(deduplicated))
                if missing and not bridge.per_geography:
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
                f"cross-grain winning area {row['geography_id']!r} maps to a blank leg."
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
                        if (target_id := _contract_target_id(member["target_id"]))
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
                if not _controls_agree(
                    left_by_coverage[coverage], right_by_coverage[coverage]
                ):
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
