"""Signed cells that are off the UK local surface still count in their leg (#1123).

A leg's control is the total of every area in the leg. The local membership
signs some cells off the surface: a cell deferred with its published value
(the City of London, the Isles of Scilly), or a cell the publisher's table
does not carry (HMRC's local-authority tables have no Torbay or Isles of
Scilly row). The reconciliation frame holds bound cells only, so without
those cells a leg's bound cells take their share of the control: the South
West's local authorities asked for 2.4% more employees than its
constituencies did for the same households (microcosm#1198 review).

The cells join the reconciliation frame, where they take their share, and
never reach the solve:

* A cell with a published value joins with that value, moved as its leg's
  bound cells were moved (their uprating index, the census household factor).
* A cell with no value is implied from the target's other local tier. Over
  the legs where both tiers are whole, the two tiers' totals stand in a
  ratio; the leg's total at the whole tier, times that ratio, less the cells
  the short tier does carry, is what the absent cells hold.

Only a leg whose bound cells a control moves needs this, so the rows are
built from a first reconciliation pass that names those legs. An absent cell
in such a leg with no whole tier to imply it from is refused: its neighbours
would take its share unseen.

An implied row is bounded. It is what the leg's bound cells do not hold, so
an undercount anywhere in the leg would pass into it and through every gate
that counts it. Its share of the leg may be at most
``UK_IMPLIED_SHARE_MAX_RATIO`` times its areas' share of the leg's census
households; past that it is refused.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: Membership geography levels and the frame's grain names for them.
_FRAME_GRAIN_BY_LEVEL = {"constituency": "constituency", "local_authority": "la"}
_LOCAL_GRAINS = tuple(_FRAME_GRAIN_BY_LEVEL.values())

LegKey = tuple[str, str, str]

#: An implied row's share of its leg, over its areas' share of the leg's
#: census households. On the 942a1fa feed Torbay and the Isles of Scilly are
#: implied at 0.32 (higher rate taxpayers) to 0.89 (employees) of their
#: household share, and one North West constituency's self-employment income
#: at 0.37 of its own.
UK_IMPLIED_SHARE_MAX_RATIO = 1.5


def uk_signed_cells_off_the_surface(
    membership: Mapping[str, Any],
    *,
    rule_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict[str, Any], ...]:
    """Every membership cell that is signed off the surface.

    A cell deferred under a small-cell rule (``rule_ids``) is left out: the
    rule's cells join the frame through ``uk_small_cell_deferred_cells``.
    ``value`` is the published value of a deferred cell, and ``None`` where
    the publisher carries no fact for the area. ``area_has_bound_cells`` says
    whether the membership binds any cell, of any target, for the area at
    that level: a whole run then carries the area, so a run whose ladder
    lacks it is a narrowed one.
    """

    bound_areas = {
        (str(level), str(candidate["geography_id"]))
        for target in membership.get("targets", {}).values()
        for level, payload in target["geography_levels"].items()
        for candidate in payload["candidates"]
        if candidate.get("status") == "active"
    }
    cells = []
    for target_id, target in sorted(membership.get("targets", {}).items()):
        for level, payload in sorted(target["geography_levels"].items()):
            if level not in _FRAME_GRAIN_BY_LEVEL:
                continue
            for candidate in payload["candidates"]:
                reason_id = candidate.get("signed_reason_id")
                if candidate.get("status") == "active" or reason_id in rule_ids:
                    continue
                value = candidate.get("deferred_value")
                cells.append(
                    {
                        "target_id": str(target_id),
                        "geography_level": str(level),
                        "geography_id": str(candidate["geography_id"]),
                        "value": None if value is None else float(value),
                        "reason_id": None if reason_id is None else str(reason_id),
                        "area_has_bound_cells": (
                            str(level),
                            str(candidate["geography_id"]),
                        )
                        in bound_areas,
                    }
                )
    return tuple(cells)


def uk_off_surface_frame_rows(
    bound_rows: Sequence[Mapping[str, Any]],
    off_surface_cells: Sequence[Mapping[str, Any]],
    *,
    leg_of_area: Callable[[str], str],
    controlled_legs: set[LegKey] | frozenset[LegKey],
    frame_only_rows: Sequence[Mapping[str, Any]] = (),
    share_reference: Mapping[tuple[str, str], float] | None = None,
    max_implied_share_ratio: float = UK_IMPLIED_SHARE_MAX_RATIO,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Frame rows for the off-surface cells of the controlled legs.

    ``bound_rows`` are the surface's bound local cells: ``target_id`` (the
    contract id), ``grain``, ``geography_id``, ``value`` (as the frame holds
    it) and ``published_value`` (before any index or factor moved it).
    ``controlled_legs`` are the (target, grain, leg) keys whose bound cells
    a first reconciliation pass moved. ``frame_only_rows`` are the rows the
    frame already holds for cells that are not bound (the small-cell rule's):
    they count in their leg's total when a tier is read to imply another.
    ``share_reference`` gives each area's census households by (grain, area):
    it bounds an implied row's share of its leg. A surface that binds no
    census households passes none, and the receipt says the row is unbounded.

    Each returned row carries ``value`` at the frame's scale and
    ``published_value`` at the publisher's, so a source-agreement test can
    count it on either side.
    """

    bound: dict[LegKey, list[float]] = {}
    for row in bound_rows:
        key = (
            str(row["target_id"]),
            str(row["grain"]),
            leg_of_area(str(row["geography_id"])),
        )
        totals = bound.setdefault(key, [0.0, 0.0])
        totals[0] += float(row["value"])
        totals[1] += float(row["published_value"])

    def moved(key: LegKey) -> float:
        frame, published = bound[key]
        return frame / published if published else 1.0

    with_value: dict[LegKey, list[Mapping[str, Any]]] = {}
    absent: dict[LegKey, list[str]] = {}
    for cell in off_surface_cells:
        grain = _FRAME_GRAIN_BY_LEVEL.get(str(cell["geography_level"]))
        if grain is None:
            continue
        key = (
            str(cell["target_id"]),
            grain,
            leg_of_area(str(cell["geography_id"])),
        )
        if key not in bound:
            # No bound cell on the leg: nothing can take this cell's share,
            # and the wholly empty leg stays on its licence.
            continue
        if cell.get("value") is None:
            absent.setdefault(key, []).append(str(cell["geography_id"]))
        else:
            with_value.setdefault(key, []).append(cell)

    rows: list[dict[str, Any]] = []
    #: Frame and published totals of the unbound rows in the frame, by leg key.
    added: dict[LegKey, list[float]] = {}
    for row in frame_only_rows:
        key = (
            str(row["target_id"]),
            str(row["grain"]),
            leg_of_area(str(row["geography_id"])),
        )
        totals = added.setdefault(key, [0.0, 0.0])
        totals[0] += float(row["value"])
        totals[1] += float(row["value"])

    def total(key: LegKey) -> float:
        return bound[key][0] + added.get(key, (0.0, 0.0))[0]

    def in_play(key: LegKey) -> bool:
        # A control moves the leg at this tier, or at the other local tier,
        # where this tier's leg total is the parent or the child.
        target_id, grain, leg = key
        return any((target_id, tier, leg) in controlled_legs for tier in _LOCAL_GRAINS)

    deferred_receipt: list[dict[str, Any]] = []
    for key in sorted(with_value):
        target_id, grain, leg = key
        factor = moved(key)
        cells = sorted(with_value[key], key=lambda c: str(c["geography_id"]))
        published_total = math.fsum(float(cell["value"]) for cell in cells)
        # Every leg's total counts its deferred cells, so a leg no control
        # moves still reads whole when the two tiers' ratio is taken.
        totals = added.setdefault(key, [0.0, 0.0])
        totals[0] += published_total * factor
        totals[1] += published_total
        if not in_play(key):
            continue
        for cell in cells:
            published = float(cell["value"])
            rows.append(_frame_row(grain, cell["geography_id"], target_id))
            rows[-1].update(value=published * factor, published_value=published)
        deferred_receipt.append(
            {
                "target_id": target_id,
                "grain": grain,
                "leg": leg,
                "areas": sorted(str(c["geography_id"]) for c in with_value[key]),
                "published_total": published_total,
                "frame_total": published_total * factor,
                "share_of_leg": published_total * factor / total(key),
            }
        )

    implied_receipt: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for key in sorted(absent):
        if key not in controlled_legs:
            continue
        target_id, grain, leg = key
        areas = sorted(absent[key])
        (sibling,) = (other for other in _LOCAL_GRAINS if other != grain)
        whole_legs = sorted(
            other_leg
            for other_target, other_grain, other_leg in bound
            if other_target == target_id
            and other_grain == grain
            and (target_id, sibling, other_leg) in bound
            and (target_id, grain, other_leg) not in absent
            and (target_id, sibling, other_leg) not in absent
        )
        sibling_key = (target_id, sibling, leg)
        if sibling_key not in bound or sibling_key in absent or not whole_legs:
            unresolved.append(
                {"target_id": target_id, "grain": grain, "leg": leg, "areas": areas}
            )
            continue
        ratio = math.fsum(
            total((target_id, grain, other)) for other in whole_legs
        ) / math.fsum(total((target_id, sibling, other)) for other in whole_legs)
        leg_total = ratio * total(sibling_key)
        implied = leg_total - total(key)
        entry = {
            "target_id": target_id,
            "grain": grain,
            "leg": leg,
            "areas": areas,
            "implied_from_grain": sibling,
            "whole_legs": len(whole_legs),
            "tier_ratio": ratio,
            "implied_value": implied,
            "share_of_leg": implied / leg_total,
        }
        implied_receipt.append(entry)
        if implied <= 0.0:
            # The short tier already holds the leg's whole total: the absent
            # cells are inside the two tables' rounding, and no row is added.
            entry["status"] = "not_positive"
            continue
        entry["status"] = "added"
        entry.update(
            _implied_share_bound(
                entry,
                share_reference,
                leg_of_area=leg_of_area,
                max_ratio=max_implied_share_ratio,
            )
        )
        rows.append(_frame_row(grain, areas[0], target_id))
        rows[-1].update(value=implied, published_value=implied / moved(key))
    if unresolved:
        raise ValueError(
            "UK local surface reconciles a leg whose absent cells cannot be "
            f"implied from the target's other local tier: {unresolved}. Their "
            "share of the control would land on the leg's bound cells: bind "
            "the target at both local tiers, or declare the cells' values "
            "(microcosm#1123)."
        )
    return rows, {
        "deferred_with_value": deferred_receipt,
        "implied_absent": implied_receipt,
        "rows_added": len(rows),
    }


def _implied_share_bound(
    entry: Mapping[str, Any],
    share_reference: Mapping[tuple[str, str], float] | None,
    *,
    leg_of_area: Callable[[str], str],
    max_ratio: float,
) -> dict[str, Any]:
    """Refuse an implied row that holds more of its leg than its areas could.

    The reference is the leg's census households at the row's grain. With no
    reference in the leg (a surface that binds no census households) the row
    cannot be bounded, and the receipt says so.
    """

    grain, leg = str(entry["grain"]), str(entry["leg"])
    leg_values = {
        area: float(value)
        for (reference_grain, area), value in (share_reference or {}).items()
        if reference_grain == grain and leg_of_area(area) == leg
    }
    if not leg_values:
        return {
            "household_share": None,
            "share_bound": "unbounded: the surface binds no census households here",
        }
    missing = sorted(area for area in entry["areas"] if area not in leg_values)
    if missing:
        raise ValueError(
            f"UK local surface cannot bound the implied {entry['target_id']} "
            f"cell(s) of {entry['areas']} ({grain}, leg {leg}): {missing} carry "
            "no census households to set their share against (microcosm#1123)."
        )
    household_share = math.fsum(leg_values[area] for area in entry["areas"]) / (
        math.fsum(leg_values.values())
    )
    share = float(entry["share_of_leg"])
    if share > max_ratio * household_share:
        raise ValueError(
            f"UK local surface implies {entry['target_id']} for {entry['areas']} "
            f"({grain}, leg {leg}) at {share:.4f} of the leg, more than "
            f"{max_ratio:g} times their {household_share:.4f} of the leg's "
            "census households: the leg's bound cells are short of its control "
            "by more than the absent areas can explain (microcosm#1123)."
        )
    return {
        "household_share": household_share,
        "share_ratio": share / household_share,
        "share_bound": f"at most {max_ratio:g} times the household share",
    }


def _frame_row(grain: str, geography_id: Any, target_id: str) -> dict[str, Any]:
    return {
        "grain": grain,
        "geography_id": str(geography_id),
        "target_id": f"contract:{target_id}",
        "_output_position": None,
    }
