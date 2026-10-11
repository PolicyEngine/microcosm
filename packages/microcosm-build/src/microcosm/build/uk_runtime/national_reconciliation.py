"""Reconcile the UK national register across its own grains (microcosm#1123).

Before #1123 cross-grain reconciliation ran only on the joint local surface and
wrote back only the local cells: a region row the operator rescaled to its
country control fed the constituencies below it, while the national register
the solve binds kept the region row's raw value, so the solve saw two values
for one quantity. This module reconciles the national register itself, in the
compile path every UK build shares (``load_uk_full_target_inputs`` and
``load_uk_national_target_inputs``), and writes the reconciled values back
into it. The joint local pass then finds every national factor equal to one.

Two relations reconcile here:

* exact measurement signatures across the country, nation and region grains
  (the CGT regional cells under the UK total), through the shared operator
  with the UK rule; and
* the declared ``band_bridges`` of ``uk/cross_grain_declarations.json``: a
  projected UK income-tax-liabilities band (HMRC ITL Table 2.5), or a run of
  them, controls the SPI regional cells (Table 3.11) whose band nests in it,
  so the regional cells sum to the projected UK band in every band.

Every reconciled row keeps ``cross_grain_value_before``, ``cross_grain_factor``
(exact ``repr`` strings, as target metadata is text) and ``cross_grain_control``
in its metadata.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.cross_grain_declarations import (
    load_uk_cross_grain_declarations,
    uk_cross_grain_grain,
    uk_fanout_sum_bridges,
)
from microcosm.build.uk_runtime.uprating_holds import uk_value_before_uprating
from microcosm.calibrate import TargetRegistry, TargetSpec

_CLOSURE_RTOL = 1e-9


@dataclass(frozen=True)
class UKBandBridge:
    """Projected UK band cells that control a set of regional band targets.

    The bridge refuses a factor further than ``max_factor_shift`` from one:
    each declaration states the gap it was reviewed at, so a wider gap means
    the regional table and the UK band no longer differ for the stated reason.
    """

    bridge_id: str
    higher_target_id: str
    higher_band_lower_bounds: tuple[float, ...]
    lower_target_ids: tuple[str, ...]
    max_factor_shift: float


def uk_band_bridges(
    declarations: Mapping[str, Any] | None = None,
) -> tuple[UKBandBridge, ...]:
    declared = declarations or load_uk_cross_grain_declarations()
    return tuple(
        UKBandBridge(
            bridge_id=str(entry["bridge_id"]),
            higher_target_id=str(entry["higher_target_id"]),
            higher_band_lower_bounds=tuple(
                float(value) for value in entry["higher_band_lower_bounds"]
            ),
            lower_target_ids=tuple(str(value) for value in entry["lower_target_ids"]),
            max_factor_shift=_band_bridge_bound(entry),
        )
        for entry in declared.get("band_bridges", ())
    )


def _band_bridge_bound(entry: Mapping[str, Any]) -> float:
    bound = entry.get("max_factor_shift")
    if not isinstance(bound, (int, float)) or not 0.0 < float(bound) < 1.0:
        raise ValueError(
            f"UK band bridge {entry.get('bridge_id')!r} must declare a "
            f"max_factor_shift in (0, 1), got {bound!r}."
        )
    return float(bound)


@dataclass(frozen=True)
class UKSumBridge:
    """Every cell of the higher targets, summed, controls every lower cell.

    Declared where the grains' precedence would run the wrong way: the four
    nations' official household estimates control the UK household
    composition partition, so the composition keeps its shares and takes the
    nations' total (microcosm#1123, decision "nations win").
    """

    bridge_id: str
    higher_target_ids: tuple[str, ...]
    lower_target_ids: tuple[str, ...]


def uk_sum_bridges(
    declarations: Mapping[str, Any] | None = None,
) -> tuple[UKSumBridge, ...]:
    declared = declarations or load_uk_cross_grain_declarations()
    return tuple(
        UKSumBridge(
            bridge_id=str(entry["bridge_id"]),
            higher_target_ids=tuple(str(value) for value in entry["higher_target_ids"]),
            lower_target_ids=tuple(str(value) for value in entry["lower_target_ids"]),
        )
        for entry in declared.get("sum_bridges", ())
    )


def _apply_sum_bridges(
    specs: list[TargetSpec],
    bridges: Sequence[UKSumBridge],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    by_target: dict[str, list[int]] = {}
    for position, spec in enumerate(specs):
        by_target.setdefault(_contract_target_id(spec), []).append(position)
    for bridge in bridges:
        sides = (*bridge.higher_target_ids, *bridge.lower_target_ids)
        missing = [target_id for target_id in sides if target_id not in by_target]
        if len(missing) == len(sides):
            continue
        if missing:
            raise ValueError(
                f"UK sum bridge {bridge.bridge_id!r} is partly bound: no cells "
                f"for {missing}."
            )
        higher = [p for t in bridge.higher_target_ids for p in by_target[t]]
        lower = [p for t in bridge.lower_target_ids for p in by_target[t]]
        control = math.fsum(float(specs[p].value) for p in higher)
        raw_total = math.fsum(float(specs[p].value) for p in lower)
        if raw_total <= 0.0 or control < 0.0:
            raise ValueError(
                f"UK sum bridge {bridge.bridge_id!r} cannot scale a total of "
                f"{raw_total!r} onto {control!r}."
            )
        factor = control / raw_total
        for position in lower:
            spec = specs[position]
            specs[position] = replace(
                spec,
                value=float(spec.value) * factor,
                metadata={
                    **spec.metadata,
                    "cross_grain_value_before": repr(float(spec.value)),
                    "cross_grain_factor": repr(factor),
                    "cross_grain_control": bridge.bridge_id,
                },
            )
        receipts.append(
            {
                "bridge_id": bridge.bridge_id,
                "control": control,
                "old_total": raw_total,
                "new_total": math.fsum(float(specs[p].value) for p in lower),
                "declared_factor": factor,
                "cells": len(lower),
            }
        )
    return receipts


def _contract_target_id(spec: TargetSpec) -> str:
    return str(spec.metadata.get("contract_target_id", spec.name.split("@", 1)[0]))


_BAND_IN_NAME = re.compile(r"\.band_(\d+)_(?:\d+|plus)(?:@|$)")


def _band_lower_bound(spec: TargetSpec) -> float | None:
    """The total-income band's lower bound a banded ITL cell carries.

    The compiled cell records the fact's band filter
    (``ledger_filter_total_income_lower_bound``); the reference name spells the
    band too (``…by_total_income_band.band_30000_50000``), read when the
    metadata does not carry it.
    """

    value = spec.metadata.get("ledger_filter_total_income_lower_bound")
    if value not in (None, ""):
        return float(value)
    match = _BAND_IN_NAME.search(spec.name)
    return None if match is None else float(match.group(1))


def _apply_band_bridges(
    specs: list[TargetSpec],
    bridges: Sequence[UKBandBridge],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    by_target: dict[str, list[int]] = {}
    for position, spec in enumerate(specs):
        by_target.setdefault(_contract_target_id(spec), []).append(position)
    for bridge in bridges:
        if bridge.higher_target_id not in by_target and not any(
            target_id in by_target for target_id in bridge.lower_target_ids
        ):
            # A register that binds neither side (a narrowed test or
            # diagnostic register) has nothing to reconcile on this bridge.
            continue
        higher_positions = [
            position
            for position in by_target.get(bridge.higher_target_id, ())
            if _band_lower_bound(specs[position]) in bridge.higher_band_lower_bounds
        ]
        found = sorted(
            _band_lower_bound(specs[position]) for position in higher_positions
        )
        if found != sorted(bridge.higher_band_lower_bounds):
            raise ValueError(
                f"UK band bridge {bridge.bridge_id!r} expects ITL band(s) with lower "
                f"bounds {sorted(bridge.higher_band_lower_bounds)}, found {found}."
            )
        lower_positions = [
            position
            for target_id in bridge.lower_target_ids
            for position in by_target.get(target_id, ())
        ]
        missing = [
            target_id
            for target_id in bridge.lower_target_ids
            if target_id not in by_target
        ]
        if missing:
            raise ValueError(
                f"UK band bridge {bridge.bridge_id!r} has no bound cells for {missing}."
            )
        control = math.fsum(float(specs[p].value) for p in higher_positions)
        raw_total = math.fsum(float(specs[p].value) for p in lower_positions)
        if raw_total <= 0.0 or control < 0.0:
            raise ValueError(
                f"UK band bridge {bridge.bridge_id!r} cannot scale a total of "
                f"{raw_total!r} onto {control!r}."
            )
        factor = control / raw_total
        if abs(factor - 1.0) > bridge.max_factor_shift:
            raise ValueError(
                f"UK band bridge {bridge.bridge_id!r} would rescale its regional "
                f"cells by {factor!r}, beyond the declared "
                f"{bridge.max_factor_shift!r}: the regional table and the UK "
                "band no longer differ by the gap the bridge was reviewed at "
                "(microcosm#1123)."
            )
        for position in lower_positions:
            spec = specs[position]
            specs[position] = replace(
                spec,
                value=float(spec.value) * factor,
                metadata={
                    **spec.metadata,
                    "cross_grain_value_before": repr(float(spec.value)),
                    "cross_grain_factor": repr(factor),
                    "cross_grain_control": bridge.bridge_id,
                },
            )
        new_total = math.fsum(float(specs[p].value) for p in lower_positions)
        if not math.isclose(new_total, control, rel_tol=_CLOSURE_RTOL, abs_tol=0.0):
            raise ValueError(
                f"UK band bridge {bridge.bridge_id!r} left {new_total!r} against "
                f"{control!r}."
            )
        receipts.append(
            {
                "bridge_id": bridge.bridge_id,
                "control": control,
                "old_total": raw_total,
                "new_total": new_total,
                "declared_factor": factor,
                "max_factor_shift": bridge.max_factor_shift,
                "cells": len(lower_positions),
            }
        )
    return receipts


def reconcile_uk_national_registry(
    registry: TargetRegistry,
    *,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None = None,
    declarations: Mapping[str, Any] | None = None,
) -> tuple[TargetRegistry, dict[str, Any]]:
    """Reconcile the national register across grains and write it back."""

    from microcosm.build.uk_runtime.ledger_targets import (
        _spec_geography,
        apply_uk_cross_grain_reconciliation,
    )

    specs = list(registry.specs)
    band_receipts = _apply_band_bridges(specs, uk_band_bridges(declarations))
    sum_receipts = _apply_sum_bridges(specs, uk_sum_bridges(declarations))

    rows: list[dict[str, Any]] = []
    cells: dict[tuple[str, str], int] = {}
    for position, spec in enumerate(specs):
        level, geography_id = _spec_geography(spec)
        target_id = _contract_target_id(spec)
        cells[(target_id, geography_id)] = cells.get((target_id, geography_id), 0) + 1
        rows.append(
            {
                "grain": uk_cross_grain_grain(spec.metadata, level, geography_id),
                "geography_id": geography_id,
                "target_id": f"contract:{target_id}",
                "value": float(spec.value),
                "_position": position,
            }
        )
    # A target with several cells at one geography is a band (or other)
    # fan-out: its cells are a distribution, not a cross-grain control or a
    # lower row of one, exactly as on the joint local surface.
    fanout = {target_id for (target_id, _), count in cells.items() if count > 1}
    frame = pd.DataFrame(
        [
            row
            for row in rows
            if row["target_id"].removeprefix("contract:") not in fanout
        ],
        columns=["grain", "geography_id", "target_id", "value", "_position"],
    )
    bound = sorted(
        {row["target_id"].removeprefix("contract:") for row in frame.to_dict("records")}
    )
    reconciled, receipt = apply_uk_cross_grain_reconciliation(
        frame[["grain", "geography_id", "target_id", "value"]],
        bound,
        reviewed_unbound_higher_targets=reviewed_unbound_higher_targets,
        licensed_empty_legs={},
    )
    moved = 0
    for value, position, group_value in zip(
        reconciled["value"].to_numpy(dtype=np.float64),
        frame["_position"],
        frame["value"].to_numpy(dtype=np.float64),
        strict=True,
    ):
        if value == group_value:
            continue
        spec = specs[int(position)]
        previous = float(spec.metadata.get("cross_grain_value_before", spec.value))
        specs[int(position)] = replace(
            spec,
            value=float(value),
            metadata={
                **spec.metadata,
                "cross_grain_value_before": repr(previous),
                # A rescaled row was nonzero: a factor cannot move a zero.
                "cross_grain_factor": repr(float(value) / previous),
                "cross_grain_control": "exact_signature",
            },
        )
        moved += 1
    if all(new is old for new, old in zip(specs, registry.specs, strict=True)):
        out = registry
    else:
        out = TargetRegistry(specs, country=registry.country)
    _assert_national_closure(out, reviewed_unbound_higher_targets, declarations)
    return out, {
        "band_bridges": band_receipts,
        "sum_bridges": sum_receipts,
        "fanout_sum_controls": uk_fanout_sum_controls(out, declarations),
        "cross_grain": receipt,
        "fanout_targets_not_controls": sorted(fanout),
        "rows_moved_by_exact_signature": moved,
    }


def uk_fanout_sum_controls(
    registry: TargetRegistry,
    declarations: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Each declared fan-out's total over every cell of the full register.

    Read from the compiled register before the measure exclusions, so a band
    the solve does not bind (the SPI £1m-and-over cells, whose spine support
    is zero) still counts in the total the area cells divide.
    """

    from microcosm.build.uk_runtime.ledger_targets import _spec_geography

    controls: dict[str, dict[str, Any]] = {}
    for bridge in uk_fanout_sum_bridges(declarations):
        cells = [
            spec
            for spec in registry.specs
            if _contract_target_id(spec) == bridge.higher_target_id
            and _spec_geography(spec)[1] == bridge.geography_id
        ]
        if not cells:
            continue
        controls[bridge.bridge_id] = {
            "higher_target_id": bridge.higher_target_id,
            "geography_id": bridge.geography_id,
            "control": math.fsum(float(spec.value) for spec in cells),
            # The same total before the bands' own uprating index, so the
            # joint surface can test source agreement apart from the indices.
            "control_before_alignment": math.fsum(
                uk_value_before_uprating(spec) for spec in cells
            ),
            "cells": sorted(spec.name for spec in cells),
        }
    return controls


#: The widest rounding allowance a subset relation may declare: two of the
#: publisher's tables of one total, each rounded on its own.
UK_SUBSET_MAX_EXCESS = 0.02


def assert_uk_subset_relations(
    national_registry: TargetRegistry,
    *,
    local_surface: pd.DataFrame | None = None,
    declarations: Mapping[str, Any] | None = None,
    local_target_ids: frozenset[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Refuse a declared part that exceeds its whole on the compiled surface.

    A ``subset`` relation lets two overlapping targets stand unreconciled
    because the part is bounded by the whole. Declared alone it only silences
    the static overlap check, so an infeasible pair would compile; this tests
    it on the values the solve binds.

    * The whole is every cell of its target in the full compiled register (a
      banded total sums its bands, a region-tier target its cells).
    * The part is summed the same way. ``other_parts`` names further targets
      that are disjoint from it and from one another: the test is on their
      sum, so nine age bands and the two 90-and-over rows are tested together
      against the UK total, where one band alone could never breach it.
    * Where the part and the whole both carry cells at one geography, the
      part is tested there too.
    * A local part is tested at each local grain on its own.

    ``max_excess`` is a rounding allowance, refused above
    ``UK_SUBSET_MAX_EXCESS``. A relation is skipped, with its reason, only
    when a register binds neither side (a narrowed test or diagnostic
    register) or when the part is local and this loader has no local surface.
    One side bound and the other not is refused: the relation would stand
    untested. ``local_target_ids`` defaults to the contract's local targets.
    """

    from microcosm.build.uk_runtime.ledger_targets import _uk_contract_targets

    declared = declarations or load_uk_cross_grain_declarations()
    if local_target_ids is None:
        local_target_ids = set(_uk_contract_targets(national_only=False)) - set(
            _uk_contract_targets(national_only=True)
        )
    national: dict[str, dict[str, float]] = {}
    for spec in national_registry.specs:
        cells = national.setdefault(_contract_target_id(spec), {})
        geography_id = str(
            spec.metadata.get("geography_id")
            or spec.metadata.get("ledger_geography_id")
            or ""
        )
        cells[geography_id] = cells.get(geography_id, 0.0) + float(spec.value)
    local_totals: dict[str, dict[str, float]] = {}
    if local_surface is not None and len(local_surface):
        grouped = local_surface.groupby(["contract_target_id", "area_type"])["value"]
        for (target_id, area_type), total in grouped.sum().items():
            local_totals.setdefault(str(target_id), {})[str(area_type)] = float(total)

    receipts: list[dict[str, Any]] = []
    breaches: list[str] = []
    untested: list[str] = []
    for relation in declared.get("relations", ()):
        if relation.get("relation") != "subset":
            continue
        part, whole = str(relation["part"]), str(relation["whole"])
        other_parts = [str(value) for value in relation.get("other_parts", ())]
        parts = [part, *other_parts]
        max_excess = float(relation.get("max_excess", 0.0))
        if not 0.0 <= max_excess <= UK_SUBSET_MAX_EXCESS:
            raise ValueError(
                f"UK subset relation {part!r} in {whole!r} declares max_excess "
                f"{max_excess!r}; a rounding allowance sits in "
                f"[0, {UK_SUBSET_MAX_EXCESS}] (microcosm#1123)."
            )
        identity = {"part": part, "whole": whole}
        if other_parts:
            identity["other_parts"] = other_parts
        bound_national = [target_id for target_id in parts if target_id in national]
        bound_local = [target_id for target_id in parts if target_id in local_totals]
        if whole not in national and not bound_national and not bound_local:
            receipts.append(
                {**identity, "status": "skipped", "reason": "neither side is bound"}
            )
            continue
        if (
            whole in national
            and not bound_national
            and not bound_local
            and local_surface is None
            and all(target_id in local_target_ids for target_id in parts)
        ):
            receipts.append(
                {
                    **identity,
                    "status": "skipped",
                    "reason": "the part is local and this loader has no local surface",
                }
            )
            continue
        missing = [
            target_id
            for target_id in (whole, *parts)
            if target_id not in national and target_id not in local_totals
        ]
        if whole not in national or missing or (bound_national and bound_local):
            untested.append(
                f"{part} in {whole} (unbound here: {missing or [whole]})"
                if missing or whole not in national
                else f"{part} in {whole} (parts at both national and local grain)"
            )
            continue
        whole_cells = national[whole]
        whole_total = math.fsum(whole_cells.values())
        ceiling = whole_total * (1.0 + max_excess) * (1.0 + _CLOSURE_RTOL)
        if bound_national:
            part_cells: dict[str, float] = {}
            for target_id in parts:
                for geography_id, value in national[target_id].items():
                    part_cells[geography_id] = part_cells.get(geography_id, 0.0) + value
            totals = {"national": math.fsum(part_cells.values())}
            shared = sorted(set(part_cells) & set(whole_cells))
        else:
            grains = sorted(
                {grain for target_id in parts for grain in local_totals[target_id]}
            )
            totals = {
                grain: math.fsum(
                    local_totals[target_id].get(grain, 0.0) for target_id in parts
                )
                for grain in grains
            }
            part_cells, shared = {}, []
        for grain, part_total in sorted(totals.items()):
            holds = part_total <= ceiling
            over = [
                geography_id
                for geography_id in shared
                if part_cells[geography_id]
                > whole_cells[geography_id] * (1.0 + max_excess) * (1.0 + _CLOSURE_RTOL)
            ]
            receipts.append(
                {
                    **identity,
                    "grain": grain,
                    "part_total": part_total,
                    "whole_total": whole_total,
                    "max_excess": max_excess,
                    "geographies_tested": len(shared),
                    "status": "holds" if holds and not over else "breached",
                }
            )
            if not holds:
                breaches.append(
                    f"{part} ({grain}) sums to {part_total!r}, above {whole} at "
                    f"{whole_total!r}"
                )
            breaches.extend(
                f"{part} at {geography_id} is {part_cells[geography_id]!r}, above "
                f"{whole} there at {whole_cells[geography_id]!r}"
                for geography_id in over
            )
    if untested:
        raise ValueError(
            "UK subset relation(s) cannot be tested on this surface: "
            + "; ".join(untested)
            + ". A relation with one side unbound would stand untested: bind "
            "both sides or drop the declaration (microcosm#1123)."
        )
    if breaches:
        raise ValueError(
            "UK subset relation(s) breached on the compiled surface: "
            + "; ".join(breaches)
            + ". A part above its whole cannot be met with it: fix the "
            "declaration or reconcile the pair (microcosm#1123)."
        )
    return receipts


def _assert_national_closure(
    registry: TargetRegistry,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None,
    declarations: Mapping[str, Any] | None,
) -> None:
    """A second pass over the written-back register must move nothing."""

    specs = list(registry.specs)
    bound = {_contract_target_id(spec) for spec in specs}
    for sum_bridge in uk_sum_bridges(declarations):
        if bound.isdisjoint(
            (*sum_bridge.higher_target_ids, *sum_bridge.lower_target_ids)
        ):
            continue
        higher = math.fsum(
            float(spec.value)
            for spec in specs
            if _contract_target_id(spec) in sum_bridge.higher_target_ids
        )
        lower = math.fsum(
            float(spec.value)
            for spec in specs
            if _contract_target_id(spec) in sum_bridge.lower_target_ids
        )
        if not math.isclose(higher, lower, rel_tol=_CLOSURE_RTOL, abs_tol=0.0):
            raise ValueError(
                f"UK national register does not close on sum bridge "
                f"{sum_bridge.bridge_id!r}: {lower!r} against {higher!r}."
            )
    for bridge in uk_band_bridges(declarations):
        if bridge.higher_target_id not in bound and bound.isdisjoint(
            bridge.lower_target_ids
        ):
            continue
        higher = math.fsum(
            float(spec.value)
            for spec in specs
            if _contract_target_id(spec) == bridge.higher_target_id
            and _band_lower_bound(spec) in bridge.higher_band_lower_bounds
        )
        lower = math.fsum(
            float(spec.value)
            for spec in specs
            if _contract_target_id(spec) in bridge.lower_target_ids
        )
        if not math.isclose(higher, lower, rel_tol=_CLOSURE_RTOL, abs_tol=0.0):
            raise ValueError(
                f"UK national register does not close on band bridge "
                f"{bridge.bridge_id!r}: {lower!r} against {higher!r}."
            )


def assert_uk_national_rows_unmoved(cross_grain_receipt: Mapping[str, Any]) -> None:
    """The joint local pass must find the national register already closed.

    The national register reaches the joint surface reconciled
    (``reconcile_uk_national_registry``), so every group whose lower grain is
    a control grain (country, nation or region) must carry a factor of one.
    A factor away from one means the joint pass would rescale a national row
    the solve binds at another value: the two passes disagree.
    """

    control_grains = set(load_uk_cross_grain_declarations()["control_grains"])
    moved = [
        (str(group["inconsistency_id"]), str(leg["leg"]), float(leg["declared_factor"]))
        for group in cross_grain_receipt.get("groups", ())
        if group.get("lower_grain") in control_grains
        for leg in group["legs"]
        if not math.isclose(
            float(leg["declared_factor"]), 1.0, rel_tol=_CLOSURE_RTOL, abs_tol=0.0
        )
    ]
    if moved:
        raise ValueError(
            "UK joint target surface rescales national rows the national "
            f"reconciliation already closed: {moved[:10]!r}"
            f"{' ...' if len(moved) > 10 else ''} (microcosm#1123)."
        )
