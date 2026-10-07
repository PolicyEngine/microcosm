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
)
from microcosm.calibrate import TargetRegistry, TargetSpec

_CLOSURE_RTOL = 1e-9


@dataclass(frozen=True)
class UKBandBridge:
    """Projected UK band cells that control a set of regional band targets."""

    bridge_id: str
    higher_target_id: str
    higher_band_lower_bounds: tuple[float, ...]
    lower_target_ids: tuple[str, ...]


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
        )
        for entry in declared.get("band_bridges", ())
    )


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
        "cross_grain": receipt,
        "fanout_targets_not_controls": sorted(fanout),
        "rows_moved_by_exact_signature": moved,
    }


def _assert_national_closure(
    registry: TargetRegistry,
    reviewed_unbound_higher_targets: Mapping[str, Mapping[str, object]] | None,
    declarations: Mapping[str, Any] | None,
) -> None:
    """A second pass over the written-back register must move nothing."""

    specs = list(registry.specs)
    bound = {_contract_target_id(spec) for spec in specs}
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
