"""Rake imputed household columns to vendored publisher facts (microcosm#890 D).

A ``rake_to_vendored_facts`` operation names the columns to rake and a list of
cells. Each cell is a set of FRS regions whose design-weighted total of the
column must equal one published FY2024-25 value read from a vendored Chronicle
resource (``selector``), optionally spread across household income quintiles
in proportion to the NTS trips per person by quintile (``allocation:
income_quintile_trips``) or uniformly (``allocation: uniform``). Cells are
disjoint by region; regions no cell names are left untouched (Wales fares,
whose publisher prints no receipts).

The arithmetic is the existing cell-mean iterative proportional fit: the cell
target mean is the published value times the cell's allocation share divided
by the design-weighted households in the raking scope of that cell, so one
pass makes the weighted total equal the fact. ``scope: users_only`` restricts
the population to households a caller-supplied flag marks as users (the
take-up style incidence draw); ``scope: all`` rakes every household in the
region set. A cell may name ``joint_columns``: the published value then binds
the sum of those columns and both are scaled by one factor (Northern Ireland's
bus-plus-rail support).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.raking import MarginSpec, iterative_proportional_fit
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

RAKE_TO_VENDORED_FACTS_KIND = "rake_to_vendored_facts"
ALLOCATION_UNIFORM = "uniform"
ALLOCATION_INCOME_QUINTILE_TRIPS = "income_quintile_trips"
SCOPE_USERS_ONLY = "users_only"
SCOPE_ALL = "all"
QUINTILE_LABELS = ("lowest", "second", "third", "fourth", "highest")
_CELL_COLUMN = "_vendored_fact_cell"
_WEIGHT_COLUMN = "_vendored_fact_weight"


@dataclass(frozen=True)
class FactCell:
    label: str
    regions: tuple[str, ...]
    value: float
    allocation: str
    quintile_trips: Mapping[str, float] | None
    joint_columns: tuple[str, ...]
    receipt: dict[str, Any]


def rake_operations(stage: SourceStageSpec) -> list[Mapping[str, Any]]:
    """Every declared ``rake_to_vendored_facts`` operation on the stage, in order."""

    return [
        dict(operation.parameters)
        for operation in stage.operations
        if operation.kind == RAKE_TO_VENDORED_FACTS_KIND
    ]


def resolve_cells(
    parameters: Mapping[str, Any], *, allowed_resources: Sequence[str]
) -> list[FactCell]:
    """Read every cell's published value (and allocation rows) from the vendored facts."""

    cells: list[FactCell] = []
    seen_regions: set[str] = set()
    for spec in parameters["cells"]:
        label = str(spec["label"])
        regions = tuple(str(region) for region in spec["regions"])
        if not regions:
            raise ValueError(f"rake cell {label!r} names no regions.")
        overlap = seen_regions & set(regions)
        if overlap:
            raise ValueError(f"rake cell {label!r} repeats regions {sorted(overlap)}.")
        seen_regions.update(regions)
        resource = str(spec["resource"])
        if resource not in allowed_resources:
            raise ValueError(
                f"rake cell {label!r} reads {resource!r}, which this stage does "
                f"not declare (allowed: {sorted(allowed_resources)})."
            )
        value, value_receipt = _published_value(resource, spec["selector"])
        allocation = str(spec["allocation"])
        quintile_trips = None
        allocation_receipt: dict[str, Any] = {"allocation": allocation}
        if allocation == ALLOCATION_INCOME_QUINTILE_TRIPS:
            allocation_resource = str(spec["allocation_resource"])
            if allocation_resource not in allowed_resources:
                raise ValueError(
                    f"rake cell {label!r} allocation reads {allocation_resource!r}, "
                    "which this stage does not declare."
                )
            quintile_trips, trips_receipt = _quintile_trips(
                allocation_resource,
                str(spec["allocation_concept"]),
                int(spec["allocation_period_value"]),
            )
            allocation_receipt.update(trips_receipt)
        elif allocation != ALLOCATION_UNIFORM:
            raise ValueError(
                f"rake cell {label!r} has unknown allocation {allocation!r}."
            )
        joint = tuple(str(column) for column in spec.get("joint_columns", ()))
        cells.append(
            FactCell(
                label=label,
                regions=regions,
                value=value,
                allocation=allocation,
                quintile_trips=quintile_trips,
                joint_columns=joint,
                receipt={
                    "label": label,
                    "regions": list(regions),
                    "resource": resource,
                    **value_receipt,
                    **allocation_receipt,
                    **({"joint_columns": list(joint)} if joint else {}),
                },
            )
        )
    return cells


def _published_value(
    resource: str, selector: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    criteria: dict[str, Any] = {"concept": str(selector["concept"])}
    for key in ("geography_id", "groupby_value_id"):
        if selector.get(key):
            criteria[key] = str(selector[key])
    if selector.get("dimension_values"):
        criteria["dimensions"] = {
            str(k): str(v) for k, v in dict(selector["dimension_values"]).items()
        }
    rows = vendored_rows(
        resource, fiscal_start=str(selector["fiscal_start"]), **criteria
    )
    sum_over = dict(selector.get("sum_over") or {})
    if sum_over:
        if len(sum_over) != 1:
            raise ValueError("sum_over names exactly one dimension.")
        ((dimension, values),) = sum_over.items()
        wanted = [str(v) for v in values]
        rows = [
            row
            for row in rows
            if str((row.get("dimensions") or {}).get(dimension)) in wanted
        ]
        found = sorted(
            str((row.get("dimensions") or {}).get(dimension)) for row in rows
        )
        if found != sorted(wanted):
            raise ValueError(
                f"{resource}: sum_over {dimension!r} expected {sorted(wanted)}, "
                f"found {found}."
            )
    elif len(rows) != 1:
        raise ValueError(
            f"{resource}: expected exactly one row for {selector}, found {len(rows)}."
        )
    value = float(sum(float(row["value"]) for row in rows))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{resource}: published value for {selector} must be positive."
        )
    units = {str(row.get("unit")) for row in rows}
    if units != {"gbp"}:
        raise ValueError(f"{resource}: expected gbp rows for {selector}, got {units}.")
    return value, {
        "selector": {k: v for k, v in selector.items()},
        "value": value,
        "source_record_ids": [str(row.get("source_record_id", "")) for row in rows],
    }


def _quintile_trips(
    resource: str, concept: str, period_value: int
) -> tuple[dict[str, float], dict[str, Any]]:
    trips: dict[str, float] = {}
    record_ids: list[str] = []
    for quintile in QUINTILE_LABELS:
        rows = vendored_rows(
            resource,
            concept=concept,
            period_type="calendar_year",
            period_value=period_value,
            dimensions={"household_income_quintile": quintile},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: expected one {concept} row for quintile {quintile!r} "
                f"in {period_value}, found {len(rows)}."
            )
        value = float(rows[0]["value"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{resource}: trips per person must be positive.")
        trips[quintile] = value
        record_ids.append(str(rows[0].get("source_record_id", "")))
    return trips, {
        "allocation_resource": resource,
        "allocation_concept": concept,
        "allocation_period_value": period_value,
        "trips_per_person_by_quintile": dict(trips),
        "allocation_source_record_ids": record_ids,
    }


def person_income_quintiles(
    household_income: np.ndarray,
    household_persons: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Design-weighted person quintiles of household income over ``mask`` rows.

    Returns each household's quintile label (empty string outside ``mask``) and
    the four internal edges. Persons, not households, are ranked (the NTS
    quintiles are of people by their household income), so a household's
    weight is its design weight times its size.
    """

    income = np.asarray(household_income, dtype=float)
    persons = np.asarray(household_persons, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    labels = np.full(len(income), "", dtype=object)
    if not mask.any():
        return labels, []
    order = np.argsort(income[mask], kind="stable")
    masked_income = income[mask][order]
    masked_weight = (weights[mask] * persons[mask])[order]
    total = float(masked_weight.sum())
    if total <= 0:
        raise ValueError("income quintiles need positive weighted persons.")
    cumulative = np.cumsum(masked_weight) / total
    edges = [
        float(
            masked_income[
                min(
                    int(np.searchsorted(cumulative, q, side="right")),
                    len(masked_income) - 1,
                )
            ]
        )
        for q in (0.2, 0.4, 0.6, 0.8)
    ]
    position = np.searchsorted(np.asarray(edges), income[mask], side="right")
    labels[mask] = np.asarray(QUINTILE_LABELS, dtype=object)[position]
    return labels, edges


def rake_to_facts(
    household: pd.DataFrame,
    *,
    columns: Sequence[str],
    cells: Sequence[FactCell],
    region: Sequence[str],
    weights: Sequence[float] | None,
    scope: str,
    users: Sequence[bool] | None = None,
    quintile_income: Sequence[float] | None = None,
    household_persons: Sequence[float] | None = None,
    iterations: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale ``columns`` so each cell's weighted total equals its published value."""

    if scope not in (SCOPE_USERS_ONLY, SCOPE_ALL):
        raise ValueError(f"unknown rake scope {scope!r}.")
    frame = household.copy()
    n = len(frame)
    region_values = np.asarray(region).astype(str)
    if len(region_values) != n:
        raise ValueError("region must align with the household table.")
    weight_values = (
        np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    )
    if scope == SCOPE_USERS_ONLY:
        if users is None:
            raise ValueError("scope users_only needs the user flag.")
        in_scope = np.asarray(users, dtype=bool)
    else:
        in_scope = np.ones(n, dtype=bool)
    needs_quintiles = any(
        cell.allocation == ALLOCATION_INCOME_QUINTILE_TRIPS for cell in cells
    )
    quintile_labels = np.full(n, "", dtype=object)
    quintile_edges: list[float] = []
    if needs_quintiles:
        if quintile_income is None or household_persons is None:
            raise ValueError(
                "income_quintile_trips allocation needs quintile_income and "
                "household_persons."
            )
        quintile_regions = {
            r
            for cell in cells
            if cell.allocation == ALLOCATION_INCOME_QUINTILE_TRIPS
            for r in cell.regions
        }
        quintile_labels, quintile_edges = person_income_quintiles(
            np.asarray(quintile_income, dtype=float),
            np.asarray(household_persons, dtype=float),
            weight_values,
            mask=np.isin(region_values, sorted(quintile_regions)),
        )
    cell_column = np.full(n, "", dtype=object)
    targets: dict[str, dict[str, float]] = {}
    cell_receipts: list[dict[str, Any]] = []
    joint_receipts: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []
    for cell in cells:
        member = np.isin(region_values, list(cell.regions))
        population = member & in_scope
        if not member.any():
            # No household of the frame lives in the cell's regions (a partial
            # or synthetic frame): the published mass has nowhere to go and the
            # cell is skipped, receipted, like an empty IPF cell. A populated
            # region with nobody in scope is refused below instead.
            skipped_cells.append(
                {
                    "cell": cell.label,
                    "regions": list(cell.regions),
                    "target_total": cell.value,
                    "reason": "no households in the cell's regions",
                }
            )
            continue
        if cell.joint_columns:
            frame, receipt = _rake_joint(
                frame, cell, population, weight_values, list(cell.joint_columns)
            )
            joint_receipts.append(receipt)
            continue
        if cell.allocation == ALLOCATION_UNIFORM:
            shares = {cell.label: 1.0}
            groups = {cell.label: population}
        else:
            assert cell.quintile_trips is not None
            persons = np.asarray(household_persons, dtype=float)
            weighted_persons = {
                q: float(
                    (weight_values * persons)[member & (quintile_labels == q)].sum()
                )
                for q in QUINTILE_LABELS
            }
            raw_shares = {
                q: weighted_persons[q] * cell.quintile_trips[q] for q in QUINTILE_LABELS
            }
            total_share = sum(raw_shares.values())
            if total_share <= 0:
                raise ValueError(
                    f"rake cell {cell.label!r} has no persons to allocate."
                )
            shares = {q: raw_shares[q] / total_share for q in QUINTILE_LABELS}
            groups = {q: population & (quintile_labels == q) for q in QUINTILE_LABELS}
        for key, group in groups.items():
            name = cell.label if key == cell.label else f"{cell.label}:{key}"
            cell_column[group] = name
            households = float(weight_values[group].sum())
            target_total = cell.value * shares[key]
            if households <= 0:
                if target_total > 0:
                    raise ValueError(
                        f"rake cell {name!r} carries GBP {target_total:,.0f} but "
                        "has no households in scope."
                    )
                continue
            target_mean = target_total / households
            targets[name] = {column: target_mean for column in columns}
            before = {
                column: float(
                    np.dot(
                        pd.to_numeric(frame[column], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float)[group],
                        weight_values[group],
                    )
                )
                for column in columns
            }
            cell_receipts.append(
                {
                    "cell": name,
                    "label": cell.label,
                    **({"quintile": key} if key != cell.label else {}),
                    "share": shares[key],
                    "target_total": target_total,
                    "weighted_households_in_scope": households,
                    "weighted_total_before": before,
                    "factor": {
                        column: (target_total / before[column])
                        if before[column] > 0
                        else None
                        for column in columns
                    },
                }
            )
    frame[_CELL_COLUMN] = cell_column
    frame[_WEIGHT_COLUMN] = weight_values
    if targets:
        frame = iterative_proportional_fit(
            frame,
            columns=tuple(columns),
            margins=(MarginSpec(_CELL_COLUMN, targets),),
            iterations=iterations,
            weight_column=_WEIGHT_COLUMN,
            fail_on_unattainable=True,
        )
    for entry in cell_receipts:
        group = frame[_CELL_COLUMN].to_numpy() == entry["cell"]
        entry["weighted_total_after"] = {
            column: float(
                np.dot(frame[column].to_numpy(dtype=float)[group], weight_values[group])
            )
            for column in columns
        }
    raked = frame.drop(columns=[_CELL_COLUMN, _WEIGHT_COLUMN])
    receipt = {
        "operation": RAKE_TO_VENDORED_FACTS_KIND,
        "columns": list(columns),
        "scope": scope,
        "iterations": int(iterations),
        "weighted": weights is not None,
        "households_in_scope": int(in_scope.sum()),
        "weighted_households_in_scope": float(weight_values[in_scope].sum()),
        "quintile_edges": quintile_edges,
        "cells": [cell.receipt for cell in cells],
        "fits": cell_receipts,
        "joint_fits": joint_receipts,
        "skipped_cells": skipped_cells,
    }
    return raked, receipt


def _rake_joint(
    frame: pd.DataFrame,
    cell: FactCell,
    population: np.ndarray,
    weights: np.ndarray,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cell.allocation != ALLOCATION_UNIFORM:
        raise ValueError(f"joint cell {cell.label!r} must allocate uniformly.")
    values = {
        column: pd.to_numeric(frame[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        for column in columns
    }
    before = {
        column: float(np.dot(values[column][population], weights[population]))
        for column in columns
    }
    current = sum(before.values())
    if current <= 0:
        raise ValueError(
            f"joint rake cell {cell.label!r} carries GBP {cell.value:,.0f} but the "
            f"columns {columns} sum to zero in scope."
        )
    factor = cell.value / current
    result = frame.copy()
    for column in columns:
        scaled = values[column].copy()
        scaled[population] *= factor
        result[column] = scaled
    return result, {
        "cell": cell.label,
        "label": cell.label,
        "columns": list(columns),
        "target_total": cell.value,
        "weighted_households_in_scope": float(weights[population].sum()),
        "weighted_total_before": before,
        "factor": factor,
        "weighted_total_after": {
            column: float(
                np.dot(
                    result[column].to_numpy(dtype=float)[population],
                    weights[population],
                )
            )
            for column in columns
        },
    }
