"""UK target loss-weight rules derived from row metadata (microcosm#1124).

The UK local doctrine admits named weighting rules, never per-target vectors
(#762 A2). ``uniform`` and ``grain_equal`` need only each row's grain and stay
in :mod:`microcosm.build.uk_runtime.local_doctrine`. The rules here also read
the row's target family, its geography and its value basis, so this module
carries that metadata in one row-aligned carrier, :class:`UKTargetRows`. The
carrier is built the same way from the declared targets
(:func:`uk_target_rows_from_targets`, used by ``prepare_uk_full_solve``) and
from a stored ordered problem (:func:`uk_target_rows_from_problem`, used by
the size-experiment harness), so a rule computed either way is one vector.

Rules (G grains, F_g families in grain g, n rows in a (grain, family) cell;
every vector sums to one):

``grain_family_equal``
    Each grain (national, constituency, local authority) takes one equal
    share, each family inside a grain one equal share of it, rows inside a
    family equal shares: ``w = 1 / (G * F_g * n)``.
``nation_grain_family_equal``
    The same with a fourth grain, ``nation_region``: every national row whose
    geography is not one of the UK-wide aggregates
    (:data:`~microcosm.calibrate.geography_constants.UK_WIDE_GEOGRAPHY_IDS`:
    the United Kingdom, Great Britain, England and Wales) — the four nations
    and the English regions, one ITL1 tier — leaves the national grain for
    it. The four grains share equally.
``*_sqrt_count``
    Inside each (grain, family) cell the count rows share the cell's count
    budget in proportion to ``max(|value|, 1) ** 0.5`` (the US value-weight
    power, microcosm#1104); amount rows keep equal shares. Every grain, family
    and cell budget is unchanged: the modifier only moves weight from small
    count cells to large ones inside one family.

Local families are the census families of
:mod:`microcosm.build.uk_runtime.local_target_census`, not the Ledger source
families, which split council tax by nation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.local_doctrine import (
    UK_LOCAL_ROW_METADATA_RULES,
    UK_LOCAL_TARGET_WEIGHT_RULES,
    uk_local_target_loss_weights,
)
from microcosm.build.uk_runtime.local_targets import AREA_TYPES, area_groups_from_codes
from microcosm.calibrate.geography_constants import UK_WIDE_GEOGRAPHY_IDS

__all__ = [
    "NATIONAL_GRAIN",
    "NATION_REGION_GRAIN",
    "UK_TARGET_WEIGHT_RULE_VERSION",
    "UKStageTargetWeighting",
    "UKTargetRows",
    "uk_rule_loss_weights",
    "uk_target_loss_weight_receipt",
    "uk_target_loss_weights_for_rows",
    "uk_target_loss_weights_sha256",
    "uk_target_rows",
    "uk_target_rows_from_problem",
    "uk_target_rows_from_targets",
    "uk_target_value_basis",
]

NATIONAL_GRAIN = "national"
NATION_REGION_GRAIN = "nation_region"
UK_WIDE_LABEL = "UK-wide"
#: Bumped whenever a rule's arithmetic changes, so receipts stay comparable.
UK_TARGET_WEIGHT_RULE_VERSION = 1
LOCAL_SURFACE_MATERIALIZATION = "uk_local_surface"

_COUNT, _AMOUNT = "count", "amount"
_LOCAL_AMOUNT_METRIC_PREFIXES = ("rent/", "ons/equiv_")
_LOCAL_COUNT_METRIC_PREFIXES = ("age/", "council_tax/", "tenure/", "uc_")
_LOCAL_COUNT_METRICS = frozenset({"households"})
_NATIONAL_COUNT_UNITS = frozenset({"count"})
_NATIONAL_AMOUNT_UNIT_PREFIX = "gbp"


@dataclass(frozen=True)
class UKTargetRows:
    """The metadata a target-weight rule may read, aligned to the target axis.

    ``grain`` is ``"constituency"`` or ``"la"`` for local rows and
    ``"national"`` for every national row; ``family`` the census family
    (local) or the register family (national); ``geography`` the area code
    (local) or the national geography id; ``measure`` the local metric or
    the national ``ledger_measure_unit``; ``values`` the target values.
    """

    names: tuple[str, ...]
    local: tuple[bool, ...]
    grain: tuple[str, ...]
    family: tuple[str, ...]
    geography: tuple[str, ...]
    measure: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.names)
        for label in ("local", "grain", "family", "geography", "measure"):
            if len(getattr(self, label)) != n:
                raise ValueError(
                    f"target rows: {label} has {len(getattr(self, label))} "
                    f"entries for {n} names."
                )
        if len(set(self.names)) != n:
            raise ValueError("target rows: row names must be unique.")
        values = np.array(self.values, dtype=np.float64)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("target rows: values must be n finite numbers.")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        for name, local, grain, family in zip(
            self.names, self.local, self.grain, self.family, strict=True
        ):
            if local and grain not in AREA_TYPES:
                raise ValueError(
                    f"local target {name!r} has grain {grain!r}, expected one of "
                    f"{AREA_TYPES}."
                )
            if not local and grain != NATIONAL_GRAIN:
                raise ValueError(
                    f"national target {name!r} has grain {grain!r}, expected "
                    f"{NATIONAL_GRAIN!r}."
                )
            if not family:
                raise ValueError(f"target {name!r} declares no family.")

    def __len__(self) -> int:
        return len(self.names)

    def take(self, indices: Sequence[int] | np.ndarray) -> UKTargetRows:
        """The rows at ``indices`` (unique, in the given order)."""

        index = np.asarray(indices, dtype=np.int64)
        if index.ndim != 1 or len(np.unique(index)) != len(index):
            raise ValueError("target rows: take() needs unique row positions.")
        if len(index) and (index.min() < 0 or index.max() >= len(self)):
            raise ValueError("target rows: take() position out of range.")
        positions = index.tolist()
        return UKTargetRows(
            names=tuple(self.names[i] for i in positions),
            local=tuple(self.local[i] for i in positions),
            grain=tuple(self.grain[i] for i in positions),
            family=tuple(self.family[i] for i in positions),
            geography=tuple(self.geography[i] for i in positions),
            measure=tuple(self.measure[i] for i in positions),
            values=self.values[index],
        )


def uk_target_rows(
    names: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    values: Sequence[float] | np.ndarray,
    *,
    local: Sequence[bool],
) -> UKTargetRows:
    """Build the carrier from per-row metadata, the one core both builders use.

    A local row needs ``area_type`` (a local grain), ``area_code`` and
    ``metric``; its ``family`` falls back to the census classifier of its
    metric. A national row needs a ``family``; its geography is the local
    spelling ``geography_id`` or the Ledger ``ledger_geography_id``, its basis
    the ``ledger_measure_unit``.
    """

    names = tuple(str(name) for name in names)
    rows = tuple(metadata)
    flags = tuple(bool(flag) for flag in local)
    if not (len(names) == len(rows) == len(flags)):
        raise ValueError(
            "target rows: names, metadata and local flags must align "
            f"({len(names)}, {len(rows)}, {len(flags)})."
        )
    grain: list[str] = []
    family: list[str] = []
    geography: list[str] = []
    measure: list[str] = []
    for name, row, is_local in zip(names, rows, flags, strict=True):
        area_type = row.get("area_type")
        if is_local:
            if area_type not in AREA_TYPES:
                raise ValueError(
                    f"local target {name!r} has area_type {area_type!r}, expected "
                    f"one of {AREA_TYPES}."
                )
            metric = str(row.get("metric") or "")
            code = str(row.get("area_code") or "")
            if not metric or not code:
                raise ValueError(
                    f"local target {name!r} needs a metric and an area code."
                )
            declared = row.get("family")
            if not declared:
                from microcosm.build.uk_runtime import local_target_census

                declared = local_target_census.family_for_metric(metric)
            grain.append(str(area_type))
            family.append(str(declared))
            geography.append(code)
            measure.append(metric)
        else:
            if area_type in AREA_TYPES:
                raise ValueError(
                    f"national target {name!r} carries the local area_type "
                    f"{area_type!r}."
                )
            grain.append(NATIONAL_GRAIN)
            family.append(str(row.get("family") or ""))
            geography.append(
                str(row.get("geography_id") or row.get("ledger_geography_id") or "")
            )
            measure.append(str(row.get("ledger_measure_unit") or ""))
    return UKTargetRows(
        names=names,
        local=flags,
        grain=tuple(grain),
        family=tuple(family),
        geography=tuple(geography),
        measure=tuple(measure),
        values=np.asarray(values, dtype=np.float64),
    )


def uk_target_rows_from_targets(
    local_targets: Sequence[Any],
    national_specs: Sequence[Any] = (),
) -> UKTargetRows:
    """The carrier for a joint solve: local targets first, then national specs.

    ``local_targets`` are the rowwise surface's :class:`~microcosm.calibrate.Target`
    rows (their metadata names the area type, area code, metric and family);
    ``national_specs`` the national register's
    :class:`~microcosm.calibrate.TargetSpec` rows, in the solve's order.
    """

    local_targets = tuple(local_targets)
    national_specs = tuple(national_specs)
    names = [target.row_name for target in local_targets] + [
        spec.to_target().row_name for spec in national_specs
    ]
    metadata = [dict(target.metadata) for target in local_targets] + [
        {**spec.metadata, "family": spec.family} for spec in national_specs
    ]
    values = [float(target.value) for target in local_targets] + [
        float(spec.value) for spec in national_specs
    ]
    local = [True] * len(local_targets) + [False] * len(national_specs)
    return uk_target_rows(names, metadata, values, local=local)


def uk_target_rows_from_problem(problem: Any) -> UKTargetRows:
    """The carrier for a stored ordered problem (``calibrate.artifacts``).

    A row is local when its metadata says ``materialization ==
    "uk_local_surface"``; a row whose materialization and area type disagree
    is refused rather than guessed.
    """

    names = tuple(problem.problem.names)
    metadata = tuple(problem.target_metadata)
    local = []
    for name, row in zip(names, metadata, strict=True):
        is_local = row.get("materialization") == LOCAL_SURFACE_MATERIALIZATION
        if is_local != (row.get("area_type") in AREA_TYPES):
            raise ValueError(
                f"target {name!r}: materialization {row.get('materialization')!r} "
                f"disagrees with area_type {row.get('area_type')!r}."
            )
        local.append(is_local)
    return uk_target_rows(
        names,
        metadata,
        np.asarray(problem.problem.target_vector, dtype=np.float64),
        local=local,
    )


def _local_basis(metric: str) -> str | None:
    if metric.startswith("hmrc/"):
        if metric.endswith("/amount"):
            return _AMOUNT
        if metric.endswith("/count"):
            return _COUNT
        return None
    if metric.startswith(_LOCAL_AMOUNT_METRIC_PREFIXES):
        return _AMOUNT
    if metric in _LOCAL_COUNT_METRICS or metric.startswith(
        _LOCAL_COUNT_METRIC_PREFIXES
    ):
        return _COUNT
    return None


def _national_basis(unit: str) -> str | None:
    if unit in _NATIONAL_COUNT_UNITS:
        return _COUNT
    if unit.startswith(_NATIONAL_AMOUNT_UNIT_PREFIX):
        return _AMOUNT
    return None


def uk_target_value_basis(rows: UKTargetRows) -> tuple[str, ...]:
    """Each row's value basis, ``"count"`` or ``"amount"``, refusing unknowns.

    Local rows by their metric (amounts: ``hmrc/*/amount``, ``rent/*``,
    ``ons/equiv_*``; counts: ``hmrc/*/count``, ``age/*``, ``households``,
    ``uc_*``, ``council_tax/*``, ``tenure/*``), national rows by their
    ``ledger_measure_unit`` (``count``; ``gbp`` and its ``gbp_*`` variants).
    """

    bases = []
    for name, local, measure in zip(rows.names, rows.local, rows.measure, strict=True):
        basis = _local_basis(measure) if local else _national_basis(measure)
        if basis is None:
            kind = "metric" if local else "ledger_measure_unit"
            raise ValueError(
                f"target {name!r}: {kind} {measure!r} has no declared count or "
                "amount basis; the *_sqrt_count rules refuse an undeclared basis."
            )
        bases.append(basis)
    return tuple(bases)


def _rule_grains(rows: UKTargetRows, *, nation: bool) -> tuple[str, ...]:
    if not nation:
        return rows.grain
    grains = []
    for name, grain, geography in zip(
        rows.names, rows.grain, rows.geography, strict=True
    ):
        if grain != NATIONAL_GRAIN:
            grains.append(grain)
            continue
        if not geography:
            raise ValueError(
                f"national target {name!r} names no geography; the nation grain "
                "needs one to place it."
            )
        grains.append(
            NATIONAL_GRAIN
            if geography in UK_WIDE_GEOGRAPHY_IDS
            else NATION_REGION_GRAIN
        )
    return tuple(grains)


def _check_row_rule(rule: str) -> None:
    if rule not in UK_LOCAL_ROW_METADATA_RULES:
        raise ValueError(
            f"row-metadata target_weight_rule must be one of "
            f"{UK_LOCAL_ROW_METADATA_RULES}, got {rule!r}."
        )


def uk_target_loss_weights_for_rows(rows: UKTargetRows, *, rule: str) -> np.ndarray:
    """The loss-weight vector of a row-metadata rule (sums to one)."""

    _check_row_rule(rule)
    if not len(rows):
        raise ValueError("target weighting requires at least one target.")
    nation = rule.startswith("nation_")
    sqrt_count = rule.endswith("_sqrt_count")
    grains = np.asarray(_rule_grains(rows, nation=nation), dtype=object)
    families = np.asarray(rows.family, dtype=object)
    bases = (
        np.asarray(uk_target_value_basis(rows), dtype=object) if sqrt_count else None
    )
    magnitudes = np.maximum(np.abs(rows.values), 1.0) ** 0.5 if sqrt_count else None
    grain_order = list(dict.fromkeys(grains.tolist()))
    weights = np.zeros(len(rows), dtype=np.float64)
    for grain in grain_order:
        in_grain = grains == grain
        family_order = list(dict.fromkeys(families[in_grain].tolist()))
        budget = 1.0 / (len(grain_order) * len(family_order))
        for family in family_order:
            cell = np.flatnonzero(in_grain & (families == family))
            if bases is None:
                weights[cell] = budget / cell.size
                continue
            counts = cell[bases[cell] == _COUNT]
            amounts = cell[bases[cell] == _AMOUNT]
            weights[amounts] = budget / cell.size
            if counts.size:
                share = magnitudes[counts]
                weights[counts] = (
                    (budget * counts.size / cell.size) * share / share.sum()
                )
    return weights


def uk_rule_loss_weights(rows: UKTargetRows, *, rule: str) -> np.ndarray:
    """The loss-weight vector of any rule in the local vocabulary.

    ``uniform`` is ones (the vector a graph problem binds for it);
    ``grain_equal`` is :func:`~microcosm.build.uk_runtime.local_doctrine.uk_local_target_loss_weights`
    on the rows' grains, bit for bit the vector ``prepare_uk_full_solve``
    binds; the row-metadata rules come from
    :func:`uk_target_loss_weights_for_rows`.
    """

    if rule not in UK_LOCAL_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {UK_LOCAL_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    if rule == "uniform":
        return np.ones(len(rows), dtype=np.float64)
    if rule == "grain_equal":
        return np.asarray(
            uk_local_target_loss_weights(list(rows.grain), rule="grain_equal"),
            dtype=np.float64,
        )
    return uk_target_loss_weights_for_rows(rows, rule=rule)


def uk_target_loss_weights_sha256(names: Sequence[str], weights: np.ndarray) -> str:
    """Content digest of a row-aligned loss-weight vector.

    The canonical form of the US national release's ``loss_vector_sha256``
    (``us_runtime.target_loss_weights.target_loss_weights_sha256``): one
    ``{"row_name", "weight_hex"}`` object per row, in row order, as compact
    sorted-key JSON. Any change to a name, a weight bit or the order changes it.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(names),):
        raise ValueError(
            f"{weights.shape} loss weights do not align with {len(names)} names."
        )
    rows = [
        {"row_name": str(name), "weight_hex": float(weight).hex()}
        for name, weight in zip(names, weights, strict=True)
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _nations(geographies: Sequence[str]) -> tuple[str, ...]:
    """Each row's nation: UK-wide aggregates, else the area-code prefix rule."""

    coded = sorted(
        {code for code in geographies if code and code not in UK_WIDE_GEOGRAPHY_IDS}
    )
    groups = area_groups_from_codes(coded) if coded else {}
    return tuple(
        UK_WIDE_LABEL if code in UK_WIDE_GEOGRAPHY_IDS else groups.get(code, "unknown")
        for code in geographies
    )


def uk_target_loss_weight_receipt(
    rows: UKTargetRows,
    weights: np.ndarray,
    *,
    rule: str,
) -> dict[str, object]:
    """Loss shares of a weight vector by rule grain, grain × family and nation.

    ``loss_share`` is a cell's share of the total weight (the share of the
    loss it carries at equal misses); ``equal_share`` what it would carry
    with every target weighted equally.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(rows),):
        raise ValueError(
            f"{weights.shape} loss weights do not align with {len(rows)} rows."
        )
    total = float(weights.sum())
    n = len(rows)
    nation_rule = rule in UK_LOCAL_ROW_METADATA_RULES and rule.startswith("nation_")
    grains = np.asarray(_rule_grains(rows, nation=nation_rule), dtype=object)
    families = np.asarray(rows.family, dtype=object)
    nations = np.asarray(_nations(rows.geography), dtype=object)
    sqrt_count = rule in UK_LOCAL_ROW_METADATA_RULES and rule.endswith("_sqrt_count")
    bases = (
        np.asarray(uk_target_value_basis(rows), dtype=object) if sqrt_count else None
    )

    def cell(mask: np.ndarray) -> dict[str, object]:
        selected = weights[mask]
        block: dict[str, object] = {
            "n_targets": int(mask.sum()),
            "loss_share": float(selected.sum()) / total if total else 0.0,
            "equal_share": float(mask.sum()) / n if n else 0.0,
            "min_weight": float(selected.min()) if selected.size else None,
            "max_weight": float(selected.max()) if selected.size else None,
        }
        if bases is not None:
            block["n_count"] = int((bases[mask] == _COUNT).sum())
            block["n_amount"] = int((bases[mask] == _AMOUNT).sum())
        return block

    grain_order = list(dict.fromkeys(grains.tolist()))
    cells = []
    for grain in grain_order:
        for family in dict.fromkeys(families[grains == grain].tolist()):
            cells.append(
                {
                    "grain": grain,
                    "family": family,
                    **cell((grains == grain) & (families == family)),
                }
            )
    return {
        "rule": rule,
        "rule_version": UK_TARGET_WEIGHT_RULE_VERSION,
        "n_targets": n,
        "weights_sha256": uk_target_loss_weights_sha256(rows.names, weights),
        "grains": [{"grain": grain, **cell(grains == grain)} for grain in grain_order],
        "cells": cells,
        "nations": [
            {"nation": nation, **cell(nations == nation)}
            for nation in sorted(set(nations.tolist()))
        ],
    }


@dataclass(frozen=True)
class UKStageTargetWeighting:
    """One stage's loss weights, computed from a named rule (the size seam).

    The dataset-size search and refit inherit the dense solve's weights
    unless a caller hands them one of these: the rule is computed on
    :attr:`rows`, which must be the stage problem's own row axis (checked by
    name). ``held_out`` rows get zero weight and the rule is applied to the
    remaining rows, as the production holdout re-applies the rule to its
    training problem. Never a raw vector: a per-target vector cannot enter.
    """

    rule: str
    rows: UKTargetRows
    held_out: tuple[int, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        if self.rule not in UK_LOCAL_TARGET_WEIGHT_RULES:
            raise ValueError(
                f"target_weight_rule must be one of {UK_LOCAL_TARGET_WEIGHT_RULES}, "
                f"got {self.rule!r}."
            )
        if not isinstance(self.rows, UKTargetRows):
            raise TypeError("stage target weighting needs a UKTargetRows carrier.")
        held = tuple(sorted(int(index) for index in self.held_out))
        if len(set(held)) != len(held):
            raise ValueError("held-out rows must be unique.")
        if held and (held[0] < 0 or held[-1] >= len(self.rows)):
            raise ValueError("held-out row position out of range.")
        if len(held) >= len(self.rows):
            raise ValueError("a stage weighting must train on at least one row.")
        object.__setattr__(self, "held_out", held)

    def training_positions(self) -> np.ndarray:
        return np.setdiff1d(
            np.arange(len(self.rows), dtype=np.int64),
            np.asarray(self.held_out, dtype=np.int64),
            assume_unique=True,
        )

    def _require_axis(self, names: Sequence[str]) -> None:
        if tuple(str(name) for name in names) != self.rows.names:
            raise ValueError(
                "stage target weighting was built for another row axis: the "
                "stage problem's target names differ from the carrier's."
            )

    def loss_weights(self, names: Sequence[str]) -> np.ndarray:
        """The full-length training weight vector for the problem rows ``names``."""

        self._require_axis(names)
        if not self.held_out:
            return uk_rule_loss_weights(self.rows, rule=self.rule)
        training = self.training_positions()
        weights = np.zeros(len(self.rows), dtype=np.float64)
        weights[training] = uk_rule_loss_weights(
            self.rows.take(training), rule=self.rule
        )
        return weights

    def held_out_weights(self) -> np.ndarray:
        """The rule applied to the held-out rows alone (their scoring weights)."""

        if not self.held_out:
            raise ValueError("no rows are held out.")
        return uk_rule_loss_weights(
            self.rows.take(np.asarray(self.held_out, dtype=np.int64)), rule=self.rule
        )

    def receipt(self, names: Sequence[str]) -> dict[str, object]:
        weights = self.loss_weights(names)
        return {
            "rule": self.rule,
            "rule_version": UK_TARGET_WEIGHT_RULE_VERSION,
            "label": self.label,
            "n_targets": len(self.rows),
            "n_held_out": len(self.held_out),
            "weights_sha256": uk_target_loss_weights_sha256(self.rows.names, weights),
        }
