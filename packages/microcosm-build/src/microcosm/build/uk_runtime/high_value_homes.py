"""The UK high-value homes family: derived facts, thresholds and the frame mask.

PolicyEngine/microcosm#1240. The committed resource
``uk/high_value_home_stock_facts.json`` carries counts of dwellings valued
above thresholds by area, computed by ``tools/build_uk_high_value_home_stock.py``
from HM Land Registry Price Paid Data (uprated with the UK HPI, placed through
the NSPL and the committed England-and-Wales atomic support). Chronicle
registers those inputs raw-only (PolicyEngine/chronicle#343) and the counts are
Microcosm's computation, so they ride in consumer-fact row shape under the
Microcosm-owned ``microcosm.derived_fact.`` key namespace and are merged beside
the pinned Chronicle feed when the local target surface compiles.

This module is the one place that knows the resource's shape: the loader
refuses a resource whose inputs drifted from the committed support provenance,
``uk_high_value_home_thresholds`` hands the frame-basis thresholds to the
metric computation, and ``high_value_home_mask`` is the owner-occupied,
above-threshold predicate shared by the local metrics and the release gates so
the two can never disagree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.tenure_constants import (
    UK_TENURE_OWNED_OUTRIGHT,
    UK_TENURE_OWNED_WITH_MORTGAGE,
    UK_TENURE_TYPE_TO_CATEGORY,
)

__all__ = [
    "DERIVED_FACT_KEY_PREFIX",
    "HIGH_VALUE_HOME_METRIC_PREFIX",
    "HighValueHomeThreshold",
    "UKHighValueHomeStockFacts",
    "high_value_home_mask",
    "uk_owner_occupied_mask",
    "load_uk_high_value_home_stock_facts",
    "load_vendored_obr_hvcts_facts",
    "uk_high_value_home_metric_names",
    "uk_high_value_home_thresholds",
]

DERIVED_FACT_KEY_PREFIX = "microcosm.derived_fact.high_value_home_stock.v1:"
HIGH_VALUE_HOME_METRIC_PREFIX = "high_value_homes/"
RESOURCE_KIND = "uk_derived_consumer_facts"
ROW_SCHEMA_VERSION = "microcosm.derived_consumer_fact.v1"
FACTS_RESOURCE = "high_value_home_stock_facts.json"
SUPPORT_PROVENANCE_RESOURCE = "uk_atomic_area_supports.provenance.json"
CROSSWALK_RESOURCE = "local_area_crosswalk.json"
_OWNER_TENURE_CATEGORIES = frozenset(
    {UK_TENURE_OWNED_OUTRIGHT, UK_TENURE_OWNED_WITH_MORTGAGE}
)
#: Threshold ids in the resource, in ascending value order; the metric names
#: derive from them so the frame and the facts cannot drift apart.
THRESHOLD_IDS: tuple[str, ...] = ("over_gbp_1m", "over_gbp_1_5m", "over_gbp_2m")
_METRIC_BY_THRESHOLD: Mapping[str, str] = {
    "over_gbp_1m": "high_value_homes/over_1m",
    "over_gbp_1_5m": "high_value_homes/over_1_5m",
    "over_gbp_2m": "high_value_homes/over_2m",
}
_LEVELS = frozenset({"constituency", "local_authority", "region", "country"})


@dataclass(frozen=True)
class HighValueHomeThreshold:
    """One threshold: its id, metric name, nominal value and frame-basis value."""

    threshold_id: str
    metric_name: str
    nominal_gbp: float
    basis_gbp: float


@dataclass(frozen=True)
class UKHighValueHomeStockFacts:
    """The validated derived-facts resource."""

    rows: tuple[Mapping[str, Any], ...]
    inputs_sha256: str
    rows_sha256: str
    parameters: Mapping[str, Any]
    inputs: Mapping[str, Any]
    generator: Mapping[str, Any]
    benchmarks: Mapping[str, Any] = field(default_factory=dict)

    @property
    def thresholds(self) -> tuple[HighValueHomeThreshold, ...]:
        nominal = self.parameters["thresholds_gbp_at_reference_month"]
        basis = self.parameters["frame_basis_thresholds"]
        return tuple(
            HighValueHomeThreshold(
                threshold_id=threshold_id,
                metric_name=_METRIC_BY_THRESHOLD[threshold_id],
                nominal_gbp=float(nominal[threshold_id]),
                basis_gbp=float(basis[threshold_id]),
            )
            for threshold_id in THRESHOLD_IDS
        )

    def rows_at(
        self, *, level: str, measure_id: str, threshold_id: str
    ) -> dict[str, float]:
        """Values by geography id for one level, measure and threshold."""
        return {
            str(row["geography"]["id"]): float(row["value"])
            for row in self.rows
            if row["geography"]["level"] == level
            and row["observed_measure"]["source_measure_id"] == measure_id
            and row["dimensions"]["microcosm.value_threshold"] == threshold_id
        }


def _package_text(name: str) -> str:
    return (
        resources.files("microcosm.build.uk").joinpath(name).read_text(encoding="utf-8")
    )


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_row(
    row: Mapping[str, Any], *, expected_vintages: Mapping[str, Any]
) -> None:
    key = str(row.get("aggregate_fact_key") or "")
    if not key.startswith(DERIVED_FACT_KEY_PREFIX):
        raise ValueError(
            f"high-value home fact {key!r} is outside the derived-fact namespace."
        )
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError(
            f"high-value home fact {key!r} has schema {row.get('schema_version')!r}."
        )
    geography = row.get("geography")
    if not isinstance(geography, Mapping) or geography.get("level") not in _LEVELS:
        raise ValueError(
            f"high-value home fact {key!r} has no supported geography level."
        )
    code = str(geography.get("id") or "")
    vintage = str(geography.get("vintage") or "")
    if not code or not vintage or not str(geography.get("name") or "").strip():
        raise ValueError(
            f"high-value home fact {key!r} lacks a geography id, vintage or name."
        )
    level = str(geography["level"])
    expected = expected_vintages.get(level)
    if isinstance(expected, Mapping):
        expected = expected.get(code[:1])
    if expected is not None:
        accepted = {expected} if isinstance(expected, str) else set(expected)
        if vintage not in accepted:
            raise ValueError(
                f"high-value home fact {key!r} at {code!r} carries vintage {vintage!r}; "
                f"the crosswalk accepts {sorted(accepted)} for level {level!r}."
            )
    value = row.get("value")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"high-value home fact {key!r} has invalid value {value!r}.")
    if not str(row.get("label") or "").strip():
        raise ValueError(f"high-value home fact {key!r} requires a label.")
    dimensions = row.get("dimensions") or {}
    if dimensions.get("microcosm.value_threshold") not in THRESHOLD_IDS:
        raise ValueError(f"high-value home fact {key!r} names an unknown threshold.")
    labels = row.get("dimension_labels") or {}
    value_labels = row.get("dimension_value_labels") or {}
    for dimension_id, value_id in dimensions.items():
        if not str(labels.get(dimension_id) or "").strip():
            raise ValueError(
                f"high-value home fact {key!r} lacks a label for {dimension_id!r}."
            )
        if not str(
            (value_labels.get(dimension_id) or {}).get(str(value_id)) or ""
        ).strip():
            raise ValueError(
                f"high-value home fact {key!r} lacks a value label for {dimension_id!r}={value_id!r}."
            )
    layout = row.get("layout") or {}
    if not str(layout.get("record_set_spec_id") or "").startswith(
        "uk.local_geography.high_value_homes."
    ):
        raise ValueError(f"high-value home fact {key!r} lacks its record set spec id.")
    if (row.get("aggregation") or {}).get("method") != "sum":
        raise ValueError(f"high-value home fact {key!r} must aggregate by sum.")


def load_uk_high_value_home_stock_facts(
    path: Path | None = None,
    *,
    support_provenance: Mapping[str, Any] | None = None,
    crosswalk: Mapping[str, Any] | None = None,
) -> UKHighValueHomeStockFacts:
    """Load and validate the derived-facts resource.

    The resource is refused when its ``rows_sha256`` does not match its rows,
    when the England-and-Wales support it placed dwellings through is not the
    committed one, or when a row is malformed or carries a boundary vintage
    the crosswalk does not accept for its level.
    """
    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else _package_text(FACTS_RESOURCE)
    )
    payload = json.loads(text)
    if payload.get("kind") != RESOURCE_KIND or payload.get("schema_version") != 1:
        raise ValueError(
            "high_value_home_stock_facts.json is not a schema-1 derived-facts resource."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("high_value_home_stock_facts.json carries no rows.")
    if payload.get("row_count") != len(rows):
        raise ValueError(
            "high_value_home_stock_facts.json row_count disagrees with its rows."
        )
    digest = _rows_sha256(rows)
    if digest != payload.get("rows_sha256"):
        raise ValueError(
            "high_value_home_stock_facts.json rows differ from their recorded digest."
        )
    provenance = support_provenance or json.loads(
        _package_text(SUPPORT_PROVENANCE_RESOURCE)
    )
    committed_support = provenance["supports"]["uk_ew_output_area_2021"]["sha256"]
    inputs = payload.get("inputs") or {}
    if inputs.get("support_sha256") != committed_support:
        raise ValueError(
            "high_value_home_stock_facts.json was built on an England-and-Wales support "
            "other than the committed one; rebuild it with tools/build_uk_high_value_home_stock.py."
        )
    levels = (crosswalk or json.loads(_package_text(CROSSWALK_RESOURCE)))["levels"]
    expected_vintages = {
        level: entry.get("expected_vintage") for level, entry in levels.items()
    }
    keys: set[str] = set()
    for row in rows:
        _validate_row(row, expected_vintages=expected_vintages)
        key = str(row["aggregate_fact_key"])
        if key in keys:
            raise ValueError(f"high-value home fact key {key!r} is duplicated.")
        keys.add(key)
    parameters = payload.get("parameters") or {}
    for required in (
        "thresholds_gbp_at_reference_month",
        "frame_basis_thresholds",
        "basis_year",
        "stock_date",
    ):
        if required not in parameters:
            raise ValueError(
                f"high_value_home_stock_facts.json lacks parameter {required!r}."
            )
    for threshold_id in THRESHOLD_IDS:
        if threshold_id not in parameters["frame_basis_thresholds"]:
            raise ValueError(
                f"high_value_home_stock_facts.json lacks the basis value for {threshold_id!r}."
            )
    return UKHighValueHomeStockFacts(
        rows=tuple(rows),
        inputs_sha256=str(payload.get("inputs_sha256") or ""),
        rows_sha256=digest,
        parameters=parameters,
        inputs=inputs,
        generator=payload.get("generator") or {},
        benchmarks=payload.get("benchmarks") or {},
    )


@cache
def _packaged_facts() -> UKHighValueHomeStockFacts:
    return load_uk_high_value_home_stock_facts()


def uk_high_value_home_thresholds() -> tuple[HighValueHomeThreshold, ...]:
    """The thresholds of the committed resource, ascending."""
    return _packaged_facts().thresholds


def uk_high_value_home_metric_names() -> tuple[str, ...]:
    """Local metric names of the family, in threshold order (append-only list member)."""
    return tuple(_METRIC_BY_THRESHOLD[threshold_id] for threshold_id in THRESHOLD_IDS)


def uk_owner_occupied_mask(tenure_type: np.ndarray) -> np.ndarray:
    """Households whose engine tenure key is an owner-occupier tenure."""
    categories = np.array(
        [
            UK_TENURE_TYPE_TO_CATEGORY.get(str(key), "")
            for key in np.asarray(tenure_type)
        ],
        dtype=object,
    )
    return np.isin(categories, list(_OWNER_TENURE_CATEGORIES))


def high_value_home_mask(
    main_residence_value: np.ndarray,
    tenure_type: np.ndarray,
    *,
    threshold_basis_gbp: float,
) -> np.ndarray:
    """Owner-occupied households whose stored home value exceeds the frame-basis threshold.

    ``tenure_type`` holds the engine's enum keys (``OWNED_OUTRIGHT`` and so on);
    ``main_residence_value`` is the stored value at the dataset period, never an
    engine-uprated one. Shared by the local metrics and the release gates.
    """
    values = np.asarray(main_residence_value, dtype=float)
    return uk_owner_occupied_mask(tenure_type) & (values > float(threshold_basis_gbp))


OBR_VENDORED_RESOURCE = "obr_hvcts_costing_facts.json"


def load_vendored_obr_hvcts_facts() -> tuple[Mapping[str, Any], ...] | None:
    """The OBR HVCTS costing facts vendored from the feed, or ``None`` before they are.

    ``uk/obr_hvcts_costing_facts.json`` carries the note's Table 1.1 and 1.2
    property counts and its non-owner-occupied share; the benchmarks gate reads
    them by concept, period and behaviour basis.
    """
    resource = resources.files("microcosm.build.uk").joinpath(OBR_VENDORED_RESOURCE)
    if not resource.is_file():
        return None
    payload = json.loads(resource.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{OBR_VENDORED_RESOURCE} carries no rows list.")
    return tuple(rows)
