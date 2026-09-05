"""Compute aggregate-only dashboard evidence from a validated population Frame.

The graph driver supplies the identity of the population artifact it loaded.
All counts, masses and distributions are computed here; callers cannot supply
precomputed statistics. Definitions name actual Frame columns, their owning
entities, units and fixed bins. This module neither simulates variables nor
chooses a representative person to assign a group's geography.

Frame.resolve_weights is the weight authority, including inherited weights.
Grouped accumulation uses one row-code vector at a time, never a Frame or a
row-by-bin matrix per geography. Every declared geography vintage independently
partitions national mass. Missing geography requires a declared unknown bucket.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from microcosm.frame import Frame, Weights

__all__ = [
    "DistributionBin",
    "DistributionSpec",
    "GeographySpec",
    "population_distributions",
    "population_support",
]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without outer whitespace.")
    return value


@dataclass(frozen=True)
class GeographySpec:
    """One complete geographic partition, with producer-declared labels/vintage.

    Levels are canonical lowercase identifiers (letters/digits separated by
    single underscores), declared by the producer rather than a country enum.
    A level names a partition; it does not establish country, source authority,
    nesting across levels/vintages, or certification for local analysis.
    National has no column and exactly one identity. Other levels name an actual
    column. Integral source values use unpadded decimal ids (6 -> "6"); string
    ids retain their spelling. Null values use unknown_id only when declared.
    A non-null unassigned code must itself appear in labels. Empty and entirely
    zero-weight declared areas remain visible in the output.
    """

    level: str
    column: str | None
    vintage: str
    labels: Mapping[str, str]
    unknown_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, str) or not re.fullmatch(
            r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", self.level
        ):
            raise ValueError(
                "Geography level must be a canonical lowercase identifier with "
                "letters/digits separated by single underscores."
            )
        _text(self.vintage, "Geography vintage")
        if not isinstance(self.labels, Mapping) or not self.labels:
            raise ValueError("Geography labels must explicitly name every area.")
        labels = {
            _text(key, "Geography id"): _text(label, "Geography label")
            for key, label in self.labels.items()
        }
        if self.level == "national":
            if (
                self.column is not None
                or len(labels) != 1
                or self.unknown_id is not None
            ):
                raise ValueError(
                    "A national geography has no column and exactly one id."
                )
        else:
            _text(self.column, "Geography column")
            if self.unknown_id is not None:
                _text(self.unknown_id, "Geography unknown_id")
                if self.unknown_id not in labels:
                    raise ValueError(
                        "Geography unknown_id must have an explicit label."
                    )
        object.__setattr__(self, "labels", MappingProxyType(labels))


@dataclass(frozen=True)
class DistributionBin:
    """A [lower, upper) bin; the final bin also includes its upper bound.

    None means negative infinity for lower and positive infinity for upper.
    Infinite observations are invalid data, even in an open-ended bin.
    """

    label: str
    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        _text(self.label, "Bin label")
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                ):
                    raise ValueError("Bin boundaries must be finite numbers or None.")
                object.__setattr__(self, name, float(value))
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError("A bin's lower bound must be below its upper bound.")


@dataclass(frozen=True)
class DistributionSpec:
    """A producer-owned variable/entity/unit definition and fixed bin scheme."""

    variable: str
    label: str
    entity: str
    unit: str
    bin_definition_id: str
    bins: tuple[DistributionBin, ...]

    def __post_init__(self) -> None:
        for name in ("variable", "label", "entity", "unit", "bin_definition_id"):
            _text(getattr(self, name), f"Distribution {name}")
        bins = tuple(self.bins)
        if not bins or any(not isinstance(item, DistributionBin) for item in bins):
            raise ValueError("Distribution bins must be explicitly declared.")
        for index, item in enumerate(bins):
            if (
                (item.lower is None and index != 0)
                or (item.upper is None and index != len(bins) - 1)
                or (index and item.lower != bins[index - 1].upper)
            ):
                raise ValueError("Distribution bins must be ordered and contiguous.")
        object.__setattr__(self, "bins", bins)


def _identity(candidate_release_id: str, population_sha256: str, period: str) -> dict:
    _text(candidate_release_id, "Candidate release id")
    _text(period, "Period")
    if not isinstance(population_sha256, str) or not re.fullmatch(
        "[a-f0-9]{64}", population_sha256
    ):
        raise ValueError("Population digest must be a lowercase SHA-256 hex string.")
    return {
        "candidate_release_id": candidate_release_id,
        "population_sha256": population_sha256,
        "period": period,
    }


def _definitions(specs: Iterable[GeographySpec]) -> tuple[GeographySpec, ...]:
    definitions = tuple(specs)
    if any(not isinstance(spec, GeographySpec) for spec in definitions):
        raise TypeError("geography_specs must contain GeographySpec definitions.")
    if sum(spec.level == "national" for spec in definitions) != 1:
        raise ValueError("Exactly one explicit national geography is required.")
    keys = [(spec.level, spec.vintage) for spec in definitions]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate geography level/vintage definitions.")
    return definitions


def _resolved(frame: Frame, entity: str) -> tuple[Weights, str, str]:
    weights = frame.resolve_weights(entity)
    if not math.isfinite(weights.total):
        raise ValueError("Resolved weight total is not finite.")
    # Identify provenance only after the canonical resolver has accepted the
    # vector. Values and kind always come from Frame, never weight columns.
    person = frame.schema.person_entity
    if entity in frame.weighted_entities:
        source, path = entity, entity
    elif entity != person and person in frame.weighted_entities:
        source, path = person, f"{person}->{entity}"
    else:
        source = next(
            group
            for group in frame.schema.group_entities
            if group in frame.weighted_entities
        )
        path = f"{source}->{person}" + (f"->{entity}" if entity != person else "")
    return weights, source, f"{weights.kind.value}:{path}"


def _geography_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("Boolean values are not geography ids.")
    if isinstance(value, str):
        return _text(value, "Source geography id")
    if (
        isinstance(value, int)
        or isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
    ):
        return str(int(value))
    raise ValueError(f"Invalid source geography id {value!r}.")


def _partitions(
    frame: Frame, entity: str, definitions: tuple[GeographySpec, ...], length: int
):
    for spec in definitions:
        ids = sorted(spec.labels)
        geographies = [
            {
                "level": spec.level,
                "id": identifier,
                "label": spec.labels[identifier],
                "vintage": spec.vintage,
            }
            for identifier in ids
        ]
        if spec.level == "national":
            yield geographies, np.zeros(length, dtype=np.int64)
            continue
        owner = frame.column_entity(spec.column)
        if owner == entity:
            series = frame.table(entity)[spec.column]
        elif entity == frame.schema.person_entity:
            series = frame.broadcast(spec.column, to=entity)
        else:
            raise ValueError(
                f"Cannot align geography {spec.column!r} on {owner!r} to entity {entity!r}."
            )
        keys = series.map(_geography_id)
        if keys.isna().any():
            if spec.unknown_id is None:
                raise ValueError(
                    f"Found missing {spec.level} geography without a declared unknown bucket."
                )
            keys = keys.fillna(spec.unknown_id)
        codes = keys.map({identifier: index for index, identifier in enumerate(ids)})
        if codes.isna().any():
            unknown = keys.loc[codes.isna()].unique()[:3].tolist()
            raise ValueError(f"Undeclared {spec.level} geography ids {unknown!r}.")
        yield geographies, codes.to_numpy(dtype=np.int64)


def _masses(codes: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    totals = np.bincount(codes, weights=weights, minlength=size)
    if not np.isfinite(totals).all():
        raise ValueError("Aggregated weight mass is not finite.")
    return totals


def _reconcile(totals: np.ndarray, expected: float) -> None:
    if not math.isclose(math.fsum(totals), expected, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(
            "Geographic partition does not reconcile with national weight mass."
        )


def population_support(
    frame: Frame,
    *,
    candidate_release_id: str,
    population_sha256: str,
    period: str,
    entity: str,
    geography_specs: Iterable[GeographySpec],
) -> dict:
    """Return support/concentration evidence for one entity and all vintages.

    ESS is Kish's (sum w)^2 / sum(w^2), calculated after per-area rescaling to
    avoid squaring large weights. It describes weight concentration, not a
    survey-design variance estimate. Empty or zero-mass areas have null ESS and
    maximum share. population_sha256 identifies the driver's source artifact;
    this function does not hash an alternative serialization of the Frame.
    """
    identity = _identity(candidate_release_id, population_sha256, period)
    definitions = _definitions(geography_specs)
    frame.revalidate()
    typed, source, basis = _resolved(frame, entity)
    weights = typed.values
    rows = []
    for geographies, codes in _partitions(frame, entity, definitions, len(weights)):
        size = len(geographies)
        totals = _masses(codes, weights, size)
        _reconcile(totals, typed.total)
        counts = np.bincount(codes, minlength=size)
        positive = np.bincount(codes[weights > 0], minlength=size)
        maxima = np.zeros(size)
        np.maximum.at(maxima, codes, weights)
        row_maxima = maxima[codes]
        scaled = np.divide(
            weights, row_maxima, out=np.zeros_like(weights), where=row_maxima > 0
        )
        scaled_sum = _masses(codes, scaled, size)
        squared_sum = _masses(codes, np.square(scaled), size)
        for index, geography in enumerate(geographies):
            mass = float(totals[index])
            rows.append(
                {
                    "geography": geography,
                    "record_count": int(counts[index]),
                    "positive_weight_count": int(positive[index]),
                    "weight_sum": mass,
                    "effective_sample_size": float(
                        scaled_sum[index] ** 2 / squared_sum[index]
                    )
                    if mass
                    else None,
                    "max_weight_share": float(maxima[index] / mass) if mass else None,
                }
            )
    return {
        "schema_version": "microcosm.population_support.v1",
        **identity,
        "entity": entity,
        "weight_entity": source,
        "weight_basis": basis,
        "rows": rows,
    }


def _bin_codes(frame: Frame, spec: DistributionSpec) -> tuple[np.ndarray, np.ndarray]:
    owner = frame.column_entity(spec.variable)
    if owner != spec.entity:
        raise ValueError(
            f"Distribution {spec.variable!r} belongs to entity {owner!r}, not {spec.entity!r}."
        )
    series = frame.table(owner)[spec.variable]
    if is_complex_dtype(series.dtype) or not (
        is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)
    ):
        raise ValueError(
            f"Distribution {spec.variable!r} must be real numeric or boolean."
        )
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"Distribution {spec.variable!r} contains infinite values.")
    present = ~np.isnan(values)
    lower = -math.inf if spec.bins[0].lower is None else spec.bins[0].lower
    upper = math.inf if spec.bins[-1].upper is None else spec.bins[-1].upper
    if ((values[present] < lower) | (values[present] > upper)).any():
        raise ValueError(
            f"Declared bins do not cover every observed {spec.variable!r} value."
        )
    uppers = np.array(
        [math.inf if item.upper is None else item.upper for item in spec.bins]
    )
    codes = np.searchsorted(uppers, values, side="right")
    # The final upper boundary is closed, unlike intermediate boundaries.
    codes = np.minimum(codes, len(spec.bins) - 1)
    return codes, present


def population_distributions(
    frame: Frame,
    *,
    candidate_release_id: str,
    population_sha256: str,
    period: str,
    distributions: Iterable[DistributionSpec],
    geography_specs: Iterable[GeographySpec],
) -> dict:
    """Return fixed-bin weighted distributions including all missing mass.

    Shares divide by total weight, including missing values. Non-exhaustive
    bins, missing/unknown geography and entity mismatches fail explicitly;
    rows are never silently dropped. Each declaration owns its source column,
    entity, unit and bin-definition id. No unit conversion is inferred.
    """
    identity = _identity(candidate_release_id, population_sha256, period)
    definitions = _definitions(geography_specs)
    specs = tuple(distributions)
    if not specs or any(not isinstance(spec, DistributionSpec) for spec in specs):
        raise ValueError("At least one DistributionSpec is required.")
    keys = [
        (
            spec.variable,
            spec.entity,
            spec.unit,
            spec.bin_definition_id,
            tuple((item.lower, item.upper) for item in spec.bins),
        )
        for spec in specs
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate distribution definitions.")
    frame.revalidate()
    output = []
    for spec in specs:
        typed, _, basis = _resolved(frame, spec.entity)
        weights = typed.values
        bin_codes, present = _bin_codes(frame, spec)
        n_bins = len(spec.bins)
        for geographies, codes in _partitions(
            frame, spec.entity, definitions, len(weights)
        ):
            size = len(geographies)
            totals = _masses(codes, weights, size)
            _reconcile(totals, typed.total)
            missing = _masses(codes[~present], weights[~present], size)
            flat_codes = codes[present] * n_bins + bin_codes[present]
            masses = _masses(flat_codes, weights[present], size * n_bins).reshape(
                size, n_bins
            )
            for index, geography in enumerate(geographies):
                total = float(totals[index])
                _reconcile(np.append(masses[index], missing[index]), total)
                output.append(
                    {
                        "variable": spec.variable,
                        "label": spec.label,
                        "entity": spec.entity,
                        "unit": spec.unit,
                        "weight_basis": basis,
                        "bin_definition_id": spec.bin_definition_id,
                        "geography": geography,
                        "bins": [
                            {
                                "label": item.label,
                                "lower": item.lower,
                                "upper": item.upper,
                                "weight": float(mass),
                                "share": float(mass / total) if total else None,
                            }
                            for item, mass in zip(spec.bins, masses[index], strict=True)
                        ],
                        "missing_weight": float(missing[index]),
                        "total_weight": total,
                    }
                )
    return {
        "schema_version": "microcosm.population_distributions.v1",
        **identity,
        "series": output,
    }
