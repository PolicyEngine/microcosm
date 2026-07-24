"""Versioned aging for raw IRS PUF tax-unit columns.

The production contract is intentionally factor-source agnostic: callers pass
an immutable bundle whose source year, target year, fact identifiers, and
source-artifact digest are all explicit. The engine scales only raw columns,
handles positive and negative factors separately, rejects duplicate ownership,
and leaves PolicyEngine-aligned derivations to the following source stage.

An ``archived_1_8_0`` bundle is packaged for audit parity. It reproduces the
retired 2015-to-2021 pass, including the accidental double aging of ``E00900``
and ``E26270``. The release's intended 2021-to-2024 processed-array pass was a
no-op, so this audit profile must not be mistaken for a corrected production
aging policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd
import yaml

__all__ = [
    "PUF_AGING_ARCHIVED_IMPLEMENTATION_URL",
    "PUF_AGING_ARCHIVED_PROFILE_RESOURCE",
    "PUF_AGING_ARCHIVED_PROFILE_VERSION",
    "PUF_AGING_ARCHIVED_TARGETS_URL",
    "PufAgingColumnFactor",
    "PufAgingFactorBundle",
    "PufAgingProvenance",
    "PufAgingSignedFactor",
    "age_raw_puf",
    "load_archived_puf_aging_factors",
    "puf_aging_factors_from_mapping",
    "puf_aging_provenance",
]

_RELEASE_COMMIT = "371f77a0aadfdeacd5856e0a3030c2db0eda65b5"
_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_RELEASE_ROOT = (
    "https://github.com/PolicyEngine/policyengine-us-data/blob/"
    f"{_RELEASE_COMMIT}/policyengine_us_data/"
)
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/policyengine-us-data/blob/"
    f"{_ARCHIVED_COMMIT}/policyengine_us_data/"
)
PUF_AGING_ARCHIVED_IMPLEMENTATION_URL = (
    _RELEASE_ROOT + "datasets/puf/uprate_puf.py#L1-L174"
)
PUF_AGING_ARCHIVED_TARGETS_URL = (
    _ARCHIVED_ROOT + "storage/calibration_targets/soi_targets.csv"
)
PUF_AGING_ARCHIVED_PROFILE_VERSION = "archived_1_8_0"
PUF_AGING_ARCHIVED_PROFILE_RESOURCE = "puf_aging_archived_1_8_0.yaml"
_PROVENANCE_ATTR = "populace_puf_aging"


@dataclass(frozen=True)
class PufAgingProvenance:
    """Coordinates for the facts used to construct one factor bundle."""

    source_kind: str
    source_artifact_sha256: str
    source_coordinates: tuple[str, ...]
    notes: str

    def validate(self) -> None:
        """Reject incomplete or mutable factor provenance."""

        if not self.source_kind:
            raise ValueError("PUF aging provenance requires source_kind.")
        digest = self.source_artifact_sha256
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "PUF aging provenance requires a lowercase SHA-256 digest."
            )
        if not self.source_coordinates or any(
            not coordinate for coordinate in self.source_coordinates
        ):
            raise ValueError("PUF aging provenance requires source coordinates.")
        if not self.notes:
            raise ValueError("PUF aging provenance requires explanatory notes.")


@dataclass(frozen=True)
class PufAgingColumnFactor:
    """One factor applied to all finite values in a raw PUF column."""

    column: str
    factor: float
    fact_ids: tuple[str, ...] = ()
    fallback_reason: str | None = None

    def validate(self, *, label: str) -> None:
        """Validate one straight or weight factor."""

        _validate_column_name(self.column, label=label)
        _validate_factor(self.factor, label=f"{label} {self.column!r}")
        _validate_factor_evidence(
            self.fact_ids,
            self.fallback_reason,
            label=f"{label} {self.column!r}",
        )


@dataclass(frozen=True)
class PufAgingSignedFactor:
    """Separate factors for positive and negative values in one raw column."""

    column: str
    positive_factor: float
    negative_factor: float
    fact_ids: tuple[str, ...] = ()
    fallback_reason: str | None = None

    def validate(self) -> None:
        """Validate a signed factor declaration."""

        _validate_column_name(self.column, label="signed factor")
        _validate_factor(
            self.positive_factor,
            label=f"signed positive factor {self.column!r}",
        )
        _validate_factor(
            self.negative_factor,
            label=f"signed negative factor {self.column!r}",
        )
        _validate_factor_evidence(
            self.fact_ids,
            self.fallback_reason,
            label=f"signed factor {self.column!r}",
        )


@dataclass(frozen=True)
class PufAgingFactorBundle:
    """Immutable, versioned ownership map for one raw-PUF aging pass."""

    schema_version: int
    aging_version: str
    source_year: int
    target_year: int
    provenance: PufAgingProvenance
    weight: PufAgingColumnFactor
    straight: tuple[PufAgingColumnFactor, ...]
    signed: tuple[PufAgingSignedFactor, ...]
    unchanged_columns: tuple[str, ...]

    def validate(self) -> None:
        """Validate dates, evidence, factors, and exclusive column ownership."""

        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported PUF aging schema_version {self.schema_version!r}."
            )
        if not self.aging_version:
            raise ValueError("PUF aging bundle requires aging_version.")
        if self.source_year < 1900 or self.target_year < 1900:
            raise ValueError("PUF aging years must be four-digit calendar years.")
        if self.target_year <= self.source_year:
            raise ValueError("PUF aging target_year must be after source_year.")
        self.provenance.validate()
        self.weight.validate(label="weight factor")
        for factor in self.straight:
            factor.validate(label="straight factor")
        for factor in self.signed:
            factor.validate()
        if len(self.unchanged_columns) != len(set(self.unchanged_columns)):
            raise ValueError("PUF aging unchanged_columns must not repeat.")
        for column in self.unchanged_columns:
            _validate_column_name(column, label="unchanged column")

        owners: dict[str, str] = {}
        declarations = (
            (self.weight.column, "weight"),
            *((factor.column, "straight") for factor in self.straight),
            *((factor.column, "signed") for factor in self.signed),
            *((column, "unchanged") for column in self.unchanged_columns),
        )
        for column, owner in declarations:
            if column in owners:
                raise ValueError(
                    f"PUF aging column {column!r} has multiple owners: "
                    f"{owners[column]!r} and {owner!r}."
                )
            owners[column] = owner

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return every column whose presence is asserted by this bundle."""

        return (
            self.weight.column,
            *(factor.column for factor in self.straight),
            *(factor.column for factor in self.signed),
            *self.unchanged_columns,
        )


@lru_cache(maxsize=1)
def load_archived_puf_aging_factors() -> PufAgingFactorBundle:
    """Load the packaged audit-only 1.8.0 parity profile."""

    resource = files("populace.build.us").joinpath(PUF_AGING_ARCHIVED_PROFILE_RESOURCE)
    with resource.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    bundle = puf_aging_factors_from_mapping(payload)
    if bundle.aging_version != PUF_AGING_ARCHIVED_PROFILE_VERSION:
        raise ValueError(
            "Packaged archived PUF aging profile has unexpected version "
            f"{bundle.aging_version!r}."
        )
    return bundle


def puf_aging_factors_from_mapping(
    payload: Mapping[str, Any] | Any,
) -> PufAgingFactorBundle:
    """Parse and strictly validate a JSON/YAML-compatible factor bundle."""

    root = _mapping(payload, label="PUF aging factor bundle")
    _exact_keys(
        root,
        required={
            "schema_version",
            "aging_version",
            "source_year",
            "target_year",
            "provenance",
            "weight",
            "straight",
            "signed",
            "unchanged_columns",
        },
        optional={"fallback"},
        label="PUF aging factor bundle",
    )
    provenance_payload = _mapping(
        root["provenance"],
        label="PUF aging provenance",
    )
    _exact_keys(
        provenance_payload,
        required={
            "source_kind",
            "source_artifact_sha256",
            "source_coordinates",
            "notes",
        },
        label="PUF aging provenance",
    )
    provenance = PufAgingProvenance(
        source_kind=_string(
            provenance_payload["source_kind"],
            label="provenance.source_kind",
        ),
        source_artifact_sha256=_string(
            provenance_payload["source_artifact_sha256"],
            label="provenance.source_artifact_sha256",
        ),
        source_coordinates=_string_tuple(
            provenance_payload["source_coordinates"],
            label="provenance.source_coordinates",
        ),
        notes=_string(
            provenance_payload["notes"],
            label="provenance.notes",
        ),
    )
    weight = _column_factor(root["weight"], label="weight")
    straight = list(
        _column_factor(item, label=f"straight[{index}]")
        for index, item in enumerate(_sequence(root["straight"], label="straight"))
    )
    signed = tuple(
        _signed_factor(item, label=f"signed[{index}]")
        for index, item in enumerate(_sequence(root["signed"], label="signed"))
    )

    if "fallback" in root:
        fallback = _mapping(root["fallback"], label="fallback")
        _exact_keys(
            fallback,
            required={"factor", "fact_ids", "reason", "columns"},
            label="fallback",
        )
        factor = _number(fallback["factor"], label="fallback.factor")
        fact_ids = _string_tuple(fallback["fact_ids"], label="fallback.fact_ids")
        reason = _string(fallback["reason"], label="fallback.reason")
        straight.extend(
            PufAgingColumnFactor(
                column=column,
                factor=factor,
                fact_ids=fact_ids,
                fallback_reason=reason,
            )
            for column in _string_tuple(
                fallback["columns"],
                label="fallback.columns",
            )
        )

    bundle = PufAgingFactorBundle(
        schema_version=_integer(root["schema_version"], label="schema_version"),
        aging_version=_string(root["aging_version"], label="aging_version"),
        source_year=_integer(root["source_year"], label="source_year"),
        target_year=_integer(root["target_year"], label="target_year"),
        provenance=provenance,
        weight=weight,
        straight=tuple(straight),
        signed=signed,
        unchanged_columns=_string_tuple(
            root["unchanged_columns"],
            label="unchanged_columns",
        ),
    )
    bundle.validate()
    return bundle


def age_raw_puf(
    puf: pd.DataFrame,
    *,
    factors: PufAgingFactorBundle,
) -> pd.DataFrame:
    """Apply a validated factor bundle to raw tax-unit PUF columns.

    The input is never mutated. Columns not named by the bundle pass through
    unchanged. A named missing, nonnumeric, or nonfinite column fails closed.
    Aggregate-only provenance is attached to ``DataFrame.attrs``.
    """

    if not isinstance(puf, pd.DataFrame):
        raise TypeError(f"Raw PUF must be a DataFrame, got {type(puf).__name__}.")
    factors.validate()
    missing = sorted(set(factors.required_columns) - set(puf.columns))
    if missing:
        raise ValueError(f"Raw PUF is missing aging column(s): {missing}.")

    result = puf.copy(deep=True)
    unchanged_snapshots = {
        column: result[column].copy(deep=True) for column in factors.unchanged_columns
    }

    for factor in (factors.weight, *factors.straight):
        values = _finite_numeric_column(result, factor.column)
        result[factor.column] = values * factor.factor

    for factor in factors.signed:
        values = _finite_numeric_column(result, factor.column)
        aged = values.copy()
        positive = values > 0.0
        negative = values < 0.0
        aged.loc[positive] = values.loc[positive] * factor.positive_factor
        aged.loc[negative] = values.loc[negative] * factor.negative_factor
        result[factor.column] = aged

    for column, original in unchanged_snapshots.items():
        if not result[column].equals(original):  # pragma: no cover - defensive
            raise AssertionError(f"PUF aging mutated unchanged column {column!r}.")

    attrs = dict(puf.attrs)
    attrs[_PROVENANCE_ATTR] = puf_aging_provenance(
        factors,
        row_count=len(result),
    )
    result.attrs = attrs
    return result


def puf_aging_provenance(
    factors: PufAgingFactorBundle,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    """Return JSON-compatible aggregate factor provenance."""

    factors.validate()
    fact_ids = sorted(
        {
            fact_id
            for factor in (
                factors.weight,
                *factors.straight,
                *factors.signed,
            )
            for fact_id in factor.fact_ids
        }
    )
    fallback_columns = sorted(
        factor.column
        for factor in (*factors.straight, *factors.signed)
        if factor.fallback_reason is not None
    )
    return {
        "schema_version": factors.schema_version,
        "aging_version": factors.aging_version,
        "source_year": factors.source_year,
        "target_year": factors.target_year,
        "source_kind": factors.provenance.source_kind,
        "source_artifact_sha256": (factors.provenance.source_artifact_sha256),
        "source_coordinates": list(factors.provenance.source_coordinates),
        "fact_ids": fact_ids,
        "weight_column": factors.weight.column,
        "straight_factor_count": len(factors.straight),
        "signed_factor_count": len(factors.signed),
        "fallback_columns": fallback_columns,
        "unchanged_columns": list(factors.unchanged_columns),
        "row_count": row_count,
        "notes": factors.provenance.notes,
    }


def _column_factor(value: Any, *, label: str) -> PufAgingColumnFactor:
    payload = _mapping(value, label=label)
    _exact_keys(
        payload,
        required={"column", "factor"},
        optional={"fact_ids", "fallback_reason"},
        label=label,
    )
    fact_ids = (
        _string_tuple(payload["fact_ids"], label=f"{label}.fact_ids")
        if "fact_ids" in payload
        else ()
    )
    fallback_reason = (
        _string(
            payload["fallback_reason"],
            label=f"{label}.fallback_reason",
        )
        if "fallback_reason" in payload
        else None
    )
    return PufAgingColumnFactor(
        column=_string(payload["column"], label=f"{label}.column"),
        factor=_number(payload["factor"], label=f"{label}.factor"),
        fact_ids=fact_ids,
        fallback_reason=fallback_reason,
    )


def _signed_factor(value: Any, *, label: str) -> PufAgingSignedFactor:
    payload = _mapping(value, label=label)
    _exact_keys(
        payload,
        required={"column", "positive_factor", "negative_factor"},
        optional={"fact_ids", "fallback_reason"},
        label=label,
    )
    fact_ids = (
        _string_tuple(payload["fact_ids"], label=f"{label}.fact_ids")
        if "fact_ids" in payload
        else ()
    )
    fallback_reason = (
        _string(
            payload["fallback_reason"],
            label=f"{label}.fallback_reason",
        )
        if "fallback_reason" in payload
        else None
    )
    return PufAgingSignedFactor(
        column=_string(payload["column"], label=f"{label}.column"),
        positive_factor=_number(
            payload["positive_factor"],
            label=f"{label}.positive_factor",
        ),
        negative_factor=_number(
            payload["negative_factor"],
            label=f"{label}.negative_factor",
        ),
        fact_ids=fact_ids,
        fallback_reason=fallback_reason,
    )


def _finite_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = int(np.count_nonzero(~np.isfinite(values.to_numpy(dtype=np.float64))))
    if invalid:
        raise ValueError(
            f"Raw PUF aging column {column!r} has {invalid} nonnumeric or "
            "nonfinite value(s)."
        )
    return values.astype("float64")


def _validate_factor(value: float, *, label: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"PUF aging {label} must be finite and nonnegative.")


def _validate_factor_evidence(
    fact_ids: tuple[str, ...],
    fallback_reason: str | None,
    *,
    label: str,
) -> None:
    if any(not fact_id for fact_id in fact_ids):
        raise ValueError(f"PUF aging {label} fact IDs must be nonempty.")
    if not fact_ids and not fallback_reason:
        raise ValueError(f"PUF aging {label} requires fact_ids or fallback_reason.")


def _validate_column_name(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"PUF aging {label} requires a nonempty column name.")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a list.")
    return tuple(value)


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    sequence = _sequence(value, label=label)
    if not all(isinstance(item, str) and item for item in sequence):
        raise ValueError(f"{label} must contain only nonempty strings.")
    if len(sequence) != len(set(sequence)):
        raise ValueError(f"{label} must not contain duplicates.")
    return tuple(sequence)


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return value


def _exact_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(payload))
    unexpected = sorted(set(payload) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing key(s): {missing}.")
    if unexpected:
        raise ValueError(f"{label} has unsupported key(s): {unexpected}.")
