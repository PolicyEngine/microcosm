"""Exact-basis, source-receipted bindings for prepared monetary measures."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from microcosm.build.ledger_targets import LedgerTargetReference
from microcosm.calibrate import TargetSpec
from microcosm.calibrate.monetary_binding import canonical_record_ids, monetary_digest


def _sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 identity.")


def _vector(values: ArrayLike, label: str) -> np.ndarray:
    result = np.asarray(values, dtype="<f8")
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty, finite one-dimensional vector.")
    return result


@dataclass(frozen=True)
class MonetaryBasis:
    """Exact currency, time, sector, perimeter, and valuation for an amount."""

    currency: str
    unit: str
    period: str
    temporal_basis: Literal["annual_flow", "closing_stock"]
    sector: str
    perimeter: str
    valuation: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.currency, str)
            or re.fullmatch(r"[A-Z]{3}", self.currency) is None
        ):
            raise ValueError("currency must be an uppercase three-letter code.")
        if self.unit != "base_currency":
            raise ValueError("unit must be base_currency; no implicit unit scaling.")
        if any(
            not isinstance(getattr(self, key), str) or not getattr(self, key).strip()
            for key in ("sector", "perimeter", "valuation")
        ):
            raise ValueError("sector, perimeter, and valuation must be explicit.")
        if self.temporal_basis == "annual_flow":
            if (
                not isinstance(self.period, str)
                or re.fullmatch(r"[0-9]{4}", self.period) is None
            ):
                raise ValueError("annual_flow period must be YYYY.")
            date(int(self.period), 1, 1)
        elif self.temporal_basis == "closing_stock":
            if (
                not isinstance(self.period, str)
                or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", self.period) is None
            ):
                raise ValueError("closing_stock period must be YYYY-MM-DD.")
            date.fromisoformat(self.period)
        else:
            raise ValueError("temporal_basis must be annual_flow or closing_stock.")


@dataclass(frozen=True, eq=False)
class PreparedMonetaryMeasure:
    """Immutable prepared direct amounts and their entity-row identity receipt."""

    values: np.ndarray = field(repr=False)
    record_ids: np.ndarray = field(repr=False)
    basis: MonetaryBasis
    factor: float
    source_identity_sha256: str
    bridge_description: str
    bridge_source_sha256: str
    readiness: str = "ready"
    measure_kind: str = "direct"
    source_values_sha256: str = field(init=False)
    values_sha256: str = field(init=False)
    record_ids_sha256: str = field(init=False)
    bridge_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.basis, MonetaryBasis):
            raise TypeError("basis must be MonetaryBasis.")
        factor = float(self.factor)
        if (
            isinstance(self.factor, (bool, np.bool_))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("factor must be explicitly positive and finite.")
        _sha256(self.source_identity_sha256, "source_identity_sha256")
        _sha256(self.bridge_source_sha256, "bridge_source_sha256")
        if (
            not isinstance(self.bridge_description, str)
            or not self.bridge_description.strip()
        ):
            raise ValueError("bridge_description must explain the explicit transport.")
        if self.readiness not in {"ready", "unbound"} or self.measure_kind not in {
            "direct",
            "policy_derived",
        }:
            raise ValueError(
                "prepared measure must declare a supported readiness and kind."
            )
        raw = _vector(self.values, "amounts")
        with np.errstate(over="ignore", invalid="ignore"):
            transported = _vector(raw * factor, "transported amounts")
        ids = canonical_record_ids(self.record_ids)
        if ids.shape != raw.shape:
            raise ValueError("record_ids must align one-to-one with amounts.")
        immutable_values = np.frombuffer(transported.tobytes(), dtype="<f8")
        immutable_ids = np.frombuffer(ids.tobytes(), dtype="<i8")
        object.__setattr__(self, "values", immutable_values)
        object.__setattr__(self, "record_ids", immutable_ids)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(
            self, "source_values_sha256", hashlib.sha256(raw.tobytes()).hexdigest()
        )
        object.__setattr__(
            self,
            "values_sha256",
            hashlib.sha256(immutable_values.tobytes()).hexdigest(),
        )
        object.__setattr__(
            self,
            "record_ids_sha256",
            hashlib.sha256(immutable_ids.tobytes()).hexdigest(),
        )
        object.__setattr__(self, "bridge_sha256", monetary_digest(self._bridge()))

    def _bridge(self) -> dict[str, object]:
        return {
            "factor": self.factor,
            "description": self.bridge_description,
            "source_sha256": self.bridge_source_sha256,
        }

    def _check_vector_metadata(self) -> None:
        if (
            self.values.dtype != np.dtype("<f8")
            or self.values.ndim != 1
            or self.record_ids.dtype != np.dtype("<i8")
            or self.record_ids.shape != self.values.shape
        ):
            raise ValueError("Prepared array dtype/shape metadata was mutated.")

    def receipt(self) -> dict[str, object]:
        """Return fresh, JSON-safe receipt data; it cannot mutate this measure."""
        self._check_vector_metadata()
        receipt = {
            "schema_version": 2,
            "basis": asdict(self.basis),
            "n_records": len(self.values),
            "source_identity_sha256": self.source_identity_sha256,
            "source_values_sha256": self.source_values_sha256,
            "values_sha256": self.values_sha256,
            "record_ids_sha256": self.record_ids_sha256,
            "bridge": self._bridge(),
            "bridge_sha256": self.bridge_sha256,
            "readiness": self.readiness,
            "measure_kind": self.measure_kind,
        }
        return {**receipt, "receipt_sha256": monetary_digest(receipt)}

    def total(self, weights: ArrayLike) -> float:
        """Return the checked weighted total over this already-aligned vector."""
        self._check_vector_metadata()
        weights = _vector(weights, "weights")
        if weights.shape != self.values.shape or (weights < 0).any():
            raise ValueError("weights must be aligned with amounts and nonnegative.")
        total = float(np.dot(self.values, weights))
        if not math.isfinite(total):
            raise ValueError("weighted monetary total must be finite.")
        return total


def prepare_monetary_measure(
    values: ArrayLike,
    *,
    record_ids: ArrayLike,
    basis: MonetaryBasis,
    factor: float,
    source_identity_sha256: str,
    bridge_description: str,
    bridge_source_sha256: str,
    readiness: str = "ready",
    measure_kind: str = "direct",
) -> PreparedMonetaryMeasure:
    """Freeze an explicit transport and exact entity-row identities."""
    return PreparedMonetaryMeasure(
        values,
        record_ids,
        basis,
        factor,
        source_identity_sha256,
        bridge_description,
        bridge_source_sha256,
        readiness,
        measure_kind,
    )


def bind_monetary_target(
    reference: LedgerTargetReference,
    *,
    value: float,
    source_basis: MonetaryBasis,
    source_assertion: str,
    source_identity_sha256: str,
    prepared: PreparedMonetaryMeasure,
    source: str | None = None,
) -> TargetSpec:
    """Bind one observed fact only after exact basis and direct-measure checks."""
    if not isinstance(reference, LedgerTargetReference):
        raise TypeError("reference must be LedgerTargetReference.")
    if not isinstance(source_basis, MonetaryBasis) or not isinstance(
        prepared, PreparedMonetaryMeasure
    ):
        raise TypeError("source_basis and prepared must be typed monetary values.")
    prepared._check_vector_metadata()
    if reference.filter is not None:
        raise ValueError(
            "Monetary filters are unsupported; prepare a masked vector instead."
        )
    if source_assertion not in {"observation", "observed", "derived_observation"}:
        raise ValueError(
            "Only observation facts may bind; projections and forecasts are unbound."
        )
    if prepared.readiness != "ready" or prepared.measure_kind != "direct":
        raise ValueError("Only ready direct measures may bind.")
    metadata = reference.metadata
    reserved_metadata = {
        "monetary_binding",
        "monetary_binding_sha256",
        "monetary_source_activation_status",
    }
    conflicts = reserved_metadata.intersection(metadata)
    if conflicts:
        raise ValueError(
            "Reference metadata may not predeclare generated monetary binding "
            f"fields: {sorted(conflicts)}."
        )
    if metadata.get("monetary_target_role", "calibration") != "calibration":
        raise ValueError(
            "Only calibration references may bind; validation is held out."
        )
    if metadata.get("measure_kind", "direct") not in {
        "direct",
        "direct_column",
        "prepared_column",
    }:
        raise ValueError("Reference measure_kind needs an unsupported policy bridge.")
    if metadata.get("activation_status", "ready") not in {
        "ready",
        "active",
        "requires_prepared_measure",
    }:
        raise ValueError("Reference activation_status is not ready.")
    if source_basis != prepared.basis:
        raise ValueError("Monetary basis mismatch.")
    if reference.value_operation not in {"identity", "sum", "count_x_mean"} or any(
        value is not None
        for value in (
            reference.uprating_index,
            reference.uprating_from_period,
            reference.uprating_to_period,
        )
    ):
        raise ValueError(
            "Implicit temporal/value conversion or uprating is not permitted."
        )
    if (
        reference.value_operation == "count_x_mean"
        and source_assertion != "derived_observation"
    ):
        raise ValueError("count_x_mean requires a resolved derived_observation.")
    allowed_periods = {source_basis.period}
    if source_basis.temporal_basis == "closing_stock":
        allowed_periods.add(source_basis.period[:4])
    if reference.period is not None and str(reference.period) not in allowed_periods:
        raise ValueError("Reference period does not match the monetary basis.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Target amount must be finite.")
    if not isinstance(reference.signed, bool):
        raise ValueError("Reference signed must be an explicit boolean.")
    if not reference.signed and (prepared.values < 0).any():
        raise ValueError("Negative prepared amounts require reference signed=True.")
    if (value > 0 and not (prepared.values > 0).any()) or (
        value < 0 and not (prepared.values < 0).any()
    ):
        raise ValueError("Target has no same-sign prepared monetary support.")
    _sha256(source_identity_sha256, "source_identity_sha256")
    source = (
        source
        or reference.source
        or reference.ledger_source_record_id
        or reference.ledger_fact_key
    )
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Resolved target source citation is required.")
    bound_metadata = {
        **metadata,
        "activation_status": "active",
        "monetary_source_activation_status": metadata.get("activation_status", "ready"),
        "monetary_target_role": "calibration",
    }
    source_reference = asdict(reference)
    binding = {
        "reference": {**source_reference, "metadata": bound_metadata},
        "source_reference": source_reference,
        "value": value,
        "source": source,
        "source_assertion": source_assertion,
        "source_identity_sha256": source_identity_sha256,
        "prepared": prepared.receipt(),
    }
    return TargetSpec(
        name=reference.name,
        entity=reference.entity,
        measure=reference.measure or "",
        value=value,
        filter=None,
        period=reference.period
        if reference.period is not None
        else source_basis.period,
        source=source,
        family=reference.family,
        signed=reference.signed,
        se=reference.se,
        tolerance=reference.tolerance,
        notes=reference.notes,
        metadata={
            **bound_metadata,
            "monetary_binding": json.dumps(binding, sort_keys=True, allow_nan=False),
            "monetary_binding_sha256": monetary_digest(binding),
        },
    )
