"""Strict, stable-ID household group bounds shared by solver and admission.

This is a numerical constraint, not authority for an origin or budget artifact.
Callers must authenticate its complete population and frozen origin bindings.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


def _id(value: object) -> int | str:
    if isinstance(value, bool):
        raise ValueError("household IDs must be integers or nonempty strings")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("household IDs must be integers or nonempty strings")


@dataclass(frozen=True, eq=False)
class GroupedUpperBounds:
    """An exhaustive, disjoint aligned map and absolute float64 upper bounds."""

    household_ids: Sequence[int | str]
    group_indices: np.ndarray
    absolute_bounds: np.ndarray
    constraint_digest: str = field(init=False)
    _members: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ids = tuple(_id(x) for x in self.household_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("household IDs must be nonempty and unique")
        groups = np.asarray(self.group_indices)
        if groups.shape != (len(ids),) or groups.dtype.kind not in "iu":
            raise ValueError(
                "group indices must be integer and aligned to household IDs"
            )
        bounds = np.asarray(self.absolute_bounds, dtype=np.float64)
        if bounds.ndim != 1 or not bounds.size:
            raise ValueError("absolute bounds must be a nonempty vector")
        if not np.isfinite(bounds).all() or (bounds < 0).any():
            raise ValueError("absolute bounds must be finite and nonnegative")
        if (groups < 0).any() or (groups >= len(bounds)).any():
            raise ValueError("unknown group index")
        if set(groups.tolist()) != set(range(len(bounds))):
            raise ValueError("group membership must cover every bound")
        groups = np.frombuffer(groups.astype("<i8").tobytes(), dtype="<i8")
        bounds = np.frombuffer(bounds.astype("<f8").tobytes(), dtype="<f8")
        members: list[np.ndarray] = []
        # Sort once. Every authoritative reduction thereafter uses this order.
        ordered = sorted(
            range(len(ids)), key=lambda i: (isinstance(ids[i], str), ids[i])
        )
        buckets: list[list[int]] = [[] for _ in bounds]
        for i in ordered:
            buckets[int(groups[i])].append(i)
        for bucket in buckets:
            members.append(
                np.frombuffer(np.asarray(bucket, dtype="<i8").tobytes(), dtype="<i8")
            )
        payload = {
            "contract": "stable-household-fsum-group-upper-v1",
            "household_ids": ids,
            "group_indices": groups.tolist(),
            "absolute_bounds_hex": [float(x).hex() for x in bounds],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
        object.__setattr__(self, "household_ids", ids)
        object.__setattr__(self, "group_indices", groups)
        object.__setattr__(self, "absolute_bounds", bounds)
        object.__setattr__(self, "_members", tuple(members))
        object.__setattr__(self, "constraint_digest", digest)

    @property
    def group_count(self) -> int:
        return len(self.absolute_bounds)

    @property
    def digest(self) -> str:
        """Canonical ordered-ID/map/absolute-bound identity."""
        return self.constraint_digest

    def _weights(self, weights: np.ndarray, *, positive: bool = False) -> np.ndarray:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(self.household_ids),):
            raise ValueError("weights must align with all household IDs")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("group weights must be finite and nonnegative")
        if positive and ((values <= 0).any() or (self.absolute_bounds <= 0).any()):
            raise ValueError(
                "grouped log-weight solver requires strictly positive weights and bounds"
            )
        return values

    def totals(self, weights: np.ndarray) -> np.ndarray:
        values = self._weights(weights)
        try:
            result = np.asarray(
                [
                    math.fsum(float(values[i]) for i in members)
                    for members in self._members
                ],
                dtype=np.float64,
            )
        except OverflowError as exc:
            raise ValueError("nonfinite group total") from exc
        if not np.isfinite(result).all():
            raise ValueError("nonfinite group total")
        return result

    def check(self, weights: np.ndarray, *, positive: bool = False) -> np.ndarray:
        """Return exact ordered fsum totals; never forgive an overage."""
        values = self._weights(weights, positive=positive)
        totals = self.totals(values)
        if (totals > self.absolute_bounds).any():
            raise ValueError("group weights exceed frozen absolute bounds")
        return totals

    def project(
        self, weights: np.ndarray, *, positive: bool = True
    ) -> tuple[np.ndarray, int]:
        """Project directly from one materialized vector; no log/exp correction.

        For an overage, at most 64 downward float64 factor corrections recompute
        from the unchanged candidate. A feasible input is a byte-preserving no-op.
        """
        candidate = self._weights(weights, positive=positive)
        totals = self.totals(candidate)
        accepted = candidate.copy()
        corrected = 0
        for group in np.flatnonzero(totals > self.absolute_bounds):
            corrected += 1
            members = self._members[int(group)]
            original = candidate[members]
            factor = np.float64(self.absolute_bounds[group] / totals[group])
            for correction in range(65):
                projected = original * factor
                total = math.fsum(float(x) for x in projected)
                if total <= self.absolute_bounds[group]:
                    accepted[members] = projected
                    break
                if correction == 64:
                    raise ValueError(
                        "group projection exceeded 64 rounding corrections"
                    )
                factor = np.nextafter(factor, np.float64(0))
            else:  # pragma: no cover
                raise AssertionError("unreachable projection state")
        self.check(accepted, positive=positive)
        return accepted, corrected

    def diagnostics(
        self, weights: np.ndarray, last_corrected_count: int = 0
    ) -> dict[str, object]:
        if (
            isinstance(last_corrected_count, bool)
            or not isinstance(last_corrected_count, Integral)
            or not 0 <= last_corrected_count <= self.group_count
        ):
            raise ValueError(
                "last corrected group count must be an integer from zero to group count"
            )
        totals = self.check(weights)
        return {
            "constraint_digest": self.constraint_digest,
            "group_count": self.group_count,
            "binding_group_count": int(
                np.count_nonzero(totals == self.absolute_bounds)
            ),
            "last_corrected_group_count": int(last_corrected_count),
        }
