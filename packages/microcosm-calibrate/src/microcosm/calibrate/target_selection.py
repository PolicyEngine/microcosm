"""Explicit target-scope selection, independent of country and pool sizing.

Selection does not admit unsupported targets, adjudicate source authority, or
reduce validation requirements. It records only the requested geography scope.
Country adapters normalize their metadata through ``geography_resolver``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .registry import TargetRegistry, TargetSpec


@dataclass(frozen=True)
class TargetSelection:
    """Selected ordered facts and a detached, reproducible scope receipt."""

    registry: TargetRegistry
    _receipt_json: str

    @property
    def receipt(self) -> dict[str, object]:
        """Return a fresh receipt; caller mutations cannot alter its identity."""
        return json.loads(self._receipt_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_bytes(self) -> bytes:
        """Return the canonical selection receipt bytes."""
        return self._receipt_json.encode("utf-8")


def _level(spec: TargetSpec) -> str:
    return spec.metadata.get("geography_level", "")


def select_targets(
    registry: TargetRegistry,
    *,
    geography_levels: Sequence[str] | None = None,
    geography_resolver: Callable[[TargetSpec], str] | None = None,
) -> TargetSelection:
    """Keep all targets by default, or explicitly select normalized levels.

    Ordering and ``(name, period)`` identity are retained. Unknown level names,
    unclassified facts and an empty selected problem refuse. A resolver must
    validate country aliases/ambiguities; it is never serialized as a callback.
    The receipt stores its resolved result for every original fact instead.
    """
    if not isinstance(registry, TargetRegistry):
        raise TypeError("Target selection requires TargetRegistry.")
    levels = None
    if geography_levels is not None:
        if isinstance(geography_levels, str | bytes):
            raise ValueError("geography_levels must be a nonempty sequence of levels.")
        levels = tuple(geography_levels)
        if not levels or any(
            not isinstance(level, str) or not level or level != level.strip()
            for level in levels
        ):
            raise ValueError("geography_levels must contain nonempty literal names.")
        if len(set(levels)) != len(levels):
            raise ValueError("geography_levels must not repeat levels.")
        levels = tuple(sorted(levels))
    resolve = _level if geography_resolver is None else geography_resolver
    resolved = []
    for spec in registry.specs:
        level = resolve(spec)
        if not isinstance(level, str) or not level or level != level.strip():
            raise ValueError(
                f"Target {spec.key!r} lacks normalized geography metadata."
            )
        resolved.append(level)
    if levels is not None and (unknown := set(levels) - set(resolved)):
        raise ValueError(
            f"Target selector contains unknown geography levels {sorted(unknown)}."
        )
    by_key = {
        spec.key: level for spec, level in zip(registry.specs, resolved, strict=True)
    }
    selected = registry.select(
        predicate=lambda spec: levels is None or by_key[spec.key] in levels
    )
    if not selected.specs:
        raise ValueError("Target selection requires a nonempty selected problem.")
    included, excluded = [], []
    for spec, level in zip(registry.specs, resolved, strict=True):
        row = {
            "name": spec.name,
            "period": spec.period,
            "geography_level": level,
            "family": spec.family,
            "source": spec.source,
        }
        if levels is None or level in levels:
            included.append(
                {
                    **row,
                    "reason": "all_geographies"
                    if levels is None
                    else "geography_selected",
                }
            )
        else:
            excluded.append({**row, "reason": "geography_not_selected"})
    receipt = {
        "schema": "microcosm.calibrate.target-selection.v1",
        "source_registry_version": registry.version,
        "selected_registry_version": selected.version,
        "selector": {"geography_levels": levels, "explicit": levels is not None},
        "included": included,
        "excluded": excluded,
    }
    return TargetSelection(
        selected,
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )
