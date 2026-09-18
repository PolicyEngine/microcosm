"""Typed presentation hierarchy carried by every published calibration target.

The hierarchy is author-owned data, not a dashboard inference.  Provider and
category are selected by Microcosm, geography describes the target's scope,
and dimensions are inherited from the matched Chronicle fact(s).  A target may
have any number of dimensions, including none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "CalibrationHierarchy",
    "CalibrationHierarchySeed",
    "HierarchyCategory",
    "HierarchyDimension",
    "HierarchyGeography",
    "HierarchyNode",
]


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class HierarchyNode:
    """A stable identifier and its author-supplied display label."""

    id: str
    label: str

    def __post_init__(self) -> None:
        _require_text(self.id, field_name="HierarchyNode.id")
        _require_text(self.label, field_name=f"HierarchyNode {self.id!r}.label")


@dataclass(frozen=True)
class HierarchyCategory(HierarchyNode):
    """A target category owned by exactly one provider."""

    provider_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_text(
            self.provider_id,
            field_name=f"HierarchyCategory {self.id!r}.provider_id",
        )


@dataclass(frozen=True)
class HierarchyGeography(HierarchyNode):
    """The target's geographic scope and geographic level."""

    level: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_text(self.level, field_name=f"HierarchyGeography {self.id!r}.level")


@dataclass(frozen=True)
class HierarchyDimension:
    """One Chronicle dimension and the categorical value selected by a target."""

    id: str
    label: str
    value_id: str
    value_label: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("id", self.id),
            ("label", self.label),
            ("value_id", self.value_id),
            ("value_label", self.value_label),
        ):
            _require_text(
                value,
                field_name=f"HierarchyDimension {self.id!r}.{field_name}",
            )


@dataclass(frozen=True)
class CalibrationHierarchySeed:
    """Microcosm-owned hierarchy data known before Chronicle matching.

    ``target_label`` is optional for a direct, unchanged single-fact target,
    whose display label belongs to Chronicle. It is required by the compiler
    when the reference combines facts, transforms their value, or assigns a
    different target period.
    """

    provider: HierarchyNode
    category: HierarchyCategory
    target_label: str | None = None

    def __post_init__(self) -> None:
        if self.category.provider_id != self.provider.id:
            raise ValueError(
                f"Hierarchy category {self.category.id!r} belongs to provider "
                f"{self.category.provider_id!r}, not {self.provider.id!r}."
            )
        if self.target_label is not None:
            _require_text(
                self.target_label,
                field_name="CalibrationHierarchySeed.target_label",
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CalibrationHierarchySeed:
        return cls(
            provider=HierarchyNode(**raw["provider"]),
            category=HierarchyCategory(**raw["category"]),
            target_label=raw.get("target_label"),
        )


@dataclass(frozen=True)
class CalibrationHierarchy:
    """The complete ordered navigation path for one calibration target.

    The fixed tier types are provider, category, geography, zero or more
    dimensions, and target.  ``dimensions`` preserves producer order; callers
    must not sort it after construction.
    """

    provider: HierarchyNode
    category: HierarchyCategory
    geography: HierarchyGeography
    dimensions: tuple[HierarchyDimension, ...]
    target: HierarchyNode

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        if self.category.provider_id != self.provider.id:
            raise ValueError(
                f"Hierarchy category {self.category.id!r} belongs to provider "
                f"{self.category.provider_id!r}, not {self.provider.id!r}."
            )
        dimension_ids = [dimension.id for dimension in self.dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError(
                f"Hierarchy target {self.target.id!r} contains duplicate dimension "
                f"identifiers: {dimension_ids!r}."
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CalibrationHierarchy:
        """Deserialize the canonical registry/diagnostics representation."""

        return cls(
            provider=HierarchyNode(**raw["provider"]),
            category=HierarchyCategory(**raw["category"]),
            geography=HierarchyGeography(**raw["geography"]),
            dimensions=tuple(
                HierarchyDimension(**dimension) for dimension in raw["dimensions"]
            ),
            target=HierarchyNode(**raw["target"]),
        )
