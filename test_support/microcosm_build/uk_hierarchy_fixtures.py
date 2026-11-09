"""Invented schema-8 calibration hierarchies for UK test registries.

#855 requires a ``CalibrationHierarchy`` on every registry-backed target before
diagnostics can be published. Tests that invent ``TargetSpec`` rows use this
one builder instead of repeating the nested constructors; the shape follows the
``_uc_hierarchy`` fixture that #855 added to the retired national-driver tests.
"""

from __future__ import annotations

from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
)


def uk_fixture_hierarchy(
    name: str,
    *,
    level: str,
    geography_id: str,
    provider_id: str = "ons",
    provider_label: str = "Office for National Statistics",
    category_id: str = "ons.households",
    category_label: str = "Households",
) -> CalibrationHierarchy:
    """A complete hierarchy whose target id equals the spec ``name``."""

    return CalibrationHierarchy(
        provider=HierarchyNode(id=provider_id, label=provider_label),
        category=HierarchyCategory(
            id=category_id, label=category_label, provider_id=provider_id
        ),
        geography=HierarchyGeography(id=geography_id, label=geography_id, level=level),
        dimensions=(),
        target=HierarchyNode(id=name, label=name),
    )
