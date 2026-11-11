"""Validation for the normalized calibration hierarchy value objects."""

import pytest

from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyDimension,
    HierarchyGeography,
    HierarchyNode,
)


def test_hierarchy_accepts_zero_or_many_ordered_dimensions() -> None:
    dimensions = (
        HierarchyDimension("income_band", "Income band", "low", "Low"),
        HierarchyDimension("sex", "Sex", "female", "Female"),
    )
    hierarchy = CalibrationHierarchy(
        provider=HierarchyNode("hmrc", "HM Revenue and Customs"),
        category=HierarchyCategory("income", "Income", "hmrc"),
        geography=HierarchyGeography("uk", "United Kingdom", "country"),
        dimensions=dimensions,
        target=HierarchyNode("hmrc/income", "Income for women in the low band"),
    )

    assert hierarchy.dimensions == dimensions


def test_category_must_belong_to_selected_provider() -> None:
    with pytest.raises(ValueError, match="belongs to provider"):
        CalibrationHierarchy(
            provider=HierarchyNode("hmrc", "HM Revenue and Customs"),
            category=HierarchyCategory("income", "Income", "ons"),
            geography=HierarchyGeography("uk", "United Kingdom", "country"),
            dimensions=(),
            target=HierarchyNode("hmrc/income", "Income"),
        )


def test_dimension_identifiers_are_unique() -> None:
    with pytest.raises(ValueError, match="duplicate dimension"):
        CalibrationHierarchy(
            provider=HierarchyNode("hmrc", "HM Revenue and Customs"),
            category=HierarchyCategory("income", "Income", "hmrc"),
            geography=HierarchyGeography("uk", "United Kingdom", "country"),
            dimensions=(
                HierarchyDimension("sex", "Sex", "female", "Female"),
                HierarchyDimension("sex", "Sex", "male", "Male"),
            ),
            target=HierarchyNode("hmrc/income", "Income"),
        )
