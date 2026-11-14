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


def _ranked_hierarchy(*dimensions: HierarchyDimension) -> CalibrationHierarchy:
    return CalibrationHierarchy(
        provider=HierarchyNode("ed_fsa", "ED · Federal Student Aid"),
        category=HierarchyCategory("ed_fsa.balance", "Balance", "ed_fsa"),
        geography=HierarchyGeography("0100000US", "United States", "country"),
        dimensions=dimensions,
        target=HierarchyNode("ed_fsa/balance", "Balance"),
    )


def test_a_dimension_value_carries_its_source_rank() -> None:
    ranked = HierarchyDimension(
        "ed_fsa.borrower_debt_size", "Borrower debt size", "debt_200k_plus", "200K+", 8
    )
    unranked = HierarchyDimension("sex", "Sex", "female", "Female")

    assert ranked.value_rank == 8
    assert unranked.value_rank is None
    assert ranked.to_dict()["value_rank"] == 8
    # A dimension without a rank keeps the four-key encoding it always had.
    assert unranked.to_dict() == {
        "id": "sex",
        "label": "Sex",
        "value_id": "female",
        "value_label": "Female",
    }
    hierarchy = _ranked_hierarchy(ranked, unranked)
    assert CalibrationHierarchy.from_dict(hierarchy.to_dict()) == hierarchy


@pytest.mark.parametrize("rank", [-1, True, 1.0, "0"])
def test_a_value_rank_must_be_a_non_negative_integer(rank: object) -> None:
    with pytest.raises(ValueError, match="value_rank must be a non-negative"):
        HierarchyDimension("sex", "Sex", "female", "Female", rank)  # type: ignore[arg-type]
