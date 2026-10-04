from __future__ import annotations

from microcosm.build.uk_runtime import local_targets
from microcosm.build.uk_runtime.frs_spine import TENURE_MAP
from microcosm.build.uk_runtime.tenure_constants import (
    UK_OWNER_TENURE_CATEGORIES,
    UK_TENURE_CATEGORIES,
    UK_TENURE_TYPE_TO_CATEGORY,
)
from microcosm.build.uk_runtime.was_wealth import (
    UK_WAS_STRATIFIED_TARGETS,
    UK_WAS_TENURE_PREDICTORS,
    WAS_OWNER_TENURE_CODES,
)


def test_every_spine_tenure_type_has_a_category() -> None:
    """The FRS spine's tenure values are exactly the mapped engine names."""

    assert set(TENURE_MAP.values()) == set(UK_TENURE_TYPE_TO_CATEGORY)
    assert set(UK_TENURE_TYPE_TO_CATEGORY.values()) == set(UK_TENURE_CATEGORIES)
    assert set(UK_OWNER_TENURE_CATEGORIES) < set(UK_TENURE_CATEGORIES)


def test_consumers_derive_their_views_from_the_shared_categories() -> None:
    assert set(local_targets._TENURE_METRIC_CATEGORIES.values()) == set(
        UK_TENURE_CATEGORIES
    )
    # One WAS flag per category but the base level, in category order.
    assert UK_WAS_TENURE_PREDICTORS == tuple(
        f"tenure_{category}" for category in UK_TENURE_CATEGORIES[1:]
    )
    assert set(WAS_OWNER_TENURE_CODES.values()) == set(UK_OWNER_TENURE_CATEGORIES)
    for categories in UK_WAS_STRATIFIED_TARGETS.values():
        assert set(categories) <= set(UK_TENURE_CATEGORIES)
