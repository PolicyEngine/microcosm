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


def _tenure_types(binding) -> set[str]:
    (predicate,) = [
        predicate
        for predicate in binding["filters"]
        if predicate["variable"] == "tenure_type"
    ]
    return set(predicate["in"]) if "in" in predicate else {predicate["equals"]}


def test_consumers_derive_their_views_from_the_shared_categories() -> None:
    # The local tenure metrics' declarations partition the engine's tenure
    # types by the shared grouping: one category per metric, each once.
    tenure_types = {
        name: _tenure_types(binding)
        for name, binding in local_targets.local_metric_bindings("la").items()
        if name.startswith("tenure/")
    }
    assert sorted(
        category
        for types in tenure_types.values()
        for (category,) in [{UK_TENURE_TYPE_TO_CATEGORY[value] for value in types}]
    ) == sorted(UK_TENURE_CATEGORIES)
    assert sorted(value for types in tenure_types.values() for value in types) == (
        sorted(UK_TENURE_TYPE_TO_CATEGORY)
    )
    # One WAS flag per category but the base level, in category order.
    assert UK_WAS_TENURE_PREDICTORS == tuple(
        f"tenure_{category}" for category in UK_TENURE_CATEGORIES[1:]
    )
    assert set(WAS_OWNER_TENURE_CODES.values()) == set(UK_OWNER_TENURE_CATEGORIES)
    for categories in UK_WAS_STRATIFIED_TARGETS.values():
        assert set(categories) <= set(UK_TENURE_CATEGORIES)
