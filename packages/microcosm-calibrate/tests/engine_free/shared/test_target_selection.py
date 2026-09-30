"""Target scope is explicit and preserves ordered fact identities."""

import pytest

from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.target_selection import select_targets


def registry():
    return TargetRegistry(
        [
            TargetSpec(
                "population",
                "person",
                20,
                "one",
                period=2024,
                source="census",
                metadata={"geography_level": "country"},
            ),
            TargetSpec(
                "population",
                "person",
                22,
                "one",
                period=2025,
                source="census",
                metadata={"geography_level": "region"},
            ),
            TargetSpec(
                "local",
                "person",
                5,
                "one",
                period=2025,
                source="census",
                metadata={"geography_level": "district"},
            ),
        ],
        country="test",
    )


def test_default_all_and_explicit_country_are_distinct():
    all_targets = select_targets(registry())
    assert all_targets.registry.specs == registry().specs
    assert all_targets.receipt["selector"] == {
        "geography_levels": None,
        "explicit": False,
    }
    selected = select_targets(registry(), geography_levels=("country",))
    assert [spec.key for spec in selected.registry.specs] == [("population", 2024)]
    assert [row["period"] for row in selected.receipt["excluded"]] == [2025, 2025]
    assert all(
        row["reason"] == "geography_not_selected"
        for row in selected.receipt["excluded"]
    )
    assert selected.receipt["source_registry_version"] == registry().version


def test_missing_unknown_empty_and_resolved_geographies():
    with pytest.raises(ValueError, match="unknown"):
        select_targets(registry(), geography_levels=("missing",))
    with pytest.raises(ValueError, match="nonempty"):
        select_targets(registry(), geography_levels=())
    missing = TargetRegistry(
        [TargetSpec("x", "person", 1, "one", source="test")], country="test"
    )
    with pytest.raises(ValueError, match="geography"):
        select_targets(missing)
    resolved = select_targets(missing, geography_resolver=lambda spec: "country")
    assert resolved.receipt["included"][0]["geography_level"] == "country"
