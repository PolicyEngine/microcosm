from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainRule,
    apply_cross_grain_reconciliation,
    detect_cross_grain_inconsistencies,
)


def _leg(area: str) -> str:
    return area[0]


def _rule(*, bridges: tuple[CrossGrainBridge, ...] = ()) -> CrossGrainRule:
    return CrossGrainRule(
        grain_precedence=("country", "constituency", "la"),
        signature_fields=("concept", "entity", "map_to", "filters"),
        bridges=bridges,
        leg_of_area=_leg,
        parent_geography_legs={
            "UK": ("E", "W", "S", "N"),
            "GB": ("E", "W", "S"),
            "E": ("E",),
            "S": ("S",),
        },
    )


def _signature(concept: str = "households") -> dict[str, object]:
    return {
        "measurement": {
            "concept": concept,
            "entity": "household",
            "map_to": None,
            "filters": [],
        }
    }


def test_exact_signature_rescales_to_country_and_receipts_without_mutation():
    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 120.0),
            ("constituency", "E1", "local", 40.0),
            ("constituency", "S1", "local", 20.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    original = surface.copy(deep=True)
    signatures = {"national": _signature(), "local": _signature()}

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, ("national",), signatures, _rule()
    )

    pd.testing.assert_frame_equal(surface, original)
    assert reconciled["value"].tolist() == [120.0, 80.0, 40.0]
    assert len(receipt["inconsistencies_in_force"]) == 1
    assert receipt["absence"] is None
    group = receipt["groups"][0]
    assert group["bridge_id"] is None
    assert group["winning_grain"] == "country"
    assert group["legs"] == [
        {
            "leg": "E+W+S+N",
            "parent_geography_id": "UK",
            "higher_target_ids": ["national"],
            "n_areas": 2,
            "old_total": 60.0,
            "new_total": 120.0,
            "relative_shift": 1.0,
            "declared_factor": 2.0,
            "reason": "standing cross-grain rule: country controls constituency",
        }
    ]


def test_bridge_sums_exhaustive_higher_partition_to_one_control():
    bridge = CrossGrainBridge(
        "partition_vs_external",
        concept="households",
        higher_target_ids=("part_a", "part_b"),
        lower_side="external:census/households",
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "part_a", 30.0),
            ("country", "UK", "part_b", 70.0),
            ("constituency", "E1", "external:census/households", 30.0),
            ("constituency", "W1", "external:census/households", 20.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("part_a", "part_b"),
        {"part_a": _signature(), "part_b": _signature()},
        _rule(bridges=(bridge,)),
    )

    assert reconciled["value"].tolist() == [30.0, 70.0, 60.0, 40.0]
    assert receipt["groups"][0]["bridge_id"] == "partition_vs_external"
    assert receipt["groups"][0]["legs"][0]["higher_target_ids"] == [
        "part_a",
        "part_b",
    ]


def test_middle_grain_wins_when_country_is_absent():
    surface = pd.DataFrame(
        [
            ("constituency", "E1", "same", 75.0),
            ("constituency", "S1", "same", 25.0),
            ("la", "E9", "same", 30.0),
            ("la", "S9", "same", 20.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, (), {"same": _signature()}, _rule()
    )

    assert reconciled["value"].tolist() == [75.0, 25.0, 75.0, 25.0]
    assert receipt["groups"][0]["winning_grain"] == "constituency"
    assert [leg["declared_factor"] for leg in receipt["groups"][0]["legs"]] == [
        2.5,
        1.25,
    ]


def test_absence_receipt_exists_when_no_higher_target_is_bound():
    surface = pd.DataFrame(
        [("constituency", "E1", "local", 10.0)],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, (), {"local": _signature()}, _rule()
    )

    pd.testing.assert_frame_equal(reconciled, surface)
    assert receipt == {
        "bound_higher_targets": [],
        "inconsistencies_in_force": [],
        "groups": [],
        "absence": "No cross-grain inconsistencies are in force on this surface.",
    }


def test_partially_bound_declared_partition_is_refused():
    bridge = CrossGrainBridge(
        "partition",
        "households",
        ("part_a", "part_b"),
        "external:census/households",
    )
    surface = pd.DataFrame(
        [("country", "UK", "part_a", 10.0)],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    with pytest.raises(ValueError, match="partially bound"):
        detect_cross_grain_inconsistencies(
            surface,
            ("part_a",),
            {"part_a": _signature(), "part_b": _signature()},
            _rule(bridges=(bridge,)),
        )


def test_target_matched_by_two_bridges_is_refused():
    bridges = (
        CrossGrainBridge("one", "households", ("national",), "contract:local"),
        CrossGrainBridge("two", "households", ("other",), "contract:local"),
    )
    surface = pd.DataFrame(
        [("constituency", "E1", "local", 10.0)],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    with pytest.raises(ValueError, match="matched by two bridges"):
        detect_cross_grain_inconsistencies(
            surface,
            (),
            {
                "national": _signature(),
                "other": _signature(),
                "local": _signature(),
            },
            _rule(bridges=bridges),
        )


@pytest.mark.parametrize(
    ("country_value", "local_values", "message"),
    [
        (10.0, (0.0, 0.0), "zero-valued lower leg"),
        (-10.0, (4.0, 6.0), "opposite-signed"),
    ],
)
def test_invalid_factor_math_is_refused(country_value, local_values, message):
    surface = pd.DataFrame(
        [
            ("country", "UK", "national", country_value),
            ("constituency", "E1", "local", local_values[0]),
            ("constituency", "S1", "local", local_values[1]),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    with pytest.raises(ValueError, match=message):
        apply_cross_grain_reconciliation(
            surface,
            ("national",),
            {"national": _signature(), "local": _signature()},
            _rule(),
        )


def test_unparented_and_empty_legs_are_refused():
    unparented = pd.DataFrame(
        [
            ("country", "GB", "national", 10.0),
            ("constituency", "N1", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    signatures = {"national": _signature(), "local": _signature()}
    with pytest.raises(ValueError, match="unparented"):
        apply_cross_grain_reconciliation(
            unparented, ("national",), signatures, _rule()
        )

    empty = pd.DataFrame(
        [
            ("country", "E", "national", 10.0),
            ("country", "S", "national", 5.0),
            ("constituency", "E1", "local", 8.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="empty leg"):
        apply_cross_grain_reconciliation(empty, ("national",), signatures, _rule())


def test_two_different_controls_at_same_grain_are_refused():
    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 10.0),
            ("country", "UK", "national", 11.0),
            ("constituency", "E1", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="two different control values"):
        apply_cross_grain_reconciliation(
            surface,
            ("national",),
            {"national": _signature(), "local": _signature()},
            _rule(),
        )


def test_incompatible_exact_and_bridged_controls_are_refused():
    bridge = CrossGrainBridge(
        "alternate_control",
        "households",
        ("bridged_national",),
        "contract:local",
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "exact_partition", 100.0),
            ("country", "UK", "bridged_national", 90.0),
            ("constituency", "E1", "local", 50.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    signatures = {
        "exact_partition": _signature(),
        "local": _signature(),
        "bridged_national": _signature("alternate"),
    }

    with pytest.raises(ValueError, match="two different control values"):
        apply_cross_grain_reconciliation(
            surface,
            ("exact_partition", "bridged_national"),
            signatures,
            _rule(bridges=(bridge,)),
        )


def test_non_finite_targets_are_refused():
    surface = pd.DataFrame(
        [("constituency", "E1", "local", np.nan)],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="finite"):
        apply_cross_grain_reconciliation(
            surface, (), {"local": _signature()}, _rule()
        )
