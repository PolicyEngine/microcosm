from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainPartition,
    CrossGrainRule,
    apply_cross_grain_partitions,
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


def test_bridge_sums_exhaustive_higher_partition_to_one_contract_control():
    bridge = CrossGrainBridge(
        "partition_vs_contract",
        concept="households",
        higher_target_ids=("part_a", "part_b"),
        lower_side="contract:census_households",
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "part_a", 30.0),
            ("country", "UK", "part_b", 70.0),
            ("constituency", "E1", "census_households", 30.0),
            ("constituency", "W1", "census_households", 20.0),
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
    assert receipt["groups"][0]["bridge_id"] == "partition_vs_contract"
    assert receipt["groups"][0]["legs"][0]["higher_target_ids"] == [
        "part_a",
        "part_b",
    ]


def test_bridge_rescales_constituency_and_la_to_same_country_control():
    bridge = CrossGrainBridge(
        "national_age_vs_local_age",
        concept="people",
        higher_target_ids=("national_age",),
        lower_side="contract:local_age",
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "national_age", 100.0),
            ("constituency", "E1", "local_age", 60.0),
            ("constituency", "S1", "local_age", 39.999),
            ("la", "E9", "local_age", 50.0),
            ("la", "S9", "local_age", 50.001),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national_age",),
        {
            "national_age": {
                "measurement": {
                    "concept": "people",
                    "entity": "household",
                    "map_to": None,
                    "filters": [{"age": {"minimum": 0, "maximum": 9}}],
                }
            },
            "local_age": {
                "measurement": {
                    "concept": "people",
                    "entity": "household",
                    "map_to": None,
                    "filters": [{"age": {"lower": 0, "upper": 10}}],
                }
            },
        },
        _rule(bridges=(bridge,)),
    )

    assert reconciled.loc[reconciled["grain"] == "constituency", "value"].sum() == (
        pytest.approx(100.0)
    )
    assert reconciled.loc[reconciled["grain"] == "la", "value"].sum() == pytest.approx(
        100.0
    )
    bridge_groups = [
        group
        for group in receipt["groups"]
        if group["bridge_id"] == "national_age_vs_local_age"
    ]
    assert {group["legs"][0]["reason"] for group in bridge_groups} == {
        "standing cross-grain rule: country controls constituency",
        "standing cross-grain rule: country controls la",
    }
    assert all(
        group["bridge_id"] == "national_age_vs_local_age" for group in bridge_groups
    )
    assert all(
        group["legs"][0]["declared_factor"] == pytest.approx(1.0, abs=2e-5)
        for group in bridge_groups
    )


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
        "unbound_bridges": [],
        "empty_legs_licensed": [],
        "controls_without_lower_rows": [],
        "delegated_legs": [],
        "absent_middle_tier_legs": [],
        "absence": "No cross-grain inconsistencies are in force on this surface.",
    }


def test_partially_bound_declared_partition_is_refused():
    bridge = CrossGrainBridge(
        "partition",
        "households",
        ("part_a", "part_b"),
        "contract:census_households",
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


def test_reviewed_unbound_bridge_keeps_the_exact_signature_group():
    bridge = CrossGrainBridge(
        "partition",
        "households",
        ("part_a", "part_b", "part_c"),
        "contract:census_households",
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "part_a", 10.0),
            ("constituency", "E1", "census_households", 30.0),
            ("constituency", "W1", "census_households", 20.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    original = surface.copy(deep=True)
    reviewed = {
        "part_b": {"tracking": "microcosm#791", "reason": "unmeasurable"},
        "part_c": {"tracking": "microcosm#791", "reason": "unmeasurable"},
    }

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("part_a",),
        {
            "part_a": _signature(),
            "part_b": _signature(),
            "part_c": _signature(),
            "census_households": _signature(),
        },
        _rule(bridges=(bridge,)),
        reviewed_unbound_higher_targets=reviewed,
    )

    pd.testing.assert_frame_equal(surface, original)
    assert reconciled["value"].tolist() == [10.0, 6.0, 4.0]
    assert len(receipt["groups"]) == 1
    assert receipt["groups"][0]["bridge_id"] is None
    assert receipt["groups"][0]["legs"][0]["higher_target_ids"] == ["part_a"]
    assert receipt["unbound_bridges"] == [
        {
            "bridge_id": "partition",
            "missing": ["part_b", "part_c"],
            "basis": "reviewed_exclusion",
            "records": reviewed,
        }
    ]


def test_partial_partition_with_unreviewed_member_names_it_in_refusal():
    bridge = CrossGrainBridge(
        "partition",
        "households",
        ("part_a", "part_b", "part_c"),
        "contract:census_households",
    )
    surface = pd.DataFrame(
        [("country", "UK", "part_a", 10.0)],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    with pytest.raises(
        ValueError,
        match=r"lack a reviewed exclusion.*part_c",
    ):
        detect_cross_grain_inconsistencies(
            surface,
            ("part_a",),
            {
                "part_a": _signature(),
                "part_b": _signature(),
                "part_c": _signature(),
            },
            _rule(bridges=(bridge,)),
            reviewed_unbound_higher_targets={"part_b": {"tracking": "microcosm#791"}},
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


def test_external_target_side_is_refused() -> None:
    bridge = CrossGrainBridge(
        "external_hatch",
        "households",
        ("national",),
        "external:census/households",
    )
    surface = pd.DataFrame(
        [("constituency", "E1", "local", 10.0)],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    with pytest.raises(ValueError, match="forbidden external side"):
        detect_cross_grain_inconsistencies(
            surface,
            (),
            {"national": _signature(), "local": _signature()},
            _rule(bridges=(bridge,)),
        )


@pytest.mark.parametrize(
    ("country_value", "local_values", "message"),
    [
        (10.0, (0.0, 0.0), "vanishing lower-leg total"),
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
        apply_cross_grain_reconciliation(unparented, ("national",), signatures, _rule())

    empty = pd.DataFrame(
        [
            ("country", "E", "national", 10.0),
            ("country", "S", "national", 5.0),
            ("constituency", "E1", "local", 8.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="empty leg.*lacks a licence"):
        apply_cross_grain_reconciliation(empty, ("national",), signatures, _rule())


def test_empty_leg_licensed_for_every_lower_target_is_receipted_and_skipped():
    surface = pd.DataFrame(
        [
            ("country", "E", "national", 100.0),
            ("country", "S", "national", 50.0),
            ("constituency", "E1", "local_a", 40.0),
            ("constituency", "E2", "local_b", 60.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    signatures = {
        target_id: _signature() for target_id in ("national", "local_a", "local_b")
    }

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        signatures,
        _rule(),
        licensed_empty_legs={
            "local_a": frozenset({"S"}),
            "local_b": frozenset({"S"}),
        },
    )

    assert reconciled["value"].tolist() == [100.0, 50.0, 40.0, 60.0]
    inconsistency_id = receipt["groups"][0]["inconsistency_id"]
    assert receipt["empty_legs_licensed"] == [
        {
            "inconsistency_id": inconsistency_id,
            "parent_geography_id": "S",
            "leg": "S",
            "lower_target_ids": ["local_a", "local_b"],
        }
    ]
    assert receipt["controls_without_lower_rows"] == [
        {
            "inconsistency_id": inconsistency_id,
            "parent_geography_id": "S",
            "covered_legs": ["S"],
            "higher_target_ids": ["national"],
            "lower_target_ids": ["local_a", "local_b"],
        }
    ]
    assert receipt["groups"][0]["legs"][0]["leg"] == "E"


def test_empty_leg_licensed_for_only_one_lower_target_is_refused():
    surface = pd.DataFrame(
        [
            ("country", "E", "national", 100.0),
            ("country", "S", "national", 50.0),
            ("constituency", "E1", "local_a", 40.0),
            ("constituency", "E2", "local_b", 60.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    signatures = {
        target_id: _signature() for target_id in ("national", "local_a", "local_b")
    }

    with pytest.raises(
        ValueError,
        match=r"empty leg.*S.*lacks a licence.*local_b",
    ):
        apply_cross_grain_reconciliation(
            surface,
            ("national",),
            signatures,
            _rule(),
            licensed_empty_legs={"local_a": frozenset({"S"})},
        )


def test_control_with_no_populated_legs_is_receipted_and_dropped():
    surface = pd.DataFrame(
        [
            ("country", "E", "national", 100.0),
            ("country", "S", "national", 50.0),
            ("constituency", "E1", "local", 100.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": _signature(), "local": _signature()},
        _rule(),
        licensed_empty_legs={"local": frozenset({"S"})},
    )

    assert reconciled["value"].tolist() == [100.0, 50.0, 100.0]
    inconsistency_id = receipt["groups"][0]["inconsistency_id"]
    assert receipt["empty_legs_licensed"] == [
        {
            "inconsistency_id": inconsistency_id,
            "parent_geography_id": "S",
            "leg": "S",
            "lower_target_ids": ["local"],
        }
    ]
    assert receipt["controls_without_lower_rows"] == [
        {
            "inconsistency_id": inconsistency_id,
            "parent_geography_id": "S",
            "covered_legs": ["S"],
            "higher_target_ids": ["national"],
            "lower_target_ids": ["local"],
        }
    ]
    assert receipt["groups"][0]["legs"][0]["parent_geography_id"] == "E"


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
        apply_cross_grain_reconciliation(surface, (), {"local": _signature()}, _rule())


def test_off_control_reconciliation_is_refused(monkeypatch):
    """Fault injection: if a rescaled leg lands off its control, the pass fails.

    Closure is the property the pass exists to establish, so it is asserted
    after the write rather than inferred from the factor arithmetic.
    """

    from microcosm.build import cross_grain as module

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 120.0),
            ("constituency", "E1", "local", 40.0),
            ("constituency", "S1", "local", 20.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    signatures = {"national": _signature(), "local": _signature()}

    monkeypatch.setattr(module.np, "isclose", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="off its control"):
        apply_cross_grain_reconciliation(surface, ("national",), signatures, _rule())


def test_closure_holds_across_many_legs_with_awkward_floats():
    """The closure assertion must not false-positive on ordinary float drift."""

    rows = [("country", "UK", "national", 28_356_000.0)]
    values = [1234.567_89 + index * 3.141_59 for index in range(200)]
    for index, value in enumerate(values):
        leg = "EWSN"[index % 4]
        rows.append(("constituency", f"{leg}{index}", "local", value))
    surface = pd.DataFrame(
        rows, columns=["grain", "geography_id", "target_id", "value"]
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": _signature(), "local": _signature()},
        _rule(),
    )

    local = reconciled.loc[reconciled["grain"] == "constituency", "value"]
    assert local.sum() == pytest.approx(28_356_000.0, rel=1e-12)
    assert receipt["groups"][0]["legs"][0]["new_total"] == pytest.approx(
        28_356_000.0, rel=1e-12
    )


def test_absent_and_empty_signature_spellings_group_together():
    """`filters: []` and an omitted `filters` are the same measurement."""

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 120.0),
            ("constituency", "E1", "local", 40.0),
            ("constituency", "S1", "local", 20.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    spelled_empty = {
        "measurement": {
            "concept": "households",
            "entity": "household",
            "map_to": None,
            "filters": [],
        }
    }
    spelled_absent = {"measurement": {"concept": "households", "entity": "household"}}

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": spelled_empty, "local": spelled_absent},
        _rule(),
    )

    assert len(receipt["groups"]) == 1
    assert reconciled.loc[1:, "value"].tolist() == [80.0, 40.0]


def test_filter_order_does_not_split_a_signature():
    conditions = [
        {"concept": "uk.benefits.universal_credit.amount", "op": ">", "value": 0},
        {"concept": "uk.household.tenure", "op": "==", "value": "rented"},
    ]
    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 100.0),
            ("constituency", "E1", "local", 25.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    forward = {
        "measurement": {
            "concept": "households",
            "entity": "household",
            "filters": conditions,
        }
    }
    reversed_order = {
        "measurement": {
            "concept": "households",
            "entity": "household",
            "filters": list(reversed(conditions)),
        }
    }

    _, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": forward, "local": reversed_order},
        _rule(),
    )

    assert len(receipt["groups"]) == 1


def test_contract_entry_without_measurement_block_is_refused():
    surface = pd.DataFrame(
        [("constituency", "E1", "local", 10.0)],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="must carry a 'measurement' mapping"):
        apply_cross_grain_reconciliation(
            surface,
            (),
            {"local": {"concept": "households", "entity": "household"}},
            _rule(),
        )


def test_vanishing_lower_leg_total_is_refused():
    surface = pd.DataFrame(
        [
            ("country", "E", "national", 100.0),
            ("constituency", "E1", "local", 1e-18),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="vanishing lower-leg total"):
        apply_cross_grain_reconciliation(
            surface,
            ("national",),
            {"national": _signature(), "local": _signature()},
            _rule(),
        )


def test_offsetting_members_reconcile_when_the_leg_total_is_well_conditioned():
    """A net-valued concept may legitimately hold offsetting members.

    Conditioning is the only hazard, and the relative floor is that test, so a
    leg summing cleanly to its control must reconcile rather than be refused
    for containing both signs.
    """

    surface = pd.DataFrame(
        [
            ("country", "E", "national", 100.0),
            ("constituency", "E1", "local", 200.0),
            ("constituency", "E2", "local", -100.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": _signature(), "local": _signature()},
        _rule(),
    )

    assert receipt["groups"][0]["legs"][0]["declared_factor"] == pytest.approx(1.0)
    assert reconciled.loc[1:, "value"].tolist() == [200.0, -100.0]


def test_cancelling_leg_is_still_refused_by_the_conditioning_floor():
    surface = pd.DataFrame(
        [
            ("country", "E", "national", 100.0),
            ("constituency", "E1", "local", 1e-3),
            ("constituency", "E2", "local", -1e-3),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="vanishing lower-leg total"):
        apply_cross_grain_reconciliation(
            surface,
            ("national",),
            {"national": _signature(), "local": _signature()},
            _rule(),
        )


def test_zero_control_over_a_zero_summing_leg_is_a_no_op():
    """The zero branch is live: it is the 0/0 case, not a dead special case."""

    surface = pd.DataFrame(
        [
            ("country", "E", "national", 0.0),
            ("constituency", "E1", "local", 5.0),
            ("constituency", "E2", "local", -5.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": _signature(), "local": _signature()},
        _rule(),
    )

    assert receipt["groups"][0]["legs"][0]["declared_factor"] == pytest.approx(1.0)
    assert reconciled.loc[1:, "value"].tolist() == [5.0, -5.0]


def test_ordered_payload_inside_a_filter_does_not_false_collide():
    """`between [0, 100]` and `between [100, 0]` are different measurements.

    Order-insensitivity is justified at the conjunction level the filter list
    occupies; applying it inside a condition would merge two distinct
    measurements into one group and rescale onto the wrong control.
    """

    def _between(lower: float, upper: float) -> dict[str, object]:
        return {
            "measurement": {
                "concept": "households",
                "entity": "household",
                "filters": [{"op": "between", "value": [lower, upper]}],
            }
        }

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 100.0),
            ("constituency", "E1", "local", 25.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )

    _, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        {"national": _between(0.0, 100.0), "local": _between(100.0, 0.0)},
        _rule(),
    )

    assert receipt["groups"] == []
    assert receipt["absence"]


def _tier_leg(area: str) -> str:
    # Region areas name their leg with their first two characters (R1a -> R1);
    # nation areas name their nation (Sa -> S).
    return area[:2] if area.startswith("R") else area[0]


def _tiered_rule() -> CrossGrainRule:
    return CrossGrainRule(
        grain_precedence=("country", "region", "la"),
        signature_fields=("concept", "entity", "map_to", "filters"),
        bridges=(),
        leg_of_area=_tier_leg,
        parent_geography_legs={
            "UK": ("R1", "R2", "S"),
            "S": ("S",),
            "R1": ("R1",),
            "R2": ("R2",),
        },
        control_grains=("country", "region"),
    )


def _tiered_signatures() -> dict[str, dict[str, object]]:
    return {
        target_id: _signature()
        for target_id in ("national", "regional", "scottish", "local")
    }


def _pairs(receipt: dict) -> dict[str, dict]:
    return {
        group["inconsistency_id"].rsplit(":", 1)[1]: group
        for group in receipt["groups"]
    }


def test_control_grains_let_region_and_country_controls_share_one_group():
    """A region row parents its own authorities; a country row the rest."""

    surface = pd.DataFrame(
        [
            ("country", "S", "scottish", 40.0),
            ("region", "R1", "regional", 100.0),
            ("region", "R2", "regional", 50.0),
            ("la", "R1a", "local", 30.0),
            ("la", "R1b", "local", 30.0),
            ("la", "R2a", "local", 20.0),
            ("la", "Sa", "local", 30.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, ("scottish", "regional"), _tiered_signatures(), _tiered_rule()
    )
    assert reconciled["value"].tolist() == pytest.approx(
        [40.0, 100.0, 50.0, 50.0, 50.0, 50.0, 40.0]
    )
    pairs = _pairs(receipt)
    assert set(pairs) == {"country_over_region", "region_over_la", "country_over_la"}
    # The Scottish row has no region rows on its leg: not a licence matter,
    # it parents the Scottish authority directly.
    assert pairs["country_over_region"]["legs"] == []
    assert [e["parent_geography_id"] for e in receipt["absent_middle_tier_legs"]] == [
        "S"
    ]
    assert [leg["parent_geography_id"] for leg in pairs["region_over_la"]["legs"]] == [
        "R1",
        "R2",
    ]
    assert [leg["parent_geography_id"] for leg in pairs["country_over_la"]["legs"]] == [
        "S"
    ]
    assert receipt["delegated_legs"] == []
    assert receipt["empty_legs_licensed"] == []


def test_nearest_control_claims_first_and_farther_legs_are_delegated():
    """Under a country row, region rows nest to it and authorities follow
    their region; the country row's region legs are delegated, never
    rescaled twice."""

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 200.0),
            ("region", "R1", "regional", 60.0),
            ("region", "R2", "regional", 40.0),
            ("la", "R1a", "local", 10.0),
            ("la", "R2a", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national", "regional"),
        _tiered_signatures(),
        _tiered_rule(),
        licensed_empty_legs={"local": frozenset({"S"})},
    )
    # Region rows rescale jointly to the UK row (100 -> 200); authorities
    # follow their reconciled region.
    assert reconciled["value"].tolist() == pytest.approx(
        [200.0, 120.0, 80.0, 120.0, 80.0]
    )
    pairs = _pairs(receipt)
    assert [
        leg["parent_geography_id"] for leg in pairs["country_over_region"]["legs"]
    ] == ["UK"]
    assert [leg["parent_geography_id"] for leg in pairs["region_over_la"]["legs"]] == [
        "R1",
        "R2",
    ]
    assert pairs["country_over_la"]["legs"] == []
    assert [
        (e["parent_geography_id"], e["legs"]) for e in receipt["delegated_legs"]
    ] == [("UK", ["R1", "R2"])]


def test_a_control_split_between_delegated_and_live_legs_is_refused():
    """A UK row over one region row, with authorities on a leg the region tier
    does not cover: the partial region tier is refused before the country row
    could be split across tiers (microcosm#1123)."""

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 200.0),
            ("region", "R1", "regional", 60.0),
            ("la", "R1a", "local", 10.0),
            ("la", "Sa", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="partial tier"):
        apply_cross_grain_reconciliation(
            surface, ("national", "regional"), _tiered_signatures(), _tiered_rule()
        )


def test_unbound_middle_tier_rows_are_not_controls():
    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 200.0),
            ("region", "R1", "regional", 60.0),
            ("la", "R1a", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national",),
        _tiered_signatures(),
        _tiered_rule(),
        licensed_empty_legs={"local": frozenset({"R2", "S"})},
    )
    # The unbound region row is out of play: the authority reconciles to the
    # UK row and the region row is untouched.
    assert reconciled["value"].tolist() == pytest.approx([200.0, 60.0, 200.0])
    assert {g["winning_grain"] for g in receipt["groups"]} == {"country"}


def test_leaf_rows_without_any_covering_control_still_refuse():
    surface = pd.DataFrame(
        [
            ("region", "R1", "regional", 60.0),
            ("la", "R1a", "local", 10.0),
            ("la", "Sa", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="unparented lower-grain leg"):
        apply_cross_grain_reconciliation(
            surface, ("regional",), _tiered_signatures(), _tiered_rule()
        )


def test_control_grains_declaration_is_validated():
    empty = pd.DataFrame(columns=["grain", "geography_id", "target_id", "value"])
    with pytest.raises(ValueError, match="not in grain_precedence"):
        apply_cross_grain_reconciliation(
            empty,
            (),
            {},
            CrossGrainRule(
                grain_precedence=("country", "la"),
                signature_fields=("concept",),
                bridges=(),
                leg_of_area=_tier_leg,
                parent_geography_legs={},
                control_grains=("country", "region"),
            ),
        )
    with pytest.raises(ValueError, match="must include the top grain"):
        apply_cross_grain_reconciliation(
            empty,
            (),
            {},
            CrossGrainRule(
                grain_precedence=("country", "region", "la"),
                signature_fields=("concept",),
                bridges=(),
                leg_of_area=_tier_leg,
                parent_geography_legs={},
                control_grains=("region",),
            ),
        )


def _nation_rule() -> CrossGrainRule:
    # Legs are the region tier: two English regions plus Wales. England is a
    # nation-grain parent spanning its regions; Wales is its own leg.
    return CrossGrainRule(
        grain_precedence=("country", "nation", "region", "la"),
        signature_fields=("concept", "entity", "map_to", "filters"),
        bridges=(),
        leg_of_area=_tier_leg,
        parent_geography_legs={
            "UK": ("R1", "R2", "W"),
            "E": ("R1", "R2"),
            "W": ("W",),
            "R1": ("R1",),
            "R2": ("R2",),
        },
        control_grains=("country", "nation", "region"),
    )


def _nation_signatures() -> dict[str, dict[str, object]]:
    return {
        target_id: _signature()
        for target_id in ("national", "nations", "regional", "local")
    }


def test_a_nation_row_spans_its_regions_under_a_country_control():
    """England covers both English legs, so the UK row rescales England and
    Wales jointly; England then parents its regions and the regions their
    authorities, each tier closing on the one above."""

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 300.0),
            ("nation", "E", "nations", 160.0),
            ("nation", "W", "nations", 40.0),
            ("region", "R1", "regional", 50.0),
            ("region", "R2", "regional", 50.0),
            ("la", "R1a", "local", 10.0),
            ("la", "R2a", "local", 10.0),
            ("la", "Wa", "local", 10.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("national", "nations", "regional"),
        _nation_signatures(),
        _nation_rule(),
    )
    values = reconciled["value"].tolist()
    assert values == pytest.approx(
        [300.0, 240.0, 60.0, 120.0, 120.0, 120.0, 120.0, 60.0]
    )
    pairs = _pairs(receipt)
    assert [
        leg["parent_geography_id"] for leg in pairs["country_over_nation"]["legs"]
    ] == ["UK"]
    assert [
        leg["parent_geography_id"] for leg in pairs["nation_over_region"]["legs"]
    ] == ["E"]


def test_a_lower_row_straddling_two_controls_is_refused():
    rule = CrossGrainRule(
        grain_precedence=("country", "nation", "la"),
        signature_fields=("concept", "entity", "map_to", "filters"),
        bridges=(),
        leg_of_area=_tier_leg,
        parent_geography_legs={
            "GB": ("R1", "W"),
            "NI": ("N",),
            "E": ("R1",),
            "X": ("W", "N"),
        },
        control_grains=("country", "nation"),
    )
    surface = pd.DataFrame(
        [
            ("country", "GB", "national", 100.0),
            ("country", "NI", "national", 10.0),
            ("nation", "E", "nations", 60.0),
            ("nation", "X", "nations", 30.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="outside control"):
        apply_cross_grain_reconciliation(
            surface, ("national", "nations"), _nation_signatures(), rule
        )


def test_controls_that_differ_only_by_summation_order_agree():
    """An exact control and a bridged control summed from two parts agree when
    they differ in the last bits only (0.1 + 0.2 against 0.3)."""

    bridge = CrossGrainBridge(
        "summed_parts", "households", ("part_a", "part_b"), "contract:local"
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "exact_partition", 0.3),
            ("country", "UK", "part_a", 0.1),
            ("country", "UK", "part_b", 0.2),
            ("constituency", "E1", "local", 1.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    signatures = {
        "exact_partition": _signature(),
        "local": _signature(),
        "part_a": _signature("part_a"),
        "part_b": _signature("part_b"),
    }
    reconciled, _ = apply_cross_grain_reconciliation(
        surface,
        ("exact_partition", "part_a", "part_b"),
        signatures,
        _rule(bridges=(bridge,)),
        licensed_empty_legs={"local": frozenset({"W", "S", "N"})},
    )
    assert reconciled["value"].iloc[3] == pytest.approx(0.3)


def _partition_surface(parent_raw: float, parent: float, members: list[float]):
    rows = [("la", "A1", "total", parent)] + [
        ("la", "A1", f"band_{index}", value) for index, value in enumerate(members)
    ]
    reconciled = pd.DataFrame(
        rows, columns=["grain", "geography_id", "target_id", "value"]
    )
    raw = reconciled.copy()
    raw.loc[0, "value"] = parent_raw
    return raw, reconciled


def test_exhaustive_partition_members_close_on_their_parent():
    raw, reconciled = _partition_surface(90.0, 120.0, [30.0, 30.0, 40.0])
    partition = CrossGrainPartition(
        "bands_sum_to_total", "total", ("band_0", "band_1", "band_2")
    )
    out, receipt = apply_cross_grain_partitions(raw, reconciled, (partition,))
    assert out["value"].tolist() == pytest.approx([120.0, 36.0, 36.0, 48.0])
    assert receipt["partitions"][0]["cells"][0]["declared_factor"] == pytest.approx(1.2)


def test_share_of_parent_members_move_by_the_parents_factor():
    raw, reconciled = _partition_surface(90.0, 120.0, [30.0, 30.0])
    partition = CrossGrainPartition(
        "tenure_moves_with_households",
        "total",
        ("band_0", "band_1"),
        kind="share_of_parent",
    )
    out, _ = apply_cross_grain_partitions(raw, reconciled, (partition,))
    # The parent grew by 120/90; the members keep their 2/3 share of it.
    assert out["value"].tolist() == pytest.approx([120.0, 40.0, 40.0])


def test_a_partition_with_a_missing_member_is_refused():
    raw, reconciled = _partition_surface(90.0, 120.0, [30.0])
    partition = CrossGrainPartition("bands", "total", ("band_0", "band_1"))
    with pytest.raises(ValueError, match="lacks member"):
        apply_cross_grain_partitions(raw, reconciled, (partition,))


def test_a_parent_without_members_is_receipted_not_refused():
    raw, reconciled = _partition_surface(90.0, 120.0, [])
    partition = CrossGrainPartition("bands", "total", ("band_0",))
    out, receipt = apply_cross_grain_partitions(raw, reconciled, (partition,))
    assert out["value"].tolist() == [120.0]
    assert receipt["parents_without_members"] == [
        {"partition_id": "bands", "grain": "la", "geography_id": "A1"}
    ]


def test_partition_declarations_are_validated():
    raw, reconciled = _partition_surface(1.0, 1.0, [1.0])
    with pytest.raises(ValueError, match="unknown kind"):
        apply_cross_grain_partitions(
            raw, reconciled, (CrossGrainPartition("p", "total", ("band_0",), kind="x"),)
        )
    with pytest.raises(ValueError, match="parent as a member"):
        apply_cross_grain_partitions(
            raw, reconciled, (CrossGrainPartition("p", "total", ("total",)),)
        )


def test_a_partial_middle_tier_over_an_unlicensed_empty_leg_is_refused():
    """A UK row over one Welsh nation row and nothing on the English legs:
    without the refusal the Welsh row would take the whole UK total."""

    surface = pd.DataFrame(
        [
            ("country", "UK", "national", 300.0),
            ("nation", "W", "nations", 40.0),
        ],
        columns=["grain", "geography_id", "target_id", "value"],
    )
    with pytest.raises(ValueError, match="partial tier"):
        apply_cross_grain_reconciliation(
            surface, ("national", "nations"), _nation_signatures(), _nation_rule()
        )
