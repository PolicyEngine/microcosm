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


def test_reviewed_partial_partition_is_unbound_with_receipt_and_no_reconciliation():
    bridge = CrossGrainBridge(
        "partition",
        "households",
        ("part_a", "part_b", "part_c"),
        "external:census/households",
    )
    surface = pd.DataFrame(
        [
            ("country", "UK", "part_a", 10.0),
            ("constituency", "E1", "external:census/households", 30.0),
            ("constituency", "W1", "external:census/households", 20.0),
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
        },
        _rule(bridges=(bridge,)),
        reviewed_unbound_higher_targets=reviewed,
    )

    pd.testing.assert_frame_equal(reconciled, original)
    assert receipt["groups"] == []
    assert receipt["inconsistencies_in_force"] == []
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
        "external:census/households",
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
