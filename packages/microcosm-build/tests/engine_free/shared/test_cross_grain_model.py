"""The general parts of the cross-grain model (microcosm#1760).

Three things that a country once did in its own code are part of the shared
operator, each off by default:

- the bound of a bridge can be measured from the median factor across the
  bridge's legs, can be one-sided, and can unbind an out-of-bound leg where
  the default refuses the pass;
- ``expected_members`` refuses a leg of a target that does not hold exactly
  its declared members;
- a bridge selector makes one bridge for each pair of targets.

``test_cross_grain_allocator.py`` holds the default path to the operator as
it stood before any of this, bit for bit.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from microcosm.build.cross_grain import (
    CrossGrainBridge,
    CrossGrainBridgeSelector,
    CrossGrainRule,
    apply_cross_grain_reconciliation,
    expand_bridge_selectors,
)

_STATES = ("A", "B", "C", "D", "E")


def _signature(concept: str) -> dict[str, object]:
    return {"measurement": {"concept": concept, "filters": []}}


_SIGNATURES = {"total": _signature("source one"), "parts": _signature("source two")}


def _rule(bridge: CrossGrainBridge | None = None, **overrides) -> CrossGrainRule:
    arguments = {
        "grain_precedence": ("nation", "state", "district"),
        "signature_fields": ("concept", "filters"),
        "bridges": () if bridge is None else (bridge,),
        "leg_of_area": lambda area: area[0],
        "parent_geography_legs": {state: (state,) for state in _STATES},
        "control_grains": ("nation", "state"),
    }
    return CrossGrainRule(**{**arguments, **overrides})


def _bridge(**bound) -> CrossGrainBridge:
    return CrossGrainBridge(
        "total_vs_parts",
        concept="one quantity",
        higher_target_ids=("total",),
        lower_side="parts",
        **bound,
    )


def _surface(factors: dict[str, float], *, parts=(30.0, 70.0)) -> pd.DataFrame:
    """A state total and two districts in each state; the factor is their gap."""

    rows = []
    for state, factor in factors.items():
        rows.append(("state", state, "total", sum(parts) * factor))
        rows += [
            ("district", f"{state}{index}", "parts", value)
            for index, value in enumerate(parts, start=1)
        ]
    return pd.DataFrame(rows, columns=["grain", "geography_id", "target_id", "value"])


_MEDIAN_BAND = {
    "max_factor_shift": 0.25,
    "min_factor_shift": -0.2,
    "shift_around": "median",
}


def test_a_median_bound_takes_out_the_gap_that_every_leg_shares() -> None:
    """Every leg is 2.4 times its rows: far from one, and no shift at all."""

    factors = dict.fromkeys(_STATES, 2.4)
    surface = _surface(factors)
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, ("total",), _SIGNATURES, _rule(_bridge(**_MEDIAN_BAND))
    )
    assert reconciled["value"].tolist() == pytest.approx([240.0, 72.0, 168.0] * 5)
    (group,) = receipt["groups"]
    assert len(group["legs"]) == 5
    for leg in group["legs"]:
        assert leg["factor_reference"] == pytest.approx(2.4)
        assert leg["shift_around"] == "median"
        assert (leg["max_factor_shift"], leg["min_factor_shift"]) == (0.25, -0.2)
    # The same legs under a bound around one are refused.
    with pytest.raises(ValueError, match="max_factor_shift 0.25 from one"):
        apply_cross_grain_reconciliation(
            surface, ("total",), _SIGNATURES, _rule(_bridge(max_factor_shift=0.25))
        )


def test_an_out_of_bound_leg_is_refused_by_default() -> None:
    factors = {"A": 1.0, "B": 1.02, "C": 1.04, "D": 0.98, "E": 2.0}
    with pytest.raises(
        ValueError, match="outside its declared shift from -0.2 to 0.25"
    ) as refused:
        apply_cross_grain_reconciliation(
            _surface(factors), ("total",), _SIGNATURES, _rule(_bridge(**_MEDIAN_BAND))
        )
    assert "leg 'E'" in str(refused.value)


def test_unbind_leaves_the_leg_as_given_and_records_it() -> None:
    factors = {"A": 1.0, "B": 1.02, "C": 1.04, "D": 0.98, "E": 2.0}
    surface = _surface(factors)
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("total",),
        _SIGNATURES,
        _rule(_bridge(**_MEDIAN_BAND, out_of_bound="unbind")),
    )
    values = reconciled["value"].tolist()
    # The four legs in bound close on their controls.
    for index, state in enumerate("ABCD"):
        assert values[3 * index + 1] + values[3 * index + 2] == pytest.approx(
            100.0 * factors[state]
        )
    # The leg out of bound keeps its rows as given.
    assert values[13:15] == [30.0, 70.0]
    (unbound,) = receipt["unbound_legs"]
    assert (unbound["leg"], unbound["parent_geography_id"]) == ("E", "E")
    assert unbound["positions"] == [13, 14]
    assert unbound["factor_on_input_totals"] == pytest.approx(2.0)
    assert unbound["factor_reference"] == pytest.approx(1.02)
    assert unbound["relative_factor"] == pytest.approx(2.0 / 1.02)
    (group,) = receipt["groups"]
    assert sorted(leg["leg"] for leg in group["legs"]) == ["A", "B", "C", "D"]


def test_the_bound_is_one_sided_when_it_declares_a_floor() -> None:
    low, high = {"A": 1.0, "B": 0.85}, {"A": 1.0, "B": 1.15}
    bound = {"max_factor_shift": 0.2, "min_factor_shift": -0.1}
    # 0.85 is below the floor of -0.1; 1.15 is inside the ceiling of 0.2.
    with pytest.raises(ValueError, match="outside its declared shift"):
        apply_cross_grain_reconciliation(
            _surface(low), ("total",), _SIGNATURES, _rule(_bridge(**bound))
        )
    reconciled, _receipt = apply_cross_grain_reconciliation(
        _surface(high), ("total",), _SIGNATURES, _rule(_bridge(**bound))
    )
    assert reconciled["value"].tolist()[4:6] == pytest.approx([34.5, 80.5])


def test_a_leg_that_sums_to_zero_under_a_zero_control_is_not_judged() -> None:
    surface = _surface({"A": 1.0, "B": 1.1, "C": 0.9})
    surface.loc[surface["geography_id"].isin(["C", "C1", "C2"]), "value"] = 0.0
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("total",),
        _SIGNATURES,
        _rule(_bridge(**_MEDIAN_BAND, out_of_bound="unbind")),
    )
    assert receipt["unbound_legs"] == []
    assert reconciled["value"].tolist()[6:] == [0.0, 0.0, 0.0]


def test_with_no_positive_factor_the_reference_is_missing_and_all_unbind() -> None:
    """A zero control over rows that are not zero: no leg gives a reference."""

    surface = _surface({"A": 0.0, "B": 0.0})
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface,
        ("total",),
        _SIGNATURES,
        _rule(_bridge(**_MEDIAN_BAND, out_of_bound="unbind")),
    )
    assert [leg["leg"] for leg in receipt["unbound_legs"]] == ["A", "B"]
    assert all(math.isnan(leg["factor_reference"]) for leg in receipt["unbound_legs"])
    assert reconciled["value"].tolist() == surface["value"].tolist()


def test_a_pass_that_does_not_check_bounds_unbinds_nothing() -> None:
    factors = {"A": 1.0, "B": 1.02, "C": 2.0}
    reconciled, receipt = apply_cross_grain_reconciliation(
        _surface(factors),
        ("total",),
        _SIGNATURES,
        _rule(_bridge(**_MEDIAN_BAND, out_of_bound="unbind")),
        check_factor_bounds=False,
    )
    assert receipt["unbound_legs"] == []
    assert reconciled["value"].tolist()[7:9] == pytest.approx([60.0, 140.0])


def test_only_a_rule_that_unbinds_carries_the_unbound_legs_key() -> None:
    surface = _surface(dict.fromkeys(_STATES, 1.0))
    for bridge in (_bridge(), _bridge(max_factor_shift=0.1), _bridge(**_MEDIAN_BAND)):
        _reconciled, receipt = apply_cross_grain_reconciliation(
            surface, ("total",), _SIGNATURES, _rule(bridge)
        )
        assert "unbound_legs" not in receipt
        (leg, *_others) = receipt["groups"][0]["legs"]
        general = bridge.shift_around != "one"
        assert ("factor_reference" in leg) == general


@pytest.mark.parametrize(
    ("bound", "message"),
    [
        ({"shift_around": "mean", "max_factor_shift": 0.1}, "shift_around"),
        ({"out_of_bound": "drop", "max_factor_shift": 0.1}, "out_of_bound"),
        ({"shift_around": "median"}, "shapes a bound it does not declare"),
        ({"out_of_bound": "unbind"}, "shapes a bound it does not declare"),
        ({"min_factor_shift": -0.1}, "shapes a bound it does not declare"),
        ({"max_factor_shift": 0.1, "min_factor_shift": 0.2}, "min_factor_shift"),
        ({"max_factor_shift": 0.1, "min_factor_shift": -1.0}, "min_factor_shift"),
    ],
)
def test_a_malformed_bound_is_refused(bound, message) -> None:
    with pytest.raises(ValueError, match=message):
        apply_cross_grain_reconciliation(
            _surface({"A": 1.0}), ("total",), _SIGNATURES, _rule(_bridge(**bound))
        )


def _members(**by_state) -> dict[str, dict[str, frozenset[str]]]:
    return {"parts": {state: frozenset(areas) for state, areas in by_state.items()}}


def test_expected_members_refuse_a_leg_that_is_not_complete() -> None:
    surface = _surface({"A": 1.1, "B": 1.1})
    complete = _members(A=["A1", "A2"], B=["B1", "B2"])
    reconciled, _receipt = apply_cross_grain_reconciliation(
        surface, ("total",), _SIGNATURES, _rule(_bridge()), expected_members=complete
    )
    assert reconciled["value"].tolist()[1:3] == pytest.approx([33.0, 77.0])

    with pytest.raises(ValueError, match=r"leg 'B': missing \['B3'\], unexpected \[\]"):
        apply_cross_grain_reconciliation(
            surface,
            ("total",),
            _SIGNATURES,
            _rule(_bridge()),
            expected_members=_members(A=["A1", "A2"], B=["B1", "B2", "B3"]),
        )
    with pytest.raises(ValueError, match=r"missing \[\], unexpected \['A2'\]"):
        apply_cross_grain_reconciliation(
            surface,
            ("total",),
            _SIGNATURES,
            _rule(_bridge()),
            expected_members=_members(A=["A1"], B=["B1", "B2"]),
        )
    # A leg that the target has rows on must declare its members.
    with pytest.raises(ValueError, match="declares no members"):
        apply_cross_grain_reconciliation(
            surface,
            ("total",),
            _SIGNATURES,
            _rule(_bridge()),
            expected_members=_members(A=["A1", "A2"]),
        )


def test_a_declared_leg_holds_its_members_with_or_without_rows() -> None:
    surface = _surface({"A": 1.1, "B": 1.1})
    without_b = surface[~surface["geography_id"].isin(["B1", "B2"])].reset_index(
        drop=True
    )
    members = _members(A=["A1", "A2"], B=["B1", "B2"])
    with pytest.raises(ValueError, match=r"leg 'B': missing \['B1', 'B2'\]"):
        apply_cross_grain_reconciliation(
            without_b,
            ("total",),
            _SIGNATURES,
            _rule(_bridge()),
            expected_members=members,
        )
    # A target with no row at all forms no group. Its declared legs are still
    # read, so the pass cannot leave the totals as they are without a word.
    totals_only = surface[surface["target_id"] == "total"].reset_index(drop=True)
    with pytest.raises(ValueError, match=r"leg 'A': missing \['A1', 'A2'\]"):
        apply_cross_grain_reconciliation(
            totals_only,
            ("total",),
            _SIGNATURES,
            _rule(_bridge()),
            expected_members=members,
        )
    # A leg that declares no member holds no row. The licences decide it.
    empty_b = _members(A=["A1", "A2"], B=[])
    with pytest.raises(ValueError, match="lacks a licence"):
        apply_cross_grain_reconciliation(
            without_b,
            ("total",),
            _SIGNATURES,
            _rule(_bridge()),
            expected_members=empty_b,
        )
    _reconciled, receipt = apply_cross_grain_reconciliation(
        without_b,
        ("total",),
        _SIGNATURES,
        _rule(_bridge()),
        expected_members=empty_b,
        licensed_empty_legs={"parts": frozenset({"B"})},
    )
    assert [entry["leg"] for entry in receipt["empty_legs_licensed"]] == ["B"]


_TARGETS = {
    "ht2:agi": {"source": "table_2.broad", "grain": "state", "measure": "agi"},
    "ht2:qbi": {"source": "table_2.broad", "grain": "state", "measure": "qbi"},
    "ht2:wages": {"source": "table_2.broad", "grain": "state", "measure": "wages"},
    "cd:agi": {"source": "district.2022", "grain": "district", "measure": "agi"},
    "cd:qbid": {"source": "district.2022", "grain": "district", "measure": "qbid"},
    "cd:eitc": {"source": "district.2022", "grain": "district", "measure": "eitc"},
    "other:agi": {"source": "table_1", "grain": "nation", "measure": "agi"},
}


def _selector(**overrides) -> CrossGrainBridgeSelector:
    arguments = {
        "bridge_id": "table_2_vs_district",
        "concept": "one tax quantity",
        "higher": {"source": {"prefix": "table_2."}, "grain": "state"},
        "lower": {"source": {"prefix": "district."}},
        "pair_on": ("measure",),
        "paired_values": {"measure": {"qbi": "qbid"}},
        **_MEDIAN_BAND,
        "out_of_bound": "unbind",
    }
    return CrossGrainBridgeSelector(**{**arguments, **overrides})


def test_a_selector_makes_one_bridge_for_each_pair() -> None:
    bridges, unpaired = expand_bridge_selectors([_selector()], _TARGETS)
    assert [
        (bridge.bridge_id, bridge.higher_target_ids, bridge.lower_side)
        for bridge in bridges
    ] == [
        ("table_2_vs_district:cd:agi", ("ht2:agi",), "cd:agi"),
        ("table_2_vs_district:cd:qbid", ("ht2:qbi",), "cd:qbid"),
    ]
    # A lower target with no higher target is named, never bridged.
    assert unpaired == {"table_2_vs_district": ("cd:eitc",)}
    # Each bridge takes the selector's concept and bound.
    for bridge in bridges:
        assert bridge.concept == "one tax quantity"
        assert (
            bridge.max_factor_shift,
            bridge.min_factor_shift,
            bridge.shift_around,
            bridge.out_of_bound,
        ) == (0.25, -0.2, "median", "unbind")


def test_a_selector_refuses_two_controls_for_one_lower_target() -> None:
    targets = {**_TARGETS, "ht2b:agi": {**_TARGETS["ht2:agi"]}}
    with pytest.raises(ValueError, match="pairs two higher targets"):
        expand_bridge_selectors([_selector()], targets)
    with pytest.raises(ValueError, match="must name the attributes"):
        expand_bridge_selectors([_selector(pair_on=())], _TARGETS)


def test_selected_bridges_run_on_the_operator() -> None:
    bridges, _unpaired = expand_bridge_selectors([_selector()], _TARGETS)
    rows = []
    for state, factor in {"A": 1.02, "B": 1.0, "C": 1.04, "D": 3.0}.items():
        for higher, lower in (("ht2:agi", "cd:agi"), ("ht2:qbi", "cd:qbid")):
            rows.append(("state", state, higher, 100.0 * factor))
            rows += [("district", f"{state}{i}", lower, 50.0) for i in (1, 2)]
    surface = pd.DataFrame(
        rows, columns=["grain", "geography_id", "target_id", "value"]
    )
    signatures = {target: _signature(target) for target in _TARGETS}
    reconciled, receipt = apply_cross_grain_reconciliation(
        surface, ("ht2:agi", "ht2:qbi"), signatures, _rule(bridges=bridges)
    )
    assert len(receipt["groups"]) == 2
    # State D is out of the band in both bridges; its rows keep their values.
    assert sorted(
        (leg["bridge_id"], leg["leg"]) for leg in receipt["unbound_legs"]
    ) == [
        ("table_2_vs_district:cd:agi", "D"),
        ("table_2_vs_district:cd:qbid", "D"),
    ]
    values = {
        (area, target): value
        for area, target, value in zip(
            surface["geography_id"],
            surface["target_id"],
            reconciled["value"],
            strict=True,
        )
    }
    assert values[("A1", "cd:agi")] == pytest.approx(51.0)
    assert values[("D1", "cd:qbid")] == 50.0
