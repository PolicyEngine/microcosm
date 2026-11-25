"""Shared target-loss rules: weights and caps from what each row is (#1729)."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.calibrate import relative_error_loss
from microcosm.calibrate.target_loss_rules import (
    AMOUNT_BASIS,
    COUNT_BASIS,
    TARGET_WEIGHT_RULES,
    TargetLossRows,
    TargetWeightRule,
    apply_family_multipliers,
    row_cap_census,
    single_cap_equivalent,
    target_loss_caps,
    target_loss_weight_shares,
    target_loss_weights_for_rule,
)


@pytest.fixture
def rows() -> TargetLossRows:
    """Three levels with unequal row counts, families and bases."""

    return TargetLossRows(
        values=[4e6, 9e4, 1.6e5, 2.5e9, 100.0, 400.0, 0.0, 3e7],
        levels=[
            "state",
            "state",
            "state",
            "state",
            "district",
            "district",
            "district",
            "nation",
        ],
        families=["pop", "soi", "soi", "soi", "pop", "pop", "pop", "pop"],
        bases=[
            COUNT_BASIS,
            COUNT_BASIS,
            COUNT_BASIS,
            AMOUNT_BASIS,
            COUNT_BASIS,
            COUNT_BASIS,
            COUNT_BASIS,
            COUNT_BASIS,
        ],
        groups=["a", "b", "b", "c", "d", "d", "d", "e"],
    )


def _share(weights: np.ndarray, mask) -> float:
    return float(weights[np.asarray(mask)].sum() / weights.sum())


def test_the_vocabulary_is_closed() -> None:
    assert sorted(TARGET_WEIGHT_RULES) == [
        "level_equal",
        "level_family_equal",
        "sqrt_value_concept_budget",
        "uniform",
    ]


def test_uniform_weighs_every_target_one(rows) -> None:
    weights = target_loss_weights_for_rule(rows, TargetWeightRule("uniform"))
    np.testing.assert_array_equal(weights, np.ones(len(rows)))


def test_level_equal_gives_each_level_one_share(rows) -> None:
    weights = target_loss_weights_for_rule(rows, TargetWeightRule("level_equal"))
    levels = np.asarray(rows.levels)
    assert weights.sum() == pytest.approx(1.0)
    for level in ("state", "district", "nation"):
        assert _share(weights, levels == level) == pytest.approx(1 / 3)
    # Uniform inside a level.
    assert len(set(weights[levels == "state"].tolist())) == 1


def test_level_family_equal_splits_a_level_across_its_families(rows) -> None:
    weights = target_loss_weights_for_rule(rows, TargetWeightRule("level_family_equal"))
    levels, families = np.asarray(rows.levels), np.asarray(rows.families)
    assert weights.sum() == pytest.approx(1.0)
    # The state level has two families: each takes half of the level's third.
    assert _share(weights, (levels == "state") & (families == "pop")) == (
        pytest.approx(1 / 6)
    )
    assert _share(weights, (levels == "state") & (families == "soi")) == (
        pytest.approx(1 / 6)
    )
    # The three soi rows share their cell equally without count_power.
    np.testing.assert_allclose(weights[1:4], 1 / 18)
    assert _share(weights, levels == "district") == pytest.approx(1 / 3)


def test_count_power_moves_weight_only_between_the_count_rows_of_a_cell(rows) -> None:
    equal = target_loss_weights_for_rule(rows, TargetWeightRule("level_family_equal"))
    powered = target_loss_weights_for_rule(
        rows, TargetWeightRule("level_family_equal", {"count_power": 0.5})
    )
    levels, families = np.asarray(rows.levels), np.asarray(rows.families)
    # Every (level, family) cell keeps its budget.
    for level, family in set(zip(rows.levels, rows.families, strict=True)):
        cell = (levels == level) & (families == family)
        assert powered[cell].sum() == pytest.approx(equal[cell].sum())
    # The amount row of the state soi cell keeps its equal share.
    assert powered[3] == pytest.approx(equal[3])
    # Its two count rows split their budget as sqrt(9e4) : sqrt(1.6e5) = 3 : 4.
    assert powered[1] / powered[2] == pytest.approx(300.0 / 400.0)
    # A zero count weighs as one unit: sqrt(100) : sqrt(400) : 1 = 10 : 20 : 1.
    np.testing.assert_allclose(powered[4:7] / powered[6], [10.0, 20.0, 1.0])


def test_count_power_refuses_a_row_without_a_declared_basis(rows) -> None:
    undeclared = TargetLossRows(
        values=rows.values,
        levels=rows.levels,
        families=rows.families,
        bases=[*rows.bases[:-1], "rate"],
    )
    with pytest.raises(ValueError, match="basis"):
        target_loss_weights_for_rule(
            undeclared, TargetWeightRule("level_family_equal", {"count_power": 0.5})
        )
    # Without count_power the rule reads no basis.
    target_loss_weights_for_rule(undeclared, TargetWeightRule("level_family_equal"))


def test_concept_budget_rule_halves_the_loss_between_the_bases(rows) -> None:
    weights = target_loss_weights_for_rule(
        rows, TargetWeightRule("sqrt_value_concept_budget", {"value_power": 0.5})
    )
    bases = np.asarray(rows.bases)
    assert weights.mean() == pytest.approx(1.0)
    assert (weights > 0).all()
    assert _share(weights, bases == COUNT_BASIS) == pytest.approx(0.5)
    assert _share(weights, bases == AMOUNT_BASIS) == pytest.approx(0.5)
    # The three district rows share one budget group: before the basis step
    # they sum to their largest member, so the largest keeps the largest share.
    assert weights[5] > weights[4] > weights[6]


def test_a_rule_refuses_rows_that_lack_what_it_reads() -> None:
    bare = TargetLossRows(values=[1.0, 2.0])
    with pytest.raises(ValueError, match="levels"):
        target_loss_weights_for_rule(bare, TargetWeightRule("level_equal"))
    with pytest.raises(ValueError, match="groups"):
        target_loss_weights_for_rule(
            TargetLossRows(values=[1.0], bases=[COUNT_BASIS]),
            TargetWeightRule("sqrt_value_concept_budget"),
        )
    with pytest.raises(ValueError, match="2 entries for 1 values"):
        TargetLossRows(values=[1.0], levels=["a", "b"])


@pytest.mark.parametrize(
    ("name", "parameters", "match"),
    [
        ("grain_equal", {}, "must be one of"),
        ("uniform", {"count_power": 0.5}, "takes"),
        ("level_family_equal", {"count_power": -1.0}, "positive finite"),
        ("level_family_equal", {"count_power": True}, "positive finite"),
        ("sqrt_value_concept_budget", {"value_power": None}, "must be a number"),
        ("uniform", {"family_multipliers": {"pop": 0.0}}, "positive and finite"),
    ],
)
def test_a_declaration_refuses_an_unknown_rule_or_parameter(
    name, parameters, match
) -> None:
    with pytest.raises(ValueError, match=match):
        TargetWeightRule(name, parameters)


def test_a_declaration_is_data() -> None:
    rule = TargetWeightRule(
        "level_family_equal",
        {"count_power": 1, "family_multipliers": {"soi": 2, "pop": 1.5}},
    )
    assert rule.to_json() == {
        "rule": "level_family_equal",
        "parameters": {
            "count_power": 1.0,
            "family_multipliers": {"pop": 1.5, "soi": 2.0},
        },
    }
    assert rule.with_family_multipliers(None).to_json() == {
        "rule": "level_family_equal",
        "parameters": {"count_power": 1.0},
    }
    with pytest.raises(TypeError):
        rule.parameters["count_power"] = 2.0  # type: ignore[index]


def test_family_multipliers_apply_to_any_rule_and_renormalize(rows) -> None:
    plain = target_loss_weights_for_rule(rows, TargetWeightRule("level_equal"))
    doubled = target_loss_weights_for_rule(
        rows, TargetWeightRule("level_equal", {"family_multipliers": {"soi": 2.0}})
    )
    families = np.asarray(rows.families)
    assert doubled.mean() == pytest.approx(1.0)
    ratio = doubled / plain
    assert ratio[families == "soi"][0] == pytest.approx(
        2.0 * ratio[families == "pop"][0]
    )
    with pytest.raises(ValueError, match="matches no target family"):
        apply_family_multipliers(plain, rows.families, {"absent": 2.0})


def test_weight_shares_report_each_level_and_family_cell(rows) -> None:
    weights = target_loss_weights_for_rule(rows, TargetWeightRule("level_family_equal"))
    cells = target_loss_weight_shares(rows, weights)
    assert [(cell["level"], cell["family"]) for cell in cells] == [
        ("state", "pop"),
        ("state", "soi"),
        ("district", "pop"),
        ("nation", "pop"),
    ]
    assert sum(cell["loss_share"] for cell in cells) == pytest.approx(1.0)
    assert sum(cell["equal_share"] for cell in cells) == pytest.approx(1.0)
    assert cells[1]["n_targets"] == 3


# ---------------------------------------------------------------------------
# Loss caps
# ---------------------------------------------------------------------------


def test_the_default_cap_stays_one_number() -> None:
    cap = target_loss_caps(1.0)
    assert cap == 1.0 and isinstance(cap, float)
    assert target_loss_caps(10, levels=["a"], families=["b"]) == 10.0


def test_a_cap_by_level_leaves_the_other_rows_at_the_default(rows) -> None:
    caps = target_loss_caps(1.0, levels=rows.levels, by_level={"district": 10.0})
    np.testing.assert_array_equal(caps, [1, 1, 1, 1, 10, 10, 10, 1])


def test_a_family_cap_wins_over_a_level_cap(rows) -> None:
    caps = target_loss_caps(
        1.0,
        levels=rows.levels,
        families=rows.families,
        by_level={"state": 5.0},
        by_family={"soi": 2.0},
    )
    np.testing.assert_array_equal(caps, [5, 2, 2, 2, 1, 1, 1, 1])


def test_a_cap_override_that_matches_no_row_is_refused(rows) -> None:
    with pytest.raises(ValueError, match="matches no target"):
        target_loss_caps(1.0, levels=rows.levels, by_level={"county": 10.0})
    with pytest.raises(ValueError, match="needs each row's level"):
        target_loss_caps(1.0, by_level={"county": 10.0})
    with pytest.raises(ValueError, match="positive and finite"):
        target_loss_caps(1.0, levels=rows.levels, by_level={"state": 0.0})
    with pytest.raises(ValueError, match="positive and finite"):
        target_loss_caps(float("nan"))


def _row_capped_loss(estimates, targets, weights, scales, caps) -> np.ndarray:
    """Each row's weighted contribution under its own cap, written directly."""

    return (
        weights
        * np.minimum(np.abs((estimates - targets) / scales), caps)
        / weights.sum()
    )


def test_equal_caps_call_the_solver_exactly_as_one_cap_does() -> None:
    weights = np.array([0.2, 0.5, 0.3])
    scales = np.array([10.0, 20.0, 1.0])
    equivalent = single_cap_equivalent(np.full(3, 10.0), weights=weights, scales=scales)
    assert equivalent.cap == 10.0
    np.testing.assert_array_equal(equivalent.weights, weights)
    np.testing.assert_array_equal(equivalent.scales, scales)


def test_the_one_cap_loss_equals_the_loss_with_a_cap_for_each_row() -> None:
    rng = np.random.default_rng(1729)
    n_rows = 400
    targets = rng.lognormal(8.0, 2.0, n_rows)
    scales = np.maximum(np.abs(targets), 1.0)
    # Misses from 0.01x to 100x the scale, so many rows sit past either cap.
    estimates = targets + scales * rng.lognormal(0.0, 2.5, n_rows) * rng.choice(
        [-1.0, 1.0], n_rows
    )
    weights = rng.uniform(0.0, 3.0, n_rows)
    weights[:5] = 0.0
    caps = np.where(rng.uniform(size=n_rows) < 0.6, 1.0, 10.0)

    equivalent = single_cap_equivalent(caps, weights=weights, scales=scales)
    direct = _row_capped_loss(estimates, targets, weights, scales, caps)
    transformed = _row_capped_loss(
        estimates,
        targets,
        equivalent.weights,
        equivalent.scales,
        np.full(n_rows, equivalent.cap),
    )

    # Row by row, not only in total.
    np.testing.assert_allclose(transformed, direct, rtol=1e-12, atol=0.0)
    assert equivalent.weights.sum() == pytest.approx(weights.sum(), rel=1e-12)
    # The solver's own loss on the transformed problem is the row-cap loss.
    assert relative_error_loss(
        estimates,
        targets,
        target_loss_weights=equivalent.weights,
        target_loss_scales=equivalent.scales,
        target_loss_cap=equivalent.cap,
    ) == pytest.approx(float(direct.sum()), rel=1e-12)
    # Both caps bind on some rows, so the identity is tested where it matters.
    scaled = np.abs((estimates - targets) / scales)
    assert ((scaled > 1.0) & (caps == 1.0)).any()
    assert ((scaled > 1.0) & (scaled < 10.0) & (caps == 10.0)).any()
    assert equivalent.receipt()["rows_by_cap"] == {
        "1.0": int((caps == 1.0).sum()),
        "10.0": int((caps == 10.0).sum()),
    }


def test_the_equivalent_refuses_misaligned_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        single_cap_equivalent([1.0, 2.0], weights=[1.0], scales=[1.0])
    with pytest.raises(ValueError, match="caps must be finite and positive"):
        single_cap_equivalent([1.0, 0.0], weights=[1.0, 1.0], scales=[1.0, 1.0])
    with pytest.raises(ValueError, match="positive total weight"):
        single_cap_equivalent([1.0, 2.0], weights=[0.0, 0.0], scales=[1.0, 1.0])


def test_the_census_reads_each_row_against_its_own_cap() -> None:
    names = ["state_far", "county_far", "county_escaped", "state_pushed", "at_cap"]
    targets = np.asarray([100.0, 100.0, 100.0, 100.0, 100.0])
    scales = np.asarray([100.0, 100.0, 100.0, 100.0, 100.0])
    caps = np.asarray([1.0, 10.0, 10.0, 1.0, 1.0])
    # Scaled misses at the start: 2, 2, 12, 0.5, 1. At the end: 2, 1.5, 3, 4, 1.
    initial = np.asarray([300.0, 300.0, 1300.0, 150.0, 200.0])
    final = np.asarray([300.0, 250.0, 400.0, 500.0, 200.0])

    census = row_cap_census(names, targets, initial, final, scales=scales, caps=caps)

    assert census["cap_basis"] == "each row's own cap"
    assert census["n_targets"] == 5
    # A miss of 2 is past a cap of 1 and inside a cap of 10.
    assert census["initial_past_cap"] == 2  # state_far, county_escaped
    assert census["final_past_cap"] == 2  # state_far, state_pushed
    assert (census["escaped"], census["frozen"], census["pushed_out"]) == (1, 1, 1)
    assert census["pushed_out_rows"] == [
        {"name": "state_pushed", "cap": 1.0, "init_rel": 0.5, "final_rel": 4.0}
    ]
    assert census["by_cap"]["1.0"] == {
        "n_targets": 3,
        "initial_past_cap": 1,
        "final_past_cap": 2,
        "escaped": 0,
        "frozen": 1,
        "pushed_out": 1,
    }
    assert census["by_cap"]["10.0"]["escaped"] == 1
    # A row exactly at its cap still has a gradient: it is not past the cap.
    assert census["by_cap"]["1.0"]["initial_past_cap"] == 1
    # One cap for every row gives the counts of a single-cap census.
    single = row_cap_census(
        names, targets, initial, final, scales=scales, caps=np.full(5, 1.0)
    )
    assert single["initial_past_cap"] == 3 and single["final_past_cap"] == 4


def test_the_census_refuses_misaligned_or_invalid_inputs() -> None:
    arguments = dict(scales=[1.0, 1.0], caps=[1.0, 1.0])
    with pytest.raises(ValueError, match="there are 2 names"):
        row_cap_census(["a", "b"], [1.0], [1.0, 1.0], [1.0, 1.0], **arguments)
    with pytest.raises(ValueError, match="must be finite"):
        row_cap_census(["a", "b"], [1.0, 1.0], [1.0, np.nan], [1.0, 1.0], **arguments)
    with pytest.raises(ValueError, match="must be positive"):
        row_cap_census(
            ["a", "b"],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            scales=[1.0, 0.0],
            caps=[1.0, 1.0],
        )
