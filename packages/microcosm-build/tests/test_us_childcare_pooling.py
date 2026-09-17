"""Synthetic pooling/evaluation contracts; never load licensed survey records."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.nsece_childcare import NSECEChildcareSource
from microcosm.build.us_runtime.nsece_childcare_pooling import (
    COMPOSITION_CHILDCARE_LEVELS,
    POOLED_CHILDCARE_LEVELS,
    POOLED_CHILDCARE_MATCH_COLUMNS,
    PooledScheduleDonors,
    fit_pooled_sibling_dependence,
    with_childcare_household_size,
)
from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
    assess_sibling_schedules,
)
from microcosm.frame import WeightKind, Weights

MONTH, DAYS, HOURS = US_CHILDCARE_ATTENDANCE_COLUMNS


def _children(households=40):
    records = []
    for household in range(households):
        for child in range(3):
            known = child != 2 or household % 4 != 0
            days = 5.0 if household % 2 else 0.0
            records.append(
                {
                    "donor_id": f"h{household}:c{child}",
                    "source_household_id": f"h{household}",
                    "age": 3,
                    "region": household % 3 + 1,
                    "parent_work_status": 2,
                    "income_band": 1,
                    "attendance_status": "complete" if known else "partial_calendar",
                    "household_weight": 2.0,
                    "child_weight": 2.0,
                    MONTH: (22.0 if days else 0.0) if known else np.nan,
                    DAYS: days if known else np.nan,
                    HOURS: (8.0 if days else 0.0) if known else np.nan,
                }
            )
    return with_childcare_household_size(pd.DataFrame(records))


def test_pooling_levels_have_one_definition_and_retain_age():
    assert POOLED_CHILDCARE_LEVELS[-1] == POOLED_CHILDCARE_MATCH_COLUMNS
    assert POOLED_CHILDCARE_LEVELS[0] == ("age",)
    for smaller, larger in zip(
        POOLED_CHILDCARE_LEVELS, POOLED_CHILDCARE_LEVELS[1:], strict=False
    ):
        assert smaller == larger[:-1]


def test_sparse_cell_blends_distributions_with_household_cluster_support():
    rows = _children().iloc[[0, 1, 3]].copy()
    rows["region"] = [1, 1, 2]
    # Two zero-care child donors come from one household, so local effective
    # support is ONE, not two. The positive root donor has one third of mass.
    values, cumulative, probability = PooledScheduleDonors(rows).distribution(
        rows.iloc[0]
    )
    assert np.average(values[:, 0], weights=probability) == pytest.approx(10 / 33)
    assert probability.sum() == pytest.approx(1)
    assert cumulative[-1] == 1
    assert (probability > 0).all()


def test_entire_household_is_excluded_even_after_unexcluded_cache_lookup():
    rows = _children().iloc[[0, 1, 3]].copy()
    model = PooledScheduleDonors(rows)
    model.distribution(rows.iloc[0])
    values, _, weights = model.distribution(rows.iloc[0], exclude_household="h0")
    assert values.tolist() == [[1, 5, 40]]
    assert weights.tolist() == [1]
    rows.loc[rows.source_household_id.eq("h1"), "age"] = 4
    with pytest.raises(ValueError, match="exact-age"):
        PooledScheduleDonors(rows).distribution(rows.iloc[0], exclude_household="h0")


def test_pooling_is_row_order_and_weight_scale_invariant_and_keeps_joint_schedules():
    children = _children()
    before = children.copy(deep=True)
    target = children.iloc[0].copy()
    target["region"] = 9  # No exact region; use the broader empirical mixture.
    original = PooledScheduleDonors(children).distribution(target)
    changed = children.iloc[::-1].copy()
    changed["child_weight"] *= 1000
    reordered = PooledScheduleDonors(changed).distribution(target)
    for a, b in zip(original, reordered, strict=True):
        np.testing.assert_allclose(a, b)
    assert set(map(tuple, original[0])) == {(0, 0, 0), (1, 5, 40)}
    assert_frame_equal(children, before)


def test_roster_count_includes_unknown_attendance_but_not_older_children():
    children = _children(2).drop(columns="childcare_household_size")
    children.loc[5, "age"] = 13
    counted = with_childcare_household_size(children)
    assert counted.childcare_household_size.tolist() == [3, 3, 3, 2, 2, 2]
    assert children.loc[2, "attendance_status"] == "partial_calendar"
    assert "childcare_household_size" not in children


def test_composition_counts_full_roster_and_never_reads_attendance():
    rows = _children(1)
    rows["age"] = [1, 4, 9]
    rows.loc[2, "attendance_status"] = "partial_calendar"
    counted = with_childcare_household_size(rows)
    assert counted.childcare_under6_count.tolist() == [2, 2, 2]
    assert counted.childcare_schoolage_count.tolist() == [1, 1, 1]
    assert counted.childcare_youngest_age_band.tolist() == [0, 0, 0]
    assert counted.childcare_has_younger_sibling.tolist() == [0, 1, 1]
    rows["attendance_status"] = "missing_calendar"
    changed = with_childcare_household_size(rows)
    columns = COMPOSITION_CHILDCARE_LEVELS[-1]
    assert_frame_equal(counted[list(columns)], changed[list(columns)])


def test_composition_distributions_exclude_the_whole_household_and_retain_age():
    rows = _children()
    model = PooledScheduleDonors(rows, composition=True)
    values, _, weights = model.distribution(rows.iloc[0], exclude_household="h0")
    assert model.levels == COMPOSITION_CHILDCARE_LEVELS
    assert len(values) == len(model.pool.loc[model.pool.source_household_id.ne("h0")])
    assert weights.sum() == pytest.approx(1)
    assert set(map(tuple, values)) == {(0, 0, 0), (1, 5, 40)}


def test_dependence_uses_observed_pairs_in_incomplete_households(monkeypatch):
    original = PooledScheduleDonors.distribution
    exclusions = []

    def checked(self, child, *, exclude_household=None):
        assert exclude_household == child.source_household_id
        exclusions.append(exclude_household)
        return original(self, child, exclude_household=exclude_household)

    monkeypatch.setattr(PooledScheduleDonors, "distribution", checked)
    fit = fit_pooled_sibling_dependence(_children())
    assert fit["households"] == 40
    assert fit["households_with_unresolved_siblings"] == 10
    assert fit["observed_pairs"] == 100
    assert len(exclusions) == 110
    assert 0 <= fit["rho"] <= 1


def test_outer_folds_refit_without_heldout_households_and_score_all_observed_children(
    monkeypatch,
):
    import microcosm.build.us_runtime.nsece_childcare_pooling as module

    children = _children()
    source = NSECEChildcareSource(
        children, Weights(children.child_weight.to_numpy(), WeightKind.DESIGN), {}
    )
    original = module.fit_pooled_sibling_dependence
    fold = 0

    def checked(training, **kwargs):
        nonlocal fold
        heldout = {
            household
            for household in children.source_household_id.unique()
            if int.from_bytes(
                hashlib.sha256(f"271828:{household}".encode()).digest()[:8], "big"
            )
            % 5
            == fold
        }
        assert set(training.source_household_id).isdisjoint(heldout)
        fold += 1
        return original(training, **kwargs)

    monkeypatch.setattr(module, "fit_pooled_sibling_dependence", checked)
    report = assess_sibling_schedules(
        source, pooled=True, include_observed_children=True
    )
    # An observed-zero subgroup has undefined relative-error screens. These
    # must remain failed, JSON-serializable results, not numpy boolean objects.
    json.dumps(report, allow_nan=False)
    assert fold == 5
    groups = {
        row["group"]: row for row in report["observed_child_validation"]["comparisons"]
    }
    assert groups["all"]["children"] == 110
    assert groups["unresolved_siblings"]["children"] == 20
    assert report["larger_households"]["households"] == 30
    assert report["all_observed_sibling_pairs"]["households"] == 40
    assert report["all_observed_sibling_pairs"]["observed_pairs"] == 100
    assert all(part["household_overlap"] == 0 for part in report["splits"])


@pytest.mark.parametrize("strength", [0, -1, np.inf, np.nan])
def test_invalid_pooling_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        PooledScheduleDonors(_children(), strength=strength)


def test_reconstructed_calendars_never_enter_the_experimental_truth_pool():
    children = _children()
    children.loc[0, "attendance_status"] = "summary_bridge"
    with pytest.raises(ValueError, match="original calendars"):
        PooledScheduleDonors(children)


def test_population_moment_objective_matches_its_analytic_coefficient():
    children = _children()
    fitted = fit_pooled_sibling_dependence(children, objective="population_moments")
    residual = np.array(fitted["normalized_mean_residual"])
    increment = np.array(fitted["normalized_mean_shared_increment"])
    optimum = residual @ increment / (increment @ increment)
    assert fitted["unconstrained_rho"] == pytest.approx(optimum)
    assert fitted["rho"] == pytest.approx(np.clip(optimum, 0, 1))
    # A population coefficient is one joint scalar, not a different rho picked
    # for each outcome to force all the evaluation moments to match.
    for alternative in (
        0,
        1,
        max(0, fitted["rho"] - 0.05),
        min(1, fitted["rho"] + 0.05),
    ):
        chosen_loss = np.sum((residual - fitted["rho"] * increment) ** 2)
        other_loss = np.sum((residual - alternative * increment) ** 2)
        assert chosen_loss <= other_loss + 1e-12


def test_dependence_objective_does_not_change_child_donor_distributions():
    children = _children()
    before = children.copy(deep=True)
    model = PooledScheduleDonors(children)
    values = model.distribution(children.iloc[0])
    fit_pooled_sibling_dependence(children, objective="population_moments")
    fit_pooled_sibling_dependence(children, objective="pair_squared_error")
    for original, after in zip(
        values, model.distribution(children.iloc[0]), strict=True
    ):
        np.testing.assert_array_equal(original, after)
    assert_frame_equal(children, before)


def test_unknown_dependence_objective_is_rejected():
    with pytest.raises(ValueError, match="objective"):
        fit_pooled_sibling_dependence(_children(), objective="unknown")


@pytest.mark.parametrize("observed_product,overlap", [(1.0, False), (3.0, True)])
def test_dependence_screen_compatibility_uses_fixed_marginal_identity(
    observed_product, overlap
):
    from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
        coupling_screen_compatibility,
        sibling_schedule_screen,
    )

    # Fixed E[X]E[Y]=1 and SD[X]SD[Y]=4: corr=(E[XY]-1)/4.
    group = {
        "observed": {
            m: {"joint_product": observed_product, "correlation": 0.5}
            for m in ("days", "weekly_hours")
        },
        "independent": {
            m: {"joint_product": 1.0, "correlation": 0.0}
            for m in ("days", "weekly_hours")
        },
        "coupled": {
            m: {"joint_product": 3.0, "correlation": 0.5}
            for m in ("days", "weekly_hours")
        },
    }
    evaluated = {"youngest_pairs": group, "larger_households": {}}
    evaluated["diagnostic_screen"] = sibling_schedule_screen(evaluated)
    report = coupling_screen_compatibility(evaluated)
    result = report["comparisons"][0]
    assert result["predicted_product_of_marginal_standard_deviations"] == 4
    assert result["joint_product_required_by_correlation_screen"] == pytest.approx(
        [2.6, 3.4]
    )
    assert result["screen_intervals_overlap"] == overlap
    assert not report["comparisons"][-1]["identified"]


@pytest.mark.parametrize("age", [np.nan, np.inf, -1, 3.5])
def test_unknown_or_invalid_roster_age_cannot_reduce_household_size(age):
    children = _children()
    children["age"] = children.age.astype(float)
    children.loc[0, "age"] = age
    with pytest.raises(ValueError, match="whole-year ages"):
        with_childcare_household_size(children)


@pytest.mark.parametrize("field", ["region", "child_weight"])
def test_nonfinite_donor_inputs_are_rejected(field):
    children = _children()
    children[field] = children[field].astype(float)
    children.loc[0, field] = np.inf
    with pytest.raises(ValueError, match="finite"):
        PooledScheduleDonors(children)
