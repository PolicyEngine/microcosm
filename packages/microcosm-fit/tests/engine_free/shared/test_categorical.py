"""The categorical classifier: weights, learners, the blend and keyed draws.

A code (an occupation group, an industry) cannot go through the quantile forest,
which interpolates. These tests pin the categorical model's contract: the
package's weight rules, the two learners and their mixture, reproducible fits,
draws that follow each row's own uniform, and loud refusals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import (
    BoostedClassifierSettings,
    CategoricalClassifier,
    SplineLogitSettings,
    inverse_cdf_draw,
)
from microcosm.fit.model import FittedModel


def _planted(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Pay picks the class: low pay mostly 'a', middling 'b', high 'c'."""
    rng = np.random.default_rng(seed)
    pay = rng.lognormal(10.0, 1.0, n)
    group = rng.choice(["north", "south"], n)
    cut = np.quantile(pay, [1 / 3, 2 / 3])
    base = np.digitize(pay, cut)
    noisy = np.where(rng.random(n) < 0.15, rng.integers(0, 3, n), base)
    return pd.DataFrame(
        {
            "pay": pay,
            "age": rng.integers(18, 80, n).astype(float),
            "group": group,
            "label": np.array(["a", "b", "c"])[noisy],
            "weight": rng.uniform(0.5, 2.0, n),
        }
    )


def _model(**overrides) -> CategoricalClassifier:
    settings = {
        "categorical_predictors": ("group",),
        "arcsinh_scales": {"pay": 1000.0},
        "boosted": BoostedClassifierSettings(max_iter=60),
        "seed": 3,
    }
    settings.update(overrides)
    return CategoricalClassifier(**settings)


PREDICTORS = ["pay", "age", "group"]


def test_dataframe_fit_requires_explicit_weights() -> None:
    with pytest.raises(ValueError, match="no typed weights"):
        _model().fit(_planted(), PREDICTORS, ["label"])


def test_unweighted_fit_must_be_asked_for() -> None:
    fitted = _model().fit(_planted(), PREDICTORS, ["label"], weights="none")
    assert fitted.weight_kind == "none"


def test_fitted_model_is_a_fitted_model_and_recovers_the_planted_signal() -> None:
    data = _planted()
    fitted = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    assert isinstance(fitted, FittedModel)
    assert list(fitted.classes_) == ["a", "b", "c"]
    proba = fitted.predict_proba(data)
    assert np.allclose(proba.sum(axis=1), 1.0)
    low, high = (
        data["pay"] < data["pay"].quantile(0.2),
        data["pay"] > data["pay"].quantile(0.8),
    )
    assert proba[low.to_numpy(), 0].mean() > 0.7
    assert proba[high.to_numpy(), 2].mean() > 0.7


@pytest.mark.parametrize("weight, learner", [(1.0, "boosted"), (0.0, "spline_logit")])
def test_a_weight_of_one_or_zero_is_one_learner(weight, learner) -> None:
    data = _planted()
    fitted = _model(boosted_weight=weight).fit(
        data, PREDICTORS, ["label"], weights="weight"
    )
    components = fitted.component_proba(data)
    assert list(components) == [learner]
    assert np.allclose(fitted.predict_proba(data), components[learner])


def test_the_blend_is_the_weighted_average_of_its_learners() -> None:
    data = _planted()
    fitted = _model(boosted_weight=0.25).fit(
        data, PREDICTORS, ["label"], weights="weight"
    )
    components = fitted.component_proba(data)
    expected = 0.25 * components["boosted"] + 0.75 * components["spline_logit"]
    assert np.allclose(fitted.predict_proba(data), expected)
    assert fitted.diagnostics["boosted_iterations"] >= 1
    assert fitted.diagnostics["logit_iterations"] >= 1


def test_the_same_seed_refits_identically() -> None:
    data = _planted()
    first = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    second = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    assert np.array_equal(first.predict_proba(data), second.predict_proba(data))


def test_draws_follow_each_rows_own_uniform() -> None:
    data = _planted()
    fitted = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    u = np.random.default_rng(9).random(len(data))
    drawn = fitted.predict_from_uniforms(data, u)["label"].to_numpy()
    expected = fitted.classes_[inverse_cdf_draw(fitted.predict_proba(data), u)]
    assert np.array_equal(drawn, expected)
    # A row's draw depends on its own uniform only: reorder the rows.
    order = np.random.default_rng(10).permutation(len(data))
    reordered = fitted.predict_from_uniforms(data.iloc[order], u[order])["label"]
    assert np.array_equal(reordered.to_numpy(), drawn[order])


def test_draws_reproduce_the_probabilities_in_aggregate() -> None:
    data = _planted(4000)
    fitted = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    u = np.random.default_rng(11).random(len(data))
    drawn = fitted.predict_from_uniforms(data, u)["label"].to_numpy()
    shares = np.array([(drawn == c).mean() for c in fitted.classes_])
    assert np.allclose(shares, fitted.predict_proba(data).mean(axis=0), atol=0.03)


def test_predict_is_a_seeded_draw_stream() -> None:
    data = _planted()
    first = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    second = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    assert first.predict(data).equals(second.predict(data))


def test_inverse_cdf_draw_places_uniforms_by_cumulative_probability() -> None:
    proba = np.array([[0.2, 0.5, 0.3]] * 4)
    assert inverse_cdf_draw(proba, np.array([0.0, 0.19, 0.2, 0.999999])).tolist() == [
        0,
        0,
        1,
        2,
    ]


@pytest.mark.parametrize(
    "proba, uniforms",
    [
        (np.array([[0.5, 0.6]]), np.array([0.1])),
        (np.array([[1.2, -0.2]]), np.array([0.1])),
        (np.array([[0.5, 0.5]]), np.array([1.0])),
        (np.array([[0.5, 0.5]]), np.array([0.1, 0.2])),
    ],
)
def test_inverse_cdf_draw_refuses_bad_input(proba, uniforms) -> None:
    with pytest.raises(ValueError):
        inverse_cdf_draw(proba, uniforms)


def test_a_logit_that_does_not_converge_is_refused() -> None:
    model = _model(boosted_weight=0.0, logit=SplineLogitSettings(max_iter=1))
    with pytest.raises(ValueError, match="did not converge"):
        model.fit(_planted(), PREDICTORS, ["label"], weights="weight")


def test_unseen_categories_at_prediction_are_refused() -> None:
    data = _planted()
    fitted = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    with pytest.raises(ValueError, match="never saw"):
        fitted.predict_proba(data.assign(group="east"))


def test_non_finite_numeric_predictors_are_refused() -> None:
    data = _planted()
    data.loc[3, "pay"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _model().fit(data, PREDICTORS, ["label"], weights="weight")


def test_a_missing_label_is_refused() -> None:
    data = _planted()
    data.loc[5, "label"] = None
    with pytest.raises(ValueError, match="no category"):
        _model().fit(data, PREDICTORS, ["label"], weights="weight")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"boosted_weight": 1.5}, "boosted_weight"),
        ({"arcsinh_scales": {"pay": 0.0}}, "positive"),
    ],
)
def test_bad_settings_are_refused(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _model(**overrides)


def test_settings_must_name_predictors() -> None:
    with pytest.raises(ValueError, match="not predictors"):
        _model(categorical_predictors=("region",)).fit(
            _planted(), PREDICTORS, ["label"], weights="weight"
        )


def test_exactly_one_target() -> None:
    data = _planted().assign(other="x")
    with pytest.raises(ValueError, match="exactly one target"):
        _model().fit(data, PREDICTORS, ["label", "other"], weights="weight")


def test_a_single_class_draws_that_class() -> None:
    data = _planted().assign(label="only")
    fitted = _model().fit(data, PREDICTORS, ["label"], weights="weight")
    assert fitted.predict_proba(data).shape == (len(data), 1)
    drawn = fitted.predict_from_uniforms(data, np.full(len(data), 0.5))
    assert set(drawn["label"]) == {"only"}
