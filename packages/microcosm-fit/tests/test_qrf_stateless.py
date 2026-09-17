"""Caller-owned uniforms make repeated QRF draws stable by identity."""

import copy

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import fit
from microcosm.fit.qrf import Regime


@pytest.fixture(scope="module")
def model():
    x = np.tile(np.arange(30, dtype=float), 6)
    donor = pd.DataFrame(
        {
            "x": x,
            "positive": x + 1,
            "negative": -x - 1,
            "zero": np.zeros(len(x)),
            "mixed": np.tile([-2.0, 0.0, 3.0], len(x) // 3),
            "inflated": np.tile([0.0, 4.0], len(x) // 2),
            "negative_inflated": np.tile([0.0, -4.0], len(x) // 2),
            "two_sign": np.tile([-3.0, 4.0], len(x) // 2),
        }
    )
    return fit(
        donor,
        ["x"],
        list(donor.columns[1:]),
        weights="none",
        n_estimators=4,
        seed=7,
    )


def uniforms(model, n):
    return {
        "quantiles": {t: np.linspace(0, 0.99, n) for t in model.targets},
        "sign_uniforms": {t: np.linspace(0.99, 0, n) for t in model.targets},
    }


def test_stateless_draws_preserve_rng_and_forests(model):
    recipient = pd.DataFrame({"x": np.arange(20, dtype=float)})
    before = copy.deepcopy(model._rng.bit_generator.state)
    draws = uniforms(model, len(recipient))
    first = model.predict_from_uniforms(recipient, **draws)
    pd.testing.assert_frame_equal(
        first, model.predict_from_uniforms(recipient, **draws)
    )
    assert model._rng.bit_generator.state == before
    assert (first.positive > 0).all()
    assert (first.negative < 0).all()
    assert (first.zero == 0).all()
    assert set(first.mixed) == {-2.0, 0.0, 3.0}
    assert set(first.inflated) == {0.0, 4.0}
    assert set(first.negative_inflated) == {0.0, -4.0}
    assert set(first.two_sign) == {-3.0, 4.0}


def test_permutation_and_chunking_preserve_chained_draws(model):
    recipient = pd.DataFrame({"x": np.arange(20, dtype=float)})
    draws = uniforms(model, len(recipient))
    expected = model.predict_from_uniforms(recipient, **draws)
    order = np.random.default_rng(4).permutation(len(recipient))
    reordered = model.predict_from_uniforms(
        recipient.iloc[order],
        **{k: {t: v[order] for t, v in d.items()} for k, d in draws.items()},
    )
    pd.testing.assert_frame_equal(expected, reordered.sort_index())
    chunks = []
    for rows in (slice(0, 7), slice(7, 20)):
        chunks.append(
            model.predict_from_uniforms(
                recipient.iloc[rows],
                **{k: {t: v[rows] for t, v in d.items()} for k, d in draws.items()},
            )
        )
    pd.testing.assert_frame_equal(expected, pd.concat(chunks))


@pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.1, 1.0], [np.nan, 0.1], [0.1]])
@pytest.mark.parametrize("field", ["quantiles", "sign_uniforms"])
def test_invalid_uniforms_rejected_before_drawing(model, bad, field):
    recipient = pd.DataFrame({"x": [1.0, 2.0]})
    draws = uniforms(model, 2)
    draws[field][model.targets[-1]] = np.array(bad)
    with pytest.raises(ValueError, match="uniform|shape"):
        model.predict_from_uniforms(recipient, **draws)


def test_uniform_target_names_must_match(model):
    draws = uniforms(model, 2)
    del draws["quantiles"][model.targets[-1]]
    with pytest.raises(ValueError, match="targets"):
        model.predict_from_uniforms(pd.DataFrame({"x": [1.0, 2.0]}), **draws)


@pytest.mark.parametrize("field", ["quantiles", "sign_uniforms"])
@pytest.mark.parametrize("imaginary", [0.0, 0.5, np.nan, np.inf])
def test_complex_uniforms_are_refused_without_lossy_conversion(model, field, imaginary):
    draws = uniforms(model, 2)
    draws[field][model.targets[-1]] = np.array(
        [complex(0.25, imaginary), complex(0.75, imaginary)]
    )
    with pytest.raises(ValueError, match="real numeric"):
        model.predict_from_uniforms(pd.DataFrame({"x": [1.0, 2.0]}), **draws)


def test_empty_recipient_batch(model):
    actual = model.predict_from_uniforms(
        pd.DataFrame({"x": pd.Series(dtype=float)}), **uniforms(model, 0)
    )
    assert list(actual.columns) == model.targets
    assert actual.empty
    assert all(dtype == np.dtype("float64") for dtype in actual.dtypes)


def test_zero_uniform_skips_a_zero_probability_sign(model, monkeypatch):
    # Exercise the inverse-CDF boundary that ordinary RNG draws almost never hit.
    gate = model._target_models["mixed"].gate
    monkeypatch.setattr(
        gate, "predict_proba", lambda x: np.tile([0.0, 0.0, 1.0], (len(x), 1))
    )
    draws = uniforms(model, 2)
    draws["sign_uniforms"]["mixed"] = np.zeros(2)
    actual = model.predict_from_uniforms(pd.DataFrame({"x": [1.0, 2.0]}), **draws)
    assert (actual.mixed == 3.0).all()


def test_a_cdf_short_of_one_cannot_fall_back_to_the_first_sign(model, monkeypatch):
    """The other half of the deliberate inverse-CDF change: the final bin closes.

    ``predict_proba`` rows are floating-point and need not sum to exactly 1.0.
    Without ``cumulative[:, -1] = 1.0`` a uniform above that sum makes every
    comparison false, and ``argmax`` on an all-false row returns 0 — silently
    selecting the *first* class, here the negative one, for a draw that should
    land in the last bin. Closing the bin is what the charter's amendment 20
    claims, so it is pinned rather than left to the strict-comparison test.
    """
    gate = model._target_models["mixed"].gate
    assert list(gate.classes_) == [-1, 0, 1]
    short = np.array([0.5, 0.5 - 2e-16, 0.0])
    assert short.cumsum()[-1] < 1.0, "the row must fall short for this to bite"
    monkeypatch.setattr(gate, "predict_proba", lambda x: np.tile(short, (len(x), 1)))
    draws = uniforms(model, 2)
    draws["sign_uniforms"]["mixed"] = np.full(2, 1.0 - 1e-16)
    actual = model.predict_from_uniforms(pd.DataFrame({"x": [1.0, 2.0]}), **draws)
    assert (actual.mixed == 3.0).all(), (
        "a uniform past a short CDF selected the first class, not the last"
    )


def test_stateless_replays_legacy_uniforms_across_all_regimes(model):
    """Ordinary uniforms preserve legacy draws; exact CDF ties differ deliberately.

    The zero-uniform test above pins the intentional strict-CDF boundary.
    This independent RNG replay detects swapped or inverted quantile streams.
    """
    legacy = copy.deepcopy(model)
    recipient = pd.DataFrame({"x": np.arange(20, dtype=float)})
    assert set(legacy.regimes().values()) == {
        Regime.POSITIVE_ONLY,
        Regime.NEGATIVE_ONLY,
        Regime.DEGENERATE_ZERO,
        Regime.THREE_SIGN,
        Regime.ZERO_INFLATED_POSITIVE,
        Regime.ZERO_INFLATED_NEGATIVE,
        Regime.SIGN_ONLY,
    }
    for _ in range(4):
        replay = np.random.default_rng()
        replay.bit_generator.state = copy.deepcopy(legacy._rng.bit_generator.state)
        quantiles, signs = {}, {}
        for target in legacy.targets:
            target_model = legacy._target_models[target]
            quantiles[target] = (
                replay.random(len(recipient))
                if target_model.regime != Regime.DEGENERATE_ZERO
                else np.zeros(len(recipient))
            )
            signs[target] = (
                replay.random(len(recipient))
                if target_model.gate is not None
                else np.zeros(len(recipient))
            )
        expected = legacy.predict(recipient)
        actual = legacy.predict_from_uniforms(
            recipient, quantiles=quantiles, sign_uniforms=signs
        )
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def test_positive_draw_matches_the_last_bin_of_the_sign_gate(model):
    """A positive-regime draw is the amount forest at the row's quantile.

    ``predict_from_uniforms`` with a sign uniform just below one lands in the
    last CDF bin, which is the positive class when one exists; that is a
    property of the sorted class order. The explicit method declares the same
    draw as an API, so the two agree bit for bit on every positive-forest
    regime.
    """
    recipient = pd.DataFrame({"x": np.arange(20, dtype=float)})
    positive_targets = ["positive", "mixed", "inflated", "two_sign"]
    n = len(recipient)
    quantiles = {t: np.linspace(0, 0.99, n) for t in positive_targets}
    full = fit(
        pd.DataFrame(
            {
                "x": np.tile(np.arange(30, dtype=float), 6),
                "positive": np.tile(np.arange(30, dtype=float), 6) + 1,
                "mixed": np.tile([-2.0, 0.0, 3.0], 60),
                "inflated": np.tile([0.0, 4.0], 90),
                "two_sign": np.tile([-3.0, 4.0], 90),
            }
        ),
        ["x"],
        positive_targets,
        weights="none",
        n_estimators=4,
        seed=7,
    )
    explicit = full.predict_positive_from_uniforms(recipient, quantiles=quantiles)
    via_gate = full.predict_from_uniforms(
        recipient,
        quantiles=quantiles,
        sign_uniforms={t: np.full(n, 1.0 - 1e-16) for t in positive_targets},
    )
    pd.testing.assert_frame_equal(explicit, via_gate)
    assert (explicit["mixed"] == 3.0).all()
    assert (explicit["inflated"] == 4.0).all()
    assert (explicit["two_sign"] == 4.0).all()
    assert (explicit["positive"] > 0).all()


@pytest.mark.parametrize("target", ["negative", "zero", "negative_inflated"])
def test_positive_draw_refuses_regimes_without_a_positive_forest(target):
    x = np.tile(np.arange(30, dtype=float), 6)
    donor = pd.DataFrame(
        {
            "x": x,
            "negative": -x - 1,
            "zero": np.zeros(len(x)),
            "negative_inflated": np.tile([0.0, -4.0], len(x) // 2),
        }
    )
    fitted = fit(donor, ["x"], [target], weights="none", n_estimators=4, seed=7)
    with pytest.raises(ValueError, match="no positive-magnitude forest"):
        fitted.predict_positive_from_uniforms(
            pd.DataFrame({"x": [1.0, 2.0]}), quantiles={target: np.array([0.1, 0.5])}
        )


def test_positive_draw_validates_uniforms_and_leaves_rng_untouched(model):
    recipient = pd.DataFrame({"x": np.arange(5, dtype=float)})
    before = copy.deepcopy(model._rng.bit_generator.state)
    with pytest.raises(ValueError, match="exactly the fitted targets"):
        model.predict_positive_from_uniforms(
            recipient, quantiles={"positive": np.zeros(5)}
        )
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        model.predict_positive_from_uniforms(
            recipient, quantiles={t: np.full(5, 1.0) for t in model.targets}
        )
    assert model._rng.bit_generator.state == before


def test_chain_step_exposes_a_fitted_one_target_view():
    from microcosm.fit.qrf import RegimeGatedQRF

    x = np.tile(np.arange(30, dtype=float), 6)
    donor = pd.DataFrame(
        {"x": x, "first": x + 1, "inflated": np.tile([0.0, 4.0], len(x) // 2)}
    )
    recipient = pd.DataFrame({"x": np.arange(10, dtype=float)})
    qrf = RegimeGatedQRF(n_estimators=4, seed=7)
    state = qrf.start_chain(donor, ["x"], ["first", "inflated"], weights="none")
    raw = pd.DataFrame(index=recipient.index)
    step_one = qrf.fit_draw_next(donor, recipient, raw, state=state, weights="none")
    raw["first"] = step_one.raw_draw
    step_two = qrf.fit_draw_next(
        donor, recipient, raw, state=step_one.state, weights="none"
    )

    view = step_two.fitted
    assert view is not None
    assert view.targets == ["inflated"]
    assert view.predictors == ["x", "first"]
    assert view.regimes() == {"inflated": Regime.ZERO_INFLATED_POSITIVE}
    augmented = recipient.assign(first=raw["first"].to_numpy())
    positive = view.predict_positive_from_uniforms(
        augmented, quantiles={"inflated": np.linspace(0, 0.99, len(recipient))}
    )
    assert (positive["inflated"] == 4.0).all()
    # The view's own predict is reproducible and independent of the chain stream.
    first = view.predict(augmented)
    step_two_again = qrf.fit_draw_next(
        donor, recipient, raw, state=step_one.state, weights="none"
    )
    pd.testing.assert_frame_equal(first, step_two_again.fitted.predict(augmented))
    np.testing.assert_array_equal(step_two.raw_draw, step_two_again.raw_draw)
