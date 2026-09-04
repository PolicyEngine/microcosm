"""Caller-owned uniforms make repeated QRF draws stable by identity."""

import copy

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import fit


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
