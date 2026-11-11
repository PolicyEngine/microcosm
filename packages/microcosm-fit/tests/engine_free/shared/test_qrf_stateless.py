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


def test_chain_step_with_caller_uniforms_draws_each_row_on_its_own():
    """Keyed uniforms make a chain row's draw independent of the other rows.

    Adding a recipient, changing another recipient's predictor and reordering
    the rest leave every untouched row's draws where they were, and a step
    with caller uniforms leaves the chain's positional draw stream unconsumed.
    """
    from microcosm.fit.qrf import RegimeGatedQRF

    x = np.tile(np.arange(30, dtype=float), 6)
    donor = pd.DataFrame(
        {
            "x": x,
            "first": x + 1 + np.tile([0.0, 0.5, 1.0], len(x) // 3),
            "inflated": np.tile([0.0, 4.0, 5.0], len(x) // 3),
        }
    )
    qrf = RegimeGatedQRF(n_estimators=4, seed=7)

    def draw(recipient: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        state = qrf.start_chain(donor, ["x"], ["first", "inflated"], weights="none")
        raw = pd.DataFrame(index=recipient.index)
        for salt in (0.137, 0.611):
            step = qrf.fit_draw_next(
                donor,
                recipient,
                raw,
                state=state,
                weights="none",
                quantiles=(ids * salt) % 1.0,
                sign_uniforms=(ids * (salt + 0.29)) % 1.0,
            )
            assert step.state.draw_rng_state_json == state.draw_rng_state_json
            raw[step.target] = step.raw_draw
            state = step.state
        return raw.set_axis(ids)

    ids = np.arange(1.0, 13.0)
    base = draw(pd.DataFrame({"x": np.arange(12, dtype=float)}), ids)

    changed_ids = np.concatenate([[99.0], ids[::-1]])
    changed_x = np.concatenate([[3.0], np.arange(12, dtype=float)[::-1]])
    changed_x[1] = 29.0  # the row with id 12 gets a new predictor
    changed = draw(pd.DataFrame({"x": changed_x}), changed_ids)

    untouched = ids[:-1]
    pd.testing.assert_frame_equal(changed.loc[untouched], base.loc[untouched])
    with pytest.raises(ValueError, match="both quantiles and sign_uniforms"):
        qrf.fit_draw_next(
            donor,
            pd.DataFrame({"x": [1.0]}),
            pd.DataFrame(index=[0]),
            state=qrf.start_chain(donor, ["x"], ["first"], weights="none"),
            weights="none",
            quantiles=np.array([0.5]),
        )


def _placements(model, n, seed):
    rng = np.random.default_rng(seed)
    return {
        "sign_placement": {t: rng.random(n) for t in model.targets},
        "quantile_placement": {t: rng.random(n) for t in model.targets},
    }


def test_conditional_uniforms_round_trip_reproduces_observed_targets(model):
    """Locating the training rows and drawing at those uniforms gives them back.

    Every fixture target is an atom of its conditional at each ``x`` (the
    donor repeats each value), so the located quantile sits on a flat stretch
    and the draw returns the atom exactly; the sign uniform lands in the
    observed class because it is placed inside that class's gate mass.
    """
    x = np.tile(np.arange(30, dtype=float), 6)
    observed = pd.DataFrame(
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
    before = copy.deepcopy(model._rng.bit_generator.state)
    located = model.conditional_uniforms(observed, **_placements(model, len(x), 1))
    assert model._rng.bit_generator.state == before
    assert located.degenerate == ("zero",)
    assert located.ungated == ("positive", "negative")
    for mapping in (located.quantiles, located.sign_uniforms):
        assert set(mapping) == set(model.targets)
        for values in mapping.values():
            assert values.shape == (len(x),)
            assert ((values >= 0) & (values < 1)).all()
    drawn = model.predict_from_uniforms(
        observed[["x"]],
        quantiles=located.quantiles,
        sign_uniforms=located.sign_uniforms,
    )
    pd.testing.assert_frame_equal(
        drawn, observed[model.targets].astype(float), check_exact=False, rtol=1e-9
    )


def test_conditional_uniforms_are_row_order_invariant(model):
    x = np.tile(np.arange(30, dtype=float), 2)
    observed = pd.DataFrame(
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
    placements = _placements(model, len(x), 2)
    expected = model.conditional_uniforms(observed, **placements)
    order = np.random.default_rng(5).permutation(len(x))
    reordered = model.conditional_uniforms(
        observed.iloc[order],
        **{k: {t: v[order] for t, v in d.items()} for k, d in placements.items()},
    )
    for target in model.targets:
        np.testing.assert_array_equal(
            expected.quantiles[target][order], reordered.quantiles[target]
        )
        np.testing.assert_array_equal(
            expected.sign_uniforms[target][order], reordered.sign_uniforms[target]
        )


def test_conditional_uniforms_are_uniform_off_sample_and_transport_standing():
    """Held-out rows locate uniformly, and the uniforms carry a row's standing.

    Population A is ``y = x + 5 + noise``; population B is ``y = 2x + 10 + noise``.
    Locating A's held-out rows under A's forest gives magnitude quantiles close
    to U[0, 1) (the probability integral transform), and drawing B's forest at
    those uniforms ranks the rows within each ``x`` as A ranked them.
    """
    rng = np.random.default_rng(11)
    x = np.repeat(np.arange(10, dtype=float), 300)
    donor_a = pd.DataFrame({"x": x, "y": x + 5 + rng.normal(size=len(x))})
    donor_b = pd.DataFrame({"x": x, "y": 2 * x + 10 + rng.normal(size=len(x))})
    model_a = fit(donor_a, ["x"], ["y"], weights="none", n_estimators=20, seed=3)
    model_b = fit(donor_b, ["x"], ["y"], weights="none", n_estimators=20, seed=4)
    held = pd.DataFrame({"x": x, "y": x + 5 + rng.normal(size=len(x))})
    placements = {
        "sign_placement": {"y": rng.random(len(x))},
        "quantile_placement": {"y": rng.random(len(x))},
    }
    located = model_a.conditional_uniforms(held, **placements)
    quantiles = np.sort(located.quantiles["y"])
    empirical = np.arange(1, len(quantiles) + 1) / len(quantiles)
    kolmogorov = np.abs(empirical - quantiles).max()
    assert kolmogorov < 0.08, kolmogorov
    transported = model_b.predict_from_uniforms(
        held[["x"]],
        quantiles=located.quantiles,
        sign_uniforms=located.sign_uniforms,
    )["y"].to_numpy()
    for value in np.unique(x):
        rows = x == value
        rank_a = pd.Series(held.y.to_numpy()[rows]).rank().to_numpy()
        rank_b = pd.Series(transported[rows]).rank().to_numpy()
        assert np.corrcoef(rank_a, rank_b)[0, 1] > 0.95
    # B's level is imported: the transported draws sit on B's scale, not A's.
    assert transported.mean() > 2 * x.mean() + 5


def test_quantile_positions_match_chunked_and_serial_passes(model, monkeypatch):
    from microcosm.fit import qrf as qrf_module

    x = np.tile(np.arange(30, dtype=float), 2)
    frame = pd.DataFrame({"x": x})
    forest = model._target_models["positive"].positive
    placement = np.random.default_rng(6).random(len(x))
    serial = forest.quantile_positions(frame, x + 1, placement)
    monkeypatch.setattr(qrf_module, "_PREDICT_CHUNK_ROWS", 7)
    chunked = forest.quantile_positions(frame, x + 1, placement)
    np.testing.assert_array_equal(serial, chunked)
    drawn = forest.draw(frame, serial)
    np.testing.assert_allclose(drawn, x + 1)


def test_locate_rows_inverts_interp_rows():
    from microcosm.fit.qrf import _interp_rows, _locate_rows

    grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    predictions = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # strictly increasing
            [1.0, 2.0, 2.0, 2.0, 5.0],  # a flat stretch at 2.0
            [1.0, 2.0, 3.0, 4.0, 5.0],  # below the first knot
            [1.0, 2.0, 3.0, 4.0, 5.0],  # above the last knot
            [2.0, 2.0, 2.0, 2.0, 2.0],  # one atom
        ]
    )
    values = np.array([2.5, 2.0, 0.5, 7.0, 2.0])
    placement = np.array([0.5, 0.25, 0.5, 0.5, 0.75])
    located = _locate_rows(values, placement, grid, predictions)
    np.testing.assert_allclose(located[0], 0.4)
    np.testing.assert_allclose(located[1], 0.3 + 0.25 * (0.7 - 0.3))
    np.testing.assert_allclose(located[2], 0.05)
    np.testing.assert_allclose(located[3], 0.95)
    np.testing.assert_allclose(located[4], 0.1 + 0.75 * 0.8)
    assert ((located >= 0) & (located < 1)).all()
    read_back = _interp_rows(located, grid, predictions)
    np.testing.assert_allclose(read_back[[0, 1, 4]], values[[0, 1, 4]])
    np.testing.assert_allclose(read_back[2], 1.0)  # clamped to the curve's ends
    np.testing.assert_allclose(read_back[3], 5.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        _locate_rows(
            values[:1], placement[:1], grid, np.array([[3.0, 2.0, 2.0, 2.0, 2.0]])
        )


def test_conditional_uniforms_refuse_inconsistent_inputs(model):
    x = np.arange(6, dtype=float)
    observed = pd.DataFrame(
        {
            "x": x,
            "positive": x + 1,
            "negative": -x - 1,
            "zero": np.zeros(6),
            "mixed": np.tile([-2.0, 0.0, 3.0], 2),
            "inflated": np.tile([0.0, 4.0], 3),
            "negative_inflated": np.tile([0.0, -4.0], 3),
            "two_sign": np.tile([-3.0, 4.0], 3),
        }
    )
    placements = _placements(model, 6, 8)
    with pytest.raises(ValueError, match="missing observed target"):
        model.conditional_uniforms(observed.drop(columns=["mixed"]), **placements)
    degenerate = observed.copy()
    degenerate.loc[0, "zero"] = 1.0
    with pytest.raises(ValueError, match="degenerate zero"):
        model.conditional_uniforms(degenerate, **placements)
    wrong_sign = observed.copy()
    wrong_sign.loc[0, "positive"] = -1.0
    with pytest.raises(ValueError, match="another sign"):
        model.conditional_uniforms(wrong_sign, **placements)
    unseen_class = observed.copy()
    unseen_class.loc[0, "inflated"] = -4.0
    with pytest.raises(ValueError, match="absent from the fitted gate"):
        model.conditional_uniforms(unseen_class, **placements)
    short = {k: {t: v[:5] for t, v in d.items()} for k, d in placements.items()}
    with pytest.raises(ValueError, match="must have shape"):
        model.conditional_uniforms(observed, **short)
    released = copy.deepcopy(model)
    released.predict(observed[["x"]], release_models=True)
    with pytest.raises(RuntimeError, match="released"):
        released.conditional_uniforms(observed, **placements)


def test_conditional_uniforms_carry_the_zero_class_between_gated_models():
    """A zero-inflated row's sign uniform lands by its class under another gate.

    Under A half the rows are zero; under B a fifth are. A row that is positive
    under A sits above A's zero mass, so B, whose zero mass is smaller, draws it
    positive every time; a row that is zero under A is spread over A's zero
    mass, so B keeps about two fifths of them at zero and the rest draw
    positive. This is the mechanism that carries "has pay" across populations.
    """
    rng = np.random.default_rng(12)
    x = np.repeat(np.arange(10, dtype=float), 300)
    positive_a = rng.random(len(x)) < 0.5
    donor_a = pd.DataFrame(
        {"x": x, "y": np.where(positive_a, x + 5 + rng.normal(size=len(x)), 0.0)}
    )
    positive_b = rng.random(len(x)) < 0.8
    donor_b = pd.DataFrame(
        {"x": x, "y": np.where(positive_b, 2 * x + 10 + rng.normal(size=len(x)), 0.0)}
    )
    model_a = fit(donor_a, ["x"], ["y"], weights="none", n_estimators=20, seed=5)
    model_b = fit(donor_b, ["x"], ["y"], weights="none", n_estimators=20, seed=6)
    assert model_a.regimes() == {"y": Regime.ZERO_INFLATED_POSITIVE}
    located = model_a.conditional_uniforms(
        donor_a,
        sign_placement={"y": rng.random(len(x))},
        quantile_placement={"y": rng.random(len(x))},
    )
    transported = model_b.predict_from_uniforms(
        donor_a[["x"]],
        quantiles=located.quantiles,
        sign_uniforms=located.sign_uniforms,
    )["y"].to_numpy()
    assert (transported[positive_a] > 0).all()
    kept_zero = (transported[~positive_a] == 0).mean()
    assert 0.3 < kept_zero < 0.5, kept_zero


def test_sign_classes_follow_the_regime(model):
    """The classes a target's gate orders, or the single sign of an ungated
    regime, so a caller drawing one model at uniforms located on another can
    check the two read a sign uniform the same way."""
    expected = {
        "positive": (1,),
        "negative": (-1,),
        "zero": (0,),
        "mixed": (-1, 0, 1),
        "inflated": (0, 1),
        "negative_inflated": (-1, 0),
        "two_sign": (-1, 1),
    }
    for target, classes in expected.items():
        assert model.sign_classes(target) == classes, target
        assert list(classes) == sorted(classes)
    assert model.regimes()["positive"] == Regime.POSITIVE_ONLY
    with pytest.raises(ValueError, match="unknown target"):
        model.sign_classes("missing")
