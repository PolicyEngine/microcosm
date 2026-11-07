"""Input contracts the microimpute bug-parity audit reproduced on microcosm-fit.

Each example test pins one input that main at ``dc00d158`` mishandled, quietly
or with an unrelated crash: an aliased seed, a NaN tolerance, a duplicated
column label, an invalid probability, a sign with no weight, an empty donor
table, an empty recipient batch, a missing distance value or an unseen
category. Each fails on that commit. The properties pin the contracts for every
input: valid input fits and draws exactly as before, and invalid input is
refused by name.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from microcosm.fit import (
    CategoricalClassifier,
    GateCalibration,
    GateOutcome,
    QRFChainState,
    RegimeGatedQRF,
    donor_distance_features,
    fit,
)
from microcosm.fit.held_out import LOG_LOSS_FLOOR, categorical_scores
from microcosm.fit.qrf import _weighted_bootstrap, detect_regime

#: Small, reproducible property runs.
PROPERTY = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

#: Property runs whose every example fits forests.
FITTING_PROPERTY = settings(
    max_examples=6,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

EMPTY_BATCH = pd.DataFrame({"x": pd.Series([], dtype=float)})

#: Both sign-gate calibrations, so the empty-batch tests reach both gate types:
#: one boosted classifier, and the cross-fitted gate that averages fold gates.
CALIBRATIONS = (GateCalibration.IN_SAMPLE, GateCalibration.CROSS_FITTED)


def _gated_donors() -> pd.DataFrame:
    """Donors for every regime, three of them gated."""
    x = np.tile(np.arange(12, dtype=float), 4)
    return pd.DataFrame(
        {
            "x": x,
            "mixed": np.tile([-2.0, 0.0, 3.0], len(x) // 3),
            "inflated": np.tile([0.0, 4.0], len(x) // 2),
            "two_sign": np.tile([-3.0, 4.0], len(x) // 2),
            "positive": x + 1.0,
            "zero": np.zeros(len(x)),
        }
    )


def _fit_gated(seed: int = 7, calibration: str = GateCalibration.IN_SAMPLE):
    donors = _gated_donors()
    model = fit(
        donors,
        ["x"],
        list(donors.columns[1:]),
        weights="none",
        n_estimators=2,
        seed=seed,
        gate_calibration=calibration,
    )
    if calibration == GateCalibration.CROSS_FITTED:
        # Every class has a row in every fold, so no gate falls back in-sample.
        gated = [outcome for outcome in model.gate_outcomes().values() if outcome]
        assert gated and set(gated) == {GateOutcome.CROSS_FITTED}
    return model


# ----------------------------------------------------------------------------
# Seeds and forest sizes are integers (D010)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [True, False, 1.9, 1.0, np.float64(2.0), "3"])
def test_qrf_refuses_a_seed_that_is_not_an_integer(seed) -> None:
    """``int()`` aliased these: ``True`` and ``1.9`` both seeded as 1."""
    with pytest.raises(TypeError, match="seed must be an integer"):
        RegimeGatedQRF(seed=seed)


def test_qrf_refuses_a_negative_seed_when_constructed() -> None:
    """A negative seed used to fail only at fit time, inside ``SeedSequence``."""
    with pytest.raises(ValueError, match="seed must be an integer >= 0"):
        RegimeGatedQRF(seed=-1)


@pytest.mark.parametrize("n_estimators", [True, 1.9, 4.0])
def test_qrf_refuses_a_forest_size_that_is_not_an_integer(n_estimators) -> None:
    """``int(1.9)`` grew one tree."""
    with pytest.raises(TypeError, match="n_estimators must be an integer"):
        RegimeGatedQRF(n_estimators=n_estimators)


def test_qrf_refuses_an_empty_forest() -> None:
    with pytest.raises(ValueError, match="n_estimators must be an integer >= 1"):
        RegimeGatedQRF(n_estimators=0)


@pytest.mark.parametrize("seed", [True, 1.9, -1])
def test_categorical_classifier_refuses_the_same_seeds(seed) -> None:
    """``SeedSequence(True)`` is ``SeedSequence(1)``: two settings, one stream."""
    with pytest.raises((TypeError, ValueError), match="seed must be an integer"):
        CategoricalClassifier(seed=seed)


def test_integer_seeds_keep_their_value() -> None:
    assert RegimeGatedQRF(seed=np.int64(5)).seed == 5
    assert type(RegimeGatedQRF(seed=np.int64(5)).seed) is int
    assert CategoricalClassifier(seed=np.uint32(9)).seed == 9


_ANY_VALUE = st.one_of(
    st.integers(-(2**70), 2**70),
    st.integers(-(2**62), 2**62).map(np.int64),
    st.integers(0, 2**64 - 1).map(np.uint64),
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(allow_nan=False, allow_infinity=False).map(np.float64),
    st.booleans(),
    st.booleans().map(np.bool_),
    st.text(max_size=3),
    st.none(),
)


@PROPERTY
@given(value=_ANY_VALUE, minimum=st.integers(-3, 3))
def test_validated_integer_accepts_exactly_the_integers_at_or_above_the_minimum(
    value, minimum
) -> None:
    """The helper keeps each integer at or above the minimum and refuses the rest."""
    from microcosm.fit.model import validated_integer

    is_integer = isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )
    if is_integer and int(value) >= minimum:
        result = validated_integer(value, name="seed", minimum=minimum)
        assert result == int(value)
        assert type(result) is int
    else:
        with pytest.raises(ValueError if is_integer else TypeError):
            validated_integer(value, name="seed", minimum=minimum)


@FITTING_PROPERTY
@given(seed=st.integers(0, 2**64 - 1))
def test_a_seed_draws_the_same_whatever_its_integer_type(seed) -> None:
    """Converting a NumPy integer seed to ``int`` moves no draw."""
    donors = _gated_donors()[["x", "mixed"]]
    as_int = fit(
        donors, ["x"], ["mixed"], weights="none", n_estimators=2, seed=seed
    ).predict(donors)
    as_numpy = fit(
        donors, ["x"], ["mixed"], weights="none", n_estimators=2, seed=np.uint64(seed)
    ).predict(donors)
    pd.testing.assert_frame_equal(as_int, as_numpy)


# ----------------------------------------------------------------------------
# The zero tolerance is finite and non-negative (D034)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("tolerance", [math.nan, math.inf, -math.inf, -1e-12])
def test_qrf_refuses_a_tolerance_that_is_not_finite_and_non_negative(
    tolerance,
) -> None:
    """A NaN tolerance fitted positive amounts as degenerate zero and drew 0."""
    with pytest.raises(ValueError, match="zero_atol must be finite and >= 0"):
        RegimeGatedQRF(zero_atol=tolerance)


@pytest.mark.parametrize("tolerance", [True, "0.1", None])
def test_qrf_refuses_a_tolerance_that_is_not_a_number(tolerance) -> None:
    with pytest.raises(TypeError, match="zero_atol must be a real number"):
        RegimeGatedQRF(zero_atol=tolerance)


def test_detect_regime_refuses_a_nan_tolerance() -> None:
    """Every sign comparison with NaN is false, so positive data read as zero."""
    with pytest.raises(ValueError, match="zero_atol"):
        detect_regime(np.array([1.0, 2.0]), zero_atol=math.nan)


def _chain_payload(tolerance: float = 0.0) -> dict:
    donors = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    state = RegimeGatedQRF(n_estimators=1, zero_atol=tolerance).start_chain(
        donors, ["x"], ["y"], weights="none"
    )
    return json.loads(json.dumps(state.to_dict()))


@pytest.mark.parametrize("tolerance", [math.nan, math.inf, -1.0, True, "0"])
def test_a_checkpoint_refuses_an_invalid_tolerance(tolerance) -> None:
    """JSON reads ``NaN`` back as a float, so a checkpoint could carry one."""
    payload = _chain_payload()
    payload["model_config"]["zero_atol"] = tolerance
    with pytest.raises(ValueError, match="zero_atol"):
        QRFChainState.from_dict(payload)


@PROPERTY
@given(tolerance=st.floats(allow_nan=True, allow_infinity=True))
def test_the_model_accepts_a_tolerance_exactly_when_finite_and_non_negative(
    tolerance,
) -> None:
    if math.isfinite(tolerance) and tolerance >= 0:
        assert RegimeGatedQRF(zero_atol=tolerance).zero_atol == tolerance
    else:
        with pytest.raises(ValueError):
            RegimeGatedQRF(zero_atol=tolerance)


@FITTING_PROPERTY
@given(tolerance=st.floats(0.0, 10.0))
def test_a_checkpoint_keeps_any_valid_tolerance(tolerance) -> None:
    """Every valid tolerance survives the JSON round trip exactly."""
    restored = QRFChainState.from_dict(_chain_payload(tolerance))
    assert restored.zero_atol == tolerance


# ----------------------------------------------------------------------------
# A column label names one column (D031)
# ----------------------------------------------------------------------------


def _duplicated(table: pd.DataFrame, label: str) -> pd.DataFrame:
    return pd.concat([table, table[[label]]], axis=1)


def _donor_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "y": np.tile([0.0, 2.0], 4),
            "w": np.full(8, 2.0),
            "g": np.repeat([0, 1, 2, 3], 2),
        }
    )


@pytest.mark.parametrize("label", ["x", "y"])
def test_a_fit_refuses_a_predictor_or_target_label_naming_two_columns(label) -> None:
    """``table['y']`` read both columns: main fitted a degenerate zero target."""
    table = _duplicated(_donor_table(), label)
    with pytest.raises(ValueError, match="more than one column"):
        RegimeGatedQRF(n_estimators=1).fit(table, ["x"], ["y"], weights="none")


def test_a_fit_refuses_a_weight_label_naming_two_columns() -> None:
    table = _duplicated(_donor_table(), "w")
    with pytest.raises(ValueError, match="more than one column"):
        RegimeGatedQRF(n_estimators=1).fit(table, ["x"], ["y"], weights="w")


def test_a_fit_refuses_a_groups_label_naming_two_columns() -> None:
    table = _duplicated(_donor_table(), "g")
    with pytest.raises(ValueError, match="more than one column"):
        RegimeGatedQRF(n_estimators=1).fit(
            table, ["x"], ["y"], weights="none", groups="g"
        )


def test_the_categorical_classifier_refuses_a_duplicated_target_label() -> None:
    table = _duplicated(_donor_table(), "y")
    with pytest.raises(ValueError, match="more than one column"):
        CategoricalClassifier().fit(table, ["x"], ["y"], weights="none")


@FITTING_PROPERTY
@given(seed=st.integers(0, 1000))
def test_an_unrelated_duplicated_label_moves_no_draw(seed) -> None:
    """Only labels the fit reads must be unique; the rest change nothing."""
    donors = _gated_donors()
    noisy = _duplicated(donors.assign(junk=1.0), "junk")
    targets = list(donors.columns[1:])
    clean = fit(donors, ["x"], targets, weights="none", n_estimators=2, seed=seed)
    dirty = fit(noisy, ["x"], targets, weights="none", n_estimators=2, seed=seed)
    pd.testing.assert_frame_equal(clean.predict(donors), dirty.predict(noisy))


# ----------------------------------------------------------------------------
# The categorical scorer scores probabilities with weights (D022)
# ----------------------------------------------------------------------------

_PROBA = np.array([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])
_LABELS = np.array(["a", "b", "b"])
_CLASSES = np.array(["a", "b"])


def test_scores_refuse_values_that_are_not_probabilities() -> None:
    """``[[2, 0], [0, 2]]`` scored a log loss of -0.69."""
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        categorical_scores(
            np.array([[2.0, 0.0], [0.0, 2.0]]),
            np.array(["a", "b"]),
            _CLASSES,
            np.ones(2),
        )


@pytest.mark.parametrize(
    "proba",
    [
        [[1.2, -0.2], [0.2, 0.8], [0.6, 0.4]],
        [[np.nan, 0.2], [0.2, 0.8], [0.6, 0.4]],
        [[np.inf, 0.0], [0.2, 0.8], [0.6, 0.4]],
    ],
)
def test_scores_refuse_negative_or_non_finite_probabilities(proba) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        categorical_scores(np.array(proba), _LABELS, _CLASSES, np.ones(3))


def test_scores_refuse_rows_that_do_not_sum_to_one() -> None:
    """A row of zeros (a learner absent from a fold) floored the loss at 1e-4."""
    proba = np.array([[0.3, 0.3], [0.0, 0.0], [0.6, 0.4]])
    with pytest.raises(ValueError, match="do not sum to 1"):
        categorical_scores(proba, _LABELS, _CLASSES, np.ones(3))


@pytest.mark.parametrize(
    "weights",
    [[1.0, np.nan, 1.0], [1.0, -0.5, 1.0], [0.0, 0.0, 0.0], [1.0, np.inf, 1.0]],
)
def test_scores_refuse_weights_that_cannot_weight(weights) -> None:
    """NaN or zero weights scored NaN; a negative weight scored a share gap."""
    with pytest.raises(ValueError, match="weights must be finite"):
        categorical_scores(_PROBA, _LABELS, _CLASSES, np.array(weights))


def test_scores_refuse_weights_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="weights need one value per label"):
        categorical_scores(_PROBA, _LABELS, _CLASSES, np.ones(2))


def test_scores_map_labels_by_the_supplied_class_order() -> None:
    """A sorted search over unsorted classes refused labels that were there."""
    in_order = categorical_scores(_PROBA, _LABELS, _CLASSES, np.ones(3))
    reversed_order = categorical_scores(
        _PROBA[:, ::-1], _LABELS, _CLASSES[::-1], np.ones(3)
    )
    assert reversed_order["log_loss"] == pytest.approx(in_order["log_loss"])
    assert reversed_order["share_gap"] == pytest.approx(in_order["share_gap"])


def test_scores_refuse_repeated_classes() -> None:
    with pytest.raises(ValueError, match="classes must be distinct"):
        categorical_scores(
            np.full((3, 2), 0.5), np.array(["a"] * 3), np.array(["a", "a"]), np.ones(3)
        )


def test_scores_refuse_a_probability_matrix_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="one column per class"):
        categorical_scores(np.full((3, 3), 1 / 3), _LABELS, _CLASSES, np.ones(3))


def test_scores_refuse_a_stratum_that_does_not_label_every_row() -> None:
    with pytest.raises(ValueError, match="needs one label per row"):
        categorical_scores(
            _PROBA, _LABELS, _CLASSES, np.ones(3), strata={"half": ["x", "y"]}
        )


def test_scores_still_name_an_unknown_label() -> None:
    with pytest.raises(ValueError, match=r"one of classes; \['c'\]"):
        categorical_scores(_PROBA, np.array(["a", "b", "c"]), _CLASSES, np.ones(3))


@st.composite
def _scorer_inputs(draw):
    """Valid scorer input: rows of probabilities, labels, weights."""
    k = draw(st.integers(1, 4))
    n = draw(st.integers(1, 25))
    raw = np.array(
        draw(
            st.lists(
                st.lists(st.floats(0.0, 10.0), min_size=k, max_size=k).filter(
                    lambda row: sum(row) > 1e-3
                ),
                min_size=n,
                max_size=n,
            )
        )
    )
    proba = raw / raw.sum(axis=1, keepdims=True)
    labels = np.array(draw(st.lists(st.integers(0, k - 1), min_size=n, max_size=n)))
    weights = np.array(draw(st.lists(st.floats(0.0, 5.0), min_size=n, max_size=n)))
    assume(weights.sum() > 0)
    return proba, labels, np.arange(k), weights


@PROPERTY
@given(inputs=_scorer_inputs())
def test_scores_stay_in_bounds_for_every_valid_input(inputs) -> None:
    """The floored log loss is in [0, -log floor]; a share gap is in [0, 1]."""
    proba, labels, classes, weights = inputs
    scores = categorical_scores(proba, labels, classes, weights)
    assert 0.0 <= scores["log_loss"] <= -math.log(LOG_LOSS_FLOOR) + 1e-12
    assert -1e-12 <= scores["share_gap"] <= 1.0 + 1e-9


@PROPERTY
@given(inputs=_scorer_inputs(), data=st.data())
def test_scores_do_not_depend_on_the_order_of_the_classes(inputs, data) -> None:
    proba, labels, classes, weights = inputs
    order = np.array(data.draw(st.permutations(range(len(classes)))))
    base = categorical_scores(proba, labels, classes, weights)
    permuted = categorical_scores(proba[:, order], labels, classes[order], weights)
    assert permuted["log_loss"] == pytest.approx(base["log_loss"], rel=1e-12, abs=1e-12)
    assert permuted["share_gap"] == pytest.approx(
        base["share_gap"], rel=1e-9, abs=1e-12
    )


@PROPERTY
@given(
    inputs=_scorer_inputs(),
    data=st.data(),
    bad=st.one_of(
        st.floats(min_value=1.0, exclude_min=True, allow_nan=False),
        st.floats(max_value=-1e-300, allow_nan=False),
        st.just(math.nan),
    ),
)
def test_scores_refuse_any_value_outside_the_unit_interval(inputs, data, bad) -> None:
    proba, labels, classes, weights = inputs
    row = data.draw(st.integers(0, len(proba) - 1))
    column = data.draw(st.integers(0, proba.shape[1] - 1))
    proba = proba.copy()
    proba[row, column] = bad
    with pytest.raises(ValueError, match="proba must hold probabilities"):
        categorical_scores(proba, labels, classes, weights)


# ----------------------------------------------------------------------------
# A sign the donors show needs weight to resample
# ----------------------------------------------------------------------------


def _signed_table(sign: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.0, 4.0 * sign, 5.0 * sign]}
    )


@pytest.mark.parametrize(("sign", "name"), [(1.0, "positive"), (-1.0, "negative")])
def test_a_fit_refuses_a_sign_whose_rows_all_carry_zero_weight(sign, name) -> None:
    """Main crashed inside the forest's bootstrap with 'Probabilities contain NaN'."""
    with pytest.raises(ValueError, match=f"2 {name} donor row.*zero weight"):
        RegimeGatedQRF(n_estimators=2).fit(
            _signed_table(sign),
            ["x"],
            ["y"],
            weights=np.array([1.0, 1.0, 0.0, 0.0]),
        )


def test_a_checkpointed_chain_refuses_a_zero_weight_sign() -> None:
    table = _signed_table(1.0)
    weights = np.array([1.0, 1.0, 0.0, 0.0])
    model = RegimeGatedQRF(n_estimators=2)
    state = model.start_chain(table, ["x"], ["y"], weights=weights)
    with pytest.raises(ValueError, match="zero weight"):
        model.fit_draw_next(
            table,
            table[["x"]],
            pd.DataFrame(index=table.index),
            state=state,
            weights=weights,
        )


def test_a_sign_with_some_weight_and_a_weightless_zero_class_still_fit() -> None:
    """The fit refuses only a sign with no weight at all; nothing else moves."""
    table = _signed_table(1.0)
    for weights in ([1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]):
        model = RegimeGatedQRF(n_estimators=2, seed=3).fit(
            table, ["x"], ["y"], weights=np.array(weights)
        )
        assert np.isfinite(model.predict(table)["y"]).all()


def test_the_weighted_bootstrap_refuses_rows_without_weight() -> None:
    with pytest.raises(ValueError, match="rows with positive weight"):
        _weighted_bootstrap(
            np.zeros((2, 1)), np.ones(2), np.zeros(2), np.random.default_rng(0)
        )


# ----------------------------------------------------------------------------
# A fit refuses an empty donor table
# ----------------------------------------------------------------------------


def test_a_fit_refuses_an_empty_training_table() -> None:
    """Main fitted the target as degenerate zero and imputed 0 for everyone."""
    with pytest.raises(ValueError, match="at least one donor row"):
        RegimeGatedQRF(n_estimators=2).fit(
            pd.DataFrame({"x": [], "y": []}), ["x"], ["y"], weights="none"
        )


def test_a_chain_refuses_an_empty_training_table() -> None:
    with pytest.raises(ValueError, match="at least one donor row"):
        RegimeGatedQRF(n_estimators=2).start_chain(
            pd.DataFrame({"x": [], "y": []}), ["x"], ["y"], weights="none"
        )


# ----------------------------------------------------------------------------
# An empty recipient batch draws an empty frame
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("calibration", CALIBRATIONS)
def test_predict_on_an_empty_batch_returns_an_empty_frame(calibration) -> None:
    """A gated target's sign gate refused a zero-row predict."""
    model = _fit_gated(calibration=calibration)
    drawn = model.predict(EMPTY_BATCH)
    assert list(drawn.columns) == model.targets
    assert drawn.empty
    assert all(dtype == np.dtype("float64") for dtype in drawn.dtypes)


def test_predict_from_uniforms_on_an_empty_batch_is_empty() -> None:
    """A guard, not a fix: main already returned early here.

    The keyed chain step shares the gate draw but skips that early return, so
    it reached the sign gate's zero-row refusal (see the chain test below).
    """
    model = _fit_gated()
    empty = {target: np.zeros(0) for target in model.targets}
    drawn = model.predict_from_uniforms(
        EMPTY_BATCH, quantiles=empty, sign_uniforms=empty
    )
    assert list(drawn.columns) == model.targets
    assert drawn.empty


@FITTING_PROPERTY
@given(seed=st.integers(0, 2**32), calibration=st.sampled_from(CALIBRATIONS))
def test_an_empty_batch_consumes_no_draws(seed, calibration) -> None:
    """The next batch draws exactly as if the empty one had never run."""
    used, fresh = _fit_gated(seed, calibration), _fit_gated(seed, calibration)
    used.predict(EMPTY_BATCH)
    recipients = pd.DataFrame({"x": np.arange(6, dtype=float)})
    pd.testing.assert_frame_equal(used.predict(recipients), fresh.predict(recipients))


def test_an_empty_batch_still_releases_the_forests() -> None:
    model = _fit_gated()
    model.predict(EMPTY_BATCH, release_models=True)
    with pytest.raises(RuntimeError, match="released"):
        model.predict(pd.DataFrame({"x": [1.0]}))


@pytest.mark.parametrize("calibration", CALIBRATIONS)
def test_conditional_uniforms_on_an_empty_batch_are_empty(calibration) -> None:
    model = _fit_gated(calibration=calibration)
    observed = pd.DataFrame(
        {column: pd.Series([], dtype=float) for column in ["x", *model.targets]}
    )
    placement = {target: np.zeros(0) for target in model.targets}
    located = model.conditional_uniforms(
        observed, sign_placement=placement, quantile_placement=placement
    )
    assert all(len(values) == 0 for values in located.quantiles.values())
    assert all(len(values) == 0 for values in located.sign_uniforms.values())


@pytest.mark.parametrize("calibration", CALIBRATIONS)
@pytest.mark.parametrize("keyed", [False, True], ids=["positional", "keyed"])
def test_a_chain_step_on_an_empty_batch_draws_nothing(keyed, calibration) -> None:
    donors = _gated_donors()
    model = RegimeGatedQRF(n_estimators=2, seed=7, gate_calibration=calibration)
    state = model.start_chain(donors, ["x"], ["mixed"], weights="none")
    uniforms = {"quantiles": np.zeros(0), "sign_uniforms": np.zeros(0)} if keyed else {}
    step = model.fit_draw_next(
        donors,
        EMPTY_BATCH,
        pd.DataFrame(index=EMPTY_BATCH.index),
        state=state,
        weights="none",
        **uniforms,
    )
    assert step.raw_draw.shape == (0,)


# ----------------------------------------------------------------------------
# Donor distances: finite ranks, present categories, an explicit unseen policy
# ----------------------------------------------------------------------------


def _distance_donors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20.0, 40.0, 60.0],
            "pay": [0.0, 5.0, 9.0],
            "area": ["x", "y", "y"],
        }
    )


@pytest.mark.parametrize("columns", ["rank_columns", "presence_rank_columns"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, None])
@pytest.mark.parametrize("side", ["recipients", "donors"])
def test_rank_values_must_be_finite(columns, value, side) -> None:
    """A missing recipient age ranked 1.0, above every donor."""
    donors = _distance_donors()
    recipients = donors.iloc[:1].copy()
    frame = recipients if side == "recipients" else donors
    frame["age"] = frame["age"].astype(object)
    frame.loc[frame.index[0], "age"] = value
    with pytest.raises(ValueError, match=f"missing or non-finite value.*{side}"):
        donor_distance_features(recipients, donors, **{columns: ["age"]})


@pytest.mark.parametrize("side", ["recipients", "donors"])
def test_distances_refuse_a_missing_category(side) -> None:
    donors = _distance_donors()
    recipients = donors.iloc[:1].copy()
    frame = recipients if side == "recipients" else donors
    frame.loc[frame.index[0], "area"] = None
    with pytest.raises(ValueError, match=f"missing value.*{side}"):
        donor_distance_features(recipients, donors, scaled_categoricals={"area": 0.3})


def test_distances_refuse_an_unseen_recipient_category_by_default() -> None:
    """It became an all-zero dummy: silently the same distance from every donor."""
    with pytest.raises(ValueError, match=r"no donor holds: \['z'\]"):
        donor_distance_features(
            pd.DataFrame({"area": ["z", "x"]}),
            _distance_donors(),
            scaled_categoricals={"area": 0.3},
        )


def test_ignore_places_an_unseen_category_the_same_distance_from_every_donor() -> None:
    recipients, donors = donor_distance_features(
        pd.DataFrame({"area": ["z"]}),
        _distance_donors(),
        scaled_categoricals={"area": 0.3},
        unseen_categories="ignore",
    )
    np.testing.assert_array_equal(recipients, [[0.0, 0.0]])
    np.testing.assert_allclose(np.linalg.norm(donors - recipients, axis=1), 0.3)


def test_distances_refuse_an_unknown_unseen_policy() -> None:
    with pytest.raises(ValueError, match="unseen_categories must be one of"):
        donor_distance_features(
            _distance_donors(),
            _distance_donors(),
            scaled_categoricals={"area": 0.3},
            unseen_categories="warn",
        )


@pytest.mark.parametrize("side", ["recipients", "donors"])
def test_distances_refuse_a_label_naming_two_columns(side) -> None:
    donors = _distance_donors()
    recipients = donors.copy()
    if side == "recipients":
        recipients = _duplicated(recipients, "age")
    else:
        donors = _duplicated(donors, "age")
    with pytest.raises(ValueError, match=f"more than one column among the {side}"):
        donor_distance_features(recipients, donors, rank_columns=["age"])


def _main_distance_features(recipients, donors, *, rank, presence, scaled):
    """``donor_distance_features`` as main had it at ``dc00d158``, verbatim.

    The properties use it to pin that every valid input still gets exactly
    the coordinates it got before this validation existed.
    """

    def reference_ranks(values, reference):
        ordered = np.sort(reference)
        if len(ordered) == 0:
            return np.zeros(len(values))
        return np.searchsorted(ordered, values, side="right") / len(ordered)

    def coordinates(frame):
        parts = []
        for column in rank:
            parts.append(
                reference_ranks(
                    frame[column].to_numpy(np.float64),
                    donors[column].to_numpy(np.float64),
                )
            )
        for column in presence:
            values = frame[column].to_numpy(np.float64)
            reference = donors[column].to_numpy(np.float64)
            positive = values > 0
            parts.append(positive.astype(np.float64))
            parts.append(
                np.where(
                    positive, reference_ranks(values, reference[reference > 0]), 0.0
                )
            )
        for column, scale in scaled.items():
            levels = np.unique(donors[column].to_numpy())
            one_hot = frame[column].to_numpy()[:, None] == levels[None, :]
            parts.extend(scale * one_hot.astype(np.float64).T)
        return np.column_stack(parts)

    return coordinates(recipients), coordinates(donors)


_FINITE = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
_AMOUNT = st.one_of(st.just(0.0), _FINITE)
_AREAS = ["x", "y", "z"]


@st.composite
def _distance_frames(draw, *, unseen: bool = False):
    """Finite donors and recipients; recipients' areas seen unless ``unseen``."""
    n_donors = draw(st.integers(1, 12))
    n_recipients = draw(st.integers(0, 8))
    donor_area = draw(
        st.lists(st.sampled_from(_AREAS), min_size=n_donors, max_size=n_donors)
    )
    pool = sorted(set(donor_area)) + (["new"] if unseen else [])
    donors = pd.DataFrame(
        {
            "age": draw(st.lists(_FINITE, min_size=n_donors, max_size=n_donors)),
            "pay": draw(st.lists(_AMOUNT, min_size=n_donors, max_size=n_donors)),
            "area": donor_area,
        }
    )
    recipients = pd.DataFrame(
        {
            "age": draw(
                st.lists(_FINITE, min_size=n_recipients, max_size=n_recipients)
            ),
            "pay": draw(
                st.lists(_AMOUNT, min_size=n_recipients, max_size=n_recipients)
            ),
            "area": draw(
                st.lists(
                    st.sampled_from(pool), min_size=n_recipients, max_size=n_recipients
                )
            ),
        },
        dtype=object,
    ).astype({"age": float, "pay": float})
    return recipients, donors


_DISTANCE_SETTINGS = {
    "rank_columns": ["age"],
    "presence_rank_columns": ["pay"],
    "scaled_categoricals": {"area": 0.3},
}


@PROPERTY
@given(frames=_distance_frames())
def test_valid_input_gets_the_coordinates_it_got_before(frames) -> None:
    """A differential against main: the checks change no valid coordinate."""
    recipients, donors = frames
    expected = _main_distance_features(
        recipients, donors, rank=["age"], presence=["pay"], scaled={"area": 0.3}
    )
    for policy in ("refuse", "ignore"):
        actual = donor_distance_features(
            recipients, donors, unseen_categories=policy, **_DISTANCE_SETTINGS
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


@PROPERTY
@given(frames=_distance_frames(unseen=True))
def test_ignore_differs_from_main_nowhere_and_refuse_refuses_only_unseen_rows(
    frames,
) -> None:
    """``ignore`` is main's behaviour; ``refuse`` raises exactly when a row is unseen."""
    recipients, donors = frames
    expected = _main_distance_features(
        recipients, donors, rank=["age"], presence=["pay"], scaled={"area": 0.3}
    )
    ignored = donor_distance_features(
        recipients, donors, unseen_categories="ignore", **_DISTANCE_SETTINGS
    )
    np.testing.assert_array_equal(ignored[0], expected[0])
    np.testing.assert_array_equal(ignored[1], expected[1])
    unseen = ~recipients["area"].isin(set(donors["area"])).to_numpy()
    if unseen.any():
        with pytest.raises(ValueError, match="no donor holds"):
            donor_distance_features(recipients, donors, **_DISTANCE_SETTINGS)
        # The unseen rows are exactly the rows with no area indicator set.
        indicators = ignored[0][:, 3:]
        np.testing.assert_array_equal(~indicators.any(axis=1), unseen)
    else:
        donor_distance_features(recipients, donors, **_DISTANCE_SETTINGS)


@PROPERTY
@given(frames=_distance_frames())
def test_ranks_lie_in_the_unit_interval_and_rise_with_the_value(frames) -> None:
    recipients, donors = frames
    features, _ = donor_distance_features(recipients, donors, **_DISTANCE_SETTINGS)
    age_rank, pay_positive, pay_rank = features[:, 0], features[:, 1], features[:, 2]
    assert ((age_rank >= 0) & (age_rank <= 1)).all()
    assert ((pay_rank >= 0) & (pay_rank <= 1)).all()
    assert set(np.unique(pay_positive)) <= {0.0, 1.0}
    np.testing.assert_array_equal(pay_rank[pay_positive == 0], 0.0)
    order = np.argsort(recipients["age"].to_numpy(), kind="stable")
    assert (np.diff(age_rank[order]) >= 0).all()
