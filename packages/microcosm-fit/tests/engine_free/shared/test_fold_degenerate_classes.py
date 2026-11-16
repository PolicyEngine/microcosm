"""Folds whose training rows are degenerate in a class: a single class, or a weightless one.

``out_of_fold_proba`` fits the categorical model on each fold's training rows.
When those rows hold a single class the model fits no learner, and each
learner's rows in that fold used to stay at zero, which
``categorical_scores`` now refuses (microcosm#1543); before that it scored
them at the log-loss floor. Those rows now take the model's probabilities:
certainty of the one class. ``_class_balanced_weights``
divides each class's weights by that class's total, so a class with no
weight gave NaN; it now refuses the class by name. The cross-fitted gate never
passed it such weights, because it falls back to the in-sample gate first
(``GateOutcome.WEIGHTLESS_FOLD``), testing the same totals; the guards below
pin that. A class whose total overflows to infinity does carry weight, so a
fit can pass it to the helper; the helper leaves it as it was (its rows weigh
zero, and if every class overflows every row's weight is NaN), and a guard
pins that such a fit is unchanged.

The example tests that name a refusal or a filled row fail before the fixes;
the guards pass on both sides. The properties check that, on the small
tables they draw, every array ``out_of_fold_proba`` returns is a distribution
in every row, that the fix
moves only rows that were all zeros, that folds dealt by household put the
rarest class in every fold and, with three or more folds, leave two classes in
every training fold, and that the balanced weights keep the previous formula's
bits wherever no class is weightless.
"""

from __future__ import annotations

import contextlib
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

from microcosm.fit import (
    BoostedClassifierSettings,
    CategoricalClassifier,
    GateCalibration,
    GateOutcome,
    RegimeGatedQRF,
    can_deal_folds,
    categorical_scores,
    out_of_fold_proba,
    qrf,
    stratified_folds,
)
from microcosm.fit.qrf import _class_balanced_weights, _cross_fit_gate

#: Cheap property runs: no fitting.
PROPERTY = settings(
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

#: Property runs whose every example fits classifiers on each fold.
FITTING_PROPERTY = settings(
    max_examples=20,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

#: A small classifier that fits any two-class table: no early-stopping split,
#: which needs two rows of every class.
SMALL = CategoricalClassifier(
    boosted=BoostedClassifierSettings(
        max_iter=10, min_samples_leaf=2, early_stopping=False
    )
)

LEARNERS = ("boosted", "spline_logit")


def _previous_out_of_fold_proba(
    model: CategoricalClassifier,
    table: pd.DataFrame,
    predictors: list[str],
    target: str,
    *,
    weights: np.ndarray,
    folds: np.ndarray,
) -> dict[str, np.ndarray]:
    """``out_of_fold_proba`` as it stood before this fix, copied verbatim."""
    weights = np.asarray(weights, dtype=np.float64)
    folds = np.asarray(folds)
    if weights.shape != (len(table),) or folds.shape != (len(table),):
        raise ValueError("weights and folds need one value per row.")
    labels = table[target].to_numpy()
    classes = np.unique(labels)
    out: dict[str, np.ndarray] = {"classes": classes}

    def full(fitted_classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
        result = np.zeros((len(proba), len(classes)))
        result[:, np.searchsorted(classes, fitted_classes)] = proba
        return result

    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        fitted = model.fit(
            table.loc[train], predictors, [target], weights=weights[train]
        )
        scored = {"model": fitted.predict_proba(table.loc[test])}
        scored.update(fitted.component_proba(table.loc[test]))
        shares = pd.Series(weights[train]).groupby(labels[train]).sum()
        shares = shares.reindex(classes, fill_value=0.0).to_numpy()
        for name, proba in scored.items():
            out.setdefault(name, np.zeros((len(table), len(classes))))
            out[name][test] = full(fitted.classes_, proba)
        out.setdefault("marginal", np.zeros((len(table), len(classes))))
        out["marginal"][test] = shares / shares.sum()
    return out


def _previous_class_balanced_weights(
    labels: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    """``_class_balanced_weights`` as it stood before this fix, copied verbatim."""
    base = (
        np.ones(len(labels), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    balanced = np.empty(len(labels), dtype=np.float64)
    for value in np.unique(labels):
        members = labels == value
        balanced[members] = base[members] / base[members].sum()
    return balanced * (len(labels) / balanced.sum())


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal dtype, shape and bytes; an object array (the class labels) compares by value."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == object:
        return a.tolist() == b.tolist()
    return a.tobytes() == b.tobytes()


def _single_class_rows(labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Rows in a fold whose training rows hold a single class."""
    rows = np.zeros(len(labels), dtype=bool)
    for fold in np.unique(folds):
        if len(np.unique(labels[folds != fold])) == 1:
            rows |= folds == fold
    return rows


def _assert_every_array_scores(
    out: dict[str, np.ndarray], labels: np.ndarray, weights: np.ndarray
) -> None:
    for name, proba in out.items():
        if name == "classes":
            continue
        assert proba.shape == (len(labels), len(out["classes"]))
        assert ((proba >= 0.0) & (proba <= 1.0)).all(), name
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9, err_msg=name)
        categorical_scores(proba, labels, out["classes"], weights)


# --- out_of_fold_proba: a fold whose training rows hold a single class -------


def _blocks() -> tuple[pd.DataFrame, np.ndarray]:
    """Three folds of 20 rows; the third holds every ``b``, so its training rows hold only ``a``."""
    rng = np.random.default_rng(0)
    n = 60
    table = pd.DataFrame(
        {"x": rng.normal(size=n), "y": np.where(np.arange(n) < 40, "a", "b")}
    )
    return table, np.repeat([0, 1, 2], 20)


def test_a_single_class_training_fold_gives_each_learner_the_models_rows() -> None:
    """Each learner's rows in that fold were zeros, which the scorer refuses."""
    table, folds = _blocks()
    weights = np.ones(len(table))
    out = out_of_fold_proba(SMALL, table, ["x"], "y", weights=weights, folds=folds)
    assert set(out) == {"classes", "model", "marginal", *LEARNERS}
    held = folds == 2
    certain_a = np.tile([1.0, 0.0], (int(held.sum()), 1))
    for name in ("model", "marginal", *LEARNERS):
        np.testing.assert_array_equal(out[name][held], certain_a, err_msg=name)
    _assert_every_array_scores(out, table["y"].to_numpy(), weights)


def test_two_grouped_folds_can_leave_a_training_fold_with_one_class() -> None:
    """Households {a, b}, {a} and {b}: dealt into two folds, one fold's training rows hold only ``a``.

    Every class sits in two households, so ``can_deal_folds`` passes, and the
    seed deals {a} alone into fold 1.
    """
    rng = np.random.default_rng(3)
    household = np.repeat([0, 0, 1, 2], 10)
    labels = np.repeat(["a", "b", "a", "b"], 10)
    table = pd.DataFrame({"x": rng.normal(size=40), "y": labels})
    assert can_deal_folds(labels, 2, groups=household)
    folds = stratified_folds(labels, 2, np.random.default_rng(0), groups=household)
    assert set(labels[folds == 1]) == {"a"}
    weights = rng.uniform(0.5, 2.0, 40)
    out = out_of_fold_proba(SMALL, table, ["x"], "y", weights=weights, folds=folds)
    assert set(out) == {"classes", "model", "marginal", *LEARNERS}
    _assert_every_array_scores(out, labels, weights)


def test_a_learner_no_fold_fitted_stays_out() -> None:
    """Guard: a target with one class, or folds that each hold one class, fit no learner.

    The UK APS stage pools a learner only where every major group fitted it,
    and a group with a single sub-major group fits neither, so these keep
    their old output.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=40)
    weights = np.ones(40)
    one_class = pd.DataFrame({"x": x, "y": "a"})
    out = out_of_fold_proba(
        SMALL, one_class, ["x"], "y", weights=weights, folds=np.arange(40) % 4
    )
    assert set(out) == {"classes", "model", "marginal"}
    _assert_every_array_scores(out, one_class["y"].to_numpy(), weights)

    split = pd.DataFrame({"x": x, "y": np.repeat(["a", "b"], 20)})
    folds = np.repeat([0, 1], 20)
    out = out_of_fold_proba(SMALL, split, ["x"], "y", weights=weights, folds=folds)
    assert set(out) == {"classes", "model", "marginal"}
    _assert_every_array_scores(out, split["y"].to_numpy(), weights)


def test_folds_that_each_keep_two_classes_are_unchanged() -> None:
    """Guard: with every training fold holding two classes, the output is the previous one, bit for bit."""
    rng = np.random.default_rng(1)
    n = 120
    x = rng.normal(size=n)
    table = pd.DataFrame(
        {"x": x, "y": np.where(x + rng.normal(0, 0.7, n) > 0, "up", "down")}
    )
    weights = rng.uniform(0.5, 2.0, n)
    folds = stratified_folds(table["y"].to_numpy(), 5, np.random.default_rng(2))
    model = CategoricalClassifier(boosted=BoostedClassifierSettings(max_iter=20))
    new = out_of_fold_proba(model, table, ["x"], "y", weights=weights, folds=folds)
    old = _previous_out_of_fold_proba(
        model, table, ["x"], "y", weights=weights, folds=folds
    )
    assert list(new) == list(old)
    for name in new:
        assert _same_bits(new[name], old[name]), name


def _fold_example(
    codes: list[int], folds: list[int], boosted_weight: float
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """A ``_fold_tables`` case with the given class codes and folds."""
    rng = np.random.default_rng(0)
    n = len(codes)
    table = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    table["y"] = np.array(["c0", "c1", "c2"])[codes]
    return table, np.asarray(folds), rng.uniform(0.25, 4.0, n), boosted_weight


#: Six ``c0`` rows over three folds and six ``c1`` rows in the third, whose
#: training rows then hold only ``c0``.
_CONFINED = ([0] * 6 + [1] * 6, [0, 1, 2] * 2 + [2] * 6)
#: Each class in a fold of its own: every training fold holds one class.
_SEPARATED = ([0] * 6 + [1] * 6, [0] * 6 + [1] * 6)


@st.composite
def _fold_tables(draw):
    """Small tables with two or three classes and two to four folds.

    ``confined`` draws put every row of the classes after the first into one
    fold, so that fold's training rows hold a single class; ``random`` draws
    deal folds at random. The property's examples add a confined fold for the
    boosted-only and spline-only models, and folds that each hold one class.
    """
    n = draw(st.integers(12, 36))
    n_classes = draw(st.integers(2, 3))
    n_folds = draw(st.integers(2, 4))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    codes = np.r_[np.arange(n_classes), rng.integers(0, n_classes, n - n_classes)]
    rng.shuffle(codes)
    labels = np.array(["c0", "c1", "c2"])[codes]
    if draw(st.sampled_from(["confined", "random"])) == "random":
        folds = rng.integers(0, n_folds, n)
        folds[:n_folds] = np.arange(n_folds)
    else:
        held = draw(st.integers(0, n_folds - 1))
        folds = np.where(codes == 0, rng.integers(0, n_folds, n), held)
        folds[np.flatnonzero(codes == 0)[:1]] = (held + 1) % n_folds
    table = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    table["y"] = labels
    weights = rng.uniform(0.25, 4.0, n)
    boosted_weight = draw(st.sampled_from([0.5, 0.0, 1.0]))
    return table, folds, weights, boosted_weight


@FITTING_PROPERTY
@given(case=_fold_tables())
@example(case=_fold_example(*_CONFINED, boosted_weight=1.0))
@example(case=_fold_example(*_CONFINED, boosted_weight=0.0))
@example(case=_fold_example(*_SEPARATED, boosted_weight=0.5))
def test_every_array_is_a_distribution_and_only_zero_rows_move(case) -> None:
    """Every array scores, and the fix moves only learner rows that were all zeros.

    Every other row keeps the previous implementation's bits, and the moved
    rows equal the model's.
    """
    table, folds, weights, boosted_weight = case
    model = CategoricalClassifier(
        boosted_weight=boosted_weight,
        boosted=BoostedClassifierSettings(
            max_iter=10, min_samples_leaf=2, early_stopping=False
        ),
    )
    labels = table["y"].to_numpy()
    new = out_of_fold_proba(model, table, ["x", "z"], "y", weights=weights, folds=folds)
    old = _previous_out_of_fold_proba(
        model, table, ["x", "z"], "y", weights=weights, folds=folds
    )
    _assert_every_array_scores(new, labels, weights)
    assert set(new) == set(old)
    moved = _single_class_rows(labels, folds)
    if moved.all():
        event("every training fold holds one class")
    elif moved.any():
        event(f"a single-class training fold, boosted_weight={boosted_weight}")
    else:
        event("no single-class fold")
    for name in new:
        if name == "classes":
            assert _same_bits(new[name], old[name])
            continue
        assert _same_bits(new[name][~moved], old[name][~moved]), name
        if name in LEARNERS:
            assert not old[name][moved].any(), name
            np.testing.assert_array_equal(new[name][moved], new["model"][moved])
        else:
            assert _same_bits(new[name][moved], old[name][moved]), name
    expected = {"classes", "model", "marginal"}
    if not moved.all():
        expected |= {"boosted"} if boosted_weight > 0 else set()
        expected |= {"spline_logit"} if boosted_weight < 1 else set()
    assert set(new) == expected


def _household_case(
    members: list[list[int]], folds: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Row labels and household codes for households holding the given classes."""
    labels = np.array([value for m in members for value in m])
    groups = np.array([g for g, m in enumerate(members) for _ in m])
    return labels, groups, folds, seed


@st.composite
def _households(draw):
    """Households, each class in at least ``folds`` of them, often exactly that many.

    ``singletons`` gives each class ``folds`` or ``folds + 1`` households of
    one member, so a dealing that ignores the classes can leave a fold without
    one. ``paired`` puts class 0 into ``folds`` households beside class 1 and
    adds households of class 1 alone, so a dealing by any class but the rarest
    can crowd class 0 into fewer folds. ``mixed`` draws households of one to
    four members.
    """
    folds = draw(st.integers(2, 6))
    n_classes = draw(st.integers(2, 4))
    shape = draw(st.sampled_from(["singletons", "paired", "mixed"]))
    if shape == "singletons":
        members = [
            [value]
            for value in range(n_classes)
            for _ in range(folds + draw(st.integers(0, 1)))
        ]
    elif shape == "paired":
        members = [[0, 1] for _ in range(folds)]
        members += [[1] for _ in range(draw(st.integers(1, 2 * folds)))]
    else:
        members = draw(
            st.lists(
                st.lists(st.integers(0, n_classes - 1), min_size=1, max_size=4),
                min_size=folds,
                max_size=14,
            )
        )
    for value in range(n_classes):
        lacking = [m for m in members if value not in m]
        short = folds - (len(members) - len(lacking))
        for household in lacking[: max(short, 0)]:
            household.append(value)
    order = draw(st.permutations(range(len(members))))
    members = [members[i] for i in order]
    return _household_case(members, folds, draw(st.integers(0, 2**32 - 1)))


@PROPERTY
@given(case=_households())
@example(case=_household_case([[0]] * 3 + [[1]] * 3, folds=3, seed=0))
@example(case=_household_case([[0, 1]] * 3 + [[1]] * 4, folds=3, seed=0))
@example(case=_household_case([[0, 1]] * 3, folds=3, seed=4))
@example(case=_household_case([[0, 1], [0], [1]], folds=2, seed=0))
def test_grouped_folds_put_the_rarest_class_in_every_fold(case) -> None:
    """Dealt by household, the rarest class sits in every fold; with three or more folds no training fold holds one class.

    Dealt by row, every class sits in every fold. Dealt by household, each
    household goes with its rarest class, counted in households (the first in
    sorted order if several tie). So every household holding the rarest class
    R is dealt round the folds in R's stratum, and R sits in every fold. A
    training fold holding one class then holds only R, and the class R2 that
    comes next in that order would have to sit wholly in the held-out fold.
    Households holding R2 but not R are dealt round the folds in R2's stratum,
    so at most one of them can sit there; the rest hold R, at most
    ``ceil(n_R / K)`` to a fold, where ``n_R <= n_R2``. That needs
    ``n_R2 - 1 <= ceil(n_R2 / K)``, which fails for every ``K >= 3`` once
    ``can_deal_folds`` gives ``n_R2 >= K``. With two folds it can happen. The
    UK SPI stage deals households into five folds.
    """
    labels, groups, folds, seed = case
    assert can_deal_folds(labels, folds, groups=groups)
    by_row = stratified_folds(labels, folds, np.random.default_rng(seed))
    for fold in range(folds):
        assert set(labels[by_row == fold]) == set(labels)
    by_group = stratified_folds(
        labels, folds, np.random.default_rng(seed), groups=groups
    )
    assert (pd.Series(by_group).groupby(groups).nunique() == 1).all()
    counts = (
        pd.DataFrame({"label": labels, "group": groups})
        .drop_duplicates()
        .groupby("label")
        .size()
    )
    rarest = counts.index[int(np.argmin(counts.to_numpy()))]
    for fold in range(folds):
        assert (labels[by_group == fold] == rarest).any(), fold
    one_class = [f for f in range(folds) if len(np.unique(labels[by_group != f])) < 2]
    at_bound = (
        "a class at the bound" if counts.min() == folds else "every class above it"
    )
    if folds >= 3:
        assert one_class == []
        event(f"three or more folds, {at_bound}")
    else:
        event("two folds" + (", a one-class training fold" if one_class else ""))


# --- _class_balanced_weights: a class with no weight -------------------------


def test_balanced_weights_refuse_a_class_with_no_weight_by_name() -> None:
    """It divided by the class's total weight and returned NaN weights."""
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match=r"class 1 carries 0\.0"):
        _class_balanced_weights(labels, np.array([1.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="class -1 carries 0.0"):
        _class_balanced_weights(np.array([-1, 0, 1]), np.array([0.0, 2.0, 1.0]))


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_balanced_weights_leave_a_class_whose_total_overflows_as_it_was() -> None:
    """Guard: a total that overflows still counts as weight, so the helper does not refuse it.

    Dividing by the infinite total gives the class's rows zero weight, as
    before. A fit can reach this (see the guard below), so refusing it would
    change that fit.
    """
    labels = np.array([0, 0, 1, 1])
    weights = np.array([1e308, 1e308, 1.0, 1.0])
    balanced = _class_balanced_weights(labels, weights)
    assert _same_bits(balanced, _previous_class_balanced_weights(labels, weights))
    np.testing.assert_array_equal(balanced, [0.0, 0.0, 2.0, 2.0])


@st.composite
def _labelled_weights(draw):
    n = draw(st.integers(1, 30))
    labels = np.asarray(
        draw(st.lists(st.sampled_from([-1, 0, 1]), min_size=n, max_size=n))
    )
    if draw(st.booleans()):
        return labels, None
    weight = st.one_of(st.just(0.0), st.floats(1e-6, 1e6), st.just(1e308))
    return labels, np.asarray(draw(st.lists(weight, min_size=n, max_size=n)))


@PROPERTY
@given(case=_labelled_weights())
def test_balanced_weights_keep_their_bits_and_refuse_exactly_the_weightless(
    case,
) -> None:
    """Where every class carries weight the result is the previous one, bit for bit.

    That includes a class whose total overflows. Otherwise the helper refuses,
    naming a class with no weight.
    """
    labels, weights = case
    base = np.ones(len(labels)) if weights is None else weights
    with np.errstate(over="ignore"):
        totals = {int(v): base[labels == v].sum() for v in np.unique(labels)}
    weightless = [v for v, total in totals.items() if not total > 0.0]
    overflowing = any(np.isinf(total) for total in totals.values())
    if weightless:
        event("a weightless class")
        with pytest.raises(ValueError, match="carries 0.0") as raised:
            with np.errstate(over="ignore"):
                _class_balanced_weights(labels, weights)
        assert f"class {weightless[0]} carries" in str(raised.value)
        return
    event("a class whose total overflows" if overflowing else "every class weighted")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # An overflowing total gives its class zero weight, and if every class
        # overflows the rescaling divides by zero: as before, so let it pass.
        with np.errstate(all="ignore") if overflowing else contextlib.nullcontext():
            balanced = _class_balanced_weights(labels, weights)
            previous = _previous_class_balanced_weights(labels, weights)
    assert _same_bits(balanced, previous)
    if overflowing:
        return
    class_totals = [balanced[labels == v].sum() for v in totals]
    np.testing.assert_allclose(class_totals, class_totals[0], rtol=1e-9)
    np.testing.assert_allclose(balanced.mean(), 1.0, rtol=1e-9)


def _one_weighted_positive(n: int = 200, positives: int = 30) -> pd.DataFrame:
    """A zero-inflated target whose positive sign carries weight on a single row."""
    rng = np.random.default_rng(9)
    x = rng.normal(size=n)
    positive = np.zeros(n, dtype=bool)
    positive[rng.choice(n, size=positives, replace=False)] = True
    weight = np.where(positive, 0.0, 1.0)
    weight[np.flatnonzero(positive)[0]] = 1.0
    return pd.DataFrame(
        {
            "x": x,
            "target": np.where(positive, rng.lognormal(1.0, 0.5, n), 0.0),
            "weight": weight,
        }
    )


def test_a_sign_weighted_on_one_row_falls_back_to_the_in_sample_gate() -> None:
    """Guard: the balanced gate never divides by a fold's missing weight.

    The positive sign carries weight overall, but the fold holding its one
    weighted row trains on none, so the fit takes the in-sample gate before
    any weights are balanced.
    """
    table = _one_weighted_positive()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fitted = RegimeGatedQRF(
            n_estimators=4,
            seed=5,
            gate_calibration=GateCalibration.CROSS_FITTED_BALANCED,
        ).fit(table, ["x"], ["target"], weights="weight")
    assert fitted.gate_outcomes() == {"target": GateOutcome.WEIGHTLESS_FOLD}


def test_a_fit_whose_class_total_overflows_is_unchanged() -> None:
    """Guard: a balanced cross-fit whose zero sign's weight total overflows fits as before.

    Every weight is finite, so the fit takes them, and every fold's training
    rows carry both signs, so the gate balances them. The fit and its draws
    match a fit with the previous helper patched in, bit for bit.
    """
    rng = np.random.default_rng(3)
    n = 120
    positive = rng.uniform(size=n) < 0.4
    table = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "target": np.where(positive, rng.lognormal(1.0, 0.5, n), 0.0),
            "weight": np.where(positive, 1.0, 1e308),
        }
    )
    model = RegimeGatedQRF(
        n_estimators=4,
        seed=5,
        gate_calibration=GateCalibration.CROSS_FITTED_BALANCED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = model.fit(table, ["x"], ["target"], weights="weight")
        draws = fitted.predict(table[["x"]])
        with mock.patch.object(
            qrf, "_class_balanced_weights", _previous_class_balanced_weights
        ):
            before = model.fit(table, ["x"], ["target"], weights="weight")
            before_draws = before.predict(table[["x"]])
    assert fitted.gate_outcomes() == before.gate_outcomes()
    assert fitted.gate_outcomes()["target"] != GateOutcome.WEIGHTLESS_FOLD
    pd.testing.assert_frame_equal(draws, before_draws, check_exact=True)


@st.composite
def _gate_cases(draw):
    """Two or three sign classes, each with at least five rows, and weights that are often zero."""
    n_classes = draw(st.integers(2, 3))
    sizes = draw(st.lists(st.integers(5, 14), min_size=n_classes, max_size=n_classes))
    labels = np.repeat(np.arange(n_classes) - (n_classes == 3), sizes)
    zero_share = draw(st.sampled_from([0.0, 0.5, 0.9, 0.9]))
    seed = draw(st.integers(0, 2**31 - 2))
    rng = np.random.default_rng(seed)
    weights = np.where(
        rng.uniform(size=len(labels)) < zero_share,
        0.0,
        rng.choice([0.5, 1.0, 3.0], size=len(labels)),
    )
    for value in np.unique(labels):
        if not weights[labels == value].any():
            weights[np.flatnonzero(labels == value)[0]] = 1.0
    x = rng.normal(size=(len(labels), 2))
    x[:, 0] += labels
    return x, labels, weights, seed


@FITTING_PROPERTY
@given(case=_gate_cases())
def test_a_cross_fit_balances_only_folds_that_carry_every_class(case) -> None:
    """Guard: the balanced gate meets a weightless class only as a fallback.

    Where some fold's training rows carry no weight for a class, the gate
    falls back before balancing anything; otherwise every class it balances
    carries weight.
    """
    features, labels, weights, seed = case
    fold_stream, _ = np.random.SeedSequence(seed).spawn(2)
    folds = qrf._stratified_folds(
        labels, qrf._GATE_CROSS_FIT_FOLDS, np.random.default_rng(fold_stream)
    )
    weightless = any(
        not weights[(folds != fold) & (labels == value)].sum() > 0.0
        for fold in range(qrf._GATE_CROSS_FIT_FOLDS)
        for value in np.unique(labels)
    )
    event("a weightless fold" if weightless else "every fold weighted")
    totals: list[float] = []

    def recording(fold_labels, fold_weights):
        totals.extend(
            float(fold_weights[fold_labels == value].sum())
            for value in np.unique(fold_labels)
        )
        return _class_balanced_weights(fold_labels, fold_weights)

    with mock.patch.object(qrf, "_class_balanced_weights", recording):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            gate, outcome = _cross_fit_gate(
                features, labels, weights, seed, balanced=True
            )
    if weightless:
        assert (gate, outcome) == (None, GateOutcome.WEIGHTLESS_FOLD)
        assert totals == []
    else:
        assert outcome != GateOutcome.WEIGHTLESS_FOLD
        assert len(totals) == qrf._GATE_CROSS_FIT_FOLDS * len(np.unique(labels))
        assert min(totals) > 0.0
