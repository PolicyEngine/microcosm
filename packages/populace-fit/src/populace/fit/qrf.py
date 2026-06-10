"""The canonical conditional model: a regime-gated, chained, weighted QRF.

This is the from-scratch successor to microimpute's regime-gated QRF imputer,
reimplemented against the :class:`~populace.frame.Frame`. Three ideas combine:

**Weighted bootstrap (forests only).** ``quantile_forest`` (and random forests
generally) cannot honor a ``sample_weight`` in their *predictive* distribution:
a fully-grown leaf holds one training row, so weighting impurity does not move
the value a draw reads out, and the backend uses ``sample_weight`` only as a
zero-weight filter on leaf membership. So weights are materialized *into the
data*: before each forest is grown, training rows are drawn with replacement
with probability proportional to weight (:func:`_weighted_bootstrap`). The leaf
distributions then reflect the weighted population. This is the microimpute#196
fix — the mechanism that makes a weighted fit actually shift the draws.

**Regime gates.** A numeric target's sign support — which of
{negative, zero, positive} appear in training — defines its *regime*. A single
regressor over a zero-inflated or sign-mixed target either loses a tail (the
"fit on ``y > 0``" pattern drops the negatives) or interpolates across the zero
crossing (predicting values in the empty gap between the negative and positive
clusters). So a classifier gates each row into its sign class, and a separate
forest models the magnitude within each nonzero sign. Regime detection is
**structural**: it reads the unweighted support, because which signs *exist* is
a fact about the variable, not about the population's weighting. The gate is
weighted *directly* by ``sample_weight`` — which the histogram classifier
honors exactly — **not** by the forests' bootstrap: an n-of-n weighted
resample would delete a vanishingly rare sign class outright (a positive row at
weight 1 among thousands of zeros at weight 50 is drawn with probability ~4e-5),
collapsing the gate to a single class that can never draw the missing sign.

**Chaining.** Targets are imputed sequentially; each conditions on the
predictors plus the targets already drawn, so the joint structure across
targets survives (sequential / chained-equations imputation).

Draws sample the weighted conditional: a quantile ``q ~ Uniform(0, 1)`` is drawn
per row from the model's seeded RNG and the forest is queried at it, so over
rows the draws reproduce the (weighted) conditional distribution, not a point
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

from populace.fit.model import (
    DESIGN_WEIGHTS,
    WeightSpec,
    predictors_targets_entity,
    resolve_fit_weights,
)
from populace.frame import Frame

__all__ = [
    "RegimeGatedQRF",
    "FittedRegimeGatedQRF",
    "Regime",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_ZERO_ATOL",
]

#: Default number of trees per forest. Enough for stable leaf distributions on
#: the pool sizes this operator targets, small enough for fast CI fits.
DEFAULT_N_ESTIMATORS = 100

#: Absolute tolerance for "equals zero" in regime detection. A magnitude at or
#: below this counts as a structural zero (the gate's zero class).
DEFAULT_ZERO_ATOL = 1e-6


class Regime:
    """Sign-support regime labels for a numeric target.

    The label records which sign classes appear in the (unweighted) training
    support and therefore which gate + forests the target needs. Exposed as
    constants so callers can match on a fitted model's regimes without magic
    strings.
    """

    #: Negative, zero, and positive all present: a three-way sign gate plus a
    #: positive-magnitude and a negative-magnitude forest.
    THREE_SIGN = "three_sign"
    #: Zero and positive only: a zero-vs-positive gate plus a positive forest.
    ZERO_INFLATED_POSITIVE = "zero_inflated_positive"
    #: Zero and negative only: a zero-vs-negative gate plus a negative forest.
    ZERO_INFLATED_NEGATIVE = "zero_inflated_negative"
    #: Both signs, no zeros: a sign gate plus a forest per sign.
    SIGN_ONLY = "sign_only"
    #: Strictly positive: one forest, no gate.
    POSITIVE_ONLY = "positive_only"
    #: Strictly negative: one forest, no gate.
    NEGATIVE_ONLY = "negative_only"
    #: Constant zero in training: every draw is exactly zero, no model.
    DEGENERATE_ZERO = "degenerate_zero"


def detect_regime(y: np.ndarray, *, zero_atol: float) -> str:
    """Classify a target's training support into a :class:`Regime`.

    A sign class counts as present when at least one training value falls in it.
    Detection is unweighted on purpose: the *existence* of a sign is structural,
    a property of the variable, not of the population the weights describe.

    Args:
        y: Training target values.
        zero_atol: Magnitudes at or below this are zeros.

    Returns:
        One of the :class:`Regime` label constants.
    """
    if y.size == 0:
        return Regime.DEGENERATE_ZERO
    has_zero = bool((np.abs(y) <= zero_atol).any())
    has_pos = bool((y > zero_atol).any())
    has_neg = bool((y < -zero_atol).any())

    if has_pos and has_neg and has_zero:
        return Regime.THREE_SIGN
    if has_pos and has_neg:
        return Regime.SIGN_ONLY
    if has_pos and has_zero:
        return Regime.ZERO_INFLATED_POSITIVE
    if has_neg and has_zero:
        return Regime.ZERO_INFLATED_NEGATIVE
    if has_pos:
        return Regime.POSITIVE_ONLY
    if has_neg:
        return Regime.NEGATIVE_ONLY
    return Regime.DEGENERATE_ZERO


def _validate_targets_finite(table: pd.DataFrame, targets: list[str]) -> None:
    """Raise unless every target column is entirely finite.

    A NaN target is the silent corruption this guards against: the sign labels
    (``y > zero_atol`` / ``y < -zero_atol``) are both ``False`` for NaN, so a
    NaN row is relabeled to the *zero* class — a missing value masquerading as a
    structural zero, NaN-blind. The model has no notion of missingness, so the
    only sound contract is to require finite targets and refuse otherwise,
    naming the offending column and its NaN count so the caller can find it.

    Predictors are not checked here: a forest can split around NaN features and
    a missing predictor is not silently miscoded the way a missing target is.

    Args:
        table: The entity table the fit reads targets from.
        targets: Target column names.

    Raises:
        ValueError: If any target column contains non-finite values. The message
            names the first offending column and its NaN/inf count.
    """
    for target in targets:
        values = table[target].to_numpy(dtype=np.float64)
        non_finite = int((~np.isfinite(values)).sum())
        if non_finite:
            raise ValueError(
                f"Target column {target!r} contains {non_finite} non-finite "
                f"value(s) (NaN/inf) out of {len(values)}. A NaN target would be "
                "silently relabeled to the zero class (the sign labels are "
                "NaN-blind); fit requires finite targets. Drop or impute the "
                f"missing {target!r} values before fitting."
            )


def _weighted_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize weights by importance-resampling the training rows.

    Draws ``len(x)`` rows with replacement with probability proportional to
    weight, so the resampled data carries the weighted distribution. With
    ``weights=None`` the data is returned unchanged (the explicit unweighted
    path). This is the operative half of the microimpute#196 fix: it is what
    makes leaf distributions — and the values drawn from them — weighted.

    Args:
        x: Feature matrix, one row per training record.
        y: Target vector aligned with ``x``.
        weights: Per-row weights (non-negative, not all zero), or ``None`` to
            return the data unchanged.
        rng: Seeded generator the resample draws from.

    Returns:
        ``(x_resampled, y_resampled)`` with the same row count as the input.
    """
    if weights is None:
        return x, y
    total = float(weights.sum())
    probabilities = weights / total
    selected = rng.choice(len(x), size=len(x), replace=True, p=probabilities)
    return x[selected], y[selected]


def _make_gate(seed: int) -> HistGradientBoostingClassifier:
    """Build the sign-gate classifier.

    A histogram gradient-boosted classifier; on the zero-inflated PolicyEngine
    targets this calibrates the zero/nonzero probability better than a small
    random forest, which matters because the gate's probability *is* the share
    of draws that come out zero.
    """
    return HistGradientBoostingClassifier(random_state=seed)


def _interp_rows(
    quantiles: np.ndarray, grid: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    """Per-row linear interpolation of grid predictions at per-row quantiles.

    Equivalent to ``[np.interp(q[i], grid, predictions[i]) for i]`` but
    vectorized: ``grid`` (the quantile knots) is shared across rows, so one
    ``searchsorted`` locates every row's bracket at once and the interpolation
    is a single weighted blend. Quantiles outside ``grid`` clamp to the end
    values (the ``np.interp`` convention), so ``q`` at/near 0 or 1 reads the
    observed conditional min/max.

    Args:
        quantiles: One quantile per row, shape ``(m,)``.
        grid: Ascending quantile knots the forest was queried at, shape
            ``(g,)``.
        predictions: Predicted values, shape ``(m, g)``, row-aligned with
            ``quantiles`` and column-aligned with ``grid``.

    Returns:
        One interpolated value per row, shape ``(m,)``.
    """
    upper = np.searchsorted(grid, quantiles, side="left")
    upper = np.clip(upper, 1, len(grid) - 1)
    lower = upper - 1
    grid_lo = grid[lower]
    grid_hi = grid[upper]
    span = grid_hi - grid_lo
    weight = np.where(span > 0, (quantiles - grid_lo) / span, 0.0)
    weight = np.clip(weight, 0.0, 1.0)  # clamp q outside the grid to the ends
    rows = np.arange(len(quantiles))
    values_lo = predictions[rows, lower]
    values_hi = predictions[rows, upper]
    return values_lo + weight * (values_hi - values_lo)


@dataclass(frozen=True)
class _Forest:
    """A fitted quantile forest plus the feature columns it was fit on."""

    model: RandomForestQuantileRegressor
    columns: tuple[str, ...]

    def draw(self, frame: pd.DataFrame, quantiles: np.ndarray) -> np.ndarray:
        """Draw one value per row at that row's quantile.

        The forest is queried on a shared fine grid of quantiles, then each
        row's value is read out by **linearly interpolating** its predicted
        grid values at its exact quantile — not by snapping to the nearest grid
        point. Snapping quantizes every draw to one of the grid's quantiles,
        which flattens the conditional and biases tail draws toward the
        grid-bracket interior; interpolation reads the true per-row quantile.

        The grid includes points adjacent to 0 and 1 (see :data:`_QUANTILE_GRID`),
        so the observed conditional min and max are drawable: ``q=1`` is the
        observed maximum, not extrapolation, and a draw with quantile near 1
        must be able to reach it.

        The predict is chunked over rows (:data:`_PREDICT_CHUNK_ROWS`) so the
        ``(n_rows x n_grid)`` prediction matrix never has to materialize whole —
        at 3M+ rows that matrix alone would be tens of GB.

        Args:
            frame: Feature rows (must carry the fitted columns).
            quantiles: One quantile in ``[0, 1]`` per row.

        Returns:
            One drawn value per row, positionally aligned with ``frame``.
        """
        features = frame.loc[:, list(self.columns)].to_numpy(dtype=np.float64)
        quantiles = np.asarray(quantiles, dtype=np.float64)
        grid = _QUANTILE_GRID
        n = len(features)
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, _PREDICT_CHUNK_ROWS):
            stop = min(start + _PREDICT_CHUNK_ROWS, n)
            block = features[start:stop]
            predictions = np.asarray(
                self.model.predict(block, quantiles=list(grid))
            ).reshape(len(block), len(grid))
            out[start:stop] = _interp_rows(quantiles[start:stop], grid, predictions)
        return out


#: Fine symmetric quantile grid used to read per-row draws. The interior is an
#: evenly spaced grid over ``(0, 1)``; points adjacent to 0 and 1 are prepended
#: and appended so the *observed* conditional extremes are drawable. The maximum
#: is the ``q=1`` order statistic (and the minimum the ``q=0`` one), which a
#: forest can return exactly — it is reading an observed value, not
#: extrapolating past it — so excluding the endpoints (as the nearest-snap grid
#: did) needlessly truncates the tails. ``np.interp`` then maps any per-row
#: quantile, including ones at or beyond the grid ends, onto these values.
_GRID_EPS = 1e-6
_QUANTILE_GRID = np.concatenate(
    [
        [_GRID_EPS],
        np.linspace(1.0 / 202.0, 1.0 - 1.0 / 202.0, 201),
        [1.0 - _GRID_EPS],
    ]
)

#: Row-batch size for the draw predict. Bounds the ``(rows x grid)`` matrix so a
#: 3M+ row draw streams in fixed-memory blocks instead of allocating the whole
#: matrix at once.
_PREDICT_CHUNK_ROWS = 50_000


def _fit_forest(
    x: np.ndarray,
    y: np.ndarray,
    columns: tuple[str, ...],
    weights: np.ndarray | None,
    *,
    seed: int,
    n_estimators: int,
    max_samples_leaf: int | float | None,
    rng: np.random.Generator,
) -> _Forest:
    """Weighted-bootstrap the rows, then grow a quantile forest on them.

    ``max_samples_leaf`` is passed through to the forest: the quantile-forest
    default of ``1`` keeps only one sample per leaf, which thins each row's
    conditional to ~``n_estimators`` atoms and undershoots tail mass; ``None``
    keeps every leaf sample, so the conditional reflects the full leaf
    population.
    """
    x_fit, y_fit = _weighted_bootstrap(x, y, weights, rng)
    model = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        max_samples_leaf=max_samples_leaf,
        random_state=seed,
    )
    model.fit(x_fit, y_fit)
    return _Forest(model=model, columns=columns)


@dataclass(frozen=True)
class _TargetModel:
    """The fitted pipeline for one numeric target: its regime, gate, forests.

    Attributes:
        regime: The detected :class:`Regime` label.
        columns: The predictor columns this target was fit on (predictors plus
            the targets chained before it).
        gate: The sign-gate classifier, or ``None`` for ungated regimes.
        positive: The positive-magnitude forest, or ``None``.
        negative: The negative-magnitude forest, or ``None``.
    """

    regime: str
    columns: tuple[str, ...]
    gate: HistGradientBoostingClassifier | None
    positive: _Forest | None
    negative: _Forest | None


class RegimeGatedQRF:
    """The canonical :class:`~populace.fit.model.ConditionalModel`.

    Fits a regime-gated, sequentially-chained quantile-regression-forest model
    of ``P(targets | predictors)`` over a :class:`~populace.frame.Frame`, with
    the frame's typed weights materialized by weighted bootstrap. See the module
    docstring for the three mechanisms (weighted bootstrap, regime gates,
    chaining).

    Args:
        n_estimators: Trees per forest.
        zero_atol: Magnitudes at or below this count as zeros in regime
            detection.
        max_samples_leaf: Samples retained per forest leaf for the conditional.
            ``None`` (the default here) keeps **all** leaf samples, so the
            per-row conditional reflects the full leaf population; the
            quantile-forest default of ``1`` keeps only one sample per leaf,
            thinning each row's conditional to ~``n_estimators`` atoms and
            undershooting tail mass (roughly halving the share above a high
            threshold). Pass an int/float to cap the leaf sample, matching the
            quantile-forest semantics.
        seed: Base random seed. Controls the weighted-bootstrap resample, the
            forest randomness, and the per-row draw quantiles, so a fixed seed
            makes a freshly fitted model's first draw reproducible.
    """

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        zero_atol: float = DEFAULT_ZERO_ATOL,
        max_samples_leaf: int | float | None = None,
        seed: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.zero_atol = float(zero_atol)
        self.max_samples_leaf = max_samples_leaf
        self.seed = int(seed)

    def fit(
        self,
        frame: Frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> FittedRegimeGatedQRF:
        """Fit the conditional model. See
        :meth:`~populace.fit.model.ConditionalModel.fit`.

        Resolves the single entity owning the predictors and targets, reads its
        typed weights per ``weights`` (``"none"`` is the only unweighted path),
        detects each target's regime structurally, and grows the gated,
        weighted-bootstrap forests in chain order.

        Raises:
            ValueError: If predictors/targets are empty, span more than one
                entity, name unknown columns, request a weight kind the
                entity's resolved weights are not, or a target column contains
                non-finite (NaN/inf) values. Messages name the culprits.
        """
        predictors = list(predictors)
        targets = list(targets)
        entity = predictors_targets_entity(frame, predictors, targets)
        weight_values = resolve_fit_weights(frame, entity, weights)
        table = frame.table(entity)
        _validate_targets_finite(table, targets)

        # Split the model seed into two independent streams: one drives the
        # fit (bootstrap resample, forest randomness, gate random_state), the
        # other drives the predict-time draw quantiles. Seeding both from the
        # same seed (as before) made the draw uniforms bit-identical to the
        # gate's bootstrap-selection uniforms — the draws were not independent
        # of the fit's resampling. SeedSequence.spawn keeps both reproducible
        # from the one model seed, so determinism is preserved.
        fit_seed, draw_seed = np.random.SeedSequence(self.seed).spawn(2)
        rng = np.random.default_rng(fit_seed)

        target_models: dict[str, _TargetModel] = {}
        for position, target in enumerate(targets):
            chained = (*predictors, *targets[:position])
            y = table[target].to_numpy(dtype=np.float64)
            features = table.loc[:, list(chained)].to_numpy(dtype=np.float64)
            target_models[target] = self._fit_target(
                features=features,
                y=y,
                columns=chained,
                weights=weight_values,
                rng=rng,
            )

        return FittedRegimeGatedQRF(
            entity=entity,
            predictors=predictors,
            targets=targets,
            target_models=target_models,
            zero_atol=self.zero_atol,
            draw_seed=draw_seed,
        )

    def _fit_target(
        self,
        *,
        features: np.ndarray,
        y: np.ndarray,
        columns: tuple[str, ...],
        weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> _TargetModel:
        """Fit the gate and per-sign forests for one numeric target."""
        regime = detect_regime(y, zero_atol=self.zero_atol)

        def forest(mask: np.ndarray) -> _Forest:
            sub_weights = None if weights is None else weights[mask]
            return _fit_forest(
                features[mask],
                y[mask],
                columns,
                sub_weights,
                seed=int(rng.integers(0, 2**31 - 1)),
                n_estimators=self.n_estimators,
                max_samples_leaf=self.max_samples_leaf,
                rng=rng,
            )

        if regime == Regime.DEGENERATE_ZERO:
            return _TargetModel(regime, columns, None, None, None)

        if regime in (Regime.POSITIVE_ONLY, Regime.NEGATIVE_ONLY):
            single = forest(np.ones(len(y), dtype=bool))
            return _TargetModel(
                regime,
                columns,
                None,
                single if regime == Regime.POSITIVE_ONLY else None,
                single if regime == Regime.NEGATIVE_ONLY else None,
            )

        # Gated regimes: a sign label per row. The gate is weighted directly by
        # sample_weight (not by bootstrap), so every sign class survives even
        # when one is vanishingly rare under the weights.
        labels = self._sign_labels(y)
        gate = self._fit_gate(features, labels, weights, rng)
        pos_mask = y > self.zero_atol
        neg_mask = y < -self.zero_atol
        positive = forest(pos_mask) if pos_mask.any() else None
        negative = forest(neg_mask) if neg_mask.any() else None
        return _TargetModel(regime, columns, gate, positive, negative)

    def _sign_labels(self, y: np.ndarray) -> np.ndarray:
        """Per-row sign code: ``-1`` negative, ``0`` zero, ``1`` positive."""
        labels = np.zeros(len(y), dtype=int)
        labels[y > self.zero_atol] = 1
        labels[y < -self.zero_atol] = -1
        return labels

    def _fit_gate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> HistGradientBoostingClassifier:
        """Fit the sign-gate classifier, weighting it directly by sample_weight.

        Unlike the forests, the gate is *not* weighted by bootstrap.
        ``HistGradientBoostingClassifier`` honors ``sample_weight`` exactly, so
        passing the weights directly weights the gate's class probabilities
        without resampling. An n-of-n weighted bootstrap would instead delete
        rare low-weight classes entirely — a single positive row at weight 1
        among thousands of zeros at weight 50 is drawn with probability ~4e-5,
        so the resampled labels routinely contain only the zero class and the
        gate could never draw the positive sign (the reproduced gate bug). With
        ``sample_weight`` every training row is present, so every sign class the
        data contains survives into ``classes_``.

        A guard then enforces internal consistency: every sign class present in
        the (unweighted) training labels must appear in the fitted gate's
        ``classes_``. If sklearn ever dropped a class, drawing it at probability
        zero would silently lose that sign, so we raise instead.

        Args:
            features: The chained predictor matrix for this target's rows.
            labels: Per-row sign codes (``-1`` / ``0`` / ``1``).
            weights: Per-row weights, or ``None`` for an unweighted gate.
            rng: Seeded generator supplying the gate's ``random_state`` (kept
                for fit reproducibility even though no resample is drawn).

        Returns:
            The fitted sign-gate classifier.

        Raises:
            ValueError: If a sign class present in ``labels`` is absent from the
                fitted gate's ``classes_`` (internal inconsistency).
        """
        gate = _make_gate(int(rng.integers(0, 2**31 - 1)))
        gate.fit(features, labels, sample_weight=weights)
        training_classes = set(np.unique(labels).tolist())
        fitted_classes = set(np.asarray(gate.classes_).tolist())
        missing = sorted(training_classes - fitted_classes)
        if missing:
            raise ValueError(
                "Sign gate dropped class(es) present in training: "
                f"{missing} are in the training sign labels "
                f"{sorted(training_classes)} but absent from the fitted gate's "
                f"classes_ {sorted(fitted_classes)}. Drawing a missing class at "
                "probability zero would silently lose that sign; refusing to "
                "fit an inconsistent gate."
            )
        return gate


class FittedRegimeGatedQRF:
    """A fitted :class:`RegimeGatedQRF`, ready to draw.

    Holds the per-target gated forests and draws by, for each target in chain
    order: gating each row to a sign class, drawing a magnitude from that
    class's forest at a per-row quantile, and carrying the drawn column forward
    as a predictor for later targets. The draw RNG is seeded from an
    independent ``SeedSequence`` child of the model seed (separate from the
    fit's resampling stream), so a freshly fitted model reproduces its first
    :meth:`predict`, while successive calls on one fitted model advance the
    state and give independent draws.

    Attributes:
        entity: The entity the predictors and targets live on.
        predictors: The conditioning columns.
        targets: The drawn columns, in chain order.
    """

    def __init__(
        self,
        *,
        entity: str,
        predictors: list[str],
        targets: list[str],
        target_models: dict[str, _TargetModel],
        zero_atol: float,
        draw_seed: np.random.SeedSequence,
    ) -> None:
        self.entity = entity
        self.predictors = list(predictors)
        self.targets = list(targets)
        self._target_models = target_models
        self._zero_atol = zero_atol
        self._rng = np.random.default_rng(draw_seed)

    def regimes(self) -> dict[str, str]:
        """The detected :class:`Regime` label per target."""
        return {name: model.regime for name, model in self._target_models.items()}

    def predict(self, frame_or_df: Frame | pd.DataFrame) -> pd.DataFrame:
        """Draw imputed targets. See
        :meth:`~populace.fit.model.FittedModel.predict`.

        Raises:
            ValueError: If a required predictor column is absent from the input.
        """
        features = self._predictor_frame(frame_or_df)
        out = pd.DataFrame(index=features.index)
        # Accumulate drawn targets so each later target can condition on them
        # (chained-equations imputation), mirroring the fit-time chain order.
        augmented = features.copy()
        for target in self.targets:
            drawn = self._draw_target(augmented, self._target_models[target])
            out[target] = drawn
            augmented[target] = np.asarray(drawn, dtype=np.float64)
        return out

    def _predictor_frame(self, frame_or_df: Frame | pd.DataFrame) -> pd.DataFrame:
        """Extract the predictor columns from a Frame or DataFrame input."""
        if isinstance(frame_or_df, Frame):
            table = frame_or_df.table(self.entity)
        else:
            table = frame_or_df
        missing = [c for c in self.predictors if c not in table.columns]
        if missing:
            kind = "frame" if isinstance(frame_or_df, Frame) else "DataFrame"
            raise ValueError(
                f"predict input ({kind}) is missing predictor column(s) "
                f"{missing}; the model conditions on {self.predictors}."
            )
        return table.loc[:, self.predictors].copy()

    def _draw_target(
        self, features: pd.DataFrame, model: _TargetModel
    ) -> np.ndarray:
        """Draw one value per row for a single target via its regime pipeline."""
        n = len(features)
        if model.regime == Regime.DEGENERATE_ZERO:
            return np.zeros(n, dtype=np.float64)

        quantiles = self._rng.random(n)
        if model.regime == Regime.POSITIVE_ONLY:
            return model.positive.draw(features, quantiles)
        if model.regime == Regime.NEGATIVE_ONLY:
            return model.negative.draw(features, quantiles)

        signs = self._gate_draw(model.gate, features, model.columns)
        values = np.zeros(n, dtype=np.float64)
        pos_mask = signs == 1
        neg_mask = signs == -1
        if pos_mask.any() and model.positive is not None:
            values[pos_mask] = model.positive.draw(
                features.loc[pos_mask], quantiles[pos_mask]
            )
        if neg_mask.any() and model.negative is not None:
            values[neg_mask] = model.negative.draw(
                features.loc[neg_mask], quantiles[neg_mask]
            )
        return values

    def _gate_draw(
        self,
        gate: HistGradientBoostingClassifier,
        features: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> np.ndarray:
        """Stochastically assign each row a sign code from the gate's proba.

        Draws the row's sign class from the categorical the gate predicts (not
        the argmax), so the share of rows assigned to each sign reproduces the
        gate's probabilities — that is what preserves the zero mass of a
        zero-inflated target rather than collapsing it to the modal class.

        Args:
            gate: The fitted sign-gate classifier.
            features: The chained predictor frame for this target's rows.
            columns: The exact column order the gate was fit on.

        Returns:
            One sign code (``-1`` / ``0`` / ``1``) per row.
        """
        x = features.loc[:, list(columns)].to_numpy(dtype=np.float64)
        proba = np.asarray(gate.predict_proba(x))
        cumulative = np.cumsum(proba, axis=1)
        u = self._rng.random(len(x))
        chosen = (cumulative >= u[:, None]).argmax(axis=1)
        return np.asarray(gate.classes_)[chosen]

    def __repr__(self) -> str:
        regimes = ", ".join(f"{k}:{v}" for k, v in self.regimes().items())
        return (
            f"FittedRegimeGatedQRF(entity={self.entity!r}, "
            f"predictors={self.predictors}, regimes[{regimes}])"
        )
