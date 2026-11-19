"""A weighted classifier for one categorical target, drawn at caller uniforms.

The quantile forest interpolates between donor values, so it cannot impute a
code such as an occupation group or an industry: a draw could fall between two
codes. :class:`CategoricalClassifier` learns ``P(target | predictors)`` for a
single categorical target with a regularised histogram gradient-boosted
classifier, a multinomial logit on spline features, or a weighted average of
the two, and draws one category per row by inverse CDF at a uniform the caller
supplies. A build passes uniforms keyed on stable record identities, so a
row's draw does not depend on which other rows are drawn with it.

The two learners err in different places. The boosted classifier is sharper
where interactions matter; the logit holds its class shares across margins it
sees only additively and extrapolates more smoothly. Their average is often
better than either, so ``boosted_weight`` sets the mixture: 1 fits only the
boosted classifier, 0 only the logit.

The fit follows the package's weight contract (:mod:`microcosm.fit.model`): a
Frame fit reads the owning entity's typed weights, a DataFrame fit needs them
stated, and ``weights="none"`` is the only unweighted path.

A category the fit never saw is refused at prediction by default. A model that
predicts for another survey's rows can choose ``unseen_categories="ignore"``
instead: such a category then sets none of its predictor's indicators, as
scikit-learn's ``handle_unknown="ignore"`` encodes it, and
:meth:`FittedCategoricalClassifier.unseen_category_rows` says which rows held
one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from microcosm.fit.model import (
    DESIGN_WEIGHTS,
    WeightSpec,
    dataframe_fit_columns,
    predictors_targets_entity,
    resolve_dataframe_fit_weights,
    resolve_fit_weights,
    resolved_weight_kind,
    validated_integer,
)
from microcosm.frame import Frame

__all__ = [
    "BoostedClassifierSettings",
    "CategoricalClassifier",
    "FittedCategoricalClassifier",
    "SplineLogitSettings",
    "inverse_cdf_draw",
]

#: Added to a feature's standard deviation so a constant column scales to zero.
_SCALE_FLOOR = 1e-9
#: How a category the fit never saw is read at prediction.
_UNSEEN_CATEGORY_RULES = ("refuse", "ignore")


@dataclass(frozen=True)
class BoostedClassifierSettings:
    """Settings of the histogram gradient-boosted classifier (scikit-learn's)."""

    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    early_stopping: bool = True
    validation_fraction: float = 0.2
    n_iter_no_change: int = 20
    max_iter: int = 500


@dataclass(frozen=True)
class SplineLogitSettings:
    """Settings of the multinomial logit on spline features.

    Each numeric predictor with at least ``n_knots`` distinct training values
    is expanded in a B-spline basis (scikit-learn's ``SplineTransformer``); a
    predictor with fewer enters as itself. Features are standardised on the
    training rows before the weighted L2-penalised fit.
    """

    n_knots: int = 6
    degree: int = 3
    knots: str = "quantile"
    extrapolation: str = "constant"
    penalty_c: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 3000


def inverse_cdf_draw(proba: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """The column index each row's uniform falls in, by inverse CDF.

    The cumulative sum's last column is forced to 1, so rounding can never
    leave a uniform below 1 without a category.

    Args:
        proba: Class probabilities, one row per draw, each row summing to 1.
        uniforms: One uniform in ``[0, 1)`` per row.

    Returns:
        An int64 array of column indices.

    Raises:
        ValueError: If the shapes disagree, a probability is negative or
            non-finite, a row does not sum to 1, or a uniform is outside
            ``[0, 1)``.
    """
    proba = np.asarray(proba, dtype=np.float64)
    u = np.asarray(uniforms, dtype=np.float64)
    if proba.ndim != 2 or u.shape != (proba.shape[0],):
        raise ValueError("inverse_cdf_draw needs one uniform per probability row.")
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("Class probabilities must be finite and non-negative.")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Each row of class probabilities must sum to 1.")
    if not np.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
        raise ValueError("Draw uniforms must be in [0, 1).")
    cumulative = np.cumsum(proba, axis=1)
    cumulative[:, -1] = 1.0
    return (cumulative > u[:, None]).argmax(axis=1).astype(np.int64)


@dataclass(frozen=True)
class CategoricalClassifier:
    """A :class:`~microcosm.fit.model.ConditionalModel` for one categorical target.

    Attributes:
        boosted_weight: The boosted classifier's share of the mixture, in
            ``[0, 1]``; the logit has the rest. 1 and 0 fit one learner only.
        boosted: The boosted classifier's settings.
        logit: The spline logit's settings.
        categorical_predictors: Predictors entered as categories (one-hot on
            their training levels); every other predictor is numeric.
        arcsinh_scales: Numeric predictor -> scale. The logit sees
            ``arcsinh(x / scale)`` for these before the spline basis, which
            tames long-tailed amounts such as incomes. The boosted classifier,
            which splits on ranks, always sees the raw values.
        seed: Seeds the boosted classifier and the :meth:`predict` draws: an
            integer ``>= 0``. The constructor refuses a bool or a float such as
            ``1.9``, since converting it would alias another seed.
        unseen_categories: ``"refuse"`` (the default) raises when a row to
            predict holds a category the fit never saw; ``"ignore"`` sets none
            of that predictor's indicators for it.
    """

    boosted_weight: float = 0.5
    boosted: BoostedClassifierSettings = BoostedClassifierSettings()
    logit: SplineLogitSettings = SplineLogitSettings()
    categorical_predictors: tuple[str, ...] = ()
    arcsinh_scales: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0
    unseen_categories: str = "refuse"

    def __post_init__(self) -> None:
        weight = float(self.boosted_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("boosted_weight must be in [0, 1].")
        object.__setattr__(
            self, "seed", validated_integer(self.seed, name="seed", minimum=0)
        )
        if self.unseen_categories not in _UNSEEN_CATEGORY_RULES:
            raise ValueError(
                f"unseen_categories must be one of {list(_UNSEEN_CATEGORY_RULES)}."
            )
        for column, scale in self.arcsinh_scales.items():
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"arcsinh scale for {column!r} must be positive.")
        object.__setattr__(
            self, "categorical_predictors", tuple(self.categorical_predictors)
        )
        object.__setattr__(
            self, "arcsinh_scales", MappingProxyType(dict(self.arcsinh_scales))
        )

    def fit(
        self,
        frame_or_df: Frame | pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> FittedCategoricalClassifier:
        """Fit ``P(target | predictors)`` for the single categorical target.

        Raises:
            ValueError: If there is not exactly one target, a categorical or
                arcsinh predictor is not a predictor, the target has a missing
                label, a numeric predictor is non-finite, the weights cannot be
                resolved, or the logit does not converge.
        """
        predictors = list(predictors)
        targets = list(targets)
        if len(targets) != 1:
            raise ValueError("CategoricalClassifier fits exactly one target.")
        unknown = [c for c in self.categorical_predictors if c not in predictors]
        unknown += [c for c in self.arcsinh_scales if c not in predictors]
        if unknown:
            raise ValueError(
                f"Settings name columns that are not predictors: {unknown}."
            )
        if set(self.arcsinh_scales) & set(self.categorical_predictors):
            raise ValueError("A categorical predictor cannot take an arcsinh scale.")

        if isinstance(frame_or_df, Frame):
            entity = predictors_targets_entity(frame_or_df, predictors, targets)
            weight_values = resolve_fit_weights(frame_or_df, entity, weights)
            table = frame_or_df.table(entity)
        else:
            entity = None
            dataframe_fit_columns(frame_or_df, predictors, targets)
            weight_values = resolve_dataframe_fit_weights(
                frame_or_df, weights, predictors=predictors, targets=targets
            )
            table = frame_or_df
        weight_kind = resolved_weight_kind(frame_or_df, entity, weight_values)
        labels = table[targets[0]]
        if labels.isna().any():
            raise ValueError(f"Target {targets[0]!r} has rows with no category.")
        y = labels.to_numpy()
        w = (
            np.ones(len(table))
            if weight_values is None
            else np.asarray(weight_values, dtype=np.float64)
        )
        features = _Features.fit(
            table, predictors, self.categorical_predictors, self.unseen_categories
        )
        classes = np.unique(y)
        fitted = FittedCategoricalClassifier(
            entity=entity,
            predictors=predictors,
            targets=targets,
            classes=classes,
            features=features,
            boosted_weight=float(self.boosted_weight),
            weight_kind=weight_kind,
            seed=self.seed,
        )
        if len(classes) == 1:
            return fitted
        if self.boosted_weight > 0:
            fitted.boosted_model = self._fit_boosted(features, table, y, w)
        if self.boosted_weight < 1:
            fitted.logit_model = _SplineLogit.fit(
                features, table, y, w, self.logit, self.arcsinh_scales
            )
        return fitted

    def _fit_boosted(
        self, features: _Features, table: pd.DataFrame, y: np.ndarray, w: np.ndarray
    ) -> HistGradientBoostingClassifier:
        settings = self.boosted
        random_state = int(
            np.random.SeedSequence(self.seed).generate_state(1, dtype=np.uint32)[0]
        )
        model = HistGradientBoostingClassifier(
            learning_rate=settings.learning_rate,
            max_leaf_nodes=settings.max_leaf_nodes,
            min_samples_leaf=settings.min_samples_leaf,
            l2_regularization=settings.l2_regularization,
            early_stopping=settings.early_stopping,
            validation_fraction=settings.validation_fraction,
            n_iter_no_change=settings.n_iter_no_change,
            max_iter=settings.max_iter,
            random_state=random_state,
        )
        return model.fit(features.boosted_matrix(table), y, sample_weight=w)


@dataclass
class _Features:
    """The numeric and one-hot categorical columns, with the training levels."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    levels: Mapping[str, np.ndarray]
    unseen: str = "refuse"

    @classmethod
    def fit(
        cls,
        table: pd.DataFrame,
        predictors: list[str],
        categorical: Sequence[str],
        unseen: str = "refuse",
    ) -> _Features:
        numeric = tuple(c for c in predictors if c not in categorical)
        features = cls(
            numeric=numeric,
            categorical=tuple(c for c in predictors if c in categorical),
            levels={},
            unseen=unseen,
        )
        features.numeric_values(table)
        features.levels = {
            column: np.unique(table[column].to_numpy())
            for column in features.categorical
        }
        return features

    def numeric_values(self, table: pd.DataFrame) -> np.ndarray:
        values = table.loc[:, list(self.numeric)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad = [
                column
                for column, ok in zip(
                    self.numeric, np.isfinite(values).all(axis=0), strict=True
                )
                if not ok
            ]
            raise ValueError(f"Numeric predictors have non-finite values: {bad}.")
        return values

    def _indicators(self, table: pd.DataFrame, column: str) -> np.ndarray:
        values = table[column].to_numpy()
        return values[:, None] == self.levels[column][None, :]

    def unseen_rows(self, table: pd.DataFrame) -> np.ndarray:
        rows = np.zeros(len(table), dtype=bool)
        for column in self.categorical:
            rows |= ~self._indicators(table, column).any(axis=1)
        return rows

    def one_hot(self, table: pd.DataFrame) -> np.ndarray:
        parts = []
        for column in self.categorical:
            indicators = self._indicators(table, column)
            seen = indicators.any(axis=1)
            if self.unseen == "refuse" and not seen.all():
                unseen = sorted({str(v) for v in table[column].to_numpy()[~seen]})
                raise ValueError(
                    f"Predictor {column!r} has categories the fit never saw: {unseen}."
                )
            parts.append(indicators.astype(np.float64))
        if not parts:
            return np.zeros((len(table), 0))
        return np.hstack(parts)

    def boosted_matrix(self, table: pd.DataFrame) -> np.ndarray:
        return np.hstack([self.numeric_values(table), self.one_hot(table)])


@dataclass
class _SplineLogit:
    """The fitted spline basis, scaling and multinomial logit."""

    transformers: Mapping[str, SplineTransformer | None]
    arcsinh_scales: Mapping[str, float]
    mean: np.ndarray
    scale: np.ndarray
    model: LogisticRegression

    @classmethod
    def fit(
        cls,
        features: _Features,
        table: pd.DataFrame,
        y: np.ndarray,
        w: np.ndarray,
        settings: SplineLogitSettings,
        arcsinh_scales: Mapping[str, float],
    ) -> _SplineLogit:
        transformed = cls._transformed(features, table, arcsinh_scales)
        transformers: dict[str, SplineTransformer | None] = {}
        for index, column in enumerate(features.numeric):
            values = transformed[:, [index]]
            if len(np.unique(values)) < settings.n_knots:
                transformers[column] = None
                continue
            transformers[column] = SplineTransformer(
                n_knots=settings.n_knots,
                degree=settings.degree,
                knots=settings.knots,
                extrapolation=settings.extrapolation,
            ).fit(values)
        logit = cls(
            transformers=transformers,
            arcsinh_scales=arcsinh_scales,
            mean=np.zeros(0),
            scale=np.ones(0),
            model=LogisticRegression(
                C=settings.penalty_c, solver=settings.solver, max_iter=settings.max_iter
            ),
        )
        raw = logit._expanded(features, table)
        logit.mean = raw.mean(axis=0)
        logit.scale = raw.std(axis=0) + _SCALE_FLOOR
        logit.model.fit((raw - logit.mean) / logit.scale, y, sample_weight=w / w.mean())
        if int(np.max(logit.model.n_iter_)) >= settings.max_iter:
            raise ValueError(
                f"The spline logit did not converge in {settings.max_iter} "
                "iterations; raise max_iter or check the predictors."
            )
        return logit

    @staticmethod
    def _transformed(
        features: _Features, table: pd.DataFrame, arcsinh_scales: Mapping[str, float]
    ) -> np.ndarray:
        values = features.numeric_values(table).copy()
        for index, column in enumerate(features.numeric):
            if column in arcsinh_scales:
                values[:, index] = np.arcsinh(values[:, index] / arcsinh_scales[column])
        return values

    def _expanded(self, features: _Features, table: pd.DataFrame) -> np.ndarray:
        transformed = self._transformed(features, table, self.arcsinh_scales)
        parts = []
        for index, column in enumerate(features.numeric):
            transformer = self.transformers[column]
            values = transformed[:, [index]]
            parts.append(
                values if transformer is None else transformer.transform(values)
            )
        parts.append(features.one_hot(table))
        return np.hstack(parts)

    def predict_proba(self, features: _Features, table: pd.DataFrame) -> np.ndarray:
        raw = self._expanded(features, table)
        return self.model.predict_proba((raw - self.mean) / self.scale)

    @property
    def iterations(self) -> int:
        return int(np.max(self.model.n_iter_))


class FittedCategoricalClassifier:
    """A fitted :class:`CategoricalClassifier`, ready to give probabilities and draws.

    Attributes:
        entity: The entity the predictors and target live on, or ``None`` for
            a DataFrame fit.
        predictors: The conditioning columns.
        targets: The one categorical target, as a one-element list.
        classes_: The target's categories, sorted; probability columns follow
            this order.
        weight_kind: The weight kind the fit resolved to, for the weights
            audit (:func:`~microcosm.fit.model.resolved_weight_kind`).
    """

    def __init__(
        self,
        *,
        entity: str | None,
        predictors: list[str],
        targets: list[str],
        classes: np.ndarray,
        features: _Features,
        boosted_weight: float,
        weight_kind: str,
        seed: int,
    ) -> None:
        self.entity = entity
        self.predictors = list(predictors)
        self.targets = list(targets)
        self.classes_ = classes
        self.weight_kind = weight_kind
        self.boosted_model: HistGradientBoostingClassifier | None = None
        self.logit_model: _SplineLogit | None = None
        self._features = features
        self._boosted_weight = boosted_weight
        self._draw_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])

    def _table(self, frame_or_df: Frame | pd.DataFrame) -> pd.DataFrame:
        if isinstance(frame_or_df, Frame):
            if self.entity is None:
                raise ValueError("A DataFrame fit predicts on DataFrames only.")
            return frame_or_df.table(self.entity)
        missing = [c for c in self.predictors if c not in frame_or_df.columns]
        if missing:
            raise ValueError(f"Rows to predict for lack predictors {missing}.")
        return frame_or_df

    def _full(self, model_classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
        out = np.zeros((len(proba), len(self.classes_)))
        out[:, np.searchsorted(self.classes_, model_classes)] = proba
        return out

    def component_proba(
        self, frame_or_df: Frame | pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Each fitted learner's class probabilities, by learner name."""
        table = self._table(frame_or_df)
        out: dict[str, np.ndarray] = {}
        if self.boosted_model is not None:
            out["boosted"] = self._full(
                self.boosted_model.classes_,
                self.boosted_model.predict_proba(self._features.boosted_matrix(table)),
            )
        if self.logit_model is not None:
            out["spline_logit"] = self._full(
                self.logit_model.model.classes_,
                self.logit_model.predict_proba(self._features, table),
            )
        return out

    def predict_proba(self, frame_or_df: Frame | pd.DataFrame) -> np.ndarray:
        """Class probabilities, one column per ``classes_`` entry."""
        table = self._table(frame_or_df)
        if len(self.classes_) == 1:
            self._features.boosted_matrix(table)  # the same input checks
            return np.ones((len(table), 1))
        components = self.component_proba(table)
        if "spline_logit" not in components:
            return components["boosted"]
        if "boosted" not in components:
            return components["spline_logit"]
        weight = self._boosted_weight
        return (
            weight * components["boosted"] + (1.0 - weight) * components["spline_logit"]
        )

    def unseen_category_rows(self, frame_or_df: Frame | pd.DataFrame) -> np.ndarray:
        """Rows holding a category of some categorical predictor the fit never saw."""
        return self._features.unseen_rows(self._table(frame_or_df))

    def predict_from_uniforms(
        self, frame_or_df: Frame | pd.DataFrame, uniforms: np.ndarray
    ) -> pd.DataFrame:
        """Draw each row's category at its own uniform, by inverse CDF."""
        table = self._table(frame_or_df)
        drawn = self.classes_[inverse_cdf_draw(self.predict_proba(table), uniforms)]
        return pd.DataFrame({self.targets[0]: drawn}, index=table.index)

    def predict(self, frame_or_df: Frame | pd.DataFrame) -> pd.DataFrame:
        """One stochastic draw per row from the model's seeded draw stream."""
        table = self._table(frame_or_df)
        return self.predict_from_uniforms(table, self._draw_rng.random(len(table)))

    @property
    def diagnostics(self) -> dict[str, int | None]:
        """The learners' iteration counts (``None`` for a learner not fitted)."""
        return {
            "boosted_iterations": (
                None if self.boosted_model is None else int(self.boosted_model.n_iter_)
            ),
            "logit_iterations": (
                None if self.logit_model is None else self.logit_model.iterations
            ),
        }
