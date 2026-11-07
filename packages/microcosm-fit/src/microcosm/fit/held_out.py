"""Held-out evidence for a categorical model.

A stage that draws categories can report how its model predicts rows it did not
train on:
- the out-of-fold log loss, beside a constant's (the training folds' weighted
  class shares);
- the calibration slope, the one-parameter Platt map of the out-of-fold log
  probabilities: 1 when the probabilities spread as much as the outcomes do,
  above 1 when they are too flat, below 1 when they are too sharp;
- how far the predicted class shares sit from the observed ones, overall and
  within the strata the caller names.

Folds deal whole groups (a household's members, say) when ``groups`` is given,
so a group never sits on both sides of a fold. The folds and the slope reuse the
sign-gate calibration's machinery in :mod:`microcosm.fit.qrf`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from microcosm.fit import qrf as _qrf
from microcosm.fit.categorical import CategoricalClassifier

__all__ = [
    "can_deal_folds",
    "categorical_scores",
    "out_of_fold_proba",
    "stratified_folds",
]

#: The probability a log loss floors a true class at, so one confident miss
#: cannot make the loss infinite.
LOG_LOSS_FLOOR = 1e-4

#: How far from 1 a row of probabilities may sum for the scorer to accept it.
PROBABILITY_SUM_TOLERANCE = 1e-6


def _group_codes(
    groups: Sequence[object] | np.ndarray | None, n: int
) -> np.ndarray | None:
    if groups is None:
        return None
    values = pd.Series(np.asarray(groups))
    if len(values) != n:
        raise ValueError(f"groups has {len(values)} values for {n} rows.")
    if values.isna().any():
        raise ValueError("groups must name a group for every row.")
    return pd.factorize(values, sort=True)[0].astype(np.int64)


def can_deal_folds(
    labels: Sequence[object] | np.ndarray,
    folds: int,
    groups: Sequence[object] | np.ndarray | None = None,
) -> bool:
    """Whether every class has at least ``folds`` rows (or groups) to deal."""
    labels = np.asarray(labels)
    codes = _group_codes(groups, len(labels))
    frame = pd.DataFrame({"label": labels})
    if codes is None:
        counts = frame.groupby("label").size()
    else:
        counts = frame.assign(group=codes).drop_duplicates().groupby("label").size()
    return bool(len(counts)) and int(counts.min()) >= folds


def stratified_folds(
    labels: Sequence[object] | np.ndarray,
    folds: int,
    rng: np.random.Generator,
    groups: Sequence[object] | np.ndarray | None = None,
) -> np.ndarray:
    """A fold per row, each class's rows (or groups) dealt round the folds.

    With ``groups`` whole groups are dealt, each with the rarest class it
    contains, so rows that share a group are never split across folds.
    """
    labels = np.asarray(labels)
    return _qrf._stratified_folds(
        labels, folds, rng, groups=_group_codes(groups, len(labels))
    )


def out_of_fold_proba(
    model: CategoricalClassifier,
    table: pd.DataFrame,
    predictors: list[str],
    target: str,
    *,
    weights: np.ndarray,
    folds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Out-of-fold class probabilities for every row of ``table``.

    Each fold's rows are scored by the model fitted on the other folds. The
    result maps ``"model"`` to the model's probabilities, each fitted learner's
    name (``"boosted"``, ``"spline_logit"``) to its own, and ``"marginal"`` to
    the training folds' weighted class shares; ``"classes"`` holds the column
    order, the sorted categories of ``table[target]``.
    """
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


def _share_gap(proba: np.ndarray, observed: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    return float(0.5 * np.abs((w @ proba - w @ observed) / total).sum())


def _scored_inputs(
    proba: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validated ``(proba, class_positions, weights)`` for :func:`categorical_scores`.

    Each label maps to its column by the supplied class order, so the classes
    may come in any order. A sorted search would misplace labels among
    unsorted classes.
    """
    proba = np.asarray(proba, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    w = np.asarray(weights, dtype=np.float64)
    if classes.ndim != 1 or not len(classes):
        raise ValueError("classes must list at least one category.")
    order = pd.Index(classes)
    if not order.is_unique:
        repeated = sorted({str(c) for c in order[order.duplicated()]})
        raise ValueError(f"classes must be distinct; {repeated} repeat.")
    if labels.ndim != 1:
        raise ValueError("labels must hold one category per row.")
    n = len(labels)
    if proba.shape != (n, len(classes)):
        raise ValueError(
            f"proba has shape {proba.shape}; it needs one row per label and one "
            f"column per class, {(n, len(classes))}."
        )
    if not np.isfinite(proba).all() or ((proba < 0.0) | (proba > 1.0)).any():
        raise ValueError(
            "proba must hold probabilities in [0, 1]: a value outside it makes "
            "the scores meaningless (a probability of 2 scores a negative log "
            "loss)."
        )
    off = np.abs(proba.sum(axis=1) - 1.0) > PROBABILITY_SUM_TOLERANCE
    if off.any():
        raise ValueError(
            f"{int(off.sum())} row(s) of proba do not sum to 1 (within "
            f"{PROBABILITY_SUM_TOLERANCE:g}); each row must be a distribution "
            "over the classes. A row of zeros usually means a learner was not "
            "fitted for that row's fold."
        )
    if w.shape != (n,):
        raise ValueError(f"weights need one value per label ({n}), got {w.shape}.")
    if not np.isfinite(w).all() or (w < 0.0).any() or not float(w.sum()) > 0.0:
        raise ValueError(
            "weights must be finite and non-negative, with a positive total."
        )
    positions = order.get_indexer(labels)
    if (positions < 0).any():
        unknown = sorted({str(label) for label in labels[positions < 0]})
        raise ValueError(f"Every label must be one of classes; {unknown[:5]} are not.")
    return proba, positions, w


def categorical_scores(
    proba: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    *,
    strata: Mapping[str, Sequence[object] | np.ndarray] | None = None,
) -> dict[str, object]:
    """Log loss, calibration slope and class-share gaps of probabilities.

    Args:
        proba: Class probabilities (out of fold), one column per ``classes``:
            each in ``[0, 1]``, each row summing to 1 (within
            :data:`PROBABILITY_SUM_TOLERANCE`).
        labels: Each row's observed category.
        classes: The distinct categories in column order, sorted or not.
        weights: Per-row weights: finite, non-negative, positive in total.
        strata: Name -> one stratum label per row. Share gaps (total
            variation between the weighted predicted and observed class
            shares) are reported within each level.

    Returns:
        ``log_loss`` (true-class probabilities floored at
        :data:`LOG_LOSS_FLOOR`), ``share_gap`` overall, ``calibration_slope``
        (``None`` where the Platt fit fell back to class rates) with its
        ``calibration_outcome``, and ``strata``: name -> level ->
        ``{"rows", "weight", "share_gap"}``. Counts are raw: a caller that
        publishes them bands them first.

    Raises:
        ValueError: If ``proba`` is not one probability distribution per row
            over ``classes``, ``classes`` repeat, a label is not one of
            ``classes``, the weights are not finite and non-negative with a
            positive total, or a stratum does not label every row.
    """
    proba, index, w = _scored_inputs(proba, labels, classes, weights)
    strata = {name: np.asarray(stratum) for name, stratum in (strata or {}).items()}
    for name, stratum in strata.items():
        if stratum.shape != index.shape:
            raise ValueError(
                f"strata[{name!r}] needs one label per row ({len(index)}), got "
                f"shape {stratum.shape}."
            )
    rows = np.arange(len(index))
    observed = np.zeros_like(proba)
    observed[rows, index] = 1.0
    true_class = np.maximum(proba[rows, index], LOG_LOSS_FLOOR)
    slope, _, outcome = _qrf._fit_gate_calibration(_qrf._log_proba(proba), index, w)
    result: dict[str, object] = {
        "log_loss": float(w @ -np.log(true_class) / w.sum()),
        "share_gap": _share_gap(proba, observed, w),
        "calibration_slope": (
            float(slope) if outcome == _qrf.GateOutcome.CROSS_FITTED else None
        ),
        "calibration_outcome": str(outcome),
        "strata": {},
    }
    for name, stratum in strata.items():
        levels: dict[object, dict[str, float | int]] = {}
        for level in np.unique(stratum):
            mask = stratum == level
            levels[level.item() if hasattr(level, "item") else level] = {
                "rows": int(mask.sum()),
                "weight": float(w[mask].sum()),
                "share_gap": _share_gap(proba[mask], observed[mask], w[mask]),
            }
        result["strata"][name] = levels
    return result
