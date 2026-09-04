"""The canonical conditional model: a regime-gated, chained, weighted QRF.

This is Microcosm's from-scratch regime-gated QRF imputer, implemented against
the :class:`~microcosm.frame.Frame`. Three ideas combine:

**Weighted bootstrap (forests only).** ``quantile_forest`` (and random forests
generally) cannot honor a ``sample_weight`` in their *predictive* distribution:
a fully-grown leaf holds one training row, so weighting impurity does not move
the value a draw reads out, and the backend uses ``sample_weight`` only as a
zero-weight filter on leaf membership. So weights are materialized *into the
data*: before each forest is grown, training rows are drawn with replacement
with probability proportional to weight (:func:`_weighted_bootstrap`). The leaf
distributions then reflect the weighted population. This is the weighting fix
that makes a weighted fit actually shift the draws.

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

The model has two front doors. Fitting on a :class:`~microcosm.frame.Frame`
resolves the owning entity and its typed weights (design by default). Fitting
on a plain :class:`pandas.DataFrame` — for use outside a microcosm stack —
requires the weights explicitly (a weight column name, a weight vector, or
``weights="none"``), because a bare table has no typed weights to default to.
Past that resolution the two paths are the same model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.fit.model import (
    DESIGN_WEIGHTS,
    WeightSpec,
    dataframe_fit_columns,
    predictors_targets_entity,
    resolve_dataframe_fit_weights,
    resolve_fit_weights,
    resolved_weight_kind,
)
from microcosm.frame import Frame

__all__ = [
    "RegimeGatedQRF",
    "FittedRegimeGatedQRF",
    "QRFChainState",
    "QRFChainStepResult",
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

    Predictors are not checked by this target-specific helper. Missing-feature
    semantics belong to the calling workflow, and the installed forest may
    reject NaN predictors; complete-case callers must filter them before fit.

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
    path). This is the operative half of the weighting fix: it is what makes
    leaf distributions — and the values drawn from them — weighted.

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


_SERIALIZED_FOREST_N_JOBS = 1


@dataclass(frozen=True)
class _Forest:
    """A fitted quantile forest plus the feature columns it was fit on."""

    model: RandomForestQuantileRegressor
    columns: tuple[str, ...]

    def __getstate__(self) -> dict[str, object]:
        """Return a worker-count-neutral pickle payload.

        ``RandomForestQuantileRegressor.n_jobs`` controls only runtime
        parallelism. A shallow model copy lets serialization pin that field
        without mutating the fitted object that will perform the first draw.
        """

        model = copy.copy(self.model)
        model.n_jobs = _SERIALIZED_FOREST_N_JOBS
        return {"model": model, "columns": self.columns}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Restore the current runtime worker setting after trusted loading."""

        model = state["model"]
        if not isinstance(model, RandomForestQuantileRegressor):
            raise TypeError("Serialized QRF forest model has an invalid type.")
        columns = state["columns"]
        if not isinstance(columns, tuple) or any(
            not isinstance(column, str) for column in columns
        ):
            raise TypeError("Serialized QRF forest columns are invalid.")
        model.n_jobs = _fit_n_jobs()
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "columns", columns)

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
        at 3M+ rows that matrix alone would be tens of GB — and the chunks run
        **in parallel** across a thread pool (:func:`_predict_workers`).
        ``quantile_forest``'s Cython ``predict`` releases the GIL over its
        per-sample aggregation, so the threads run that aggregation — the serial
        bottleneck when ``max_samples_leaf`` keeps full leaf populations — truly
        concurrently, without serializing the fitted forest to any worker.

        The parallel draw is **bit-identical** to a serial one: each row's value
        depends only on that row (the forest predict is pure per row), so the
        chunk boundaries never change a result. The tests pin this exact
        equality rather than an approximate match.

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
        if n == 0:
            return out

        workers = _predict_workers()
        # Bound the (rows x grid) matrix per chunk (memory) while cutting enough
        # chunks to balance across workers. Boundaries are draw-invariant, so
        # this only changes *when* each row is computed, never *what* it is.
        chunk = max(1, min(_PREDICT_CHUNK_ROWS, -(-n // (4 * workers))))
        bounds = [(start, min(start + chunk, n)) for start in range(0, n, chunk)]

        def _draw_chunk(bound: tuple[int, int]) -> None:
            start, stop = bound
            predictions = np.asarray(
                self.model.predict(features[start:stop], quantiles=list(grid))
            ).reshape(stop - start, len(grid))
            # Disjoint output slices: each row is written by exactly one worker,
            # so the shared ``out`` needs no lock.
            out[start:stop] = _interp_rows(quantiles[start:stop], grid, predictions)

        if workers <= 1 or len(bounds) <= 1:
            for bound in bounds:
                _draw_chunk(bound)
            return out

        # Force the forest's own ``apply`` to run serially inside each worker:
        # the outer pool already saturates the cores, so leaving it at the fit's
        # ``n_jobs=-1`` would fan a thread pool under every worker and thrash.
        # This synchronous draw owns the model, so the swap-and-restore is safe.
        saved_n_jobs = self.model.n_jobs
        self.model.n_jobs = 1
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(_draw_chunk, bounds):
                    pass
        finally:
            self.model.n_jobs = saved_n_jobs
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

#: Row-batch size for the draw predict. Bounds the ``(rows x grid)`` matrix and
#: the quantile-forest workspace so large support frames stream in fixed-memory
#: blocks instead of allocating the whole draw at once. Also the per-chunk cap
#: for the parallel draw, so a chunk's prediction matrix stays bounded no matter
#: how few workers split how many rows.
_PREDICT_CHUNK_ROWS = 10_000

#: Environment override for the parallel draw's worker count. A build tool sets
#: it to leave cores for concurrent work or to pin timing; unset defaults to
#: :func:`os.cpu_count`.
_PREDICT_WORKERS_ENV = "POPULACE_FIT_PREDICT_WORKERS"

#: Environment override for quantile-forest fit parallelism. Unset retains the
#: historical ``n_jobs=-1`` behavior; checkpointed production builds can pin it
#: to one (or another positive width) without changing the model API.
_N_JOBS_ENV = "POPULACE_FIT_N_JOBS"

#: A target whose ``y`` has at most this many distinct values is treated as
#: near-discrete for leaf-storage purposes. Continuous dollar targets have
#: distinct counts in the thousands; the pathological class (QBI
#: ``*_would_be_qualified`` six-level ratios, ``business_is_sstb``, mortgage
#: origination years) tops out at 17.
_DISCRETE_Y_UNIQUE_MAX = 32

#: Per-leaf sample cap applied to near-discrete targets in place of full leaf
#: retention. A leaf's conditional over at most a few dozen atoms is estimated
#: from this many seeded samples with multinomial error ~1/sqrt(4096) per atom
#: share — invisible at draw time — while bounding the forest's dense per-leaf
#: value store, which with full retention reached 57-67GB on a single such
#: target (Build M base, target 40).
_DISCRETE_Y_LEAF_BOUND = 4_096


def _fit_n_jobs() -> int:
    """Resolve quantile-forest fit parallelism.

    Unset preserves the historical ``n_jobs=-1`` default. If the environment
    variable is present it must be a strict positive base-10 integer; in
    particular, an empty string is an invalid configured value rather than a
    second spelling of "unset".

    Raises:
        ValueError: If ``POPULACE_FIT_N_JOBS`` is set to anything other than a
            positive integer.
    """
    override = os.environ.get(_N_JOBS_ENV)
    if override is None:
        return -1
    try:
        jobs = int(override)
    except ValueError:
        raise ValueError(
            f"{_N_JOBS_ENV} must be a positive integer, got {override!r}."
        ) from None
    if jobs < 1 or str(jobs) != override:
        raise ValueError(f"{_N_JOBS_ENV} must be a positive integer, got {override!r}.")
    return jobs


def _predict_workers() -> int:
    """Resolve the draw's thread-pool width.

    Threads, not processes: ``quantile_forest``'s Cython ``predict`` runs its
    per-sample aggregation under ``with nogil`` (a serial loop, no internal
    ``prange``), so it releases the GIL and a *shared* fitted forest is queried
    concurrently across row chunks with no per-worker serialization. A process
    pool would instead pickle the whole forest into every worker — hundreds of
    MB to gigabytes each, re-paid on every fit in the fit-then-predict-per-
    pattern transfer path — for no speed gain over the GIL-free threads.

    Defaults to :func:`os.cpu_count`; ``POPULACE_FIT_PREDICT_WORKERS`` overrides
    it with a positive integer.

    Raises:
        ValueError: If the override is set but not a positive integer.
    """
    override = os.environ.get(_PREDICT_WORKERS_ENV)
    if override is None or not override.strip():
        return os.cpu_count() or 1
    try:
        workers = int(override)
    except ValueError:
        raise ValueError(
            f"{_PREDICT_WORKERS_ENV} must be a positive integer, got {override!r}."
        ) from None
    if workers < 1:
        raise ValueError(
            f"{_PREDICT_WORKERS_ENV} must be a positive integer, got {workers}."
        )
    return workers


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

    Near-discrete targets are the exception to the ``None`` policy: nodes whose
    ``y`` is already pure stop splitting, so a dominant repeated value
    concentrates into leaves holding hundreds of thousands of identical
    samples, and the forest's dense per-leaf value store
    (``trees x leaves x largest_leaf``) reaches tens of GB. Such a leaf's
    conditional is a distribution over a handful of atoms, so a seeded
    subsample of :data:`_DISCRETE_Y_LEAF_BOUND` values per leaf reproduces
    every leaf quantile to multinomial noise while bounding memory; the guard
    triggers on the pre-bootstrap ``y`` so it is a deterministic property of
    the donor column, and it never touches continuous targets or explicit
    ``max_samples_leaf`` configurations.
    """
    x_fit, y_fit = _weighted_bootstrap(x, y, weights, rng)
    effective_max_samples_leaf = max_samples_leaf
    if max_samples_leaf is None and 2 <= len(np.unique(y)) <= _DISCRETE_Y_UNIQUE_MAX:
        effective_max_samples_leaf = _DISCRETE_Y_LEAF_BOUND
        print(
            "microcosm-fit: near-discrete target "
            f"({len(np.unique(y))} distinct values); bounding "
            f"max_samples_leaf at {_DISCRETE_Y_LEAF_BOUND} for this forest.",
            flush=True,
        )
    model = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        max_samples_leaf=effective_max_samples_leaf,
        random_state=seed,
        # Tree fitting and prediction parallelize without affecting the
        # seed-determined draws; forests are deterministic per random_state
        # regardless of worker count.
        n_jobs=_fit_n_jobs(),
    )
    model.fit(x_fit, y_fit)
    return _Forest(model=model, columns=columns)


#: Sentinel marking a target whose fitted forests were freed by
#: ``predict(release_models=True)``.
_RELEASED = object()


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


_CHAIN_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _IndexIdentity:
    """Compact, JSON-safe identity for a pandas row index.

    Checkpoint state must not embed millions of index values, but resuming on a
    reordered recipient would corrupt every chained prior. A SHA-256 digest of
    pandas' stable per-value hashes, combined with the exact index class, dtype,
    names, and length, gives the state a fixed-size row-order identity.
    """

    length: int
    class_name: str
    dtype: str
    names_repr: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "length": self.length,
            "class_name": self.class_name,
            "dtype": self.dtype,
            "names_repr": self.names_repr,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> _IndexIdentity:
        """Restore an identity from :meth:`to_dict` output."""
        expected = {"length", "class_name", "dtype", "names_repr", "sha256"}
        if set(value) != expected:
            raise ValueError(
                "QRF chain index identity keys must be exactly "
                f"{sorted(expected)}, got {sorted(value)}."
            )
        length = value["length"]
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise ValueError("QRF chain index identity length must be >= 0.")
        strings = {
            key: value[key] for key in ("class_name", "dtype", "names_repr", "sha256")
        }
        if any(not isinstance(item, str) for item in strings.values()):
            raise ValueError("QRF chain index identity metadata must be strings.")
        digest = strings["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("QRF chain index identity sha256 is malformed.")
        return cls(length=length, **strings)


def _index_identity(index: pd.Index) -> _IndexIdentity:
    """Return a stable, row-order-sensitive identity for ``index``."""
    hashes = pd.util.hash_pandas_object(index, index=False, categorize=False).to_numpy(
        dtype="<u8", copy=False
    )
    digest = hashlib.sha256()
    digest.update(hashes.tobytes(order="C"))
    return _IndexIdentity(
        length=len(index),
        class_name=type(index).__name__,
        dtype=str(index.dtype),
        names_repr=repr(tuple(index.names)),
        sha256=digest.hexdigest(),
    )


def _weight_identity(weights: np.ndarray | None) -> str:
    """Fingerprint the exact resolved fit weights for safe resume checks."""
    if weights is None:
        return "none"
    values = np.asarray(weights)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def _rng_state_json(rng: np.random.Generator) -> str:
    """Serialize a generator state to immutable canonical JSON text."""
    return json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _rng_from_state_json(value: str, *, stream: str) -> np.random.Generator:
    """Restore a generator from a chain state's canonical JSON text."""
    try:
        state = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"QRF chain {stream} RNG state is not valid JSON.") from exc
    if not isinstance(state, dict) or state.get("bit_generator") != "PCG64":
        raise ValueError(f"QRF chain {stream} RNG state must describe NumPy PCG64.")
    rng = np.random.default_rng()
    try:
        rng.bit_generator.state = state
    except (TypeError, ValueError) as exc:
        raise ValueError(f"QRF chain {stream} RNG state is invalid.") from exc
    return rng


@dataclass(frozen=True)
class QRFChainState:
    """Immutable checkpoint state for a target-at-a-time QRF chain.

    The state deliberately contains no frame, fitted model, or mutable NumPy
    object. It records the exact model configuration and chain order, the
    completed target prefix, donor/recipient row identities, resolved weight
    identity, and *separate* fit/draw RNG states. :meth:`to_dict` output can be
    passed through ``json.dumps``/``json.loads`` and restored with
    :meth:`from_dict` between subprocesses.

    Donor priors are never stored here: each target fit reads the observed prior
    targets from the donor input. Recipient priors are likewise external and
    must be supplied to :meth:`RegimeGatedQRF.fit_draw_next` as the exact raw
    draw prefix, allowing a build to checkpoint them losslessly.
    """

    predictors: tuple[str, ...]
    targets: tuple[str, ...]
    completed_targets: tuple[str, ...]
    entity: str | None
    weight_kind: str
    weight_sha256: str
    n_estimators: int
    zero_atol: float
    max_samples_leaf: int | float | None
    max_samples_leaf_kind: str
    seed: int
    fit_n_jobs: int
    donor_index: _IndexIdentity
    recipient_index: _IndexIdentity | None
    fit_rng_state_json: str
    draw_rng_state_json: str
    schema_version: int = _CHAIN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject malformed or internally inconsistent checkpoint state."""
        if self.schema_version != _CHAIN_STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported QRF chain state schema_version "
                f"{self.schema_version}; expected {_CHAIN_STATE_SCHEMA_VERSION}."
            )
        if not self.predictors or not self.targets:
            raise ValueError("QRF chain state requires predictors and targets.")
        if len(set(self.predictors)) != len(self.predictors):
            raise ValueError("QRF chain state predictors contain duplicates.")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("QRF chain state targets contain duplicates.")
        if set(self.predictors) & set(self.targets):
            raise ValueError("QRF chain state predictors and targets overlap.")
        expected_prefix = self.targets[: len(self.completed_targets)]
        if self.completed_targets != expected_prefix:
            raise ValueError(
                "QRF chain completed_targets must be an exact target-order prefix; "
                f"expected {expected_prefix}, got {self.completed_targets}."
            )
        if len(self.completed_targets) > len(self.targets):
            raise ValueError("QRF chain completed target prefix is too long.")
        expected_leaf_kind = (
            "none"
            if self.max_samples_leaf is None
            else "int"
            if isinstance(self.max_samples_leaf, int)
            and not isinstance(self.max_samples_leaf, bool)
            else "float"
            if isinstance(self.max_samples_leaf, float)
            else "invalid"
        )
        if self.max_samples_leaf_kind != expected_leaf_kind:
            raise ValueError(
                "QRF chain max_samples_leaf kind/value mismatch: "
                f"{self.max_samples_leaf_kind!r} vs {self.max_samples_leaf!r}."
            )
        _rng_from_state_json(self.fit_rng_state_json, stream="fit")
        _rng_from_state_json(self.draw_rng_state_json, stream="draw")

    @property
    def next_target(self) -> str | None:
        """The next target in the locked order, or ``None`` when complete."""
        if self.is_complete:
            return None
        return self.targets[len(self.completed_targets)]

    @property
    def is_complete(self) -> bool:
        """Whether every target in the locked order has been fit and drawn."""
        return len(self.completed_targets) == len(self.targets)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-roundtrippable checkpoint mapping."""
        return {
            "schema_version": self.schema_version,
            "predictors": list(self.predictors),
            "targets": list(self.targets),
            "completed_targets": list(self.completed_targets),
            "entity": self.entity,
            "weight_kind": self.weight_kind,
            "weight_sha256": self.weight_sha256,
            "model_config": {
                "n_estimators": self.n_estimators,
                "zero_atol": self.zero_atol,
                "max_samples_leaf": self.max_samples_leaf,
                "max_samples_leaf_kind": self.max_samples_leaf_kind,
                "seed": self.seed,
                "fit_n_jobs": self.fit_n_jobs,
            },
            "donor_index": self.donor_index.to_dict(),
            "recipient_index": (
                None if self.recipient_index is None else self.recipient_index.to_dict()
            ),
            "fit_rng_state": json.loads(self.fit_rng_state_json),
            "draw_rng_state": json.loads(self.draw_rng_state_json),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> QRFChainState:
        """Restore and validate state produced by :meth:`to_dict`."""
        expected = {
            "schema_version",
            "predictors",
            "targets",
            "completed_targets",
            "entity",
            "weight_kind",
            "weight_sha256",
            "model_config",
            "donor_index",
            "recipient_index",
            "fit_rng_state",
            "draw_rng_state",
        }
        if set(value) != expected:
            raise ValueError(
                "QRF chain state keys must be exactly "
                f"{sorted(expected)}, got {sorted(value)}."
            )

        def names(key: str) -> tuple[str, ...]:
            raw = value[key]
            if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
                raise ValueError(f"QRF chain state {key} must be a list of strings.")
            return tuple(raw)

        config = value["model_config"]
        if not isinstance(config, Mapping):
            raise ValueError("QRF chain state model_config must be an object.")
        config_keys = {
            "n_estimators",
            "zero_atol",
            "max_samples_leaf",
            "max_samples_leaf_kind",
            "seed",
            "fit_n_jobs",
        }
        if set(config) != config_keys:
            raise ValueError(
                "QRF chain model_config keys must be exactly "
                f"{sorted(config_keys)}, got {sorted(config)}."
            )
        donor_index = value["donor_index"]
        recipient_index = value["recipient_index"]
        if not isinstance(donor_index, Mapping):
            raise ValueError("QRF chain donor_index must be an object.")
        if recipient_index is not None and not isinstance(recipient_index, Mapping):
            raise ValueError("QRF chain recipient_index must be null or an object.")
        entity = value["entity"]
        if entity is not None and not isinstance(entity, str):
            raise ValueError("QRF chain entity must be null or a string.")
        string_fields = ("weight_kind", "weight_sha256")
        if any(not isinstance(value[key], str) for key in string_fields):
            raise ValueError("QRF chain weight metadata must be strings.")

        # Keep JSON's int and float spellings distinct: max_samples_leaf=1 and
        # 1.0 have different sklearn semantics even though Python says they are
        # equal, so the explicit kind field is load-bearing.
        n_estimators = config["n_estimators"]
        seed = config["seed"]
        fit_n_jobs = config["fit_n_jobs"]
        zero_atol = config["zero_atol"]
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool):
            raise ValueError("QRF chain n_estimators must be an integer.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("QRF chain seed must be an integer.")
        if (
            not isinstance(fit_n_jobs, int)
            or isinstance(fit_n_jobs, bool)
            or fit_n_jobs == 0
            or fit_n_jobs < -1
        ):
            raise ValueError("QRF chain fit_n_jobs must be -1 or a positive integer.")
        if not isinstance(zero_atol, (int, float)) or isinstance(zero_atol, bool):
            raise ValueError("QRF chain zero_atol must be numeric.")
        leaf = config["max_samples_leaf"]
        leaf_kind = config["max_samples_leaf_kind"]
        if not isinstance(leaf_kind, str):
            raise ValueError("QRF chain max_samples_leaf_kind must be a string.")

        fit_rng_state = json.dumps(
            value["fit_rng_state"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        draw_rng_state = json.dumps(
            value["draw_rng_state"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(
            schema_version=value["schema_version"],
            predictors=names("predictors"),
            targets=names("targets"),
            completed_targets=names("completed_targets"),
            entity=entity,
            weight_kind=value["weight_kind"],
            weight_sha256=value["weight_sha256"],
            n_estimators=n_estimators,
            zero_atol=float(zero_atol),
            max_samples_leaf=leaf,
            max_samples_leaf_kind=leaf_kind,
            seed=seed,
            fit_n_jobs=fit_n_jobs,
            donor_index=_IndexIdentity.from_dict(donor_index),
            recipient_index=(
                None
                if recipient_index is None
                else _IndexIdentity.from_dict(recipient_index)
            ),
            fit_rng_state_json=fit_rng_state,
            draw_rng_state_json=draw_rng_state,
        )


@dataclass(frozen=True)
class QRFChainStepResult:
    """One raw target draw plus the state *after* that target completed.

    ``state`` is the advanced state and is the one checkpoint callers must
    persist for the next subprocess; it is not the predecessor state passed to
    :meth:`RegimeGatedQRF.fit_draw_next`.

    Attributes:
        target: The target just fit and drawn.
        raw_draw: Positionally aligned float64 recipient draws. The array is
            read-only so checkpoint code cannot accidentally mutate the result.
        state: Chain state after this target, ready for the next subprocess.
        regime: The target's detected :class:`Regime`.
        weight_kind: The resolved donor fit weight kind.
    """

    target: str
    raw_draw: np.ndarray
    state: QRFChainState
    regime: str
    weight_kind: str


@dataclass(frozen=True)
class _ResolvedFitInput:
    """One fit input after the shared entity/weight resolution path."""

    entity: str | None
    table: pd.DataFrame
    weights: np.ndarray | None
    weight_kind: str


def _resolve_qrf_fit_input(
    frame_or_df: Frame | pd.DataFrame,
    predictors: list[str],
    targets: list[str],
    weights: WeightSpec,
) -> _ResolvedFitInput:
    """Resolve the two QRF front doors identically for all fit modes."""
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
    _validate_targets_finite(table, targets)
    return _ResolvedFitInput(entity, table, weight_values, weight_kind)


def _max_samples_leaf_kind(value: int | float | None) -> str:
    """Return the type tag needed to distinguish sklearn's int/float modes."""
    if value is None:
        return "none"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "invalid"


def _gate_draw_with_rng(
    gate: HistGradientBoostingClassifier,
    features: pd.DataFrame,
    columns: tuple[str, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw sign classes from a fitted gate using the supplied RNG stream."""
    x = features.loc[:, list(columns)].to_numpy(dtype=np.float64)
    proba = np.asarray(gate.predict_proba(x))
    cumulative = np.cumsum(proba, axis=1)
    u = rng.random(len(x))
    chosen = (cumulative >= u[:, None]).argmax(axis=1)
    return np.asarray(gate.classes_)[chosen]


def _draw_target_with_rng(
    features: pd.DataFrame,
    model: _TargetModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one target through its regime pipeline with an explicit RNG.

    Both monolithic prediction and targetwise subprocess steps call this helper,
    pinning RNG consumption and draw semantics to one implementation.
    """
    if model is _RELEASED:
        raise RuntimeError(
            "This target's fitted forests were released by a prior "
            "predict(release_models=True) call; refit or predict without "
            "releasing to draw again."
        )
    n = len(features)
    if model.regime == Regime.DEGENERATE_ZERO:
        return np.zeros(n, dtype=np.float64)

    quantiles = rng.random(n)
    if model.regime == Regime.POSITIVE_ONLY:
        return model.positive.draw(features, quantiles)
    if model.regime == Regime.NEGATIVE_ONLY:
        return model.negative.draw(features, quantiles)

    signs = _gate_draw_with_rng(model.gate, features, model.columns, rng)
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


def _draw_target_from_uniforms(
    features: pd.DataFrame,
    model: _TargetModel,
    quantiles: np.ndarray,
    sign_uniforms: np.ndarray,
) -> np.ndarray:
    """Evaluate a target without consuming or replacing any model RNG state."""
    if model is _RELEASED:
        raise RuntimeError("This target's fitted forests were released; refit to draw.")
    if model.regime == Regime.DEGENERATE_ZERO:
        return np.zeros(len(features), dtype=np.float64)
    if model.regime == Regime.POSITIVE_ONLY:
        return model.positive.draw(features, quantiles)
    if model.regime == Regime.NEGATIVE_ONLY:
        return model.negative.draw(features, quantiles)
    x = features.loc[:, list(model.columns)].to_numpy(dtype=np.float64)
    cumulative = np.cumsum(model.gate.predict_proba(x), axis=1)
    # Uniforms occupy [0, 1): strict comparison skips zero-probability classes
    # even at u=0. Close the final CDF bin against floating-point roundoff.
    cumulative[:, -1] = 1.0
    chosen = (cumulative > sign_uniforms[:, None]).argmax(axis=1)
    signs = np.asarray(model.gate.classes_)[chosen]
    values = np.zeros(len(features), dtype=np.float64)
    for sign, forest in ((1, model.positive), (-1, model.negative)):
        mask = signs == sign
        if mask.any() and forest is not None:
            values[mask] = forest.draw(features.loc[mask], quantiles[mask])
    return values


class RegimeGatedQRF:
    """The canonical :class:`~microcosm.fit.model.ConditionalModel`.

    Fits a regime-gated, sequentially-chained quantile-regression-forest model
    of ``P(targets | predictors)`` over a :class:`~microcosm.frame.Frame` — or a
    plain :class:`pandas.DataFrame` with explicit weights — with the weights
    materialized by weighted bootstrap. See the module docstring for the three
    mechanisms (weighted bootstrap, regime gates, chaining) and the two front
    doors.

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
        frame_or_df: Frame | pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> FittedRegimeGatedQRF:
        """Fit the conditional model. See
        :meth:`~microcosm.fit.model.ConditionalModel.fit`.

        On a Frame: resolves the single entity owning the predictors and
        targets and reads its typed weights per ``weights`` (``"none"`` is the
        only unweighted path). On a DataFrame: validates the columns and
        resolves the *explicit* ``weights`` — a weight column name, a weight
        vector, or ``"none"``; omitting them raises, because a bare table has
        no typed weights to default to. Either way it then detects each
        target's regime structurally and grows the gated, weighted-bootstrap
        forests in chain order.

        Raises:
            ValueError: If predictors/targets are empty, span more than one
                entity (Frame) or are missing from the columns, request a
                weight spec the input cannot resolve, or a target column
                contains non-finite (NaN/inf) values. Messages name the
                culprits.
        """
        predictors = list(predictors)
        targets = list(targets)
        resolved = _resolve_qrf_fit_input(frame_or_df, predictors, targets, weights)

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
            y = resolved.table[target].to_numpy(dtype=np.float64)
            features = resolved.table.loc[:, list(chained)].to_numpy(dtype=np.float64)
            target_models[target] = self._fit_target(
                features=features,
                y=y,
                columns=chained,
                weights=resolved.weights,
                rng=rng,
            )

        return FittedRegimeGatedQRF(
            entity=resolved.entity,
            predictors=predictors,
            targets=targets,
            target_models=target_models,
            zero_atol=self.zero_atol,
            draw_seed=draw_seed,
            weight_kind=resolved.weight_kind,
        )

    def start_chain(
        self,
        frame_or_df: Frame | pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> QRFChainState:
        """Initialize a safe target-at-a-time chain checkpoint.

        This performs the same entity, column, target-finiteness, and weight
        resolution as :meth:`fit`, then locks the donor row order, resolved
        weights, exact model configuration, target order, and independent fit
        and draw RNG streams into immutable JSON-roundtrippable state. No model
        is fit until :meth:`fit_draw_next`.
        """
        predictors = list(predictors)
        targets = list(targets)
        resolved = _resolve_qrf_fit_input(frame_or_df, predictors, targets, weights)
        fit_seed, draw_seed = np.random.SeedSequence(self.seed).spawn(2)
        fit_rng = np.random.default_rng(fit_seed)
        draw_rng = np.random.default_rng(draw_seed)
        return QRFChainState(
            predictors=tuple(predictors),
            targets=tuple(targets),
            completed_targets=(),
            entity=resolved.entity,
            weight_kind=resolved.weight_kind,
            weight_sha256=_weight_identity(resolved.weights),
            n_estimators=self.n_estimators,
            zero_atol=self.zero_atol,
            max_samples_leaf=self.max_samples_leaf,
            max_samples_leaf_kind=_max_samples_leaf_kind(self.max_samples_leaf),
            seed=self.seed,
            fit_n_jobs=_fit_n_jobs(),
            donor_index=_index_identity(resolved.table.index),
            recipient_index=None,
            fit_rng_state_json=_rng_state_json(fit_rng),
            draw_rng_state_json=_rng_state_json(draw_rng),
        )

    def fit_draw_next(
        self,
        frame_or_df: Frame | pd.DataFrame,
        recipient_predictors: Frame | pd.DataFrame,
        raw_prior_draws: pd.DataFrame,
        *,
        state: QRFChainState,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> QRFChainStepResult:
        """Fit and draw exactly the next target in a checkpointed chain.

        Fit features use the donor's *observed* completed-target prefix. Draw
        features use ``raw_prior_draws`` — never finalized, snapped, or
        otherwise post-processed values — so targetwise subprocesses preserve
        the monolithic chain's conditioning semantics bit for bit.

        Every call re-runs the ordinary input/weight resolution and refuses
        model-config drift, target-prefix drift, donor/recipient row reordering,
        weight drift, an entity/front-door change, or non-float64 raw priors.

        Returns:
            A result whose ``state`` is the state *after* this target and must
            be checkpointed for the next subprocess.
        """
        if not isinstance(state, QRFChainState):
            raise TypeError("state must be a QRFChainState.")
        self._validate_chain_config(state)
        if state.is_complete:
            raise ValueError("QRF chain is already complete; there is no next target.")

        predictors = list(state.predictors)
        targets = list(state.targets)
        resolved = _resolve_qrf_fit_input(frame_or_df, predictors, targets, weights)
        self._validate_chain_donor(state, resolved)
        recipient = self._chain_recipient_table(recipient_predictors, state)
        recipient_identity = _index_identity(recipient.index)
        if state.recipient_index is not None and (
            recipient_identity != state.recipient_index
        ):
            raise ValueError(
                "QRF chain recipient index/order changed since the first target."
            )
        self._validate_raw_prior_draws(raw_prior_draws, recipient.index, state)

        position = len(state.completed_targets)
        target = state.targets[position]
        chained = (*state.predictors, *state.completed_targets)
        fit_rng = _rng_from_state_json(state.fit_rng_state_json, stream="fit")
        draw_rng = _rng_from_state_json(state.draw_rng_state_json, stream="draw")
        target_model = self._fit_target(
            features=resolved.table.loc[:, list(chained)].to_numpy(dtype=np.float64),
            y=resolved.table[target].to_numpy(dtype=np.float64),
            columns=chained,
            weights=resolved.weights,
            rng=fit_rng,
        )

        augmented = recipient.loc[:, list(state.predictors)].copy()
        for prior in state.completed_targets:
            augmented[prior] = raw_prior_draws[prior].to_numpy(
                dtype=np.float64, copy=False
            )
        raw_draw = np.asarray(
            _draw_target_with_rng(augmented, target_model, draw_rng),
            dtype=np.float64,
        )
        raw_draw.setflags(write=False)
        advanced = replace(
            state,
            completed_targets=(*state.completed_targets, target),
            recipient_index=recipient_identity,
            fit_rng_state_json=_rng_state_json(fit_rng),
            draw_rng_state_json=_rng_state_json(draw_rng),
        )
        return QRFChainStepResult(
            target=target,
            raw_draw=raw_draw,
            state=advanced,
            regime=target_model.regime,
            weight_kind=resolved.weight_kind,
        )

    def _validate_chain_config(self, state: QRFChainState) -> None:
        """Refuse any model configuration drift across subprocesses."""
        actual = (
            self.n_estimators,
            self.zero_atol,
            self.max_samples_leaf,
            _max_samples_leaf_kind(self.max_samples_leaf),
            self.seed,
            _fit_n_jobs(),
        )
        expected = (
            state.n_estimators,
            state.zero_atol,
            state.max_samples_leaf,
            state.max_samples_leaf_kind,
            state.seed,
            state.fit_n_jobs,
        )
        if actual != expected or actual[3] != expected[3]:
            raise ValueError(
                "QRF chain model configuration changed across subprocesses: "
                f"expected {expected}, got {actual}."
            )

    @staticmethod
    def _validate_chain_donor(
        state: QRFChainState, resolved: _ResolvedFitInput
    ) -> None:
        """Refuse donor front-door, row-order, or resolved-weight drift."""
        if resolved.entity != state.entity:
            raise ValueError(
                "QRF chain donor entity/input kind changed: expected "
                f"{state.entity!r}, got {resolved.entity!r}."
            )
        if _index_identity(resolved.table.index) != state.donor_index:
            raise ValueError("QRF chain donor index/order changed since start_chain.")
        if resolved.weight_kind != state.weight_kind:
            raise ValueError(
                "QRF chain resolved weight kind changed: expected "
                f"{state.weight_kind!r}, got {resolved.weight_kind!r}."
            )
        if _weight_identity(resolved.weights) != state.weight_sha256:
            raise ValueError("QRF chain resolved weight values/order changed.")

    @staticmethod
    def _chain_recipient_table(
        recipient_predictors: Frame | pd.DataFrame, state: QRFChainState
    ) -> pd.DataFrame:
        """Resolve recipient predictors with the ordinary predict semantics."""
        if isinstance(recipient_predictors, Frame):
            if state.entity is None:
                raise ValueError(
                    "This QRF chain was started on a plain DataFrame, so it has "
                    "no entity to read from a recipient Frame."
                )
            table = recipient_predictors.table(state.entity)
            kind = "frame"
        else:
            table = recipient_predictors
            kind = "DataFrame"
        missing = [column for column in state.predictors if column not in table]
        if missing:
            raise ValueError(
                f"recipient predictors ({kind}) are missing column(s) {missing}; "
                f"the chain conditions on {list(state.predictors)}."
            )
        return table

    @staticmethod
    def _validate_raw_prior_draws(
        raw_prior_draws: pd.DataFrame,
        recipient_index: pd.Index,
        state: QRFChainState,
    ) -> None:
        """Require the exact ordered float64 raw-recipient target prefix."""
        if not isinstance(raw_prior_draws, pd.DataFrame):
            raise TypeError("raw_prior_draws must be a pandas DataFrame.")
        expected = list(state.completed_targets)
        actual = list(raw_prior_draws.columns)
        if actual != expected:
            raise ValueError(
                "raw_prior_draws columns must be the exact completed target "
                f"prefix in order: expected {expected}, got {actual}."
            )
        if not raw_prior_draws.index.equals(recipient_index):
            raise ValueError(
                "raw_prior_draws index/order must exactly match recipient predictors."
            )
        for target in expected:
            if raw_prior_draws[target].dtype != np.dtype(np.float64):
                raise ValueError(
                    f"raw_prior_draws[{target!r}] must retain float64 raw QRF "
                    f"draws, got {raw_prior_draws[target].dtype}."
                )
            values = raw_prior_draws[target].to_numpy(copy=False)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"raw_prior_draws[{target!r}] contains non-finite values."
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
        entity: The entity the predictors and targets live on, or ``None`` for
            a model fit on a plain DataFrame (which has no entities).
        predictors: The conditioning columns.
        targets: The drawn columns, in chain order.
        weight_kind: The weight kind this fit *resolved* to
            (:func:`~microcosm.fit.model.resolved_weight_kind`) — ``"design"`` /
            ``"importance"`` / ``"calibrated"`` for a Frame fit (the resolved,
            possibly inherited kind), ``"explicit"`` for a DataFrame fit weighted
            by a caller-supplied column or vector, or ``"none"`` for an
            unweighted fit. Read-only; the value a build records for the
            weights audit (microcosm #300).
    """

    def __init__(
        self,
        *,
        entity: str | None,
        predictors: list[str],
        targets: list[str],
        target_models: dict[str, _TargetModel],
        zero_atol: float,
        draw_seed: np.random.SeedSequence,
        weight_kind: str,
    ) -> None:
        self.entity = entity
        self.predictors = list(predictors)
        self.targets = list(targets)
        self._target_models = target_models
        self._zero_atol = zero_atol
        self._rng = np.random.default_rng(draw_seed)
        self._weight_kind = weight_kind

    @property
    def weight_kind(self) -> str:
        """The weight kind this fit resolved to. See the class docstring."""
        return self._weight_kind

    def regimes(self) -> dict[str, str]:
        """The detected :class:`Regime` label per target."""
        return {name: model.regime for name, model in self._target_models.items()}

    def predict(
        self,
        frame_or_df: Frame | pd.DataFrame,
        *,
        release_models: bool = False,
    ) -> pd.DataFrame:
        """Draw imputed targets. See
        :meth:`~microcosm.fit.model.FittedModel.predict`.

        Args:
            frame_or_df: Input carrying the predictor columns.
            release_models: Free each target's fitted forests as soon as its
                draws complete. The chain conditions on the DRAWN VALUES, not
                the fitted models, so draws are bit-identical either way; the
                only change is model lifetime. With ``max_samples_leaf=None``
                every forest retains its full leaf-sample store, so a chained
                stage's resident peak drops from the sum of all targets'
                forests to the largest single one. A released model cannot
                predict again — single-pass pipelines (the build stages)
                should pass ``True``; reusable fits keep the default.

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
            if release_models:
                self._target_models[target] = _RELEASED
        return out

    def predict_from_uniforms(
        self,
        frame_or_df: Frame | pd.DataFrame,
        *,
        quantiles: Mapping[str, np.ndarray],
        sign_uniforms: Mapping[str, np.ndarray],
    ) -> pd.DataFrame:
        """Draw using caller-supplied per-row uniforms, without advancing RNG.

        Each mapping must contain exactly the fitted targets, with one finite
        one-dimensional array in ``[0, 1)`` per target, aligned to input rows.
        Supply both arrays even for single-sign or all-zero targets. Later
        targets condition on earlier draws, just as in :meth:`predict`.

        Pairing uniforms with stable entity IDs makes results invariant to
        recipient ordering and batching. Fitted forests remain reusable. The
        legacy :meth:`predict` stream and its consumption order are unchanged.
        """
        features = self._predictor_frame(frame_or_df)
        arrays = {}
        for name, supplied in (
            ("quantiles", quantiles),
            ("sign_uniforms", sign_uniforms),
        ):
            if not isinstance(supplied, Mapping) or set(supplied) != set(self.targets):
                raise ValueError(f"{name} must contain exactly the fitted targets.")
            arrays[name] = {}
            for target in self.targets:
                values = np.asarray(supplied[target], dtype=np.float64)
                if values.shape != (len(features),):
                    raise ValueError(
                        f"{name}[{target!r}] must have shape ({len(features)},)."
                    )
                if (
                    not np.isfinite(values).all()
                    or ((values < 0) | (values >= 1)).any()
                ):
                    raise ValueError(f"{name}[{target!r}] uniforms must be in [0, 1).")
                arrays[name][target] = values
        out = pd.DataFrame(index=features.index)
        if features.empty:
            return out.reindex(columns=self.targets).astype(np.float64)
        augmented = features.copy()
        for target in self.targets:
            drawn = _draw_target_from_uniforms(
                augmented,
                self._target_models[target],
                arrays["quantiles"][target],
                arrays["sign_uniforms"][target],
            )
            out[target] = drawn
            augmented[target] = drawn
        return out

    def _predictor_frame(self, frame_or_df: Frame | pd.DataFrame) -> pd.DataFrame:
        """Extract the predictor columns from a Frame or DataFrame input."""
        if isinstance(frame_or_df, Frame):
            if self.entity is None:
                raise ValueError(
                    "This model was fit on a plain DataFrame, so it has no "
                    "entity to read from a Frame. Pass a DataFrame with the "
                    "predictor columns, or fit on a Frame to draw from frames."
                )
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

    def _draw_target(self, features: pd.DataFrame, model: _TargetModel) -> np.ndarray:
        """Draw one value per row for a single target via its regime pipeline."""
        return _draw_target_with_rng(features, model, self._rng)

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
        return _gate_draw_with_rng(gate, features, columns, self._rng)

    def __repr__(self) -> str:
        regimes = ", ".join(f"{k}:{v}" for k, v in self.regimes().items())
        return (
            f"FittedRegimeGatedQRF(entity={self.entity!r}, "
            f"predictors={self.predictors}, regimes[{regimes}])"
        )
