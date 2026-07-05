"""The conditional-model protocol and the weight-resolution it enforces.

A conditional model fits ``P(y | x)`` over a :class:`~populace.frame.Frame` —
or a plain :class:`pandas.DataFrame` — and draws from it. The defining property
of this operator is that fitting is **weight-aware by construction**. On a
Frame, the weights come from its typed weight vectors, never from a raw array a
caller might forget to pass, and an unweighted fit is impossible to request
without writing ``weights="none"`` and meaning it. A bare DataFrame carries no
typed weights, so there the rule inverts from "safe default" to "no default":
the caller must state weights explicitly — a weight column name, a weight
vector, or ``weights="none"`` — and omitting them is an error, not a silent
unweighted fit.

:func:`resolve_fit_weights` (Frame) and :func:`resolve_dataframe_fit_weights`
(DataFrame) are the single authorities for that rule. Every model in
:mod:`populace.fit` routes its weight handling through them, so "no silent
unweighted default" is enforced in one place rather than re-litigated per
model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from populace.frame import Frame, WeightKind, assert_kind_transition

__all__ = [
    "ConditionalModel",
    "FittedModel",
    "WeightSpec",
    "DESIGN_WEIGHTS",
    "NO_WEIGHTS",
    "EXPLICIT_WEIGHTS",
    "resolve_fit_weights",
    "resolve_dataframe_fit_weights",
    "resolved_weight_kind",
    "predictors_targets_entity",
    "dataframe_fit_columns",
]

#: ``weights="design"`` — fit on the owning entity's design weights. The
#: default: a populace fit is weighted unless the caller opts out.
DESIGN_WEIGHTS = "design"

#: ``weights="none"`` — the *only* way to fit unweighted. Explicit by design:
#: an unweighted fit is a deliberate statistical choice, not a silent fallback.
NO_WEIGHTS = "none"

#: The *resolved* weight kind of a DataFrame fit weighted by an explicit column
#: or vector: weighted, but with no kernel :class:`~populace.frame.WeightKind`
#: provenance (design/importance/calibrated are Frame concepts). Distinct from
#: ``"none"`` — an explicit-vector fit is weighted, so the build weights audit
#: does not flag it — and reported honestly rather than mislabeled as a typed
#: kind. See :func:`resolved_weight_kind`.
EXPLICIT_WEIGHTS = "explicit"

#: The weight specification accepted by a fit. On a Frame: a
#: :class:`WeightKind` (or its string value
#: ``"design" | "importance" | "calibrated"``) selects which typed weight
#: vector of the owning entity to use. On a DataFrame: the name of a weight
#: column, or a 1-D weight vector (array / Series, read positionally) aligned
#: to the rows. ``"none"`` fits unweighted on either input — the only way to.
WeightSpec = "WeightKind | str | np.ndarray | pd.Series | Sequence[float]"

# Map the accepted string spellings to the kernel's WeightKind. ``"none"`` is
# handled separately (it is the absence of a kind, not a kind).
_KIND_BY_NAME: dict[str, WeightKind] = {kind.value: kind for kind in WeightKind}


@runtime_checkable
class FittedModel(Protocol):
    """A fitted conditional model that can draw imputed targets.

    A fitted model holds the conditional distribution learned at fit time and
    draws from it on demand. Draws are stochastic: each :meth:`predict` call
    samples fresh values from the conditional, advancing the model's seeded
    random state so repeated calls give independent draws (while a freshly
    fitted model with the same seed reproduces the same first draw).
    """

    def predict(self, frame_or_df: Frame | pd.DataFrame) -> pd.DataFrame:
        """Draw imputed target values for the rows of ``frame_or_df``.

        Args:
            frame_or_df: The rows to impute for. A :class:`~populace.frame.Frame`
                supplies the predictor columns from the entity that owns them; a
                :class:`pandas.DataFrame` is used directly (its columns must
                cover the predictors). Row count and index are preserved.

        Returns:
            A :class:`pandas.DataFrame` with one column per target, indexed to
            match the input rows. Values are a single stochastic draw from the
            fitted conditional.
        """
        ...


@runtime_checkable
class ConditionalModel(Protocol):
    """A model that fits a conditional distribution over a Frame or DataFrame.

    The protocol is intentionally minimal so trajectory and sequence models can
    slot in behind it later (DESIGN.md, "populace-fit: conditional models").
    The single non-negotiable is the weight contract: on a Frame, ``weights``
    selects a typed weight vector of the entity that owns the predictors and
    targets, and defaults to that entity's design weights; on a DataFrame —
    which has no typed weights — ``weights`` must be stated explicitly (a
    weight column name or a weight vector). ``weights="none"`` is the only way
    to fit unweighted on either input.
    """

    def fit(
        self,
        frame_or_df: Frame | pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> FittedModel:
        """Fit the conditional distribution ``P(targets | predictors)``.

        Args:
            frame_or_df: The rows to fit on. For a
                :class:`~populace.frame.Frame`, predictors and targets must all
                live on the same entity (resolved from the frame's column
                ownership). For a :class:`pandas.DataFrame`, they must all be
                columns of it.
            predictors: Conditioning variable names.
            targets: Variable names to learn the conditional of.
            weights: On a Frame: which typed weight vector of the owning entity
                to weight the fit by — a :class:`~populace.frame.WeightKind` or
                its string value; the default ``"design"`` uses that entity's
                design weights. On a DataFrame: required — the name of a weight
                column or a 1-D weight vector aligned to the rows (the
                ``"design"`` default is a Frame concept and raises). ``"none"``
                fits unweighted — the only way to do so, and a deliberate
                choice.

        Returns:
            A :class:`FittedModel`.
        """
        ...


def _duplicates(names: list[str]) -> list[str]:
    """Return the sorted set of names that appear more than once."""
    seen: set[str] = set()
    dups: set[str] = set()
    for name in names:
        (dups if name in seen else seen).add(name)
    return sorted(dups)


def _validate_fit_names(predictors: list[str], targets: list[str]) -> None:
    """Refuse empty, duplicated, or overlapping predictor/target names.

    The name discipline is the same on both front doors (Frame and DataFrame),
    so it lives here and the per-input validators layer ownership/membership
    checks on top.

    Raises:
        ValueError: Naming the duplicates or the overlap.
    """
    if not predictors:
        raise ValueError("fit requires at least one predictor.")
    if not targets:
        raise ValueError("fit requires at least one target.")
    predictor_dups = _duplicates(predictors)
    if predictor_dups:
        raise ValueError(f"Duplicate predictors: {predictor_dups}.")
    target_dups = _duplicates(targets)
    if target_dups:
        raise ValueError(f"Duplicate targets: {target_dups}.")
    overlap = sorted(set(predictors) & set(targets))
    if overlap:
        raise ValueError(
            f"Columns are both predictor and target: {overlap}; a target "
            "cannot condition on itself."
        )


def predictors_targets_entity(
    frame: Frame, predictors: list[str], targets: list[str]
) -> str:
    """Resolve the single entity that owns every predictor and target.

    Conditional fitting is a one-entity operation: a row of the fit is a row of
    one entity, and the weights that weight the fit are that entity's. Mixing
    entities (e.g. a household predictor and a person target) would silently
    cross-join rows at different grains, so it is refused here rather than
    producing a misaligned fit. Broadcast a group column onto persons first
    (``Frame.broadcast``) to fit across grains deliberately.

    Args:
        frame: The frame whose column ownership resolves the entity.
        predictors: Predictor variable names (at least one).
        targets: Target variable names (at least one).

    Returns:
        The name of the entity that owns all of ``predictors`` and ``targets``.

    Raises:
        ValueError: If ``predictors`` or ``targets`` is empty, a name is not a
            column on any entity table, or the names span more than one entity.
            The message names the offending columns and their entities.
    """
    _validate_fit_names(predictors, targets)
    owners: dict[str, str] = {}
    for column in (*predictors, *targets):
        owners[column] = frame.column_entity(column)  # raises naming the column
    entities = set(owners.values())
    if len(entities) != 1:
        by_entity: dict[str, list[str]] = {}
        for column, entity in owners.items():
            by_entity.setdefault(entity, []).append(column)
        described = "; ".join(
            f"{entity}: {sorted(columns)}"
            for entity, columns in sorted(by_entity.items())
        )
        raise ValueError(
            "Predictors and targets must all live on one entity, but they span "
            f"{sorted(entities)} ({described}). Broadcast a column onto the "
            "person entity (Frame.broadcast) to fit across grains deliberately."
        )
    return entities.pop()


def resolve_fit_weights(
    frame: Frame,
    entity: str,
    weights: WeightSpec,
) -> np.ndarray | None:
    """Resolve a ``weights`` spec to the per-row vector a fit trains on.

    This is the enforcement point for the operator's defining rule: a populace
    fit is weighted unless the caller writes ``weights="none"``. The vector
    returned is positionally aligned with ``entity``'s table — the grain every
    predictor and target shares — so a model can hand it straight to its
    weighted bootstrap.

    Weights are resolved through :meth:`~populace.frame.Frame.resolve_weights`,
    not only an entity's *own* stored vector: a person-level fit on a
    household-weighted frame reads the household weights broadcast onto persons,
    carrying the household's kind. The kind discipline is unchanged — the
    requested kind must match the *resolved* (possibly inherited) kind — so a
    fit can never silently weight by a kind the caller did not ask for.

    Args:
        frame: The frame carrying the typed weights.
        entity: The entity that owns the predictors and targets (its effective
            weights are the ones that weight the fit).
        weights: ``"none"`` to fit unweighted (the only unweighted path), or a
            :class:`~populace.frame.WeightKind` / its string value selecting
            which typed weight vector of ``entity`` to use.

    Returns:
        ``None`` when ``weights="none"`` (the model fits unweighted), otherwise
        a float64 array of length ``frame.n(entity)``.

    Raises:
        TypeError: If ``weights`` is neither a string nor a
            :class:`~populace.frame.WeightKind`.
        ValueError: If the spec is an unknown string, or the requested kind is
            not the kind of ``entity``'s resolved weights. The message names the
            valid specs / the resolved kind and an actionable fix.
    """
    if isinstance(weights, WeightKind):
        requested = weights
    elif isinstance(weights, str):
        if weights == NO_WEIGHTS:
            return None
        requested = _KIND_BY_NAME.get(weights)
        if requested is None:
            valid = [NO_WEIGHTS, *_KIND_BY_NAME]
            raise ValueError(
                f"Unknown weights spec {weights!r}; expected one of {valid}. "
                "An unweighted fit must be requested explicitly with "
                f"weights={NO_WEIGHTS!r}."
            )
    else:
        raise TypeError(
            "weights must be a WeightKind or one of the strings "
            f"{[NO_WEIGHTS, *_KIND_BY_NAME]}, got {type(weights).__name__}. "
            f"To fit unweighted, pass weights={NO_WEIGHTS!r} explicitly."
        )

    # Resolve through the frame's effective weights, not only the entity's own
    # stored vector: a person-level fit on a household-weighted frame inherits
    # the household weights (and their kind). Raises naming the entity / the
    # ambiguity if the weights cannot be resolved.
    resolved = frame.resolve_weights(entity)
    if resolved.kind is not requested:
        try:
            # Kinds only move forward (design -> importance -> calibrated).
            # If the requested kind is reachable from the resolved one, telling
            # the caller to advance the frame's weights is actionable.
            assert_kind_transition(resolved.kind, requested)
        except ValueError:
            # The requested kind ranks *below* the resolved kind, so advancing
            # is impossible (calibrated weights never revert to design). The
            # only actionable fix is to request the kind the frame actually
            # carries.
            raise ValueError(
                f"Requested {requested.value!r} weights for entity {entity!r}, "
                f"but its resolved weights are {resolved.kind.value!r}. Weight "
                "kinds only move forward "
                "(design -> importance -> calibrated), so the frame cannot be "
                f"reverted to {requested.value!r}; pass "
                f"weights={resolved.kind.value!r} to fit on the weights the "
                "frame carries."
            ) from None
        raise ValueError(
            f"Requested {requested.value!r} weights for entity {entity!r}, but "
            f"its resolved weights are {resolved.kind.value!r}. Either pass "
            f"weights={resolved.kind.value!r}, or advance the frame's weights "
            f"to {requested.value!r} first."
        )
    return np.asarray(resolved.values, dtype=np.float64)


def dataframe_fit_columns(
    df: pd.DataFrame, predictors: list[str], targets: list[str]
) -> None:
    """Validate that a DataFrame fit's predictors and targets are its columns.

    The DataFrame analogue of :func:`predictors_targets_entity`: the same name
    discipline (non-empty, no duplicates, no predictor/target overlap), with
    column membership in ``df`` standing in for entity ownership — a single
    table is a single grain, so there is no entity to resolve.

    Args:
        df: The DataFrame to fit on.
        predictors: Predictor variable names (at least one).
        targets: Target variable names (at least one).

    Raises:
        ValueError: If a name rule is violated or a name is missing from
            ``df``'s columns. The message names the missing columns.
    """
    _validate_fit_names(predictors, targets)
    missing = [c for c in (*predictors, *targets) if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame fit is missing column(s) {missing}; the fit "
            f"conditions on {predictors} and imputes {targets}."
        )


def resolve_dataframe_fit_weights(
    df: pd.DataFrame,
    weights: WeightSpec,
    *,
    predictors: list[str],
    targets: list[str],
) -> np.ndarray | None:
    """Resolve an explicit ``weights`` spec for a DataFrame fit.

    This is the enforcement point for the DataFrame front door's defining rule:
    a bare DataFrame has no typed weights, so there is no safe default to fall
    back on — the caller must state the weights. Accepted forms:

    - the name of a numeric weight column of ``df``,
    - a 1-D weight vector (array / Series / sequence) aligned positionally to
      ``df``'s rows (a Series' index is ignored), or
    - ``NO_WEIGHTS`` (``"none"``) to deliberately fit unweighted.

    The typed-kind spellings (``"design"`` / ``"importance"`` / ``"calibrated"``
    / a :class:`~populace.frame.WeightKind`) are Frame concepts and are refused
    — including the protocol's ``"design"`` default, which is what makes
    *omitting* ``weights`` on a DataFrame an error rather than a silent
    unweighted fit. A column literally named ``"none"`` (or a kind spelling)
    cannot be selected by name; pass its values as a vector instead.

    Args:
        df: The DataFrame to fit on.
        weights: The explicit weight spec (see above).
        predictors: The fit's predictor names — a weight column may not also
            be a predictor (the weight would leak into the features).
        targets: The fit's target names — a weight column may not also be a
            target.

    Returns:
        ``None`` for ``weights="none"``, otherwise a float64 vector of length
        ``len(df)``.

    Raises:
        TypeError: If ``weights`` is neither a string, a 1-D numeric
            vector, nor ``"none"``.
        ValueError: If a typed-kind spelling is passed, the named column is
            absent / non-numeric / also a predictor or target, or the vector
            has the wrong length or dimension, is non-finite, is negative, or
            sums to zero. Messages name the culprit and the fix.
    """
    if isinstance(weights, WeightKind) or (
        isinstance(weights, str) and weights in _KIND_BY_NAME
    ):
        spelled = weights.value if isinstance(weights, WeightKind) else weights
        raise ValueError(
            f"A DataFrame has no typed weights, so weights={spelled!r} cannot "
            "be resolved (typed weight kinds are a Frame concept, and the "
            f"{DESIGN_WEIGHTS!r} default only applies to Frames). Pass the "
            "name of a weight column in the DataFrame, a weight vector "
            f"aligned to its rows, or weights={NO_WEIGHTS!r} to deliberately "
            "fit unweighted."
        )

    if isinstance(weights, str):
        if weights == NO_WEIGHTS:
            return None
        if weights not in df.columns:
            raise ValueError(
                f"weights={weights!r} is not a column of the DataFrame. Pass "
                "the name of a weight column, a weight vector, or "
                f"weights={NO_WEIGHTS!r} to fit unweighted."
            )
        if weights in predictors:
            raise ValueError(
                f"Weight column {weights!r} is also a predictor; the weight "
                "would leak into the features. Drop it from predictors or "
                "pass the weights as a vector."
            )
        if weights in targets:
            raise ValueError(
                f"Weight column {weights!r} is also a target; a fit cannot "
                "impute its own weights. Drop it from targets or pass the "
                "weights as a vector."
            )
        column = df[weights]
        if not pd.api.types.is_numeric_dtype(column):
            raise ValueError(
                f"Weight column {weights!r} is not numeric (dtype {column.dtype})."
            )
        vector = column.to_numpy(dtype=np.float64)
    else:
        try:
            vector = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(
                "weights must be a weight column name, a 1-D numeric weight "
                f"vector, or {NO_WEIGHTS!r}; got {type(weights).__name__}."
            ) from None
        if vector.ndim != 1:
            raise ValueError(f"Weight vector must be 1-D, got shape {vector.shape}.")
        if len(vector) != len(df):
            raise ValueError(
                f"Weight vector length {len(vector)} does not match the "
                f"DataFrame's {len(df)} rows."
            )

    if not np.isfinite(vector).all():
        raise ValueError("Weights must be finite; found NaN or infinite values.")
    if (vector < 0).any():
        raise ValueError("Weights must be non-negative; found negative values.")
    if float(vector.sum()) <= 0.0:
        raise ValueError(
            "Weights sum to zero; a weighted fit needs positive mass. To fit "
            f"unweighted, pass weights={NO_WEIGHTS!r} explicitly."
        )
    return vector


def resolved_weight_kind(
    frame_or_df: Frame | pd.DataFrame,
    entity: str | None,
    weight_values: np.ndarray | None,
) -> str:
    """Name the weight kind a fit *resolved* to, for the build weights audit.

    Called after :func:`resolve_fit_weights` / :func:`resolve_dataframe_fit_weights`
    have produced ``weight_values``, so it reports what the fit actually resolved
    — not the spec the caller passed. On a Frame that means the *resolved*
    (possibly inherited) :class:`~populace.frame.WeightKind`, so a person-level
    fit on a household-calibrated frame reads back ``"calibrated"`` even though
    the caller wrote ``weights="design"`` would have raised; on either input an
    unweighted fit reads back ``"none"`` however it was spelled. This is the
    value a fitted model records for the build-level weights audit (populace
    #300), which blocks a release on an unlisted ``"none"``.

    Args:
        frame_or_df: The input the fit ran on.
        entity: The resolved entity for a Frame fit, or ``None`` for a DataFrame
            fit (which has no entities).
        weight_values: The resolved per-row weights, or ``None`` for an
            unweighted fit.

    Returns:
        ``"none"`` when ``weight_values is None`` (unweighted, either door); the
        resolved :class:`~populace.frame.WeightKind` value
        (``"design"`` / ``"importance"`` / ``"calibrated"``) for a Frame fit; or
        :data:`EXPLICIT_WEIGHTS` (``"explicit"``) for a DataFrame fit weighted by
        a caller-supplied column or vector (weighted, but with no typed kernel
        provenance).
    """
    if weight_values is None:
        return NO_WEIGHTS
    if isinstance(frame_or_df, Frame):
        if entity is None:  # pragma: no cover - defensive; a Frame fit has one
            raise ValueError(
                "A weighted Frame fit must resolve an entity to name its weight "
                "kind; got entity=None."
            )
        return frame_or_df.resolve_weights(entity).kind.value
    return EXPLICIT_WEIGHTS
