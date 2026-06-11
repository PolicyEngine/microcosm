"""The conditional-model protocol and the weight-resolution it enforces.

A conditional model fits ``P(y | x)`` over a :class:`~populace.frame.Frame` and
draws from it. The defining property of this operator — the one the 2026-06
microimpute landmine violated — is that fitting is **weight-aware by
construction**: the weights come from the Frame's typed weight vectors, never
from a raw array a caller might forget to pass, and an unweighted fit is
impossible to request without writing ``weights="none"`` and meaning it.

:func:`resolve_fit_weights` is the single authority for that rule. Every model
in :mod:`populace.fit` routes its weight handling through it, so "no silent
unweighted default" is enforced in one place rather than re-litigated per model.
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
    "resolve_fit_weights",
    "predictors_targets_entity",
]

#: ``weights="design"`` — fit on the owning entity's design weights. The
#: default: a populace fit is weighted unless the caller opts out.
DESIGN_WEIGHTS = "design"

#: ``weights="none"`` — the *only* way to fit unweighted. Explicit by design:
#: an unweighted fit is a deliberate statistical choice, not a silent fallback.
NO_WEIGHTS = "none"

#: The weight specification accepted by a fit. A :class:`WeightKind` (or its
#: string value ``"design" | "importance" | "calibrated"``) selects which typed
#: weight vector of the owning entity to use; ``"none"`` fits unweighted.
WeightSpec = "WeightKind | str"

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
    """A model that fits a conditional distribution over a Frame.

    The protocol is intentionally minimal so trajectory and sequence models can
    slot in behind it later (DESIGN.md, "populace-fit: conditional models").
    The single non-negotiable is the weight contract: ``weights`` selects a
    typed weight vector of the entity that owns the predictors and targets, and
    defaults to that entity's design weights. ``weights="none"`` is the only
    way to fit unweighted.
    """

    def fit(
        self,
        frame: Frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: WeightSpec = DESIGN_WEIGHTS,
    ) -> FittedModel:
        """Fit the conditional distribution ``P(targets | predictors)``.

        Args:
            frame: The frame to fit on. Predictors and targets must all live on
                the same entity (resolved from the frame's column ownership).
            predictors: Conditioning variable names.
            targets: Variable names to learn the conditional of.
            weights: Which typed weight vector of the owning entity to weight
                the fit by. A :class:`~populace.frame.WeightKind` or its string
                value selects the kind; the default ``"design"`` uses that
                entity's design weights. ``"none"`` fits unweighted — the only
                way to do so, and a deliberate choice.

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
