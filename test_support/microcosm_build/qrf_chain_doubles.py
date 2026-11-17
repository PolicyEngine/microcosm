"""QRF test doubles and checks for callers that fit one target at a time.

The CPS-ORG wage imputation and the ACS input transfer fit and draw each
target through ``RegimeGatedQRF.start_chain`` and ``fit_draw_next``, and drop
that target's forests and sign classifier before fitting the next. This
module provides:

- :class:`FitPredictChainDouble`, which gives a ``fit``/``predict`` test
  double the two chain methods those callers call;
- :class:`FitThenPredictQRF`, the real model computed the way the callers
  computed it before: one ``fit`` over every target, then one ``predict``;
- :func:`tracked_target_fits`, which keeps weak references to each fitted
  target and checks, when the next target's fit starts, that they are dead;
- :func:`table_bytes_digest`, a SHA-256 of a table's exact bytes.
"""

from __future__ import annotations

import gc
import hashlib
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import RegimeGatedQRF
from microcosm.frame import Frame

# ---------------------------------------------------------------------------
# Chain methods over a fit/predict double
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitPredictChainState:
    """Chain state of a :class:`FitPredictChainDouble`.

    ``drawn`` holds the fitted double's single ``predict`` result once the
    first step has asked for it; later steps read their target's column.
    """

    fitted: object
    targets: tuple[str, ...]
    weight_kind: str | None
    completed_targets: tuple[str, ...] = ()
    drawn: pd.DataFrame | None = None

    @property
    def next_target(self) -> str | None:
        if len(self.completed_targets) == len(self.targets):
            return None
        return self.targets[len(self.completed_targets)]

    @property
    def is_complete(self) -> bool:
        return self.next_target is None


class FitPredictChainDouble:
    """``start_chain`` and ``fit_draw_next`` built on a double's ``fit``/``predict``.

    ``start_chain`` calls the double's ``fit`` once, with the frame,
    predictors, targets and weights the callers used to pass to ``fit``. The
    first ``fit_draw_next`` calls the fitted double's ``predict`` once on the
    recipient predictors, as the callers used to, and every step returns the
    next target's column of that result. A subclass keeps its ``fit``,
    ``predict`` and recorded calls unchanged.
    """

    def start_chain(
        self,
        frame_or_df: Frame | pd.DataFrame,
        predictors: Sequence[str],
        targets: Sequence[str],
        *,
        weights: object,
        groups: object = None,
    ) -> FitPredictChainState:
        assert groups is None
        fitted = self.fit(frame_or_df, list(predictors), list(targets), weights=weights)
        return FitPredictChainState(
            fitted=fitted,
            targets=tuple(targets),
            weight_kind=getattr(fitted, "weight_kind", None),
        )

    def fit_draw_next(
        self,
        frame_or_df: Frame | pd.DataFrame,
        recipient_predictors: pd.DataFrame,
        raw_prior_draws: pd.DataFrame,
        *,
        state: FitPredictChainState,
        weights: object,
        quantiles: object = None,
        sign_uniforms: object = None,
        groups: object = None,
    ) -> SimpleNamespace:
        assert quantiles is None and sign_uniforms is None and groups is None
        target = state.next_target
        if target is None:
            raise ValueError("The chain is already complete.")
        assert list(raw_prior_draws.columns) == list(state.completed_targets)
        drawn = state.drawn
        if drawn is None:
            drawn = state.fitted.predict(recipient_predictors)
        regimes = getattr(state.fitted, "regimes", None)
        return SimpleNamespace(
            target=target,
            raw_draw=np.asarray(drawn[target], dtype=np.float64),
            state=replace(
                state,
                completed_targets=(*state.completed_targets, target),
                drawn=drawn,
            ),
            weight_kind=state.weight_kind,
            regime=regimes()[target] if callable(regimes) else None,
        )


class FitThenPredictQRF(FitPredictChainDouble, RegimeGatedQRF):
    """The real QRF, fitting every target before predicting any of them.

    A caller run with this class in place of ``RegimeGatedQRF`` computes what
    it computed before it moved to the one-target-at-a-time chain: one
    ``RegimeGatedQRF.fit`` over all targets, then one
    ``FittedRegimeGatedQRF.predict``. Comparing the two runs shows the chain
    reproduces those draws on the caller's own inputs.
    """


# ---------------------------------------------------------------------------
# Lifetimes of fitted targets
# ---------------------------------------------------------------------------


@dataclass
class TargetFitLifetimes:
    """Weak references to the objects each ``RegimeGatedQRF._fit_target`` built."""

    fitted_targets: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    _references: list[tuple[str, str, weakref.ref]] = field(default_factory=list)

    def record(self, target: str, model: object) -> None:
        self.fitted_targets.append(target)
        parts: list[tuple[str, object | None]] = [
            ("target model", model),
            ("sign classifier", getattr(model, "gate", None)),
        ]
        for sign in ("positive", "negative"):
            forest = getattr(model, sign, None)
            parts.append((f"{sign} forest", forest))
            parts.append((f"{sign} regressor", getattr(forest, "model", None)))
        for kind, part in parts:
            if part is not None:
                self._references.append((target, kind, weakref.ref(part)))

    def alive(self) -> list[tuple[str, str]]:
        return [
            (target, kind)
            for target, kind, reference in self._references
            if reference() is not None
        ]

    def require_released(self, *, before: str) -> None:
        alive = self.alive()
        if alive:
            message = (
                f"{before} started while earlier fitted targets were alive: {alive}"
            )
            self.violations.append(message)
            raise AssertionError(message)


@contextmanager
def tracked_target_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TargetFitLifetimes]:
    """Check that each fitted target is freed before the next target's fit.

    Wraps ``RegimeGatedQRF._fit_target``: on entry it requires every object an
    earlier call built (the target model, its sign classifier, its forests
    and their regressors) to be dead, then records weak references to the new
    ones. The garbage collector is off inside the block, so an object counts
    as freed only when nothing references it any more, which is when its
    memory goes back; a reference cycle would keep it alive and fail the
    check. Violations are also kept on the returned object, in case a caller
    catches the assertion.
    """

    lifetimes = TargetFitLifetimes()
    original = RegimeGatedQRF._fit_target

    def fit_target(self: RegimeGatedQRF, *, target: str, **kwargs: object) -> object:
        lifetimes.require_released(before=f"the fit of target {target!r}")
        model = original(self, target=target, **kwargs)
        lifetimes.record(target, model)
        return model

    monkeypatch.setattr(RegimeGatedQRF, "_fit_target", fit_target)
    collector_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield lifetimes
    finally:
        if collector_was_enabled:
            gc.enable()
        monkeypatch.setattr(RegimeGatedQRF, "_fit_target", original)


# ---------------------------------------------------------------------------
# Exact table digests
# ---------------------------------------------------------------------------


def _update_digest(
    digest: hashlib._Hash, label: object, values: pd.Index | pd.Series
) -> None:
    digest.update(repr(label).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    if isinstance(values.dtype, pd.CategoricalDtype):
        digest.update(repr(list(values.dtype.categories)).encode("utf-8"))
        digest.update(repr(values.dtype.ordered).encode("utf-8"))
    if isinstance(values.dtype, np.dtype) and values.dtype.kind in "biuf":
        digest.update(np.ascontiguousarray(values.to_numpy()).tobytes())
    else:
        digest.update(
            pd.util.hash_pandas_object(pd.Series(values), index=False)
            .to_numpy()
            .tobytes()
        )


def table_bytes_digest(table: pd.DataFrame) -> str:
    """SHA-256 over a table's index and columns: names, dtypes and value bytes.

    Numeric and boolean columns contribute their raw bytes, so a float that
    differs in its last bit (or a ``-0.0`` for a ``0.0``) changes the digest;
    a categorical column also contributes its categories and their order.
    """

    digest = hashlib.sha256()
    _update_digest(digest, ("index", tuple(table.index.names)), table.index)
    for column in table.columns:
        _update_digest(digest, column, table[column])
    return digest.hexdigest()


__all__ = [name for name in globals() if not name.startswith("__")]
