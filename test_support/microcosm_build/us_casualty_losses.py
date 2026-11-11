"""Contracts for the IRS PUF casualty-loss input restoration."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.casualty_losses import (
    derive_us_casualty_loss_from_puf,
    us_casualty_loss_signal_gate,
    us_casualty_loss_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_casualty_loss_from_source,
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(
        self,
        person: pd.DataFrame,
        weights: np.ndarray | None = None,
    ) -> None:
        self._person = person
        self._weights = np.ones(len(person)) if weights is None else weights

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.asarray(self._weights, dtype=np.float64))


__all__ = [name for name in globals() if not name.startswith("__")]
