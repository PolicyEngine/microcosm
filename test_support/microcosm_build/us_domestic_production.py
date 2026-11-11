"""Contracts for the IRS PUF domestic-production-ALD restoration."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.domestic_production import (
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_DERIVATION_URL,
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_EXPORT_URL,
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_IMPUTATION_URL,
    DOMESTIC_PRODUCTION_ALD_ARCHIVED_PUF_ARTIFACT_URL,
    derive_us_domestic_production_ald_from_puf,
    us_domestic_production_ald_signal_gate,
    us_domestic_production_ald_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_domestic_production_ald_from_source,
)
from microcosm.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _TaxUnitFrame:
    def __init__(
        self,
        tax_unit: pd.DataFrame,
        weights: np.ndarray | None = None,
    ) -> None:
        self._tax_unit = tax_unit
        self._weights = np.ones(len(tax_unit)) if weights is None else weights

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "tax_unit"
        return self._tax_unit

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "tax_unit"
        return _ResolvedWeights(np.asarray(self._weights, dtype=np.float64))


__all__ = [name for name in globals() if not name.startswith("__")]
