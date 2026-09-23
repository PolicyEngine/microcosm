"""Contracts for source-backed PUF capital-gain detail inputs."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.capital_gain_details import (
    CAPITAL_GAIN_DETAILS_ARCHIVED_DERIVATION_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_EXPORT_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_IMPUTATION_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_PERSON_ALLOCATION_URL,
    CAPITAL_GAIN_DETAILS_ARCHIVED_PUF_ARTIFACT_URL,
    US_CAPITAL_GAIN_DETAILS_NONCONSTANT_PERSON_COLUMNS,
    US_CAPITAL_GAIN_DETAILS_NONCONSTANT_TAX_UNIT_COLUMNS,
    US_CAPITAL_GAIN_DETAILS_OUTPUT_COLUMNS,
    derive_us_capital_gain_details_from_puf,
    us_capital_gain_details_signal_gate,
    us_capital_gain_details_stage_spec,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_capital_gain_details_from_sources,
    derive_puf_policyengine_variables,
)
from microcosm.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays
from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _CapitalGainFrame:
    def __init__(
        self,
        *,
        person: pd.DataFrame,
        tax_unit: pd.DataFrame,
        person_weights: np.ndarray | None = None,
        tax_unit_weights: np.ndarray | None = None,
    ) -> None:
        self._tables = {"person": person, "tax_unit": tax_unit}
        self._weights = {
            "person": (
                np.ones(len(person)) if person_weights is None else person_weights
            ),
            "tax_unit": (
                np.ones(len(tax_unit)) if tax_unit_weights is None else tax_unit_weights
            ),
        }

    def table(self, entity: str) -> pd.DataFrame:
        return self._tables[entity]

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        return _ResolvedWeights(np.asarray(self._weights[entity], dtype=np.float64))


__all__ = [name for name in globals() if not name.startswith("__")]
