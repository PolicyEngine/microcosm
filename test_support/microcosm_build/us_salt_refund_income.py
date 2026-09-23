"""Contracts for the IRS PUF state-and-local-tax-refund income input."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
    support_channel_column,
)
from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_salt_refund_income_from_source,
    derive_puf_policyengine_variables,
)
from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.salt_refund_income import (
    SALT_REFUND_ARCHIVED_DERIVATION_URL,
    SALT_REFUND_ARCHIVED_EXPORT_URL,
    SALT_REFUND_ARCHIVED_IMPUTATION_URL,
    SALT_REFUND_ARCHIVED_PERSON_ALLOCATION_URL,
    SALT_REFUND_ARCHIVED_PUF_ARTIFACT_URL,
    derive_us_salt_refund_income_from_puf,
    us_salt_refund_income_signal_gate,
    us_salt_refund_income_stage_spec,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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


def _joint_support_frame() -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2], dtype="int64"),
                "person_household_id": np.asarray([10, 10], dtype="int64"),
                "person_tax_unit_id": np.asarray([100, 100], dtype="int64"),
                "person_spm_unit_id": np.asarray([1_000, 1_000], dtype="int64"),
                "person_family_id": np.asarray([10_000, 10_000], dtype="int64"),
                "person_marital_unit_id": np.asarray([100_000, 100_000], dtype="int64"),
                "employment_income_before_lsr": [50_000.0, 20_000.0],
                "self_employment_income_before_lsr": [0.0, 40_000.0],
            }
        ),
        "household": pd.DataFrame({"household_id": np.asarray([10], dtype="int64")}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([100], dtype="int64"),
                "filing_status_input": ["JOINT"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([1_000], dtype="int64")}),
        "family": pd.DataFrame({"family_id": np.asarray([10_000], dtype="int64")}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([100_000], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0]), WeightKind.DESIGN)},
    )


__all__ = [name for name in globals() if not name.startswith("__")]
