"""Contracts for the IRS PUF educator-expense input restoration."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version

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
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.educator_expenses import (
    EDUCATOR_EXPENSE_ARCHIVED_ALLOCATION_URL,
    EDUCATOR_EXPENSE_ARCHIVED_DERIVATION_URL,
    EDUCATOR_EXPENSE_ARCHIVED_EXPORT_URL,
    EDUCATOR_EXPENSE_ARCHIVED_PUF_IMPUTATION_URL,
    derive_us_educator_expense_from_puf,
    us_educator_expense_signal_gate,
    us_educator_expense_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_educator_expense_from_source,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(self, person: pd.DataFrame, weights: np.ndarray | None = None) -> None:
        self._person = person
        self._weights = np.ones(len(person)) if weights is None else weights

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.asarray(self._weights, dtype=np.float64))


def _minimal_us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 100, 200], dtype="int64"),
            "person_family_id": np.asarray([1_000, 1_000, 2_000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [10_000, 10_000, 20_000], dtype="int64"
            ),
            "employment_income_before_lsr": np.asarray([75_000.0, 25_000.0, 50_000.0]),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2], dtype="int64"),
                "state_fips": np.asarray([6, 36], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.asarray([100, 200], dtype="int64")}
        ),
        "family": pd.DataFrame(
            {"family_id": np.asarray([1_000, 2_000], dtype="int64")}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([10_000, 20_000], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray([100.0, 100.0]),
                kind=WeightKind.DESIGN,
            )
        },
        pd.Series(["asec_2024", "asec_2024", "asec_2024"], name="stratum"),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
