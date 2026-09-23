"""Contracts for the retired eCPS alimony input family."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime.alimony import (
    ALIMONY_ASEC_ARCHIVED_DERIVATION_URL,
    ALIMONY_PUF_ARCHIVED_DERIVATION_URL,
    STRIKE_BENEFITS_ASEC_ARCHIVED_DERIVATION_URL,
    derive_us_alimony_from_asec,
    derive_us_alimony_from_puf,
    us_alimony_signal_gate,
    us_alimony_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_alimony_from_sources,
)


class _ResolvedWeights:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _PersonFrame:
    def __init__(self, person: pd.DataFrame) -> None:
        self._person = person

    def table(self, entity: str) -> pd.DataFrame:
        assert entity == "person"
        return self._person

    def resolve_weights(self, entity: str) -> _ResolvedWeights:
        assert entity == "person"
        return _ResolvedWeights(np.ones(len(self._person)))


def _stacked_alimony_person() -> pd.DataFrame:
    source_count = 500
    source_numbers = np.repeat(np.arange(source_count), 2)
    clone_indices = np.tile([0, 1], source_count)
    source_channels = np.where(source_numbers < 250, "asec", "acs")
    asec_source = source_channels == "asec"

    codes = np.full(len(source_numbers), np.nan)
    amounts = np.full(len(source_numbers), np.nan)
    codes[asec_source] = 0.0
    amounts[asec_source] = 0.0
    reported_alimony = source_numbers == 10
    reported_strike = source_numbers == 20
    codes[reported_alimony] = 20.0
    amounts[reported_alimony] = 2_000.0
    codes[reported_strike] = 12.0
    amounts[reported_strike] = 700.0

    alimony_income = np.where(reported_alimony, amounts, 0.0)
    alimony_expense = np.where(source_numbers == 300, 3_000.0, 0.0)
    strike_benefits = np.where(reported_strike, amounts, 0.0)
    return pd.DataFrame(
        {
            "person_spine_source_id": source_numbers,
            "person_support_channel": source_channels,
            "person_support_clone_index": clone_indices,
            "OI_OFF": codes,
            "OI_VAL": amounts,
            "alimony_income": alimony_income,
            "alimony_expense": alimony_expense,
            "strike_benefits": strike_benefits,
            "miscellaneous_income": np.zeros(len(source_numbers)),
        }
    )


__all__ = [name for name in globals() if not name.startswith("__")]
