# ruff: noqa: F401
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.frs_disability import (
    UKDWPDisabilityCategoryRates,
    UKDWPDisabilityFlagRates,
    _assert_frs_disability_stage_parameters,
    derive_frs_disability,
    uk_dwp_disability_category_rates,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def _category_rates() -> UKDWPDisabilityCategoryRates:
    return UKDWPDisabilityCategoryRates(
        aa_lower=10,
        aa_higher=20,
        dla_sc_lower=10,
        dla_sc_middle=20,
        dla_sc_higher=30,
        dla_m_lower=10,
        dla_m_higher=20,
        pip_m_standard=10,
        pip_m_enhanced=20,
        pip_dl_standard=10,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )


def _flags() -> UKDWPDisabilityFlagRates:
    return UKDWPDisabilityFlagRates(
        aa_higher=20,
        dla_sc_higher=30,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )


__all__ = [name for name in globals() if not name.startswith("__")]
