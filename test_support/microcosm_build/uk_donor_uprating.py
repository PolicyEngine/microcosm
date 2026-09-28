"""Declared donor uprating to the FRS 2024-25 base year (microcosm#890 U)."""

# ruff: noqa: F401

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import donor_uprating as module
from microcosm.build.uk_runtime.donor_uprating import (
    apply_donor_uprating,
    donor_uprating_factors,
    uprating_operation,
)
from microcosm.build.uk_runtime.etb_services import (
    etb_donor_uprating,
    rail_fare_index_denominator_key,
)
from microcosm.build.uk_runtime.lcfs_consumption import lcfs_donor_uprating
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"
CPI = "gov.economic_assumptions.indices.obr.consumer_price_index"

FAKE_INDICES = {
    CPI: {2023: 1.51964, 2024: 1.55763},
    "gov.economic_assumptions.indices.obr.average_earnings": {2023: 1.0, 2024: 1.051},
    "gov.economic_assumptions.indices.obr.per_capita.mixed_income": {
        2023: 2.0,
        2024: 2.0546,
    },
    "gov.economic_assumptions.indices.obr.private_pension_index": {
        2023: 1.0,
        2024: 1.05,
    },
    "gov.economic_assumptions.indices.obr.petrol_spending_litre_proxy": {
        2023: 1.61756,
        2024: 1.37799,
    },
    "gov.economic_assumptions.indices.obr.diesel_spending_litre_proxy": {
        2023: 1.34477,
        2024: 1.38733,
    },
}


def _fake_reader(path: str, year: int) -> float:
    try:
        return FAKE_INDICES[path][year]
    except KeyError as error:  # pragma: no cover - guards a typo in a test
        raise ValueError(f"unexpected read {path!r} at {year}") from error


def _stages():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return spec.sources.stage_map()


__all__ = [name for name in globals() if not name.startswith("__")]
