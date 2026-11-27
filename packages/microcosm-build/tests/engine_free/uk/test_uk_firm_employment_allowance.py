"""Assumed Employment Allowance flags and their held-out HMRC check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_employment_allowance import (
    UK_FIRM_EMPLOYMENT_ALLOWANCE_CHECK_FILE,
    UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS,
    UK_FIRM_EMPLOYMENT_ALLOWANCE_STATUS,
    assign_uk_firm_employment_allowance,
    read_uk_firm_employment_allowance_check,
    uk_firm_employment_allowance_comparison,
    uk_firm_employment_allowance_flags,
)
from microcosm.build.uk_runtime.firm_table import UK_FIRM_POLICYENGINE_UK_INPUT_STATUS

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "uk"
    / "firm_employment_allowance_2024_25"
)


def test_flags_follow_legal_status_and_employment() -> None:
    # company, sole proprietor, partnership, central government, non-profit
    status = np.array([1, 1, 2, 2, 3, 3, 3, 5, 7])
    employment = np.array([1, 2, 1, 2, 1, 2, 3, 1, 1])
    flags = uk_firm_employment_allowance_flags(status, employment)
    assert flags["firm_claims_employment_allowance"].tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        True,
    ]
    assert flags["firm_is_single_director_company"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(values.dtype == bool for values in flags.values())
    # No headcount at all: neither a sole proprietor nor a partnership claims.
    none = uk_firm_employment_allowance_flags(np.array([2, 3, 1]), np.array([0, 0, 0]))
    assert none["firm_claims_employment_allowance"].tolist() == [False, False, True]
    assert not none["firm_is_single_director_company"].any()


def test_flags_refuse_bad_input() -> None:
    with pytest.raises(ValueError, match="same shape"):
        uk_firm_employment_allowance_flags(np.array([1, 2]), np.array([1]))
    with pytest.raises(ValueError, match="non-negative"):
        uk_firm_employment_allowance_flags(np.array([1]), np.array([-1]))
    with pytest.raises(ValueError, match="ONS attributes first"):
        assign_uk_firm_employment_allowance(pd.DataFrame({"employment": [1]}))


def test_assign_adds_only_the_flags() -> None:
    firms = pd.DataFrame(
        {"ons_legal_status": [1, 2, 3], "employment": [1, 1, 5], "x": [1.0, 2.0, 3.0]}
    )
    out = assign_uk_firm_employment_allowance(firms)
    assert list(out.columns) == [*firms.columns, *UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS]
    pd.testing.assert_frame_equal(out[list(firms.columns)], firms)
    assert out["firm_claims_employment_allowance"].tolist() == [True, False, True]
    assert out["firm_is_single_director_company"].tolist() == [True, False, False]


def test_flags_are_assumed_inputs() -> None:
    assert UK_FIRM_EMPLOYMENT_ALLOWANCE_STATUS == "assumed"
    for column in UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS:
        assert UK_FIRM_POLICYENGINE_UK_INPUT_STATUS[column] == "assumed"


def test_held_out_check_fixture_is_sourced() -> None:
    check = read_uk_firm_employment_allowance_check(FIXTURE)
    assert check.tax_year == "2024-25"
    assert check.claimants == 1_223_000
    assert check.cost_gbp == 3_300_000_000
    csv_path = FIXTURE / UK_FIRM_EMPLOYMENT_ALLOWANCE_CHECK_FILE
    assert check.sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    provenance = json.loads(
        (
            FIXTURE / f"{UK_FIRM_EMPLOYMENT_ALLOWANCE_CHECK_FILE}.provenance.json"
        ).read_text()
    )
    assert provenance["period"] == "2024-25"
    assert provenance["accessed"] == "2026-11-22"
    assert "Never a calibration target" in provenance["use"]
    (source,) = provenance["sources"].values()
    assert source["url"].startswith("https://assets.publishing.service.gov.uk/")
    assert source["edition"] == "Tax relief statistics, January 2026"
    assert len(source["sha256"]) == 64


def test_comparison_weights_claimants_and_cost() -> None:
    check = read_uk_firm_employment_allowance_check(FIXTURE)
    allowance = np.array([5_000.0, 0.0, 2_000.0])
    weight = np.array([100.0, 50.0, 200.0])
    result = uk_firm_employment_allowance_comparison(allowance, weight, check)
    assert result["model_claimants"] == 300.0
    assert result["model_cost_gbp"] == 900_000.0
    assert result["model_cost_per_claimant_gbp"] == 3_000.0
    assert result["claimants_ratio"] == pytest.approx(300.0 / 1_223_000)
    assert result["cost_ratio"] == pytest.approx(900_000.0 / 3.3e9)
