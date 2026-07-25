"""Synthetic-fixture and packaged-resource tests for QBI v3 evidence."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.qbi_v3_evidence import (
    SCF_MINIMUM_UNWEIGHTED_N,
    build_qbi_employer_structure_resource,
    build_scf_business_records,
    validate_qbi_employer_structure_resource,
    weighted_inverse_cdf,
)


def _scf_fixture(
    businesses: list[dict[str, float | int]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for household_id, business in enumerate(businesses, start=1):
        business_count = int(business.get("business_count", 1))
        for implicate in range(1, 6):
            row: dict[str, float | int] = {
                "y1": household_id * 10 + implicate,
                "x42001": float(business.get("weight", 100.0)),
                "x3103": 1,
                "x3104": 1,
                "x3105": business_count,
                "x3107": int(business.get("industry", 1)),
                "x3111": int(business.get("headcount", 1)),
                "x3113": 1,
                "x3114": 5,
                "x3119": int(business.get("legal_form", 2)),
                "x3128": int(business.get("ownership", 10_000)),
                "x3131": float(business.get("receipts", 20_000.0)),
                "x3132": float(business.get("net_income", 10_000.0)),
                "x3207": int(business.get("second_industry", 6)),
                "x3211": int(business.get("second_headcount", 1)),
                "x3213": 1,
                "x3214": 5,
                "x3219": int(business.get("second_legal_form", 11)),
                "x3228": int(business.get("second_ownership", 5_000)),
                "x3231": float(business.get("second_receipts", 40_000.0)),
                "x3232": float(business.get("second_net_income", 8_000.0)),
            }
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _provenance() -> dict[str, object]:
    return {
        "generated_by": "synthetic test",
        "run_command": "synthetic fixture",
        "inputs": [{"filename": "synthetic.dta", "sha256": "0" * 64}],
    }


def _employer_cell(
    payload: dict[str, object],
    *,
    income_band: str,
    legal_form_group: str,
    industry_code: int,
) -> dict[str, object]:
    cells = payload["cells"]
    assert isinstance(cells, list)
    return next(
        cell
        for cell in cells
        if cell["income_band"] == income_band
        and cell["legal_form_group"] == legal_form_group
        and cell["industry_code"] == industry_code
    )


def test_scf_implicates_pool_with_weight_divided_by_five() -> None:
    source = _scf_fixture(
        [
            {"weight": 100.0, "net_income": -1.0, "receipts": -1.0},
            {
                "weight": 200.0,
                "business_count": 2,
                "second_ownership": 2_500,
                "second_net_income": 8_000.0,
            },
        ]
    )

    records = build_scf_business_records(source)

    assert len(records) == 15
    assert records["weight"].sum() == pytest.approx(500.0)
    assert records.loc[records["business_slot"].eq(1), "weight"].sum() == 300.0
    first = records.loc[records["household_id"].eq(1)].iloc[0]
    assert first["gross_receipts"] == 0.0
    assert first["whole_net_income"] == 0.0
    second = records.loc[records["business_slot"].eq(2)].iloc[0]
    assert second["owned_net_income"] == 2_000.0


def test_scf_thin_cells_follow_independent_nested_fallbacks() -> None:
    businesses = [
        {
            "industry": 1,
            "headcount": 1 if index < 5 else 3,
            "receipts": 20_000.0,
            "net_income": 10_000.0,
        }
        for index in range(40)
    ]
    businesses.extend(
        {
            "industry": 2,
            "headcount": 6,
            "receipts": 50_000.0,
            "net_income": 10_000.0,
        }
        for _ in range(10)
    )

    payload = build_qbi_employer_structure_resource(
        _scf_fixture(businesses),
        provenance=_provenance(),
        minimum_unweighted_n=30.0,
    )

    exact = _employer_cell(
        payload,
        income_band="0_to_25k",
        legal_form_group="sole_or_informal",
        industry_code=1,
    )
    assert exact["requested_counts"]["implicate_adjusted_unweighted_n"] == 40.0
    assert exact["employer_presence"]["estimate_level"] == "exact"
    assert exact["employer_presence"]["probability_headcount_gt_1"] == pytest.approx(
        0.875
    )
    assert exact["headcount_size_distribution"]["estimate_level"] == "exact"

    thin = _employer_cell(
        payload,
        income_band="0_to_25k",
        legal_form_group="sole_or_informal",
        industry_code=2,
    )
    assert thin["requested_counts"]["implicate_adjusted_unweighted_n"] == 10.0
    assert thin["employer_presence"]["estimate_level"] == "income_form"
    assert thin["headcount_size_distribution"]["estimate_level"] == "income_form"

    empty_income = _employer_cell(
        payload,
        income_band="over_1m",
        legal_form_group="sole_or_informal",
        industry_code=7,
    )
    assert empty_income["employer_presence"]["estimate_level"] == "form"
    assert empty_income["headcount_size_distribution"]["estimate_level"] == "form"
    assert sum(
        empty_income["headcount_size_distribution"]["shares"].values()
    ) == pytest.approx(1.0)


def test_weighted_profit_margin_quantiles_use_inverse_cdf() -> None:
    quantiles = weighted_inverse_cdf(
        np.array([0.1, 0.2, 0.9]),
        np.array([1.0, 3.0, 1.0]),
        (0.05, 0.25, 0.5, 0.75, 0.95),
    )

    assert quantiles == {
        "q05": 0.1,
        "q25": 0.2,
        "q50": 0.2,
        "q75": 0.2,
        "q95": 0.9,
    }


def test_employer_resource_schema_rejects_nonprovisional_payload() -> None:
    source = _scf_fixture(
        [
            {
                "industry": (index % 6) + 1,
                "headcount": 3,
                "receipts": 20_000.0,
                "net_income": 10_000.0,
            }
            for index in range(35)
        ]
    )
    payload = build_qbi_employer_structure_resource(
        source,
        provenance=_provenance(),
        minimum_unweighted_n=SCF_MINIMUM_UNWEIGHTED_N,
    )
    broken = deepcopy(payload)
    broken["provisional"] = False

    with pytest.raises(ValueError, match="must remain provisional"):
        validate_qbi_employer_structure_resource(broken)
