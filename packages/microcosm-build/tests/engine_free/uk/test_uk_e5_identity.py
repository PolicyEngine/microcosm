from __future__ import annotations

import importlib.util

import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from test_support.microcosm_build.uk_area_home_values import (
    toy_area_home_values_resources,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository


def _identity_module():
    path = ROOT / "tools/verify_uk_identity_stability.py"
    spec = importlib.util.spec_from_file_location("verify_uk_identity_stability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_e5_identity_receipt_is_stable_under_row_permutation() -> None:
    # The packaged uprating divides by the WAS chain's kept draw, so the
    # London owner's main residence is recomputed through the area home
    # values on an invented seat and the Scottish owner's draw uprated
    # (microcosm#931).
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [101, 102, 201],
                "person_benunit_id": [10, 10, 20],
                "person_household_id": [1, 1, 2],
                "age": [25, 40, 20],
                "student_loan_repayments": [10.0, 30.0, 0.0],
                "student_loans": [0.0, 0.0, 1.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [10, 20]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "household_weight": [1.0, 1.0],
                "region": ["LONDON", "SCOTLAND"],
                "tenure_type": ["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE"],
                "constituency_code": pd.array(["E14000001", "S14000001"]),
                "source_household_id": [1, 2],
                "main_residence_value": [100.0, 200.0],
                "main_residence_value_rank_basis": [90.0, 200.0],
                "property_wealth": [150.0, 300.0],
                "corporate_wealth_excl_isa": [10.0, 20.0],
                "stocks_and_shares_isa": [1.0, 2.0],
                "student_loan_balance": [400.0, 100.0],
            }
        ),
        time_period="2023",
    )
    resource = {
        "values": [
            {"region": "LONDON", "avg_house_price": 200.0},
            {"region": "SCOTLAND", "avg_house_price": 400.0},
        ]
    }

    receipt = _identity_module().e5_identity_receipt(
        frame,
        regional_resource=resource,
        permutation_seed=42,
        area_home_values_resources=toy_area_home_values_resources(
            {"E14000001": (80.0, 120.0, 160.0)},
            resource,
            grid=(0.0, 0.5, 1.0),
            cap=160.0,
        ),
    )

    assert receipt["check"] == "uk_e5_identity_stability"
    assert receipt["uprating_denominator"] == "main_residence_value_rank_basis"
    assert receipt["identical_under_permutation"] is True
    assert receipt["permutation_mismatches"] == {}
    assert receipt["columns_by_entity"] == {
        "household": ["corporate_wealth", "main_residence_value"],
        "person": ["student_loan_balance"],
    }
    assert set(receipt["not_recomputed"]["household"]) == {"property_wealth"}
