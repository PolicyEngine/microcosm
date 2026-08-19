from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from microcosm.build.uk_runtime.national_frame import uk_national_frame

ROOT = Path(__file__).resolve().parents[3]


def _identity_module():
    path = ROOT / "tools/verify_uk_identity_stability.py"
    spec = importlib.util.spec_from_file_location("verify_uk_identity_stability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_e5_identity_receipt_is_stable_under_row_permutation() -> None:
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
                "main_residence_value": [100.0, 200.0],
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
    )

    assert receipt["check"] == "uk_e5_identity_stability"
    assert receipt["identical_under_permutation"] is True
    assert receipt["permutation_mismatches"] == {}
    assert receipt["columns_by_entity"] == {
        "household": [
            "corporate_wealth",
            "main_residence_value",
            "property_wealth",
        ],
        "person": ["student_loan_balance"],
    }
