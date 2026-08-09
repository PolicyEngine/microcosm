"""Weighted input-mass totals: the evidence the parity gate runs on."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.input_mass import input_mass_totals
from microcosm.build.us_runtime import us_input_mass_totals as package_input_mass_totals
from microcosm.build.us_runtime.input_mass import us_input_mass_totals
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _frame(**person_extra: object) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 20], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 200], dtype="int64"),
            "person_spm_unit_id": np.asarray([1000, 2000, 2000], dtype="int64"),
            "person_family_id": np.asarray([10000, 20000, 20000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [100000, 200000, 200001], dtype="int64"
            ),
            "student_loan_interest": [100.0, 50.0, 0.0],
            "has_esi": [True, False, True],
            "ssn_card_type": ["CITIZEN", "CITIZEN", "NONE"],
            **person_extra,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": np.asarray([10, 20], dtype="int64")}
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([100, 200], dtype="int64"),
                    "health_savings_account_ald": [500.0, 250.0],
                }
            ),
            "spm_unit": pd.DataFrame(
                {
                    "spm_unit_id": np.asarray([1000, 2000], dtype="int64"),
                    "spm_unit_pre_subsidy_childcare_expenses": [0.0, 1_200.0],
                }
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([10000, 20000], dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([100000, 200000, 200001], dtype="int64")}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([10.0, 30.0]), WeightKind.CALIBRATED)},
    )


def test_legacy_us_exports_are_exact_shared_function_aliases() -> None:
    assert us_input_mass_totals is input_mass_totals
    assert package_input_mass_totals is input_mass_totals


def test_totals_weight_each_entity_through_household_membership() -> None:
    totals = us_input_mass_totals(_frame())

    # Persons inherit their household's weight: 100*10 + 50*30 + 0*30.
    assert totals["student_loan_interest"] == pytest.approx(2_500.0)
    # Boolean columns total their weighted True mass: 10 + 30.
    assert totals["has_esi"] == pytest.approx(40.0)
    # Group entities inherit household weights too.
    assert totals["health_savings_account_ald"] == pytest.approx(
        500.0 * 10.0 + 250.0 * 30.0
    )
    assert totals["spm_unit_pre_subsidy_childcare_expenses"] == pytest.approx(
        1_200.0 * 30.0
    )


def test_structural_and_string_columns_are_skipped() -> None:
    totals = us_input_mass_totals(_frame())

    assert "person_id" not in totals
    assert "person_household_id" not in totals
    assert "household_id" not in totals
    assert "ssn_card_type" not in totals


def test_column_restriction_filters_raw_source_columns() -> None:
    totals = us_input_mass_totals(
        _frame(WSAL_VAL=[1.0, 2.0, 3.0]),
        columns=["student_loan_interest"],
    )

    assert set(totals) == {"student_loan_interest"}
