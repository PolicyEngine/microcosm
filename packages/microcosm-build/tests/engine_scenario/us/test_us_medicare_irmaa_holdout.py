"""The IRMAA holdout's measures on the locked policyengine-us (microcosm#1206)."""

import numpy as np
import pytest

from microcosm.build.us_runtime.medicare_irmaa_holdout import (
    MEDICARE_IRMAA_HOLDOUT_VARIABLES,
    medicare_irmaa_holdout_payload,
)

# 2026 Trustees Report Tables V.E3 (p. 209) and V.E4 (p. 212), 2024 row,
# 85 percent column: the top monthly income-related adjustment amounts.
_TRUSTEES_2024_PART_B_TOP_MONTHLY = 419.30
_TRUSTEES_2024_PART_D_TOP_MONTHLY = 81.00


def _person(take_up: bool) -> dict:
    return {"age": {"2024": 70}, "takes_up_medicare_if_eligible": {"2024": take_up}}


def test_holdout_measures_follow_the_engine_on_a_household() -> None:
    """An enrollee and an eligible non-enrollee, each a single filer whose
    IRMAA MAGI ($600,000) is given for 2024, so it sits in the top tier.

    The engine charges the enrollee the Trustees' 2024 top-tier amounts, and
    gives the non-enrollee the eligibility-gated base premium but no
    enrollment-gated gross premium: the holdout counts only the enrollee.
    """
    from policyengine_us import Simulation

    situation = {
        "people": {"enrollee": _person(True), "non_enrollee": _person(False)},
        "tax_units": {
            "enrollee_unit": {
                "members": ["enrollee"],
                "medicare_irmaa_magi_two_years_prior": {"2024": 600_000},
            },
            "non_enrollee_unit": {
                "members": ["non_enrollee"],
                "medicare_irmaa_magi_two_years_prior": {"2024": 600_000},
            },
        },
        "families": {"family": {"members": ["enrollee", "non_enrollee"]}},
        "spm_units": {"spm_unit": {"members": ["enrollee", "non_enrollee"]}},
        "marital_units": {
            "enrollee_marital_unit": {"members": ["enrollee"]},
            "non_enrollee_marital_unit": {"members": ["non_enrollee"]},
        },
        "households": {
            "household": {
                "members": ["enrollee", "non_enrollee"],
                "state_code": {"2024": "TX"},
            }
        },
    }
    simulation = Simulation(situation=situation)
    values = {
        variable: np.asarray(simulation.calculate(variable, 2024))
        for variable in MEDICARE_IRMAA_HOLDOUT_VARIABLES
    }
    enrolled = values["medicare_enrolled"]
    gross = values["gross_medicare_part_b_premium_if_enrolled"]
    base = values["base_part_b_premium"]
    part_d = values["income_adjusted_part_d_premium_surcharge"]

    np.testing.assert_array_equal(enrolled, [True, False])
    assert base[1] == base[0] > 0
    assert gross[1] == 0
    assert gross[0] - base[0] == pytest.approx(12 * _TRUSTEES_2024_PART_B_TOP_MONTHLY)
    assert part_d[0] == pytest.approx(12 * _TRUSTEES_2024_PART_D_TOP_MONTHLY)
    assert part_d[1] == 0

    payload = medicare_irmaa_holdout_payload(
        enrolled, gross, base, part_d, np.array([1.0, 1.0]), period=2024
    )
    part_b_row, part_d_row = payload["parts"]
    assert part_b_row["weighted_payers"] == 1
    assert part_b_row["average_monthly_premium_per_payer"] == pytest.approx(
        _TRUSTEES_2024_PART_B_TOP_MONTHLY
    )
    assert part_d_row["weighted_payers"] == 1
    assert part_d_row["average_monthly_premium_per_payer"] == pytest.approx(
        _TRUSTEES_2024_PART_D_TOP_MONTHLY
    )
