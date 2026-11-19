"""The realised childcare take-up report (Vahid's round-2 review of #1338, item 3)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from test_support.paths import paths_for

ROOT = paths_for("microcosm-build").repository


def _tool():
    path = ROOT / "tools" / "report_uk_childcare_realised_take_up.py"
    spec = importlib.util.spec_from_file_location("realised", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Contract:
    rates = {
        "tax_free_childcare": 0.9,
        "extended_childcare": 0.5,
        "universal_childcare": 0.2,
        "targeted_childcare": 0.6,
        "extended_childcare_age_0_usage": 0.5,
        "extended_childcare_age_1_usage": 0.7,
    }

    def rate(self, key: str, build_year: int | None = None) -> float:
        return self.rates[key]


def _tables(units: int = 30, high_pay: int = 2):
    ids = np.arange(1, units + 1)
    # One adult and one child per unit; the child is under 2 in every unit.
    person = pd.DataFrame(
        {
            "person_id": np.concatenate([ids * 10, ids * 10 + 1]),
            "person_benunit_id": np.concatenate([ids, ids]),
            "person_household_id": np.concatenate([ids, ids]),
            "age": np.concatenate([np.full(units, 35), np.full(units, 1)]),
            "months_since_last_birthday": np.zeros(2 * units),
            "employment_income": np.concatenate(
                [np.where(ids <= high_pay, 150_000.0, 30_000.0), np.zeros(units)]
            ),
            "max_free_entitlement_hours_used": np.concatenate(
                [np.full(units, 30.0), np.where(ids % 2 == 0, 0.0, 30.0)]
            ),
        }
    )
    extended = ids % 3 != 0
    benunit = pd.DataFrame(
        {
            "benunit_id": ids,
            "would_claim_tfc": ids % 2 == 0,
            "would_claim_extended_childcare": extended,
            "would_claim_universal_childcare": extended | (ids % 5 == 0),
            "would_claim_targeted_childcare": ids % 4 == 0,
        }
    )
    household = pd.DataFrame({"household_id": ids, "source_household_id": ids})
    weights = np.where(ids % 2 == 0, 3.0, 1.0)
    return person, benunit, household, weights


def test_realised_take_up_reports_shares_beside_the_contract_rates() -> None:
    tool = _tool()
    person, benunit, household, weights = _tables()
    receipt = tool.realised_childcare_take_up(
        person, benunit, household, weights, contract=_Contract()
    )

    tfc = receipt["flags"]["would_claim_tfc"]
    assert tfc["contract_rate"] == 0.9
    cell = tfc["cells"]["units_with_a_child_under_5"]
    assert cell["source_households"] == 30
    assert cell["unweighted_share"] == pytest.approx(0.5)
    # Even units claim and carry weight 3: 45 / 60.
    assert cell["weighted_share"] == pytest.approx(0.75)
    universal = receipt["flags"]["would_claim_universal_childcare"]
    assert universal["contract_rate"] == pytest.approx(0.5 + 0.5 * 0.2)
    # Two high-pay units: under the floor of three, so no share is written.
    assert universal["cells"]["units_with_a_child_under_5_and_pay_over_100k"] == {
        "source_households": "<3"
    }
    usage = receipt["flags"]["max_free_entitlement_hours_used.age_1"]
    assert usage["contract_key"] == "extended_childcare_age_1_usage"
    assert usage["contract_rate"] == 0.7
    children = usage["cells"]["children_aged_1_in_claiming_units"]
    claiming = benunit["would_claim_extended_childcare"].to_numpy()
    assert children["source_households"] == int(claiming.sum())
    odd = (benunit["benunit_id"].to_numpy() % 2 == 1) & claiming
    assert children["unweighted_share"] == pytest.approx(odd.sum() / claiming.sum())


def test_realised_take_up_splits_natives_and_copies_when_known() -> None:
    tool = _tool()
    person, benunit, household, weights = _tables()
    copies = benunit["benunit_id"].to_numpy() > 15
    receipt = tool.realised_childcare_take_up(
        person, benunit, household, weights, contract=_Contract(), copies=copies
    )

    cells = receipt["flags"]["would_claim_extended_childcare"]["cells"]
    assert cells["native_units_with_a_child_under_5"]["source_households"] == 15
    assert cells["copy_units_with_a_child_under_5"]["source_households"] == 15
