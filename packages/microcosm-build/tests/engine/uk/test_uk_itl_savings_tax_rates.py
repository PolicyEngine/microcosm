"""Engine lockstep for HMRC's Income Tax on savings rows (microcosm#1418).

Each ``hmrc.itl.savings_tax_at_<rate>_rate`` row binds Table 2.6's tax on
savings at one rate as that rate times the engine's savings income at it.
These tests fail if policyengine-uk's savings rates move in either tax year of
the calendar-2025 window the rows bind, or if a row names another rate's
income.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files

import pytest

_RATES = {
    "basic": ("basic", "basic_rate_savings_income"),
    "higher": ("higher", "higher_rate_savings_income"),
    "additional": ("additional", "add_rate_savings_income"),
}


def _rows() -> dict[str, dict]:
    targets = json.loads(
        files("microcosm.build.uk").joinpath("uk_population_targets.json").read_text()
    )["targets"]
    return {
        target["target_id"]: target
        for target in targets
        if target["target_id"].startswith("hmrc.itl.savings_tax_at_")
    }


@pytest.mark.parametrize("rate", sorted(_RATES))
def test_each_row_scales_its_rates_income_by_the_engines_rate(rate) -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    parameter, variable = _RATES[rate]
    row = _rows()[f"hmrc.itl.savings_tax_at_{rate}_rate"]
    expression = row["bindings"]["policyengine"]["value_expression"]
    match = re.fullmatch(r"(\d+(?:\.\d+)?) \* ([a-z_]+)", expression)
    assert match, expression
    assert match.group(2) == f"{variable}_annual"
    assert row["ledger_selector"]["dimension_values"] == {
        "income_tax_component": f"savings_at_{rate}_rate"
    }
    parameters = CountryTaxBenefitSystem().parameters
    # The window's two tax years: 2024-25 and 2025-26, read as the engine
    # reads a year's parameters.
    for instant in ("2024-04-30", "2025-04-30"):
        savings = parameters(instant).gov.hmrc.income_tax.rates.savings
        assert float(match.group(1)) == pytest.approx(getattr(savings, parameter))
