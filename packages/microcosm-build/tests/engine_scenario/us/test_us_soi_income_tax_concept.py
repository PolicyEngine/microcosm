"""SOI's income tax after credits in policyengine-us.

SOI's income tax after credits (A06500, N06500) is Form 1040 line 22: income
tax before credits less the nonrefundable credits, before every refundable
credit and without the Schedule 2 Part II taxes. The release tool reads it as
``income_tax_before_credits - income_tax_capped_non_refundable_credits``
(``SOI_DIFFERENCE_CONCEPTS``). These households pin the identities that
reading rests on at the locked engine:

- ``income_tax_capped_non_refundable_credits`` equals
  ``min(income_tax_before_credits, income_tax_non_refundable_credits)``, so
  line 22 lies between zero and the tax before credits;
- ``income_tax_before_refundable_credits`` equals line 22 plus the net
  investment income tax, the recapture of investment credit, unreported
  payroll tax and the qualified retirement penalty; and
- ``income_tax`` equals that less ``income_tax_refundable_credits``.

They also show that ``max(income_tax, 0)``, which the SOI rows read before,
misses line 22 in both directions: refundable credits pull it below, and the
net investment income tax pushes it above.
"""

from __future__ import annotations

import numpy as np
import pytest

YEAR = 2024
_PEOPLE = {
    # Head of household, one child, $45,000 of wages: line 22 tax remains
    # after the nonrefundable child tax credit, and the EITC refunds more.
    "parent": {"age": {YEAR: 35}, "employment_income": {YEAR: 45_000}},
    "child": {"age": {YEAR: 5}},
    # $250,000 of wages and $150,000 of interest: net investment income tax.
    "investor": {
        "age": {YEAR: 50},
        "employment_income": {YEAR: 250_000},
        "taxable_interest_income": {YEAR: 150_000},
    },
    "retiree": {"age": {YEAR: 70}},
}
_UNITS = {
    "parent_unit": (("parent", "child"), "HEAD_OF_HOUSEHOLD"),
    "investor_unit": (("investor",), "SINGLE"),
    "retiree_unit": (("retiree",), "SINGLE"),
}
_PART_II_TAXES = (
    "net_investment_income_tax",
    "recapture_of_investment_credit",
    "unreported_payroll_tax",
    "qualified_retirement_penalty",
)


@pytest.fixture(scope="module")
def tax_units() -> dict[str, dict[str, float]]:
    from policyengine_us import Simulation

    situation = {
        "people": _PEOPLE,
        "tax_units": {
            name: {"members": list(members), "filing_status": {YEAR: status}}
            for name, (members, status) in _UNITS.items()
        },
        "families": {
            f"{name}_family": {"members": list(members)}
            for name, (members, _) in _UNITS.items()
        },
        "spm_units": {
            f"{name}_spm": {"members": list(members)}
            for name, (members, _) in _UNITS.items()
        },
        "households": {
            f"{name}_household": {
                "members": list(members),
                "state_code_str": {YEAR: "TX"},
            }
            for name, (members, _) in _UNITS.items()
        },
        "marital_units": {
            f"{person}_marital": {"members": [person]} for person in _PEOPLE
        },
    }
    simulation = Simulation(situation=situation)
    variables = (
        "income_tax_before_credits",
        "income_tax_non_refundable_credits",
        "income_tax_capped_non_refundable_credits",
        "income_tax_before_refundable_credits",
        "income_tax_refundable_credits",
        "income_tax",
        *_PART_II_TAXES,
    )
    values = {
        variable: np.asarray(simulation.calculate(variable, YEAR), dtype=np.float64)
        for variable in variables
    }
    return {
        unit: {variable: float(values[variable][index]) for variable in variables}
        for index, unit in enumerate(_UNITS)
    }


def _line_22(unit: dict[str, float]) -> float:
    return (
        unit["income_tax_before_credits"]
        - unit["income_tax_capped_non_refundable_credits"]
    )


@pytest.mark.parametrize("name", sorted(_UNITS))
def test_capped_credits_are_the_credits_up_to_the_tax(tax_units, name):
    unit = tax_units[name]
    assert unit["income_tax_capped_non_refundable_credits"] == pytest.approx(
        min(
            unit["income_tax_before_credits"],
            unit["income_tax_non_refundable_credits"],
        ),
        abs=0.01,
    )
    assert -0.01 <= _line_22(unit) <= unit["income_tax_before_credits"] + 0.01


@pytest.mark.parametrize("name", sorted(_UNITS))
def test_engine_income_tax_adds_part_ii_taxes_and_nets_refundable_credits(
    tax_units, name
):
    unit = tax_units[name]
    before_refundable = _line_22(unit) + sum(unit[tax] for tax in _PART_II_TAXES)
    assert unit["income_tax_before_refundable_credits"] == pytest.approx(
        before_refundable, abs=0.01
    )
    assert unit["income_tax"] == pytest.approx(
        unit["income_tax_before_refundable_credits"]
        - unit["income_tax_refundable_credits"],
        abs=0.01,
    )


def test_positive_income_tax_misses_line_22_both_ways(tax_units):
    parent = tax_units["parent_unit"]
    assert _line_22(parent) > 0
    assert parent["income_tax_refundable_credits"] > 0
    assert max(parent["income_tax"], 0.0) < _line_22(parent)

    investor = tax_units["investor_unit"]
    assert investor["net_investment_income_tax"] > 0
    assert investor["income_tax"] == pytest.approx(
        _line_22(investor) + investor["net_investment_income_tax"], abs=0.01
    )
    assert investor["income_tax"] > _line_22(investor)

    retiree = tax_units["retiree_unit"]
    assert _line_22(retiree) == 0
    assert retiree["income_tax"] == 0
