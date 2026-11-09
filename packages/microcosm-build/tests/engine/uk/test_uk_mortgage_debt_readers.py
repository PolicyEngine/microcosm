"""The secured debts nest inside ``mortgage_debt``, and nothing counts both (microcosm#1095).

The WAS stage draws the debt secured on other residential and on
non-residential property inside the mortgages on other property, which
``mortgage_debt`` holds beside the main-residence mortgage (``was_wealth``).
policyengine-uk subtracts ``mortgage_debt`` in ``net_wealth``, and each secured
debt from its own property's UC and Pension Credit capital, so each measure
counts a debt once. María's ruling of 2026-10-25 keeps ``mortgage_debt`` as all
secured debt; this test fails if a formula of the locked engine starts to read
it beside a secured debt, which would count the nested debt twice.
"""

from __future__ import annotations

import inspect
import re

from microcosm.build.uk_runtime.was_wealth import (
    UK_WAS_DERIVED_TOTALS,
    UK_WAS_OTHER_MORTGAGE_COLUMN,
    UK_WAS_PROPERTY_SECURED_DEBT,
    UK_WAS_SECURED_DEBT_WITHIN,
)

_QUOTED_NAME = re.compile(r"[\"']([a-z][a-z0-9_]*)[\"']")


def test_the_secured_debts_sit_inside_mortgage_debt() -> None:
    assert UK_WAS_SECURED_DEBT_WITHIN == UK_WAS_OTHER_MORTGAGE_COLUMN
    assert UK_WAS_OTHER_MORTGAGE_COLUMN in UK_WAS_DERIVED_TOTALS["mortgage_debt"]


def test_no_engine_formula_reads_mortgage_debt_beside_a_secured_debt() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    secured = set(UK_WAS_PROPERTY_SECURED_DEBT)
    readers: set[str] = set()
    both: set[str] = set()
    for name, variable in CountryTaxBenefitSystem().variables.items():
        for formula in variable.formulas.values():
            names = set(_QUOTED_NAME.findall(inspect.getsource(formula)))
            if "mortgage_debt" in names:
                readers.add(name)
                if names & secured:
                    both.add(name)
    # The scan sees the engine's reads: net_wealth subtracts mortgage_debt.
    assert "net_wealth" in readers
    assert both == set()
