"""CenTax's main-source measure compares every source of the engine's total income.

CenTax's employees are taxpayers whose main source of income is employment, so
the measure must weigh pay against each source the engine counts in
``total_income`` (ITA 2007 s. 23). A source left out cannot outrank pay, and a
parent whose largest income is that source would count as an employee.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES,
    CENTAX_EMPLOYEE_SOURCE_INPUTS,
)


def _components(variables, name):
    """``name``'s leaf components, ``social_security_income`` taken apart."""

    adds = getattr(variables[name], "adds", None) or ()
    if name == "total_income" or name == "social_security_income":
        return [leaf for part in adds for leaf in _components(variables, part)]
    return [name]


def test_the_main_source_measure_compares_every_total_income_component():
    from policyengine_uk import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    components = _components(variables, "total_income")
    compared = [
        CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES.get(name, name)
        for name in CENTAX_EMPLOYEE_SOURCE_INPUTS
    ]
    assert sorted(compared) == sorted(components)
    assert len(set(compared)) == len(compared)
    assert compared[0] == "employment_income"
