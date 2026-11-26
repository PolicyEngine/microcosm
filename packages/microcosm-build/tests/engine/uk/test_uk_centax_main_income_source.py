"""CenTax's main-source measure compares every component of the engine's ANI.

CenTax's employees are taxpayers whose main source of income is employment, so
the measure must weigh pay against each source the engine counts in adjusted
net income: the components ``gov.hmrc.income_tax.adjusted_net_income_components``
lists, which ``total_income_charged_to_tax`` (ITA 2007 s. 23) also sums. A
source left out cannot outrank pay, and a parent whose largest income is that
source would count as an employee.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_EMPLOYEE_SOURCE_COMPONENTS,
    CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES,
    CENTAX_EMPLOYEE_SOURCE_INPUTS,
)
from microcosm.build.uk_runtime.ledger_targets import UK_TARGET_REFERENCE_PERIOD


def test_the_main_source_measure_compares_every_ani_component():
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    components = system.parameters.gov.hmrc.income_tax.adjusted_net_income_components(
        f"{UK_TARGET_REFERENCE_PERIOD}-01-01"
    )
    assert sorted(CENTAX_EMPLOYEE_SOURCE_COMPONENTS) == sorted(components)
    assert len(set(CENTAX_EMPLOYEE_SOURCE_INPUTS)) == len(CENTAX_EMPLOYEE_SOURCE_INPUTS)
    assert CENTAX_EMPLOYEE_SOURCE_INPUTS[0] == (
        "employment_income_with_benefits_in_kind_annual"
    )


def test_the_measure_takes_social_security_and_pensions_as_the_engine_adds_them():
    from policyengine_uk import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    assert variables["taxable_social_security_income"].adds == [
        "social_security_income"
    ]
    compared = [
        CENTAX_EMPLOYEE_SOURCE_ENGINE_NAMES.get(name, name)
        for name in CENTAX_EMPLOYEE_SOURCE_COMPONENTS["taxable_social_security_income"]
    ]
    assert sorted(compared) == sorted(variables["social_security_income"].adds)
    assert variables["taxable_pension_income"].adds == ["private_pension_income"]
    # Employment is pay plus the company car and car fuel benefits, which the
    # engine's taxable employment income counts and keeps apart from pay.
    assert variables["benefits_in_kind"].adds == [
        "company_car_benefit",
        "car_fuel_benefit",
    ]
