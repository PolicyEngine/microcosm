"""An illustrative change to the Scottish rates on property income (microcosm#1326 B).

policyengine-uk taxes a Scottish taxpayer's property income at the Scottish
rates, stacked on top of their other non-savings income
(``property_income_tax``). It has no separate Scottish property rate. Adding
``delta`` to every Scottish rate on that slice adds exactly ``delta`` times the
slice, ``taxable_property_income_after_allowances``, whichever bands it
crosses. That is what this reform charges.

The finance-cost reduction is given at the UK property basic rate
(``property_finance_cost_relief``). By default it stays there. With
``relief_follows``, Scottish taxpayers' reduction rises by ``delta`` too,
within the same Step 6 limit the engine applies.
"""

from __future__ import annotations

from policyengine_core.reforms import Reform
from policyengine_uk.model_api import (
    GBP,
    YEAR,
    Person,
    Variable,
    add,
    max_,
    min_,
    where,
)
from policyengine_uk.variables.gov.hmrc.income_tax.liability.property_income_tax import (
    property_income_tax as _engine_property_income_tax,
)


def scottish_property_rate_reform(
    delta: float = 0.01, *, relief_follows: bool = False
) -> type[Reform]:
    """A reform adding ``delta`` to the Scottish rates charged on property income."""

    class property_income_tax(Variable):  # noqa: N801 (engine variable names)
        value_type = float
        entity = Person
        label = "Income tax on property income"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):  # noqa: N805
            charged = _engine_property_income_tax.formula(person, period, parameters)
            scottish = person("pays_scottish_income_tax", period)
            property_slice = person("taxable_property_income_after_allowances", period)
            return charged + where(scottish, delta * property_slice, 0)

    class property_finance_cost_relief(Variable):  # noqa: N801
        value_type = float
        entity = Person
        label = "residential property finance costs tax reduction"
        definition_period = YEAR
        unit = GBP

        def formula(person, period, parameters):  # noqa: N805
            income_tax = parameters(period).gov.hmrc.income_tax
            scottish = person("pays_scottish_income_tax", period)
            rate = income_tax.rates.property.basic + where(scottish, delta, 0)
            reduction = rate * person("property_finance_costs_relieved", period)
            other_reductions = [
                variable
                for variable in income_tax.income_tax_subtractions
                if variable != "property_finance_cost_relief"
            ]
            tax_left = max_(
                0,
                person("income_tax_pre_charges", period)
                - add(person, period, other_reductions),
            )
            return min_(reduction, tax_left)

    class ScottishPropertyRateReform(Reform):
        def apply(self):
            self.update_variable(property_income_tax)
            if relief_follows:
                self.update_variable(property_finance_cost_relief)

    return ScottishPropertyRateReform
