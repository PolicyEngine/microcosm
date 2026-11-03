"""The registry flags are policyengine-us inputs the engine cannot compute.

Each registry flag must exist in the pinned engine as an input variable with
no formula, a boolean defaulting to ``False`` on the registry's entity, and
each listed consumer must read it. If a later engine gives a flag a formula,
this fails and the registry entry should be retired rather than kept writing
a column the engine would treat as formula-owned.
"""

from __future__ import annotations

import inspect
import json
from importlib.resources import files

import pytest

from microcosm.build.us_runtime.local_jurisdictions import (
    US_LOCAL_JURISDICTION_REGISTRY_RESOURCE,
    packaged_us_local_jurisdiction_registry,
)

REGISTRY = packaged_us_local_jurisdiction_registry()
DOCUMENT = json.loads(
    files("microcosm.build.us_runtime")
    .joinpath(US_LOCAL_JURISDICTION_REGISTRY_RESOURCE)
    .read_text()
)
CONSUMERS = {
    item["flag"]: item["engine_consumers"] for item in DOCUMENT["jurisdictions"]
}


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.mark.parametrize("item", REGISTRY.jurisdictions, ids=lambda item: item.flag)
def test_registry_flag_is_an_input_only_boolean_on_its_entity(system, item) -> None:
    variable = system.variables[item.flag]

    assert variable.is_input_variable()
    assert not getattr(variable, "formulas", None)
    assert variable.entity.key == item.entity
    assert variable.value_type is bool
    assert variable.default_value is False
    assert str(variable.definition_period).lower() == "year"


@pytest.mark.parametrize("item", REGISTRY.jurisdictions, ids=lambda item: item.flag)
def test_each_listed_consumer_reads_the_flag(system, item) -> None:
    for consumer in CONSUMERS[item.flag]:
        variable = system.variables[consumer]
        defined_for = getattr(variable, "defined_for", None)
        reads_in_formula = any(
            f'"{item.flag}"' in inspect.getsource(formula)
            for formula in variable.formulas.values()
        )
        assert reads_in_formula or getattr(defined_for, "name", defined_for) == (
            item.flag
        ), consumer
