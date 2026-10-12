"""Tests split from packages/microcosm-build/tests/test_us_early_head_start_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_early_head_start_exclusion import *


def test_policyengine_2_2_1_requires_under_3_or_pregnant_person_input() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "2.2.1"
    system = CountryTaxBenefitSystem()
    evidence = _entry()["evidence"]["policyengine_variable"]
    assert evidence == {
        "version": "2.2.1",
        "entity": "person",
        "definition_period": "year",
        "value_type": "bool",
        "default": True,
        "eligibility_domain": (
            "age < 3 or is_pregnant, plus Head Start income or categorical eligibility"
        ),
    }
    variable = system.variables["takes_up_early_head_start_if_eligible"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.value_type is bool
    assert variable.default_value is True

    formula = system.variables["is_early_head_start_eligible"].get_formula("2024")
    assert formula is not None
    source = inspect.getsource(formula)
    assert "age < p.early_head_start.age_limit" in source
    assert "is_pregnant" in source
    assert system.parameters("2024").gov.hhs.head_start.early_head_start.age_limit == 3
