from __future__ import annotations

import inspect
import json
from importlib import resources as importlib_resources

import pytest

pytestmark = pytest.mark.requires_uk


def _system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def test_childcare_target_formula_contracts_match_the_installed_engine() -> None:
    system = _system()
    universal_source = inspect.getsource(
        system.variables["universal_childcare_entitlement_eligible"].formula
    )
    tax_free_source = inspect.getsource(system.variables["tax_free_childcare"].formula)

    assert "~has_extended_childcare" in universal_source
    assert "tax_free_childcare_spend_routed_share" in tax_free_source

    for name in (
        "is_child_receiving_tax_free_childcare",
        "is_child_receiving_extended_childcare",
        "is_child_receiving_targeted_childcare",
        "is_child_receiving_universal_childcare",
    ):
        variable = system.variables[name]
        assert variable.entity.key == "person", name
        assert variable.value_type is bool, name


def test_extended_childcare_age_one_hours_change_on_2025_01_01() -> None:
    system = _system()

    before = system.parameters("2024-12-31").gov.dfe.extended_childcare_entitlement
    after = system.parameters("2025-01-01").gov.dfe.extended_childcare_entitlement

    assert before.hours.calc(1) == 0
    assert after.hours.calc(1) == 15


def test_childcare_activation_preserves_existing_calibration_gates() -> None:
    from microcosm.build.uk_runtime.national_doctrine import (
        UK_NATIONAL_SOLVE_DOCTRINE,
    )

    gates = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("gates.json")
        .read_text()
    )["gates"]
    by_id = {gate["id"]: gate for gate in gates}

    assert by_id["uk_target_fit"]["criticality"] == "release_blocking"
    assert by_id["uk_target_fit"]["parameters"]["max_abs_relative_error"] == 0.25
    assert by_id["uk_calibration_reference_coverage"]["criticality"] == (
        "release_blocking"
    )
    assert UK_NATIONAL_SOLVE_DOCTRINE.target_weight_rule == "uniform"
