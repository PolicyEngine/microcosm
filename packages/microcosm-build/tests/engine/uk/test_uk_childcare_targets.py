from __future__ import annotations

import inspect
import json
from importlib import resources as importlib_resources


def _system():
    from policyengine_uk import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def test_childcare_target_formula_contracts_match_the_installed_engine() -> None:
    system = _system()
    universal_source = inspect.getsource(
        system.variables["universal_childcare_entitlement_eligible"].formula
    )
    tax_free_source = inspect.getsource(system.variables["tax_free_childcare"].formula)
    # From 2.165.1 the top-up is one helper, shared with the comparison that
    # lets a family keep Tax-Free Childcare over a smaller UC award
    # (pe-uk#2754), and the helper reads the spend the build routes.
    from policyengine_uk.variables.gov.hmrc.tax_free_childcare import _top_up

    top_up_source = inspect.getsource(_top_up.tax_free_childcare_top_up)
    without_uc_source = inspect.getsource(
        system.variables["tax_free_childcare_without_universal_credit"].formula
    )

    # From 2.128.1 a child on the working parent entitlement keeps the
    # universal hours (pe-uk#2177), so the universal-only row must leave the
    # working-parent children out itself.
    assert "has_extended_childcare" not in universal_source
    universal_only = next(
        target
        for target in json.loads(
            importlib_resources.files("microcosm.build.uk")
            .joinpath("uk_population_targets.json")
            .read_text(encoding="utf-8")
        )["targets"]
        if target["target_id"] == "dfe.funded_childcare.universal_only_children"
    )
    assert {
        "variable": "dfe_spring_working_parent",
        "equals": False,
    } in universal_only["bindings"]["policyengine"]["filters"]
    assert "tax_free_childcare_top_up(" in tax_free_source
    assert "tax_free_childcare_top_up(" in without_uc_source
    assert "tax_free_childcare_spend_routed_share" in top_up_source

    # The spring-term measures (dfe_spring_term.py) count a family that does
    # not claim the universal or targeted hours on their own when the engine
    # allocates them from the child's gross working-parent hours. Pin that
    # allocation, so an engine that moved it would fail here.
    for name in ("universal_childcare_entitlement", "targeted_childcare_entitlement"):
        source = inspect.getsource(system.variables[name].formula)
        assert "extended_childcare_entitlement_total_per_child" in source, name

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
    # María's ruling (2026-09-20, microcosm#965): the doctrine rule is family_equal.
    assert UK_NATIONAL_SOLVE_DOCTRINE.target_weight_rule == "family_equal"
