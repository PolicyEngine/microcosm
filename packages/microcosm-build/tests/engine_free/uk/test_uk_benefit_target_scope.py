"""Country scope of the benefit rows bound against DWP's lines (microcosm#1095, uk-data#490).

OBR table 4.9 reports "DWP social security" lines, whose figures cover Great Britain (DWP benefit
expenditure and caseload tables, Spring 2026, Notes 3), with Northern Ireland social security on
rows of its own. DWP reports Attendance Allowance, Carer's Allowance, DLA/PIP and Winter Fuel
Payment for England and Wales only since executive competence moved to the Scottish Government.
Each row binds the model column over the same households the publisher counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import microcosm.build
from microcosm.build.uk_runtime.local_targets import load_uk_population_contract

GB_REGIONS = {
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
}
GB_COUNTRIES = {"ENGLAND", "WALES", "SCOTLAND"}
GREAT_BRITAIN_CODE = "K03000001"
GREAT_BRITAIN_ROWS = (
    "obr.pension_credit",
    "obr.esa",
    "obr.jobseekers_allowance",
    "obr.universal_credit",
    "obr.universal_credit_in_cap",
    "obr.universal_credit_outside_cap",
    "obr.housing_benefit",
    "obr.statutory_maternity_pay",
    "dwp.esa_claimants",
    "dwp.esa_contrib_claimants",
    "dwp.esa_income_claimants",
    "dwp.jsa_claimants",
)
ENGLAND_AND_WALES_ROWS = (
    "obr.attendance_allowance",
    "obr.carers_allowance",
    "obr.pip",
    "obr.winter_fuel_allowance",
)
# OBR table 4.9 concepts reported inside "DWP social security".
DWP_SOCIAL_SECURITY_CONCEPTS = {
    "obr.attendance_allowance",
    "obr.carers_allowance",
    "obr.dla_and_pip",
    "obr.housing_benefit",
    "obr.incapacity_benefits",
    "obr.jobseekers_allowance",
    "obr.pension_credit",
    "obr.statutory_maternity_pay",
    "obr.universal_credit_in_cap",
    "obr.universal_credit_outside_cap",
    "obr.winter_fuel_payment",
    "obr.state_pension",
}


def _targets() -> dict[str, dict]:
    return {
        target["target_id"]: target
        for target in load_uk_population_contract()["targets"]
    }


def _concepts(target: dict) -> set[str]:
    concept = target.get("ledger_selector", {}).get("source_concept")
    if concept is None:
        return set()
    return {concept} if isinstance(concept, str) else set(concept)


def _country_filter(target: dict) -> dict:
    filters = [
        predicate
        for predicate in target["measurement"].get("filters", ())
        if predicate.get("concept") == "uk.geography.country"
    ]
    assert len(filters) == 1, target["target_id"]
    return filters[0]


def _condition(target: dict) -> dict:
    binding = target["bindings"]["policyengine"]
    conditions = binding.get("household_conditions", ())
    assert len(conditions) == 1, target["target_id"]
    condition = conditions[0]
    assert condition["entity"] == "household", target["target_id"]
    assert condition["reduce"] == "any", target["target_id"]
    assert condition["operator"] == "in", target["target_id"]
    assert condition["map_to"] == binding.get("from_entity", "household")
    return condition


def test_dwp_social_security_rows_bind_great_britain_households() -> None:
    targets = _targets()
    for target_id in GREAT_BRITAIN_ROWS:
        target = targets[target_id]
        condition = _condition(target)
        assert condition["variable"] == "region", target_id
        assert set(condition["value"]) == GB_REGIONS, target_id
        assert len(condition["value"]) == len(GB_REGIONS), target_id
        assert set(_country_filter(target)["value"]) == {
            "england",
            "scotland",
            "wales",
        }, target_id


def test_devolved_benefit_rows_bind_england_and_wales_households() -> None:
    targets = _targets()
    for target_id in ENGLAND_AND_WALES_ROWS:
        target = targets[target_id]
        condition = _condition(target)
        assert condition["variable"] == "country", target_id
        assert condition["value"] == ["ENGLAND", "WALES"], target_id
        assert set(_country_filter(target)["value"]) == {"england", "wales"}, target_id


def test_every_dwp_social_security_row_is_scoped() -> None:
    """A row added against a DWP line cannot bind the UK column by default."""

    scoped = set(GREAT_BRITAIN_ROWS) | set(ENGLAND_AND_WALES_ROWS)
    for target_id, target in _targets().items():
        if _concepts(target) & DWP_SOCIAL_SECURITY_CONCEPTS:
            assert target_id in scoped, target_id
            assert target["bindings"]["policyengine"].get("household_conditions"), (
                target_id
            )


def _scoped_to_great_britain(binding: dict) -> bool:
    for predicate in (
        *binding.get("household_conditions", ()),
        *binding.get("filters", ()),
    ):
        variable = predicate.get("variable")
        operator = predicate.get("operator")
        value = predicate.get("value")
        values = set(value) if isinstance(value, list) else {value}
        allowed = {"region": GB_REGIONS, "country": GB_COUNTRIES}.get(variable)
        if allowed is None or not values:
            continue
        if operator in ("in", "==") and values <= allowed:
            return True
        if operator in ("!=", "not_in") and "NORTHERN_IRELAND" in values:
            return True
    return False


def test_every_active_great_britain_fact_binds_great_britain_households() -> None:
    """A row whose fact is pinned to Great Britain never measures the UK column."""

    membership = json.loads(
        (
            Path(microcosm.build.__file__).parent
            / "uk"
            / "target_reference_membership.json"
        ).read_text()
    )
    targets = _targets()
    unscoped = sorted(
        target_id
        for target_id, pin in membership["geography_pins"].items()
        if pin.get("geography_id") == GREAT_BRITAIN_CODE
        and membership["targets"].get(target_id, {}).get("status") == "active"
        and not _scoped_to_great_britain(targets[target_id]["bindings"]["policyengine"])
    )
    assert unscoped == []
