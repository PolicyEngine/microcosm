"""Country scope of the benefit rows bound against DWP's lines (microcosm#1095, uk-data#490).

OBR table 4.9 reports "DWP social security" lines, whose figures cover Great Britain (DWP benefit
expenditure and caseload tables, Spring 2026, Notes 3), with Northern Ireland social security on
rows of its own. DWP reports Attendance Allowance, Carer's Allowance, DLA/PIP and Winter Fuel
Payment for England and Wales only since executive competence moved to the Scottish Government.
Each row binds the model column over the same households the publisher counts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import microcosm.build
from microcosm.build.uk_runtime.local_targets import load_uk_population_contract
from test_support.paths import paths_for

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
# DWP's PIP daily-living caseload (FOI2025/24990), which Chronicle stamps
# K04000001: diagnostics on obr.pip, not targets (microcosm#1095).
PIP_CASELOAD_ROWS = (
    "dwp.pip.daily_living_standard_claimants",
    "dwp.pip.daily_living_enhanced_claimants",
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


def test_obr_pip_binds_dla_and_pip_together() -> None:
    """OBR's line is "Disability living allowance and personal independence payments"."""

    binding = _targets()["obr.pip"]["bindings"]["policyengine"]
    assert binding["value_expression"] == "pip + dla"
    assert "value_variable" not in binding


def test_pip_caseload_rows_are_england_and_wales_diagnostics_on_obr_pip() -> None:
    """The two rows resolve at K04000001, the geography Chronicle stamps DWP's
    caseload with, and fit; fitting them breaks the weight-ratio fence, so
    they sit beside obr.pip as diagnostics (María's ruling of 2026-10-10,
    microcosm#1095)."""

    from microcosm.build.uk_runtime.ledger_targets import (
        UK_REQUIRED_TARGET_DIAGNOSTICS,
    )

    contract = load_uk_population_contract()
    target_ids = {target["target_id"] for target in contract["targets"]}
    assert not target_ids.intersection(PIP_CASELOAD_ROWS)
    assert UK_REQUIRED_TARGET_DIAGNOSTICS["obr.pip"] == PIP_CASELOAD_ROWS
    parity = contract["registry_parity"]
    for target_id in PIP_CASELOAD_ROWS:
        assert target_id not in parity["scope_target_ids"]
        declaration = contract["diagnostic_references"][target_id]
        assert declaration["attach_to_target"] == "obr.pip"
        assert declaration["required_period_type"] == "month"
        assert declaration["required_assertion"] == "observation"
        reference = declaration["reference"]
        assert reference["period"] == "2025-01"
        assert reference["period_match_policy"] == "exact"
        assert reference["ledger_selector"]["geography_id"] == "K04000001"
        assert reference["ledger_fact_key"]
        assert reference["measure"] in parity["excluded"]
        assert reference["measure"] not in parity["mapped"]


def test_pip_caseload_diagnostics_carry_the_january_2025_caseload() -> None:
    """Feed-gated: each diagnostic resolves its exact January 2025 fact."""

    from microcosm.build.uk_runtime.ledger_targets import (
        _attached_diagnostic_metadata,
    )

    root = paths_for("microcosm-build").repository
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = (
        Path(configured) if configured else root / ".codex-work/consumer_facts_uk.jsonl"
    )
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    with feed.open(encoding="utf-8") as handle:
        facts = tuple(
            json.loads(line)
            for line in handle
            if '"dwp.pip_daily_living_claimants"' in line
        )

    metadata = _attached_diagnostic_metadata(facts, "obr.pip")

    for rate, value in (("standard", 1_283_000.0), ("enhanced", 1_608_000.0)):
        prefix = f"dwp_pip_daily_living_{rate}_diagnostic"
        assert metadata[f"{prefix}_role"] == "diagnostic_only_not_in_fit"
        assert metadata[f"{prefix}_status"] == "available"
        assert float(metadata[f"{prefix}_value_count"]) == value
        assert metadata[f"{prefix}_period"] == "2025-01"
