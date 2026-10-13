"""The UK cross-grain declarations are exhaustive and enforced (#1123)."""

from __future__ import annotations

import copy
import json
from importlib import resources as importlib_resources

import pytest

from microcosm.build.cross_grain import _measurement_signature
from microcosm.build.uk_runtime.cross_grain_declarations import (
    assert_uk_cross_grain_coverage,
    load_uk_cross_grain_declarations,
    uk_cross_grain_bridges,
    uk_cross_grain_coverage_violations,
    uk_cross_grain_grain,
)
from microcosm.build.uk_runtime.ledger_targets import (
    UK_CROSS_GRAIN_BRIDGES,
    UK_CROSS_GRAIN_RULE,
    uk_cross_grain_contract_signatures,
    uk_cross_grain_coverage_receipt,
)


def _contract() -> dict[str, dict]:
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    return {target["target_id"]: target for target in payload["targets"]}


def test_the_committed_contract_passes_the_coverage_check():
    # microcosm#1123 closed every gap: the check tolerates none.
    receipt = uk_cross_grain_coverage_receipt()
    assert receipt["checked_targets"] == len(_contract())


def test_the_runtime_bridges_are_the_declared_bridges():
    assert UK_CROSS_GRAIN_BRIDGES == uk_cross_grain_bridges()
    for entry in load_uk_cross_grain_declarations()["bridges"]:
        assert entry["reason"].strip()


def test_a_local_target_with_no_route_is_refused():
    contract = _contract()
    added = copy.deepcopy(contract["ons.tenure.owned_outright"])
    added["target_id"] = "ons.tenure.shared_ownership"
    added["measurement"]["filters"] = [
        {"concept": "uk.household.tenure", "equals": "shared_ownership"}
    ]
    contract["ons.tenure.shared_ownership"] = added
    with pytest.raises(ValueError, match="ons.tenure.shared_ownership"):
        assert_uk_cross_grain_coverage(contract)


def test_an_overlap_across_grains_must_be_declared():
    contract = _contract()
    added = copy.deepcopy(contract["hmrc.cgt.taxpayers_total"])
    added["target_id"] = "hmrc.cgt.taxpayers_wales"
    added["geography_levels"] = ["local_authority"]
    added["measurement"]["filters"] = [
        *added["measurement"]["filters"],
        {"concept": "uk.geography.country", "equals": "WALES"},
    ]
    contract["hmrc.cgt.taxpayers_wales"] = added
    violations = uk_cross_grain_coverage_violations(contract)
    assert {
        "kind": "undeclared_overlap",
        "target_id": "hmrc.cgt.taxpayers_total,hmrc.cgt.taxpayers_wales",
        "where": "contract",
    } in violations


def test_a_local_route_on_a_covered_target_is_stale():
    declarations = copy.deepcopy(dict(load_uk_cross_grain_declarations()))
    declarations["local_routes"] = {
        **declarations["local_routes"],
        "ons.census.households": {"route": "no_higher_control", "reason": "x"},
    }
    violations = uk_cross_grain_coverage_violations(_contract(), declarations)
    assert {
        "kind": "stale_local_route",
        "target_id": "ons.census.households",
        "where": "local_routes",
    } in violations


@pytest.mark.parametrize(
    ("metadata", "level", "geography_id", "grain"),
    [
        ({}, "country", "K02000001", "country"),
        ({}, "country", "E92000001", "nation"),
        ({}, "country", "W92000004", "nation"),
        ({"cross_grain_grain": "region"}, "country", "W92000004", "region"),
        ({}, "region", "E12000007", "region"),
    ],
)
def test_country_level_cells_on_nation_codes_sit_at_the_nation_grain(
    metadata, level, geography_id, grain
):
    assert uk_cross_grain_grain(metadata, level, geography_id) == grain


def test_signature_incomplete_targets_never_exact_match_another_target():
    """The GB two-child-limit rows and the Scottish under-one UC row share a
    measurement block; with the nation grain the Scottish row would sit under
    the GB rows as their part if the operator grouped them."""

    raw = _contract()
    fields = UK_CROSS_GRAIN_RULE.signature_fields
    assert _measurement_signature(
        raw["dwp.uc.two_child_limit.households_affected"], fields
    ) == _measurement_signature(raw["dwp.uc.scotland_households_child_under_1"], fields)
    grouped = uk_cross_grain_contract_signatures()
    incomplete = {
        target_id
        for group in load_uk_cross_grain_declarations()["signature_incomplete"]
        for target_id in group["targets"]
    }
    signatures = {
        target_id: _measurement_signature(target, fields)
        for target_id, target in grouped.items()
    }
    for target_id in incomplete:
        twins = [
            other
            for other, signature in signatures.items()
            if other != target_id and signature == signatures[target_id]
        ]
        assert twins == [], target_id
