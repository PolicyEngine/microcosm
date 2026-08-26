"""Guarantees for the UK local-geography calibration contract resource.

``uk/uk_local_geography_targets.json`` is the consumer-side selection contract
over Chronicle facts (chronicle#166 ruling: contracts live in Microcosm;
Chronicle is facts-only). Values resolve from a Chronicle consumer artifact at
build time. This resource stays value-free and closed over the corrected
wave-3 selector vocabulary used to author the area-grain local target surface.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources as importlib_resources

from microcosm.build.uk_runtime.local_targets import (
    load_uk_local_geography_contract,
    metric_names,
    metric_names_from_target_profile,
)

FORBIDDEN_VALUE_KEYS = {"aggregation", "operation", "registry", "target_value", "value"}
FORBIDDEN_RUNTIME_KEYS = {
    "callable",
    "command",
    "execute",
    "executable",
    "function",
    "import",
    "imports",
    "module",
    "python_code",
    "runtime_code",
    "script",
    "solver",
}
LOCAL_SELECTOR_KEYS = {"source_name", "source_measure_id", "record_set_spec_id"}


def _load() -> dict:
    return load_uk_local_geography_contract()


def _is_filter_predicate(node: Mapping) -> bool:
    return (
        "value" in node
        and "operator" in node
        and ("concept" in node or "variable" in node)
    )


def _carried_forbidden_keys(node, *, in_defaults: bool = False) -> set[str]:
    carried: set[str] = set()
    if isinstance(node, Mapping):
        forbidden = set(FORBIDDEN_VALUE_KEYS | FORBIDDEN_RUNTIME_KEYS)
        if _is_filter_predicate(node):
            forbidden.discard("value")
        if in_defaults:
            forbidden.discard("operation")
        carried.update(forbidden.intersection(node))
        for child in node.values():
            carried.update(_carried_forbidden_keys(child))
    elif isinstance(node, (list, tuple)):
        for child in node:
            carried.update(_carried_forbidden_keys(child))
    return carried


def test_uk_local_geography_targets_are_registered_in_the_country_package() -> None:
    package = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("country_package.json")
        .read_text()
    )

    paths = [
        row["path"] if isinstance(row, dict) else row for row in package["resources"]
    ]
    assert "uk_local_geography_targets.json" in paths


def test_uk_local_geography_targets_shape_and_profile_parity() -> None:
    resource = _load()

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == ["identity", "sum"]
    assert len(resource["targets"]) == 25

    parity = resource["profile_parity"]
    assert parity["source_profile_id"] == "uk_local_geography"
    assert parity["source_target_count"] == 25
    assert parity["contract_target_count"] == 25
    assert parity["corrected_rows"] == len(parity["corrected"]) == 25

    targets = {target["target_id"]: target for target in resource["targets"]}
    corrected_ids = {entry["target_id"] for entry in parity["corrected"]}
    assert corrected_ids == set(targets)
    for entry in parity["corrected"]:
        target = targets[entry["target_id"]]
        assert entry["corrected_selector"] == target["ledger_selector"]
        assert entry["profile_selector"]
        assert entry["reason"]


def test_uk_local_geography_targets_have_unique_target_ids() -> None:
    resource = _load()

    target_ids = [target["target_id"] for target in resource["targets"]]
    assert len(target_ids) == len(set(target_ids))


def test_uk_local_geography_targets_are_value_free_and_hook_free() -> None:
    resource = _load()

    carried = _carried_forbidden_keys(
        {key: value for key, value in resource.items() if key != "resolution_defaults"}
    )
    carried |= _carried_forbidden_keys(
        resource["resolution_defaults"], in_defaults=True
    )
    assert not carried, sorted(carried)
    assert resource["resolution_defaults"]["operation"] == "sum"


def test_uk_local_geography_targets_use_corrected_selector_vocabulary() -> None:
    resource = _load()
    targets = {target["target_id"]: target for target in resource["targets"]}

    for target in resource["targets"]:
        selector = target["ledger_selector"]
        assert selector
        assert set(selector) <= LOCAL_SELECTOR_KEYS, target["target_id"]
        assert "chronicle_selector" not in target
        assert set(target["bindings"]) == {"policyengine", "axiom"}

    assert targets["ons.age.0_10"]["ledger_selector"] == {
        "source_measure_id": "population",
        "record_set_spec_id": "uk.local_geography.population.age_0_10.v1",
    }
    assert "source_name" not in targets["ons.age.70_80"]["ledger_selector"]
    assert targets["ons.age.70_80"]["ledger_selector"]["record_set_spec_id"] == (
        "uk.local_geography.population.age_70_80.v1"
    )
    assert targets["dwp.universal_credit.households.3plus_children"][
        "ledger_selector"
    ] == {
        "source_name": "dwp",
        "source_measure_id": "universal_credit_households_by_children",
        "record_set_spec_id": "uk.local_geography.uc_households.children_3plus.v1",
    }
    assert targets["ons.tenure.private_rent"]["ledger_selector"] == {
        "source_measure_id": "households",
        "record_set_spec_id": "uk.local_geography.tenure.private_rent.v1",
    }
    assert targets["hmrc.employment_income.amount"]["ledger_selector"][
        "source_measure_id"
    ] == ["employment_income_count", "employment_income_mean"]
    assert "count_x_mean" in targets["hmrc.employment_income.amount"]["selector_note"]
    assert (
        targets["ons.equiv_net_income_bhc"]["ledger_selector"]["source_measure_id"]
        == "equivalised_net_income_before_housing_costs"
    )
    assert (
        targets["ons.equiv_housing_costs"]["bindings"]["policyengine"][
            "value_expression"
        ]
        == "equiv_hbai_household_net_income - equiv_hbai_household_net_income_ahc"
    )


def test_uk_local_geography_targets_preserve_metric_ordering_contract() -> None:
    resource = _load()

    assert (
        metric_names_from_target_profile(resource, "constituency")
        == metric_names("constituency")[:-1]
    )
    assert metric_names_from_target_profile(resource, "la") == metric_names("la")[:-1]
    assert len(metric_names_from_target_profile(resource, "constituency")) == 17
    assert len(metric_names_from_target_profile(resource, "la")) == 21
