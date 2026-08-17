"""Guarantees for the UK national calibration contract resource.

``uk/uk_national_targets.json`` is the consumer-side selection contract over
Chronicle facts (chronicle#166 ruling: contracts live in Microcosm; Chronicle
is facts-only). Values resolve from a Chronicle consumer artifact at build
time — the resource itself must stay value-free, hook-free, and closed-world
on its declarative counterfactual vocabulary. Compilation is microcosm#622.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources as importlib_resources

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
BINDING_KINDS = {
    "input_substitution_counterfactual",
    "parameter_gated_threshold",
    "baseline_flag_crosstab",
}
PROJECTION_FAMILIES = {"obr", "slc_borrowers", "scotgov_social_security"}


def _load() -> dict:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text()
    )
    return json.loads(payload)


def _is_filter_predicate(node: Mapping) -> bool:
    return "value" in node and "operator" in node and (
        "concept" in node or "variable" in node
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


def test_uk_national_targets_is_registered_in_the_country_package():
    package = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("country_package.json")
        .read_text()
    )

    assert "uk_national_targets.json" in package["resources"]


def test_uk_national_targets_shape_and_accounting():
    resource = _load()

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == ["identity"]
    assert len(resource["targets"]) == 186

    parity = resource["registry_parity"]
    assert parity["pinned_ref"] == "ebf733c"
    assert parity["mapped_rows"] + parity["excluded_rows"] == parity["registry_rows"]
    assert len(parity["mapped"]) == parity["mapped_rows"]
    assert len(parity["excluded"]) == parity["excluded_rows"]
    mapped_target_ids = set(parity["mapped"].values())
    declared_target_ids = {target["target_id"] for target in resource["targets"]}
    assert mapped_target_ids <= declared_target_ids


def test_uk_national_targets_are_value_free_and_hook_free():
    resource = _load()

    carried = _carried_forbidden_keys(
        {key: value for key, value in resource.items() if key != "resolution_defaults"}
    )
    carried |= _carried_forbidden_keys(
        resource["resolution_defaults"], in_defaults=True
    )
    assert not carried, sorted(carried)
    assert resource["resolution_defaults"]["operation"] == "sum"


def test_uk_national_targets_declare_selectors_and_closed_world_kinds():
    resource = _load()

    metric_names: list[str] = []
    for target in resource["targets"]:
        assert target["ledger_selector"], target["target_id"]
        assert "assertion" not in target["ledger_selector"], target["target_id"]
        binding = target["bindings"]["policyengine"]
        metric_names.append(binding["metric_name"])
        kind = binding.get("kind")
        assert kind is None or kind in BINDING_KINDS, target["target_id"]
        if target["family"] in PROJECTION_FAMILIES:
            assert target.get("assertion_policy") == "allow_source_projection", (
                target["target_id"]
            )
        else:
            assert "assertion_policy" not in target, target["target_id"]
    assert len(metric_names) == len(set(metric_names))
