"""Guarantees for the UK firm calibration contract resource."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources as importlib_resources

from microcosm.build.uk_runtime.firm_generation import (
    DEFAULT_UK_FIRM_TARGET_PROFILE,
    UK_FIRM_TARGET_IDS,
    load_uk_firms_contract,
    uk_firm_target_profile_from_mapping,
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
FIRM_SELECTOR_KEYS = {
    "source_name",
    "source_measure_id",
    "record_set_id",
    "groupby_dimension",
    "dimensions",
}


def _load() -> dict:
    return load_uk_firms_contract()


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


def test_uk_firms_targets_are_registered_in_the_country_package() -> None:
    package = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("country_package.json")
        .read_text()
    )

    paths = [
        row["path"] if isinstance(row, dict) else row for row in package["resources"]
    ]
    assert "uk_firms_targets.json" in paths


def test_uk_firms_targets_shape_and_profile_parity() -> None:
    resource = _load()

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == ["identity"]
    assert len(resource["targets"]) == 8
    assert resource["profile_parity"]["source_profile_id"] == "uk_firms"
    assert resource["profile_parity"]["source_target_count"] == 8
    assert resource["profile_parity"]["contract_target_count"] == 8
    assert resource["profile_parity"]["selector_rename"] == (
        "chronicle_selector -> ledger_selector"
    )


def test_uk_firms_targets_have_unique_target_ids_in_runtime_order() -> None:
    resource = _load()

    target_ids = [target["target_id"] for target in resource["targets"]]
    assert target_ids == list(UK_FIRM_TARGET_IDS.values())
    assert len(target_ids) == len(set(target_ids))


def test_uk_firms_targets_are_value_free_and_hook_free() -> None:
    resource = _load()

    carried = _carried_forbidden_keys(
        {key: value for key, value in resource.items() if key != "resolution_defaults"}
    )
    carried |= _carried_forbidden_keys(
        resource["resolution_defaults"], in_defaults=True
    )
    assert not carried, sorted(carried)


def test_uk_firms_targets_use_ledger_selector_vocabulary() -> None:
    resource = _load()

    for target in resource["targets"]:
        selector = target["ledger_selector"]
        assert selector
        assert "chronicle_selector" not in target
        assert set(selector) <= FIRM_SELECTOR_KEYS, target["target_id"]
        assert set(target["bindings"]) == {"microcosm", "axiom"}


def test_default_uk_firm_target_profile_matches_committed_contract() -> None:
    committed = uk_firm_target_profile_from_mapping(_load())

    assert DEFAULT_UK_FIRM_TARGET_PROFILE == committed
