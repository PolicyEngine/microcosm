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

from microcosm.build.uk_runtime.fiscal_targets import UK_CGT_TARGET_SPECS

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
SELECTOR_KEYS = {
    "source_name",
    "source_concept",
    "source_measure_id",
    "groupby_dimension",
    "dimensions",
    "dimension_values",
}
POLICYENGINE_BINDING_KEYS = {
    "affected_flag_variable",
    "band",
    "band_period_factor",
    "count_of",
    "filters",
    "folded_into",
    "from_entity",
    "gate_comparison",
    "gate_parameter",
    "gated_variable",
    "groupby_variable",
    "household_conditions",
    "kind",
    "map_to",
    "metric_name",
    "notes",
    "output_delta",
    "output_variable",
    "reduce",
    "source_lines",
    "threshold_price_base_year",
    "value_expression",
    "value_variable",
    "zeroed_input",
}
ASSERTION_POLICIES = {"allow_source_projection"}
BINDING_REDUCERS = {"any"}
PREDICATE_REDUCERS = {"any", "any_child_under", "count", "sum"}
PREDICATE_ENTITIES = {"person", "benunit", "household"}


def _load() -> dict:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text()
    )
    return json.loads(payload)


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


def test_uk_national_targets_is_registered_in_the_country_package():
    package = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("country_package.json")
        .read_text()
    )

    paths = [
        row["path"] if isinstance(row, dict) else row
        for row in package["resources"]
    ]
    assert "uk_national_targets.json" in paths


def test_uk_national_targets_shape_and_accounting():
    resource = _load()

    assert resource["country"] == "uk"
    assert resource["allowed_value_operations"] == ["identity"]
    assert len(resource["targets"]) == 189

    parity = resource["registry_parity"]
    assert parity["pinned_ref"] == "12a1e028afeef08d8b2d74ee03fd9de3a78b2dd3"
    assert parity["pinned_version"] == "1.56.16"
    assert parity["mapped_rows"] + parity["excluded_rows"] == parity["registry_rows"]
    assert len(parity["mapped"]) == parity["mapped_rows"]
    assert len(parity["excluded"]) == parity["excluded_rows"]
    mapped_target_ids = set(parity["mapped"].values())
    unmapped_declarations = parity["unmapped_declarations"]
    assert set(mapped_target_ids).isdisjoint(unmapped_declarations)
    declared_target_ids = {target["target_id"] for target in resource["targets"]}
    assert mapped_target_ids | set(unmapped_declarations) == declared_target_ids
    assert all(reason for reason in unmapped_declarations.values())
    assert len(mapped_target_ids) == 184
    assert len(unmapped_declarations) == 5
    suppressed_ancestors = parity["suppressed_ancestors"]
    assert len(suppressed_ancestors) == 5
    assert set(suppressed_ancestors).isdisjoint(parity["mapped"])
    assert set(suppressed_ancestors).isdisjoint(parity["excluded"])
    assert {
        entry["contract_target_id"] for entry in suppressed_ancestors.values()
    } <= declared_target_ids
    assert all(
        "410-Gone" in entry["rationale"] for entry in suppressed_ancestors.values()
    )


def test_uk_national_targets_split_terminal_sex_age_bands():
    resource = _load()
    parity = resource["registry_parity"]

    assert parity["mapped"]["ons/female_85_90"] == "ons.population.female_85_89"
    assert parity["mapped"]["ons/male_85_90"] == "ons.population.male_85_89"
    assert {
        "ons.population.female_90_plus",
        "ons.population.male_90_plus",
    } <= set(parity["unmapped_declarations"])
    assert "single-age-90 share" in parity["accounting_notes"][
        "ons_terminal_sex_band_split"
    ]

    female_85_89 = _target_by_id(resource, "ons.population.female_85_89")
    female_90_plus = _target_by_id(resource, "ons.population.female_90_plus")
    male_85_89 = _target_by_id(resource, "ons.population.male_85_89")
    male_90_plus = _target_by_id(resource, "ons.population.male_90_plus")

    assert female_85_89["ledger_selector"]["dimension_values"] == {
        "sex": "female",
        "age": [85, 86, 87, 88, 89],
    }
    assert male_85_89["ledger_selector"]["dimension_values"] == {
        "sex": "male",
        "age": [85, 86, 87, 88, 89],
    }
    assert female_90_plus["ledger_selector"]["dimension_values"] == {
        "sex": "female",
        "age": ["90_plus"],
    }
    assert male_90_plus["ledger_selector"]["dimension_values"] == {
        "sex": "male",
        "age": ["90_plus"],
    }
    assert female_85_89["measurement"]["filters"] == [
        {"concept": "uk.demographics.gender", "equals": "female"},
        {"concept": "uk.demographics.age", "operator": ">=", "value": 85},
        {"concept": "uk.demographics.age", "operator": "<=", "value": 89},
    ]
    assert female_90_plus["measurement"]["filters"] == [
        {"concept": "uk.demographics.gender", "equals": "female"},
        {"concept": "uk.demographics.age", "operator": ">=", "value": 90},
    ]


def test_uk_national_targets_have_unique_target_ids():
    resource = _load()

    target_ids = [target["target_id"] for target in resource["targets"]]
    assert len(target_ids) == len(set(target_ids))


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
        selector = target["ledger_selector"]
        assert selector, target["target_id"]
        assert set(selector) <= SELECTOR_KEYS, target["target_id"]
        assert "assertion" not in selector, target["target_id"]
        binding = target["bindings"]["policyengine"]
        assert set(binding) <= POLICYENGINE_BINDING_KEYS, target["target_id"]
        metric_names.append(binding["metric_name"])
        kind = binding.get("kind")
        assert kind is None or kind in BINDING_KINDS, target["target_id"]
        if target["family"] in PROJECTION_FAMILIES:
            assert target.get("assertion_policy") == "allow_source_projection", target[
                "target_id"
            ]
        else:
            assert "assertion_policy" not in target, target["target_id"]
    assert len(metric_names) == len(set(metric_names))


def test_uk_national_targets_declare_chronicle_loader_guarantees():
    resource = _load()

    for target in resource["targets"]:
        selector = target["ledger_selector"]
        if "dimension_values" in selector:
            assert _valid_dimension_values(selector["dimension_values"]), target[
                "target_id"
            ]
        if "dimensions" in selector:
            dimensions = selector["dimensions"]
            assert isinstance(dimensions, list), target["target_id"]
            assert all(isinstance(name, str) and name for name in dimensions), target[
                "target_id"
            ]

        binding = target["bindings"]["policyengine"]
        kind = binding.get("kind")
        if kind == "input_substitution_counterfactual":
            assert {
                "zeroed_input",
                "folded_into",
                "output_variable",
                "output_delta",
            } <= set(binding), target["target_id"]
        elif kind == "parameter_gated_threshold":
            assert {
                "gate_parameter",
                "gated_variable",
                "gate_comparison",
            } <= set(binding), target["target_id"]
        elif kind == "baseline_flag_crosstab":
            assert {"affected_flag_variable", "count_of"} <= set(binding), target[
                "target_id"
            ]

        if "reduce" in binding:
            assert binding["reduce"] in BINDING_REDUCERS, target["target_id"]
        for field in ("filters", "household_conditions"):
            for predicate in binding.get(field, ()):
                if "reduce" in predicate:
                    assert predicate["reduce"] in PREDICATE_REDUCERS, (
                        target["target_id"],
                        predicate,
                    )
                if "entity" in predicate:
                    assert predicate["entity"] in PREDICATE_ENTITIES, (
                        target["target_id"],
                        predicate,
                    )

        assertion_policy = target.get("assertion_policy")
        if assertion_policy is not None:
            assert assertion_policy in ASSERTION_POLICIES, target["target_id"]

    assert (
        _target_by_id(resource, "obr.esa")["bindings"]["policyengine"][
            "value_expression"
        ]
        == "esa_income + esa_contrib"
    )


def test_uk_uc_households_target_counts_benunits():
    resource = _load()
    target = _target_by_id(resource, "dwp.uc.households")

    assert target["measurement"] == {
        "entity": "benunit",
        "concept": "uk.benefit_unit.count",
        "filters": [
            {
                "concept": "uk.benefits.universal_credit.amount",
                "operator": ">",
                "value": 0,
            }
        ],
    }
    assert target["bindings"]["policyengine"]["value_variable"] == "benunit_count"
    assert target["bindings"]["policyengine"]["from_entity"] == "benunit"
    assert target["bindings"]["policyengine"]["filters"] == [
        {
            "variable": "universal_credit",
            "operator": ">",
            "value": 0,
        }
    ]
    assert target["ledger_selector"] == {
        "source_name": "dwp",
        "source_concept": "dwp.uc_benefit_units",
    }


def test_uk_national_cgt_contract_names_match_runtime_specs():
    resource = _load()

    cgt_metric_names = sorted(
        target["bindings"]["policyengine"]["metric_name"]
        for target in resource["targets"]
        if target["target_id"].startswith("hmrc.cgt.")
    )
    mapped = resource["registry_parity"]["mapped"]

    assert UK_CGT_TARGET_SPECS == ()
    assert cgt_metric_names == [
        "hmrc/capital_gains_total",
        "hmrc/cgt_taxpayers",
    ]
    assert mapped["hmrc/capital_gains_total"] == "hmrc.cgt.gains_total"
    assert mapped["hmrc/cgt_taxpayers"] == "hmrc.cgt.taxpayers_total"


def _target_by_id(resource: Mapping, target_id: str) -> Mapping:
    for target in resource["targets"]:
        if target["target_id"] == target_id:
            return target
    raise AssertionError(f"missing target {target_id!r}")


def _valid_dimension_values(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not value:
        return False
    for key, expected in value.items():
        if not isinstance(key, str) or not key:
            return False
        if _is_dimension_scalar(expected):
            continue
        if (
            isinstance(expected, list)
            and expected
            and all(_is_dimension_scalar(item) for item in expected)
        ):
            continue
        return False
    return True


def _is_dimension_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)
