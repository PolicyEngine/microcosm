"""Specification tests for value-free population and monetary profiles."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.trace import sha256_file


def _population_profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "country": "xx",
        "profile_id": "benefit_population_inputs",
        "activation": "explicit_only",
        "description": "Value-free population input fixture.",
        "inputs": [
            {
                "input_id": "benefit_receipt",
                "column": "receives_benefit",
                "entity": "person",
                "dtype": "bool",
                "nullable": True,
                "semantic_kind": "receipt",
                "data_kind": "latent",
                "owner": "Microcosm",
                "consumer": "PolicyEngine",
                "mechanics_owner": "PolicyEngine",
                "axiom_role": "none",
                "description": "Measured or latent receipt status, not behavior.",
            }
        ],
        "mappings": [
            {
                "mapping_id": "benefit_receipt_mapping",
                "target_reference": "benefit_recipients",
                "input_id": "benefit_receipt",
                "chronicle_source_record_id": "official.benefit.recipients",
                "chronicle_entity": "person",
                "chronicle_entity_role": "recipient",
                "chronicle_geography_level": "statistical_scope",
                "chronicle_geography_id": "XX-SCHEME",
                "chronicle_geography_vintage": "scheme_scope_2024",
                "chronicle_period_type": "calendar_year",
                "chronicle_period": 2024,
                "microcosm_entity": "person",
                "microcosm_geography_level": "statistical_scope",
                "microcosm_geography_id": "XX-SCHEME",
                "microcosm_geography_vintage": "scheme_scope_2024",
                "microcosm_period_type": "calendar_year",
                "microcosm_period": 2024,
                "publisher_source_readiness": "ready",
                "input_readiness": "required_missing",
                "mapping_readiness": "required_missing",
                "period_readiness": "ready",
                "completeness_readiness": "complete_imputation_required_missing",
                "notes": "Execution remains blocked until the input exists.",
            }
        ],
    }


def _monetary_profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "country": "xx",
        "profile_id": "monetary_targets_2024",
        "activation": "explicit_only",
        "description": "Value-free monetary target fixture.",
        "targets": [
            {
                "reference": {
                    "name": "income/payroll",
                    "ledger_source_record_id": "official.payroll.2024",
                    "entity": "person",
                    "measure": "employment_income",
                    "period": 2024,
                    "family": "income",
                    "metadata": {
                        "monetary_target_role": "calibration",
                        "activation_status": "requires_prepared_measure",
                        "measure_kind": "prepared_column",
                    },
                },
                "basis": {
                    "currency": "XXX",
                    "unit": "base_currency",
                    "period": "2024",
                    "temporal_basis": "annual_flow",
                    "sector": "S14",
                    "perimeter": "resident employee payroll",
                    "valuation": "nominal",
                },
                "readiness": "requires_prepared_measure",
                "source_url": "https://example.test/payroll",
                "notes": "Requires a prepared entity-aligned amount column.",
            }
        ],
    }


def _write_package(root: Path, resources: dict[str, dict[str, object]]) -> Path:
    package = root / "xx"
    package.mkdir()
    manifest = {
        "schema_version": 1,
        "country": "xx",
        "policy": "spec-only test package",
        "resources": list(resources),
    }
    (package / "country_package.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    for name, payload in resources.items():
        (package / name).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    return package


def _profile_resources() -> dict[str, dict[str, object]]:
    return {
        "population_inputs.json": _population_profile(),
        "monetary_target_profile.json": _monetary_profile(),
    }


def test_profiles_are_optional_when_not_declared(tmp_path: Path) -> None:
    package = _write_package(tmp_path, {"evidence.json": {"status": "present"}})

    spec = load_country_spec(package)

    assert spec.population_input_profile is None
    assert spec.monetary_target_profile is None


def test_declared_profiles_are_typed_and_content_hashed(tmp_path: Path) -> None:
    resources = _profile_resources()
    package = _write_package(tmp_path, resources)

    before = load_country_spec(package)

    assert before.population_input_profile is not None
    assert before.population_input_profile.profile_id == "benefit_population_inputs"
    assert before.population_input_profile.inputs[0].owner == "Microcosm"
    assert before.monetary_target_profile is not None
    assert before.monetary_target_profile.profile_id == "monetary_targets_2024"
    assert before.monetary_target_profile.targets[0].basis.period == "2024"
    for name in resources:
        assert before.resource_hashes[name] == sha256_file(package / name)

    population_path = package / "population_inputs.json"
    resources["population_inputs.json"]["description"] = (
        "Changed value-free population input fixture."
    )
    population_path.write_text(
        json.dumps(resources["population_inputs.json"], sort_keys=True),
        encoding="utf-8",
    )
    after = load_country_spec(package)

    assert (
        after.resource_hashes["population_inputs.json"]
        != before.resource_hashes["population_inputs.json"]
    )
    assert (
        after.resource_hashes["monetary_target_profile.json"]
        == before.resource_hashes["monetary_target_profile.json"]
    )
    assert after.fingerprint != before.fingerprint


@pytest.mark.parametrize(
    ("profile_name", "failure"),
    [
        ("population_inputs.json", "country"),
        ("monetary_target_profile.json", "country"),
        ("population_inputs.json", "malformed"),
        ("monetary_target_profile.json", "malformed"),
    ],
)
def test_declared_profiles_fail_on_country_mismatch_or_malformed_content(
    tmp_path: Path,
    profile_name: str,
    failure: str,
) -> None:
    resources = copy.deepcopy(_profile_resources())
    profile = resources[profile_name]
    if failure == "country":
        profile["country"] = "yy"
    elif profile_name == "population_inputs.json":
        del profile["activation"]
    else:
        profile["targets"][0]["readiness"] = "ready"
    package = _write_package(tmp_path, resources)

    with pytest.raises(ValueError):
        load_country_spec(package)


@pytest.mark.parametrize("container", ["ledger_selector", "metadata"])
def test_monetary_profile_refuses_nested_carried_values(
    tmp_path: Path, container: str
) -> None:
    resources = copy.deepcopy(_profile_resources())
    reference = resources["monetary_target_profile.json"]["targets"][0]["reference"]
    reference.setdefault(container, {})["observed_value"] = 42
    package = _write_package(tmp_path, resources)

    with pytest.raises(ValueError, match="must be value-free"):
        load_country_spec(package)


def test_monetary_profile_refuses_selector_period_drift(tmp_path: Path) -> None:
    resources = copy.deepcopy(_profile_resources())
    reference = resources["monetary_target_profile.json"]["targets"][0]["reference"]
    reference["ledger_selector"] = {
        "period_type": "calendar_year",
        "period_value": 2023,
    }
    package = _write_package(tmp_path, resources)

    with pytest.raises(ValueError, match="selector period must match"):
        load_country_spec(package)
