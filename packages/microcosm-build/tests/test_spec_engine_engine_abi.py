"""Generated engine-ABI-lock derivation and refusal gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from microcosm.build.spec_engine import (
    SpecValidationError,
    assert_engine_abi_lock_current,
    engine_abi_lock_bytes_from_domains,
    engine_abi_lock_payload_from_domains,
    load_schema_registry,
    load_yaml12_file,
)
from microcosm.build.spec_engine import engine_abi as engine_abi_module
from microcosm.build.spec_engine.canonical import sha256_json, spec_envelope

ROOT = Path(__file__).resolve().parents[3]
US_PACKAGE_ROOT = ROOT / "packages/microcosm-build/src/microcosm/build/us"
US_SPEC_ROOT = US_PACKAGE_ROOT / "spec"
ENGINE_VERSION_REF = {
    "kind": "engine_abi_lock",
    "pointer": "/engine/version",
}


def _count_scalar(value: object, expected: object) -> int:
    if isinstance(value, dict):
        return sum(_count_scalar(child, expected) for child in value.values())
    if isinstance(value, list):
        return sum(_count_scalar(child, expected) for child in value)
    return int(value == expected)


def _fake_domains() -> dict[str, object]:
    return {
        "sources": {"stages": []},
        "vintages": {
            "records": [
                {
                    "id": "policyengine_us_surface",
                    "kind": "policy_engine_surface_ref",
                    "authority_ref": {
                        "kind": "engine_abi_lock",
                        "pointer": "/engine/version",
                    },
                }
            ]
        },
        "take_up": {
            "programs": [
                {"id": f"program_{index:02d}", "variable": f"variable_{index:02d}"}
                for index in range(13)
            ]
        },
    }


def _fake_engine_contract() -> dict[str, dict[str, object]]:
    return {
        f"variable_{index:02d}": {
            "entity": "person",
            "value_type": "bool",
            "default": True,
            "engine_computed": False,
            "consumers": [f"consumer_{index:02d}"],
            "engine_class": "data_seeded",
        }
        for index in range(13)
    }


def _fake_remaining_stage_input_manifest(
    *_args: object, **_kwargs: object
) -> dict[str, object]:
    return {
        "rows": [
            {
                "stage": "simulate",
                "consumer": "fake_consumer",
                "entity": "person",
                "variable": "age",
                "execution_scope": "whole_pool",
                "provision": "fake_provision",
                "available_by": "assembled",
                "fallback": None,
            }
        ],
        "receipt": {
            "schema_version": 1,
            "entry_count": 1,
            "stage_counts": {"derive": 0, "seed": 0, "simulate": 1},
            "consumer_counts": {"fake_consumer": 1},
            "ssi_dependency_contract": {
                "engine_version": "1.2.3",
                "root": "ssi",
                "input_leaf_count": 1,
                "formula_node_count": 1,
                "edge_count": 1,
                "sha256": "0" * 64,
            },
            "engine_input_projection_contract": {
                "engine_version": "1.2.3",
                "input_count": 1,
                "default_count": 1,
                "sha256": "1" * 64,
                "defaults_sha256": "2" * 64,
            },
            "manifest_sha256": "3" * 64,
            "sha256": "4" * 64,
        },
    }


@pytest.fixture
def fake_fresh_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_abi_module,
        "_installed_engine_version",
        lambda package: "1.2.3",
    )
    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_policyengine_us_contract",
        _fake_engine_contract,
    )
    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        _fake_remaining_stage_input_manifest,
    )


def test_engine_version_is_owned_only_by_the_fresh_generated_lock(
    fake_fresh_engine: None,
) -> None:
    domains = _fake_domains()
    payload = engine_abi_lock_payload_from_domains(domains)

    assert payload["engine"] == {
        "package": "policyengine-us",
        "version": "1.2.3",
    }
    receipt = payload["remaining_stage_input_manifest"]["receipt"]
    assert receipt["ssi_dependency_contract"]["engine_version_ref"] == (
        ENGINE_VERSION_REF
    )
    assert receipt["engine_input_projection_contract"]["engine_version_ref"] == (
        ENGINE_VERSION_REF
    )
    assert "engine_version" not in receipt["ssi_dependency_contract"]
    assert "engine_version" not in receipt["engine_input_projection_contract"]
    assert receipt["ssi_dependency_contract"]["sha256"] == "0" * 64
    assert receipt["engine_input_projection_contract"]["sha256"] == "1" * 64
    assert receipt["engine_input_projection_contract"]["defaults_sha256"] == ("2" * 64)
    assert receipt["sha256"] == "4" * 64
    assert _count_scalar(payload, "1.2.3") == 1
    assert "value" not in domains["vintages"]["records"][0]


def test_nested_receipt_version_must_match_the_single_engine_pin(
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched_manifest(*_args: object, **_kwargs: object) -> dict[str, object]:
        manifest = _fake_remaining_stage_input_manifest()
        manifest["receipt"]["ssi_dependency_contract"]["engine_version"] = "9.9.9"
        return manifest

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        mismatched_manifest,
    )
    with pytest.raises(
        SpecValidationError,
        match="engine version differs from the exact generated engine pin",
    ):
        engine_abi_lock_payload_from_domains(_fake_domains())


def test_program_variable_mapping_must_be_total_and_injective(
    fake_fresh_engine: None,
) -> None:
    domains = _fake_domains()
    programs = domains["take_up"]["programs"]
    programs[1]["variable"] = programs[0]["variable"]
    with pytest.raises(SpecValidationError, match="must be injective"):
        engine_abi_lock_payload_from_domains(domains)

    domains = _fake_domains()
    domains["take_up"]["programs"][0]["variable"] = "unknown_variable"
    with pytest.raises(
        SpecValidationError, match="not total over the fresh engine ABI"
    ):
        engine_abi_lock_payload_from_domains(domains)


def test_stale_or_noncanonical_generated_lock_is_refused(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domains = _fake_domains()
    lock_path = tmp_path / "engine_abi.lock.json"
    lock_path.write_bytes(engine_abi_lock_bytes_from_domains(domains))
    registry = load_schema_registry()
    assert_engine_abi_lock_current(
        tmp_path,
        domains,
        schema_registry=registry,
    )

    monkeypatch.setattr(
        engine_abi_module,
        "_installed_engine_version",
        lambda package: "9.9.9",
    )

    def bumped_manifest(*_args: object, **_kwargs: object) -> dict[str, object]:
        manifest = _fake_remaining_stage_input_manifest()
        receipt = manifest["receipt"]
        receipt["ssi_dependency_contract"]["engine_version"] = "9.9.9"
        receipt["engine_input_projection_contract"]["engine_version"] = "9.9.9"
        return manifest

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        bumped_manifest,
    )
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        assert_engine_abi_lock_current(
            tmp_path,
            domains,
            schema_registry=registry,
        )

    monkeypatch.setattr(
        engine_abi_module,
        "_installed_engine_version",
        lambda package: "1.2.3",
    )
    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        _fake_remaining_stage_input_manifest,
    )

    stale = engine_abi_lock_payload_from_domains(domains)
    stale["programs"]["program_00"]["default"] = False
    lock_path.write_text(json.dumps(stale, sort_keys=True), encoding="utf-8")
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        assert_engine_abi_lock_current(
            tmp_path,
            domains,
            schema_registry=registry,
        )

    stale = engine_abi_lock_payload_from_domains(domains)
    stale["remaining_stage_input_manifest"]["rows"][0]["provision"] = (
        "mutated_provision"
    )
    lock_path.write_bytes(
        json.dumps(stale, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        assert_engine_abi_lock_current(
            tmp_path,
            domains,
            schema_registry=registry,
        )


def test_remaining_stage_manifest_lock_rows_are_closed_world(
    fake_fresh_engine: None,
) -> None:
    payload = engine_abi_lock_payload_from_domains(_fake_domains())
    payload["remaining_stage_input_manifest"]["rows"][0]["undeclared"] = True

    with pytest.raises(SpecValidationError, match="undeclared"):
        load_schema_registry().validate(
            payload,
            "locks.schema.json#/$defs/engine_abi_lock",
        )


def test_remaining_stage_engine_version_refs_are_closed_and_exact(
    fake_fresh_engine: None,
) -> None:
    payload = engine_abi_lock_payload_from_domains(_fake_domains())
    receipt = payload["remaining_stage_input_manifest"]["receipt"]
    dependency = receipt["ssi_dependency_contract"]
    dependency["engine_version_ref"]["pointer"] = "/engine/package"

    with pytest.raises(SpecValidationError, match="/engine/version"):
        load_schema_registry().validate(
            payload,
            "locks.schema.json#/$defs/engine_abi_lock",
        )


def test_generated_remaining_stage_manifest_is_a_canonical_identity_binding(
    fake_fresh_engine: None,
) -> None:
    before = engine_abi_lock_payload_from_domains(_fake_domains())
    after = copy.deepcopy(before)
    after["remaining_stage_input_manifest"]["rows"][0]["provision"] = (
        "mutated_provision"
    )

    def identity(payload: object) -> str:
        return sha256_json(
            spec_envelope(
                country="us",
                schema_version=1,
                normative_files={},
                resolved_bindings={
                    "generated_authorities": {"engine_abi_lock": payload}
                },
            )
        )

    assert identity(before) != identity(after)


def test_checked_in_us_lock_is_fresh_schema_valid_and_not_authored() -> None:
    domains = {
        "vintages": load_yaml12_file(US_SPEC_ROOT / "vintages.yaml"),
        "take_up": load_yaml12_file(US_SPEC_ROOT / "take_up.yaml"),
        "sources": load_yaml12_file(US_SPEC_ROOT / "sources.yaml"),
    }
    path = US_PACKAGE_ROOT / "engine_abi.lock.json"
    payload = json.loads(path.read_bytes())

    assert path.read_bytes() == engine_abi_lock_bytes_from_domains(domains)
    load_schema_registry().validate(
        payload,
        "locks.schema.json#/$defs/engine_abi_lock",
    )
    assert payload["engine"] == {
        "package": "policyengine-us",
        "version": "1.764.6",
    }
    assert len(payload["programs"]) == 13
    assert len({row["variable"] for row in payload["programs"].values()}) == 13
    assert all(
        set(row)
        == {
            "variable",
            "entity",
            "value_type",
            "default",
            "engine_class",
            "consumers",
        }
        for row in payload["programs"].values()
    )
    remaining = payload["remaining_stage_input_manifest"]
    assert len(remaining["rows"]) == 993
    assert remaining["receipt"]["entry_count"] == 993
    assert remaining["receipt"]["stage_counts"] == {
        "derive": 34,
        "seed": 29,
        "simulate": 930,
    }
    assert remaining["receipt"]["manifest_sha256"] == (
        "8247a93e5f8f63d3ae71c1de681c29524d4bb8f07e3c6a50dcaf431b1377020f"
    )
    assert (
        remaining["receipt"]["ssi_dependency_contract"]["engine_version_ref"]
        == ENGINE_VERSION_REF
    )
    assert (
        remaining["receipt"]["engine_input_projection_contract"]["engine_version_ref"]
        == ENGINE_VERSION_REF
    )
    assert _count_scalar(payload, payload["engine"]["version"]) == 1

    manifest = json.loads((US_PACKAGE_ROOT / "country_package.json").read_bytes())
    assert "engine_abi.lock.json" not in {row["path"] for row in manifest["resources"]}


def test_lock_mutation_does_not_change_authored_domain_payloads(
    fake_fresh_engine: None,
) -> None:
    domains = _fake_domains()
    before = copy.deepcopy(domains)
    engine_abi_lock_payload_from_domains(domains)
    assert domains == before
