"""Tests split from packages/microcosm-build/tests/test_spec_engine_engine_abi.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_engine_abi import *


def test_checked_in_us_lock_is_fresh_schema_valid_and_not_authored() -> None:
    __import__("policyengine_us")
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
        "version": "2.145.3",
    }
    assert len(payload["programs"]) == 21
    assert len({row["variable"] for row in payload["programs"].values()}) == 21
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
    assert len(remaining["rows"]) == 2106
    assert remaining["receipt"]["entry_count"] == 2106
    assert remaining["receipt"]["stage_counts"] == {
        "derive": 34,
        "seed": 37,
        "simulate": 2035,
    }
    assert remaining["receipt"]["manifest_sha256"] == (
        "3bd50a8db708f3c08aecf14277a549d82c85fd9c88fc60b441ab8750de531879"
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
