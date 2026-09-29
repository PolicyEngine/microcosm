"""Generated engine-ABI-lock derivation and refusal gates."""

# ruff: noqa: F401

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
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
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
                for index in range(17)
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
        for index in range(17)
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


__all__ = [name for name in globals() if not name.startswith("__")]
