"""Normalized calibration authority and constants-era projection gates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from microcosm.build.spec_engine import (
    CALIBRATION_SUMMARY_ALIASES,
    SpecValidationError,
    load_schema_registry,
    project_legacy_calibration_contract,
    scoped_take_up_manifest_program_bindings,
)
from microcosm.build.spec_engine.canonical import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[3]
US_ROOT = ROOT / "packages/microcosm-build/src/microcosm/build/us"
LEGACY_CANONICAL_SHA256 = (
    "8cc14b405d775640123eafeda89405ab0cb28455c554df8e9bf0eb950d78a806"
)


def _load_contract_builder():
    if "tools" not in sys.modules:
        tools_package = types.ModuleType("tools")
        tools_package.__path__ = [str(ROOT / "tools")]
        sys.modules["tools"] = tools_package
    name = "f0_us_bundle_contracts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "tools/us_bundle_generation/contracts.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_calibration_contract = _load_contract_builder().build_calibration_contract


def _bootstrap_bindings() -> Iterator[tuple[str, str, str]]:
    frozen = json.loads((US_ROOT / "take_up_contract.json").read_text())
    return (
        (row["variable"], row["entity"], row["populace_treatment"])
        for row in frozen["programs"]
    )


@pytest.fixture(scope="module")
def calibration() -> dict[str, object]:
    # Constants extraction is deliberately scoped to frozen generation-0
    # bindings so this test remains runnable before the typed YAML is rewritten.
    with scoped_take_up_manifest_program_bindings(tuple(_bootstrap_bindings())):
        return build_calibration_contract()


def test_normalized_calibration_is_closed_and_has_no_summary_aliases(
    calibration: dict[str, object],
) -> None:
    load_schema_registry().validate(calibration, "calibration.schema.json")
    solver = calibration["solver"]
    assert not CALIBRATION_SUMMARY_ALIASES.intersection(solver)


def test_normalized_contract_reconstructs_the_prior_canonical_object(
    calibration: dict[str, object],
) -> None:
    projected = project_legacy_calibration_contract(calibration)

    assert hashlib.sha256(canonical_json_bytes(projected)).hexdigest() == (
        LEGACY_CANONICAL_SHA256
    )
    assert CALIBRATION_SUMMARY_ALIASES <= projected["solver"].keys()


def test_projection_is_pure(calibration: dict[str, object]) -> None:
    before = copy.deepcopy(calibration)
    project_legacy_calibration_contract(calibration)
    assert calibration == before


@pytest.mark.parametrize(
    ("path", "legacy_field", "replacement"),
    [
        (
            ("solver", "hard_constraints", "max_weight_ratio"),
            "max_weight_ratio",
            7.5,
        ),
        (
            ("solver", "initialization_contract", "policy_id"),
            "initialization",
            "reviewed_mutation",
        ),
        (
            ("solver", "stopping_contract", "max_epochs"),
            "stopping",
            12,
        ),
        (
            ("solver", "infeasibility_contract", "soft_target_miss"),
            "infeasibility_policy",
            "reviewed_mutation",
        ),
        (
            ("solver", "target_priority_contract", "policy_id"),
            "target_priority",
            "reviewed_mutation",
        ),
    ],
)
def test_normalized_mutations_change_the_named_legacy_alias(
    calibration: dict[str, object],
    path: tuple[str, ...],
    legacy_field: str,
    replacement: object,
) -> None:
    mutated = copy.deepcopy(calibration)
    cursor = mutated
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement

    before = project_legacy_calibration_contract(calibration)["solver"][legacy_field]
    after = project_legacy_calibration_contract(mutated)["solver"][legacy_field]
    assert after != before


def test_unsupported_target_policy_mutation_is_refused(
    calibration: dict[str, object],
) -> None:
    mutated = copy.deepcopy(calibration)
    mutated["targets"]["zero_target_policy"] = "drop"

    with pytest.raises(
        SpecValidationError,
        match="calibration/targets: zero/negative policies",
    ):
        project_legacy_calibration_contract(mutated)


def test_retired_summary_alias_cannot_be_reintroduced(
    calibration: dict[str, object],
) -> None:
    mutated = copy.deepcopy(calibration)
    mutated["solver"]["max_weight_ratio"] = 5.0

    with pytest.raises(SpecValidationError, match="retired derived aliases"):
        project_legacy_calibration_contract(mutated)
    with pytest.raises(SpecValidationError, match="max_weight_ratio"):
        load_schema_registry().validate(mutated, "calibration.schema.json")
