"""Generation-1 US bundle migration and package-seam gates."""

# ruff: noqa: F401

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from microcosm.build.country_spec import ResolvedCountrySpec, load_country_spec
from microcosm.build.spec_engine import (
    F0_KERNEL_REGISTRY,
    ResolvedSpec,
    ResourceKind,
    SpecResolutionError,
    SpecValidationError,
    load_bundle,
    load_schema_registry,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.resolver import resolve_cross_references
from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
from microcosm.build.spec_engine.yaml12 import load_yaml12_file
from microcosm.build.us_runtime.take_up_contract import take_up_contract_identity
from test_support.paths import paths_for
from tools.us_bundle_generation.contracts import derive_battery_registry_views
from tools.us_bundle_generation.core import (
    project_publication_legacy_release,
    project_spine_legacy_sampling,
)
from tools.us_bundle_generation.imputation import (
    derive_primary_effective_predictor_tuples,
    project_imputation_legacy_payloads,
)

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
US_PACKAGE_ROOT = ROOT / "packages/microcosm-build/src/microcosm/build/us"
US_SPEC_ROOT = US_PACKAGE_ROOT / "spec"
AUTHORING_POINTER_ROOT = ROOT / "specs/us"


def _load_generator_module():
    path = ROOT / "tools/generate_us_bundle_from_constants.py"
    spec = importlib.util.spec_from_file_location(
        "generate_us_bundle_from_constants",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator_module()

TYPED_DOMAIN_KINDS = frozenset(
    {
        "battery",
        "bundle",
        "calibration",
        "catalogs",
        "geography",
        "imputation",
        "publication",
        "selection",
        "sources",
        "spine",
        "take_up",
        "vintages",
    }
)

F_P_WAIVED_TARGETS = frozenset(
    {
        "has_champva_health_coverage_at_interview",
        "has_esi",
        "has_indian_health_service_coverage_at_interview",
        "has_marketplace_health_coverage_at_interview",
        "has_medicaid_health_coverage_at_interview",
        "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        "has_other_means_tested_health_coverage_at_interview",
        "has_tricare_health_coverage_at_interview",
        "has_va_health_coverage_at_interview",
        "receives_tanf",
        "receives_housing_assistance",
        "receives_snap",
        "receives_wic",
        "takes_up_housing_assistance_if_eligible",
        "takes_up_medicare_if_eligible",
        "takes_up_wic_if_eligible",
    }
)

EXPECTED_F_P_CONCEPTS = {
    "american_indian_status": ["is_american_indian_or_alaska_native"],
    "citizenship_status": ["is_us_citizen"],
    "dependent_child_status": ["own_children_in_household"],
    "disability_status": ["has_hearing_difficulty", "has_vision_difficulty"],
    "employment_attachment": ["hours_worked_last_week", "weeks_worked_last_year"],
    "household_income_eligibility": ["spm_unit_net_income", "spm_unit_size"],
    "housing_need": ["pre_subsidy_rent", "tenure_type"],
    "medicare_coverage_context": ["acs_hins_medicare"],
    "military_coverage_context": ["acs_hins_va"],
    "pregnancy_status": ["is_pregnant"],
    "private_coverage_context": [
        "acs_hins_employer",
        "acs_hins_direct_purchase",
    ],
    "public_coverage_context": [
        "acs_hins_medicaid",
        "acs_hins_other_public",
    ],
    "veteran_status": ["is_veteran", "receives_va_payments"],
}

EXPECTED_F_P_TARGET_CONCEPTS = {
    "has_champva_health_coverage_at_interview": [
        "veteran_status",
        "military_coverage_context",
    ],
    "has_esi": ["employment_attachment", "private_coverage_context"],
    "has_indian_health_service_coverage_at_interview": ["american_indian_status"],
    "has_marketplace_health_coverage_at_interview": [
        "citizenship_status",
        "household_income_eligibility",
        "private_coverage_context",
    ],
    "has_medicaid_health_coverage_at_interview": [
        "dependent_child_status",
        "disability_status",
        "household_income_eligibility",
        "public_coverage_context",
    ],
    "has_non_marketplace_direct_purchase_health_coverage_at_interview": [
        "household_income_eligibility",
        "private_coverage_context",
    ],
    "has_other_means_tested_health_coverage_at_interview": [
        "disability_status",
        "household_income_eligibility",
        "public_coverage_context",
    ],
    "has_tricare_health_coverage_at_interview": ["veteran_status"],
    "has_va_health_coverage_at_interview": [
        "veteran_status",
        "military_coverage_context",
    ],
    "receives_tanf": [
        "dependent_child_status",
        "household_income_eligibility",
    ],
    "receives_housing_assistance": [
        "household_income_eligibility",
        "housing_need",
    ],
    "receives_snap": [
        "dependent_child_status",
        "disability_status",
        "household_income_eligibility",
    ],
    "receives_wic": [
        "dependent_child_status",
        "household_income_eligibility",
        "pregnancy_status",
    ],
    "takes_up_housing_assistance_if_eligible": [
        "household_income_eligibility",
        "housing_need",
    ],
    "takes_up_medicare_if_eligible": [
        "disability_status",
        "medicare_coverage_context",
    ],
    "takes_up_wic_if_eligible": ["dependent_child_status", "pregnancy_status"],
}

EXPECTED_RUNGS = [
    {"token": "f001", "fraction": 0.01, "percent_basis_points": 100},
    {"token": "f004", "fraction": 0.04, "percent_basis_points": 400},
    {"token": "f010", "fraction": 0.1, "percent_basis_points": 1_000},
    {"token": "f025", "fraction": 0.25, "percent_basis_points": 2_500},
    {"token": "f100", "fraction": 1.0, "percent_basis_points": 10_000},
]

LEGACY_COMPATIBILITY_SHA256 = {
    "source_stages.json": (
        "9f4f983091fb18dea8f4524cc100be20f9fca2b006435461592f8edf53c9a5d8"
    ),
    "support_spine.json": (
        "68f37dc6ae6e0cde7ebccb53f88dd4a800e63456f838fa214ff98d1db8d815be"
    ),
    "take_up_contract.json": (
        "282dbc4c31e30134d008152138a95dd1a84f4355b44ac83607760dcaeef35db5"
    ),
}


@pytest.fixture(autouse=True)
def _prime_worker_identity(prime_primary_qrf_worker_identity: None) -> None:
    """Share the real session attestation unless a test opts into live identity."""


@pytest.fixture(scope="module")
def resolved_us_spec() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def resolved_country_spec() -> ResolvedCountrySpec:
    return load_country_spec("us")


@pytest.fixture
def generated_documents(
    prime_primary_qrf_worker_identity: None,
) -> dict[str, dict[str, object]]:
    # The function fixture primes before the first cached generation, including
    # when earlier tests already initialized and then cleared the session memo.
    return _cached_generated_documents()


@cache
def _cached_generated_documents() -> dict[str, dict[str, object]]:
    # The extractor is intentionally exercised once per module: this proves
    # the checked-in package remains a projection of the live constants while
    # keeping the comparatively expensive PolicyEngine ABI read bounded.
    pytest.importorskip(
        "policyengine_us",
        reason="live-engine oracle: the wheels gate's venv installs no engine",
        exc_type=ModuleNotFoundError,
    )
    return generator.build_documents()


def _domain(spec: ResolvedSpec, kind: ResourceKind | str) -> dict[str, object]:
    value = spec.domain(kind).to_wire()
    assert isinstance(value, dict)
    return value


def _json_resource(name: str) -> dict[str, object]:
    value = json.loads((US_PACKAGE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _count_scalar(value: object, expected: object) -> int:
    if isinstance(value, dict):
        return sum(_count_scalar(child, expected) for child in value.values())
    if isinstance(value, list):
        return sum(_count_scalar(child, expected) for child in value)
    return int(value == expected)


def _generated_resolution_resources(
    generated_documents: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        filename.removesuffix(".yaml"): copy.deepcopy(document)
        for filename, document in generated_documents.items()
    }


def _resolve_generated_resources(resources: dict[str, object]) -> None:
    resolve_cross_references(
        resources,
        kernel_registry=F0_KERNEL_REGISTRY,
        generated_authorities={
            "engine_abi_lock": _json_resource("engine_abi.lock.json")
        },
    )


def _participation_target(
    imputation: dict[str, object], target_name: str
) -> dict[str, object]:
    return next(
        target
        for family in imputation["families"]
        for target in family["targets"]
        if target["name"] == target_name
    )


__all__ = [name for name in globals() if not name.startswith("__")]
