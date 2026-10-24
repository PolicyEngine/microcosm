"""Tests split from packages/microcosm-build/tests/test_spec_engine_inventory_coverage.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.spec_engine_inventory_coverage import *


def test_us_inventory_is_structure_exact_and_complete(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    __import__("policyengine_us")
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )

    assert_inventory_coverage_complete(report)
    assert report["required_item_count"] == 41
    assert report["covered_item_count"] == 41
    assert report["missing_item_count"] == 0
    assert report["missing_items"] == []
    assert set(report["items"]) == EXPECTED_CHECKS
    assert report["counts"] == EXPECTED_COUNTS


def test_full_checkpoint_vector_binds_dynamic_inputs_and_scale_controls(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    __import__("policyengine_us")
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    item = report["items"]["stacked_checkpoint_base_identity_exact"]
    assert item["status"] == "covered"
    assert item["observed"]["input_roles"] == ["alpha", "zeta"]
    assert item["observed"]["fraction_token"] == "f025"
    assert item["observed"]["sha256"] == item["expected"]["sha256"]
