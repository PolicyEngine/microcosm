"""End-to-end gates for the committed F0 coverage attestation."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from microcosm.build.spec_engine.compiler_ir import CompiledSpecIR, compile_spec
from microcosm.build.spec_engine.legacy_adapter import compile_to_legacy_payload
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import ResolvedSpec
from tools.spec_engine_coverage import (
    DEFAULT_REPORT_PATH,
    CoverageError,
    assert_coverage_complete,
    build_coverage_report,
    coverage_report_bytes,
)

pytest.importorskip(
    "policyengine_us",
    reason="live-engine oracle: the wheels gate's venv installs no engine",
    exc_type=ModuleNotFoundError,
)


@pytest.fixture(scope="module")
def coverage_inputs() -> tuple[ResolvedSpec, CompiledSpecIR, dict[str, object]]:
    spec = load_bundle("us")
    return spec, compile_spec(spec), compile_to_legacy_payload(spec)


@pytest.fixture(scope="module")
def coverage_report(
    coverage_inputs: tuple[ResolvedSpec, CompiledSpecIR, dict[str, object]],
) -> dict[str, object]:
    spec, compiled, legacy = coverage_inputs
    return build_coverage_report(
        spec,
        compiled=compiled,
        legacy_payload=legacy,
    )


def test_us_coverage_is_exact_complete_and_honest(
    coverage_report: dict[str, object],
) -> None:
    assert_coverage_complete(coverage_report)
    assert coverage_report["status"] == "pass"
    fields = coverage_report["field_usage"]
    assert fields["configuration_field_count"] == 42_154
    assert fields["authored_normative_field_count"] == 32_384
    assert fields["resolved_binding_field_count"] == 9_770
    assert fields["consumed_field_count"] == 42_154
    assert fields["unused_field_count"] == 0
    assert fields["multiple_primary_use_field_count"] == 0
    assert fields["claim_count"] == 49
    assert fields["mode_counts"] == {
        "legacy_behavior": 13_988,
        "compiler_semantic": 27_715,
        "front_end_validation": 348,
        "identity_only": 103,
    }
    assert fields["generation0_effect_counts"] == {
        "legacy_behavior": 38_476,
        "no_generation0_effect": 3_678,
    }

    inventory = coverage_report["inventory_coverage"]
    assert inventory["required_item_count"] == 41
    assert inventory["covered_item_count"] == 41
    assert inventory["missing_item_count"] == 0
    assert inventory["missing_items"] == []
    assert inventory["counts"]["producer_inputs"] == 2_744
    assert inventory["counts"]["ownership_rows"] == 18
    assert inventory["counts"]["tail_control_fields"] == 934
    assert inventory["counts"]["seed_owner_bindings"] == 112


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda report: report.__setitem__("report_schema_version", 2),
            "report schema version differs",
        ),
        (
            lambda report: report["spec_binding"].__setitem__(
                "schema_version", 99
            ),
            "spec_binding contract differs",
        ),
        (
            lambda report: report["field_usage"]["claims"].pop(),
            "claim count differs",
        ),
        (
            lambda report: report["field_usage"]["mode_counts"].__setitem__(
                "legacy_behavior", 0
            ),
            "mode_counts do not match",
        ),
    ],
)
def test_coverage_assertion_recomputes_tampered_version_binding_and_claims(
    coverage_report: dict[str, object],
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    corrupted = deepcopy(coverage_report)
    mutation(corrupted)

    with pytest.raises(CoverageError, match=match):
        assert_coverage_complete(corrupted)


def test_missing_legacy_sink_fails_even_with_compiled_copies_present(
    coverage_inputs: tuple[ResolvedSpec, CompiledSpecIR, dict[str, object]],
) -> None:
    spec, compiled, legacy = coverage_inputs
    broken = deepcopy(legacy)
    calibration = broken["calibration_contract"]
    assert isinstance(calibration, dict)
    solver = calibration["solver"]
    assert isinstance(solver, dict)
    stopping = solver["stopping_contract"]
    assert isinstance(stopping, dict)
    del stopping["max_epochs"]

    with pytest.raises(
        CoverageError,
        match=r"calibration_contract/solver/stopping_contract/max_epochs",
    ):
        build_coverage_report(
            spec,
            compiled=compiled,
            legacy_payload=broken,
        )


def test_committed_report_is_current(coverage_report: dict[str, object]) -> None:
    report_path = Path(DEFAULT_REPORT_PATH)
    assert report_path.read_bytes() == coverage_report_bytes(coverage_report)
