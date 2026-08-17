"""Compiler-to-constants adapter and exact field-diff gates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from microcosm.build.spec_engine import ResolvedSpec, load_bundle
from microcosm.build.spec_engine.canonical import canonical_json_bytes
from microcosm.build.spec_engine.legacy_adapter import (
    LegacyPayloadMismatchError,
    assert_legacy_payload_equal,
    compile_to_legacy_payload,
    diff_legacy_payloads,
)
from microcosm.build.us_runtime.stacked_spine import (
    stacked_gap_fill_plan,
    stacked_gap_fill_producer_schedule_receipt,
    stacked_spine_authority_receipt,
)
from microcosm.build.us_runtime.take_up_contract import take_up_contract_identity
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    us_late_producer_schedule_receipt,
)

ROOT = Path(__file__).resolve().parents[3]
US_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/us"


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_ready(child) for child in value)
    return value


@pytest.fixture(scope="module")
def resolved_us_spec() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def legacy_payload(resolved_us_spec: ResolvedSpec) -> dict[str, object]:
    return compile_to_legacy_payload(resolved_us_spec)


def test_legacy_payload_diff_lists_every_named_field() -> None:
    expected = {
        "schedule": {"version": 1, "rows": [{"name": "first", "value": 3}]},
        "removed": True,
    }
    actual = {
        "schedule": {
            "version": 2,
            "rows": [{"name": "renamed", "value": 3}, "extra"],
        },
        "added": False,
    }

    differences = diff_legacy_payloads(expected, actual)

    assert [row.path for row in differences] == [
        "/removed",
        "/added",
        "/schedule/rows/0/name",
        "/schedule/rows/1",
        "/schedule/version",
    ]
    with pytest.raises(LegacyPayloadMismatchError) as failure:
        assert_legacy_payload_equal(expected, actual)
    message = str(failure.value)
    for difference in differences:
        assert difference.path in message


def test_legacy_payload_diff_is_type_exact() -> None:
    differences = diff_legacy_payloads(
        {"boolean": True, "integer": 1},
        {"boolean": 1, "integer": 1.0},
    )

    assert [row.path for row in differences] == ["/boolean", "/integer"]
    assert all(row.reason == "scalar type differs" for row in differences)


def test_equal_payload_has_no_differences() -> None:
    payload = {"ordered": [1, {"value": None}], "finite": 2.5}
    assert diff_legacy_payloads(payload, payload) == ()
    assert_legacy_payload_equal(payload, payload)


def test_compiler_adapter_covers_all_named_legacy_surfaces(
    legacy_payload: dict[str, object],
) -> None:
    assert set(legacy_payload) == {
        "battery_contract",
        "calibration_contract",
        "calibration_tail_contracts",
        "imputation",
        "publication_release",
        "source_manifest",
        "spine_sampling",
        "stacked_authority_receipt",
        "stacked_checkpoint_static_components",
        "support_spine",
        "take_up_contract",
        "take_up_contract_identity",
    }


def test_compiled_gate_is_byte_identical_to_constants_era_payloads(
    legacy_payload: dict[str, object],
) -> None:
    imputation = legacy_payload["imputation"]
    assert isinstance(imputation, dict)
    compiled_gate = {
        "source_manifest": legacy_payload["source_manifest"],
        "support_spine": legacy_payload["support_spine"],
        "take_up_contract": legacy_payload["take_up_contract"],
        "take_up_contract_identity": legacy_payload["take_up_contract_identity"],
        "stacked_authority_receipt": legacy_payload["stacked_authority_receipt"],
        "gap_fill_plan": imputation["gap_fill_plan"],
        "gap_fill_producer_schedule_receipt": imputation[
            "gap_fill_producer_schedule_receipt"
        ],
        "late_producer_schedule_receipt": imputation[
            "late_producer_schedule_receipt"
        ],
        "overlap_ownership": imputation["overlap_ownership"],
    }
    live_gate = {
        "source_manifest": json.loads(
            (US_PACKAGE / "source_stages.json").read_text(encoding="utf-8")
        ),
        "support_spine": json.loads(
            (US_PACKAGE / "support_spine.json").read_text(encoding="utf-8")
        ),
        "take_up_contract": json.loads(
            (US_PACKAGE / "take_up_contract.json").read_text(encoding="utf-8")
        ),
        "take_up_contract_identity": take_up_contract_identity(),
        "stacked_authority_receipt": dict(stacked_spine_authority_receipt()),
        "gap_fill_plan": _json_ready(stacked_gap_fill_plan()),
        "gap_fill_producer_schedule_receipt": _json_ready(
            stacked_gap_fill_producer_schedule_receipt()
        ),
        "late_producer_schedule_receipt": _json_ready(
            us_late_producer_schedule_receipt()
        ),
        "overlap_ownership": _json_ready(us_late_overlap_ownership_receipt()),
    }

    assert_legacy_payload_equal(live_gate, compiled_gate)
    assert canonical_json_bytes(compiled_gate) == canonical_json_bytes(live_gate)


def test_adapter_preserves_generation_zero_identity_components(
    legacy_payload: dict[str, object],
) -> None:
    imputation = legacy_payload["imputation"]
    assert isinstance(imputation, dict)
    assert legacy_payload["stacked_authority_receipt"]["sha256"] == (
        "f0b676f6508dbf6bb2b787c42e6b85331bacc57c6649ac7ad15fdaa5884a1b2d"
    )
    assert imputation["late_producer_schedule_receipt"]["schedule_sha256"] == (
        "b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5"
    )
    assert imputation["overlap_ownership"]["sha256"] == (
        "5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356"
    )
    assert legacy_payload["take_up_contract_identity"]["resource_sha256"] == (
        "495dc6ed195eae372a6ba098c6fb894323638a4a7dce1b4fe7efaaf6beb69446"
    )
