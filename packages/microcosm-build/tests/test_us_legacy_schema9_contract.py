"""The contract attested schema-9 stacked pools sealed stays frozen (#767).

Authenticated legacy scoring compares a historical pool's sealed schedule and
authority against reconstructions. Those reconstructions used to be derived
from the *current* late-producer registry, so any contract change silently
redefined history and every real schema-9 pool would be refused — while the
positive tests, which downgrade a freshly built artifact, kept passing. These
pins are the regression: the historical content is data, and its hashes are
the ones real pools carry.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from microcosm.build.us_runtime import h5_io, stacked_spine
from microcosm.build.us_runtime import us_late_producer_registry as registry

#: What pools built before manifest schema 10 actually sealed, computed on
#: main at 7c4a5ac0 (the base of microcosm #779) and reproduced here.
HISTORICAL_SCHEDULE_PAYLOAD_SHA256 = (
    "02e618cc656eb39990ed99dca2b30a52794e01e2b06a3c2df87ca4a7d85ab086"
)
HISTORICAL_V11_AUTHORITY_SHA256 = (
    "e660a8ce42b69a39d29c5f0ec37264bc69d61b03f27adc386336ec8889531bb2"
)


def test_legacy_schedule_is_the_sealed_historical_content() -> None:
    receipt = registry.legacy_us_late_producer_schedule_receipt()
    assert receipt["payload_sha256"] == HISTORICAL_SCHEDULE_PAYLOAD_SHA256
    assert (
        registry.LEGACY_SCHEMA16_LATE_PRODUCER_SCHEDULE_PAYLOAD_SHA256
        == HISTORICAL_SCHEDULE_PAYLOAD_SHA256
    )
    assert receipt["schema_version"] == 16
    contract = receipt["execution_receipt_contract"]
    assert contract["version"] == 3
    assert contract["transition_authority"]["version"] == 1


def test_legacy_schedule_payload_hash_is_a_real_content_hash() -> None:
    receipt = dict(registry.legacy_us_late_producer_schedule_receipt())
    payload = {
        key: value
        for key, value in receipt.items()
        if key not in registry._LEGACY_SCHEDULE_DERIVED_KEYS
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == HISTORICAL_SCHEDULE_PAYLOAD_SHA256


def test_legacy_schedule_does_not_follow_the_current_registry() -> None:
    # The current contract differs from history exactly where #767 changed
    # it; if these ever coincide the frozen resource has stopped mattering.
    current = registry.us_late_producer_schedule_receipt()
    legacy = registry.legacy_us_late_producer_schedule_receipt()
    assert current["payload_sha256"] != legacy["payload_sha256"]

    def immigration_inputs(receipt) -> set[str]:
        inventory = next(
            row
            for row in receipt["source_input_inventories"]
            if row["operator"] == "with_us_immigration_inputs"
        )
        return {requirement["label"] for requirement in inventory["requirements"]}

    assert {"raw_person:WSAL_VAL", "raw_person:SEMP_VAL"} <= immigration_inputs(legacy)
    assert "raw_person:A_LFSR" not in immigration_inputs(legacy)
    assert "raw_person:A_LFSR" in immigration_inputs(current)
    assert legacy["execution_receipt_contract"]["top_binding"] == (
        "entry_and_output_frame_sha256_execution_chain_source_completion_"
        "and_nineteen_transfer_groups"
    )


def test_legacy_authority_is_the_sealed_v11_authority() -> None:
    receipt = stacked_spine._legacy_stacked_authority_receipt()
    assert receipt["sha256"] == HISTORICAL_V11_AUTHORITY_SHA256


def test_tampered_frozen_schedule_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry,
        "LEGACY_SCHEMA16_LATE_PRODUCER_SCHEDULE_PAYLOAD_SHA256",
        "0" * 64,
    )
    registry.legacy_us_late_producer_schedule_receipt.cache_clear()
    try:
        with pytest.raises(ValueError, match="sealed content hash"):
            registry.legacy_us_late_producer_schedule_receipt()
    finally:
        registry.legacy_us_late_producer_schedule_receipt.cache_clear()


def test_schema9_operator_order_is_the_literal_historical_order() -> None:
    historical = h5_io._SCHEMA9_STACKED_POOL_OPERATOR_ORDER
    assert historical == (
        "assemble_stacked_spine",
        "assign_us_puma_ladder",
        "prepare_multispine_source_inputs_for_clone",
        "gap_fill_stacked_spine",
        "run_stacked_late_producer_dag",
        "prepare_stacked_tail_derivation",
        "derive_multispine_pool_inputs",
        "seed_multispine_pool_inputs",
        "materialize_multispine_agreement_outputs",
        "stacked_completeness_gate",
        "by_origin_battery",
    )
    assert "us_immigration_composition_gate" not in historical
    assert h5_io.US_STACKED_POOL_OPERATOR_ORDER[-1] == "us_immigration_composition_gate"
