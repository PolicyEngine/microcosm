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
    registry._verified_legacy_schedule_json.cache_clear()
    try:
        with pytest.raises(ValueError, match="sealed content hash"):
            registry.legacy_us_late_producer_schedule_receipt()
    finally:
        registry._verified_legacy_schedule_json.cache_clear()


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


def test_each_caller_gets_its_own_frozen_schedule() -> None:
    first = registry.legacy_us_late_producer_schedule_receipt()
    first["execution_receipt_contract"]["version"] = 99

    second = registry.legacy_us_late_producer_schedule_receipt()
    assert second["execution_receipt_contract"]["version"] == 3


# ---------------------------------------------------------------------------
# The execution contracts schema-9 pools sealed (#767, second review round)
# ---------------------------------------------------------------------------

IMMIGRATION_SOURCE = "source:with_us_immigration_inputs"
IMMIGRATION_TRANSFER = "transfer:person/source_operator_immigration"


def test_inventory_payload_round_trips_exactly() -> None:
    for inventories in (
        registry.US_LATE_SOURCE_INPUT_INVENTORIES,
        registry.US_LATE_TRANSFER_INPUT_INVENTORIES,
    ):
        for name, inventory in inventories.items():
            assert inventory.operator == name
            parsed = registry._inventory_from_payload(
                registry._inventory_payload(inventory)
            )
            assert parsed == inventory


def test_legacy_contracts_reproduce_the_frozen_schedule() -> None:
    legacy = registry.legacy_us_late_producer_contracts()
    frozen = registry.legacy_us_late_producer_schedule_receipt()
    assert legacy.schedule.sha256 == frozen["schedule_sha256"]
    assert list(legacy.schedule.order) == frozen["order"]
    assert set(legacy.registry) == set(registry.CANONICAL_US_LATE_PRODUCER_REGISTRY)


def test_legacy_contracts_differ_from_live_only_where_767_changed_them() -> None:
    legacy = registry.legacy_us_late_producer_contracts().registry
    live = registry.CANONICAL_US_LATE_PRODUCER_REGISTRY
    changed = {name for name in live if legacy[name] != live[name]}
    assert changed == {IMMIGRATION_SOURCE, IMMIGRATION_TRANSFER}
    # The historical input surfaces schema-9 pools sealed, which today's
    # contracts would refuse as "not the exact N-input readiness surface".
    assert len(legacy[IMMIGRATION_SOURCE].inputs) == 57
    assert len(live[IMMIGRATION_SOURCE].inputs) == 56
    assert len(legacy[IMMIGRATION_TRANSFER].inputs) == 93
    assert len(live[IMMIGRATION_TRANSFER].inputs) == 100
    for name in (IMMIGRATION_SOURCE, IMMIGRATION_TRANSFER):
        assert legacy[name].outputs == live[name].outputs
        assert legacy[name].kind == live[name].kind


@pytest.mark.parametrize("drift", ["overlap_ownership", "contract_inputs"])
def test_legacy_contracts_refuse_a_drifted_derivation(
    monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    if drift == "overlap_ownership":
        live = registry.us_late_overlap_ownership_receipt

        def drifted_ownership():
            receipt = dict(live())
            receipt["schema_version"] = -1
            return receipt

        monkeypatch.setattr(
            registry, "us_late_overlap_ownership_receipt", drifted_ownership
        )
    else:
        live_inputs = registry._inventory_contract_inputs

        def drifted_inputs(node_name, inventory, *, required_scope):
            return live_inputs(node_name, inventory, required_scope=required_scope)[1:]

        monkeypatch.setattr(registry, "_inventory_contract_inputs", drifted_inputs)
    registry.legacy_us_late_producer_contracts.cache_clear()
    try:
        with pytest.raises(ValueError, match="no longer reproduces the frozen"):
            registry.legacy_us_late_producer_contracts()
    finally:
        registry.legacy_us_late_producer_contracts.cache_clear()
