"""Attested schema-9 pools keep #767's immigration contracts at schema 16.

Authenticated legacy scoring compares a historical pool's sealed late-producer
schedule, authority and execution rows against reconstructions. Main derives
those reconstructions from today's registry and relabels them at schema 16.
Microcosm #767 changed the inputs of two producers: the immigration source
stage and its transfer group. A schema-9 pool's execution rows carry those two
producers' schema-16 input surfaces (57 and 93 inputs), which today's
contracts (56 and 100) refuse. So the legacy path takes those two inventories,
and the schema-16 execution receipt contract, from the frozen schema-16
schedule, and checks each rebuilt contract against the digest the schema-16
registry derived. It otherwise follows today's registry exactly as main does.

Main's own drift since schema 16 is out of scope here: microcosm #1239 and
#1258 changed the primary-PUF and PUF-itemization contracts, so the legacy
schedule no longer hashes to the sealed one.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.build.us_runtime import h5_io, stacked_spine
from microcosm.build.us_runtime import us_late_producer_registry as registry
from microcosm.build.us_runtime.late_producer_dag import _contract_payload

#: The sealed content hash of the frozen schema-16 schedule, computed on main
#: at 7c4a5ac0 (the base of microcosm #779), which reproduced it.
HISTORICAL_SCHEDULE_PAYLOAD_SHA256 = (
    "02e618cc656eb39990ed99dca2b30a52794e01e2b06a3c2df87ca4a7d85ab086"
)
IMMIGRATION_SOURCE = "source:with_us_immigration_inputs"
IMMIGRATION_TRANSFER = "transfer:person/source_operator_immigration"
#: Canonical-JSON SHA-256 of the two contracts as main at 7c4a5ac0 derived
#: them; that registry reproduced the frozen schedule's ``schedule_sha256``.
HISTORICAL_IMMIGRATION_CONTRACT_SHA256 = {
    IMMIGRATION_SOURCE: (
        "8ffdcf5128e54956ea63669033cddcb894af93333fdf61e2262d9f54105ec5af"
    ),
    IMMIGRATION_TRANSFER: (
        "81c5bdc4567090d8c68aac8101214b9f2db3a16485cc14521b81e530a0860f96"
    ),
}
SCHEMA16_TOP_BINDING = (
    "entry_and_output_frame_sha256_execution_chain_source_completion_"
    "and_nineteen_transfer_groups"
)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _frozen() -> dict[str, object]:
    return json.loads(registry._verified_legacy_schedule_json())


def _row(rows, operator: str) -> dict[str, object]:
    return next(row for row in rows if row["operator"] == operator)


def _labels(row) -> set[str]:
    return {requirement["label"] for requirement in row["requirements"]}


def _clear_legacy_caches() -> None:
    registry._verified_legacy_schedule_json.cache_clear()
    registry.legacy_us_late_producer_contracts.cache_clear()
    registry._legacy_schedule_receipt_json.cache_clear()


def test_frozen_schedule_resource_is_the_sealed_historical_content() -> None:
    frozen = _frozen()
    payload = {
        key: value
        for key, value in frozen.items()
        if key not in registry._LEGACY_SCHEDULE_DERIVED_KEYS
    }
    assert _canonical_sha256(payload) == HISTORICAL_SCHEDULE_PAYLOAD_SHA256
    assert frozen["payload_sha256"] == HISTORICAL_SCHEDULE_PAYLOAD_SHA256
    assert (
        registry.LEGACY_SCHEMA16_LATE_PRODUCER_SCHEDULE_PAYLOAD_SHA256
        == HISTORICAL_SCHEDULE_PAYLOAD_SHA256
    )
    assert frozen["schema_version"] == 16


def test_legacy_receipt_carries_the_schema16_receipt_contract() -> None:
    legacy = registry.legacy_us_late_producer_schedule_receipt()
    live = registry.us_late_producer_schedule_receipt()
    assert legacy["schema_version"] == 16
    assert (
        legacy["execution_receipt_contract"] == _frozen()["execution_receipt_contract"]
    )
    contract = legacy["execution_receipt_contract"]
    assert contract["version"] == 3
    assert contract["transition_authority"]["version"] == 1
    assert contract["top_binding"] == SCHEMA16_TOP_BINDING
    assert live["execution_receipt_contract"]["top_binding"] != SCHEMA16_TOP_BINDING


def test_legacy_receipt_payload_hash_is_a_real_content_hash() -> None:
    receipt = dict(registry.legacy_us_late_producer_schedule_receipt())
    payload = {
        key: value
        for key, value in receipt.items()
        if key not in registry._LEGACY_SCHEDULE_DERIVED_KEYS
    }
    assert _canonical_sha256(payload) == receipt["payload_sha256"]
    assert receipt["schedule_sha256"] == (
        registry.legacy_us_late_producer_contracts().schedule.sha256
    )


def test_legacy_receipt_differs_from_live_only_in_767s_surface() -> None:
    legacy = dict(registry.legacy_us_late_producer_schedule_receipt())
    live = json.loads(json.dumps(dict(registry.us_late_producer_schedule_receipt())))
    assert set(legacy) == set(live)
    changed = {key for key in live if legacy[key] != live[key]}
    assert changed <= {
        "schema_version",
        "execution_receipt_contract",
        "schedule_sha256",
        "payload_sha256",
        "source_input_inventories",
        "transfer_input_inventories",
        "edges",
        "waves",
    }
    assert {
        "schema_version",
        "execution_receipt_contract",
        "schedule_sha256",
        "payload_sha256",
        "source_input_inventories",
        "transfer_input_inventories",
    } <= changed
    assert legacy["order"] == live["order"]
    for key, operator in (
        ("source_input_inventories", "with_us_immigration_inputs"),
        ("transfer_input_inventories", IMMIGRATION_TRANSFER),
    ):
        differing = {
            row["operator"]
            for row in live[key]
            if row != _row(legacy[key], row["operator"])
        }
        assert differing == {operator}


def test_legacy_immigration_inventories_are_the_frozen_rows() -> None:
    frozen = _frozen()
    legacy = registry.legacy_us_late_producer_contracts()
    source = registry._inventory_payload(
        legacy.source_inventories["with_us_immigration_inputs"]
    )
    transfer = registry._inventory_payload(
        legacy.transfer_inventories[IMMIGRATION_TRANSFER]
    )
    assert source == _row(
        frozen["source_input_inventories"], "with_us_immigration_inputs"
    )
    assert transfer == _row(frozen["transfer_input_inventories"], IMMIGRATION_TRANSFER)
    live = registry._inventory_payload(
        registry.US_LATE_SOURCE_INPUT_INVENTORIES["with_us_immigration_inputs"]
    )
    # #767 replaced the wage and self-employment reads with the ASEC labor
    # force recode; the schema-16 row keeps the reads those pools sealed.
    assert {"raw_person:WSAL_VAL", "raw_person:SEMP_VAL"} <= _labels(source)
    assert "raw_person:A_LFSR" not in _labels(source)
    assert "raw_person:A_LFSR" in _labels(live)


def test_legacy_contracts_differ_from_live_exactly_at_767s_two_producers() -> None:
    legacy = registry.legacy_us_late_producer_contracts().registry
    live = registry.CANONICAL_US_LATE_PRODUCER_REGISTRY
    assert set(legacy) == set(live)
    changed = {name for name in live if legacy[name] != live[name]}
    assert changed == {IMMIGRATION_SOURCE, IMMIGRATION_TRANSFER}
    assert len(legacy[IMMIGRATION_SOURCE].inputs) == 57
    assert len(live[IMMIGRATION_SOURCE].inputs) == 56
    assert len(legacy[IMMIGRATION_TRANSFER].inputs) == 93
    assert len(live[IMMIGRATION_TRANSFER].inputs) == 100
    for name in (IMMIGRATION_SOURCE, IMMIGRATION_TRANSFER):
        assert legacy[name].outputs == live[name].outputs
        assert legacy[name].kind == live[name].kind


def test_legacy_immigration_contracts_hash_to_their_sealed_digests() -> None:
    legacy = registry.legacy_us_late_producer_contracts().registry
    assert dict(registry._LEGACY_SCHEMA16_SEALED_CONTRACT_SHA256) == (
        HISTORICAL_IMMIGRATION_CONTRACT_SHA256
    )
    for name, expected in HISTORICAL_IMMIGRATION_CONTRACT_SHA256.items():
        assert _canonical_sha256(_contract_payload(legacy[name])) == expected
        assert (
            _canonical_sha256(
                _contract_payload(registry.CANONICAL_US_LATE_PRODUCER_REGISTRY[name])
            )
            != expected
        )


def test_legacy_authority_binds_the_legacy_schedule() -> None:
    authority = stacked_spine._legacy_stacked_authority_receipt()
    assert authority["version"] == 11
    component = authority["components"]["late_producer_schedule"]
    legacy = stacked_spine._json_ready(
        registry.legacy_us_late_producer_schedule_receipt()
    )
    assert stacked_spine._json_ready(component["identity"]) == legacy
    assert component["schedule_sha256"] == legacy["schedule_sha256"]
    live = stacked_spine._json_ready(registry.us_late_producer_schedule_receipt())
    assert stacked_spine._json_ready(component["identity"]) != live


def test_tampered_frozen_schedule_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry,
        "LEGACY_SCHEMA16_LATE_PRODUCER_SCHEDULE_PAYLOAD_SHA256",
        "0" * 64,
    )
    _clear_legacy_caches()
    try:
        with pytest.raises(ValueError, match="sealed content hash"):
            registry.legacy_us_late_producer_schedule_receipt()
    finally:
        _clear_legacy_caches()


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


def test_each_caller_gets_its_own_legacy_schedule() -> None:
    first = registry.legacy_us_late_producer_schedule_receipt()
    first["execution_receipt_contract"]["version"] = 99
    first["source_input_inventories"].clear()

    second = registry.legacy_us_late_producer_schedule_receipt()
    assert second["execution_receipt_contract"]["version"] == 3
    assert second["source_input_inventories"]


def test_inventory_payload_round_trips_exactly() -> None:
    for inventories in (
        registry.US_LATE_SOURCE_INPUT_INVENTORIES,
        registry.US_LATE_TRANSFER_INPUT_INVENTORIES,
    ):
        for name, inventory in inventories.items():
            assert inventory.operator == name
            payload = registry._inventory_payload(inventory)
            parsed = registry._inventory_from_payload(payload)
            assert parsed == inventory
            assert registry._inventory_payload(parsed) == payload


_TOKEN = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_:@", min_size=1, max_size=12)
_SCOPED_INPUT = st.builds(
    registry.ScopedInput,
    _TOKEN,
    _TOKEN,
    st.sampled_from(("column_present", "non_null", "finite_numeric")),
)
# Constructors canonicalize: alternatives are deduplicated and sorted, so two
# alternatives over the same columns are one repeated alternative (refused),
# and requirement labels must be unique within an inventory.
_REQUIREMENT = st.builds(
    registry.EffectiveInputRequirement,
    label=_TOKEN,
    optional=st.booleans(),
    required_scope=st.one_of(st.none(), _TOKEN),
    alternatives=st.lists(
        st.lists(_SCOPED_INPUT, min_size=1, max_size=3).map(tuple),
        min_size=1,
        max_size=3,
        unique_by=frozenset,
    ).map(tuple),
)
_INVENTORY = st.builds(
    registry.SourceInputInventory,
    operator=_TOKEN,
    requirements=st.lists(
        _REQUIREMENT, min_size=1, max_size=4, unique_by=lambda item: item.label
    ).map(tuple),
)


@given(_INVENTORY)
def test_inventory_payload_round_trip_property(inventory) -> None:
    payload = registry._inventory_payload(inventory)
    assert registry._inventory_from_payload(payload) == inventory
    assert registry._inventory_from_payload(json.loads(json.dumps(payload))) == (
        inventory
    )


def test_legacy_contracts_refuse_a_drifted_immigration_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_inputs = registry._inventory_contract_inputs

    def drifted_inputs(node_name, inventory, *, required_scope):
        inputs = live_inputs(node_name, inventory, required_scope=required_scope)
        return inputs[1:] if node_name == IMMIGRATION_SOURCE else inputs

    monkeypatch.setattr(registry, "_inventory_contract_inputs", drifted_inputs)
    _clear_legacy_caches()
    try:
        with pytest.raises(ValueError, match="no longer matches the schema-16"):
            registry.legacy_us_late_producer_contracts()
    finally:
        _clear_legacy_caches()


@pytest.mark.parametrize("fault", ["missing", "not_round_tripping"])
def test_legacy_contracts_refuse_an_unusable_sealed_inventory(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "missing":
        monkeypatch.setattr(
            registry,
            "LEGACY_SCHEMA16_SEALED_SOURCE_INVENTORIES",
            ("not_a_source_operator",),
        )
        message = "has no 'not_a_source_operator' inventory"
    else:
        live_parse = registry._inventory_from_payload

        def lossy_parse(payload):
            inventory = live_parse(payload)
            return registry.SourceInputInventory(
                operator=inventory.operator,
                requirements=inventory.requirements[1:],
            )

        monkeypatch.setattr(registry, "_inventory_from_payload", lossy_parse)
        message = "does not round-trip"
    _clear_legacy_caches()
    try:
        with pytest.raises(ValueError, match=message):
            registry.legacy_us_late_producer_contracts()
    finally:
        _clear_legacy_caches()
