"""Invented-fixture checks for the SIPP nominal December-2022 equity consumer.

No genuine SIPP parent, projection, receipt or content store is opened here.
Every byte under test is authored in ``us_sipp_equity_fixture``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import us_sipp_equity_fixture as fx

import microcosm.build.sipp_december_readback as readback
import microcosm.build.sipp_december_source as source_module
import microcosm.build.sipp_equity_contract as contract_module
import microcosm.build.sipp_equity_graph as equity
from microcosm.graph.errors import StoreCorruptError
from microcosm.graph.store import ContentStore

TARGET = contract_module.TARGET_NAME
PREDICTOR = contract_module.PREDICTOR_NAME
ROSTER = contract_module.ROSTER_PREDICTOR_NAME


@pytest.fixture
def accepted(tmp_path):
    root, contract, projection, receipt = fx.accepted_store(tmp_path)
    return root, contract, projection, receipt


def _scratch(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def _prepared(tmp_path, root, contract, name="source-store"):
    return equity.prepare_source_store(root, tmp_path / name, contract=contract)


def _donor_frame(tmp_path, accepted, name="source-store"):
    root, contract, _, _ = accepted
    return equity.load_sipp_equity_donor_frame(
        _prepared(tmp_path, root, contract, name), contract=contract
    )


# --- packaged contract -----------------------------------------------------
def test_packaged_contract_is_importable_and_pins_the_accepted_identities():
    contract = contract_module.load_contract()
    pins = contract.source
    assert pins["receipt_bytes"] == 6183
    assert (
        pins["receipt_sha256"]
        == "ce89d7712f61907754896a41d38bff58c3aaa647e886d2a7a91ac70b3bdad7bf"
    )
    assert (
        pins["producer_binding"]
        == "3905d7a931d61c971143f676b9961c288f2cbf81c67fa381e9888aac9834a574"
    )
    assert pins["projection_bytes"] == 4287344
    assert contract.admission["expected_admitted_residences"] == 17588
    assert contract.admission["expected_excluded_zero_weight"] == 20
    assert contract.admission["expected_excluded_unknown_target"] == 38
    assert contract.model == {
        "kernel": "fit.qrf.train@1",
        "apply_kernel": "fit.qrf.apply@1",
        "n_estimators": 2,
        "seed": 0,
        "zero_atol": 0.0,
        "max_samples_leaf": None,
    }
    assert len(contract.identity) == 64


def test_a_2024_or_wrong_concept_request_refuses_before_any_source_or_kernel():
    contract = contract_module.load_contract()
    with pytest.raises(contract_module.ContractError, match="YEAR_BRIDGE_REQUIRED"):
        equity.build_sipp_equity_graph(concept=TARGET, period=2024)
    with pytest.raises(contract_module.ContractError, match="UNSUPPORTED_CONCEPT"):
        equity.build_sipp_equity_graph(concept="household_net_worth", period=2022)
    with pytest.raises(contract_module.ContractError):
        contract.require_request(concept=TARGET, period=True)


# --- readback --------------------------------------------------------------
def test_readback_reconstructs_the_full_roster_and_measurement_digests(
    tmp_path, accepted
):
    root, contract, _, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    result = readback.read_sipp_december_projection(
        ContentStore(prepared), _scratch(tmp_path, "scratch-ok"), contract=contract
    )
    assert len(result.households) == 15
    assert result.producer_binding == fx.INVENTED_BINDING


def test_a_changed_same_size_projection_payload_refuses(tmp_path, accepted):
    """Both routes refuse: the store's own integrity check, and the pin check."""
    root, contract, projection, receipt = accepted
    changed = fx.mutate_same_size(projection)
    assert len(changed) == len(projection)

    prepared = _prepared(tmp_path, root, contract, "s-inplace")
    obj = ContentStore(prepared).object_path(contract.source["projection_key"])
    (obj / "payload.bin").write_bytes(changed)
    with pytest.raises(Exception) as inplace:
        readback.read_sipp_december_projection(
            ContentStore(prepared), _scratch(tmp_path, "scratch1"), contract=contract
        )
    assert "payload.bin" in str(inplace.value)

    # A store that is internally consistent with the changed bytes still fails
    # the packaged pin: caller-supplied metadata confers no authority.
    consistent = tmp_path / "consistent"
    forged_receipt = fx.build_receipt(changed, fx.invented_rows())
    fx.publish(consistent, changed, forged_receipt)
    republished = equity.prepare_source_store(
        consistent,
        tmp_path / "s-consistent",
        contract=_contract_for(forged_receipt, changed),
    )
    with pytest.raises(readback.ReadbackError, match="RECEIPT_PIN|PROJECTION_PIN"):
        readback.read_sipp_december_projection(
            ContentStore(republished),
            _scratch(tmp_path, "scratch1b"),
            contract=contract,
        )


def _contract_for(receipt, projection):
    import microcosm.build.sipp_december_source as source_module

    return fx.invented_contract(
        receipt,
        projection,
        source_module._key(
            "sipp-december-receipt/1",
            (fx.INVENTED_BINDING, source_module._sha(receipt), len(receipt)),
        ),
        json.loads(receipt)["projection"]["key"],
    )


def test_forged_object_metadata_refuses(tmp_path, accepted):
    root, contract, _, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    fx.forge_metadata(prepared, contract.source["receipt_key"], node_key="a" * 64)
    with pytest.raises((source_module.SourceError, readback.ReadbackError)):
        readback.read_sipp_december_projection(
            ContentStore(prepared), _scratch(tmp_path, "scratch2"), contract=contract
        )


def test_a_receipt_stored_under_the_wrong_kind_refuses(tmp_path, accepted):
    root, contract, projection, receipt = accepted
    prepared = _prepared(tmp_path, root, contract)
    obj = ContentStore(prepared).object_path(contract.source["receipt_key"])
    meta = json.loads((obj / "meta.json").read_bytes())
    meta["kind"] = "not_the_declared_kind"
    (obj / "meta.json").write_bytes(json.dumps(meta, sort_keys=True).encode())
    with pytest.raises((StoreCorruptError, readback.ReadbackError)):
        readback.read_sipp_december_projection(
            ContentStore(prepared), _scratch(tmp_path, "scratch3"), contract=contract
        )


def test_receipt_counts_inconsistent_with_the_reconstructed_rows_refuse(tmp_path):
    counts = {
        "december_people": 999,
        "december_residences": 15,
        "december_ssuids": 14,
        "multi_residence_ssuids": 1,
        "zero_weight_december_people": 3,
    }
    root, contract, _, _ = fx.accepted_store(tmp_path, counts=counts)
    prepared = equity.prepare_source_store(root, tmp_path / "s", contract=contract)
    with pytest.raises(readback.ReadbackError, match="COUNTS"):
        readback.read_sipp_december_projection(
            ContentStore(prepared), _scratch(tmp_path, "scratch4"), contract=contract
        )


def test_scratch_inside_the_source_store_refuses(tmp_path, accepted):
    root, contract, _, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    inside = prepared / "tmp"
    with pytest.raises(readback.ReadbackError, match="SCRATCH_INSIDE_SOURCE_STORE"):
        readback.read_sipp_december_projection(
            ContentStore(prepared), inside, contract=contract
        )


# --- prepared source store -------------------------------------------------
def test_the_prepared_store_has_both_directories_and_only_two_objects(
    tmp_path, accepted
):
    root, contract, _, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    assert sorted(p.name for p in prepared.iterdir()) == ["objects", "tmp"]
    assert not any((prepared / "tmp").iterdir())
    keys = {
        obj.name
        for shard in (prepared / "objects").iterdir()
        for obj in shard.iterdir()
    }
    assert keys == {contract.source["receipt_key"], contract.source["projection_key"]}


def test_an_extra_member_in_the_source_store_refuses(tmp_path, accepted):
    root, contract, _, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    (prepared / "unexpected.txt").write_text("x")
    with pytest.raises(equity.EquityGraphError, match="SOURCE_STORE_MEMBERS"):
        equity.load_sipp_equity_donor_frame(prepared, contract=contract)


def test_a_third_stored_object_refuses(tmp_path, accepted):
    root, contract, projection, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    ContentStore(prepared).put_bytes("b" * 64, b"extra", node_key=fx.INVENTED_BINDING)
    with pytest.raises(equity.EquityGraphError, match="SOURCE_STORE_ROSTER"):
        equity.load_sipp_equity_donor_frame(prepared, contract=contract)


def test_a_nonempty_tmp_refuses(tmp_path, accepted):
    root, contract, _, _ = accepted
    prepared = _prepared(tmp_path, root, contract)
    (prepared / "tmp" / "leftover").write_text("x")
    with pytest.raises(equity.EquityGraphError, match="TMP_NOT_EMPTY"):
        equity.load_sipp_equity_donor_frame(prepared, contract=contract)


# --- admission and coverage ------------------------------------------------
def test_admission_keeps_signed_targets_and_positive_source_weights(tmp_path, accepted):
    frame = _donor_frame(tmp_path, accepted)
    table = frame.table("household")
    targets = table[TARGET].to_numpy()
    assert len(table) == fx.ADMITTED_RESIDENCES
    assert (targets < 0).any() and (targets == 0).any() and (targets > 0).any()
    weights = frame.weights_for("household")
    assert weights.kind.value == "design"
    assert (weights.values > 0).all()
    assert weights.values.dtype == np.float64
    assert sorted(set(int(v) for v in table[PREDICTOR])) == fx.ADMITTED_SIZES


def test_every_excluded_residence_stays_in_the_full_coverage_report(tmp_path, accepted):
    frame = _donor_frame(tmp_path, accepted)
    metadata = frame.metadata
    admission = metadata["admission"]
    coverage = metadata["coverage"]
    assert coverage["roster"]["december_residences"] == 15
    assert admission["denominator_residences"] == 15
    assert admission["admitted_residences"] == fx.ADMITTED_RESIDENCES
    reasons = dict(admission["excluded_reasons_exclusive"])
    assert reasons == {
        "target_unknown": fx.EXCLUDED_UNKNOWN_TARGET,
        "weight_not_positive": fx.EXCLUDED_ZERO_WEIGHT,
        "weight_unknown": fx.EXCLUDED_UNKNOWN_WEIGHT,
    }
    assert len(admission["excluded"]) == 15 - fx.ADMITTED_RESIDENCES
    # An unknown weight is unknown, never zero.
    assert coverage["household_weights"]["unknown_count"] == 1
    assert coverage["household_weights"]["zero_count"] == 1


def test_flag_9_is_retained_as_component_derived_and_never_relabelled(
    tmp_path, accepted
):
    frame = _donor_frame(tmp_path, accepted)
    coverage = frame.metadata["coverage"]
    qualities = dict(coverage["target"]["quality_observations"])
    assert qualities["component_derived_allocation_unresolved"] > 0
    assert "reported" not in coverage["flag_9_policy"]
    assert coverage["vehicle_count_is_not_an_admission_requirement"] is True
    assert "TVEH_NUM" in coverage["diagnostics"]


def test_exact_decimal_evidence_survives_beside_the_checked_float64_weights(
    tmp_path, accepted
):
    frame = _donor_frame(tmp_path, accepted)
    exact = frame.metadata["exact_source_weights"]
    weights = frame.weights_for("household").values
    assert len(exact) == len(weights)
    assert all(
        float(value) == observed for value, observed in zip(exact, weights, strict=True)
    )
    assert frame.metadata["weight_is_calibrated"] is False


def test_donor_lineage_is_deterministic_and_recorded(tmp_path, accepted):
    first = _donor_frame(tmp_path, accepted, "s-a")
    second = _donor_frame(tmp_path, accepted, "s-b")
    assert first.metadata["lineage_digest"] == second.metadata["lineage_digest"]
    lineage = first.metadata["lineage"]
    assert len(lineage["households"]) == fx.ADMITTED_RESIDENCES
    assert {entry["ssuid"] for entry in lineage["households"]}


def test_a_multi_residence_ssuid_produces_two_admitted_donors(tmp_path, accepted):
    frame = _donor_frame(tmp_path, accepted)
    lineage = frame.metadata["lineage"]["households"]
    repeated = [e for e in lineage if e["ssuid"] == "00000010"]
    assert len(repeated) == 2
    assert {e["residence"] for e in repeated} == {100010, 100011}
