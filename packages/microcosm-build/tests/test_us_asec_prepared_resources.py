"""Packaged admission evidence; no survey data or rules-engine execution."""

from hashlib import sha256
from importlib import resources

import pytest

from microcosm.build.us_runtime import asec_engine_evaluation as evaluation
from microcosm.build.us_runtime.graph_implementation import implementation_manifest

# These are the exact bytes re-admitted when the US engine lock moved to
# policyengine-us 2.2.1 / policyengine-core 3.32.5 (#936, merged into this
# integration branch on 17 September 2026), re-derived from the installed
# distributions through `asec_engine_evaluation._runtime_package_identity` —
# the same code `engine_runtime_identity()` asserts against. They are a pinned
# admission record, not newly observed identities of whatever engine a test
# environment happens to install; a different installed engine must fail this
# assertion rather than silently redefine it. The superseded 1.819.0 / 3.31.0
# record is b0e7f54b28145ee920ad0f4c5d0e2090c8b0785a.
_RESTORED_RESOURCE_SHA256 = (
    "448fe39eb6f8237a7b7da457ac76e7f81c666987a3dee0c37390861f60498626"
)
_RESTORED_RESOURCES = {
    "asec_current_money_engine_defaults_v1.json": _RESTORED_RESOURCE_SHA256,
    # Exact historical 4cfcd0723 and 41f2a7f72 resources, respectively.
    "asec_current_money_graph_consumers_v1.json": (
        "67432a10d05e11dd4b97a80cd4f04ed0eb6e56c4c8e21d5dffe87c5f67e27c66"
    ),
    "asec_income_observations_v1.json": (
        "6e37855ddd6642d02231efc664644029ed6131885eb743a00197261bc5a6a51f"
    ),
}


def test_packaged_defaults_restore_the_admitted_evidence_without_new_claims():
    payload = (
        resources.files("microcosm.build.us_runtime")
        .joinpath(evaluation.ENGINE_DEFAULTS_RESOURCE)
        .read_bytes()
    )
    assert sha256(payload).hexdigest() == _RESTORED_RESOURCE_SHA256
    assert evaluation.ENGINE_DEFAULTS_SHA256 == _RESTORED_RESOURCE_SHA256
    document = evaluation.load_engine_defaults()
    assert document["allowlist"] == []
    assert document["release_eligible"] is False
    assert tuple(document["admitted_roots"]) == evaluation.ADMITTED_ROOTS
    assert tuple(sorted(document["blocked_roots"])) == evaluation.BLOCKED_ENGINE_OUTPUTS
    assert document["baseline_runtime"]["policyengine-us"]["version"] == "2.2.1"
    assert document["baseline_runtime"]["policyengine-core"]["version"] == "3.32.5"
    assert all(
        not any(
            marker in leaf for marker in ("last_year", "previous_year", "prior_year")
        )
        for contract in evaluation.ADMITTED_ENGINE_OUTPUT_CONTRACTS
        for leaf in contract.input_leaves
    )


@pytest.mark.parametrize(
    "stage",
    ("asec_prepared_v3", "composed_asec_binding_v1", "composed_population_v1"),
)
def test_prepared_stage_manifest_binds_the_packaged_resources(stage):
    pytest.importorskip("microunit")
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    manifest = implementation_manifest(stage)
    for name, fingerprint in _RESTORED_RESOURCES.items():
        assert (
            manifest["resources"][f"microcosm.build/us_runtime/{name}"] == fingerprint
        )


def test_restored_graph_consumer_status_keeps_its_source_parent_and_scope():
    from microcosm.build.us_runtime.asec_current_money_graph_resources import (
        load_graph_current_money_consumers,
    )

    document = load_graph_current_money_consumers()
    assert document["scope"] == "current_money_only_no_prior_wages"
    assert document["engine_defaults_allowlist"] == []
    assert document["release_eligible"] is False
    assert document["all_current_money_consumers_wired"] is False


def test_restored_income_contract_preserves_reported_source_semantics():
    from microcosm.build.us_runtime.asec_income_observations import _contract

    document = _contract()
    assert set(document["fields"]) == {"PAW_VAL", "PAW_YN"}
    assert document["consumer"]["semantic"] == (
        "reported_cash_public_assistance_not_simulated_tanf"
    )
    assert document["consumer"]["raw_frame_aliases"] == []
    assert document["consumer"]["release_eligible"] is False


@pytest.mark.parametrize("alteration", ("missing", "tampered"))
def test_defaults_never_fall_back_when_packaged_evidence_is_unavailable(
    tmp_path, monkeypatch, alteration
):
    payload = (
        resources.files("microcosm.build.us_runtime")
        .joinpath(evaluation.ENGINE_DEFAULTS_RESOURCE)
        .read_bytes()
    )
    monkeypatch.setattr(evaluation.resources, "files", lambda package: tmp_path)
    if alteration == "tampered":
        (tmp_path / evaluation.ENGINE_DEFAULTS_RESOURCE).write_bytes(payload + b"\n")
        with pytest.raises(
            evaluation.EngineAdmissionError, match="ENGINE_DEFAULTS_FINGERPRINT"
        ):
            evaluation.load_engine_defaults()
    else:
        with pytest.raises(FileNotFoundError):
            evaluation.load_engine_defaults()
