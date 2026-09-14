"""Canonical CLI preparation authenticates real H5 checkpoint boundaries."""

import json

import pytest
from test_uk_calibration_run import _bound_checkpoint
from test_uk_full_population_graph import source_frame
from test_uk_ladder_rowwise_clone import toy_ladder as toy_ladder

from microcosm.build.uk_runtime import full_build_cli as cli
from microcosm.build.uk_runtime import spine_build
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.graph import ContentStore, compile_graph, run_graph


@pytest.fixture
def checkpoint_request(tmp_path, toy_ladder, monkeypatch):
    pytest.importorskip("tables")
    _, ladder_path = toy_ladder
    path = write_uk_national_frame(source_frame(), tmp_path / "spine.h5")
    frame, _ = load_uk_national_frame(path)
    sidecar_path, gates_path, sidecar = _bound_checkpoint(tmp_path, frame)
    sidecar["stages"] = ["frs_spine"]
    sidecar["sampling"] = {"fraction": 1.0, "seed": 7}
    sidecar_path.write_text(json.dumps(sidecar))
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "facts.csv").write_text("fixture-only; unused during preparation")
    monkeypatch.setattr(spine_build, "_rules_engine", lambda: object())
    monkeypatch.setattr(
        spine_build, "_rules_engine_provenance", lambda: {"version": "fixture"}
    )
    args = cli.parse_args(
        [
            "--input-h5",
            str(path),
            "--input-sidecar",
            str(sidecar_path),
            "--input-spine-gates",
            str(gates_path),
            "--ladder",
            str(ladder_path),
            "--ledger-facts",
            str(ledger),
            "--out",
            str(tmp_path / "out"),
            "--n-clones",
            "1",
            "--epochs",
            "8",
        ]
    )
    return args, sidecar_path, gates_path


def test_prepare_real_checkpoint_preserves_source_year_and_wires_preflight(
    checkpoint_request, tmp_path
):
    args, _, _ = checkpoint_request
    prepared = cli.prepare_full_build(args)
    assert prepared.full.config.source_year == 2023
    assert prepared.full.config.geography_levels is None
    dense = prepared.full.graph.node("uk.full.dense")
    assert any(
        a.name == "preflight" and a.producer == "uk.full.gates.preflight"
        for a in dense.artifact_inputs
    )
    endpoint = cli._through(prepared.full.graph, "uk.full.spine_checkpoint")
    store = ContentStore(tmp_path / "store")
    first = run_graph(
        compile_graph(endpoint),
        sources=prepared.sources,
        store=store,
        kernels=prepared.kernels,
    )
    provenance_key = first.nodes["uk.full.spine_checkpoint"].opaque_artifacts[
        "spine_provenance"
    ]
    provenance = json.loads(store.load_bytes(provenance_key))
    assert provenance["stages"] == ["frs_spine"]
    assert provenance["fit_weight_records"] == {"model": {"fit_weights_used": True}}
    fresh = cli.prepare_full_build(args)
    replay = run_graph(
        compile_graph(cli._through(fresh.full.graph, "uk.full.spine_checkpoint")),
        sources=fresh.sources,
        store=store,
        kernels=fresh.kernels,
        resume="require",
    )
    assert replay.nodes["uk.full.spine_checkpoint"].hit
    assert not args.out.exists()


@pytest.mark.parametrize("damage", ["identity", "gate_bytes"])
def test_prepare_refuses_checkpoint_drift_before_registering_a_full_build(
    checkpoint_request, damage
):
    args, sidecar_path, gates_path = checkpoint_request
    if damage == "identity":
        value = json.loads(sidecar_path.read_text())
        value["uk_frame_content_identity"] = "f" * 64
        sidecar_path.write_text(json.dumps(value))
    else:
        gates_path.write_text(gates_path.read_text() + "\n")
    with pytest.raises(ValueError, match="identity mismatch|SHA-256 mismatch"):
        cli.prepare_full_build(args)
    assert not args.out.exists()
