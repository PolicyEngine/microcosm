"""Canonical CLI preparation authenticates real H5 checkpoint boundaries."""

import hashlib
import json

import pytest

from microcosm.build.uk_runtime import full_build_cli as cli
from microcosm.build.uk_runtime import hmrc_uprating, spine_build
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.uk_atomic_support_fixtures import write_toy_supports
from test_support.microcosm_build.uk_calibration_run import _bound_checkpoint
from test_support.microcosm_build.uk_full_build_cli import patch_support_register
from test_support.microcosm_build.uk_full_population_graph import source_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


@pytest.fixture
def checkpoint_argv(tmp_path, toy_ladder, monkeypatch):
    """A dense request over a real bound checkpoint, as an argument list."""
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
    # The engine stands in here, so its freeze check does too: the preparation
    # runs it before reading any input (microcosm#1095).
    monkeypatch.setattr(
        hmrc_uprating, "assert_installed_engine_is_pinned", lambda: "fixture"
    )
    argv = [
        "--release-role",
        "dense",
        "--input-h5",
        str(path),
        "--input-sidecar",
        str(sidecar_path),
        "--input-spine-gates",
        str(gates_path),
        "--ladder",
        str(ladder_path),
        *support_arguments(tmp_path),
        "--ledger-facts",
        str(ledger),
        "--out",
        str(tmp_path / "out"),
        "--n-clones",
        "1",
        "--epochs",
        "8",
    ]
    return argv, sidecar_path, gates_path


@pytest.fixture
def checkpoint_request(checkpoint_argv):
    argv, sidecar_path, gates_path = checkpoint_argv
    return cli.parse_args(argv), sidecar_path, gates_path


def support_arguments(tmp_path):
    _, paths = write_toy_supports(tmp_path / "supports")
    ew, scotland, ni = (paths[system] for system in sorted(paths, key=_system_order))
    return [
        "--atomic-support-ew",
        str(ew),
        "--atomic-support-scotland",
        str(scotland),
        "--atomic-support-ni",
        str(ni),
    ]


def _system_order(system):
    from microcosm.build.uk_runtime.atomic_area_support import SYSTEMS

    return SYSTEMS.index(system)


def test_prepare_real_checkpoint_preserves_source_year_and_wires_preflight(
    checkpoint_request, tmp_path
):
    args, _, _ = checkpoint_request
    prepared = cli.prepare_full_build(args)
    assert prepared.full.config.source_year == 2023
    assert prepared.full.config.geography_levels is None
    assert prepared.full.config.geography_assignment == "atomic"
    geography = prepared.bindings["geography"]
    assert geography["assignment"] == "atomic"
    assert geography["identity"] == "geography_household_key"
    assert geography["stream"][:2] == ["sha256-u53-v1", "uk-post-clone-atomic-area-v1"]
    assert set(geography["support_pins"]) == {
        "uk_ew_output_area_2021",
        "uk_scotland_output_area_2022",
        "uk_ni_data_zone_2021",
    }
    assert {
        "uk_ew_output_area_2021_support",
        "uk_scotland_output_area_2022_support",
        "uk_ni_data_zone_2021_support",
    } <= set(prepared.sources)
    assert prepared.full.graph.node("uk.full.pool").artifact_inputs[0].producer == (
        "uk.full.geography.gate"
    )
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


def test_prepare_release_candidate_requires_the_registered_supports(
    checkpoint_request, monkeypatch
):
    """Under ``--release-candidate`` the operator's support pins must be the
    sources.yaml rows (microcosm#932 round 1): a pin that proves the bytes are
    the ones requested is not proof they are the ones reviewed."""
    import hashlib
    from pathlib import Path

    args, sidecar_path, gates_path = checkpoint_request

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    argv = [
        "--release-role",
        "dense",
        "--input-h5",
        str(args.input_h5),
        "--input-sha256",
        sha(Path(args.input_h5)),
        "--input-sidecar",
        str(sidecar_path),
        "--input-spine-gates",
        str(gates_path),
        "--ladder",
        str(args.ladder),
        "--ladder-sha256",
        sha(Path(args.ladder)),
        "--ledger-facts",
        str(args.ledger_facts),
        "--out",
        str(args.out),
        "--release-candidate",
    ]
    pins = {}
    for system, label in cli._SUPPORT_ARGUMENTS.items():
        path = Path(getattr(args, f"atomic_support_{label}"))
        pins[system] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
        argv += [
            f"--atomic-support-{label}",
            str(path),
            f"--atomic-support-sha256-{label}",
            pins[system]["sha256"],
        ]
    release = cli.parse_args(argv)
    with pytest.raises(ValueError, match="sources.yaml"):
        cli.prepare_full_build(release)
    patch_support_register(monkeypatch, pins)
    prepared = cli.prepare_full_build(release)
    assert prepared.bindings["geography"]["support_pins"] == pins


# ---------------------------------------------------------------------------
# microcosm#1301: a gate-only sign-off and a later gate clock leave the solve
# ---------------------------------------------------------------------------


@pytest.fixture
def stand_in_dependency_versions(monkeypatch):
    """Kernels hash their declared dependencies' versions; the engine-free
    environment lacks policyengine-uk, so absent ones read as a stand-in."""
    from importlib import metadata

    installed = metadata.version

    def version(distribution):
        try:
            return installed(distribution)
        except metadata.PackageNotFoundError:
            return "0+stand-in"

    monkeypatch.setattr(metadata, "version", version)


def _node_keys(prepared):
    """Every node key of the prepared graph, over fixed source identities."""
    from microcosm.graph.keys import node_key

    compiled = compile_graph(prepared.full.graph)
    sources = {
        source.name: hashlib.sha256(source.name.encode()).hexdigest()
        for source in compiled.graph.sources
    }
    keys: dict[str, str] = {}
    for node_id in compiled.order:
        kernel = prepared.kernels.get(compiled.graph.node(node_id).kernel)
        keys[node_id] = node_key(
            compiled,
            node_id,
            keys,
            kernel.implementation_hash(),
            sources,
            kernel_capabilities=kernel.capabilities,
        )
    return compiled, keys


def _descendants(compiled, node_id):
    found, pending = set(), [node_id]
    while pending:
        current = pending.pop()
        for successor, predecessors in compiled.predecessors.items():
            if current in predecessors and successor not in found:
                found.add(successor)
                pending.append(successor)
    return found


def _perturbed(monkeypatch, *resources):
    """Serve the packaged UK spec with ``resources`` edited (their hashes)."""
    from dataclasses import replace
    from types import MappingProxyType

    from microcosm.build import country_spec

    spec = country_spec.load_country_spec("uk")
    hashes = dict(spec.resource_hashes)
    for resource in resources:
        hashes[resource] = hashlib.sha256(hashes[resource].encode()).hexdigest()
    edited = replace(spec, resource_hashes=MappingProxyType(hashes))
    monkeypatch.setattr(country_spec, "_load_packaged_country_spec", lambda _: edited)


def test_a_gate_only_sign_off_rekeys_the_calibrated_gates_and_nothing_before(
    checkpoint_argv, monkeypatch, stand_in_dependency_versions
):
    """Every node the calibrated gates depend on keeps its key when every
    gate-only register is edited at once; the calibrated gates re-key."""
    from microcosm.build.uk_runtime.solve_scope import UK_GATE_ONLY_RESOURCES

    argv, _, _ = checkpoint_argv
    args = cli.parse_args([*argv, "--dataset-households", "2"])
    prepared = cli.prepare_full_build(args)
    compiled, before = _node_keys(prepared)
    gates = "uk.full.gates.calibrated"
    downstream = {gates, *_descendants(compiled, gates)}
    solve = set(compiled.order) - downstream
    assert {
        "uk.full.target_compilation",
        "uk.full.problem",
        "uk.full.gates.preflight",
        "uk.full.dense",
        "uk.full.size_refit",
        "uk.full.calibrated",
        "uk.full.holdout",
    } <= solve
    with monkeypatch.context() as patch:
        _perturbed(patch, *UK_GATE_ONLY_RESOURCES)
        _, signed = _node_keys(prepared)
    assert {node: signed[node] for node in solve} == {
        node: before[node] for node in solve
    }
    assert signed[gates] != before[gates]
    # Control: a solve-scoped register re-keys the solve from compilation on.
    with monkeypatch.context() as patch:
        _perturbed(patch, "calibration_measure_exclusions.json")
        _, edited = _node_keys(prepared)
    for node in ("uk.full.target_compilation", "uk.full.dense", "uk.full.holdout"):
        assert edited[node] != before[node], node


def test_a_later_gate_clock_rekeys_only_the_calibrated_gates_and_the_export(
    checkpoint_argv, stand_in_dependency_versions
):
    from datetime import date, timedelta

    argv, _, _ = checkpoint_argv
    solve = (date.today() - timedelta(days=2)).isoformat()
    gate = date.today().isoformat()
    first = cli.prepare_full_build(cli.parse_args([*argv, "--review-date", solve]))
    second = cli.prepare_full_build(
        cli.parse_args([*argv, "--review-date", solve, "--gate-review-date", gate])
    )
    assert (first.bindings["review_date"], first.bindings["gate_review_date"]) == (
        solve,
        solve,
    )
    assert (second.bindings["review_date"], second.bindings["gate_review_date"]) == (
        solve,
        gate,
    )
    _, before = _node_keys(first)
    _, after = _node_keys(second)
    assert before.keys() == after.keys()
    assert {node for node in before if before[node] != after[node]} == {
        "uk.full.gates.calibrated",
        "uk.full.export.prepare",
    }
    calibrated = second.full.graph.node("uk.full.gates.calibrated")
    assert calibrated.params["review_date"] == gate
    for node in ("uk.full.gates.preflight", "uk.full.target_compilation"):
        assert second.full.graph.node(node).params["review_date"] == solve
