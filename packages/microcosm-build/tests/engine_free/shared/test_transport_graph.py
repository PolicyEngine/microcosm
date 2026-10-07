"""Charter properties of the shared calibration/terminal kernels on a toy graph.

The toy graph (``test_support/microcosm_build/transport_graph.py``) is
synthetic: six made-up households and invented facts. It runs every G5b
kernel inside ``run_graph`` on a content store.

Properties:

- A1 determinism: two fresh stores give identical node keys and identical
  artifact bytes. The export H5 is an outer input: two writes of one
  descriptor can differ in bytes (the HDF5 writer records object times at
  one-second resolution), so the second build reads a byte copy of the first
  build's file.
- A2 memoization: a second run on the same store executes zero kernels.
- A4 inert fields: editing ``description`` or ``citation`` moves no key.
- A6 input identity: renaming source paths changes nothing; changing one
  source's bytes re-keys exactly that source's consumers and their
  descendants.
- Differential: ``calibrate.ordered_adam@1`` over the compiled problem and
  ``calibrate.adam@1`` over the same targets as params install
  byte-identical weights. This holds because both paths build their
  constraint matrix from equal-valued target sets through
  :mod:`microcosm.calibrate.matrix` and solve with the same
  :func:`microcosm.calibrate.calibrate` call.
- D3 (Hypothesis property): the executor's ``realized_max_weight_ratio`` is
  in the calibrate receipt and equals the largest installed/design ratio
  recomputed from the stored weights, and the node is accepted exactly when
  that ratio is within the cap, even where the solver's own cap, measured
  against its starting weights, is satisfied.
- Source rules: the transport modules import no country runtime, name no
  country, and carry no float literal (no target value, threshold or rate).
"""

from __future__ import annotations

import ast
import functools
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.target_kernels import decode_target_surface
from microcosm.calibrate.kernels import CALIBRATE_ADAM
from microcosm.frame import WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    NodeRejectedError,
    Slice,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)
from test_support.microcosm_build.transport_graph import (
    CALIBRATION_TARGETS,
    DEFAULT_CONFIG,
    TOY_BINDINGS,
    ToySources,
    descendants,
    run_through,
    run_toy,
    toy_facts,
    toy_frame,
    toy_graph,
    toy_registry,
    with_config,
    write_facts,
    write_toy_sources,
)
from test_support.paths import paths_for

TRANSPORT = (
    paths_for("microcosm-build").package / "src" / "microcosm" / "build" / "transport"
)
CALIBRATE = "toy.calibrate"


def _bytes_of(run) -> dict[str, dict[str, bytes]]:
    return {
        node_id: {
            name: run.store.load_bytes(key)
            for name, key in receipt.opaque_artifacts.items()
        }
        for node_id, receipt in run.manifest.nodes.items()
    }


def _keys(manifest) -> dict[str, str]:
    return {node_id: receipt.key for node_id, receipt in manifest.nodes.items()}


def test_a1_two_fresh_stores_give_identical_keys_and_bytes(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    first = run_toy(tmp_path / "one", sources)
    copied = tmp_path / "two" / "copied_export.h5"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(first.exported, copied)
    second = run_toy(tmp_path / "two", sources, exported=copied)

    assert _keys(first.manifest) == _keys(second.manifest)
    assert _bytes_of(first) == _bytes_of(second)
    assert first.manifest.key == second.manifest.key


def test_a2_a_second_run_on_the_same_store_executes_no_kernel(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    first = run_toy(tmp_path, sources)
    again = run_toy(tmp_path, sources, store=first.store, exported=first.exported)

    assert all(receipt.hit for receipt in again.prepared.nodes.values())
    assert all(receipt.hit for receipt in again.manifest.nodes.values())
    assert _keys(again.manifest) == _keys(first.manifest)


def _editable_gate_bindings():
    binding = replace(
        TOY_BINDINGS["weight_ratio"],
        artifact_arguments=dict(TOY_BINDINGS["weight_ratio"].artifact_arguments),
    )
    return {**TOY_BINDINGS, "weight_ratio": binding}


def _store_files(store):
    return {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }


def test_binding_edit_after_first_run_raises_without_changing_store(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    bindings = _editable_gate_bindings()
    registry = toy_registry(bindings)
    first, store = run_through(
        tmp_path, sources, DEFAULT_CONFIG, "toy.gates.terminal", registry=registry
    )
    assert first.nodes["toy.gates.terminal"].receipt["outcome"] == "pass"
    before = _store_files(store)
    bindings["weight_ratio"].artifact_arguments["solution"] = "missing"

    with pytest.raises(ValueError, match="binding registry changed"):
        run_through(
            tmp_path,
            sources,
            DEFAULT_CONFIG,
            "toy.gates.terminal",
            store=store,
            registry=registry,
        )

    assert _store_files(store) == before


def test_binding_edit_before_first_run_raises_then_correct_registry_runs_fresh(
    tmp_path,
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    bindings = _editable_gate_bindings()
    registry = toy_registry(bindings)
    store = ContentStore(tmp_path / "store")
    before = _store_files(store)
    bindings["weight_ratio"].artifact_arguments["solution"] = "missing"

    with pytest.raises(ValueError, match="binding registry changed"):
        run_through(
            tmp_path,
            sources,
            DEFAULT_CONFIG,
            "toy.gates.terminal",
            store=store,
            registry=registry,
        )

    assert _store_files(store) == before
    correct, _ = run_through(
        tmp_path,
        sources,
        DEFAULT_CONFIG,
        "toy.gates.terminal",
        store=store,
        registry=toy_registry(_editable_gate_bindings()),
    )
    assert correct.nodes["toy.gates.terminal"].receipt["outcome"] == "pass"
    assert all(not receipt.hit for receipt in correct.nodes.values())


def test_a4_description_and_citation_move_no_key(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    plain, _ = run_through(
        tmp_path / "a", sources, DEFAULT_CONFIG, "toy.export.prepare"
    )
    described = with_config(DEFAULT_CONFIG, description="Edited prose, never hashed.")
    graph = toy_graph(described, through_prepare=True)
    assert graph.node("toy.create").description
    assert graph.node("toy.targets.compile").citation
    edited, _ = run_through(tmp_path / "b", sources, described, "toy.export.prepare")

    assert _keys(plain) == _keys(edited)


def _renamed(sources: ToySources, root: Path) -> ToySources:
    root.mkdir(parents=True)
    moved = {}
    for name in ("population", "calibration_facts", "holdout_facts"):
        target = root / f"renamed_{name}.bin"
        shutil.copyfile(getattr(sources, name), target)
        moved[name] = target
    return ToySources(**moved)


def test_a6_renaming_sources_changes_nothing(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    first, store = run_through(tmp_path, sources, DEFAULT_CONFIG, "toy.export.prepare")
    moved, _ = run_through(
        tmp_path,
        _renamed(sources, tmp_path / "moved"),
        DEFAULT_CONFIG,
        "toy.export.prepare",
        store=store,
    )

    assert _keys(moved) == _keys(first)
    assert all(receipt.hit for receipt in moved.nodes.values())


@pytest.mark.parametrize(
    ("source", "consumer"),
    [
        ("holdout_facts", "toy.validate.compare"),
        ("calibration_facts", "toy.targets.compile"),
        ("population", "toy.create"),
    ],
)
def test_a6_one_changed_byte_rekeys_exactly_its_consumers(
    tmp_path, source, consumer
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    before, store = run_through(tmp_path, sources, DEFAULT_CONFIG, "toy.export.prepare")
    changed_root = tmp_path / "changed"
    changed = _renamed(sources, changed_root)
    path = getattr(changed, source)
    # One trailing byte: the loaders ignore it, so only identity changes.
    path.write_bytes(path.read_bytes() + b" ")
    after, _ = run_through(
        tmp_path, changed, DEFAULT_CONFIG, "toy.export.prepare", store=store
    )
    graph = toy_graph(DEFAULT_CONFIG, through_prepare=True)
    expected = descendants(graph, {consumer})
    moved = {
        node for node in before.nodes if before.nodes[node].key != after.nodes[node].key
    }

    assert moved == expected
    assert {
        node for node, receipt in after.nodes.items() if not receipt.hit
    } == expected


# ---------------------------------------------------------------------------
# The ordered solve against calibrate.adam@1, inside run_graph
# ---------------------------------------------------------------------------

#: Household-grain targets, which ``calibrate.adam@1`` can take as params.
HOUSEHOLD_TARGETS = (
    ("toy_renters", "household", "is_renter", None, 1.05),
    ("toy_rent", "household", "rent", None, 0.9),
    ("toy_renter_rent", "household", "rent", "is_renter", 1.2),
)


def test_differential_ordered_and_adam_install_byte_identical_weights(
    tmp_path,
) -> None:
    config = with_config(DEFAULT_CONFIG, calibration_targets=HOUSEHOLD_TARGETS)
    sources = write_toy_sources(tmp_path / "src", config)
    ordered, store = run_through(tmp_path / "ordered", sources, config, CALIBRATE)
    surface = decode_target_surface(
        store.load_bytes(
            ordered.nodes["toy.targets.compile"].opaque_artifacts["surface"]
        )
    )
    targets = tuple(
        (spec.name, spec.measure, spec.filter, spec.value, None)
        for spec in surface.registry.specs
    )
    graph = toy_graph(config, through_prepare=True)
    solve = graph.node(CALIBRATE)
    adam = replace(
        solve,
        kernel=CALIBRATE_ADAM.ref,
        inputs=(Slice("household", ("rent", "is_renter")),),
        params={**solve.params, "targets": targets},
        artifact_inputs=(),
        artifact_outputs=(),
    )
    adam_graph = replace(
        graph,
        nodes=(graph.node("toy.create"), graph.node("toy.open"), adam),
    )
    registry = toy_registry()
    registry.register(CALIBRATE_ADAM)
    adam_store = ContentStore(tmp_path / "adam" / "store")
    adam_manifest = run_graph(
        compile_graph(adam_graph),
        sources={"population": sources.population},
        store=adam_store,
        kernels=registry,
    )

    ordered_weights = store.load_column(ordered.nodes[CALIBRATE].weight_key)
    adam_weights = adam_store.load_column(adam_manifest.nodes[CALIBRATE].weight_key)
    assert np.asarray(ordered_weights).tobytes() == np.asarray(adam_weights).tobytes()
    assert (
        ordered.nodes[CALIBRATE].receipt["realized_max_weight_ratio"]
        == adam_manifest.nodes[CALIBRATE].receipt["realized_max_weight_ratio"]
    )


# ---------------------------------------------------------------------------
# D3: the cap is measured against the CREATE design anchor
# ---------------------------------------------------------------------------


def test_d3_realized_ratio_is_recorded_and_bounded(tmp_path) -> None:
    sources = write_toy_sources(tmp_path / "src")
    manifest, store = run_through(tmp_path, sources, DEFAULT_CONFIG, CALIBRATE)
    receipt = manifest.nodes[CALIBRATE].receipt
    weights = np.asarray(store.load_column(manifest.nodes[CALIBRATE].weight_key))

    assert receipt["weight_anchor"] == "design"
    assert receipt["max_weight_ratio"] == DEFAULT_CONFIG.max_weight_ratio
    assert 0 < receipt["realized_max_weight_ratio"] <= DEFAULT_CONFIG.max_weight_ratio
    assert (weights >= 0).all()


class _Importance(KernelBase):
    """Design -> importance: scale every household weight by ``factor``."""

    ref = "test.importance@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def run(self, context: KernelContext) -> KernelResult:
        values = context.weights["household"].values * float(context.params["factor"])
        return KernelResult(weights=Weights(values, WeightKind.IMPORTANCE))


def _capped_graph(cap: float, factor: float = 2, **solve) -> Graph:
    graph = toy_graph(
        with_config(DEFAULT_CONFIG, max_weight_ratio=cap, **solve),
        through_prepare=True,
    )
    importance = Node(
        "toy.importance",
        _Importance.ref,
        structural=StructuralDelta.REWEIGHT,
        base="toy.open",
        inputs=(Slice("household", ("rent",)),),
        params={"factor": factor},
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
    )
    problem = replace(graph.node("toy.targets.problem"), population=importance.id)
    compile_ = replace(graph.node("toy.targets.compile"), population=importance.id)
    solve = replace(graph.node(CALIBRATE), base=importance.id)
    return replace(
        graph,
        nodes=(
            graph.node("toy.create"),
            graph.node("toy.open"),
            importance,
            compile_,
            problem,
            solve,
        ),
    )


def test_d3_a_cap_violation_against_the_design_anchor_rejects_the_node(
    tmp_path,
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    # Facts valued at twice the design weights, which is where the importance
    # step starts the solve: calibrated weights stay near 2x their design
    # anchor, within the solver's own 1.5x cap on its starting weights but
    # outside the executor's 1.5x cap on the CREATE design weights.
    doubled = toy_frame(
        weights=[2 * w for w in toy_frame().weights_for("household").values]
    )
    write_facts(sources.calibration_facts, toy_facts(CALIBRATION_TARGETS, doubled))
    registry = toy_registry()
    registry.register(_Importance())
    with pytest.raises(NodeRejectedError, match="original design weight"):
        run_graph(
            compile_graph(_capped_graph(1.5)),
            sources={
                "population": sources.population,
                "calibration_facts": sources.calibration_facts,
            },
            store=ContentStore(tmp_path / "store"),
            kernels=registry,
        )
    loose = run_graph(
        compile_graph(_capped_graph(4.0)),
        sources={
            "population": sources.population,
            "calibration_facts": sources.calibration_facts,
        },
        store=ContentStore(tmp_path / "loose"),
        kernels=registry,
    )
    receipt = loose.nodes[CALIBRATE].receipt
    assert 1.5 < receipt["realized_max_weight_ratio"] <= 4.0


# ---------------------------------------------------------------------------
# Source rules
# ---------------------------------------------------------------------------


def _modules() -> list[Path]:
    modules = sorted(TRANSPORT.glob("*.py"))
    assert {path.name for path in modules} >= {
        "__init__.py",
        "artifact_types.py",
        "gate_kernels.py",
        "graph_inputs.py",
        "target_kernels.py",
        "terminal_kernels.py",
    }
    return modules


def test_transport_modules_import_no_country_runtime_and_name_no_country() -> None:
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "uk_runtime" not in name and "us_runtime" not in name, path
        words = {word.lower() for word in re.findall(r"[A-Za-z]+", path.read_text())}
        assert "nz" not in words, path


def test_transport_modules_carry_no_float_literal() -> None:
    """No target value, threshold or rate lives in the kernels' Python."""

    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        floats = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        assert floats == [], (path, floats)


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
@given(
    factor=st.sampled_from((0.5, 0.8, 1.25, 2.0, 3.0)),
    cap=st.sampled_from((1.1, 1.5, 2.5, 4.0)),
)
def test_property_d3_the_cap_is_measured_against_the_design_anchor(
    tmp_path, factor, cap
) -> None:
    """Accepted exactly when max(installed / CREATE design) <= cap.

    The importance step scales every weight by ``factor``, the facts are the
    totals at those starting weights, and the solve barely moves (one epoch at
    a small rate), so every installed weight sits at about ``factor`` times its
    design anchor while the solver's own cap is measured against the starting
    weights. The grid keeps ``factor`` at least 10% away from ``cap``.
    """

    root = tmp_path / f"f{factor}_c{cap}"
    shutil.rmtree(root, ignore_errors=True)
    sources = write_toy_sources(root / "src")
    design = toy_frame().weights_for("household").values
    exact = tuple((*target[:4], 1.0) for target in CALIBRATION_TARGETS)
    start = toy_frame(weights=[factor * w for w in design])
    write_facts(sources.calibration_facts, toy_facts(exact, start))
    registry = toy_registry()
    registry.register(_Importance())
    graph = _capped_graph(cap, factor, epochs=1, learning_rate=0.001)
    run = functools.partial(
        run_graph,
        compile_graph(graph),
        sources={
            "population": sources.population,
            "calibration_facts": sources.calibration_facts,
        },
        store=ContentStore(root / "store"),
        kernels=registry,
    )
    if factor > cap:
        with pytest.raises(NodeRejectedError, match="original design weight"):
            run()
        return
    manifest = run()
    store = ContentStore(root / "store")
    installed = np.asarray(store.load_column(manifest.nodes[CALIBRATE].weight_key))
    ratio = float(np.max(installed / design))
    receipt = manifest.nodes[CALIBRATE].receipt
    assert receipt["realized_max_weight_ratio"] == ratio
    assert ratio <= cap
    assert ratio == pytest.approx(factor, rel=0.01)
