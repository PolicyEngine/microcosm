"""Invented common-frame fiscal solve through existing grouped Adam and graph."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_fiscal_measurement import InventedSource, context, declaration, frame

from microcosm.build.us_runtime import graph_fiscal_dense_calibration as stage
from microcosm.build.us_runtime import graph_fiscal_measurement as measurement
from microcosm.calibrate import TargetRegistry, group_bounds
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejectedError,
    Numeric,
    NumericScope,
    Owned,
    Slice,
    SourceRef,
    StoreMissError,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)
from microcosm.graph.keys import opaque_artifact_key


def setup(*, incoming=None, row_upper=None, ids=None):
    value = frame()
    if ids is not None:
        old = value.table("household").household_id.to_numpy().copy()
        mapping = dict(zip(old.tolist(), ids.tolist(), strict=True))
        value.table("household")["household_id"] = ids
        value.table("person")["person_household_id"] = np.array(
            [mapping[v] for v in value.person.person_household_id], dtype=ids.dtype
        )
    incoming = np.array([1.0, 2.0, 3.0] if incoming is None else incoming)
    value = Frame(
        {e: value.table(e) for e in value.entities},
        value.schema,
        {"household": Weights(incoming, WeightKind.IMPORTANCE)},
    )
    args = declaration()
    # Exact common-frame targets correspond to desired weights [2, 2, 4].
    args["registry"] = TargetRegistry(
        [
            replace(s, value=v)
            for s, v in zip(args["registry"], [10, 6, 4, 4, 2, 4], strict=True)
        ],
        country="us",
    )
    if incoming[1] == 0:
        args["registry"] = TargetRegistry(
            [
                replace(s, value=v)
                for s, v in zip(args["registry"], [8, 4, 4, 4, 0, 4], strict=True)
            ],
            country="us",
        )
    mc = context(args, value)
    measured = measurement.FiscalMeasurementKernel().run(mc).artifacts["measurement"]
    hh_ids = value.table("household").household_id.to_numpy()
    groups = group_bounds.GroupedUpperBounds(hh_ids, np.array([0, 0, 1]), [6.0, 5.0])
    bounds = stage.encode_fiscal_calibration_bounds(
        population=mc.node.population,
        household_ids=hh_ids,
        original_design_weights=Weights(np.array([8.0, 9.0, 10.0]), WeightKind.DESIGN),
        incoming_weights=value.weights_for("household"),
        grouped_upper_bounds=groups,
        row_upper=np.array([6.0, 6.0, 5.0] if row_upper is None else row_upper),
        budget_sha256=hashlib.sha256(b"invented budget; not an issuer").hexdigest(),
    )
    node = stage.fiscal_dense_calibration_node(
        measurement_node=mc.node,
        bounds_node="invented.bounds",
        epochs=120,
        learning_rate=0.03,
    )
    artifacts = {}
    for name, payload, type_, output in (
        ("measurement", measured, measurement.MEASUREMENT_TYPE, "measurement"),
        ("bounds", bounds, stage.BOUNDS_TYPE, "numeric_bounds"),
    ):
        key = hashlib.sha256(name.encode()).hexdigest()
        artifacts[name] = ArtifactValue(
            payload,
            type_,
            opaque_artifact_key(key, output),
            key,
            NumericScope(Numeric.BITWISE),
        )
    return value, mc, replace(mc, node=node, params=node.params, artifacts=artifacts)


def test_real_grouped_solver_rebuilds_exact_matrix_and_residuals(monkeypatch):
    value, mc, c = setup()
    captured = []
    original_calibrate = stage.calibrate

    def record(*args, **kwargs):
        result = original_calibrate(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(stage, "calibrate", record)
    result = stage.FiscalDenseCalibrationKernel().run(c)
    solved = captured[0]
    expected = measurement.decode_fiscal_measurement(c.artifacts["measurement"].payload)
    np.testing.assert_array_equal(
        solved.problem.matrix.toarray(), expected.matrix.toarray()
    )
    assert solved.problem.names == tuple(
        s.to_target().row_name for s in expected.registry
    )
    assert solved.problem.target_vector.tobytes() == expected.target_values.tobytes()
    assert solved.final_loss < solved.initial_loss
    assert result.weights.kind is WeightKind.CALIBRATED
    assert result.frame is None and not result.columns
    assert result.receipt["release_eligible"] is False
    assert result.receipt["source_admission"] == "required_from_complete_parent_owner"
    diagnostic = json.loads(result.artifacts["diagnostics"])
    assert diagnostic["schema_version"] == 8
    origin = json.loads(result.artifacts["origin_diagnostics"])
    achieved = expected.matrix @ result.weights.values
    for row, actual in zip(diagnostic["targets"], achieved, strict=True):
        assert row["final_estimate"] == actual
    assert origin["accepted_weights"] == measurement._array(result.weights.values)
    assert origin["group_totals"] == measurement._array(
        np.array([result.weights.values[:2].sum(), result.weights.values[2]])
    )
    assert origin["fixed_zero_rows"] == 0
    assert diagnostic["build"]["matrix_matches_measurement"] is True
    # The direct kernel receives observations only, never a writable parent.
    for entity in value.entities:
        if entity in mc.tables:
            pd.testing.assert_frame_equal(c.tables[entity], mc.tables[entity])
    assert (
        value.weights_for("household").values.tobytes()
        == np.array([1.0, 2.0, 3.0]).tobytes()
    )


def test_uint64_order_original_design_and_strict_zero_support():
    ids = np.array([2**63, 2**63 + 1, 2**64 - 1], dtype=np.uint64)
    _, _, c = setup(incoming=[1.0, -0.0, 3.0], ids=ids)
    bounds = stage.verify_fiscal_calibration_bounds(
        c.artifacts["bounds"].payload,
        household_ids=ids,
        incoming_weights=c.weights["household"],
        original_design_weights=Weights(np.array([8.0, 9.0, 10.0]), WeightKind.DESIGN),
    )
    assert bounds.household_ids.dtype == ids.dtype
    assert (
        bounds.original_design.values.tobytes() == np.array([8.0, 9.0, 10.0]).tobytes()
    )
    output = stage.FiscalDenseCalibrationKernel().run(c)
    assert output.weights.values[1:2].tobytes() == np.array([-0.0]).tobytes()
    assert np.all(output.weights.values[[0, 2]] > 0)
    stage.check_fiscal_calibration_weights(bounds, output.weights.values)
    assert json.loads(output.artifacts["origin_diagnostics"])["fixed_zero_rows"] == 1
    with pytest.raises(ValueError, match="DESIGN_ANCHOR"):
        stage.verify_fiscal_calibration_bounds(
            c.artifacts["bounds"].payload,
            household_ids=ids,
            incoming_weights=c.weights["household"],
            original_design_weights=Weights(
                np.array([8.0, 8.0, 10.0]), WeightKind.DESIGN
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("epochs", True),
        ("epochs", 1.5),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", 0.0),
        ("learning_rate", True),
    ],
)
def test_solver_settings_refuse_nonfinite_or_undeclared_modes(field, value):
    _, mc, _ = setup()
    with pytest.raises(ValueError):
        stage.fiscal_dense_calibration_node(
            measurement_node=mc.node, bounds_node="bounds", **{field: value}
        )


def test_misaligned_measurement_bounds_and_calibrated_parent_refuse_before_solve(
    monkeypatch,
):
    _, _, c = setup()

    def forbidden(*a, **kw):
        pytest.fail("solver ran before admission")

    monkeypatch.setattr(stage, "calibrate", forbidden)
    with pytest.raises(ValueError, match="IMPORTANCE"):
        stage.FiscalDenseCalibrationKernel().run(
            replace(
                c,
                weights={
                    "household": Weights(
                        np.array([1.0, 2.0, 3.0]), WeightKind.CALIBRATED
                    )
                },
            )
        )
    changed = dict(c.tables)
    changed["person"] = c.tables["person"].copy()
    changed["person"]["amount"] += 1
    with pytest.raises(ValueError, match="PROJECTION_BINDING"):
        stage.FiscalDenseCalibrationKernel().run(replace(c, tables=changed))
    raw = json.loads(c.artifacts["bounds"].payload)
    raw["household_ids"] = measurement._array(np.array([30, 20, 10], dtype=np.int64))
    artifacts = dict(c.artifacts)
    artifacts["bounds"] = replace(
        artifacts["bounds"], payload=measurement.canonical_json(raw)
    )
    with pytest.raises(ValueError, match="HOUSEHOLD_ALIGNMENT"):
        stage.FiscalDenseCalibrationKernel().run(replace(c, artifacts=artifacts))


def test_group_and_row_caps_refuse_without_clipping():
    _, _, c = setup(row_upper=[1.0, 6.0, 5.0])
    with pytest.raises(ValueError, match="ROW_REFERENCE_CAP"):
        stage.FiscalDenseCalibrationKernel().run(c)
    bounds = stage.decode_fiscal_calibration_bounds(c.artifacts["bounds"].payload)
    with pytest.raises(ValueError, match="frozen absolute bounds"):
        stage.check_fiscal_calibration_weights(bounds, np.array([5.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "change,reason",
    [
        ("names", "TARGET_ALIGNMENT"),
        ("matrix", "MATRIX_IDENTITY"),
        ("diagnostic", "DIAGNOSTIC_ESTIMATES"),
    ],
)
def test_solver_return_with_wrong_matrix_or_target_order_is_refused(
    monkeypatch, change, reason
):
    _, _, c = setup()
    actual = stage.calibrate

    def changed(*a, **kw):
        r = actual(*a, **kw)
        if change == "names":
            return replace(
                r, problem=replace(r.problem, names=tuple(reversed(r.problem.names)))
            )
        if change == "matrix":
            matrix = r.problem.matrix.copy()
            matrix.data[0] += 1
            return replace(r, problem=replace(r.problem, matrix=matrix))
        return replace(
            r,
            diagnostics=(
                replace(r.diagnostics[0], final_estimate=999.0),
                *r.diagnostics[1:],
            ),
        )

    monkeypatch.setattr(stage, "calibrate", changed)
    with pytest.raises(ValueError, match=reason):
        stage.FiscalDenseCalibrationKernel().run(c)


class InventedImportance(KernelBase):
    ref = "test.fiscal_dense_importance@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def run(self, context):
        return KernelResult(
            weights=Weights(np.array(context.params["incoming"]), WeightKind.IMPORTANCE)
        )


class InventedBounds(KernelBase):
    ref = "test.fiscal_dense_bounds@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        payload = context.sources["bounds"].read_bytes()
        stage.decode_fiscal_calibration_bounds(payload)
        return KernelResult(artifacts={"numeric_bounds": payload})


def graph_fixture(tmp_path):
    value, mc, c = setup(incoming=[1.0, 0.0, 3.0])
    value = Frame(
        {e: value.table(e) for e in value.entities},
        value.schema,
        {"household": Weights(np.array([8.0, 9.0, 10.0]), WeightKind.DESIGN)},
    )
    create = Node(
        "invented.design",
        InventedSource.ref,
        sources=("fixture",),
        structural=StructuralDelta.CREATE,
        outputs=(
            Owned("person", "amount", "float64"),
            Owned("person", "keep", "bool"),
            Owned("tax_unit", "tax_unit_amount", "float64"),
            Owned("household", "state_fips", "int64"),
            Owned("household", "congressional_district_geoid", "int64"),
        ),
    )
    importance = Node(
        "invented",
        InventedImportance.ref,
        base=create.id,
        structural=StructuralDelta.REWEIGHT,
        inputs=(Slice("household", ("state_fips",)),),
        weights=WeightTransition("household", "importance", mass="free"),
        mass="free",
        params={"incoming": (1.0, 0.0, 3.0)},
    )
    bounds = Node(
        "invented.bounds",
        InventedBounds.ref,
        population=importance.id,
        sources=("bounds",),
        artifact_outputs=(ArtifactOutput("numeric_bounds", stage.BOUNDS_TYPE),),
    )
    nodes = (create, importance, bounds, mc.node, c.node)
    sources = (SourceRef("fixture", "frame-store"), SourceRef("bounds", "raw-bytes-v1"))
    kernels = KernelRegistry()
    for kernel in (
        InventedSource(),
        InventedImportance(),
        InventedBounds(),
        measurement.FiscalMeasurementKernel(),
        stage.FiscalDenseCalibrationKernel(),
    ):
        kernels.register(kernel)
    store = ContentStore(tmp_path / "store")
    path = store.put_frame(
        hashlib.sha256(b"invented fiscal dense fixture").hexdigest(), value
    )
    bound_path = tmp_path / "bounds.json"
    bound_path.write_bytes(c.artifacts["bounds"].payload)
    arguments = {
        "store": store,
        "kernels": kernels,
        "sources": {"fixture": path, "bounds": bound_path},
    }
    return value, sources, nodes, arguments


def test_real_graph_cold_required_replay_preserves_parent_and_original_anchors(
    tmp_path,
):
    original, sources, nodes, arguments = graph_fixture(tmp_path)
    compiled = compile_graph(Graph("us", sources, nodes))
    anchors = {}

    def observe(node_id, population):
        anchors[node_id] = population.design_weights["household"].tobytes()

    cold = run_graph(compiled, **arguments, _population_observer=observe)
    cold_anchors = dict(anchors)
    warm = run_graph(
        compiled, **arguments, resume="require", _population_observer=observe
    )
    assert cold.key == warm.key and all(n.store_hit for n in warm.nodes.values())
    before = warm.population("invented")
    after = warm.population(nodes[-1].id)
    # The graph performs precisely DESIGN -> IMPORTANCE -> CALIBRATED.
    assert after.weights_for("household").kind is WeightKind.CALIBRATED
    assert before.weights_for("household").kind is WeightKind.IMPORTANCE
    for entity in original.entities:
        pd.testing.assert_frame_equal(before.table(entity), original.table(entity))
        pd.testing.assert_frame_equal(after.table(entity), original.table(entity))
    assert after.weights_for("household").values[1] == 0
    assert (
        cold.population(nodes[-1].id).weights_for("household").values.tobytes()
        == after.weights_for("household").values.tobytes()
    )
    # Detached observer snapshots expose original anchors without owner authority.
    assert anchors == cold_anchors
    assert anchors[nodes[-1].id] == np.array([8.0, 9.0, 10.0]).tobytes()
    payload = arguments["store"].load_bytes(
        warm.node(nodes[-1].id).opaque_artifacts["origin_diagnostics"]
    )
    assert json.loads(payload)["fixed_zero_rows"] == 1
    # A repeated CALIBRATED -> CALIBRATED stage is refused before another solve.
    new_measure = replace(nodes[-2], id="repeat.measurement", population=nodes[-1].id)
    repeated = stage.fiscal_dense_calibration_node(
        measurement_node=new_measure,
        bounds_node=nodes[2].id,
        node_id="repeat.calibration",
    )
    compiled_repeat = compile_graph(
        Graph("us", sources, (*nodes, new_measure, repeated))
    )
    with pytest.raises(
        NodeRejectedError, match="(?i)(importance|calibrated|transition|forward)"
    ):
        run_graph(compiled_repeat, **arguments)


def test_real_graph_replay_refuses_changed_bounds_and_target_declaration(tmp_path):
    _, sources, nodes, arguments = graph_fixture(tmp_path)
    compiled = compile_graph(Graph("us", sources, nodes))
    run_graph(compiled, **arguments)
    changed_measure = measurement.fiscal_measurement_node(**declaration())
    changed_solve = stage.fiscal_dense_calibration_node(
        measurement_node=changed_measure, bounds_node=nodes[2].id
    )
    changed = compile_graph(
        Graph("us", sources, (*nodes[:3], changed_measure, changed_solve))
    )
    with pytest.raises(
        (NodeRejectedError, StoreMissError), match="(?i)(require|cache|missing)"
    ):
        run_graph(changed, **arguments, resume="require")
    path = arguments["sources"]["bounds"]
    raw = json.loads(path.read_bytes())
    raw["group_upper"] = measurement._array(np.array([5.5, 5.0]))
    path.write_bytes(measurement.canonical_json(raw))
    with pytest.raises(
        (NodeRejectedError, StoreMissError), match="(?i)(require|cache|missing)"
    ):
        run_graph(compiled, **arguments, resume="require")


def test_bound_codec_refuses_noncanonical_values_and_wrong_types():
    _, _, c = setup()
    payload = c.artifacts["bounds"].payload
    for field, value in (
        ("incoming", np.array([1.0, float("nan"), 3.0])),
        ("group_indices", np.array([0, 0, 9], dtype=np.int64)),
        ("original_design", np.array([8.0, 9.0])),
        ("household_ids", np.array([10, 10, 30], dtype=np.int64)),
    ):
        raw = json.loads(payload)
        raw[field] = measurement._array(value)
        with pytest.raises(ValueError):
            stage.decode_fiscal_calibration_bounds(measurement.canonical_json(raw))
    with pytest.raises(ValueError, match="CANONICAL"):
        stage.decode_fiscal_calibration_bounds(payload + b"\n")


def test_implementation_identity_binds_existing_solver_and_target_math(
    monkeypatch, tmp_path
):
    # Point source identity at changed synthetic bytes without editing package source.
    baseline = stage.FiscalDenseCalibrationKernel().implementation_hash()
    changed = tmp_path / "changed_target.py"
    changed.write_text("# invented changed target arithmetic\n")
    monkeypatch.setattr(stage.target_module, "__file__", str(changed))
    assert stage.FiscalDenseCalibrationKernel().implementation_hash() != baseline
