"""Invented demographic margins through the real calibration graph seam."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.demographic_calibration_graph import (
    DemographicCalibrationKernel,
    demographic_calibration_node,
)
from microcosm.calibrate import calibrate, diagnostics_payload
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    load_source,
    run_graph,
    source_hash,
)


def fixture_registry():
    """Fictional values and identities; these are not Census observations."""
    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="household",
                measure=measure,
                value=value,
                period="2024",
                se=se,
                source="Invented fixture; no published Census value",
                family="acs.S0101",
                metadata={
                    "table": "S0101",
                    "reference_sha256": "a" * 64,
                    "geography": "0100000US",
                    "universe": "population",
                    "role": "calibration",
                    "evidence_scope": "invented",
                },
            )
            for name, measure, value, se in (
                ("S0101_C01_001", "people", 16.0, None),
                ("S0101_C01_002", "children", 5.0, 0.7),
            )
        ],
        country="us",
    )


def fixture_frame():
    ids = np.array([2, 6, 9, 15], dtype=np.int64)
    return Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [11, 12, 3, 20, 8, 9],
                    "person_household_id": [9, 9, 2, 15, 6, 6],
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": ids,
                    "people": [1.0, 2.0, 2.0, 1.0],
                    "children": [0.0, 1.0, 1.0, 0.0],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([2.0, 1.0, 1.0, 3.0]), WeightKind.IMPORTANCE)},
    )


def fixture_node(registry=None):
    return demographic_calibration_node(
        fixture_registry() if registry is None else registry,
        base="assembled",
        epochs=48,
        learning_rate=0.03,
        max_weight_ratio=4.0,
        max_initial_weight_ratio=4.0,
    )


def context(frame, node):
    return KernelContext(
        node=node,
        tables={"household": frame.table("household")},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
    )


def test_registry_metadata_and_actual_solver_diagnostics_survive():
    registry, frame = fixture_registry(), fixture_frame()
    direct = calibrate(
        frame,
        registry.to_target_set(),
        weight_entity="household",
        method="adam",
        epochs=48,
        learning_rate=0.03,
        max_weight_ratio=4.0,
        mass="free",
        seed=0,
    )
    result = DemographicCalibrationKernel().run(context(frame, fixture_node(registry)))
    np.testing.assert_array_equal(result.weights.values, direct.weights)
    payload = json.loads(result.artifacts["diagnostics"])
    assert payload == diagnostics_payload(direct, target_registry=registry)
    assert payload["schema_version"] == 6
    assert payload["final_loss"] < payload["initial_loss"]
    assert payload["targets"][0]["period"] == "2024"
    assert result.receipt["consumes_standard_errors"] is False
    assert result.receipt["solver_weight_anchor"] == "incoming_importance"
    assert result.receipt["executor_weight_anchor"] == "original_design"
    assert result.receipt["release_eligible"] is False
    assert result.weights.kind is WeightKind.CALIBRATED


@pytest.mark.parametrize("table", ["B19001", "B25003"])
def test_reserved_tables_cannot_become_calibration_targets(table):
    registry = fixture_registry()
    spec = registry.specs[0]
    changed = replace(
        spec, family=f"acs.{table}", metadata={**spec.metadata, "table": table}
    )
    with pytest.raises(ValueError, match="reserved"):
        fixture_node(TargetRegistry([changed], country="us"))


@pytest.mark.parametrize(
    "key,value",
    [("role", "holdout"), ("universe", "households"), ("geography", "5001900US0101")],
)
def test_nonmatching_measurement_scope_refused(key, value):
    spec = fixture_registry().specs[0]
    changed = replace(spec, metadata={**spec.metadata, key: value})
    with pytest.raises(ValueError, match="scope"):
        fixture_node(TargetRegistry([changed], country="us"))


def test_registry_mutation_after_declaration_cannot_change_node():
    registry = fixture_registry()
    node = fixture_node(registry)
    registry.specs[0].metadata["reference_sha256"] = "b" * 64
    assert fixture_node(registry).params != node.params
    result = DemographicCalibrationKernel().run(context(fixture_frame(), node))
    assert (
        result.receipt["registry_sha256"]
        != DemographicCalibrationKernel()
        .run(context(fixture_frame(), fixture_node(registry)))
        .receipt["registry_sha256"]
    )


def test_negative_or_missing_demographic_counts_are_not_silently_skipped():
    frame, node = fixture_frame(), fixture_node()
    for value in [-1.0, np.nan, 0.5]:
        table = frame.table("household").copy()
        table.loc[0, "people"] = value
        bad = replace(context(frame, node), tables={"household": table})
        with pytest.raises(ValueError, match="nonnegative finite"):
            DemographicCalibrationKernel().run(bad)


@pytest.mark.parametrize("dtype", [str, complex])
def test_strings_and_complex_values_are_not_coerced_into_counts(dtype):
    frame, node = fixture_frame(), fixture_node()
    table = frame.table("household").copy()
    table["people"] = table["people"].astype(dtype)
    with pytest.raises(ValueError, match="numeric count"):
        DemographicCalibrationKernel().run(
            replace(context(frame, node), tables={"household": table})
        )


def test_altered_input_scope_is_refused_even_with_valid_params():
    frame, node = fixture_frame(), fixture_node()
    node = replace(node, inputs=(Slice("household", ("people",)),))
    with pytest.raises(ValueError, match="complete declaration"):
        DemographicCalibrationKernel().run(context(frame, node))


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", True),
        ("learning_rate", 0),
        ("max_weight_ratio", float("inf")),
        ("max_initial_weight_ratio", 0.5),
    ],
)
def test_invalid_solver_controls_refused_before_execution(key, value):
    options = {
        "epochs": 48,
        "learning_rate": 0.03,
        "max_weight_ratio": 4.0,
        "max_initial_weight_ratio": 4.0,
    }
    with pytest.raises(ValueError):
        demographic_calibration_node(
            fixture_registry(), base="assembled", **{**options, key: value}
        )


class FixtureSource(KernelBase):
    ref = "test.demographic_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["fixture"])
        )

    def implementation_hash(self):
        # Pytest and the fresh child import this fixture under different module
        # names. Bind its same actual file bytes, not that incidental name.
        return hashlib.sha256(
            Path(__file__).read_bytes() + source_hash(load_source).encode()
        ).hexdigest()


class FixtureImportance(FixtureSource):
    ref = "test.demographic_importance@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def run(self, context):
        return KernelResult(
            weights=Weights(
                context.weights["household"].values / 2, WeightKind.IMPORTANCE
            )
        )


def graph_fixture(output, warm=False):
    """Cold and warm use the same on-disk fixture and real production kernel."""
    from microcosm.build.transfer_example import _source_frame_path

    output = Path(output)
    store = ContentStore(output / "store")
    if not warm:
        frame = fixture_frame()
        source = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {
                "household": Weights(
                    frame.weights_for("household").values * 2, WeightKind.DESIGN
                )
            },
            frame.strata,
        )
        path = _source_frame_path(store, source)
        (output / "source-path.txt").write_text(str(path))
    path = Path((output / "source-path.txt").read_text())
    graph = Graph(
        "us",
        (SourceRef("fixture", "frame-store"),),
        (
            Node(
                "source",
                FixtureSource.ref,
                sources=("fixture",),
                structural=StructuralDelta.CREATE,
                outputs=(
                    Owned("household", "people", "float64"),
                    Owned("household", "children", "float64"),
                ),
            ),
            Node(
                "assembled",
                FixtureImportance.ref,
                base="source",
                inputs=(Slice("household", ("people",)),),
                structural=StructuralDelta.REWEIGHT,
                mass="free",
                weights=WeightTransition("household", "importance", mass="free"),
            ),
            fixture_node(),
        ),
    )
    registry = KernelRegistry()
    for kernel in (
        FixtureSource(),
        FixtureImportance(),
        DemographicCalibrationKernel(),
    ):
        if warm:

            def forbidden(context):
                raise AssertionError("A warm replay must execute no kernels")

            kernel.run = forbidden
        registry.register(kernel)
    result = run_graph(
        compile_graph(graph), sources={"fixture": path}, store=store, kernels=registry
    )
    calibrated = result.population("national.demographic_calibration")
    node = result.node("national.demographic_calibration")
    payload = store.load_bytes(node.opaque_artifacts["diagnostics"])
    direct = calibrate(
        fixture_frame(),
        fixture_registry().to_target_set(),
        weight_entity="household",
        method="adam",
        epochs=48,
        learning_rate=0.03,
        max_weight_ratio=4.0,
        mass="free",
        seed=0,
    )
    np.testing.assert_array_equal(
        calibrated.weights_for("household").values, direct.weights
    )
    pd.testing.assert_frame_equal(
        calibrated.table("person"), fixture_frame().table("person")
    )
    assert calibrated.weights_for("household").kind is WeightKind.CALIBRATED
    assert json.loads(payload) == diagnostics_payload(
        direct, target_registry=fixture_registry()
    )
    return {
        "key": result.key,
        "diagnostics_sha256": hashlib.sha256(payload).hexdigest(),
        "weight_bytes_sha256": hashlib.sha256(direct.weights.tobytes()).hexdigest(),
        "hits": [n.hit for n in result.nodes.values()],
    }


def test_real_graph_and_fresh_process_warm_diagnostics(tmp_path):
    cold = graph_fixture(tmp_path)
    assert cold["hits"] == [False, False, False]
    # The child removes foreign editable namespace entries after site startup.
    source_root = Path(__file__).resolve().parents[3]
    code = """import sys, pathlib, importlib.util, json
w = pathlib.Path(sys.argv[1])
sys.path[:] = [p for p in sys.path if not ('/packages/microcosm-' in p and p.endswith('/src'))]
sys.path[:0] = [str(p) for p in w.glob('packages/microcosm-*/src')]
spec = importlib.util.spec_from_file_location('calibration_graph_fixture', sys.argv[2])
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
pathlib.Path(sys.argv[4]).write_text(json.dumps(m.graph_fixture(sys.argv[3], warm=True)))
"""
    result_path = tmp_path / "fresh-warm.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(source_root),
            __file__,
            str(tmp_path),
            str(result_path),
        ],
        check=True,
        timeout=90,
    )
    warm = json.loads(result_path.read_text())
    assert warm["hits"] == [True, True, True]
    assert {k: v for k, v in warm.items() if k != "hits"} == {
        k: v for k, v in cold.items() if k != "hits"
    }
