"""Export artifacts bind exact H5 content and survive filesystem recreation."""

import numpy as np
import pandas as pd
import pytest
from uk_hierarchy_fixtures import uk_fixture_hierarchy

from microcosm.build.uk_runtime.graph_terminal import (
    add_uk_export_continuation,
    add_uk_export_preparation,
    describe_uk_export,
    materialize_uk_export,
    register_uk_terminal_kernels,
    validate_uk_export,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


def _frame():
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_clone_index": [0, 0],
            "region": ["LONDON", "SOUTH_EAST"],
            "oa_code": ["E00000001", "E00000002"],
            "lsoa_code": ["E01000001", "E01000002"],
            "msoa_code": ["E02000001", "E02000002"],
            "local_authority_code": ["E09000001", "E07000002"],
            "ward_code": ["E05000001", "E05000002"],
            "constituency_code": ["E14000001", "E14000002"],
            "region_code": ["E12000007", "E12000008"],
            "itl3_code": ["TLI31", "TLJ31"],
            "itl2_code": ["TLI3", "TLJ3"],
            "itl1_code": ["TLI", "TLJ"],
        }
    )
    for column in household.select_dtypes(include=["str", "object"]).columns:
        household[column] = household[column].astype("string")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_clone_index": [0, 0],
                "income": pd.Series([10.5, 20.5], dtype="float32"),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2], "benunit_clone_index": [0, 0]}),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.array([13.0, 87.0]),
        mass_log=(MassChangeRecord("household", 100.0, 100.0, 1.0, "calibration"),),
    )


def test_export_drops_native_aliases_and_keeps_identity_keyed_assignment(tmp_path):
    pytest.importorskip("tables")
    from microcosm.build.uk_runtime.atomic_area_support import (
        UK_NATIVE_ALIAS_COLUMNS,
    )
    from microcosm.build.uk_runtime.graph_terminal import _tables

    frame = _frame()
    household = frame.table("household")
    household["geography_household_key"] = pd.array(["k1", "k2"], dtype="string")
    household["atomic_area_code"] = household["oa_code"]
    household["atomic_area_system"] = pd.array(
        ["uk_ew_output_area_2021"] * 2, dtype="string"
    )
    household["atomic_area_basis"] = pd.array(["assigned"] * 2, dtype="string")
    household["output_area_code"] = household["oa_code"]
    for alias in UK_NATIVE_ALIAS_COLUMNS[1:]:
        household[alias] = pd.array([pd.NA, pd.NA], dtype="string")
    exported = _tables(frame)["household"].columns
    assert not set(UK_NATIVE_ALIAS_COLUMNS) & set(exported)
    assert {
        "geography_household_key",
        "atomic_area_code",
        "atomic_area_system",
        "atomic_area_basis",
        "oa_code",
        "ward_code",
        "itl1_code",
    } <= set(exported)
    descriptor = describe_uk_export(frame, bindings={"target_scope": "all"})
    assert "output_area_code" not in descriptor["tables"]["household"]["columns"]
    path = tmp_path / "full.h5"
    materialize_uk_export(frame, descriptor, path)
    assert validate_uk_export(path, descriptor)["passed"] is True
    with pd.HDFStore(path) as store:
        stored = store["household"].columns
    assert "geography_household_key" in stored and "data_zone_code" not in stored


def test_export_roundtrip_preserves_dtype_weights_lineage_period_and_gate(tmp_path):
    pytest.importorskip("tables")
    frame = _frame()
    descriptor = describe_uk_export(
        frame,
        bindings={"pool_replicates": 1, "dataset_households": 2, "target_scope": "all"},
    )
    path = tmp_path / "full.h5"
    record = materialize_uk_export(frame, descriptor, path)
    report = validate_uk_export(path, descriptor)
    assert report["passed"] is True
    assert record == report["dataset"]
    assert descriptor["tables"]["person"]["dtypes"][-1] == "float32"
    assert descriptor["time_period"] == "2024"
    assert descriptor["weight_kind"] == "calibrated"
    path.unlink()
    materialize_uk_export(frame, descriptor, path)
    assert validate_uk_export(path, descriptor)["passed"] is True


def test_export_refuses_population_changed_after_descriptor(tmp_path):
    frame = _frame()
    descriptor = describe_uk_export(frame, bindings={})
    frame.table("person").loc[0, "income"] = 99.0
    with pytest.raises(ValueError, match="descriptor"):
        materialize_uk_export(frame, descriptor, tmp_path / "changed.h5")


def test_export_readback_reports_changed_stored_values(tmp_path):
    pytest.importorskip("tables")
    frame = _frame()
    descriptor = describe_uk_export(frame, bindings={})
    path = tmp_path / "full.h5"
    materialize_uk_export(frame, descriptor, path)
    with pd.HDFStore(path) as store:
        person = store["person"]
        person.loc[0, "income"] = np.float32(77.0)
        store.put("person", person, format="table")
    report = validate_uk_export(path, descriptor)
    assert report["passed"] is False
    assert any("person" in failure for failure in report["failures"])


def test_graph_export_continuation_reuses_numerics_and_validates_recreated_file(
    tmp_path,
):
    import json

    from microcosm.graph import (
        Capabilities,
        ContentStore,
        Determinism,
        Graph,
        KernelBase,
        KernelRegistry,
        KernelResult,
        Node,
        Owned,
        SourceRef,
        StructuralDelta,
        compile_graph,
        run_graph,
    )

    pytest.importorskip("tables")

    class Create(KernelBase):
        ref = "fixture.create@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def run(self, context):
            return KernelResult(frame=_frame())

    frame = _frame()
    identifiers = {
        "person_id",
        "benunit_id",
        "household_id",
        "person_household_id",
        "person_benunit_id",
    }
    source = tmp_path / "fixture.txt"
    source.write_text("deterministic export fixture")
    graph = Graph(
        "uk",
        (SourceRef("fixture", "raw-bytes-v1"),),
        (
            Node(
                id="root",
                kernel=Create.ref,
                structural=StructuralDelta.CREATE,
                sources=("fixture",),
                outputs=tuple(
                    Owned(entity, column, str(table[column].dtype))
                    for entity in frame.entities
                    for table in (frame.table(entity),)
                    for column in table.columns
                    if column not in identifiers
                ),
            ),
        ),
    )
    graph = add_uk_export_preparation(
        graph,
        population="root",
        bindings={"pool_replicates": 1, "dataset_households": 2, "target_scope": "all"},
    )
    registry = KernelRegistry()
    registry.register(Create())
    register_uk_terminal_kernels(registry)
    store = ContentStore(tmp_path / "store")
    numerical = run_graph(
        compile_graph(graph), sources={"fixture": source}, store=store, kernels=registry
    )
    key = numerical.nodes["uk.full.export.prepare"].opaque_artifacts[
        "export_descriptor"
    ]
    descriptor = json.loads(store.load_bytes(key))
    path = tmp_path / "full.h5"
    materialize_uk_export(numerical.population("root"), descriptor, path)
    continued = add_uk_export_continuation(
        graph, population="root", manifest_binding={"key": numerical.key}
    )
    terminal = run_graph(
        compile_graph(continued),
        sources={"fixture": source, "exported_dataset": path},
        store=store,
        kernels=registry,
    )
    assert terminal.nodes["root"].store_hit
    assert terminal.nodes["uk.full.export.prepare"].store_hit
    assert terminal.nodes["uk.full.export.readback"].receipt["outcome"] == "pass"
    inventory = json.loads(
        store.load_bytes(
            terminal.nodes["uk.full.package"].opaque_artifacts["package_inventory"]
        )
    )
    assert inventory["numerical_graph"]["key"] == numerical.key
    assert inventory["build_bindings"]["target_scope"] == "all"
    path.unlink()
    materialize_uk_export(numerical.population("root"), descriptor, path)
    replay = run_graph(
        compile_graph(continued),
        sources={"fixture": source, "exported_dataset": path},
        store=store,
        kernels=registry,
    )
    assert replay.nodes["root"].store_hit
    assert replay.nodes["uk.full.export.readback"].receipt["outcome"] == "pass"


@pytest.mark.parametrize("households", [None, 2])
def test_full_gate_nodes_precede_dense_and_bind_final_problem_axis(households):
    from uk_atomic_support_fixtures import toy_support_payloads

    from microcosm.build.uk_runtime.atomic_area_support import (
        uk_atomic_assignment_definition,
    )
    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
    from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
    from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
    from microcosm.graph import compile_graph

    raw = uk_spine_graph(source_mode="split")
    config = UKFullBuildConfig(
        calibration_year=2025,
        calibration=UKGraphCalibrationConfig(dataset_households=households),
    )
    full = uk_full_graph(
        config,
        spine=raw,
        atomic_geography_definition=uk_atomic_assignment_definition(
            toy_support_payloads(), seed=config.seed
        ),
    )
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=uk_spine_endpoint(raw).stage_names,
        engine_identity="fixture-engine",
        review_date="2026-09-10",
    )
    compiled = compile_graph(graph)
    assert (
        "uk.full.gates.preflight"
        in compiled.predecessors[full.calibration.dense_producer]
    )
    preflight = graph.node("uk.full.gates.preflight")
    assert not preflight.inputs
    assert not {"problem", "solution"} & {
        item.name for item in preflight.artifact_inputs
    }
    final = graph.node("uk.full.gates.calibrated")
    assert final.population == full.calibration.population
    inputs = {item.name: item for item in final.artifact_inputs}
    assert inputs["problem"].producer == full.calibration.problem_producer
    assert inputs["solution"].artifact == (
        "solution" if households is None else "refit_solution"
    )


def test_full_preflight_persists_real_source_failures_without_matrix_diagnostics(
    monkeypatch,
):
    import json
    from types import SimpleNamespace

    from microcosm.build.gate_battery import _gates_manifest_payload
    from microcosm.build.uk_runtime import full_gates
    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullGateKernel,
        decode_full_gate_report,
    )
    from microcosm.graph.canonical import canonical_json

    selection = {
        "schema": "microcosm.calibrate.target-selection.v1",
        "selector": {"geography_levels": None},
        "included": [
            {"name": "n", "period": 2025, "geography_level": "country"},
            {"name": "l", "period": 2025, "geography_level": "constituency"},
        ],
        "excluded": [],
    }
    empty_registry = {"country": "uk", "specs": []}

    def artifact(payload):
        return SimpleNamespace(payload=canonical_json(payload), key="a" * 64)

    context = SimpleNamespace(
        params={
            "engine_identity": "fixture",
            "phase": "preflight",
            "gate_manifest": canonical_json(
                _gates_manifest_payload(full_gates.uk_full_gate_manifest())
            ).decode(),
            "spine_stage_names": ("frs_spine",),
            "review_date": "2026-09-10",
            "sample_fraction": 1.0,
            "release_candidate": True,
        },
        artifacts={
            "selection": artifact({"receipt": selection}),
            "surface": artifact(
                {
                    "national_registry": empty_registry,
                    "uk_ledger_compiled_registries": {
                        "2023": empty_registry,
                        "2025": empty_registry,
                    },
                    "uk_ledger_compiled_local_registries": {"2025": empty_registry},
                }
            ),
            "spine_provenance": artifact(
                {"stages": ["frs_spine"], "stage_evidence": {}}
            ),
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Source preflight must not compute final matrix diagnostics"
        )

    monkeypatch.setattr(full_gates, "build_full_gate_context", forbidden)
    result = UKFullGateKernel(coverage_engine=object(), engine_identity="fixture").run(
        context
    )
    report, enforcement = decode_full_gate_report(result.artifacts["gate_report"])
    assert report.phase == "preflight"
    assert len(report.outcomes) == 6
    assert enforcement["artifact_permitted"] is False
    assert "uk_release_family_build_stages" in enforcement["structural_failures"]
    document = json.loads(result.artifacts["gate_report"])
    assert document["target_diagnostics"] == []
    document["enforcement"]["artifact_permitted"] = True
    with pytest.raises(ValueError, match="enforcement"):
        decode_full_gate_report(document)


def test_terminal_byte_materialization_recreates_exact_files(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.graph_terminal import (
        materialize_uk_terminal_artifacts,
    )
    from microcosm.graph import ContentStore

    payloads = {
        "calibration_diagnostics": b'{"score":1}',
        "target_diagnostics_csv": b"target,estimate\na,1\n",
        "area_support_csv": b"area,rows\na,2\n",
        "holdout": b'{"report_only":true}',
        "selection": b'{"registry":{}}',
    }
    store = ContentStore(tmp_path / "store")
    keys = {}
    for name, payload in payloads.items():
        key = hashlib.sha256(payload).hexdigest()
        store.put_bytes(key, payload)
        keys[name] = key
    manifest = SimpleNamespace(
        nodes={
            "uk.full.gates.calibrated": SimpleNamespace(opaque_artifacts=keys),
            "uk.full.holdout": SimpleNamespace(opaque_artifacts=keys),
            "uk.full.target_selection": SimpleNamespace(opaque_artifacts=keys),
        }
    )
    first = materialize_uk_terminal_artifacts(
        manifest, store, directory=tmp_path, stem="full"
    )
    for record in first.values():
        (tmp_path / record["filename"]).unlink()
    assert (
        materialize_uk_terminal_artifacts(
            manifest, store, directory=tmp_path, stem="full"
        )
        == first
    )


def test_holdout_kernel_preserves_existing_rotations_and_seed_settings(monkeypatch):
    import json
    from types import SimpleNamespace

    from test_uk_full_calibration_graph import preflight_payload
    from test_uk_local_rowwise import _assigned, _clone_frame

    from microcosm.build.uk_runtime import graph_targets, local_rowwise
    from microcosm.build.uk_runtime.graph_terminal import UKFullHoldoutKernel
    from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
    from microcosm.calibrate.artifacts import encode_problem

    frame = _clone_frame()
    names = ("households", "tenure/social_rent", "tenure/private_rent")
    problem = local_rowwise.build_uk_rowwise_local_matrix(
        pd.DataFrame({name: [1.0, 2.0, 3.0] for name in names}, index=[101, 102, 103]),
        _assigned(),
        pd.DataFrame(
            {"code": ["E001", "S001"], **{name: [3.0, 3.0] for name in names}}
        ),
    )
    calls = []

    def solve(frame, training, **kwargs):
        calls.append(
            (kwargs["seed"], kwargs["dataset_households"], kwargs["selection_seed"])
        )
        return SimpleNamespace(weights=np.ones(2), selected_support=np.array([0, 2]))

    monkeypatch.setattr(local_rowwise, "solve_uk_rowwise_weights_under_doctrine", solve)
    monkeypatch.setattr(
        local_rowwise,
        "_derive_uk_local_bound_families_from_target_frame",
        lambda *a, **k: (),
    )
    monkeypatch.setattr(
        graph_targets,
        "reconstruct_uk_full_problem_inputs",
        lambda context: SimpleNamespace(
            frame=frame, local_problem=problem, national_rows=None, bound_families=()
        ),
    )
    original = encode_problem(
        build_constraint_matrix(
            frame,
            TargetSet([Target("count", "household", lambda f: np.ones(3), 3.0)]),
            "household",
        ),
        entity_ids=[101, 102, 103],
    )
    context = SimpleNamespace(
        params={
            "skip_holdout": False,
            "target_weight_rule": "uniform",
            "epochs": 1,
            "learning_rate": 0.1,
            "dataset_households": 2,
            "seed": 42,
            "selection_seed": 17,
            "selection_pi_hi": 1.0,
        },
        artifacts={
            "preflight": SimpleNamespace(payload=preflight_payload(), key="a" * 64),
            "problem": SimpleNamespace(payload=original, key="b" * 64),
        },
    )
    result = json.loads(UKFullHoldoutKernel().run(context).artifacts["holdout"])
    expected = local_rowwise.rotated_uk_local_holdout(
        frame,
        problem,
        bound_families=(),
        epochs=1,
        learning_rate=0.1,
        dataset_households=2,
        solve_seed=42,
        selection_seed=17,
    )
    assert {
        key: value for key, value in result.items() if key != "graph_binding"
    } == expected
    assert calls == [(42, 2, 17)] * 10


def test_final_gate_kernel_owns_complete_diagnostics_and_reuses_decoded_result(
    monkeypatch,
):
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    from test_uk_full_calibration_graph import preflight_payload

    from microcosm.build.gate_battery import _gates_manifest_payload
    from microcosm.build.uk_runtime import full_gates, geography_ladder
    from microcosm.build.uk_runtime.graph_targets import registry_payload
    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullGateKernel,
        decode_full_gate_report,
    )
    from microcosm.calibrate import (
        TargetRegistry,
        TargetSet,
        TargetSpec,
        build_constraint_matrix,
        calibrate,
    )
    from microcosm.calibrate.artifacts import (
        decode_problem,
        encode_calibration_result,
        encode_problem,
        encode_solution,
    )
    from microcosm.frame import Frame, Weights
    from microcosm.graph.canonical import canonical_json

    original = _frame()
    tables = {entity: original.table(entity).copy() for entity in original.entities}
    tables["household"]["source_household_id"] = [1, 2]
    tables["household"]["household_is_spi_synthetic"] = False
    tables["household"]["household_is_capital_gains_clone"] = False
    initial = Frame(
        tables,
        original.schema,
        {"household": Weights(np.array([13.0, 87.0]), WeightKind.IMPORTANCE)},
        original.strata,
        metadata=original.metadata,
    )
    specs = [
        TargetSpec(
            name="national",
            entity="household",
            value=100.0,
            measure="household_id",
            period=2024,
            source="fixture",
            family="households",
            hierarchy=uk_fixture_hierarchy(
                "national", level="country", geography_id="UK"
            ),
            metadata={
                "geography_level": "country",
                "geography_id": "UK",
                "materialization": "uk_national_measure",
            },
        ),
        TargetSpec(
            name="local",
            entity="household",
            value=13.0,
            measure="household_id",
            period=2024,
            source="fixture",
            family="census_households",
            hierarchy=uk_fixture_hierarchy(
                "local", level="constituency", geography_id="E14000001"
            ),
            metadata={
                "geography_level": "constituency",
                "geography_id": "E14000001",
                "materialization": "uk_local_surface",
                "area_type": "constituency",
                "area_code": "E14000001",
                "metric": "households",
            },
        ),
    ]
    targets = TargetSet(
        [
            replace(specs[0].to_target(), measure=lambda f: np.ones(2)),
            replace(specs[1].to_target(), measure=lambda f: np.array([1.0, 0.0])),
        ]
    )
    selection = {
        "schema": "microcosm.calibrate.target-selection.v1",
        "selector": {"geography_levels": None},
        "included": [
            {
                "name": spec.name,
                "period": 2024,
                "geography_level": spec.metadata["geography_level"],
            }
            for spec in specs
        ],
        "excluded": [],
    }
    problem = build_constraint_matrix(initial, targets, "household")
    problem_bytes = encode_problem(
        problem,
        entity_ids=[1, 2],
        target_metadata=[{**spec.metadata, "family": spec.family} for spec in specs],
        bindings={"target_selection": selection},
    )
    ordered = decode_problem(problem_bytes)
    result = calibrate(initial, targets, weight_entity="household", epochs=1, seed=42)
    frame = result.frame

    def artifact(payload, raw=False):
        return SimpleNamespace(
            payload=payload if raw else canonical_json(payload), key="a" * 64
        )

    empty = {"country": "uk", "specs": []}
    artifacts = {
        "problem": artifact(problem_bytes, True),
        "result": artifact(
            encode_calibration_result(
                result, entity_ids=[1, 2], problem_sha256=ordered.sha256
            ),
            True,
        ),
        "solution": artifact(
            encode_solution(
                result.weights, entity_ids=[1, 2], problem_sha256=ordered.sha256
            ),
            True,
        ),
        "selection": artifact(
            {
                "receipt": selection,
                "registry": registry_payload(TargetRegistry(specs, country="uk")),
            }
        ),
        "preflight": artifact(preflight_payload(selection=selection), True),
        "spine_provenance": artifact(
            {"stages": ["frs_spine"], "stage_evidence": {}, "fit_weight_records": {}}
        ),
        "surface": artifact(
            {
                "national_registry": registry_payload(
                    TargetRegistry(specs[:1], country="uk")
                ),
                "uk_ledger_compiled_registries": {"2023": empty, "2025": empty},
                "uk_ledger_compiled_local_registries": {"2025": empty},
            }
        ),
        "holdout": artifact({"report_only": True, "outcome": "fixture"}),
    }
    monkeypatch.setattr(
        geography_ladder,
        "load_uk_oa_ladder",
        lambda path: SimpleNamespace(
            constituency_code=np.array(["E14000001", "E14000002"]),
            local_authority_code=np.array(["E09000001", "E07000002"]),
        ),
    )
    monkeypatch.setattr(
        full_gates,
        "load_efrs_parity_reference",
        lambda: SimpleNamespace(input_entities={}),
    )
    monkeypatch.setattr(
        full_gates, "uk_aggregate_admin_totals", lambda frame, gates: ({}, [])
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("The decoded result diagnostics must be reused")

    monkeypatch.setattr(full_gates, "_build_diagnostics", forbidden)
    context = SimpleNamespace(
        tables={entity: frame.table(entity) for entity in frame.entities},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        frame_metadata=frame.metadata,
        frame_mass_log=frame.mass_log,
        frame_column_order={},
        params={
            "phase": "terminal",
            "engine_identity": "fixture",
            "review_date": "2026-09-10",
            "sample_fraction": 1.0,
            "release_candidate": False,
            "spine_stage_names": ("frs_spine",),
            "gate_manifest": canonical_json(
                _gates_manifest_payload(full_gates.uk_full_gate_manifest())
            ).decode(),
        },
        artifacts=artifacts,
        sources={"uk_ladder": "fixture"},
    )
    stored = UKFullGateKernel(coverage_engine=object(), engine_identity="fixture").run(
        context
    )
    phase, _ = decode_full_gate_report(stored.artifacts["gate_report"])
    assert phase.phase == "terminal"
    document = json.loads(stored.artifacts["calibration_diagnostics"])
    assert document["uk_diagnostics"]["rotated_holdout"]["outcome"] == "fixture"
    assert len(document["targets"]) == 2
    assert (
        len(pd.read_csv(__import__("io").BytesIO(stored.artifacts["area_support_csv"])))
        == 4
    )


def test_package_validates_materialized_evidence_against_graph_bytes(tmp_path):
    import json
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.graph_terminal import UKPackageInventoryKernel
    from microcosm.graph.canonical import canonical_json

    payload = b'{"diagnostics":"graph-owned"}'
    evidence = tmp_path / "full.diagnostics.json"
    evidence.write_bytes(payload)
    readback = {
        "schema_version": 1,
        "kind": "uk_full_build_export_readback",
        "passed": True,
        "dataset": {"filename": "full.h5", "sha256": "b" * 64, "size_bytes": 10},
        "content_sha256": "c" * 64,
        "bindings": {"K": 20, "k": 2, "scope": "all"},
    }
    context = SimpleNamespace(
        params={
            "manifest_binding": "{}",
            "evidence_files": json.dumps({"diagnostics": evidence.name}),
        },
        sources={"exported_evidence_diagnostics": evidence},
        artifacts={
            "export_readback": SimpleNamespace(
                payload=canonical_json(readback), key="d" * 64
            ),
            "diagnostics": SimpleNamespace(payload=payload, key="a" * 64),
        },
    )
    document = json.loads(
        UKPackageInventoryKernel().run(context).artifacts["package_inventory"]
    )
    assert document["evidence_files"]["diagnostics"]["graph_artifact_key"] == "a" * 64
    evidence.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from its graph artifact"):
        UKPackageInventoryKernel().run(context)
    readback["passed"] = False
    context.artifacts["export_readback"].payload = canonical_json(readback)
    with pytest.raises(ValueError, match="H5 readback failed"):
        UKPackageInventoryKernel().run(context)


@pytest.mark.requires_uk
def test_gate_cache_identity_includes_country_reference_resource_bytes(monkeypatch):
    from types import SimpleNamespace

    from microcosm.build import country_spec
    from microcosm.build.uk_runtime.graph_terminal import UKFullGateKernel

    kernel = UKFullGateKernel(coverage_engine=object(), engine_identity="fixture")
    monkeypatch.setattr(
        country_spec,
        "load_country_spec",
        lambda country: SimpleNamespace(fingerprint="a" * 64),
    )
    first = kernel.implementation_hash()
    monkeypatch.setattr(
        country_spec,
        "load_country_spec",
        lambda country: SimpleNamespace(fingerprint="b" * 64),
    )
    assert kernel.implementation_hash() != first
