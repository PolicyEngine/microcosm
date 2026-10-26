"""Export artifacts bind exact H5 content and survive filesystem recreation."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.graph_terminal import (
    add_uk_export_continuation,
    add_uk_export_preparation,
    describe_uk_export,
    materialize_uk_export,
    register_uk_terminal_kernels,
    validate_uk_export,
)
from microcosm.build.uk_runtime.weight_year import (
    UK_WEIGHT_YEAR_MASS_REASON,
    uk_weight_projection_factor,
)
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_graph_terminal import _frame
from test_support.microcosm_build.uk_hierarchy_fixtures import (
    uk_fixture_hierarchy,
)
from test_support.microcosm_diagnostics.json_schema import (
    load_strict_json,
    published_json_schema_errors,
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
    descriptor = describe_uk_export(
        frame, bindings={"target_scope": "all"}, solve_year=2025
    )
    assert "output_area_code" not in descriptor["tables"]["household"]["columns"]
    path = tmp_path / "full.h5"
    materialize_uk_export(frame, descriptor, path)
    assert validate_uk_export(path, descriptor)["passed"] is True
    with pd.HDFStore(path) as store:
        stored = store["household"].columns
    assert "geography_household_key" in stored and "data_zone_code" not in stored


def test_local_export_drops_the_national_boundarys_reviewed_columns():
    """The local file leaves out what the national release boundary drops.

    microcosm#1278 (from #796): the K=25 build's degenerate-surface gate failed
    on ``incapacity_benefit_reported``, which the national boundary dropped and
    the local export did not. Both exports now apply the same reviewed list.
    """
    from microcosm.build.uk_runtime.graph_terminal import _tables
    from microcosm.build.uk_runtime.national_frame import (
        UK_RELEASE_EXPORT_DROPPED_COLUMNS,
    )

    assert "incapacity_benefit_reported" in UK_RELEASE_EXPORT_DROPPED_COLUMNS["person"]
    frame = _frame()
    for entity, columns in UK_RELEASE_EXPORT_DROPPED_COLUMNS.items():
        table = frame.table(entity)
        for column in columns:
            table[column] = 0.0
    exported = _tables(frame)
    descriptor = describe_uk_export(
        frame, bindings={"target_scope": "all"}, solve_year=2025
    )
    for entity, columns in UK_RELEASE_EXPORT_DROPPED_COLUMNS.items():
        assert not set(columns) & set(exported[entity].columns)
        assert not set(columns) & set(descriptor["tables"][entity]["columns"])


def test_export_writes_area_codes_under_consumer_names_and_refuses_ladder_names(
    tmp_path,
):
    """microcosm#1114: the artifact carries ``constituency_code_oa``,
    ``la_code_oa`` and ``region_code_oa`` and never the ladder names."""
    pytest.importorskip("tables")
    from microcosm.build.uk_runtime.geography_ladder import (
        UK_EXPORT_AREA_CODE_COLUMNS,
        export_area_code_columns,
    )
    from microcosm.build.uk_runtime.graph_terminal import _tables

    frame = _frame()
    ladder = frame.table("household")
    exported = _tables(frame)["household"]
    for ladder_name, export_name in UK_EXPORT_AREA_CODE_COLUMNS.items():
        assert ladder_name not in exported.columns
        assert exported[export_name].tolist() == ladder[ladder_name].tolist()
    # The in-memory frame is untouched: gates and diagnostics keep the ladder names.
    assert "constituency_code" in frame.table("household").columns
    descriptor = describe_uk_export(
        frame, bindings={"target_scope": "all"}, solve_year=2025
    )
    assert descriptor["area_codes"]["columns"] == dict(UK_EXPORT_AREA_CODE_COLUMNS)
    frames = descriptor["area_codes"]["frames"]
    assert set(frames) == set(UK_EXPORT_AREA_CODE_COLUMNS.values())
    assert frames["constituency_code_oa"]["uk_ew_output_area_2021"] == "2024_pcon"
    assert frames["la_code_oa"]["uk_ew_output_area_2021"] == "2023_april_lad"
    assert frames["region_code_oa"]["uk_ew_output_area_2021"] == "2024_rgn"
    assert "constituency_code_oa" in descriptor["tables"]["household"]["columns"]
    assert "constituency_code" not in descriptor["tables"]["household"]["columns"]
    path = tmp_path / "full.h5"
    materialize_uk_export(frame, descriptor, path)
    assert validate_uk_export(path, descriptor)["passed"] is True
    with pd.HDFStore(path) as store:
        stored = store["household"]
    assert {"constituency_code_oa", "la_code_oa", "region_code_oa"} <= set(
        stored.columns
    )
    # A stored table that slipped a ladder name or an empty code through is refused.
    with pd.HDFStore(path) as store:
        household = store["household"].rename(
            columns={"constituency_code_oa": "constituency_code"}
        )
        household["la_code_oa"] = ["", "E07000008"]
        store.put("household", household, format="table")
    report = validate_uk_export(path, descriptor)
    assert report["passed"] is False
    assert any("ladder area code 'constituency_code'" in f for f in report["failures"])
    assert any(
        "lacks the declared area code 'constituency_code_oa'" in f
        for f in report["failures"]
    )
    assert any("'la_code_oa' is empty on 1 household" in f for f in report["failures"])
    # A stale consumer column on the frame cannot ride through the boundary.
    stale = ladder.assign(constituency_code_oa=["stale", "stale"])
    with pytest.raises(ValueError, match="already carries consumer area-code"):
        export_area_code_columns(stale)
    # A table without geography passes unchanged (the national line).
    plain = pd.DataFrame({"household_id": [1], "region": ["LONDON"]})
    assert export_area_code_columns(plain) is plain


def test_export_roundtrip_preserves_dtype_weights_lineage_period_and_gate(tmp_path):
    pytest.importorskip("tables")
    frame = _frame()
    descriptor = describe_uk_export(
        frame,
        bindings={"pool_replicates": 1, "dataset_households": 2, "target_scope": "all"},
        solve_year=2025,
    )
    path = tmp_path / "full.h5"
    record = materialize_uk_export(frame, descriptor, path)
    report = validate_uk_export(path, descriptor)
    assert report["passed"] is True
    assert record == report["dataset"]
    assert descriptor["tables"]["person"]["dtypes"][-1] == "float32"
    assert descriptor["time_period"] == "2024"
    assert descriptor["weight_kind"] == "calibrated"
    # The artifact carries the solved weights at the dataset's year
    # (microcosm#796 A19); the frame the graph holds keeps them as solved.
    factor = uk_weight_projection_factor(2024, 2025)
    assert descriptor["weight_year"] == {
        "solve_year": 2025,
        "dataset_year": 2024,
        "engine_projection_factor": factor,
    }
    assert descriptor["mass_log"][-1]["reason"] == UK_WEIGHT_YEAR_MASS_REASON
    assert descriptor["mass_log"][-1]["declared_factor"] == 1.0 / factor
    with pd.HDFStore(path) as store:
        np.testing.assert_array_equal(
            store["household"]["household_weight"].to_numpy(),
            np.array([13.0, 87.0]) / factor,
        )
    np.testing.assert_array_equal(frame.weights_for("household").values, [13.0, 87.0])
    path.unlink()
    materialize_uk_export(frame, descriptor, path)
    assert validate_uk_export(path, descriptor)["passed"] is True


def test_export_refuses_population_changed_after_descriptor(tmp_path):
    frame = _frame()
    descriptor = describe_uk_export(frame, bindings={}, solve_year=2025)
    frame.table("person").loc[0, "income"] = 99.0
    with pytest.raises(ValueError, match="descriptor"):
        materialize_uk_export(frame, descriptor, tmp_path / "changed.h5")


def test_export_readback_reports_changed_stored_values(tmp_path):
    pytest.importorskip("tables")
    frame = _frame()
    descriptor = describe_uk_export(frame, bindings={}, solve_year=2025)
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
        solve_year=2025,
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
    from microcosm.build.uk_runtime.atomic_area_support import (
        uk_atomic_assignment_definition,
    )
    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
    from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
    from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
    from microcosm.graph import compile_graph
    from test_support.microcosm_build.uk_atomic_support_fixtures import (
        toy_support_payloads,
    )

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
    # microcosm#1301: the preflight feeds the solve and keys on the solve
    # fingerprint under its own kernel; the calibrated gates keep the full one.
    assert preflight.kernel == "uk.full-gates.preflight@1"
    assert final.kernel == "uk.full-gates@1"
    assert final.population == full.calibration.population
    inputs = {item.name: item for item in final.artifact_inputs}
    assert inputs["problem"].producer == full.calibration.problem_producer
    assert inputs["solution"].artifact == (
        "solution" if households is None else "refit_solution"
    )
    # microcosm#1301: the holdout's fold searches share the size search's
    # warm start and L0 floor, both holdout parameters.
    holdout = graph.node("uk.full.holdout").params
    assert holdout["selection_l0_search_floor"] == (
        None if households is None else 1e-10
    )


def test_full_preflight_persists_real_source_failures_without_matrix_diagnostics(
    monkeypatch,
):
    import json
    from types import SimpleNamespace

    from microcosm.build.gate_battery import _gates_manifest_payload
    from microcosm.build.uk_runtime import full_gates
    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullPreflightGateKernel,
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
                    "uk_ledger_compiled_registries": {"2025": empty_registry},
                    "uk_ledger_compiled_local_registries": {"2025": empty_registry},
                    "uk_ledger_compile_unsupported": {"2025": []},
                    "uk_ledger_compile_local_unsupported": {"2025": []},
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
    result = UKFullPreflightGateKernel(
        coverage_engine=object(), engine_identity="fixture"
    ).run(context)
    report, enforcement = decode_full_gate_report(result.artifacts["gate_report"])
    assert report.phase == "preflight"
    assert len(report.outcomes) == 5
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
        "high_value_home_support_csv": b"level,area_code,rows\nconstituency,a,2\n",
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
    from datetime import date
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import graph_targets, local_rowwise
    from microcosm.build.uk_runtime.graph_terminal import UKFullHoldoutKernel
    from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
    from microcosm.calibrate.artifacts import encode_problem
    from test_support.microcosm_build.uk_full_calibration_graph import (
        preflight_payload,
    )
    from test_support.microcosm_build.uk_local_rowwise import (
        _assigned,
        _clone_frame,
    )

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
    reviewed_on = []

    def solve(frame, training, **kwargs):
        calls.append(
            (
                kwargs["seed"],
                kwargs["dataset_households"],
                kwargs["selection_seed"],
                kwargs["selection_initial_lambda"],
            )
        )
        reviewed_on.append(kwargs["now"])
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
            "baseline_pi_floor": 0.0,
            # Every fold's budget search starts where the main search started.
            "selection_initial_lambda": 1.7e-7,
            # Inherited from uk.full.problem: the binding's review date.
            "review_date": "2026-10-20",
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
        selection_initial_lambda=1.7e-7,
    )
    assert {
        key: value for key, value in result.items() if key != "graph_binding"
    } == expected
    assert calls == [(42, 2, 17, 1.7e-7)] * 10
    # Every fold the kernel ran reviewed the binding on the build's review
    # date (microcosm#1278); the direct call above reviewed on today's.
    assert reviewed_on == [date(2026, 10, 20)] * 5 + [None] * 5


def _bound_problem_inputs():
    """A clone frame and a local problem that binds two fenced families."""

    from types import SimpleNamespace

    from microcosm.build.uk_runtime import local_rowwise
    from test_support.microcosm_build.uk_local_rowwise import (
        _assigned,
        _clone_frame,
    )

    problem = local_rowwise.build_uk_rowwise_local_matrix(
        pd.DataFrame(
            {"households": [2.0, 1.0, 3.0], "tenure/social_rent": [1.0, 0.0, 1.0]},
            index=[101, 102, 103],
        ),
        _assigned(),
        pd.DataFrame(
            {
                "code": ["E001", "S001"],
                "households": [4.0, 2.0],
                "tenure/social_rent": [1.0, 1.0],
            }
        ),
    )
    return SimpleNamespace(
        frame=_clone_frame(),
        local_problem=problem,
        national_rows=None,
        bound_families=["census_households/constituency", "tenure/constituency"],
        selection=None,
        selected=(),
        metadata={},
        full=None,
        cross=None,
        rung=None,
        country_only=False,
    )


def test_problem_kernel_reviews_the_binding_on_the_build_review_date(monkeypatch):
    """microcosm#1278: uk.full.problem reviews the local binding
    adjudications on its review_date parameter, not the wall clock, so a date
    past the committed register's windows is refused."""
    from datetime import date
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import graph_targets
    from microcosm.build.uk_runtime.graph_targets import UKFullProblemKernel

    inputs = _bound_problem_inputs()
    monkeypatch.setattr(
        graph_targets, "reconstruct_uk_full_problem_inputs", lambda context: inputs
    )

    def context(review_date):
        return SimpleNamespace(
            params={"target_weight_rule": "uniform", "review_date": review_date},
            artifacts={},
        )

    class ReviewedError(Exception):
        pass

    reviewed_on = []

    def prepare(*args, **kwargs):
        reviewed_on.append(kwargs["now"])
        raise ReviewedError

    with monkeypatch.context() as patch:
        patch.setattr(graph_targets, "prepare_uk_full_solve", prepare)
        with pytest.raises(ReviewedError):
            UKFullProblemKernel().run(context("2026-10-20"))
    assert reviewed_on == [date(2026, 10, 20)]
    with pytest.raises(ValueError, match="outside its reviewed window"):
        UKFullProblemKernel().run(context("2099-01-01"))


def test_holdout_kernel_refuses_a_review_date_past_the_binding_windows(monkeypatch):
    """microcosm#1278: every fold of the rotated holdout reviews the binding
    on the review date the holdout inherits from uk.full.problem."""
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import graph_targets, local_rowwise
    from microcosm.build.uk_runtime.graph_terminal import UKFullHoldoutKernel
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.calibrate import Target, TargetSet, build_constraint_matrix
    from microcosm.calibrate.artifacts import encode_problem
    from test_support.microcosm_build.uk_full_calibration_graph import (
        preflight_payload,
    )

    # Five areas, so each of the five folds holds one out; the household
    # counts bind a fenced family.
    household_ids = [101, 102, 103, 104, 105]
    area_codes = ["E001", "W001", "S001", "N001", "E002"]
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "person_household_id": household_ids,
                "person_benunit_id": [11, 12, 13, 14, 15],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [11, 12, 13, 14, 15]}),
        household=pd.DataFrame(
            {"household_id": household_ids, "household_weight": [1.0] * 5}
        ),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
    )
    problem = local_rowwise.build_uk_rowwise_local_matrix(
        pd.DataFrame({"households": np.ones(5)}, index=household_ids),
        pd.Series(area_codes, index=household_ids),
        pd.DataFrame({"code": area_codes, "households": [2.0] * 5}),
    )
    inputs = SimpleNamespace(
        frame=frame, local_problem=problem, national_rows=None, bound_families=None
    )
    monkeypatch.setattr(
        graph_targets, "reconstruct_uk_full_problem_inputs", lambda context: inputs
    )
    original = encode_problem(
        build_constraint_matrix(
            frame,
            TargetSet([Target("count", "household", lambda f: np.ones(5), 5.0)]),
            "household",
        ),
        entity_ids=household_ids,
    )
    context = SimpleNamespace(
        params={
            "skip_holdout": False,
            "target_weight_rule": "uniform",
            "epochs": 1,
            "learning_rate": 0.1,
            "seed": 42,
            "selection_pi_hi": 1.0,
            "baseline_pi_floor": 0.0,
            "review_date": "2099-01-01",
        },
        artifacts={
            "preflight": SimpleNamespace(payload=preflight_payload(), key="a" * 64),
            "problem": SimpleNamespace(payload=original, key="b" * 64),
        },
    )
    with pytest.raises(ValueError, match="outside its reviewed window"):
        UKFullHoldoutKernel().run(context)


def test_holdout_kernel_refuses_a_floor_its_folds_cannot_honour():
    """microcosm#1301: each fold's size search runs under the doctrine's L0
    floor, so a holdout keyed on another floor refuses before any fold."""
    from types import SimpleNamespace

    import pytest

    from microcosm.build.uk_runtime.graph_terminal import UKFullHoldoutKernel
    from test_support.microcosm_build.uk_full_calibration_graph import (
        preflight_payload,
    )

    context = SimpleNamespace(
        params={"skip_holdout": False, "selection_l0_search_floor": 1e-7},
        artifacts={
            "preflight": SimpleNamespace(payload=preflight_payload(), key="a" * 64)
        },
    )
    with pytest.raises(ValueError, match="L0 floor of 1e-07"):
        UKFullHoldoutKernel().run(context)


def test_holdout_kernel_records_an_unrequested_holdout(monkeypatch):
    """microcosm#1301: the rotated holdout is an opt-in diagnostic; the kernel
    records the skip with its reason and runs no fold."""
    import json
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import graph_targets
    from microcosm.build.uk_runtime.graph_terminal import UKFullHoldoutKernel
    from test_support.microcosm_build.uk_full_calibration_graph import (
        preflight_payload,
    )

    monkeypatch.setattr(
        graph_targets,
        "reconstruct_uk_full_problem_inputs",
        lambda context: pytest.fail("a skipped holdout reconstructs no problem"),
    )
    context = SimpleNamespace(
        params={"skip_holdout": True},
        artifacts={
            "preflight": SimpleNamespace(payload=preflight_payload(), key="a" * 64),
            "problem": SimpleNamespace(payload=b"unread", key="b" * 64),
        },
    )
    result = json.loads(UKFullHoldoutKernel().run(context).artifacts["holdout"])
    assert result["skipped"] is True and result["report_only"] is True
    assert result["reason"] == (
        "Not requested: opt-in diagnostic; no out-of-sample claim."
    )
    assert result["graph_binding"]["original_problem_artifact"] == "b" * 64


def test_full_gate_nodes_skip_the_holdout_unless_requested():
    from microcosm.build.uk_runtime.atomic_area_support import (
        uk_atomic_assignment_definition,
    )
    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
    from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
    from test_support.microcosm_build.uk_atomic_support_fixtures import (
        toy_support_payloads,
    )

    raw = uk_spine_graph(source_mode="split")
    config = UKFullBuildConfig(calibration_year=2025)
    full = uk_full_graph(
        config,
        spine=raw,
        atomic_geography_definition=uk_atomic_assignment_definition(
            toy_support_payloads(), seed=config.seed
        ),
    )
    common = {
        "calibration": full.calibration,
        "spine_stage_names": uk_spine_endpoint(raw).stage_names,
        "engine_identity": "fixture-engine",
        "review_date": "2026-09-10",
    }
    default = append_uk_full_gate_nodes(full.graph, **common)
    assert default.node("uk.full.holdout").params["skip_holdout"] is True
    requested = append_uk_full_gate_nodes(full.graph, **common, skip_holdout=False)
    assert requested.node("uk.full.holdout").params["skip_holdout"] is False
    assert requested.node("uk.full.holdout").description.startswith(
        "Optional (--rotated-holdout)"
    )


def _gate_nodes(**extra):
    from microcosm.build.uk_runtime.atomic_area_support import (
        uk_atomic_assignment_definition,
    )
    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
    from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
    from test_support.microcosm_build.uk_atomic_support_fixtures import (
        toy_support_payloads,
    )

    raw = uk_spine_graph(source_mode="split")
    config = UKFullBuildConfig(calibration_year=2025)
    full = uk_full_graph(
        config,
        spine=raw,
        review_date="2026-09-10",
        atomic_geography_definition=uk_atomic_assignment_definition(
            toy_support_payloads(), seed=config.seed
        ),
    )
    return append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=uk_spine_endpoint(raw).stage_names,
        engine_identity="fixture-engine",
        review_date="2026-09-10",
        **extra,
    )


def test_only_the_calibrated_gates_take_the_gate_clock():
    """microcosm#1301: the solve clock keys the solve; the gate clock only the
    calibrated gates, so a later sign-off date re-runs the gates alone."""
    default = _gate_nodes()
    same = _gate_nodes(gate_review_date="2026-09-10")
    later = _gate_nodes(gate_review_date="2026-10-02")
    # Absent, or equal to the solve clock, the calibrated parameters are
    # byte-identical to a build that never named a gate clock.
    assert default == same
    assert default.node("uk.full.gates.calibrated").params["review_date"] == (
        "2026-09-10"
    )
    assert later.node("uk.full.gates.calibrated").params == {
        **default.node("uk.full.gates.calibrated").params,
        "review_date": "2026-10-02",
    }
    changed = {node.id for node in later.nodes if node != default.node(node.id)}
    assert changed == {"uk.full.gates.calibrated"}
    for node in ("uk.full.gates.preflight", "uk.full.target_compilation"):
        assert later.node(node).params["review_date"] == "2026-09-10", node


def test_a_gate_clock_before_the_solve_clock_is_refused():
    with pytest.raises(ValueError, match="precedes the solve review date"):
        _gate_nodes(gate_review_date="2026-09-09")


@pytest.mark.parametrize("phase", ["preflight", "terminal"])
def test_each_gate_kernel_refuses_the_other_phase(phase):
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullGateKernel,
        UKFullPreflightGateKernel,
    )

    kernel = (UKFullGateKernel if phase == "preflight" else UKFullPreflightGateKernel)(
        coverage_engine=object(), engine_identity="fixture"
    )
    context = SimpleNamespace(params={"phase": phase, "engine_identity": "fixture"})
    with pytest.raises(ValueError, match=f"phase only, not '{phase}'"):
        kernel.run(context)


def test_both_gate_kernels_are_registered():
    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullGateKernel,
        UKFullHoldoutKernel,
        UKFullPreflightGateKernel,
        register_uk_full_gate_kernels,
    )
    from microcosm.graph import KernelRegistry

    registry = KernelRegistry()
    register_uk_full_gate_kernels(
        registry, coverage_engine=object(), engine_identity="fixture"
    )
    assert type(registry.get(UKFullPreflightGateKernel.ref)) is (
        UKFullPreflightGateKernel
    )
    assert type(registry.get(UKFullGateKernel.ref)) is UKFullGateKernel
    assert type(registry.get(UKFullHoldoutKernel.ref)) is UKFullHoldoutKernel


@pytest.fixture(autouse=True)
def _toy_national_rows_published(monkeypatch):
    """Publish the two-household toy frame's national rows.

    A national row's model figures leave a build only when ten distinct
    source households carry it (María's rule of 2026-10-21), and no row here
    rests on more than two. These kernel tests lower that floor to one; the
    BRMA floor stays at ten, so the withheld BRMA row keeps its own test, and
    the national withheld path has its tests in test_uk_diagnostics.py and
    test_uk_national_graph.py.
    """

    from microcosm.build.uk_runtime import diagnostics

    monkeypatch.setattr(diagnostics, "UK_DIAGNOSTIC_MIN_SOURCE_HOUSEHOLDS", 1)


def _run_final_gate_kernel(monkeypatch, *, brma_row: bool = False, solved=None):
    """Run the terminal gate kernel on a two-household fixture.

    Rebuilding the diagnostics is forbidden, so a returned document proves
    the kernel reused the decoded calibration result.

    ``solved``, a list, receives the calibration result the kernel decodes.

    ``brma_row`` adds a Scottish BRMA rent row far from its target: its two
    carriers sit under the ten-household floor, so its figures are withheld
    (microcosm#1262). Its rent differs between the two households, so the
    national count, which both carry once, does not give it back (Vahid's
    second review of #1333: a row that did would be withheld beside it).
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from microcosm.build.gate_battery import _gates_manifest_payload
    from microcosm.build.uk_runtime import full_gates, geography_ladder
    from microcosm.build.uk_runtime.graph_targets import registry_payload
    from microcosm.build.uk_runtime.graph_terminal import (
        UKFullGateKernel,
        uk_terminal_only_code_hash,
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
    from test_support.microcosm_build.uk_full_calibration_graph import (
        preflight_payload,
    )

    original = _frame()
    tables = {entity: original.table(entity).copy() for entity in original.entities}
    tables["household"]["source_household_id"] = [1, 2]
    tables["household"]["household_is_spi_synthetic"] = False
    tables["household"]["household_is_capital_gains_clone"] = False
    # microcosm#1240: the terminal kernel builds the high-value homes
    # evidence on the calibrated pool, so the fixture carries tenure and value.
    from microcosm.build.uk_runtime.tenure_constants import (
        UK_TENURE_OWNED_OUTRIGHT,
        UK_TENURE_PRIVATE_RENT,
    )

    tables["household"]["tenure_type"] = [
        UK_TENURE_OWNED_OUTRIGHT,
        UK_TENURE_PRIVATE_RENT,
    ]
    tables["household"]["main_residence_value"] = [2_500_000.0, 0.0]
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
                # A local row names its contract target (Vahid's third review
                # of #1383, item 33); this one ties to no family.
                "contract_target_id": "fixture.census_households",
            },
        ),
    ]
    if brma_row:
        specs.append(
            TargetSpec(
                name="brma_rent",
                entity="household",
                value=1000.0,
                measure="household_id",
                period=2024,
                source="fixture",
                family="private_rent",
                hierarchy=uk_fixture_hierarchy(
                    "brma_rent", level="brma", geography_id="S33999999"
                ),
                metadata={
                    "geography_level": "brma",
                    "geography_id": "S33999999",
                    "materialization": "uk_local_surface",
                    "area_type": "brma",
                    "area_code": "S33999999",
                    "metric": "rent/private_rent_brma",
                },
            )
        )
    unsolved_spec = replace(
        specs[1],
        name="dropped_local",
        metadata={
            **specs[1].metadata,
            "geography_level": "la",
            "geography_id": "E07000002",
            "area_type": "la",
            "area_code": "E07000002",
        },
        hierarchy=uk_fixture_hierarchy(
            "dropped_local", level="la", geography_id="E07000002"
        ),
    )
    targets = TargetSet(
        [
            replace(specs[0].to_target(), measure=lambda f: np.ones(2)),
            replace(specs[1].to_target(), measure=lambda f: np.array([1.0, 0.0])),
            *(
                [replace(specs[2].to_target(), measure=lambda f: np.array([1.0, 0.5]))]
                if brma_row
                else []
            ),
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
    if solved is not None:
        solved.append(result)
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
                # Below the f100 rung the problem drops cells unreachable in the
                # sample, so the selected registry is wider than the solved one;
                # the battery partitions the registry that entered the solve.
                "registry": registry_payload(
                    TargetRegistry([*specs, unsolved_spec], country="uk")
                ),
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
                "uk_ledger_compiled_registries": {"2025": empty},
                "uk_ledger_compiled_local_registries": {"2025": empty},
                "uk_ledger_compile_unsupported": {"2025": []},
                "uk_ledger_compile_local_unsupported": {"2025": []},
            }
        ),
        "holdout": artifact(
            {
                "report_only": True,
                "method": "rotated_folds",
                "target_loss_cap": 10.0,
                "loss_weight_scale": "held_local_grains_only",
                "target_weight_rule": "grain_equal",
                "population": "held_out_local_targets",
                "grains": ["constituency", "local_authority"],
                "n_folds": 2,
                "seed": 20260529,
                "solve_seed": 42,
                "mean_holdout_loss": 0.1,
                "worst_holdout_loss": 0.15,
                "fold_losses": [0.05, 0.15],
                "folds": [
                    {
                        "fold": fold,
                        "n_train_targets": 1,
                        "n_holdout_targets": 1,
                        "holdout_target_indices": [fold],
                        "training_national_rows": 1,
                        "holdout_loss": loss,
                    }
                    for fold, loss in enumerate([0.05, 0.15])
                ],
            }
        ),
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
            "terminal_code": uk_terminal_only_code_hash(),
        },
        artifacts=artifacts,
        sources={"uk_ladder": "fixture"},
    )
    return UKFullGateKernel(coverage_engine=object(), engine_identity="fixture").run(
        context
    )


def test_final_gate_kernel_owns_complete_diagnostics_and_reuses_decoded_result(
    monkeypatch,
):
    import json

    from microcosm.build.uk_runtime.graph_terminal import decode_full_gate_report

    stored = _run_final_gate_kernel(monkeypatch)
    phase, _ = decode_full_gate_report(stored.artifacts["gate_report"])
    assert phase.phase == "terminal"
    document = json.loads(stored.artifacts["calibration_diagnostics"])
    assert document["uk_diagnostics"]["rotated_holdout"]["method"] == "rotated_folds"
    assert document["uk_diagnostics"]["rotated_holdout"]["worst_holdout_loss"] == 0.15
    assert len(document["targets"]) == 2
    support_csv = pd.read_csv(
        __import__("io").BytesIO(stored.artifacts["area_support_csv"])
    )
    assert len(support_csv) == 4
    # Consumers read which areas sit below the floor (microcosm#1278).
    assert support_csv["support_below_floor"].dtype == bool


@pytest.mark.parametrize(
    "brma_row", [False, True], ids=["published_rows", "withheld_brma_row"]
)
def test_final_gate_kernel_diagnostics_match_the_published_json_schema(
    monkeypatch, brma_row
):
    # The withheld BRMA row exercises the parts #1305 added: the ``brma``
    # geography level and ``uk_diagnostics.model_figures_withheld``.
    stored = _run_final_gate_kernel(monkeypatch, brma_row=brma_row)
    document = load_strict_json(stored.artifacts["calibration_diagnostics"])

    assert published_json_schema_errors(document) == []
    if brma_row:
        assert document["uk_diagnostics"]["model_figures_withheld"]["targets"] == 1


def test_the_gate_kernel_keeps_a_withheld_brma_row_out_of_its_gates(monkeypatch):
    """The third review of #1305: the real bindings, through the kernel.

    The BRMA row misses its target by about 90% on two source households.
    Both fit gates decide on the published rows and count it (María's ruling
    of 2026-10-19); the targets table, the embedded rows and the calibration
    diagnostics leave its figures out, and nothing the kernel writes gives
    them back.
    """
    import io
    import json

    from microcosm.build.uk_runtime.diagnostics import uk_refuse_named_withheld_rows

    solved: list = []
    stored = _run_final_gate_kernel(monkeypatch, brma_row=True, solved=solved)
    key = "private_rent/S33999999/rent/private_rent_brma"
    document = json.loads(stored.artifacts["gate_report"])
    outcomes = {row["id"]: row for row in document["report"]["outcomes"]}
    fit = outcomes["uk_local_target_fit"]
    assert fit["details"]["withheld_targets"] == 1
    assert key not in fit["details"]["failing_targets"]
    assert not any(key in line for line in fit["failures"])
    family = outcomes["uk_local_per_family_fit"]["details"]
    assert family["withheld_targets"] == {"private_rent": 1}
    assert "private_rent" not in family["family_sizes"]
    (embedded,) = [
        row for row in document["target_diagnostics"] if row["family"] == "private_rent"
    ]
    assert embedded["model_figures_withheld"] is True
    assert embedded["estimate"] is None and embedded["relative_error"] is None
    uk_refuse_named_withheld_rows(
        document["report"], names=frozenset({embedded["name"]}), keys=frozenset({key})
    )
    table = pd.read_csv(io.BytesIO(stored.artifacts["target_diagnostics_csv"]))
    flags = dict(zip(table["family"], table["model_figures_withheld"], strict=True))
    assert flags == {
        "households": False,
        "census_households": False,
        "private_rent": True,
    }
    withheld = table.loc[table["family"] == "private_rent"].iloc[0]
    assert pd.isna(withheld["estimate"]) and pd.isna(withheld["relative_error"])
    diagnostics = json.loads(stored.artifacts["calibration_diagnostics"])
    (row,) = [r for r in diagnostics["targets"] if r["target_name"] == "brma_rent"]
    assert row["final_estimate"] is None and row["initial_estimate"] is None
    assert row.get("final_loss_contribution") is None
    # The totals cover the published rows (María's ruling of 2026-10-18), and
    # no loss over every row survives, in the options or anywhere else (the
    # third review of #1305).
    assert diagnostics["loss_trajectory"] == []
    assert diagnostics["uk_diagnostics"]["model_figures_withheld"]["targets"] == 1
    (result,) = solved
    receipt = dict(result.options.get("iterate_selection_receipt") or {})
    every_row = [
        float(result.final_loss),
        float(result.initial_loss),
        *(
            float(receipt[field])
            for field in ("selected_loss_float32", "closing_iterate_loss_float32")
            if isinstance(receipt.get(field), float)
        ),
    ]

    def numbers(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            yield float(value)
        elif isinstance(value, dict):
            for item in value.values():
                yield from numbers(item)
        elif isinstance(value, list):
            for item in value:
                yield from numbers(item)

    import math

    for document in (diagnostics, json.loads(stored.artifacts["gate_report"])):
        for number in numbers(document):
            assert not any(
                math.isclose(number, loss, rel_tol=1e-9) for loss in every_row
            )


def test_the_gate_kernel_fails_closed_on_a_binding_that_leaks(monkeypatch):
    """A binding that wrote a withheld row's error stops the build."""
    from microcosm.build.uk_runtime import battery_bindings

    monkeypatch.setattr(
        battery_bindings, "uk_withheld_gate_keys", lambda rows: frozenset()
    )
    with pytest.raises(ValueError, match="names a withheld row"):
        _run_final_gate_kernel(monkeypatch, brma_row=True)


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


def test_diagnostics_receive_the_holdout_in_the_schemas_shape():
    """The holdout artifact carries the kernel's binding and, when skipped, a
    reason; the diagnostics records forbid undeclared fields and state a skipped
    holdout as the bare marker. The first K=25 build (2026-10-06) reached its
    terminal gates after 28 hours and the battery refused the graph's marker."""
    from pydantic import ValidationError

    from microcosm.build.uk_runtime.graph_terminal import diagnostics_rotated_holdout
    from microcosm.diagnostics.schema import UKSkippedRotatedHoldout

    skipped = {
        "report_only": True,
        "skipped": True,
        "reason": "Not requested: opt-in diagnostic; no out-of-sample claim.",
        "graph_binding": {"original_problem_artifact": "a" * 64, "artifacts": {}},
    }
    assert diagnostics_rotated_holdout(skipped) == {"skipped": True}
    UKSkippedRotatedHoldout.model_validate({"skipped": True})
    # The schema stays strict: the graph's artifact fields are the converter's
    # to strip, and an undeclared field is refused.
    with pytest.raises(ValidationError):
        UKSkippedRotatedHoldout.model_validate(skipped)
    with pytest.raises(ValidationError):
        UKSkippedRotatedHoldout.model_validate({**skipped, "invented": 1})

    measured = {
        "report_only": True,
        "method": "rotated_folds",
        "n_folds": 5,
        "fold_losses": [0.1, 0.2, 0.3, 0.4, 0.5],
        "graph_binding": {"original_problem_artifact": "a" * 64, "artifacts": {}},
    }
    shaped = diagnostics_rotated_holdout(measured)
    assert "graph_binding" not in shaped
    assert shaped == {k: v for k, v in measured.items() if k != "graph_binding"}
    assert "graph_binding" in measured  # the artifact's report is left intact


def test_the_candidate_manifest_records_the_exclusions_each_signed_gate_stood_on():
    """microcosm#1301: the full gate report does not ship, so the candidate
    manifest records the names each signed-exclusion gate stood on, leaving
    out dormant entries as the release assembler does."""
    from microcosm.build.uk_runtime.graph_terminal import (
        _gate_rows,
        _reviewed_exclusions_stood_on,
    )
    from microcosm.data.contract import _UK_DENSE_SIGNED_EXCLUSION_GATES

    document = {
        "report": {
            "outcomes": [
                {
                    "id": "uk_target_fit",
                    "details": {
                        "reviewed_exclusions": {
                            "synthetic.row_b@2025": {"reason": "synthetic"},
                            "synthetic.row_a@2025": {"reason": "synthetic"},
                        },
                        "dormant_exclusions": [],
                    },
                },
                {
                    "id": "uk_input_mass_parity",
                    "details": {
                        "reviewed_exclusions": {
                            "synthetic_column": "synthetic reason",
                            "synthetic_dormant": "never measured",
                        },
                        "dormant_exclusions": ["synthetic_dormant"],
                    },
                },
                # A gate outside the signed set is not recorded.
                {
                    "id": "uk_weight_ratio",
                    "details": {"reviewed_exclusions": {"synthetic": "x"}},
                },
            ]
        }
    }
    record = _reviewed_exclusions_stood_on(_gate_rows(document))
    assert tuple(record) == _UK_DENSE_SIGNED_EXCLUSION_GATES
    assert record == {
        "uk_target_fit": ["synthetic.row_a@2025", "synthetic.row_b@2025"],
        # The report lacks the gate: it stood on none.
        "uk_local_target_fit": [],
        "uk_input_mass_parity": ["synthetic_column"],
    }


def test_the_gate_kernel_keys_the_solve_without_the_high_value_homes_code(
    monkeypatch,
):
    # The dense solve keys on the source preflight, which runs this kernel, so
    # the report-only high-value homes code keys the calibrated node through
    # its parameters and stays out of the kernel's hash (microcosm#1306
    # review).
    from microcosm.build.uk_runtime import (
        graph_terminal,
        high_value_home_gates,
        high_value_homes,
    )

    hashed: list[set[str]] = []
    real = graph_terminal.source_hash

    def recording(*objects, **kwargs):
        hashed.append({obj.__name__ for obj in objects})
        return real(*objects, **kwargs)

    monkeypatch.setattr(graph_terminal, "source_hash", recording)
    high_value = {high_value_home_gates.__name__, high_value_homes.__name__}
    # The preflight kernel inherits the calibrated kernel's code hash.
    for kernel in (
        graph_terminal.UKFullGateKernel,
        graph_terminal.UKFullPreflightGateKernel,
    ):
        kernel(
            coverage_engine=object(), engine_identity="fixture"
        ).implementation_hash()
        assert not hashed[-1] & high_value, kernel.__name__
    # The hash is memoised per process; its undecorated body hashes the two
    # high-value homes modules.
    graph_terminal.uk_terminal_only_code_hash.__wrapped__()
    assert hashed[-1] == high_value
    first = graph_terminal.uk_terminal_only_code_hash()
    calls = len(hashed)
    assert graph_terminal.uk_terminal_only_code_hash() == first
    assert len(hashed) == calls


def test_the_calibrated_gates_refuse_a_stale_terminal_code() -> None:
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.graph_terminal import UKFullGateKernel

    kernel = UKFullGateKernel(coverage_engine=object(), engine_identity="fixture")
    context = SimpleNamespace(
        params={
            "engine_identity": "fixture",
            "phase": "terminal",
            "terminal_code": "0" * 64,
        },
        artifacts={},
    )
    with pytest.raises(ValueError, match="terminal-only code"):
        kernel.run(context)
