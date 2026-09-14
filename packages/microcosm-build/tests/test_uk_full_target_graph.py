"""Actual full graph on synthetic target and engine-source adapters."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_uk_full_calibration_graph import preflight_payload
from test_uk_full_population_graph import Source, graph_and_registry
from test_uk_ladder_rowwise_clone import toy_ladder as toy_ladder

from microcosm.build.uk_runtime import full_targets, graph_targets, ledger_targets
from microcosm.build.uk_runtime.graph_build import (
    UKFullBuildConfig,
    register_uk_full_kernels,
    uk_full_graph,
)
from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
from microcosm.build.uk_runtime.graph_terminal import FULL_GATE_REPORT_TYPE
from microcosm.build.uk_runtime.local_rowwise import UKRowwiseNationalRows
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.artifacts import decode_problem
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    compile_graph,
    run_graph,
)


@pytest.fixture
def target_inputs(monkeypatch, toy_ladder):
    national = TargetRegistry(
        [
            TargetSpec(
                name="country_households",
                entity="household",
                measure="test_ones",
                value=34.0,
                period=2026,
                family="fixture",
                source="fixture",
                metadata={
                    "geography_level": "country",
                    "geography_id": "UK",
                    "contract_target_id": "obr.vat",
                },
            ),
            TargetSpec(
                # Repeated names across periods must retain distinct metadata.
                name="country_households",
                entity="household",
                measure="test_ones",
                value=4.0,
                period=2025,
                family="fixture",
                source="fixture",
                filter="test_london",
                metadata={
                    "geography_level": "region",
                    "geography_id": "LONDON",
                    "contract_target_id": "voa.council_tax_stock.band_a",
                },
            ),
        ],
        country="uk",
    )
    ladder, _ = toy_ladder
    local = TargetRegistry(
        [
            TargetSpec(
                name=f"ons.census.households@{level}:{code}",
                entity="household",
                measure="household_count",
                value=35.0 if level == "local_authority" and i == 0 else 40.0,
                period=2026,
                family="census_households",
                source="chronicle_fixture",
                metadata={
                    "contract_target_id": "ons.census.households",
                    "geography_level": level,
                    "geography_id": str(code),
                    "uprating_from_period": 2022 if str(code).startswith("S") else 2021,
                    "uprating_to_period": 2026,
                },
            )
            for level, codes in (
                ("constituency", ladder.constituency_code),
                ("local_authority", ladder.local_authority_code),
            )
            for i, code in enumerate(codes)
        ],
        country="uk",
    )
    inputs = {
        "national_registry": national,
        "band_edge_registry": national,
        "local_registry": local,
        "artifact": SimpleNamespace(facts=()),
        "calibration_year": 2026,
        "measure_exclusions": {},
        "reviewed_unbound_higher_targets": {},
        "national_source_pin": {"fixture": True},
        "local_source_pin": {"fixture": True},
        "register_completeness": {"fixture": True},
        "ledger_provenance": {"fixture": True},
        "uk_ledger_compiled_registries": {2026: national},
        "uk_ledger_compiled_local_registries": {2026: local},
    }
    monkeypatch.setattr(
        full_targets, "load_uk_full_target_inputs", lambda *args, **kwargs: inputs
    )
    reference = {"value": 33.0, "period": 2026}
    monkeypatch.setattr(
        graph_targets, "uk_ledger_households_total", lambda *args, **kwargs: reference
    )

    def surface():
        return ledger_targets.uk_local_target_surface(
            graph_targets.full_problem._joint_surface_registry(local, national),
            bound_national_target_ids=graph_targets.full_problem._national_contract_target_ids(
                national
            ),
            period=2026,
            census_household_uprating=ledger_targets.uk_census_household_uprating(
                local, reference, period=2026
            ),
        )

    def measures(frame, national_registry, *, local_grains, **kwargs):
        tables = {e: frame.table(e).copy() for e in frame.entities}
        tables["household"]["test_ones"] = 1.0
        tables["household"]["test_london"] = (
            tables["household"]["region"] == "LONDON"
        ).astype(float)
        prepared = Frame(
            tables,
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
        return (
            prepared,
            lambda _: frame,
            UKRowwiseNationalRows(
                national_registry.to_target_set(), national_registry, ("fixture",)
            ),
            {
                g: pd.DataFrame(
                    {"households": np.ones(frame.n("household"))},
                    index=frame.table("household")["household_id"],
                )
                for g in local_grains
            },
            {"fixture": True},
        )

    monkeypatch.setattr(graph_targets, "resolve_uk_full_measures", measures)
    return {
        "national": national,
        "local": local,
        "inputs": inputs,
        "surface": surface,
        "measures": measures,
    }


class Preflight(KernelBase):
    """Synthetic source verdict: tests below exercise numerical graph ownership."""

    ref = "uk.test.target-preflight@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        selection = json.loads(context.artifacts["selection"].payload)["receipt"]
        return KernelResult(
            artifacts={"preflight": preflight_payload(selection=selection)}
        )


def build(
    tmp_path,
    ladder_path,
    levels,
    *,
    n_clones=1,
    seed=7,
    dataset_households=None,
    resume="auto",
    forbid_execution=False,
):
    primitive, _ = graph_and_registry(1)
    base = Graph(
        "uk",
        tuple(s for s in primitive.sources if s.name == "fixture"),
        (primitive.node("source"),),
    )
    config = UKFullBuildConfig(
        calibration_year=2026,
        time_period="2023",
        source_year=2023,
        n_clones=n_clones,
        geography_levels=levels,
        seed=seed,
        calibration=UKGraphCalibrationConfig(
            epochs=8, seed=seed, dataset_households=dataset_households
        ),
    )
    full = uk_full_graph(config, spine=base, spine_population="source")
    preflight = Node(
        "fixture.preflight",
        Preflight.ref,
        population="uk.full.pool",
        artifact_inputs=(
            ArtifactInput(
                "selection",
                "uk.full.target_selection",
                "selection",
                graph_targets.TARGET_SELECTION_TYPE,
            ),
        ),
        artifact_outputs=(ArtifactOutput("preflight", FULL_GATE_REPORT_TYPE),),
    )
    full = replace(
        full,
        graph=replace(
            full.graph,
            nodes=(
                *(
                    replace(
                        node,
                        artifact_inputs=(
                            *node.artifact_inputs,
                            ArtifactInput(
                                "preflight",
                                preflight.id,
                                "preflight",
                                FULL_GATE_REPORT_TYPE,
                            ),
                        ),
                    )
                    if node.id == full.calibration.dense_producer
                    else node
                    for node in full.graph.nodes
                ),
                preflight,
            ),
        ),
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(Preflight())
    register_uk_full_kernels(registry)
    if forbid_execution:
        for kernel in registry.as_mapping().values():
            kernel.run = lambda *args, **kwargs: pytest.fail(
                "cached full graph executed"
            )
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compile_graph(full.graph),
        sources={
            "fixture": ladder_path,
            "uk_ladder": ladder_path,
            "uk_ledger_facts": ladder_path,
        },
        store=store,
        kernels=registry,
        resume=resume,
    )
    problem = decode_problem(
        store.load_bytes(manifest.nodes["uk.full.problem"].opaque_artifacts["problem"])
    )
    return full, manifest, problem


@pytest.mark.requires_uk
def test_explicit_country_filter_runs_same_full_graph_without_local_constraints(
    target_inputs, toy_ladder, tmp_path
):
    _, path = toy_ladder
    full, manifest, problem = build(tmp_path, path, ("country",))
    assert problem.problem.names == ("country_households@2026",)
    assert {m["geography_level"] for m in problem.target_metadata} == {"country"}
    assert manifest.population(full.population).n("household") == 4
    assert len(problem.bindings["target_selection"]["excluded"]) > 0
    assert "uk.full.dense" in manifest.nodes
    assert "uk.full.locations" in manifest.nodes
    surface = json.loads(
        ContentStore(tmp_path / "store").load_bytes(
            manifest.nodes["uk.full.target_compilation"].opaque_artifacts["surface"]
        )
    )
    assert len(surface["surface"]) == 10
    assert len(surface["household_dispersion"]["cells"]) == 10
    assert surface["census_household_uprating"]["household_cells"]["cells"] == 10
    assert surface["source_validation"]["targets"] == {
        "chronicle": {"fixture": True},
        "paired_ladder_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_default_scope_contains_all_levels_and_never_depends_on_k_or_k_small():
    default = UKFullBuildConfig(calibration_year=2026)
    for config in (
        default,
        replace(default, n_clones=1),
        replace(
            default, calibration=replace(default.calibration, dataset_households=20)
        ),
    ):
        graph = uk_full_graph(config).graph
        assert graph.node("uk.full.target_selection").params["geography_levels"] is None


@pytest.mark.requires_uk
def test_default_all_has_direct_matrix_and_solver_parity_and_replays(
    target_inputs, toy_ladder, tmp_path
):
    from test_uk_full_population_graph import source_frame

    from microcosm.build.uk_runtime.full_problem import build_uk_full_local_problem
    from microcosm.build.uk_runtime.local_rowwise import (
        prepare_uk_full_solve,
        solve_uk_dense_reference,
    )
    from microcosm.build.uk_runtime.rowwise_dataset import (
        clone_uk_dataset_with_ladder_geography,
    )

    ladder, path = toy_ladder
    default, default_run, default_problem = build(
        tmp_path / "default", path, None, n_clones=10
    )
    explicit, explicit_run, explicit_problem = build(
        tmp_path / "explicit",
        path,
        ("country", "region", "constituency", "la"),
        n_clones=10,
    )
    assert {row["geography_level"] for row in default_problem.target_metadata} == {
        "country",
        "region",
        "constituency",
        "la",
    }
    assert default_problem.problem.n_targets == 12
    by_grain = {}
    for value, metadata in zip(
        default_problem.problem.target_vector,
        default_problem.target_metadata,
        strict=True,
    ):
        if metadata["materialization"] == "uk_local_surface":
            assert metadata["contract_target_id"] == "ons.census.households"
            by_grain.setdefault(metadata["geography_level"], []).append(value)
    assert sum(by_grain["constituency"]) == pytest.approx(33.0)
    assert sum(by_grain["la"]) == pytest.approx(33.0)
    assert len(set(by_grain["constituency"])) == 1
    assert len(set(by_grain["la"])) > 1
    census_receipt = default_problem.bindings["cross_geography"][
        "census_household_uprating"
    ]
    assert census_receipt["grains"]["constituency"]["factor"] == 33.0 / 200.0
    assert census_receipt["grains"]["local_authority"]["factor"] == 33.0 / 195.0
    assert default_problem.problem.names == explicit_problem.problem.names
    np.testing.assert_array_equal(
        default_problem.problem.matrix.toarray(),
        explicit_problem.problem.matrix.toarray(),
    )
    np.testing.assert_array_equal(
        default_problem.problem.target_vector, explicit_problem.problem.target_vector
    )
    np.testing.assert_array_equal(
        default_run.population(default.population).weights_for("household").values,
        explicit_run.population(explicit.population).weights_for("household").values,
    )
    assert default_problem.bindings["target_selection"]["selector"]["explicit"] is False
    assert explicit_problem.bindings["target_selection"]["selector"]["explicit"] is True

    # Independently execute the maintained pre-graph numerical helpers on the
    # same original spine, legacy location draw, target rows and solver options.
    assignment = clone_uk_dataset_with_ladder_geography(
        source_frame(),
        ladder,
        n_clones=10,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
    )
    prepared_frame, _, national_rows, metrics, _ = target_inputs["measures"](
        assignment.frame, target_inputs["national"], local_grains=("constituency", "la")
    )
    surface, cross = target_inputs["surface"]()
    _, local, _, families, _ = build_uk_full_local_problem(
        SimpleNamespace(result=SimpleNamespace(frame=prepared_frame), ladder=ladder),
        local_registry=target_inputs["local"],
        national_registry=target_inputs["national"],
        local_metrics=metrics,
        period=2026,
        sample_fraction=1.0,
        reviewed_unbound_higher_targets={},
        selected_surface=surface,
        surface_receipt=cross,
    )
    prepared = prepare_uk_full_solve(
        prepared_frame,
        local,
        bound_families=families,
        national_rows=national_rows,
        target_weight_rule="uniform",
    )
    direct = solve_uk_dense_reference(prepared, epochs=8, seed=7)
    np.testing.assert_array_equal(
        default_problem.problem.matrix.toarray(), direct.problem.matrix.toarray()
    )
    np.testing.assert_array_equal(
        default_run.population(default.population).weights_for("household").values,
        direct.weights,
    )

    _, replay, _ = build(
        tmp_path / "default",
        path,
        None,
        n_clones=10,
        resume="require",
        forbid_execution=True,
    )
    assert all(receipt.hit for receipt in replay.nodes.values())
    np.testing.assert_array_equal(
        replay.population(default.population).weights_for("household").values,
        direct.weights,
    )


@pytest.mark.requires_uk
def test_unsupported_default_all_refuses_without_narrowing(
    target_inputs, toy_ladder, tmp_path
):
    from microcosm.graph.errors import NodeRejectedError

    _, path = toy_ladder
    # At K=1 London's only household cannot occupy both positive area cells.
    with pytest.raises(NodeRejectedError, match="[Ss]upport|unassigned|positive"):
        build(tmp_path / "all", path, None, n_clones=1)
    country, result, problem = build(
        tmp_path / "country", path, ("country",), n_clones=1
    )
    assert problem.problem.names == ("country_households@2026",)
    assert result.population(country.population).n("household") == 4


def test_local_surface_selection_keeps_exact_target_periods():
    rows = pd.DataFrame(
        {
            "target_name": ["same", "same", "different"],
            "period": [2025, 2026, 2026],
            "value": [2.0, 3.0, 4.0],
        }
    )
    selected = [
        TargetSpec(
            name="same",
            entity="household",
            measure="count",
            value=3.0,
            period=2026,
            source="fixture",
            family="fixture",
        )
    ]
    actual = graph_targets._selected_local_surface(rows, selected)
    assert actual.to_dict(orient="records") == [
        {"target_name": "same", "period": 2026, "value": 3.0}
    ]


@pytest.mark.requires_uk
@pytest.mark.parametrize("levels", [None, ("country",)])
def test_source_census_validation_precedes_any_geography_filter(
    target_inputs, toy_ladder, tmp_path, levels
):
    from microcosm.graph.errors import NodeRejectedError

    local = target_inputs["local"]
    # A malformed NI mapping must fail even when no local constraint is selected.
    target_inputs["inputs"]["local_registry"] = TargetRegistry(
        [
            replace(spec, value=2.0)
            if spec.metadata["geography_id"] == "N05000001"
            else spec
            for spec in local
        ],
        country="uk",
    )
    with pytest.raises(NodeRejectedError, match="dispersion exceeds"):
        build(tmp_path, toy_ladder[1], levels)


def test_local_problem_targets_come_from_chronicle_with_assignment_rosters(
    target_inputs, toy_ladder
):
    from test_uk_full_population_graph import source_frame

    from microcosm.build.uk_runtime.full_problem import build_uk_full_local_problem
    from microcosm.build.uk_runtime.rowwise_dataset import (
        clone_uk_dataset_with_ladder_geography,
    )

    ladder, _ = toy_ladder
    clone = clone_uk_dataset_with_ladder_geography(
        source_frame(),
        ladder,
        n_clones=10,
        seed=7,
        source_year=2023,
        expected_constituency_vintage="2024_pcon",
    )
    ids = clone.frame.table("household")["household_id"]
    metrics = {
        grain: pd.DataFrame({"households": np.ones(len(ids))}, index=ids)
        for grain in ("constituency", "la")
    }
    local = target_inputs["local"]
    uprating = ledger_targets.uk_census_household_uprating(
        local, {"value": 33.0, "period": 2026}, period=2026
    )
    _, problem, receipt, _, _ = build_uk_full_local_problem(
        SimpleNamespace(result=clone, ladder=ladder),
        local_registry=local,
        national_registry=TargetRegistry([], country="uk"),
        local_metrics=metrics,
        period=2026,
        sample_fraction=1.0,
        reviewed_unbound_higher_targets={},
        census_household_uprating=uprating,
    )
    expected = target_inputs["surface"]()[0].set_index("target_name")["value"]
    actual = problem.target_frame.set_index("target_name")["value"]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index())
    assert set(actual.index) == {spec.name for spec in local}
    assert receipt["census_household_uprating"]["household_cells"]["cells"] == 10


def test_target_kernel_identity_includes_reconciliation_and_ladder_diagnostics(
    monkeypatch,
):
    hashed = []
    monkeypatch.setattr(
        graph_targets, "source_hash", lambda *objects: hashed.extend(objects) or "test"
    )
    graph_targets.UKFullTargetCompilationKernel().implementation_hash()
    assert graph_targets.cross_grain in hashed
    assert graph_targets.ladder_targets in hashed


@pytest.mark.requires_uk
def test_target_kernel_identity_binds_country_reference_resources(monkeypatch):
    kernel = graph_targets.UKFullProblemKernel()
    original = kernel.implementation_hash()
    monkeypatch.setattr(
        graph_targets,
        "load_country_spec",
        lambda _country: SimpleNamespace(fingerprint="changed-reference-resource"),
    )
    assert kernel.implementation_hash() != original
