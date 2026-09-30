"""Actual full graph on synthetic adapters: the engine-backed target paths."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_full_target_graph import *


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
    assert "uk.full.locations" not in manifest.nodes
    assert {
        "uk.full.identity",
        "uk.full.geography.assign",
        "uk.full.geography_gate",
    } <= set(manifest.nodes)
    assert manifest.nodes["uk.full.geography_gate"].receipt["outcome"] == "pass"
    assert (
        manifest.nodes["uk.full.pool"].receipt["atomic_validation"]["outcome"] == "pass"
    )
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


def test_default_all_has_direct_matrix_and_solver_parity_and_replays(
    target_inputs, toy_ladder, tmp_path
):
    from microcosm.build.uk_runtime.full_problem import build_uk_full_local_problem
    from microcosm.build.uk_runtime.local_rowwise import (
        prepare_uk_full_solve,
        solve_uk_dense_reference,
    )
    from microcosm.build.uk_runtime.rowwise_dataset import (
        clone_uk_dataset_with_ladder_geography,
    )
    from test_support.microcosm_build.uk_full_population_graph import source_frame

    ladder, path = toy_ladder
    # The pre-graph numerical helpers below draw geography with the legacy
    # sequential ladder; the graph is built the same way for this parity proof.
    default, default_run, default_problem = build(
        tmp_path / "default", path, None, n_clones=10, geography_assignment="legacy"
    )
    explicit, explicit_run, explicit_problem = build(
        tmp_path / "explicit",
        path,
        ("country", "region", "constituency", "la"),
        n_clones=10,
        geography_assignment="legacy",
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
        geography_assignment="legacy",
    )
    assert all(receipt.hit for receipt in replay.nodes.values())
    np.testing.assert_array_equal(
        replay.population(default.population).weights_for("household").values,
        direct.weights,
    )


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


def test_target_kernel_identity_binds_country_reference_resources(monkeypatch):
    kernel = graph_targets.UKFullProblemKernel()
    original = kernel.implementation_hash()
    monkeypatch.setattr(
        graph_targets,
        "load_country_spec",
        lambda _country: SimpleNamespace(fingerprint="changed-reference-resource"),
    )
    assert kernel.implementation_hash() != original
