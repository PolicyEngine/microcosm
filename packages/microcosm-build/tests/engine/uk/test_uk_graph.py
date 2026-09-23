"""Tests split from packages/microcosm-build/tests/test_uk_graph.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_graph import *
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def test_spi_support_fixture_changes_person_mass_and_conserves_household_mass() -> None:
    """Keep H2 sensitive to support-channel composition changes."""

    from microcosm.build.uk_runtime.frs_spine import uk_frs_spine_seed_frame
    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs

    root = _TEST_PATHS.repository
    fixture = root / "packages/microcosm-graph/tests/fixtures/parity/uk_spine"
    _, implementations = fixture_stage_plan_inputs(fixture / "sources")
    before = implementations["frs_spine"](uk_frs_spine_seed_frame())
    after = implementations["spi_support_channel"](before)

    before_person_mass = before.stratum_mass()
    after_person_mass = after.stratum_mass()
    assert before_person_mass.index.equals(after_person_mass.index)
    assert not np.isclose(
        float(after_person_mass.sum()),
        float(before_person_mass.sum()),
        rtol=1e-9,
        atol=0.0,
    )

    before_household_mass = before.weights_for("household").total
    assert after.weights_for("household").total == before_household_mass
    assert after.mass_log[-1].old_total == before_household_mass
    assert after.mass_log[-1].new_total == before_household_mass


def test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture(
    tmp_path,
) -> None:
    """The driver's record projection must cover every declared output.

    ``frs_spine`` declares the entity ids and memberships among its outputs,
    but the executor carries those outside owned cells, so the root node
    exposes no artifact for them.  The first full licensed run through the
    graph completed every stage and then died here, on ``person_id``; this
    test runs the projection on the hermetic H2 fixture so the class fails
    in CI's engine lane instead.
    """

    import importlib.util

    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
    from microcosm.graph import ContentStore, run_graph

    root = _TEST_PATHS.repository
    fixture = root / "packages/microcosm-graph/tests/fixtures/parity/uk_spine"
    if not fixture.exists():
        pytest.skip("UK spine parity fixture is not present")
    spec = importlib.util.spec_from_file_location(
        "build_uk_frs_spine", root / "tools" / "build_uk_frs_spine.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    country = load_country_spec("uk")
    stages = [
        stage
        for stage in country.sources.stages
        if stage.stage not in UK_SPINE_EXCLUSIONS
    ]
    _, implementations = fixture_stage_plan_inputs(fixture / "sources")
    graph = uk_spine_graph()
    compiled = compile_graph(graph)
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compiled,
        sources={"frs": fixture / "sources"},
        store=store,
        kernels=uk_registry(dict(implementations)),
        resume="forbid",
        decisions=(),
    )
    final = manifest.population(compiled.versions[compiled.order[-1]])

    records = driver._graph_stage_records(
        manifest=manifest, store=store, stages=stages, frame=final
    )

    assert [record.stage for record in records] == [stage.stage for stage in stages]
    by_stage = {record.stage: record for record in records}
    for stage in stages:
        assert set(by_stage[stage.stage].nonzero_share) == set(stage.outputs), (
            stage.stage
        )
    root_shares = by_stage["frs_spine"].nonzero_share
    for column in (
        "person_id",
        "person_benunit_id",
        "person_household_id",
        "benunit_id",
        "household_id",
    ):
        assert root_shares[column] == 1.0
