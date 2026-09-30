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


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    """Run the whole spine graph on the hermetic H2 fixture once, with the
    real engine, for every test in this module that reads its final frame."""

    from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
    from microcosm.graph import ContentStore, run_graph

    root = _TEST_PATHS.repository
    fixture = root / "packages/microcosm-graph/tests/fixtures/parity/uk_spine"
    if not fixture.exists():
        pytest.skip("UK spine parity fixture is not present")
    _, implementations = fixture_stage_plan_inputs(fixture / "sources")
    graph = uk_spine_graph()
    compiled = compile_graph(graph)
    store = ContentStore(tmp_path_factory.mktemp("uk_graph") / "store")
    manifest = run_graph(
        compiled,
        sources={"frs": fixture / "sources"},
        store=store,
        kernels=uk_registry(dict(implementations)),
        resume="forbid",
        decisions=(),
    )
    final = manifest.population(compiled.versions[compiled.order[-1]])
    return manifest, store, final


def test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture(
    fixture_run,
) -> None:
    """The driver's record projection must cover every declared output.

    ``frs_spine`` declares the entity ids and memberships among its outputs,
    but the executor carries those outside owned cells, so the root node
    exposes no artifact for them.  The first full licensed run through the
    graph completed every stage and then died here, on ``person_id``; this
    test runs the projection on the hermetic H2 fixture so the class fails
    in CI's engine lane instead.
    """

    # The driver lives in the package; tools/build_uk_frs_spine.py is a shim.
    from microcosm.build.uk_runtime import spine_build as driver

    manifest, store, final = fixture_run
    country = load_country_spec("uk")
    stages = list(country.sources.stages)

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


@pytest.mark.parametrize("period", ["2024", "2026"])
def test_take_up_population_reads_the_same_on_the_final_spine_frame(
    fixture_run, period
) -> None:
    """The take-up gate's engine read works on a real multi-channel frame.

    The final spine frame carries the SPI support copies (zero-weight, with
    by-design NaN auxiliaries the engine refuses), the CGT clones and every
    stage's columns. The population read from its narrow projection equals
    the engine's is_WA_adult on the whole frame (NaN floats filled, as
    tools/verify_uk_identity_stability.py does), at the fixture's period and
    at 2026-27, when State Pension age splits the 66-year-olds and the answer
    depends on every person's id and weight. The gate then measures over it.
    """

    from microcosm.build.uk_runtime.frs_take_up import (
        uc_age_eligible_benunits,
        uk_take_up_population,
        uk_take_up_signal_gate,
    )
    from microcosm.build.uk_runtime.national_frame import (
        uk_household_weight_kind,
        uk_national_frame,
    )
    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    _, _, final = fixture_run
    tables = {}
    for entity in ("person", "benunit", "household"):
        table = final.table(entity).copy()
        for column in table.columns:
            if table[column].dtype.kind == "f" and table[column].isna().any():
                table[column] = table[column].fillna(0.0)
        tables[entity] = table
    whole = uk_national_frame(
        **tables,
        time_period=period,
        weight_kind=uk_household_weight_kind(final),
        household_weights=final.weights_for("household").values,
    )
    frame = uk_national_frame(
        person=final.table("person").copy(),
        benunit=final.table("benunit").copy(),
        household=final.table("household").copy(),
        time_period=period,
        weight_kind=uk_household_weight_kind(final),
        household_weights=final.weights_for("household").values,
    )
    engine = PolicyEngineUKEngine()

    population = uk_take_up_population(frame, engine)
    direct = engine.materialize(whole, ("is_WA_adult",), period)["is_WA_adult"]

    assert population.working_age_adult.tolist() == np.asarray(direct).tolist()
    result = uk_take_up_signal_gate(frame)
    units = uc_age_eligible_benunits(
        frame.table("person"), frame.table("benunit"), population
    )
    detail = result.details["benunit.would_claim_uc"]
    assert detail["population_units"] == int(units.sum())
    assert result.details["universal_credit_population"]["period"] == period
