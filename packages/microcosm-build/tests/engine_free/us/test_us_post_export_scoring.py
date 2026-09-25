"""Tests split from packages/microcosm-build/tests/test_us_post_export_scoring.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_post_export_scoring import *


def test_batched_scoring_matches_the_unbatched_path(
    builder, fake_reforms, tmp_path
) -> None:
    """(a) The smoke's GateResult, every reform_validation row and the
    demographics payload are identical for batch sizes 1, 2 and unbatched, and
    identical to one whole-frame engine per simulate() call."""
    frame = _nested_frame()
    reference = _run_consumers(
        builder,
        lambda name, consumer: _unbatched_simulate(frame, _EngineLog()),
    )
    reference_gate, reference_payload, reference_demographics = reference
    assert reference_payload["out_of_sample_simulated"] is True
    assert reference_gate.details["probes"] == len(us_release_reform_coverage_probes())

    for batch_size in (1, 2, None):
        (gate, payload, demographics), _, scorer, _ = _batched_run(
            builder, frame, batch_size, tmp_path / f"b{batch_size}"
        )
        assert scorer.n_batches == (1 if batch_size is None else -(-7 // batch_size))
        assert gate.passed == reference_gate.passed
        assert gate.failures == reference_gate.failures
        assert gate.details == reference_gate.details
        assert payload == reference_payload
        assert demographics == reference_demographics

def test_batched_engines_partition_release_and_order_their_work(
    builder, fake_reforms, tmp_path
) -> None:
    """(b) no construction sees more than the batch size and every pass
    partitions the pool; (c) one engine alive at a time; (d) the SPM selection
    on every construction; (e) one reform system per reform, 41 for the
    shipped smoke; (f) one baseline pass per plan; (g) ascending periods."""
    frame = _nested_frame()
    household_ids = frame.table("household")["household_id"].tolist()
    batch_size = 3
    (gate, payload, _), log, scorer, consumers = _batched_run(
        builder, frame, batch_size, tmp_path
    )
    n_batches = scorer.n_batches
    assert n_batches == 3

    # (b) No construction saw more than the batch size, and the constructions
    # come in passes of n_batches, each covering every household exactly once
    # and in the written order, with one reform (or none) per pass.
    constructions = log.constructions
    assert len(constructions) % n_batches == 0
    for construction in constructions:
        assert construction.dataset is None  # released (see (c))
        assert 0 < len(construction.household_ids) <= batch_size
    for start in range(0, len(constructions), n_batches):
        chunk = constructions[start : start + n_batches]
        assert [id_ for engine in chunk for id_ in engine.household_ids] == (
            household_ids
        )
        assert len({id(engine.tax_benefit_system) for engine in chunk}) == 1

    # (c) At every construction, every earlier engine had been released.
    assert log.max_unreleased_at_construction == 0

    # (d) Every construction declares the release SPM selection.
    assert all(engine.spm == US_RELEASE_SPM_SELECTION for engine in constructions)

    # (e) One reform system per reform, built once and shared by the batches.
    # Each batch engine gets that system alone: ``reform=`` too would make
    # policyengine-us clone the system on every construction.
    smoke = consumers["reform_coverage_smoke"].record()
    validation = consumers["reform_validation"].record()
    demographics = consumers["demographics"].record()
    probes = len(us_release_reform_coverage_probes())
    assert smoke["reform_systems"] == smoke["reform_passes"] == probes == 41
    assert validation["reform_systems"] == validation["reform_passes"] > 0
    assert demographics["reform_systems"] == demographics["reform_passes"] == 0
    assert all(engine.reform is None for engine in constructions)
    reform_constructions = [
        engine for engine in constructions if engine.tax_benefit_system is not None
    ]
    assert len(log.systems) == smoke["reform_systems"] + validation["reform_systems"]
    assert {id(engine.tax_benefit_system) for engine in reform_constructions} == {
        id(system) for system in log.systems
    }
    assert all(system.reform is not None for system in log.systems)
    assert len(reform_constructions) == len(log.systems) * n_batches

    # (f) One baseline pass per consumer plan.
    baseline_constructions = [
        engine for engine in constructions if engine.tax_benefit_system is None
    ]
    assert len(baseline_constructions) == 3 * n_batches
    for record in (smoke, validation, demographics):
        assert record["baseline_passes"] == 1
        assert record["n_batches"] == n_batches
        assert record["max_batch_households"] == batch_size

    # (g) Each engine computed its keys in non-decreasing period order, and
    # the validation baseline spans several periods.
    assert all(engine.periods == sorted(engine.periods) for engine in constructions)
    validation_periods = validation["baseline_plan"]["period_order"]
    assert validation_periods == sorted(validation_periods)
    assert len(validation_periods) > 1
    assert payload["out_of_sample_simulated"] is True
    assert gate.details["probes"] == probes

def test_engine_guard_refuses_an_earlier_period(builder) -> None:
    """(g) The guard refuses a key earlier than one the engine computed."""
    calls = []
    engine = builder._AscendingPeriodEngine(
        SimpleNamespace(
            calculate=lambda variable, period, **kwargs: calls.append(
                (variable, period, kwargs)
            )
        ),
        label="fixture",
    )
    engine.calculate(("income_tax", 2024, None))
    engine.calculate(("state_income_tax", 2024, "household"))
    engine.calculate(("ga_income_tax", 2026, None))
    with pytest.raises(RuntimeError, match="nd_income_tax@2025 after it computed"):
        engine.calculate(("nd_income_tax", 2025, None))
    assert calls == [
        ("income_tax", 2024, {}),
        ("state_income_tax", 2024, {"map_to": "household"}),
        ("ga_income_tax", 2026, {}),
    ]

def test_ascending_plan_is_stable_and_consumers_refuse_other_orders(
    builder, tmp_path
) -> None:
    keys = [
        ("income_tax", 2024, None),
        ("ga_income_tax", 2026, None),
        ("nd_income_tax", 2025, None),
        ("state_fips", 2024, "person"),
        ("income_tax", 2024, None),
    ]
    assert builder._ascending_period_plan(keys) == (
        ("income_tax", 2024, None),
        ("state_fips", 2024, "person"),
        ("nd_income_tax", 2025, None),
        ("ga_income_tax", 2026, None),
    )
    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    with pytest.raises(ValueError, match="ascending period order"):
        scorer.open_consumer("fixture", tuple(dict.fromkeys(keys)))

def test_served_baseline_refuses_a_key_outside_its_plan(builder, tmp_path) -> None:
    log = _EngineLog()
    scorer = _scorer(builder, _nested_frame(), log, 3, tmp_path)
    consumer = scorer.open_consumer("fixture", (("income_tax", 2024, None),))
    baseline = consumer.simulate(None)
    assert baseline.calculate("income_tax", 2024).sum() > 0
    with pytest.raises(RuntimeError, match="does not hold"):
        baseline.calculate("income_tax", 2026)
    # The served baseline answered from the plan: one pass, no new engines.
    assert len(log.constructions) == scorer.n_batches

def test_a_failing_key_still_releases_its_batch_engine(builder, tmp_path) -> None:
    """(c) The engine that raised is released before the error propagates, so
    a failed pass never leaves a batch engine's arrays alive."""
    log = _EngineLog()
    log.fail_on = "state_income_tax"
    scorer = _scorer(builder, _nested_frame(), log, 3, tmp_path)
    with pytest.raises(RuntimeError, match="fixture engine refuses"):
        scorer.open_consumer(
            "fixture",
            (("income_tax", 2024, None), ("state_income_tax", 2024, None)),
        )
    assert len(log.constructions) == 1
    assert log.constructions[0].dataset is None

def test_post_export_values_sum_matches_microseries_sum(builder) -> None:
    """``.sum()`` equals policyengine-core's ``MicroSeries.sum`` (NaN skipped)
    and the arrays are read-only."""
    microdf = pytest.importorskip("microdf")
    values = np.asarray([1.5, np.nan, 2.0, 4.0])
    weights = np.asarray([10.0, 20.0, np.nan, 0.5])
    result = builder._PostExportValues(values, weights)
    assert result.sum() == float(microdf.MicroSeries(values, weights=weights).sum())
    flags = np.asarray([True, False, True, True])
    assert builder._PostExportValues(flags, weights).sum() == float(
        microdf.MicroSeries(flags, weights=weights).sum()
    )
    with pytest.raises(ValueError):
        np.asarray(result)[0] = 9.0
    with pytest.raises(ValueError, match="one weight per value"):
        builder._PostExportValues(values, weights[:2])

def test_a_multi_batch_pass_refuses_a_population_aggregate(builder, tmp_path) -> None:
    """A measure that reaches a formula aggregating over its whole simulation
    (here ``medicaid_cost`` -> ``medicaid_slcsp_state_denominator``) would sum
    batch-local aggregates, so every engine of a multi-batch pass refuses it,
    after scoring and before its release. One batch is the whole file."""
    plan = (("medicaid_cost", 2024, None),)
    log = _EngineLog()
    log.computes["medicaid_cost"] = ("medicaid_slcsp_state_denominator",)
    scorer = _scorer(builder, _nested_frame(), log, 3, tmp_path / "batched")
    with pytest.raises(
        RuntimeError,
        match=(
            r"not batch-invariant: a batch engine computed "
            r"medicaid_slcsp_state_denominator@2024 \(aggregates over"
        ),
    ):
        scorer.open_consumer("fixture", plan)
    assert len(log.constructions) == 1
    assert log.constructions[0].dataset is None

    whole = _scorer(builder, _nested_frame(), log, None, tmp_path / "whole")
    assert whole.n_batches == 1
    consumer = whole.open_consumer("fixture", plan)
    assert consumer.simulate(None).calculate("medicaid_cost", 2024).sum() > 0

    # A value the written H5 stores is read, not recomputed per batch; the same
    # variable computed for another period is refused.
    stored = _EngineLog()
    stored.stored_inputs["household_income_decile"] = (2024,)
    scorer = _scorer(builder, _nested_frame(), stored, 3, tmp_path / "stored")
    scorer.open_consumer("fixture", (("household_income_decile", 2024, None),))
    assert len(stored.constructions) == scorer.n_batches
    with pytest.raises(RuntimeError, match="household_income_decile@2025"):
        scorer.open_consumer("fixture", (("household_income_decile", 2025, None),))

def test_known_periods_include_live_branches(builder) -> None:
    def engine(known):
        return SimpleNamespace(
            get_holder=lambda name: SimpleNamespace(
                get_known_periods=lambda: known.get(name, [])
            ),
            branches={},
        )

    root = engine({"medicaid_slcsp_state_denominator": ["2024"]})
    branch = engine({"household_income_decile": ["2025"]})
    root.branches["mtr"] = branch
    branch.branches["loop"] = root  # a cycle is walked once
    assert builder._engine_known_periods(
        root, builder.US_POPULATION_AGGREGATE_VARIABLES
    ) == {
        ("medicaid_slcsp_state_denominator", "2024"),
        ("household_income_decile", "2025"),
    }

def test_known_period_walker_matches_a_reference_traversal(builder) -> None:
    """The shared walker returns exactly the known (variable, period) pairs of
    every engine reachable from the root through live branches, recorded
    detached branches and ``baseline``, for the watched variables only, and
    terminates on cycles. Checked against an independent breadth-first walk
    over random engine graphs. Hypothesis is a workspace dependency; the
    wheels job installs no test extras, so it skips there."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @st.composite
    def engine_graphs(draw):
        n = draw(st.integers(1, 7))
        known = [
            {
                variable: draw(st.lists(st.integers(2020, 2035), max_size=3))
                for variable in _WALKER_VARIABLES
            }
            for _ in range(n)
        ]
        nodes = [
            SimpleNamespace(
                get_holder=(
                    lambda name, k=k: SimpleNamespace(
                        get_known_periods=lambda: k.get(name, [])
                    )
                ),
                branches={},
                _target_materialization_branches=[],
                baseline=None,
            )
            for k in known
        ]
        edges: list[tuple[int, int]] = []
        for i, node in enumerate(nodes):
            for j in draw(st.lists(st.integers(0, n - 1), max_size=3)):
                node.branches[f"b{j}"] = nodes[j]
                edges.append((i, j))
            for j in draw(st.lists(st.integers(0, n - 1), max_size=2)):
                node._target_materialization_branches.append(nodes[j])
                edges.append((i, j))
            baseline = draw(st.one_of(st.none(), st.integers(0, n - 1)))
            if baseline is not None:
                node.baseline = nodes[baseline]
                edges.append((i, baseline))
        return nodes, known, edges

    @settings(max_examples=200, deadline=None)
    @given(graph=engine_graphs(), watched=st.sets(st.sampled_from(_WALKER_VARIABLES)))
    def check(graph, watched):
        nodes, known, edges = graph
        reachable, frontier = {0}, [0]
        while frontier:
            current = frontier.pop()
            for source, target in edges:
                if source == current and target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        expected = {
            (variable, str(period))
            for index in reachable
            for variable in watched
            for period in known[index][variable]
        }
        assert (
            builder._engine_known_periods(nodes[0], tuple(sorted(watched))) == expected
        )

    check()

def test_batch_invariance_check_reads_a_value_held_only_on_baseline(
    builder,
) -> None:
    """Post-export scoring and target materialization share one walker. It
    also reads a ``reform=`` engine's ``baseline`` simulation, which holds its
    own values and is not among the engine's branches."""

    def engine(known, **extra):
        return SimpleNamespace(
            get_holder=lambda name: SimpleNamespace(
                get_known_periods=lambda: known.get(name, [])
            ),
            branches={},
            **extra,
        )

    baseline = engine({"medicaid_slcsp_state_denominator": ["2024"]})
    root = engine({}, baseline=baseline)
    assert builder._engine_known_periods(
        root, builder.US_POPULATION_AGGREGATE_VARIABLES
    ) == {("medicaid_slcsp_state_denominator", "2024")}
    with pytest.raises(
        RuntimeError,
        match=r"not batch-invariant.*medicaid_slcsp_state_denominator@2024 "
        r"\(aggregates over",
    ):
        builder._assert_post_export_scoring_is_batch_invariant(
            root,
            builder.US_POPULATION_AGGREGATE_VARIABLES,
            set(),
            label="fixture",
            batched=True,
            reform=False,
        )

def test_a_reform_engine_refuses_a_baseline_branch_reader(builder, tmp_path) -> None:
    """A batch reform engine is built from the reform's system alone, so it
    has no baseline branch: a reform pass refuses a formula that reads one,
    even in a single batch. A baseline pass has no baseline branch in a
    whole-file simulation either and scores it, and a single-batch reform pass
    may compute a population aggregate, which then spans the whole file."""
    log = _EngineLog()
    log.computes["fixture_hours"] = ("relative_wage_change",)
    log.computes["fixture_decile"] = ("household_income_decile",)
    scorer = _scorer(builder, _nested_frame(), log, None, tmp_path)
    consumer = scorer.open_consumer("fixture", (("fixture_hours", 2024, None),))
    assert consumer.simulate(None).calculate("fixture_hours", 2024).sum() > 0
    reformed = consumer.simulate(_FakeReform("fixture"))
    assert reformed.calculate("fixture_decile", 2024).sum() > 0
    with pytest.raises(
        RuntimeError,
        match=(
            r"relative_wage_change@2024 \(reads the engine's baseline branch, "
            r"which a batch reform engine does not carry\)"
        ),
    ):
        reformed.calculate("fixture_hours", 2024)
    assert all(engine.dataset is None for engine in log.constructions)
    assert builder._post_export_watched_variables(batched=False, reform=False) == ()
    assert set(builder._post_export_watched_variables(batched=True, reform=True)) == {
        *builder.US_POPULATION_AGGREGATE_VARIABLES,
        *builder.POST_EXPORT_BASELINE_BRANCH_READERS,
    }

def test_a_reform_that_moves_a_behavioral_parameter_is_refused(
    builder, tmp_path
) -> None:
    """Without a baseline branch policyengine-us scores labor-supply and
    capital-gains responses as 0, which a whole-file reform engine matches
    only while the reform leaves their parameters at baseline. A reform that
    moves one is refused before any engine is built."""
    log = _EngineLog()
    scorer = _scorer(builder, _nested_frame(), log, 3, tmp_path)
    consumer = scorer.open_consumer("fixture", ())
    dynamic = _FakeReform(
        "dynamic",
        parameters={"gov.simulation.labor_supply_responses.elasticities.income": -0.05},
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"off baseline at 2024: "
            r"gov\.simulation\.labor_supply_responses\.elasticities\.income\. "
        ),
    ):
        consumer.simulate(dynamic).calculate("income_tax", 2024)
    assert log.constructions == []
    assert consumer.record()["reform_passes"] == 0
    # Setting a behavioral parameter to its baseline value moves nothing.
    static = _FakeReform(
        "static",
        parameters={"gov.simulation.capital_gains_responses.elasticity": 0.0},
    )
    assert consumer.simulate(static).calculate("income_tax", 2024).sum() > 0
    assert len(log.constructions) == scorer.n_batches

def test_recording_dry_runs_construct_no_engine(
    builder, fake_reforms, monkeypatch
) -> None:
    """(h) Learning the plans never constructs an engine, and the plans hold
    exactly the baseline keys the consumers ask for."""

    class Refused:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a recording dry run constructed an engine")

        @staticmethod
        def default_tax_benefit_system(*args, **kwargs):
            raise AssertionError("a recording dry run built a reform system")

    monkeypatch.setitem(
        sys.modules, "policyengine_us", SimpleNamespace(Microsimulation=Refused)
    )
    args = SimpleNamespace(
        skip_reform_coverage_smoke=False,
        skip_reform_validation=False,
        skip_out_of_sample_reforms=False,
        skip_demographics=False,
    )
    plan = builder._post_export_scoring_plan(
        n_households=352_932,
        maximum_microsim_batch_size=2_000,
        consumers=builder._post_export_consumers(
            args, result=_empty_calibration_result(), release_id="fixture"
        ),
    )
    record = plan.record()
    assert record["n_batches"] == 177
    assert record["spm"] == US_RELEASE_SPM_SELECTION
    assert list(record["consumers"]) == [
        "reform_coverage_smoke",
        "reform_validation",
        "demographics",
    ]
    smoke_keys = {
        (probe.budget_measure, int(probe.period or builder.PERIOD), None)
        for probe in us_release_reform_coverage_probes()
    }
    assert set(plan.baseline_plan("reform_coverage_smoke")) == smoke_keys
    assert len(plan.baseline_plan("reform_coverage_smoke")) == len(smoke_keys)
    assert plan.baseline_plan("demographics") == (("age", builder.PERIOD, None),)
    validation = plan.baseline_plan("reform_validation")
    assert ("income_tax", builder.PERIOD, None) in validation
    assert ("state_code_str", builder.PERIOD, None) in validation
    for consumer in record["consumers"].values():
        periods = [key["period"] for key in consumer["baseline_plan"]["keys"]]
        assert periods == sorted(periods)

    # The skip flags drop their consumers from the plan.
    skipped = builder._post_export_consumers(
        SimpleNamespace(
            skip_reform_coverage_smoke=True,
            skip_reform_validation=False,
            skip_out_of_sample_reforms=True,
            skip_demographics=False,
        ),
        result=_empty_calibration_result(),
        release_id="fixture",
    )
    assert list(skipped) == ["demographics"]

def test_reform_validation_sweeps_only_released_engines(
    builder, fake_reforms, monkeypatch
) -> None:
    """The dry run's recorders hold no engine state, so reform_validation runs
    no full collection for them; an engine simulation, which carries a
    ``populations`` map, is still swept
    after its reform row and after the shared baseline."""
    collections: list = []
    monkeypatch.setattr(
        reform_validation_module,
        "gc",
        SimpleNamespace(collect=lambda *args: collections.append(args) or 0),
    )
    plan = builder._record_post_export_baseline_plan(
        builder._reform_validation_consumer(
            result=_empty_calibration_result(), release_id="fixture"
        )
    )
    assert plan and collections == []

    class EngineLike:
        def __init__(self) -> None:
            self.populations: dict = {}

        def calculate(self, variable, period, map_to=None):
            return _FakeSeries(np.ones(2), np.ones(2))

    spec = ReformValidationSpec(
        id="fixture_repeal",
        name="fixture repeal",
        category="fixture",
        in_sample=False,
        period=2024,
        jct_score=None,
        jct_window="",
        jct_source="",
        jct_source_url="",
        neutralized_variable="fixture_credit",
    )
    reform_validation_module.reform_validation_payload(
        (spec,), period=2024, simulate=lambda reform: EngineLike()
    )
    assert len(collections) == 2

def test_an_unbuildable_plan_is_recorded_and_refused_not_raised(
    builder, fake_reforms, monkeypatch, tmp_path
) -> None:
    """microcosm#547: building the plan never raises before the diagnostics.
    The error rides the plan record, becomes one terminal-batch line, and the
    scorer refuses the plan if a run ever reached the export."""
    args = SimpleNamespace(
        skip_reform_coverage_smoke=False,
        skip_reform_validation=False,
        skip_out_of_sample_reforms=False,
        skip_demographics=False,
        maximum_microsim_batch_size=3,
    )
    # A calibration result without its compiled problem cannot feed the
    # in-sample reform-validation rows.
    failed = builder._record_post_export_scoring_plan(
        args,
        n_households=7,
        result=SimpleNamespace(diagnostics=()),
        release_id="fixture",
    )
    assert failed.error is not None and failed.error.startswith("AttributeError")
    assert failed.record() == {
        "method": builder.POST_EXPORT_SCORING_METHOD,
        "error": failed.error,
    }
    (line,) = failed.terminal_failures()
    assert line.startswith("Post-export scoring plan could not be built")
    assert failed.error in line
    with pytest.raises(RuntimeError, match="not built before export: Attribute"):
        failed.baseline_plan("demographics")
    with pytest.raises(RuntimeError, match="not built before export: Attribute"):
        builder._open_post_export_scorer(failed, _written_h5(tmp_path))

    built = builder._record_post_export_scoring_plan(
        args,
        n_households=7,
        result=_empty_calibration_result(),
        release_id="fixture",
    )
    assert built.error is None and built.terminal_failures() == []
    assert list(built.baseline_plans) == [
        "reform_coverage_smoke",
        "reform_validation",
        "demographics",
    ]
    empty = builder._PostExportScoringPlan(
        n_households=7, maximum_batch_size=3, baseline_plans={}
    )
    assert builder._open_post_export_scorer(empty, _written_h5(tmp_path)) is None

def test_calibration_diagnostics_carry_the_post_export_plan(
    builder, monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def fake_write_calibration_diagnostics(result, path, *, target_registry, build):
        captured["build"] = build
        return path

    monkeypatch.setattr(
        builder, "write_calibration_diagnostics", fake_write_calibration_diagnostics
    )
    gate = SimpleNamespace(passed=True, failures=(), details={})
    plan = builder._PostExportScoringPlan(
        n_households=7,
        maximum_batch_size=3,
        baseline_plans={
            "demographics": (("age", builder.PERIOD, None),),
        },
    )
    common = dict(
        result=SimpleNamespace(),
        release_dir=tmp_path,
        registry=TargetRegistry((), country="us"),
        base_dataset_sha256="base-sha",
        compilation={"dropped_target_names": []},
        target_profile_gate=gate,
        health_input_gate=gate,
        base_population_gate=gate,
        support_value_repairs={},
        audit_export_targets=False,
        gate_failures=[],
    )
    builder._write_release_calibration_diagnostics(
        **common, post_export_scoring=plan.record()
    )
    assert captured["build"]["post_export_scoring"] == {
        "method": "household_batched_written_h5",
        "maximum_batch_size": 3,
        "n_households": 7,
        "n_batches": 3,
        "spm": {"geography_kind": "county"},
        "period_order": "ascending",
        "consumers": {
            "demographics": {
                "baseline_plan": {
                    "keys": [
                        {"variable": "age", "period": builder.PERIOD, "map_to": None}
                    ],
                    "period_order": [builder.PERIOD],
                }
            }
        },
    }
    builder._write_release_calibration_diagnostics(**common)
    assert "post_export_scoring" not in captured["build"]

def test_main_and_writers_never_build_a_whole_pool_simulation(builder) -> None:
    """(i) Neither _main nor the reform-validation and demographics writers
    call default_simulate_factory or construct a Microsimulation, and _main
    feeds the smoke from the batched scorer."""
    for name in ("_main", "_write_reform_validation", "_write_demographics"):
        _, tree = _function_source(builder, name)
        called = _called_names(tree)
        assert "default_simulate_factory" not in called, name
        assert "Microsimulation" not in called, name
        assert "USSingleYearDataset" not in called, name

    source, tree = _function_source(builder, "_main")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def calls_to(name: str) -> list[ast.Call]:
        return [
            call
            for call in calls
            if name
            in (getattr(call.func, "id", None), getattr(call.func, "attr", None))
        ]

    # The scorer opens on the written H5 with the plan recorded before export.
    assert [ast.unparse(call) for call in calls_to("_open_post_export_scorer")] == [
        "_open_post_export_scorer(post_export_scoring_plan, dataset_path)"
    ]
    # It opens the file this run has just written, never a previous run's
    # bytes at the same path (microcosm#443), and before the smoke scores.
    (write_call,) = calls_to("write_dataset")
    (open_call,) = calls_to("_open_post_export_scorer")
    (smoke_gate_call,) = calls_to("us_reform_coverage_smoke_gate")
    assert write_call.lineno < open_call.lineno < smoke_gate_call.lineno
    # The smoke's served baseline is its own recorded plan, and the gate reads
    # that consumer's seam.
    assert [ast.unparse(call) for call in calls_to("open_consumer")] == [
        "post_export_scorer.open_consumer('reform_coverage_smoke', "
        "post_export_scoring_plan.baseline_plan('reform_coverage_smoke'))"
    ]
    (smoke_call,) = calls_to("us_reform_coverage_smoke_gate")
    simulate = next(kw for kw in smoke_call.keywords if kw.arg == "simulate")
    assert ast.unparse(simulate.value) == "smoke_scoring.simulate"
    # Each writer shares the scorer and is served its own consumer's plan.
    for writer, consumer in (
        ("_write_reform_validation", "reform_validation"),
        ("_write_demographics", "demographics"),
    ):
        (call,) = calls_to(writer)
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert ast.unparse(keywords["post_export_scorer"]) == "post_export_scorer"
        plan_calls = [
            ast.unparse(node)
            for node in ast.walk(keywords["baseline_plan"])
            if isinstance(node, ast.Call)
        ]
        assert plan_calls == [f"post_export_scoring_plan.baseline_plan({consumer!r})"]
    (validation_call,) = calls_to("_write_reform_validation")
    assert {kw.arg: ast.unparse(kw.value) for kw in validation_call.keywords}[
        "simulate_out_of_sample"
    ] == "not args.skip_out_of_sample_reforms"
    # The manifest binds the sha of the bytes the closed scorer scored, and
    # nothing else assigns it.
    sha_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and "scored_dataset_sha256"
        in {
            target.id
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
    ]
    assert [ast.unparse(node.value) for node in sha_assignments] == [
        "_close_post_export_scorer(post_export_scorer)"
    ]
    (demographics_call,) = calls_to("_write_demographics")
    (manifest_call,) = calls_to("_build_manifests")
    assert demographics_call.lineno < sha_assignments[0].lineno < manifest_call.lineno
    diagnostics_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_write_release_calibration_diagnostics"
    )
    assert {kw.arg: ast.unparse(kw.value) for kw in diagnostics_call.keywords}[
        "post_export_scoring"
    ] == "post_export_scoring_plan.record()"
    # The smoke's record is registered with the scorer before its evidence is
    # written, and the evidence carries that same record.
    assert (
        "smoke_scoring_record = post_export_scorer.finish_consumer(smoke_scoring)"
        in source
    )
    assert '"post_export_scoring": smoke_scoring_record' in source
    # Both manifests get the scorer's block, taken after every consumer ran
    # and before the scorer closes.
    block_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "post_export_scoring"
            for target in node.targets
        )
    ]
    assert [ast.unparse(node.value) for node in block_assignments] == [
        "_post_export_scoring_manifest_block(post_export_scorer)"
    ]
    assert (
        demographics_call.lineno
        < block_assignments[0].lineno
        < sha_assignments[0].lineno
        < manifest_call.lineno
    )
    assert {kw.arg: ast.unparse(kw.value) for kw in manifest_call.keywords}[
        "post_export_scoring"
    ] == "post_export_scoring"
    # The plan is recorded (never raised) before the diagnostics, and its own
    # failure joins the terminal batch ahead of the export write.
    recorded = source.index(
        "post_export_scoring_plan = _record_post_export_scoring_plan("
    )
    joined = source.index(
        "terminal_gate_failures.extend(post_export_scoring_plan.terminal_failures())"
    )
    assert recorded < source.index("_write_release_calibration_diagnostics(") < joined
    assert joined < source.index("release_engine.write_dataset(")

def test_main_opens_the_scorer_its_plan_recorded(
    builder, fake_reforms, engine_free_loader, tmp_path
) -> None:
    """(i) The plan ``_main`` records before export opens the scorer
    ``_main`` scores with: the written H5, bound to its sha, split by the
    plan's batch size, and refused if its household count is not the plan's.
    Closing it returns the scored sha the manifest binds."""
    log, loads = engine_free_loader
    plan = builder._record_post_export_scoring_plan(
        _plan_args(),
        n_households=7,
        result=_empty_calibration_result(),
        release_id="fixture-release",
    )
    assert plan.error is None and plan.n_batches == 3
    path = _written_h5(tmp_path)
    scorer = builder._open_post_export_scorer(plan, path)
    assert loads == [(path, builder._sha256(path))]
    assert scorer.dataset_path == path
    assert scorer.n_batches == 3 and scorer.max_batch_households == 3
    # Served through the engine's own Microsimulation, one batch at a time.
    demographics = scorer.open_consumer(
        "demographics", plan.baseline_plan("demographics")
    )
    assert [len(engine.household_ids) for engine in log.constructions] == [3, 3, 1]
    # Only finished consumers reach the manifests' block, once each.
    assert builder._post_export_scoring_manifest_block(scorer)["consumers"] == {}
    finished = scorer.finish_consumer(demographics)
    with pytest.raises(ValueError, match="already finished"):
        scorer.finish_consumer(demographics)
    block = builder._post_export_scoring_manifest_block(scorer)
    assert block["dataset_sha256"] == builder._sha256(path)
    assert (block["n_batches"], block["max_batch_households"]) == (3, 3)
    assert block["consumers"] == {
        "demographics": {
            key: finished[key]
            for key in (
                "baseline_plan",
                "baseline_passes",
                "reform_passes",
                "reform_systems",
            )
        }
    }
    assert builder._post_export_scoring_manifest_block(None) is None
    assert builder._close_post_export_scorer(scorer) == builder._sha256(path)
    with pytest.raises(RuntimeError, match="scorer is closed"):
        scorer.open_consumer("demographics", plan.baseline_plan("demographics"))
    assert builder._close_post_export_scorer(None) is None

    other = builder._record_post_export_scoring_plan(
        _plan_args(),
        n_households=8,
        result=_empty_calibration_result(),
        release_id="fixture-release",
    )
    with pytest.raises(ValueError, match="post-export scoring plan was built for 8"):
        builder._open_post_export_scorer(other, path)

def test_writers_score_through_the_shared_scorer(
    builder, fake_reforms, monkeypatch, tmp_path
) -> None:
    """(h) The writers ``_main`` calls write what the unbatched path computes,
    with ``out_of_sample_simulated`` true (the publisher refuses false), and
    score through the scorer ``_main`` hands them: opening a second one fails
    this test, and demographics costs exactly one baseline pass. The
    calibration's in-sample JCT estimates and targets reach their rows."""
    frame = _nested_frame()
    log = _EngineLog()
    scorer = _scorer(builder, frame, log, 3, tmp_path / "h5")
    fit = _calibration_result_with_in_sample_fit(builder.PERIOD)
    plan = builder._record_post_export_scoring_plan(
        _plan_args(),
        n_households=7,
        result=fit,
        release_id="fixture-release",
    )

    def second_scorer(*args, **kwargs):
        raise AssertionError("a post-export writer opened a second scorer")

    monkeypatch.setattr(builder, "_HouseholdBatchedPostExportScorer", second_scorer)
    monkeypatch.setattr(
        builder, "geography_coverage_payload", lambda path: {"h5": Path(path).name}
    )
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    builder._write_reform_validation(
        release_dir=release_dir,
        dataset_path=scorer.dataset_path,
        result=fit,
        registry=TargetRegistry((), country="us"),
        release_id="fixture-release",
        simulate_out_of_sample=True,
        post_export_scorer=scorer,
        baseline_plan=plan.baseline_plan("reform_validation"),
    )
    written = json.loads((release_dir / "reform_validation.json").read_text())
    assert written["out_of_sample_simulated"] is True
    # The reference calls the payload directly, with the fit spelled out,
    # rather than through the consumer the writer uses.
    fitted = reform_validation_module.in_sample_reform_specs(period=builder.PERIOD)[:2]
    reference = reform_validation_module.reform_validation_payload(
        reform_validation_module.load_default_reform_specs(period=builder.PERIOD),
        period=builder.PERIOD,
        simulate=_unbatched_simulate(frame, _EngineLog()),
        in_sample_estimates={
            fitted[0].id: _IN_SAMPLE_FIT[0][0],
            fitted[1].id: _IN_SAMPLE_FIT[1][0],
            "fixture_other_target": 7.0,
        },
        in_sample_targets={
            fitted[0].id: _IN_SAMPLE_FIT[0][1],
            fitted[1].id: _IN_SAMPLE_FIT[1][1],
            "fixture_other_target": 8.0,
        },
        baseline_levels=reform_validation_module.default_baseline_level_specs(),
        release_id="fixture-release",
    )
    assert written == json.loads(json.dumps(reference, allow_nan=False))
    rows = {row["id"]: row for row in written["reforms"]}
    for spec, (estimate, target) in zip(fitted, _IN_SAMPLE_FIT, strict=True):
        row = rows[spec.id]
        assert row["in_sample"] is True
        assert row["microcosm"]["budget_effect"] == estimate
        assert row["microcosm"]["baseline_total"] is None
        assert row["jct"]["score"] == target
    # The in-sample reforms the fit does not cover are simulated.
    unfitted = reform_validation_module.in_sample_reform_specs(period=builder.PERIOD)[2]
    assert rows[unfitted.id]["microcosm"]["baseline_total"] is not None
    validation_constructions = len(log.constructions)
    assert validation_constructions > scorer.n_batches

    builder._write_demographics(
        release_dir=release_dir,
        dataset_path=scorer.dataset_path,
        release_id="fixture-release",
        post_export_scorer=scorer,
        baseline_plan=plan.baseline_plan("demographics"),
    )
    demographics_engines = log.constructions[validation_constructions:]
    assert len(demographics_engines) == scorer.n_batches
    assert all(engine.tax_benefit_system is None for engine in demographics_engines)
    ages, weights = builder._demographics_consumer(
        _unbatched_simulate(frame, _EngineLog())
    )
    expected = demographics_payload(
        ages, weights, period=builder.PERIOD, release_id="fixture-release"
    )
    expected["geography_coverage"] = {"h5": "populace_us_2024.h5"}
    written = json.loads((release_dir / "demographics.json").read_text())
    assert written == json.loads(json.dumps(expected, allow_nan=False))

def test_a_direct_writer_call_opens_and_closes_its_own_scorer(
    builder, fake_reforms, engine_free_loader, monkeypatch, tmp_path
) -> None:
    """A writer called without ``_main``'s scorer records its plan by the
    same dry run and scores through a scorer of its own, which it closes."""
    log, loads = engine_free_loader
    closed = []
    scorer_cls = builder._HouseholdBatchedPostExportScorer
    close = scorer_cls.close
    monkeypatch.setattr(
        scorer_cls, "close", lambda self: closed.append(self) or close(self)
    )
    monkeypatch.setattr(builder, "geography_coverage_payload", lambda path: {})
    path = _written_h5(tmp_path / "h5")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    builder._write_demographics(
        release_dir=release_dir,
        dataset_path=path,
        release_id="fixture-release",
        maximum_microsim_batch_size=3,
    )
    assert loads == [(path, builder._sha256(path))]
    assert [len(engine.household_ids) for engine in log.constructions] == [3, 3, 1]
    assert len(closed) == 1 and closed[0].dataset_path == path
    written = json.loads((release_dir / "demographics.json").read_text())
    assert written["period"] == builder.PERIOD
    assert written["total_population"] > 0

def test_main_frees_the_target_frame_before_the_export(builder) -> None:
    """``del target_frame`` follows its last reader and precedes the export
    write, and nothing reads the name afterwards."""
    _, tree = _function_source(builder, "_main")
    deletes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == "target_frame"
            for target in node.targets
        )
    ]
    assert len(deletes) == 1
    deleted_at = deletes[0].lineno
    writes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_dataset"
    ]
    assert writes and deleted_at < min(writes)
    reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "target_frame"
        and isinstance(node.ctx, ast.Load)
    ]
    assert reads and max(reads) < deleted_at

def test_calibration_result_holds_no_reference_to_the_target_frame() -> None:
    """The premise of ``del target_frame``: ``calibrate`` keeps no reference
    to its input frame, and ``result.frame`` shares no memory with it."""
    frame, targets = _small_calibration_problem()
    result = calibrate(frame, targets, epochs=3, mass="conserve", seed=0)
    gc.collect()
    # Nothing the calibration returned (or cached) refers to the input frame.
    referrers = [
        type(referrer).__name__
        for referrer in gc.get_referrers(frame)
        if not inspect.isframe(referrer)
    ]
    assert referrers == []
    assert result.frame is not frame
    for entity in ("person", "household"):
        for column in frame.table(entity).columns:
            assert not np.shares_memory(
                frame.table(entity)[column].to_numpy(),
                result.frame.table(entity)[column].to_numpy(),
            )

def test_calibration_results_drop_their_frames_and_keep_everything_else(
    builder, monkeypatch
) -> None:
    """Dropping target tables preserves diagnostics, estimates and weights."""
    # The export helpers check formula ownership against the installed engine's
    # metadata index; that check is not under test here, and the fast CI groups
    # run without policyengine-us.
    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", lambda frame: None)
    from microcosm.build.us_runtime.exact_k_ladder import (
        ExactKLadderCalibration,
        assert_exact_k_realized_count,
    )
    from microcosm.calibrate import calibrate_l0_refit
    from microcosm.calibrate.diagnostics import diagnostics_payload

    frame, targets = _small_calibration_problem()
    dense = calibrate(frame, targets, epochs=3, mass="conserve", seed=0)
    dropped = builder._without_calibrated_frames(dense)
    assert dropped.frame is None
    assert dropped.weights is dense.weights and dropped.problem is dense.problem
    assert diagnostics_payload(dropped) == diagnostics_payload(dense)
    assert builder._in_sample_estimates(dropped) == builder._in_sample_estimates(dense)
    # Once the full result goes, nothing but this test holds its frame.
    held = dense.frame
    del dense
    gc.collect()
    assert [
        type(referrer).__name__
        for referrer in gc.get_referrers(held)
        if not inspect.isframe(referrer)
    ] == []
    del held

    refit = calibrate_l0_refit(
        frame, targets, epochs=5, refit_epochs=5, target_records=20, seed=0
    )
    dropped = builder._without_calibrated_frames(refit)
    assert dropped.selection.frame is None and dropped.refit.frame is None
    assert diagnostics_payload(dropped) == diagnostics_payload(refit)
    np.testing.assert_array_equal(
        dropped.selected_entity_ids, refit.selected_entity_ids
    )
    exported = builder._with_l0_refit_weights(frame, refit)
    np.testing.assert_array_equal(
        builder._with_l0_refit_weights(frame, dropped).weights_for("household").values,
        exported.weights_for("household").values,
    )
    kept = builder._without_calibrated_frames(refit, export_frame=exported)
    assert kept.selection.frame is None and kept.refit.frame is exported
    assert kept.refit.frame is not refit.refit.frame
    k = int(refit.refit.frame.n("household"))
    outcome = ExactKLadderCalibration(
        result=kept,
        support=np.arange(k),
        selected_inclusion_probabilities=np.ones(k),
        selection_receipt={},
        refit_baseline_diagnostics={},
    )
    assert assert_exact_k_realized_count(outcome, k) == k
    # A test double carries no calibrated frame and passes through.
    double = _empty_calibration_result()
    assert builder._without_calibrated_frames(double) is double

def test_main_drops_the_calibration_frames_before_the_export(builder) -> None:
    """Cleanup follows export construction and precedes the H5 write."""
    _, tree = _function_source(builder, "_main")
    (deleted,) = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        and any(
            getattr(target, "id", None) == "target_frame" for target in node.targets
        )
    ]
    frame_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "frame"
        and getattr(node.value, "id", None) == "result"
    ]
    later_bindings = sorted(
        (node.lineno, ast.unparse(node.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "result" for target in node.targets)
        and node.lineno > deleted
    )
    (write_call,) = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "write_dataset"
    ]
    assert frame_reads and max(frame_reads) < deleted
    assert [value for _, value in later_bindings] == [
        "_without_calibrated_frames(result, "
        "export_frame=export_frame if ladder_outcome is not None else None)",
    ]
    assert all(deleted < lineno < write_call for lineno, _ in later_bindings)
    export_bindings = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "export_frame" for target in node.targets
        )
    ]
    assert export_bindings and max(export_bindings) < later_bindings[0][0]
    assert "dataclasses.replace(ladder_outcome, result=result)" in ast.unparse(tree)

@pytest.mark.parametrize("full_pool", [True, False])
@pytest.mark.parametrize("exact_k", [True, False])
def test_main_has_no_calibrated_target_frame_at_the_export_write(
    builder, full_pool, exact_k, monkeypatch
) -> None:
    """Execute the export/cleanup statements with real calibration results."""
    # As above: formula ownership needs the engine and is not under test.
    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", lambda frame: None)
    from microcosm.calibrate import calibrate_l0_refit

    target_frame, targets = _small_calibration_problem()
    base_frame = Frame(
        {
            "person": target_frame.table("person"),
            "household": target_frame.table("household").drop(columns="x"),
        },
        target_frame.schema,
        {"household": target_frame.weights_for("household")},
    )
    if full_pool:
        result = calibrate(target_frame, targets, epochs=3, mass="conserve", seed=0)
    else:
        result = calibrate_l0_refit(
            target_frame, targets, epochs=5, refit_epochs=5, target_records=20, seed=0
        )
    ladder_outcome = (
        builder.ExactKLadderCalibration(
            result=result,
            support=np.arange(len(result.weights)),
            selected_inclusion_probabilities=np.ones(len(result.weights)),
            selection_receipt={},
            refit_baseline_diagnostics={},
        )
        if exact_k
        else None
    )
    _, tree = _function_source(builder, "_main")
    statements = tree.body[0].body
    export_start = next(
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "full_pool_calibration"
        and any(
            isinstance(child, ast.Assign)
            and any(
                getattr(target, "id", None) == "export_frame"
                for target in child.targets
            )
            for child in ast.walk(node)
        )
    )
    cleanup_end = next(
        index
        for index, node in enumerate(statements)
        if index > export_start
        and isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "isinstance(ladder_outcome, ExactKLadderCalibration)"
    )
    (write,) = [
        node
        for node in statements
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) == "write_dataset"
    ]
    writes = []

    def check_write(frame, path, *, period):
        writes.append(frame)
        assert "x" not in frame.table("household")
        cleaned = namespace["result"]
        if not full_pool:
            assert cleaned.selection.frame is None
        assert cleaned.frame is (frame if exact_k else None)
        if exact_k:
            outcome = namespace["ladder_outcome"]
            assert outcome.result is cleaned
            assert builder.assert_exact_k_realized_count(
                outcome, len(result.weights)
            ) == len(result.weights)

    namespace = {
        **vars(builder),
        "base_frame": base_frame,
        "result": result,
        "ladder_outcome": ladder_outcome,
        "full_pool_calibration": full_pool,
        "release_engine": SimpleNamespace(write_dataset=check_write),
        "dataset_path": Path("fixture.h5"),
    }
    module = ast.Module(
        body=[*statements[export_start : cleanup_end + 1], write], type_ignores=[]
    )
    exec(compile(module, "<main export statements>", "exec"), namespace)
    assert len(writes) == 1

def test_scored_sha_is_bound_at_load_and_checked_by_the_manifest(
    builder, tmp_path
) -> None:
    """(j) The scorer loads the written H5 against the sha it records, and
    _build_manifests refuses a dataset whose sha differs from the scored one."""
    loads: list = []
    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path, loads=loads)
    path = tmp_path / "populace_us_2024.h5"
    assert scorer.dataset_sha256 == builder._sha256(path)
    assert loads == [(path, scorer.dataset_sha256)]
    record = scorer.open_consumer("fixture", ()).record()
    assert record["dataset_sha256"] == builder._sha256(path)
    assert record["baseline_passes"] == 0

    # The manifest check runs before anything else reads the arguments.
    required = {
        name: None
        for name, parameter in inspect.signature(
            builder._build_manifests
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    required.update(
        release_dir=tmp_path,
        artifact_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="never scored"):
        builder._build_manifests(
            **required,
            dataset_filename=path.name,
            scored_dataset_sha256="0" * 64,
        )
    # The manifests' scoring block must describe the same bytes.
    with pytest.raises(RuntimeError, match="does not pin"):
        builder._build_manifests(
            **required,
            dataset_filename=path.name,
            post_export_scoring={"dataset_sha256": "0" * 64},
        )
    # A consumer opened on another scorer cannot be finished on this one.
    other = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path / "other")
    with pytest.raises(ValueError, match="not opened on this scorer"):
        scorer.finish_consumer(other.open_consumer("fixture", ()))

    _, tree = _function_source(builder, "_main")
    manifest_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_build_manifests"
    )
    keywords = {kw.arg: ast.unparse(kw.value) for kw in manifest_call.keywords}
    assert keywords["scored_dataset_sha256"] == "scored_dataset_sha256"

def test_writers_register_their_consumer_with_the_shared_scorer(
    builder, tmp_path
) -> None:
    """(j) Reform validation and demographics score through
    ``_score_post_export_consumer``; on ``_main``'s shared scorer each
    registers its finished record, so the manifests' block names it."""
    scorer = _scorer(builder, _nested_frame(), _EngineLog(), 3, tmp_path)
    output = builder._score_post_export_consumer(
        "reform_validation",
        lambda simulate: "payload",
        dataset_path=scorer.dataset_path,
        post_export_scorer=scorer,
        baseline_plan=(),
        maximum_microsim_batch_size=3,
    )
    assert output == "payload"
    assert list(scorer.consumer_records) == ["reform_validation"]
    block = builder._post_export_scoring_manifest_block(scorer)
    assert list(block["consumers"]) == ["reform_validation"]
    assert block["consumers"]["reform_validation"]["reform_passes"] == 0

def test_scorer_refuses_a_unit_that_spans_households(builder, tmp_path) -> None:
    """(k) Frame.select does not check that group units nest in households,
    so the scorer does, before any engine exists."""
    frame = _nested_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    # Move the second member of household 1 into household 2's tax unit.
    person.loc[person["person_id"] == 2, "person_tax_unit_id"] = 20
    tables["tax_unit"] = pd.DataFrame(
        {"tax_unit_id": np.unique(person["person_tax_unit_id"].to_numpy())}
    )
    spanning = Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")})
    log = _EngineLog()
    with pytest.raises(ValueError, match="nested in households"):
        _scorer(builder, spanning, log, 1, tmp_path)
    assert log.constructions == []

    # The partition check is a second line: a batch set that misses or
    # repeats a household is refused.
    batches = [
        builder._select_households_by_position(frame, np.asarray([0, 1, 2])),
        builder._select_households_by_position(frame, np.asarray([2, 3, 4, 5, 6])),
    ]
    with pytest.raises(ValueError, match="do not partition the pool"):
        builder._assert_post_export_batching_premises(frame, batches)

    # Batches that partition the pool must also keep the written household
    # order and household weights, or the concatenated arrays would misalign.
    batches = [
        builder._select_households_by_position(frame, np.asarray(positions))
        for positions in ([0, 1, 2], [3, 4, 5, 6])
    ]
    builder._assert_post_export_batching_premises(frame, batches)
    with pytest.raises(ValueError, match="household order"):
        builder._assert_post_export_batching_premises(frame, batches[::-1])
    last = batches[1]
    reweighted = Frame(
        {entity: last.table(entity) for entity in last.entities},
        US_SCHEMA,
        {
            "household": Weights(
                last.weights_for("household").values * 2.0, WeightKind.CALIBRATED
            )
        },
    )
    with pytest.raises(ValueError, match="household weights"):
        builder._assert_post_export_batching_premises(frame, [batches[0], reweighted])

def test_scorer_refuses_a_household_count_other_than_planned(builder, tmp_path) -> None:
    with pytest.raises(ValueError, match="built for 8"):
        builder._HouseholdBatchedPostExportScorer(
            _written_h5(tmp_path),
            maximum_microsim_batch_size=3,
            expected_n_households=8,
            microsimulation_cls=_fake_engine(_EngineLog()),
            dataset_from_frame=lambda batch_frame: batch_frame,
            load_frame=lambda path, *, expected_sha256: _nested_frame(),
        )

def test_guard_sweep_records_and_scores_shipped_reform_passes(
    builder, fake_reforms, monkeypatch, tmp_path
) -> None:
    sweep = _load_sweep_module()
    plans, requests, counts = sweep._record_requests(builder)
    assert counts["smoke_probes"] == len(us_release_reform_coverage_probes()) == 41
    specs = reform_validation_module.load_default_reform_specs(period=builder.PERIOD)
    assert counts["validation_specs"] == sum(not spec.in_sample for spec in specs) == 52
    assert (
        counts["in_sample_fallback_specs"]
        == sum(spec.in_sample for spec in specs)
        == 12
    )
    assert counts["baseline_keys"] == {
        "reform_coverage_smoke": 18,
        "reform_validation": 75,
        "demographics": 1,
    }
    smoke_requests = [
        request for request in requests if request[0] == "reform_coverage_smoke"
    ]
    assert [reform.label for _, reform, _ in smoke_requests] == [
        probe.id for probe in us_release_reform_coverage_probes()
    ]
    validation_requests = [
        request for request in requests if request[0] == "reform_validation"
    ]
    ordinary_specs = [
        spec for spec in specs if not reform_validation_module._is_obbba_spec(spec)
    ]
    assert {spec.id for spec in ordinary_specs}.issubset(
        {reform.label for _, reform, _ in validation_requests}
    )
    obbba_specs = [
        spec for spec in specs if reform_validation_module._is_obbba_spec(spec)
    ]
    pre_baseline = reform_validation_module._merged_parameter_changes(obbba_specs)
    pre_baseline_requests = [
        key
        for _, reform, key in validation_requests
        if reform.label == json.dumps(pre_baseline, sort_keys=True, default=str)
    ]
    obbba_groups = {(spec.budget_measure, spec.period) for spec in obbba_specs}
    assert {(key[0], key[1]) for key in pre_baseline_requests} == obbba_groups
    assert len(pre_baseline_requests) == len(obbba_groups) == 2
    assert counts["validation_reform_passes"] == len(specs) + len(obbba_groups) == 66
    assert (
        len(requests)
        == counts["smoke_probes"] + counts["validation_reform_passes"]
        == 107
    )

    frame, log = _nested_frame(), _EngineLog()
    monkeypatch.setattr(
        reform_validation_module,
        "default_simulate_factory",
        lambda path: _unbatched_simulate(frame, _EngineLog()),
    )
    scorer = _scorer(builder, frame, log, 3, tmp_path)
    sweep._score_baselines(builder, scorer, tmp_path / "fixture.h5", plans)
    records = [
        sweep._score_reforms(
            builder, scorer, tmp_path / "fixture.h5", requests, start, 4
        )
        for start in range(0, len(requests), 4)
    ]
    assert sum(record["reform_passes"] for record in records) == len(requests)
    assert sum(len(record["whole_file_comparisons"]) for record in records) == 4
    assert len(log.systems) == len(requests)
    scorer.close()

def test_guard_sweep_runs_every_chunk_and_propagates_worker_failure(tmp_path) -> None:
    import subprocess

    sweep = _load_sweep_module()
    starts = []

    def run(command, *, check):
        assert check is True
        output = Path(command[command.index("--output") + 1])
        if "--worker-start" in command:
            start = int(command[command.index("--worker-start") + 1])
            starts.append(start)
            record = {
                "reform_passes": min(4, 9 - start),
                "whole_file_comparisons": [start] if start == 0 else [],
            }
        else:
            record = {"total_reform_passes": 9}
        output.write_text(
            json.dumps(
                {
                    **record,
                    "peak_rss_bytes": 1234,
                    "elapsed_seconds": 1.0,
                }
            )
        )

    args = SimpleNamespace(
        dataset_path=tmp_path / "fixture.h5",
        batch_size=3,
        reforms_per_worker=4,
        output=tmp_path / "report.json",
    )
    report = sweep._run_workers(args, run=run)
    assert starts == [0, 4, 8]
    assert report["scored_reform_passes"] == 9
    assert report["whole_file_comparisons"] == [0]
    assert len(report["workers"]) == 4
    assert report["peak_rss_bytes"] == 1234

    def fail(command, *, check):
        raise subprocess.CalledProcessError(2, command)

    with pytest.raises(subprocess.CalledProcessError):
        sweep._run_workers(args, run=fail)
