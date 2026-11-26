"""Tests split from packages/microcosm-build/tests/test_us_post_export_scoring.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_post_export_scoring import *


def test_watched_formula_lists_match_the_installed_engine(builder) -> None:
    """Pin the watched lists to the installed engine's known source patterns.

    The scan matches weight reads, selected population-aggregate operations
    and baseline-branch references in ``variables`` and contrib ``reforms``.
    It follows matching helper names too. Changes detected by these patterns
    require the lists to be revisited; this is not a general additivity proof.
    """
    modules = _engine_formula_modules()
    aggregates = _variables_reaching(
        modules, _POPULATION_AGGREGATE_MARKER, weight_reads=True
    )
    assert sorted(aggregates) == sorted(builder.US_POPULATION_AGGREGATE_VARIABLES), (
        aggregates
    )
    readers = _variables_reaching(modules, _BASELINE_BRANCH_MARKER)
    assert sorted(readers) == sorted(
        {
            *builder.POST_EXPORT_BASELINE_BRANCH_READERS,
            *builder.POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS,
        }
    ), readers
    # The two responses left to the parameter check read their own subtree.
    sources = {
        node.name: ast.get_source_segment(source, node)
        for source, tree in modules
        for node in ast.walk(tree)
        if _is_variable_class(node)
    }
    for response, subtree in builder.POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS.items():
        assert f"parameters(period).{subtree}" in sources[response], response


@pytest.mark.slow
def test_batched_scoring_matches_whole_pool_policyengine_us(builder, tmp_path) -> None:
    """(l) On a small written H5 with Maryland households, three probes and
    two validation reforms score the same totals on batch engines (each given
    the reform's system alone) as on a whole-file engine given ``reform=``, a
    whole-file reform engine scores no behavioral response at the engine's
    defaults, and the 75-key reform-validation baseline plan runs in ascending
    period order. The request-order MD CCS premise is observed, with a warning
    if the engine no longer raises the expected error."""
    from policyengine_us import Microsimulation

    path = _write_engine_h5(builder, tmp_path, _ENGINE_HOUSEHOLDS)

    probes = (
        _engine_probe(
            "wages_income_tax_2024",
            "income_tax",
            2024,
            neutralized_variable="employment_income_before_lsr",
        ),
        _engine_probe(
            "wages_state_income_tax_2025",
            "state_income_tax",
            2025,
            neutralized_variable="employment_income_before_lsr",
        ),
        _engine_probe(
            "zero_single_standard_deduction_2026",
            "income_tax",
            2026,
            parameter_changes={
                "gov.irs.deductions.standard.amount.SINGLE": {"2026-01-01": 0}
            },
        ),
    )
    whole = builder.us_reform_coverage_smoke_gate(
        simulate=reform_validation_module.default_simulate_factory(path),
        probes=probes,
        period=builder.PERIOD,
    )
    scorer = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=2
    )
    assert scorer.n_batches == 2

    def smoke(simulate):
        return builder.us_reform_coverage_smoke_gate(
            simulate=simulate, probes=probes, period=builder.PERIOD
        )

    plan = builder._record_post_export_baseline_plan(smoke)
    assert [key[1] for key in plan] == [2024, 2025, 2026]
    scoring = scorer.open_consumer("reform_coverage_smoke", plan)
    batched = smoke(scoring.simulate)
    for probe in probes:
        expected = whole.details["results"][probe.id]
        observed = batched.details["results"][probe.id]
        for field in ("baseline_total", "reform_total", "effect"):
            assert observed[field] == pytest.approx(
                expected[field], rel=1e-12, abs=1e-6
            ), (probe.id, field)
        assert expected["effect"] != 0.0, probe.id
    assert scoring.record()["reform_systems"] == len(probes)

    # Validation reforms, a credit repeal and a structural contrib reform: the
    # batch engines (the reform's system alone, no baseline branch) score what
    # a whole-file engine given ``reform=`` scores.
    specs = {
        spec.id: spec
        for spec in reform_validation_module.load_default_reform_specs(
            period=builder.PERIOD
        )
    }
    for spec_id in ("state_repeal_md_eitc", "federal.ubi_mechanical"):
        spec = specs[spec_id]
        reform = spec.build_reform()
        whole_engine = reform_validation_module.default_simulate_factory(path)(reform)
        expected = float(whole_engine.calculate(spec.budget_measure, spec.period).sum())
        builder.release_engine_simulation(whole_engine)
        del whole_engine
        observed = (
            scoring.simulate(reform).calculate(spec.budget_measure, spec.period).sum()
        )
        assert expected != 0.0, spec_id
        assert observed == pytest.approx(expected, rel=1e-12, abs=1e-6), spec_id

    # The premise of dropping the baseline branch: at the installed engine's
    # defaults, a whole-file reform engine (baseline branch present) scores no
    # labor-supply or capital-gains response, as the batch engines do.
    dynamic_probe = probes[2]
    whole_engine = reform_validation_module.default_simulate_factory(path)(
        smoke_module._build_reform(dynamic_probe)
    )
    assert whole_engine.baseline is not None
    for response in builder.POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS:
        values = whole_engine.calculate(response, dynamic_probe.period)
        assert float(np.abs(np.asarray(values)).sum()) == 0.0, response
    builder.release_engine_simulation(whole_engine)
    del whole_engine

    validation_plan = builder._record_post_export_baseline_plan(
        builder._reform_validation_consumer(
            result=_empty_calibration_result(), release_id="fixture"
        )
    )
    periods = [key[1] for key in validation_plan]
    assert len(validation_plan) == 75
    assert periods == sorted(periods) and max(periods) > 2025
    # Probe the ascending-order premise on the first batch by requesting a
    # 2025 key after 2024 and 2027 keys. The warning permits an upstream fix
    # without requiring an unrelated lock bump to preserve the error.
    (first_batch, _) = scorer._batches()
    engine = Microsimulation(
        dataset=builder._dataset_from_frame(
            first_batch,
            assert_no_formula_owned_columns=False,
        ),
        spm=dict(US_RELEASE_SPM_SELECTION),
    )
    engine.calculate("income_tax", 2024)
    engine.calculate("md_income_tax", 2027)
    try:
        engine.calculate("nd_income_tax", 2025)
    except Exception as error:  # noqa: BLE001 - the engine's own error type
        assert "md.msde.ccs.payment.informal.rates" in str(error), error
    else:
        warnings.warn(
            "policyengine-us no longer raises the MD CCS ParameterNotFoundError "
            "for request-order periods; revisit the ascending-period guard "
            "(microcosm#956).",
            stacklevel=1,
        )
    finally:
        builder.release_engine_simulation(engine)
        del engine
    # The same batch scores every reform-validation baseline key in ascending
    # period order without error.
    validation = scorer.open_consumer("reform_validation", validation_plan)
    income_tax = validation.simulate(None).calculate("income_tax", builder.PERIOD)
    assert income_tax.sum() == pytest.approx(
        whole.details["results"]["wages_income_tax_2024"]["baseline_total"],
        rel=1e-12,
    )
    scorer.close()


@pytest.mark.slow
def test_additivity_guards_fire_on_the_installed_engine(builder, tmp_path) -> None:
    """The three additivity guards refuse on policyengine-us itself.

    The fake-engine tests above pin the guards' logic; these read what a real
    engine computed (its holders' known periods) and what a real reform
    system sets (its parameter leaves), covering API behavior that the fake
    engine does not establish.
    ``medicaid_cost`` reaches the Medicaid SLCSP state sums: two batches
    refuse it, one batch scores the whole-file total, and a reform engine
    refuses it even in one batch, because the state denominator reads the
    baseline branch. A reform that moves the behavioral-response elasticities
    is refused by name before a batch engine for that reform is built."""
    from policyengine_core.reforms import Reform
    from policyengine_us import Microsimulation

    path = _write_engine_h5(builder, tmp_path, _GUARD_HOUSEHOLDS)
    key = ("medicaid_cost", 2024, None)
    whole = reform_validation_module.default_simulate_factory(path)(None)
    expected = float(whole.calculate("medicaid_cost", 2024).sum())
    enrolled = int(np.asarray(whole.calculate("medicaid_enrolled", 2024)).sum())
    builder.release_engine_simulation(whole)
    del whole
    assert expected > 0.0 and enrolled > 0

    batched = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=4
    )
    assert batched.n_batches == 2
    with pytest.raises(
        RuntimeError,
        match=(
            r"not batch-invariant: a batch engine computed .*"
            r"medicaid_slcsp_state_denominator@2024 \(aggregates over"
        ),
    ):
        batched.open_consumer("guard", (key,))
    batched.close()

    scorer = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=None
    )
    assert scorer.n_batches == 1
    consumer = scorer.open_consumer("guard", (key,))
    scored = consumer.simulate(None).calculate("medicaid_cost", 2024).sum()
    assert scored == pytest.approx(expected, rel=1e-12)

    # A reform that leaves the behavioral parameters alone passes that check
    # and reaches an engine, which refuses the denominator.
    static = Reform.from_dict(
        {"gov.irs.deductions.standard.amount.SINGLE": {"2024-01-01": 0}},
        country_id="us",
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"computed medicaid_slcsp_state_denominator@2024 \(reads the "
            r"engine's baseline branch, which a batch reform engine does not "
            r"carry\)"
        ),
    ):
        consumer.simulate(static).calculate("medicaid_cost", 2024)
    income_elasticity = "gov.simulation.labor_supply_responses.elasticities.income"
    dynamic = Reform.from_dict(
        {income_elasticity: {"2024-01-01": -0.05}},
        country_id="us",
    )
    assert builder._moved_behavioral_response_parameters(
        Microsimulation.default_tax_benefit_system(reform=dynamic),
        Microsimulation.default_tax_benefit_system_instance,
        2024,
    ) == [income_elasticity]
    with pytest.raises(
        RuntimeError,
        match=(
            r"off baseline at 2024: "
            r"gov\.simulation\.labor_supply_responses\.elasticities\.income\. "
        ),
    ):
        consumer.simulate(dynamic).calculate("income_tax", 2024)
    record = consumer.record()
    assert (record["reform_systems"], record["reform_passes"]) == (2, 1)
    scorer.close()


@pytest.mark.slow
def test_shipped_baseline_plans_match_whole_file_policyengine_us(
    builder, tmp_path
) -> None:
    """The smoke, validation and demographics baseline values, served from
    one merged pass in 3 batches, match a whole-file engine.

    The complete reform sweep runs separately through
    ``tools/sweep_us_post_export_scoring.py`` in bounded worker processes.
    """
    sweep = _load_sweep_module()
    path = _write_engine_h5(builder, tmp_path, _GUARD_HOUSEHOLDS)
    plans, _, counts = sweep._record_requests(builder)
    # microcosm#1251's passive partnership/S-corp probe adds one smoke baseline key.
    assert counts["baseline_keys"] == {
        "reform_coverage_smoke": 20,
        "reform_validation": 75,
        "demographics": 1,
    }
    scorer = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=3, baseline_plans=plans
    )
    assert scorer.n_batches == 3
    passes = _record_pass_labels(scorer)
    try:
        sweep._score_baselines(builder, scorer, path, plans)
        # One merged pass, not a fallback to one pass per plan.
        assert passes == ["merged baseline plan"]
    finally:
        scorer.close()


@pytest.mark.slow
def test_merged_baseline_and_shared_pre_obbba_pass_are_byte_identical(
    builder, tmp_path
) -> None:
    """On policyengine-us, in 3 household batches: the one merged baseline
    pass serves every planned consumer the same bytes (values, weights and
    dtypes) as that consumer's own pass, and the pre-OBBBA state scored once
    for both OBBBA groups gives each group's total the bytes of a pass of its
    own (microcosm#1595). One person holds a taxable estate, so the estate
    tax, which the shared pass computes after the income tax, is nonzero."""
    people = sum(len(members) for _, _, members in _GUARD_HOUSEHOLDS)
    path = _write_engine_h5(
        builder,
        tmp_path,
        _GUARD_HOUSEHOLDS,
        person_inputs={"taxable_estate_value": [30_000_000.0] + [0.0] * (people - 1)},
    )
    fit = _full_in_sample_fit(builder.PERIOD)
    plan = builder._record_post_export_scoring_plan(
        _plan_args(),
        n_households=len(_GUARD_HOUSEHOLDS),
        result=fit,
        release_id="fixture-release",
    )
    assert plan.error is None and len(plan.baseline_plans) == 5
    separate = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=3
    )
    merged = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=3, baseline_plans=plan.baseline_plans
    )
    assert merged.n_batches == separate.n_batches == 3
    merged_passes = _record_pass_labels(merged)
    try:
        for name in plan.baseline_plans:
            expected = separate.open_consumer(name, plan.baseline_plan(name))
            observed = merged.open_consumer(name, plan.baseline_plan(name))
            for key in plan.baseline_plan(name):
                assert _same_bytes(observed._baseline[key], expected._baseline[key]), (
                    name,
                    key,
                )
            del expected, observed
        # One merged pass, not a fallback to one pass per consumer.
        assert merged_passes == ["merged baseline plan"]
        assert not merged._merged_baseline

        specs = reform_validation_module.load_default_reform_specs(
            period=builder.PERIOD
        )
        obbba = [
            spec for spec in specs if reform_validation_module._is_obbba_spec(spec)
        ]
        reform = reform_validation_module._build_parameter_reform(
            reform_validation_module._merged_parameter_changes(obbba)
        )
        requests = tuple(
            dict.fromkeys((spec.budget_measure, spec.period) for spec in obbba)
        )
        assert len(requests) == 2
        consumer = separate.open_consumer("pre-OBBBA state", ())
        shared = consumer.simulate(reform)
        shared.prefetch(requests)
        # Read the shared pass's values first: opening the next reform
        # finishes this one, which then refuses further keys.
        shared_values = {request: shared.calculate(*request) for request in requests}
        for request in requests:
            alone = consumer.simulate(reform).calculate(*request)
            assert _same_bytes(shared_values[request], alone), request
            assert alone.sum() != 0.0, request
        assert consumer.record()["reform_passes"] == 1 + len(requests)
    finally:
        separate.close()
        merged.close()


@pytest.mark.slow
def test_reform_passes_leave_no_rule_modules_behind_and_score_the_same(
    builder, tmp_path, monkeypatch
) -> None:
    """Each reform's rules build adds about 6,000 variable modules to
    ``sys.modules`` that policyengine-core never removes (about 0.04 GiB a
    reform pass on this fixture). The scorer removes them once the consumer
    opens the next reform or finishes, and the scored arrays are byte-identical
    to scoring without the removal."""
    import policyengine_us

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    source = _write_engine_h5(builder, tmp_path / "export", _GUARD_HOUSEHOLDS)
    probes = {probe.id: probe for probe in us_release_reform_coverage_probes()}
    chosen = [
        probes["keogh_distribution_neutralization"],
        probes["collectibles_gain_neutralization"],
    ]
    variables_dir = str(
        Path(policyengine_us.CountryTaxBenefitSystem.variables_dir).absolute()
    )

    def rule_modules() -> set[str]:
        return {
            name
            for name, module in tuple(sys.modules.items())
            if re.fullmatch(r"\d+_-?\d+_\w+", name)
            and str(getattr(module, "__file__", "")).startswith(variables_dir)
        }

    def score():
        scorer = builder._HouseholdBatchedPostExportScorer(
            source, maximum_microsim_batch_size=None
        )
        consumer = scorer.open_consumer("fixture", ())
        before = rule_modules()
        scored = []
        for probe in chosen:
            values = consumer.simulate(_build_reform(probe)).calculate(
                probe.budget_measure, period=probe.period
            )
            scored.append((np.asarray(values).tobytes(), values.weights.tobytes()))
        scorer.finish_consumer(consumer)
        scorer.close()
        return before, rule_modules(), scored

    before, after, released = score()
    assert after == before

    class KeepEveryModule:
        """A scope that removes nothing: post-export scoring before the fix."""

        def __init__(self, variables_dir=None) -> None:
            pass

        def open(self):
            return self

        def close(self) -> None:
            pass

    monkeypatch.setattr(builder, "EngineVariableModuleScope", KeepEveryModule)
    kept_before, kept_after, kept = score()
    assert len(kept_after - kept_before) >= 2 * 1_000
    assert released == kept


def test_batches_read_from_their_files_equal_the_written_h5s_batches(
    builder, tmp_path
) -> None:
    """Each batch the scorer reads back from its file holds exactly
    the rows, columns, dtypes, values and weights of that household batch of a
    fixture H5 written by the engine's own writer, as the scorer selected it
    in memory before it kept batches in files."""
    from test_support.microcosm_build.us_post_export_batch_files import (
        assert_same_batch,
    )

    path = _write_engine_h5(builder, tmp_path, _GUARD_HOUSEHOLDS)
    written = builder._load_frame(path, expected_sha256=builder._sha256(path))
    expected = [
        builder.select_households_by_position(written, positions)
        for positions in builder.household_position_batches(written.n("household"), 3)
    ]
    scorer = builder._HouseholdBatchedPostExportScorer(
        path, maximum_microsim_batch_size=3
    )
    try:
        batches = scorer._batches()
        assert len(batches) == len(expected) == scorer.n_batches >= 2
        for number, selected in enumerate(expected):
            assert_same_batch(batches[number], selected)
    finally:
        scorer.close()
