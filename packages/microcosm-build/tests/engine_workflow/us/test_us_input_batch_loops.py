"""The release's input-stage household-batch loops (tracker task 1.6 · 2).

Five loops run policyengine-us over household batches before target
compilation: ACA source inputs, Medicaid eligibility, December SSI candidate
amounts, SNAP eligibility and the other-health-insurance modeled premiums.
Each must write exactly what main wrote, refuse a batch engine that computed a
population-aggregate variable (as target compilation does), refuse a frame
that household batches would split, and release every batch engine before the
next one is built, also when a batch fails.
"""

from __future__ import annotations

import pytest

from test_support.microcosm_build.us_input_batch_loops import (
    INPUT_LOOPS,
    POPULATION_AGGREGATE_PROBE_VARIABLE,
    PROBE_FAILURE_MESSAGE,
    BatchEngineRecorder,
    assert_same_bytes,
    engine_released,
    load_builder_module,
    medicaid_claim_outside_household_frame,
    real_engine_frame,
    spm_unit_spanning_households_frame,
)

LOOP_PARAMETERS = [pytest.param(loop, id=loop.name) for loop in INPUT_LOOPS]


@pytest.fixture(scope="module")
def builder():
    return load_builder_module()


def _options(batch_size, refuse_population_aggregates):
    options = {"maximum_microsim_batch_size": batch_size}
    if refuse_population_aggregates is not None:
        options["refuse_population_aggregates"] = refuse_population_aggregates
    return options


@pytest.mark.parametrize("loop", LOOP_PARAMETERS)
def test_input_loop_writes_mains_bytes_and_releases_each_batch_engine(
    builder, loop, monkeypatch
) -> None:
    """One batch and two batches of two households: the same arrays, dtypes
    and column order as main's loop, and each engine released before the
    next is built."""

    frame = real_engine_frame()
    for batch_size, expected_engines in ((None, 1), (2, 2)):
        expected = loop.run_main(builder, frame, maximum_microsim_batch_size=batch_size)
        recorder = BatchEngineRecorder(monkeypatch, period=builder.PERIOD)
        actual = loop.run(builder, frame, maximum_microsim_batch_size=batch_size)
        monkeypatch.undo()

        assert loop.reaches_engine_values(expected), loop.name
        assert_same_bytes(
            actual, expected, label=f"{loop.name}, batch size {batch_size}"
        )
        assert len(recorder.engines) == expected_engines
        assert all(recorder.earlier_engines_released_at_build)
        assert all(engine_released(engine) for engine in recorder.engines)


@pytest.mark.parametrize("loop", LOOP_PARAMETERS)
@pytest.mark.parametrize(
    ("probe", "batch_size", "refuse_population_aggregates", "batch_label"),
    [
        pytest.param("engine", 2, None, "1/2", id="aggregate-on-batch-engine"),
        pytest.param(
            "detached_branch", 2, None, "1/2", id="aggregate-on-dropped-branch"
        ),
        pytest.param("engine", None, True, "1/1", id="one-batch-refusal-switched-on"),
    ],
)
def test_input_loop_refuses_a_batch_engine_that_computed_a_population_aggregate(
    builder,
    loop,
    probe,
    batch_size,
    refuse_population_aggregates,
    batch_label,
    monkeypatch,
) -> None:
    recorder = BatchEngineRecorder(
        monkeypatch,
        period=builder.PERIOD,
        first_requested_variable=loop.first_requested_variable,
        probe=probe,
    )
    with pytest.raises(ValueError) as refusal:
        loop.run(
            builder,
            real_engine_frame(),
            **_options(batch_size, refuse_population_aggregates),
        )

    message = str(refusal.value)
    assert message.startswith(f"{loop.pass_label} is not batch-invariant"), message
    assert f"household batch {batch_label} computed " in message
    # The probe's variable reads the state average cost index too, so the
    # refusal lists both.
    assert f"{POPULATION_AGGREGATE_PROBE_VARIABLE}@{builder.PERIOD}" in message
    # The refusal stops the first batch, after its check and its release.
    assert len(recorder.engines) == 1
    assert engine_released(recorder.engines[0])


@pytest.mark.parametrize("loop", LOOP_PARAMETERS)
@pytest.mark.parametrize(
    ("batch_size", "refuse_population_aggregates"),
    [
        pytest.param(None, None, id="one-batch-by-default"),
        pytest.param(2, False, id="refusal-switched-off"),
    ],
)
def test_input_loop_keeps_an_aggregate_when_its_refusal_is_off(
    builder, loop, batch_size, refuse_population_aggregates, monkeypatch
) -> None:
    """One batch holds the whole pool, so the refusal is off by default; a
    caller may also switch it off. Engines are released either way."""

    recorder = BatchEngineRecorder(
        monkeypatch,
        period=builder.PERIOD,
        first_requested_variable=loop.first_requested_variable,
        probe="engine",
    )
    loop.run(
        builder,
        real_engine_frame(),
        **_options(batch_size, refuse_population_aggregates),
    )

    assert len(recorder.engines) == (1 if batch_size is None else 2)
    assert all(recorder.earlier_engines_released_at_build)
    assert all(engine_released(engine) for engine in recorder.engines)


@pytest.mark.parametrize("loop", LOOP_PARAMETERS)
def test_a_failing_input_batch_still_releases_its_engine(
    builder, loop, monkeypatch
) -> None:
    recorder = BatchEngineRecorder(
        monkeypatch,
        period=builder.PERIOD,
        first_requested_variable=loop.first_requested_variable,
        probe="failure",
    )
    with pytest.raises(RuntimeError, match=PROBE_FAILURE_MESSAGE):
        loop.run(builder, real_engine_frame(), maximum_microsim_batch_size=2)

    assert len(recorder.engines) == 1
    assert engine_released(recorder.engines[0])


@pytest.mark.parametrize("loop", LOOP_PARAMETERS)
@pytest.mark.parametrize(
    ("make_frame", "batch_size", "refuse_population_aggregates", "message"),
    [
        pytest.param(
            spm_unit_spanning_households_frame,
            2,
            None,
            "spm_unit units must be nested in households",
            id="spm-unit-spans-households",
        ),
        pytest.param(
            medicaid_claim_outside_household_frame,
            2,
            None,
            "medicaid_claiming_tax_unit_id does not resolve within the "
            "person's own household",
            id="medicaid-claim-outside-household",
        ),
        pytest.param(
            medicaid_claim_outside_household_frame,
            None,
            True,
            "medicaid_claiming_tax_unit_id does not resolve within the "
            "person's own household",
            id="medicaid-claim-outside-household-one-batch-refusal-on",
        ),
    ],
)
def test_input_loop_refuses_a_frame_that_household_batches_would_split(
    builder,
    loop,
    make_frame,
    batch_size,
    refuse_population_aggregates,
    message,
    monkeypatch,
) -> None:
    """Refused before any engine is built, as in target compilation."""

    recorder = BatchEngineRecorder(monkeypatch, period=builder.PERIOD)
    with pytest.raises(ValueError, match=message):
        loop.run(
            builder,
            make_frame(),
            **_options(batch_size, refuse_population_aggregates),
        )

    assert recorder.engines == []


def test_other_health_insurance_stage_writes_mains_tables(builder, monkeypatch) -> None:
    """The whole stage, in two batches of five households, writes the same
    tables as with main's premium loop: the modeled premiums on each tax
    unit's first person, the ASEC residual and the PUF-support imputation."""

    import microcosm.build.us_runtime.other_health_insurance as other_health_insurance
    from test_support.microcosm_build import (
        us_input_batch_loops_main_baseline as main_baseline,
    )
    from test_support.microcosm_build.us_other_health_insurance import (
        _frame as other_health_insurance_frame,
    )

    frame = other_health_insurance_frame()

    def run_stage():
        return other_health_insurance.with_us_other_health_insurance_inputs(
            frame,
            seed=0,
            time_period=builder.PERIOD,
            maximum_microsim_batch_size=5,
        )

    def main_premium_loop(frame, *, refuse_population_aggregates, **options):
        # Main's loop has no refusal; the stage passes its default through.
        assert refuse_population_aggregates is None
        return main_baseline.main_materialize_modeled_premiums(frame, **options)

    actual = run_stage()
    monkeypatch.setattr(
        other_health_insurance, "_materialize_modeled_premiums", main_premium_loop
    )
    expected = run_stage()

    # A modeled premium reaches the stage's output: the residual of a person
    # with a marketplace premium is below their reported premium.
    reported, residual = other_health_insurance.US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS
    person = expected.table("person")
    assert (person[residual] < person[reported]).any()
    assert list(actual.entities) == list(expected.entities)
    for entity in expected.entities:
        assert_same_bytes(actual.table(entity), expected.table(entity), label=entity)
