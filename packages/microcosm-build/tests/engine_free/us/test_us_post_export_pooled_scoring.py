"""Pooled post-export scoring (tracker task 1.4 · 7).

``_PooledPostExportScorer`` scores the requests the engine-free dry run
recorded in the shared worker pool and serves each consumer, when it runs
again in the parent, from the results. These tests run the shipped consumers
through it on the scoring tests' fake engine, in spawned workers, and require:

- every file post-export scoring writes, and the manifests' scoring block, to
  be byte-identical to the serial scorer's, at one and at three workers
  (``assert_identical_across_worker_counts``), with only the block's
  ``execution`` record (run metadata) apart;
- the serial scorer's refusals to happen in the workers, with the serial
  error's class name and text: a population aggregate in a multi-batch pass,
  a baseline-branch reader in a reform pass, a reform that moves a
  behavioral-response parameter;
- the merged baseline pass's failure to fall back to each consumer's own pass,
  as the serial scorer does, a changed batch file to stop scoring, and a
  request the dry run did not record to be refused;
- certification (``_certify_us_build``) to close the scorer when an error
  stops it during scoring (tracker task 1.4 · 8): the reform-coverage check's
  reform units run first, and its refusal, or any stage's error, kills the
  units still running and removes the batch files, while a failed check that
  certification goes on past cancels nothing.
"""

# The ``builder`` fixture is imported, and each test takes it by name.
# ruff: noqa: F401, F811
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import pytest

from microcosm.build.termination import BuildTerminatedError
from microcosm.build.us_runtime.post_export_requests import (
    RECORDS_REQUESTS_ONLY,
    ParameterChangesReform,
)
from microcosm.build.us_runtime.reform_validation import US_RELEASE_SPM_SELECTION
from microcosm.build.worker_pool import (
    WorkerPoolError,
    WorkerPoolSettings,
    WorkerPoolUnitError,
)
from microcosm.calibrate import TargetRegistry
from test_support.microcosm_build.us_post_export_pool import (
    COMPUTES,
    ENGINE_LOG,
    FAIL_ON,
    FAIL_ON_HOUSEHOLDS,
    MEMORY_KEYS,
    MEMORY_ON_HOUSEHOLDS,
    SLOW,
    WAIT_FOR,
    WAIT_FOR_OTHER_REFORMS,
    FakeMicrosimulation,
    build_fake_parameter_reform,
    build_fake_reform,
    engine_log,
    identity_dataset,
)
from test_support.microcosm_build.us_post_export_scoring import (
    ReformValidationSpec,
    _full_in_sample_fit,
    _nested_frame,
    _plan_args,
    _written_h5,
    builder,
    reform_validation_module,
    smoke_module,
)
from test_support.microcosm_build.worker_pool_identity import (
    assert_identical_across_worker_counts,
    canonical_result_bytes,
)

N_HOUSEHOLDS = 7
BATCH_SIZE = 3


@pytest.fixture
def pooled_reforms(monkeypatch, builder):
    """The parent builds the fake reforms the workers build from descriptions,
    and the demographics writer reads no H5 for its geography counts."""
    monkeypatch.setattr(smoke_module, "_build_reform", build_fake_reform)
    monkeypatch.setattr(ReformValidationSpec, "build_reform", build_fake_reform)
    monkeypatch.setattr(
        reform_validation_module, "_build_parameter_reform", build_fake_parameter_reform
    )
    monkeypatch.setattr(
        builder, "geography_coverage_payload", lambda path: {"h5": Path(path).name}
    )
    for name in (
        FAIL_ON,
        COMPUTES,
        ENGINE_LOG,
        MEMORY_KEYS,
        SLOW,
        WAIT_FOR,
        WAIT_FOR_OTHER_REFORMS,
        FAIL_ON_HOUSEHOLDS,
        MEMORY_ON_HOUSEHOLDS,
    ):
        monkeypatch.delenv(name, raising=False)


def _release_plan(builder):
    """The plan ``_main`` records for the nested frame, with a full fit."""
    fit = _full_in_sample_fit(builder.PERIOD)
    plan = builder._record_post_export_scoring_plan(
        _plan_args(),
        n_households=N_HOUSEHOLDS,
        result=fit,
        release_id="fixture-release",
    )
    assert plan.error is None
    return plan, fit


def _pooling(builder, workers: int, **settings):
    return builder._PostExportPooling(
        WorkerPoolSettings(workers=workers, **settings),
        chosen_by="test",
        microsimulation_cls=FakeMicrosimulation,
        dataset_from_frame=identity_dataset,
        reform_builder=build_fake_reform,
    )


def _load(frame):
    def load_frame(path, *, expected_sha256=None):
        return frame

    return load_frame


def _serial_scorer(builder, plan, directory: Path, *, batch_parent=None):
    return builder._HouseholdBatchedPostExportScorer(
        _written_h5(directory),
        maximum_microsim_batch_size=BATCH_SIZE,
        batch_parent=batch_parent,
        expected_n_households=N_HOUSEHOLDS,
        microsimulation_cls=FakeMicrosimulation,
        dataset_from_frame=identity_dataset,
        load_frame=_load(_nested_frame()),
        baseline_plans=plan.baseline_plans,
    )


def _pooled_scorer(
    builder, plan, directory: Path, workers: int, *, batch_parent=None, **settings
):
    return builder._PooledPostExportScorer(
        _written_h5(directory),
        maximum_microsim_batch_size=BATCH_SIZE,
        batch_parent=batch_parent,
        expected_n_households=N_HOUSEHOLDS,
        consumer_requests=plan.consumer_requests,
        pooling=_pooling(builder, workers, **settings),
        load_frame=_load(_nested_frame()),
    )


def _certify_outputs(builder, scorer, plan, fit, release_dir: Path):
    """Run every post-export stage as ``_certify_us_build`` does.

    Returns the files written and the manifests' scoring block, and, apart,
    that block's ``execution`` record (run metadata).
    """
    release_dir.mkdir(parents=True)
    try:
        smoke = scorer.open_consumer(
            "reform_coverage_smoke",
            plan.baseline_plan("reform_coverage_smoke"),
            planned_reform_passes=plan.planned_reform_passes("reform_coverage_smoke"),
        )
        gate = builder.us_reform_coverage_smoke_gate(
            simulate=smoke.simulate, period=builder.PERIOD
        )
        smoke_record = scorer.finish_consumer(smoke)
        (release_dir / "reform_coverage_smoke.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "enforced": True,
                    "reform_coverage_smoke": {
                        "passed": gate.passed,
                        "failures": list(gate.failures),
                        "details": dict(gate.details),
                    },
                    "post_export_scoring": smoke_record,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        builder._write_reform_validation(
            release_dir=release_dir,
            dataset_path=scorer.dataset_path,
            result=fit,
            registry=TargetRegistry((), country="us"),
            release_id="fixture-release",
            simulate_out_of_sample=True,
            post_export_scorer=scorer,
            baseline_plan=plan.baseline_plan("reform_validation"),
            planned_reform_passes=plan.planned_reform_passes("reform_validation"),
        )
        common = dict(
            release_dir=release_dir,
            dataset_path=scorer.dataset_path,
            release_id="fixture-release",
            post_export_scorer=scorer,
        )
        builder._write_demographics(
            **common, baseline_plan=plan.baseline_plan("demographics")
        )
        builder._write_ssi_social_security_holdout(
            **common, baseline_plan=plan.baseline_plan("ssi_social_security_holdout")
        )
        builder._write_medicare_irmaa_holdout(
            **common, baseline_plan=plan.baseline_plan("medicare_irmaa_holdout")
        )
        block = builder._post_export_scoring_manifest_block(scorer)
    finally:
        builder._close_post_export_scorer(scorer)
    execution = block.pop("execution")
    files = {path.name: path.read_bytes() for path in sorted(release_dir.iterdir())}
    return {"files": files, "block": block}, execution


def _reform_unit_count(plan) -> int:
    return sum(
        len(recorded.simulations) for recorded in plan.consumer_requests.values()
    )


def test_pooled_scoring_writes_the_serial_scorers_bytes_at_any_worker_count(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """Every output file and the manifests' block, scored in spawned workers,
    are byte-identical to the serial scorer's at one and three workers. Each
    worker engine is built as the serial scorer builds it: one batch at a
    time, released before the next, with the SPM selection, a reform's
    system alone, and its keys in ascending period order."""
    plan, fit = _release_plan(builder)
    serial, serial_execution = _certify_outputs(
        builder,
        _serial_scorer(builder, plan, tmp_path / "serial"),
        plan,
        fit,
        tmp_path / "serial" / "release",
    )
    assert serial_execution == {"mode": "serial"}
    assert set(serial["files"]) == {
        "demographics.json",
        "reform_coverage_smoke.json",
        "reform_validation.json",
        "us_medicare_irmaa_holdout.json",
        "us_ssi_social_security_holdout.json",
    }
    executions, logs = {}, {}

    def run(workers: int):
        directory = tmp_path / f"pooled-{workers}"
        logs[workers] = directory / "engines"
        logs[workers].mkdir(parents=True)
        monkeypatch.setenv(ENGINE_LOG, str(logs[workers]))
        outputs, executions[workers] = _certify_outputs(
            builder,
            _pooled_scorer(builder, plan, directory, workers),
            plan,
            fit,
            directory / "release",
        )
        return outputs

    pooled = assert_identical_across_worker_counts(run, worker_counts=(1, 3))
    assert canonical_result_bytes(pooled) == canonical_result_bytes(serial)

    reform_units = _reform_unit_count(plan)
    for workers, execution in executions.items():
        report = execution.pop("report")
        assert execution == {
            "mode": "pooled",
            "chosen_by": "test",
            "requested_workers": workers,
            "workers": workers,
            "memory_cap_bytes": None,
            "max_units_per_worker": None,
            "max_attempts": 2,
            "max_units_ahead": None,
            "worker_environment_variables": [],
            "units": {"baseline": workers, "reform": reform_units},
            "resources": None,
            "merged_baseline_fallback": None,
        }
        assert report["units_completed"] == workers + reform_units
        assert report["failed_attempts"] == []
        assert 1 <= report["max_live_workers"] <= workers

    events = engine_log(logs[3])
    constructions = [event for event in events if event["event"] == "construct"]
    assert os.getpid() not in {event["pid"] for event in events}
    assert len({event["pid"] for event in constructions}) > 1
    assert all(
        event["spm"] == dict(US_RELEASE_SPM_SELECTION) for event in constructions
    )
    assert all(not event["reform_argument"] for event in constructions)
    assert all(event["unreleased"] == 0 for event in constructions)
    assert all(len(event["households"]) <= BATCH_SIZE for event in constructions)
    periods = defaultdict(list)
    for event in events:
        if event["event"] == "calculate":
            periods[(event["pid"], event["engine"])].append(event["period"])
    assert periods and all(found == sorted(found) for found in periods.values())


def _fixture_consumer(asks):
    """A consumer that reads the baseline, opens one reformed simulation per
    description in ``asks`` and asks it for its keys."""

    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True
        baseline = simulate(None)
        totals = [baseline.calculate("age", 2024).sum()]
        for description, keys in asks(totals[0]):
            reformed = simulate(
                description if records_only else build_fake_reform(description)
            )
            for variable, period in keys:
                totals.append(reformed.calculate(variable, period).sum())
        return totals

    return consumer


def _fixture_plan(builder, consumer):
    return builder._post_export_scoring_plan(
        n_households=N_HOUSEHOLDS,
        maximum_microsim_batch_size=BATCH_SIZE,
        consumers={"fixture": consumer},
    )


FIXTURE_REFORM = ParameterChangesReform({"gov.irs.fixture": {"2024-01-01": 1}})
OTHER_REFORM = ParameterChangesReform({"gov.irs.other": {"2024-01-01": 2}})


def test_a_reform_request_the_dry_run_did_not_record_is_refused(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    """The dry run sees zeros; a consumer that then asks for another key
    would be served values scored for no recorded request."""

    def asks(total):
        return [(FIXTURE_REFORM, [("income_tax" if total == 0 else "snap", 2024)])]

    plan = _fixture_plan(builder, _fixture_consumer(asks))
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        with pytest.raises(
            RuntimeError, match="not the next reform pass its engine-free dry run"
        ):
            _fixture_consumer(asks)(scoring.simulate)
    finally:
        scorer.close()


def test_a_consumer_that_skips_a_recorded_reform_pass_is_refused(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    def asks(total):
        first = [(FIXTURE_REFORM, [("income_tax", 2024)])]
        return first + [(OTHER_REFORM, [("snap", 2024)])] if total == 0 else first

    consumer = _fixture_consumer(asks)
    plan = _fixture_plan(builder, consumer)
    assert plan.planned_reform_passes("fixture") == 2
    scorer = _pooled_scorer(builder, plan, tmp_path, 2)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        consumer(scoring.simulate)
        with pytest.raises(RuntimeError, match=r"reformed simulation\(s\) \[1\]"):
            scorer.finish_consumer(scoring)
    finally:
        scorer.close()


def test_a_baseline_key_outside_the_plan_is_refused(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    def consumer(simulate):
        baseline = simulate(None)
        total = baseline.calculate("age", 2024).sum()
        return baseline.calculate("age" if total == 0 else "snap", 2024).sum()

    plan = _fixture_plan(builder, consumer)
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        with pytest.raises(RuntimeError, match="which its recorded plan does not hold"):
            consumer(scoring.simulate)
    finally:
        scorer.close()


def test_only_recorded_consumers_open_each_once(
    builder, pooled_reforms, tmp_path
) -> None:
    plan = _fixture_plan(builder, _fixture_consumer(lambda total: []))
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        with pytest.raises(ValueError, match="has no recorded requests"):
            scorer.open_consumer("demographics", ())
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        scorer.finish_consumer(scoring)
        with pytest.raises(ValueError, match="opened on this scorer already"):
            scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
    finally:
        scorer.close()


RESPONSE_REFORM = ParameterChangesReform(
    {"gov.simulation.labor_supply_responses.elasticities.income": {"2024-01-01": 0.25}}
)


@pytest.mark.parametrize(
    ("description", "variable", "computes", "counts"),
    [
        # A population aggregate, computed in a multi-batch pass.
        (
            FIXTURE_REFORM,
            "medicaid",
            {"medicaid": ["medicaid_slcsp_state_average_cost_index"]},
            (1, 1),
        ),
        # A baseline-branch reader, computed by a reform engine.
        (
            FIXTURE_REFORM,
            "income_tax",
            {"income_tax": ["relative_income_change"]},
            (1, 1),
        ),
        # A reform that moves a behavioral-response parameter: refused before
        # its pass, as the serial scorer refuses it.
        (RESPONSE_REFORM, "income_tax", {}, (1, 0)),
    ],
    ids=["population_aggregate", "baseline_branch_reader", "behavioral_response"],
)
def test_the_workers_refuse_what_the_serial_scorer_refuses(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    description,
    variable,
    computes,
    counts,
) -> None:
    """The refusal reaches the consumer as the serial scorer's error, with its
    class name and text, and the consumer's counts move as the serial
    scorer's: (reform systems, reform passes)."""
    monkeypatch.setenv(COMPUTES, json.dumps(computes))

    def asks(total):
        return [(description, [(variable, 2024)])]

    consumer = _fixture_consumer(asks)
    plan = _fixture_plan(builder, consumer)
    errors, observed = {}, {}
    for mode in ("serial", "pooled"):
        scorer = (
            _serial_scorer(builder, plan, tmp_path / mode)
            if mode == "serial"
            else _pooled_scorer(builder, plan, tmp_path / mode, 1)
        )
        try:
            scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
            with pytest.raises(RuntimeError) as raised:
                consumer(scoring.simulate)
            errors[mode] = raised.value
            observed[mode] = (scoring.reform_systems, scoring.reform_passes)
        finally:
            scorer.close()
    assert type(errors["pooled"]).__name__ == type(errors["serial"]).__name__
    assert str(errors["pooled"]) == str(errors["serial"])
    assert observed == {"serial": counts, "pooled": counts}
    assert "Traceback" in errors["pooled"].worker_traceback


def test_a_failed_merged_pass_falls_back_as_the_serial_scorer_does(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """A key only the Medicare holdout planned fails the merged pass; every
    consumer then scores its own plan, and only that holdout records itself
    unavailable, with the error the serial scorer records."""
    monkeypatch.setenv(FAIL_ON, "medicare_enrolled")
    plan, fit = _release_plan(builder)
    serial, _ = _certify_outputs(
        builder,
        _serial_scorer(builder, plan, tmp_path / "serial"),
        plan,
        fit,
        tmp_path / "serial" / "release",
    )
    capsys.readouterr()
    pooled, execution = _certify_outputs(
        builder,
        _pooled_scorer(builder, plan, tmp_path / "pooled", 2),
        plan,
        fit,
        tmp_path / "pooled" / "release",
    )
    assert "merged baseline plan failed" in capsys.readouterr().out
    assert canonical_result_bytes(pooled) == canonical_result_bytes(serial)
    holdout = json.loads(pooled["files"]["us_medicare_irmaa_holdout.json"])
    assert holdout["status"] == "unavailable"
    assert holdout["error"] == "RuntimeError: fixture engine refuses medicare_enrolled"
    planned = [name for name, keys in plan.baseline_plans.items() if keys]
    assert execution["report"]["units_completed"] == (
        2 + 2 * len(planned) + _reform_unit_count(plan)
    )


def test_a_changed_batch_file_stops_pooled_scoring(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    plan, _ = _release_plan(builder)
    scorer = _pooled_scorer(builder, plan, tmp_path, 2)
    try:
        path, _ = scorer._batches().files[1]
        Path(path).write_bytes(Path(path).read_bytes() + b"changed")
        with pytest.raises(builder.PostExportBatchFileError, match="has SHA-256"):
            scorer.open_consumer(
                "reform_coverage_smoke", plan.baseline_plan("reform_coverage_smoke")
            )
        # Nothing more is served, and no consumer falls back to its own pass:
        # a pool that could not start refuses every later consumer with a
        # WorkerPoolError, which the held-out reports pass on.
        with pytest.raises(builder.PostExportPoolStoppedError, match="could not start"):
            scorer.open_consumer(
                "reform_validation", plan.baseline_plan("reform_validation")
            )
    finally:
        scorer.close()


def test_a_worker_batch_file_failure_is_raised_as_the_tools_own_error(builder) -> None:
    from microcosm.build.us_runtime.post_export_pool import PostExportUnitFailure

    try:
        raise builder.PostExportBatchFileError("batch 2 changed")
    except builder.PostExportBatchFileError as error:
        failure = builder._post_export_unit_failure("scoring", error)
    assert failure.batch_file_error
    with pytest.raises(builder.PostExportBatchFileError, match="batch 2 changed"):
        builder._raise_post_export_worker_failure(failure)
    other = PostExportUnitFailure.from_exception("scoring", KeyError("snap"))
    with pytest.raises(KeyError) as raised:
        builder._raise_post_export_worker_failure(other)
    assert isinstance(raised.value.__cause__, builder._PostExportWorkerTracebackError)


@pytest.mark.parametrize(
    "writer", ["_write_ssi_social_security_holdout", "_write_medicare_irmaa_holdout"]
)
def test_the_holdouts_stop_on_a_pool_failure(builder, tmp_path, writer) -> None:
    """A pool that cannot score the H5 is not an unavailable measurement."""

    class FailingScorer:
        def open_consumer(self, *args, **kwargs):
            raise WorkerPoolError("a worker could not be started")

    with pytest.raises(WorkerPoolError, match="could not be started"):
        getattr(builder, writer)(
            release_dir=tmp_path,
            dataset_path=tmp_path / "populace_us_2024.h5",
            release_id="fixture-release",
            post_export_scorer=FailingScorer(),
            baseline_plan=(("age", builder.PERIOD, None),),
        )
    assert not list(tmp_path.iterdir())


class _ParentOnly:
    """A value whose class a spawned worker cannot import."""


def test_a_reform_a_worker_cannot_unpickle_stops_the_run_and_names_its_unit(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    module = types.ModuleType("parent_only_post_export_values")
    _ParentOnly.__module__ = module.__name__
    module._ParentOnly = _ParentOnly
    monkeypatch.setitem(sys.modules, module.__name__, module)
    description = ParameterChangesReform(
        {"gov.irs.fixture": {"2024-01-01": _ParentOnly()}}
    )

    def asks(total):
        return [(description, [("income_tax", 2024)])]

    consumer = _fixture_consumer(asks)
    plan = _fixture_plan(builder, consumer)
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        with pytest.raises(
            WorkerPoolUnitError, match=r"fixture reformed simulation 0 .*all 3 batch"
        ):
            consumer(scoring.simulate)
    finally:
        scorer.close()


def test_the_pool_starts_no_more_workers_than_units(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    plan = _fixture_plan(builder, _fixture_consumer(lambda total: []))
    scorer = _pooled_scorer(builder, plan, tmp_path, 8)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        scorer.finish_consumer(scoring)
        execution = scorer.execution_record()
    finally:
        scorer.close()
    assert execution["units"] == {"baseline": 3, "reform": 0}
    assert (execution["requested_workers"], execution["workers"]) == (8, 3)
    assert execution["report"]["max_live_workers"] <= 3


def test_workers_get_the_lock_check_record_and_the_set_environment(
    builder,
    monkeypatch,
) -> None:
    from microcosm.build.us_runtime import take_up_contract

    monkeypatch.setattr(
        take_up_contract,
        "us_engine_abi_lock_check_environment",
        lambda: {"MICROCOSM_ENGINE_ABI_LOCK_CHECK_RECORD": "fixture record"},
    )
    pooling = _pooling(
        builder, 6, worker_environment={"OMP_NUM_THREADS": "1"}, memory_cap_bytes=2**30
    )
    settings = builder._post_export_worker_settings(pooling, workers=4)
    assert settings.workers == 4
    assert settings.memory_cap_bytes == 2**30
    assert settings.worker_environment == {
        "OMP_NUM_THREADS": "1",
        "MICROCOSM_ENGINE_ABI_LOCK_CHECK_RECORD": "fixture record",
    }
    assert builder._post_export_worker_settings(pooling, workers=9).workers == 6
    monkeypatch.setattr(
        take_up_contract, "us_engine_abi_lock_check_environment", lambda: {}
    )
    inherited = builder._post_export_worker_settings(_pooling(builder, 2), workers=2)
    assert inherited.worker_environment is None


def test_the_plan_records_each_consumers_requests_and_passes(builder) -> None:
    """The reform passes hold every reform request, in order, and the
    pre-OBBBA state's two keys are one pass, as the serial scorer runs it."""
    plan, _ = _release_plan(builder)
    assert set(plan.consumer_requests) == set(plan.baseline_plans)
    for name, recorded in plan.consumer_requests.items():
        assert recorded.baseline_plan == plan.baseline_plans[name]
        assert len(recorded.reform_passes) == plan.planned_reform_passes(name)
        reform_keys = [
            (request.reformed_simulation, request.key)
            for request in recorded.descriptors
            if request.reform is not None
        ]
        assert reform_keys == [
            (simulation, key)
            for simulation, keys in recorded.reform_passes
            for key in keys
        ]
    validation = plan.consumer_requests["reform_validation"]
    assert max(len(keys) for _, keys in validation.reform_passes) == 2
    assert "consumer_requests" not in json.dumps(plan.record())


def test_the_batch_ranges_cover_every_batch_once(builder) -> None:
    for n_batches in (1, 2, 3, 7, 177):
        for ranges in (1, 2, 4, 17, 200):
            split = builder._post_export_batch_ranges(n_batches, ranges)
            assert [position for batches in split for position in batches] == list(
                range(n_batches)
            )
            assert len(split) == min(ranges, n_batches)
            sizes = {len(batches) for batches in split}
            assert max(sizes) - min(sizes) <= 1


# ---------------------------------------------------------------------------
# Pooled scoring is certification's default (1.4 · 6's settings, approved by
# Anthony on 27 November)
# ---------------------------------------------------------------------------

GIB = 2**30
WORKERS_VARIABLE = "MICROCOSM_POST_EXPORT_WORKERS"


def test_the_environment_chooses_pooled_or_serial_scoring(builder, capsys) -> None:
    resources = {"total_memory_bytes": 32 * GIB, "usable_cores": 10}
    pooling = builder._post_export_pooling_from_environment({}, **resources)
    assert pooling.chosen_by == "formula"
    settings = pooling.settings
    assert (settings.workers, settings.max_units_per_worker) == (4, 2)
    assert settings.memory_cap_bytes == 21 * GIB // 4
    assert settings.worker_environment is None
    assert pooling.resources["planned_peak_bytes"] == 23 * GIB
    assert "4 pool workers" in capsys.readouterr().out
    named = builder._post_export_pooling_from_environment(
        {WORKERS_VARIABLE: "6"}, **resources
    )
    assert (named.settings.workers, named.chosen_by) == (6, f"{WORKERS_VARIABLE}=6")
    assert (
        builder._post_export_pooling_from_environment(
            {WORKERS_VARIABLE: "serial"}, **resources
        )
        is None
    )
    assert "in this process" in capsys.readouterr().out
    for environment, machine in (
        ({WORKERS_VARIABLE: "lots"}, resources),
        ({}, {"total_memory_bytes": 8 * GIB, "usable_cores": 10}),
    ):
        with pytest.raises(SystemExit, match=WORKERS_VARIABLE):
            builder._post_export_pooling_from_environment(environment, **machine)


def test_the_execution_record_holds_the_plan_it_ran(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    plan = _fixture_plan(builder, _fixture_consumer(lambda total: []))
    pooling = builder._PostExportPooling(
        WorkerPoolSettings(workers=2),
        chosen_by="formula",
        microsimulation_cls=FakeMicrosimulation,
        dataset_from_frame=identity_dataset,
        reform_builder=build_fake_reform,
        resources={"usable_cores": 3},
    )
    scorer = builder._PooledPostExportScorer(
        _written_h5(tmp_path),
        maximum_microsim_batch_size=BATCH_SIZE,
        consumer_requests=plan.consumer_requests,
        pooling=pooling,
        load_frame=_load(_nested_frame()),
    )
    scorer.close()
    record = scorer.execution_record()
    assert (record["chosen_by"], record["resources"]) == (
        "formula",
        {"usable_cores": 3},
    )


def test_certification_chooses_its_workers_before_it_places_the_build(
    builder,
) -> None:
    """A bad setting or a machine too small for one worker stops
    certification before it copies the build's files or loads the H5, and
    the pool settings reach the post-export scorer."""
    import ast
    import inspect

    source = inspect.getsource(builder._certify_main_with_close_out)
    chosen = source.index(
        "post_export_pooling = _post_export_pooling_from_environment()"
    )
    assert chosen < source.index("_place_build_for_certification(")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_certify_us_build"
    ]
    assert [ast.unparse(call) for call in calls] == [
        "_certify_us_build(record, certification=certification, "
        "telemetry=telemetry, post_export_pooling=post_export_pooling)"
    ]


def test_the_probe_command_scores_as_certification_does(builder) -> None:
    """The export probe's command reads the same setting, before it loads
    the export, so it times pooled scoring by default."""
    import ast
    import inspect

    from test_support.microcosm_build.us_export_subsample import _load_tool

    probe_tool = _load_tool("probe_us_post_export", "probe_us_post_export.py")
    source = inspect.getsource(probe_tool.main)
    chosen = source.index("post_export_pooling = post_export_pooling_for(stages)")
    assert chosen < source.index("report = probe_export(")
    # Only a run with an engine stage reads the setting, so a probe of the
    # frame-based stages alone runs on any machine.
    calls = []

    class Recording:
        def _post_export_pooling_from_environment(self):
            calls.append(True)
            return "pooling"

    assert (
        probe_tool.post_export_pooling_for(
            ("stored_inputs", "qrf_tail_concentration"), Recording()
        )
        is None
    )
    assert calls == []
    assert probe_tool.post_export_pooling_for(("demographics",), Recording()) == (
        "pooling"
    )
    (call,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "probe_export"
    ]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["post_export_pooling"] == "post_export_pooling"


# ---------------------------------------------------------------------------
# Review round 1
# ---------------------------------------------------------------------------


def _reform_consumer(*opened):
    """A consumer that reads the baseline, then opens each reform of the
    dry run's order (``opened(total)`` picks the real run's order) and asks
    each for income tax."""

    (choose,) = opened

    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True
        total = simulate(None).calculate("age", 2024).sum()
        totals = []
        for description in choose(total):
            reformed = simulate(
                description if records_only else build_fake_reform(description)
            )
            totals.append(reformed.calculate("income_tax", 2024).sum())
        return totals

    return consumer


def test_a_reform_other_than_the_recorded_one_is_refused(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    """The dry run opened A then B; the real run opens B then A with the
    same keys. Served by position alone, B would get A's values."""

    def opened(total):
        return (
            [FIXTURE_REFORM, OTHER_REFORM]
            if total == 0
            else [OTHER_REFORM, FIXTURE_REFORM]
        )

    consumer = _reform_consumer(opened)
    plan = _fixture_plan(builder, consumer)
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        with pytest.raises(
            RuntimeError, match="reformed simulation 0 with a reform other than"
        ):
            consumer(scoring.simulate)
    finally:
        scorer.close()


def test_a_reform_without_its_description_or_beyond_the_recorded_ones_is_refused(
    builder,
    pooled_reforms,
    tmp_path,
) -> None:
    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True
        simulate(None).calculate("age", 2024)
        if records_only:
            simulate(FIXTURE_REFORM).calculate("income_tax", 2024)
        return simulate

    plan = _fixture_plan(builder, consumer)
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        with pytest.raises(RuntimeError, match="carries no description"):
            scoring.simulate(object())
        built = build_fake_reform(FIXTURE_REFORM)
        with pytest.raises(RuntimeError, match="more reformed simulations"):
            scoring.simulate(built)
    finally:
        scorer.close()


def _baseline_consumer(*variables):
    def consumer(simulate):
        baseline = simulate(None)
        return [baseline.calculate(variable, 2024).sum() for variable in variables]

    return consumer


def test_a_merged_pass_a_worker_cannot_finish_falls_back_as_the_serial_scorer_does(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """An engine runs out of memory on the merged pass's three keys but not
    on either consumer's two. The serial scorer catches the MemoryError and
    scores each consumer's own plan; so does the pool, once the merged unit
    has failed on every attempt."""
    monkeypatch.setenv(MEMORY_KEYS, "2")
    consumers = {
        "first": _baseline_consumer("age", "snap"),
        "second": _baseline_consumer("age", "income_tax"),
    }
    plan = builder._post_export_scoring_plan(
        n_households=N_HOUSEHOLDS,
        maximum_microsim_batch_size=BATCH_SIZE,
        consumers=consumers,
    )

    executions = []

    def run(scorer):
        try:
            outputs = {}
            for name, consumer in consumers.items():
                scoring = scorer.open_consumer(name, plan.baseline_plan(name))
                outputs[name] = consumer(scoring.simulate)
                scorer.finish_consumer(scoring)
            block = scorer.manifest_record()
        finally:
            scorer.close()
        executions.append(block.pop("execution"))
        return outputs, block

    serial = run(_serial_scorer(builder, plan, tmp_path / "serial"))
    assert "merged baseline plan failed" in capsys.readouterr().out
    pooled = run(_pooled_scorer(builder, plan, tmp_path / "pooled", 1))
    assert "merged baseline plan failed" in capsys.readouterr().out
    assert canonical_result_bytes(pooled) == canonical_result_bytes(serial)
    # The record keeps the failed pool's report: the attempts that failed and
    # the workers they started, with the fallback's units.
    execution = executions[1]
    assert execution["merged_baseline_fallback"].startswith("WorkerPoolUnitError")
    assert execution["units"] == {"baseline": 1, "reform": 0, "fallback_baseline": 2}
    report = execution["report"]
    assert report["pools"] == 2
    assert len(report["failed_attempts"]) == 2
    assert report["worker_starts"] >= 3
    assert report["units_completed"] == 2


def test_a_unit_the_pool_cannot_finish_fails_its_own_consumer_alone(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """A reform unit that fails on every attempt fails the consumer that asks
    for it, with the pool's error, which the held-out reports pass on instead
    of recording an unavailable measurement. The pool then goes on in fresh
    workers: a consumer served from the merged pass and one with a reform
    unit of its own are served what the serial scorer serves them."""
    monkeypatch.setenv(MEMORY_KEYS, "2")

    def prefetching(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True
        simulate(None).calculate("age", 2024)
        reformed = simulate(
            FIXTURE_REFORM if records_only else build_fake_reform(FIXTURE_REFORM)
        )
        reformed.prefetch((("income_tax", 2024), ("snap", 2024), ("ssi", 2024)))
        return reformed.calculate("income_tax", 2024).sum()

    consumers = {
        "first": prefetching,
        "second": _baseline_consumer("age"),
        "third": _fixture_consumer(
            lambda total: [(OTHER_REFORM, [("income_tax", 2024)])]
        ),
    }
    plan = builder._post_export_scoring_plan(
        n_households=N_HOUSEHOLDS,
        maximum_microsim_batch_size=BATCH_SIZE,
        consumers=consumers,
    )
    serial = _serial_scorer(builder, plan, tmp_path / "serial")
    try:
        with pytest.raises(MemoryError):
            prefetching(
                serial.open_consumer("first", plan.baseline_plan("first")).simulate
            )
        expected = {
            name: consumers[name](
                serial.open_consumer(name, plan.baseline_plan(name)).simulate
            )
            for name in ("second", "third")
        }
    finally:
        serial.close()
    scorer = _pooled_scorer(builder, plan, tmp_path / "pooled", 1)
    try:
        first = scorer.open_consumer("first", plan.baseline_plan("first"))
        first.simulate(None).calculate("age", 2024)
        reformed = first.simulate(build_fake_reform(FIXTURE_REFORM))
        keys = (("income_tax", 2024), ("snap", 2024), ("ssi", 2024))
        with pytest.raises(WorkerPoolUnitError, match="MemoryError") as failed:
            reformed.prefetch(keys)
        # Asked again, the unit's own error again, not a refusal of the order.
        with pytest.raises(WorkerPoolUnitError) as again:
            reformed.prefetch(keys)
        assert str(again.value) == str(failed.value)
        for name in ("second", "third"):
            scoring = scorer.open_consumer(name, plan.baseline_plan(name))
            assert consumers[name](scoring.simulate) == expected[name], name
            scorer.finish_consumer(scoring)
        report = scorer.execution_record()["report"]
    finally:
        scorer.close()
    assert report["pools"] == 2
    assert len(report["failed_attempts"]) == 2
    assert [attempt["pool"] for attempt in report["failed_attempts"]] == [0, 0]


def test_a_later_units_failure_waits_for_its_own_consumer(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """Workers run ahead of the consumers. A later consumer's unit that fails
    on every attempt while an earlier unit is still running is that later
    consumer's failure, not the earlier one's: the earlier unit's engine
    waits until the later unit has failed twice, then runs again in a fresh
    pool, and the stored error holds none of the stopped run's frames."""
    import threading

    monkeypatch.setenv(MEMORY_KEYS, "2")

    def prefetching(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True
        simulate(None).calculate("age", 2024)
        reformed = simulate(
            OTHER_REFORM if records_only else build_fake_reform(OTHER_REFORM)
        )
        reformed.prefetch((("income_tax", 2024), ("ssi", 2024), ("wic", 2024)))
        return reformed.calculate("income_tax", 2024).sum()

    consumers = {
        "slow": _fixture_consumer(lambda total: [(FIXTURE_REFORM, [("snap", 2024)])]),
        "failing": prefetching,
    }
    plan = builder._post_export_scoring_plan(
        n_households=N_HOUSEHOLDS,
        maximum_microsim_batch_size=BATCH_SIZE,
        consumers=consumers,
    )
    serial = _serial_scorer(builder, plan, tmp_path / "serial")
    try:
        expected = consumers["slow"](
            serial.open_consumer("slow", plan.baseline_plan("slow")).simulate
        )
    finally:
        serial.close()
    released = tmp_path / "released"
    monkeypatch.setenv(WAIT_FOR, json.dumps({"snap": str(released)}))
    scorer = _pooled_scorer(builder, plan, tmp_path / "pooled", 2)
    stop = threading.Event()
    try:
        slow = scorer.open_consumer("slow", plan.baseline_plan("slow"))
        first_pool = scorer._pool

        def release_once_the_later_unit_failed():
            while not stop.wait(0.2):
                if len(first_pool.report().failed_attempts) >= 2:
                    released.touch()
                    return

        watcher = threading.Thread(target=release_once_the_later_unit_failed)
        watcher.start()
        assert consumers["slow"](slow.simulate) == expected
        scorer.finish_consumer(slow)
        assert all(
            error.__traceback__ is None for error in scorer._unit_errors.values()
        )
        failing = scorer.open_consumer("failing", plan.baseline_plan("failing"))
        with pytest.raises(WorkerPoolUnitError, match="failing reformed simulation 0"):
            prefetching(failing.simulate)
        report = scorer.execution_record()["report"]
    finally:
        stop.set()
        scorer.close()
    assert released.exists()
    assert report["pools"] == 2


def test_a_pool_that_cannot_start_refuses_every_consumer(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """Workers start only with the parent's release tool: given another
    digest, none starts, and every consumer is refused with a WorkerPoolError
    that names the refusal."""
    monkeypatch.setattr(builder, "_RELEASE_TOOL_SHA256", "0" * 64)
    plan = builder._post_export_scoring_plan(
        n_households=N_HOUSEHOLDS,
        maximum_microsim_batch_size=BATCH_SIZE,
        consumers={
            "first": _baseline_consumer("age"),
            "second": _baseline_consumer("snap"),
        },
    )
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        with pytest.raises(WorkerPoolError, match="SHA-256"):
            scorer.open_consumer("first", plan.baseline_plan("first"))
        with pytest.raises(builder.PostExportPoolStoppedError):
            scorer.open_consumer("second", plan.baseline_plan("second"))
    finally:
        scorer.close()


def test_asking_again_after_a_failed_pass_raises_what_the_serial_scorer_raises(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(COMPUTES, json.dumps({"income_tax": ["relative_income_change"]}))

    def consumer(simulate):
        records_only = getattr(simulate, RECORDS_REQUESTS_ONLY, False) is True
        simulate(None).calculate("age", 2024)
        reformed = simulate(
            FIXTURE_REFORM if records_only else build_fake_reform(FIXTURE_REFORM)
        )
        errors = []
        for period in (2024, 2024, 2025, 2024):
            try:
                reformed.calculate("income_tax", period)
            except RuntimeError as error:
                errors.append(f"{type(error).__name__}: {error}")
        return errors

    plan = _fixture_plan(builder, consumer)
    observed = {}
    for mode in ("serial", "pooled"):
        scorer = (
            _serial_scorer(builder, plan, tmp_path / mode)
            if mode == "serial"
            else _pooled_scorer(builder, plan, tmp_path / mode, 1)
        )
        try:
            scoring = scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
            errors = consumer(scoring.simulate)
            observed[mode] = (errors, scoring.reform_systems, scoring.reform_passes)
        finally:
            scorer.close()
    assert len(observed["serial"][0]) == 4
    assert observed["pooled"] == observed["serial"]


def test_the_execution_record_names_the_environment_workers_got(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    from microcosm.build.us_runtime import take_up_contract

    monkeypatch.setattr(
        take_up_contract,
        "us_engine_abi_lock_check_environment",
        lambda: {"MICROCOSM_TEST_HANDED_DOWN": "handed down"},
    )
    plan = _fixture_plan(builder, _fixture_consumer(lambda total: []))
    scorer = _pooled_scorer(builder, plan, tmp_path, 1)
    try:
        scorer.finish_consumer(
            scorer.open_consumer("fixture", plan.baseline_plan("fixture"))
        )
        execution = scorer.execution_record()
    finally:
        scorer.close()
    assert execution["worker_environment_variables"] == ["MICROCOSM_TEST_HANDED_DOWN"]


def test_the_consumers_builders_mark_each_reform_with_its_description(
    monkeypatch,
) -> None:
    """The smoke check's and reform validation's own builders mark every
    reform they build with the description they built it from, which pooled
    scoring compares with the reform its dry run recorded."""
    from microcosm.build.us_runtime.post_export_requests import reform_description

    class StubReform:
        @classmethod
        def from_dict(cls, changes, country_id):
            return type("FromDict", (cls,), {"changes": dict(changes)})

    core = types.ModuleType("policyengine_core")
    reforms = types.ModuleType("policyengine_core.reforms")
    reforms.Reform = StubReform
    core.reforms = reforms
    monkeypatch.setitem(sys.modules, "policyengine_core", core)
    monkeypatch.setitem(sys.modules, "policyengine_core.reforms", reforms)
    from microcosm.build.us_runtime.release_input_coverage import (
        us_release_reform_coverage_probes,
    )

    probes = us_release_reform_coverage_probes()
    neutralizing = next(probe for probe in probes if probe.neutralized_variable)
    changing = next(
        probe
        for probe in probes
        if not probe.neutralized_variable and not probe.list_edits
    )
    for probe in (neutralizing, changing):
        assert reform_description(smoke_module._build_reform(probe)) == probe
    specs = reform_validation_module.load_default_reform_specs(period=2024)
    for spec in (
        next(spec for spec in specs if spec.neutralized_variable),
        next(spec for spec in specs if spec.parameter_changes),
    ):
        assert reform_description(spec.build_reform()) == spec
    changes = {"gov.irs.fixture": {"2024-01-01": 1}}
    assert reform_description(
        reform_validation_module._build_parameter_reform(changes)
    ) == ParameterChangesReform(changes)
    # Two builds are two reforms, each with its own description.
    assert smoke_module._build_reform(changing) is not smoke_module._build_reform(
        changing
    )


def test_workers_check_they_run_the_parents_release_tool(builder) -> None:
    import hashlib

    tool = Path(builder.__file__)
    assert builder._RELEASE_TOOL_SHA256 == hashlib.sha256(tool.read_bytes()).hexdigest()


def test_a_stage_own_baseline_raises_its_first_problem_in_batch_order(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """After the merged pass fails, a stage's own baseline pass that raises in
    its first batch raises that error, as the serial scorer does, even when a
    later range of it also fails on every attempt in the pool."""
    monkeypatch.setenv(FAIL_ON_HOUSEHOLDS, json.dumps({"snap": [1]}))
    monkeypatch.setenv(MEMORY_ON_HOUSEHOLDS, json.dumps({"snap": [7]}))
    consumers = {
        "first": _baseline_consumer("age", "income_tax"),
        "second": _baseline_consumer("age", "snap"),
    }
    plan = builder._post_export_scoring_plan(
        n_households=N_HOUSEHOLDS,
        maximum_microsim_batch_size=BATCH_SIZE,
        consumers=consumers,
    )

    def run(scorer):
        observed = {}
        try:
            for name, consumer in consumers.items():
                try:
                    scoring = scorer.open_consumer(name, plan.baseline_plan(name))
                except (RuntimeError, MemoryError) as error:
                    observed[name] = f"{type(error).__name__}: {error}"
                    continue
                observed[name] = consumer(scoring.simulate)
        finally:
            scorer.close()
        return observed

    monkeypatch.setenv(MEMORY_ON_HOUSEHOLDS, "{}")
    serial = run(_serial_scorer(builder, plan, tmp_path / "serial"))
    assert serial["second"] == "RuntimeError: fixture engine refuses snap"
    monkeypatch.setenv(MEMORY_ON_HOUSEHOLDS, json.dumps({"snap": [7]}))
    assert run(_pooled_scorer(builder, plan, tmp_path / "pooled", 2)) == serial


# ---------------------------------------------------------------------------
# Certification stops the pool when scoring stops it (tracker task 1.4 · 8)
# ---------------------------------------------------------------------------


class _ScoringFinishedError(Exception):
    """Raised in place of the population-structure holdout, which certification
    writes after it has closed the scorer, so a test ends there."""


def _use_probes(monkeypatch, *, passing: bool):
    """Three of the shipped reform-coverage probes, each made to pass or to
    fail on the fake engine, in place of the shipped set: the plan's dry run,
    the check in the certifying process and the workers' units all use them."""
    probes = tuple(
        dataclasses.replace(
            probe,
            expected_sign="either",
            min_abs_effect=1e-6 if passing else 1e18,
        )
        for probe in smoke_module.us_release_reform_coverage_probes()[:3]
    )
    monkeypatch.setattr(
        smoke_module, "us_release_reform_coverage_probes", lambda: probes
    )
    return probes


def _record_for_certification(builder, plan, fit, tmp_path: Path, **settings):
    """What ``_certify_us_build`` reads of a build record, for the fixture plan.

    The record's options are the build's: every stage on, the fixture's
    batching, and ``settings`` on top. Everything the manifests alone read is
    empty, since these tests end before the manifests.
    """
    artifact_root = tmp_path / "build" / "artifacts"
    artifact_root.mkdir(parents=True)
    options = {name: False for name in builder.US_BUILD_RECORD_SETTINGS}
    options.update(maximum_microsim_batch_size=BATCH_SIZE, **settings)

    def empty():
        return defaultdict(lambda: None)

    return types.SimpleNamespace(
        certify_args=lambda *, evidence_release: argparse.Namespace(
            **options, evidence_release=evidence_release
        ),
        artifact_root=artifact_root,
        release_root=tmp_path / "build",
        dataset_key=builder.DATASET_KEY,
        dataset_filename=builder.DATASET_FILENAME,
        calibration_key=builder.CALIBRATION_KEY,
        calibration_filename=builder.CALIBRATION_FILENAME,
        terminal_gate_failures=(),
        timing={},
        calibration_result=lambda: fit,
        registry_summary=lambda: TargetRegistry((), country="us"),
        post_export_scoring_plan=lambda args, *, result: plan,
        manifest_inputs=defaultdict(lambda: None, release_gate_failures=[]),
        gates=empty,
        diagnostics_outcome=lambda release_dir: None,
        source_coverage=empty(),
        take_up=empty(),
    )


def _certification(builder, tmp_path: Path, *, evidence_release: bool, owners=()):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    return builder._Certification(
        evidence_release=evidence_release,
        owner_patterns=owners,
        owners_file=None,
        owners_file_sha256=None,
        release_dir=release_dir,
        build="fixture-release",
        commit="c" * 40,
        changed_since_build=(),
    )


def _certify_with(builder, monkeypatch, make_scorer, record, certification, opened):
    """Run ``_certify_us_build`` with ``make_scorer(batch_parent)`` as the
    scorer it opens, in the batch parent certification passes (the scorer
    holds the fixture frame). Each scorer opened is added to ``opened``."""

    def open_scorer(plan, dataset_path, *, batch_parent=None, pooling=None):
        opened.append(make_scorer(batch_parent))
        return opened[-1]

    monkeypatch.setattr(builder, "_open_post_export_scorer", open_scorer)
    builder._certify_us_build(record, certification=certification, telemetry=None)


def _close_opened(builder, opened) -> None:
    for scorer in opened:
        builder._close_post_export_scorer(scorer)


def _child_processes() -> set[int]:
    """The process IDs of this process's live ``multiprocessing`` children."""
    return {process.pid for process in multiprocessing.active_children()}


def _engine_events(log: Path) -> list[dict]:
    """The engine log so far; a line a worker is still writing is skipped."""
    events = []
    for path in sorted(log.glob("*.jsonl")):
        for text in path.read_text().splitlines():
            try:
                events.append({"pid": int(path.stem), **json.loads(text)})
            except ValueError:
                continue
    return events


@pytest.mark.parametrize(
    ("evidence_release", "refusal"),
    [
        (False, "Release gates failed: Reform coverage smoke failed"),
        (True, "Evidence release refused"),
    ],
    ids=["certified", "evidence_unowned"],
)
def test_a_failed_check_that_stops_certification_stops_the_pools_workers(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    evidence_release,
    refusal,
) -> None:
    """The reform-coverage check's reform units are the first reform units the
    pool runs, after the merged baseline pass. When the check fails and that
    stops certification (the certified tier, or a failure no evidence owner
    matches), certification closes the scorer before its refusal leaves
    ``_certify_us_build``: the reform-validation unit a worker had already
    started is killed at once, not left to run until the process exits, and
    the scorer's batch files and the batch parent it made are removed."""
    probes = _use_probes(monkeypatch, passing=False)
    probe_ids = [probe.id for probe in probes]
    plan, fit = _release_plan(builder)
    log = tmp_path / "engine-log"
    log.mkdir()
    monkeypatch.setenv(ENGINE_LOG, str(log))
    # Every reform other than the check's waits for a file nobody writes, so
    # reform validation's first unit is still running when the check fails.
    monkeypatch.setenv(
        WAIT_FOR_OTHER_REFORMS,
        json.dumps({"file": str(tmp_path / "never-written"), "except": probe_ids}),
    )
    checked_gate = builder.us_reform_coverage_smoke_gate
    decided: list[float] = []

    def gate_once_reform_validation_started(**arguments):
        # The pool hands reform validation's first unit to the worker as the
        # check's last result comes back; wait until the worker has built
        # that unit's first engine before the check's result is returned.
        result = checked_gate(**arguments)
        deadline = time.monotonic() + 60
        while not any(
            event["event"] == "construct" and event["reform"] not in (None, *probe_ids)
            for event in _engine_events(log)
        ):
            assert time.monotonic() < deadline, "reform validation never started"
            time.sleep(0.05)
        decided.append(time.monotonic())
        return result

    # After the plan's dry run, which runs the check on a recording seam.
    monkeypatch.setattr(
        builder, "us_reform_coverage_smoke_gate", gate_once_reform_validation_started
    )
    children_before = _child_processes()
    opened: list = []
    try:
        with pytest.raises(RuntimeError, match=refusal):
            _certify_with(
                builder,
                monkeypatch,
                lambda batch_parent: _pooled_scorer(
                    builder, plan, tmp_path / "scoring", 1, batch_parent=batch_parent
                ),
                _record_for_certification(builder, plan, fit, tmp_path),
                _certification(builder, tmp_path, evidence_release=evidence_release),
                opened,
            )
        # The busy worker was killed: the pool waits 10 seconds for a worker
        # it has told to stop, and the unit it ran waits two minutes.
        assert time.monotonic() - decided[0] < 8.0
        assert _child_processes() - children_before == set()
        [scorer] = opened
        assert not scorer.batch_directory.exists()
        assert not (
            tmp_path / "build" / builder.US_POST_EXPORT_BATCH_DIRECTORY
        ).exists()

        # One worker ran every unit, in the order the pool handed them out:
        # the merged baseline pass, the check's reformed simulations, then
        # reform validation's first, which never finished.
        events = _engine_events(log)
        assert len({event["pid"] for event in events}) == 1
        built = [event["reform"] for event in events if event["event"] == "construct"]
        first_reform = next(
            index for index, reform in enumerate(built) if reform is not None
        )
        assert first_reform > 0
        assert None not in built[first_reform:]
        started = [
            reform
            for index, reform in enumerate(built[first_reform:], start=first_reform)
            if built[index - 1] != reform
        ]
        assert started[: len(probes)] == probe_ids
        assert len(started) == len(probes) + 1
        execution = scorer.execution_record()
        assert execution["report"]["units_completed"] == (
            execution["units"]["baseline"] + len(probes)
        )
        release_dir = tmp_path / "release"
        assert (release_dir / "reform_coverage_smoke.json").exists()
        assert not (release_dir / "reform_validation.json").exists()
    finally:
        _close_opened(builder, opened)


@pytest.mark.parametrize(
    ("pooled", "error_type"),
    [
        (True, RuntimeError),
        (False, RuntimeError),
        (True, BuildTerminatedError),
    ],
    ids=["pooled", "serial", "pooled_sigterm"],
)
def test_any_error_during_scoring_closes_the_scorer(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    pooled,
    error_type,
) -> None:
    """An error in any post-export stage, not only the check's refusal, stops
    certification with the scorer closed: no worker is left running and the
    batch files are removed. That includes a SIGTERM, which certification
    raises as ``BuildTerminatedError`` (a ``KeyboardInterrupt``, not an
    ``Exception``). The serial scorer is closed the same way."""
    _use_probes(monkeypatch, passing=True)
    plan, fit = _release_plan(builder)
    error = error_type("the demographics writer broke")

    def make_scorer(batch_parent):
        if pooled:
            return _pooled_scorer(
                builder, plan, tmp_path / "scoring", 1, batch_parent=batch_parent
            )
        return _serial_scorer(
            builder, plan, tmp_path / "scoring", batch_parent=batch_parent
        )

    def broken_demographics(**arguments):
        raise error

    monkeypatch.setattr(builder, "_write_demographics", broken_demographics)
    children_before = _child_processes()
    opened: list = []
    try:
        with pytest.raises(type(error)) as raised:
            _certify_with(
                builder,
                monkeypatch,
                make_scorer,
                _record_for_certification(builder, plan, fit, tmp_path),
                _certification(builder, tmp_path, evidence_release=False),
                opened,
            )
        assert raised.value is error
        assert (tmp_path / "release" / "reform_validation.json").exists()
        [scorer] = opened
        assert not scorer.batch_directory.exists()
        assert not (
            tmp_path / "build" / builder.US_POST_EXPORT_BATCH_DIRECTORY
        ).exists()
        assert _child_processes() - children_before == set()
    finally:
        _close_opened(builder, opened)


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise ValueError("this error cannot be printed")


@pytest.mark.parametrize(
    ("close_error_type", "close_error_arguments", "printed"),
    [
        (
            OSError,
            ("the batch directory is gone",),
            "OSError: the batch directory is gone",
        ),
        (_UnprintableError, (), "_UnprintableError"),
    ],
    ids=["ordinary", "unprintable"],
)
def test_an_error_closing_the_scorer_does_not_replace_the_one_that_stopped_it(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    capsys,
    close_error_type,
    close_error_arguments,
    printed,
) -> None:
    """If closing the scorer after an error raises an ``Exception`` too,
    certification prints that second error, even one that cannot be turned
    into text, and raises the first, the check's refusal."""
    _use_probes(monkeypatch, passing=False)
    plan, fit = _release_plan(builder)

    def failing_close():
        raise close_error_type(*close_error_arguments)

    def make_scorer(batch_parent):
        scorer = _serial_scorer(
            builder, plan, tmp_path / "scoring", batch_parent=batch_parent
        )
        monkeypatch.setattr(scorer, "close", failing_close)
        return scorer

    opened: list = []
    try:
        with pytest.raises(RuntimeError, match="Release gates failed"):
            _certify_with(
                builder,
                monkeypatch,
                make_scorer,
                _record_for_certification(builder, plan, fit, tmp_path),
                _certification(builder, tmp_path, evidence_release=False),
                opened,
            )
        assert printed in capsys.readouterr().err
    finally:
        for scorer in opened:
            type(scorer).close(scorer)


class _BrokenStream:
    """A standard error stream whose reader has gone."""

    def write(self, text):
        raise BrokenPipeError("standard error is closed")

    def flush(self):
        raise BrokenPipeError("standard error is closed")


def test_a_warning_that_cannot_be_written_does_not_replace_the_error(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """If closing the scorer after an error raises, and the warning about it
    cannot be written either (standard error's reader has gone), the error
    that stopped certification is still the one raised."""
    _use_probes(monkeypatch, passing=False)
    plan, fit = _release_plan(builder)

    def failing_close():
        raise OSError("the batch directory is gone")

    def make_scorer(batch_parent):
        scorer = _serial_scorer(
            builder, plan, tmp_path / "scoring", batch_parent=batch_parent
        )
        monkeypatch.setattr(scorer, "close", failing_close)
        return scorer

    opened: list = []
    try:
        with monkeypatch.context() as broken:
            broken.setattr(sys, "stderr", _BrokenStream())
            with pytest.raises(RuntimeError, match="Release gates failed"):
                _certify_with(
                    builder,
                    monkeypatch,
                    make_scorer,
                    _record_for_certification(builder, plan, fit, tmp_path),
                    _certification(builder, tmp_path, evidence_release=False),
                    opened,
                )
    finally:
        for scorer in opened:
            type(scorer).close(scorer)


def test_an_interrupt_while_closing_the_scorer_is_raised(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """An interrupt while closing the scorer after an error (Ctrl-C, or a
    first SIGTERM) is raised in place of that error, which it carries as its
    context: certification then ends as an interrupted run."""
    _use_probes(monkeypatch, passing=False)
    plan, fit = _release_plan(builder)

    def interrupted_close():
        raise BuildTerminatedError()

    def make_scorer(batch_parent):
        scorer = _serial_scorer(
            builder, plan, tmp_path / "scoring", batch_parent=batch_parent
        )
        monkeypatch.setattr(scorer, "close", interrupted_close)
        return scorer

    opened: list = []
    try:
        with pytest.raises(BuildTerminatedError) as raised:
            _certify_with(
                builder,
                monkeypatch,
                make_scorer,
                _record_for_certification(builder, plan, fit, tmp_path),
                _certification(builder, tmp_path, evidence_release=False),
                opened,
            )
        assert isinstance(raised.value.__context__, RuntimeError)
        assert "Release gates failed" in str(raised.value.__context__)
    finally:
        for scorer in opened:
            type(scorer).close(scorer)


def test_a_stage_error_kills_the_units_still_running(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
) -> None:
    """A stage's own error while workers are still running its later units
    stops certification with those units killed at once: here reform
    validation fails as it builds its second reform, while the worker runs
    that reformed simulation's unit."""
    probes = _use_probes(monkeypatch, passing=True)
    plan, fit = _release_plan(builder)
    validation = plan.consumer_requests["reform_validation"]
    assert len(validation.simulations) >= 2
    first_validation = build_fake_reform(
        validation.reforms[validation.simulations[0]]
    ).label
    held = [probe.id for probe in probes] + [first_validation]
    log = tmp_path / "engine-log"
    log.mkdir()
    monkeypatch.setenv(ENGINE_LOG, str(log))
    # Every reform after reform validation's first waits for a file nobody
    # writes, so its second unit is still running when the stage fails.
    monkeypatch.setenv(
        WAIT_FOR_OTHER_REFORMS,
        json.dumps({"file": str(tmp_path / "never-written"), "except": held}),
    )
    built: list[object] = []
    failed: list[float] = []

    def failing_on_the_second(build):
        def build_reform(*arguments, **options):
            built.append(build)
            if len(built) < 2:
                return build(*arguments, **options)
            # Wait until the worker has started the second unit's engine.
            deadline = time.monotonic() + 60
            while not any(
                event["event"] == "construct" and event["reform"] not in (None, *held)
                for event in _engine_events(log)
            ):
                assert time.monotonic() < deadline, "the second unit never started"
                time.sleep(0.05)
            failed.append(time.monotonic())
            raise RuntimeError("reform validation broke")

        return build_reform

    # Reform validation builds a spec's reform or a parameter reform.
    monkeypatch.setattr(
        ReformValidationSpec,
        "build_reform",
        failing_on_the_second(ReformValidationSpec.build_reform),
    )
    monkeypatch.setattr(
        reform_validation_module,
        "_build_parameter_reform",
        failing_on_the_second(reform_validation_module._build_parameter_reform),
    )
    children_before = _child_processes()
    opened: list = []
    try:
        with pytest.raises(RuntimeError, match="reform validation broke"):
            _certify_with(
                builder,
                monkeypatch,
                lambda batch_parent: _pooled_scorer(
                    builder, plan, tmp_path / "scoring", 1, batch_parent=batch_parent
                ),
                _record_for_certification(builder, plan, fit, tmp_path),
                _certification(builder, tmp_path, evidence_release=False),
                opened,
            )
        # The busy worker was killed: the pool waits 10 seconds for a worker
        # it has told to stop, and the unit it ran waits two minutes.
        assert time.monotonic() - failed[0] < 8.0
        assert _child_processes() - children_before == set()
        [scorer] = opened
        assert not scorer.batch_directory.exists()
        assert not (
            tmp_path / "build" / builder.US_POST_EXPORT_BATCH_DIRECTORY
        ).exists()
        execution = scorer.execution_record()
        assert execution["report"]["units_completed"] == (
            execution["units"]["baseline"] + len(probes) + 1
        )
        assert not (tmp_path / "release" / "reform_validation.json").exists()
    finally:
        _close_opened(builder, opened)


@pytest.mark.parametrize(
    ("evidence_release", "settings", "owners"),
    [
        (False, {"allow_reform_coverage_smoke_failures": True}, ()),
        (True, {}, (("Reform coverage smoke failed", "PolicyEngine/microcosm#368"),)),
    ],
    ids=["allowed", "evidence_owned"],
)
def test_a_failed_check_that_does_not_stop_certification_cancels_nothing(
    builder,
    pooled_reforms,
    tmp_path,
    monkeypatch,
    evidence_release,
    settings,
    owners,
) -> None:
    """A failed check that certification goes on past (the allow option, or an
    evidence-tier failure with an owner) cancels nothing: every later stage
    is scored in the pool and writes its file."""
    _use_probes(monkeypatch, passing=False)
    plan, fit = _release_plan(builder)

    def finished(**arguments):
        raise _ScoringFinishedError

    monkeypatch.setattr(builder, "_write_population_structure_holdout", finished)
    opened: list = []
    try:
        with pytest.raises(_ScoringFinishedError):
            _certify_with(
                builder,
                monkeypatch,
                lambda batch_parent: _pooled_scorer(
                    builder, plan, tmp_path / "scoring", 1, batch_parent=batch_parent
                ),
                _record_for_certification(builder, plan, fit, tmp_path, **settings),
                _certification(
                    builder,
                    tmp_path,
                    evidence_release=evidence_release,
                    owners=owners,
                ),
                opened,
            )
        release_dir = tmp_path / "release"
        assert {path.name for path in release_dir.iterdir()} == {
            "demographics.json",
            "reform_coverage_smoke.json",
            "reform_validation.json",
            "us_medicare_irmaa_holdout.json",
            "us_ssi_social_security_holdout.json",
        }
        smoke = json.loads((release_dir / "reform_coverage_smoke.json").read_text())
        assert smoke["reform_coverage_smoke"]["passed"] is False
        assert smoke["enforced"] is not settings.get(
            "allow_reform_coverage_smoke_failures", False
        )
        # One worker ran every unit once: none was killed and run again.
        [scorer] = opened
        report = scorer.execution_record()["report"]
        assert report["pools"] == 1
        assert report["worker_starts"] == 1
        assert report["failed_attempts"] == []
    finally:
        _close_opened(builder, opened)
