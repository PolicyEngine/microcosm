"""The d844 early-stop estimate, against simulated and real scorer logs.

``test_support.microcosm_build.h2h_pool_sim`` stands in for the scorer's two
pools (see its docstring for what it follows). It and the estimate's replay
are separate implementations of one model, so agreement checks the rebuild
from log lines, not the model; the real lines from a local three-worker run
of the pinned scorer (``fixtures/us_h2h``) check the model against CPython.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from test_support.microcosm_build import h2h_pool_sim as sim
from test_support.microcosm_build.us_modal_stage_plan_tool import ROOT


def _load():
    path = ROOT / "tools" / "us_head_to_head_projection.py"
    spec = importlib.util.spec_from_file_location("us_head_to_head_projection", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


projection = _load()
PLAN = ROOT / "docs" / "us-modal-stage-route-a-h2h-plan.json"
WEIGHTS = ROOT / "docs" / "evidence" / "us-h2h-speed" / "chunk-weights.json"
REAL_LOG = (
    ROOT
    / "packages"
    / "microcosm-build"
    / "tests"
    / "fixtures"
    / "us_h2h"
    / "pool-run-3-workers-2cfe3043d.log"
)
#: Review r3's toy of the scorer's pool on CPython 3.14 (twelve spawned
#: workers, three chunks of 24 slices, seed 44), whose whole-second ties
#: once left the rebuild unable to explain a worker start.
TOY_LOG = REAL_LOG.with_name("toy-pool-12-workers-g44.log")
#: The toy's own record of each chunk's mean slice time and each worker's
#: first task, from its per-process timestamps.
TOY_CHUNK_SECONDS = {1: 37.4, 2: 41.82, 3: 38.6}
TOY_FIRST_TASKS = [1, 2, 4, 8, 14, 17, 23, 30, 36, 45, 50, 61]
SCORER = ROOT / "tools" / "score_us_release_head_to_head.py"
START = 1_794_816_000  # 2026-11-16T08:00:00Z
#: The estimate allows 600 s after the last slice; the simulator takes 300.
SURPLUS = 600 - sim.SCORECARD_SECONDS


def _run(**params) -> sim.Run:
    return sim.run(started_at=START, **params)


def _prior(run: sim.Run) -> projection.Prior:
    """The run's own slice times: what a perfect local measurement gives."""

    return projection.Prior(
        candidate=tuple(float(s) for s in run.candidate_seconds),
        incumbent=tuple(float(s) for s in run.incumbent_seconds),
        candidate_slices=run.candidate_slices,
    )


def _limits(**overrides) -> projection.Limits:
    values = {"workers": 4, "started_at": START, "max_wall_seconds": 10**9}
    return projection.Limits(**{**values, **overrides})


def _judge(run: sim.Run, limits: projection.Limits, at: float) -> dict:
    return projection.project(projection.parse_log(run.log(at)), limits, at)


def _polls(run: sim.Run, count: int = 30) -> list[float]:
    """Record times spread over the run, up to the candidate's last slice."""

    times = [t for t in run.times() if t < run.candidate.end]
    return times[:: max(1, len(times) // count)]


def _estimated_end(report: dict) -> float:
    return START + report["estimate"]["estimated_tool_seconds_from_start"]


def _runs(max_workers: int) -> st.SearchStrategy:
    return st.integers(1, max_workers).flatmap(
        lambda workers: st.integers(1, 5).flatmap(
            lambda chunks: st.fixed_dictionaries(
                {
                    "workers": st.just(workers),
                    # Each chunk's time drawn on its own, for both artifacts.
                    "incumbent_seconds": st.lists(
                        st.integers(20, 300), min_size=chunks, max_size=chunks
                    ),
                    "candidate_seconds": st.lists(
                        st.integers(20, 300), min_size=chunks, max_size=chunks
                    ),
                    "incumbent_slices": st.integers(1, 3 * workers),
                    "candidate_slices": st.integers(1, 10 * workers),
                    # Starts sometimes slower than several slices.
                    "start_cost": st.integers(1, 400),
                    "first_slice_extra": st.integers(0, 200),
                }
            )
        )
    )


@settings(max_examples=150, deadline=None)
@given(_runs(16), st.floats(0.0, 1.0))
def test_the_rebuilt_schedule_is_the_simulated_one(params, cut) -> None:
    run = _run(**params)
    # Cut while a pool is open, so every example compares at least one.
    opened = run.incumbent.opened
    now = opened + cut * (run.candidate.end - opened)
    events = projection.parse_log(run.log(now)).events
    for name, simulated in (("incumbent", run.incumbent), ("candidate", run.candidate)):
        pool = projection._pool(events, name)
        history = projection.reconstruct(pool, params["workers"], now)
        if simulated.opened > now:
            assert history is None
            continue
        assert history is not None and history.notes == []
        assert history.started == {
            task: at for task, at in simulated.starts.items() if at <= now
        }
        started = sum(at <= now for at in simulated.inits)
        assert len(history.firsts) == started
        # A worker that never had a task has no first task to compare.
        assert all(
            ours == theirs
            for ours, theirs in zip(
                history.firsts, simulated.first_tasks[:started], strict=True
            )
            if theirs is not None
        )
        assert history.start_costs == [params["start_cost"]] * started


@settings(max_examples=100, deadline=None)
@given(_runs(12))
def test_with_the_runs_own_times_as_prior_the_estimate_is_its_end(params) -> None:
    run = _run(**params)
    limits = _limits(workers=params["workers"], prior=_prior(run))
    for at in _polls(run):
        report = _judge(run, limits, at)
        if "estimate" not in report:
            continue
        error = _estimated_end(report) - SURPLUS - run.tool_end
        if at >= run.candidate.opened:
            assert abs(error) <= 0.5, report
        else:
            # Loading the candidate is left out.
            assert -sim.CANDIDATE_LOAD_SECONDS - 0.5 <= error <= 0.5, report


@settings(max_examples=100, deadline=None)
@given(_runs(12), st.floats(0.2, 0.99))
def test_a_run_that_cannot_finish_is_stopped_at_its_first_estimate(
    params, share
) -> None:
    run = _run(**params)
    budget = share * (run.tool_end - START)
    limits = _limits(
        workers=params["workers"], max_wall_seconds=budget, prior=_prior(run)
    )
    for at in _polls(run):
        if at > START + budget:
            return
        report = _judge(run, limits, at)
        if "estimate" in report:
            assert report["verdict"] == "NO-GO"
            assert report["reasons"][0].startswith("time:")
            return


@settings(max_examples=100, deadline=None)
@given(_runs(12))
def test_a_run_that_fits_is_never_stopped(params) -> None:
    run = _run(**params)
    limits = _limits(
        workers=params["workers"],
        max_wall_seconds=run.tool_end - START + SURPLUS + 1,
        prior=_prior(run),
    )
    for at in _polls(run):
        assert _judge(run, limits, at)["verdict"] != "NO-GO"


def test_the_real_local_run_is_consistent_with_the_model() -> None:
    # The pinned scorer on this Mac (CPython 3.14.7, spawn), three workers:
    # every logged start is explained, every rebuilt slice ends after it
    # starts and none runs far from its chunk's pace, and the times match
    # one-process timing (docs/evidence). All three starts fall in the
    # opening burst, so this log cannot tell the start rules apart; the toy
    # log below and the source check of the scorer's line order cover what
    # it cannot.
    events = projection.parse_log(REAL_LOG.read_text().splitlines()).events
    pool = projection._pool(events, "incumbent")
    end = max(event.at for event in events)
    history = projection.reconstruct(pool, 3, end)
    assert history is not None and history.notes == []
    assert history.start_costs == [42.0, 36.0, 44.0]
    assert history.firsts == [1, 2, 4]
    timing = projection.timing(pool, history, 3)
    assert timing.notes == []
    assert all(pool.by_task[t] > history.started[t] for t in pool.by_task)
    per_chunk = pool.slices_per_chunk
    assert per_chunk is not None
    for task, at in pool.by_task.items():
        if task not in history.firsts:
            pace = timing.chunk_seconds[(task - 1) // per_chunk + 1]
            assert 0.5 * pace <= at - history.started[task] <= 1.5 * pace, task
    for chunk in (1, 2, 3, 4):
        assert 35 <= timing.chunk_seconds[chunk] <= 50
    assert 180 <= timing.chunk_seconds[5] <= 320
    prior = projection._prior(WEIGHTS)
    limits = _limits(workers=3, started_at=end - 3600, prior=prior)
    report = projection.project(
        projection.parse_log(REAL_LOG.read_text().splitlines()), limits, end
    )
    assert report["verdict"] == "GO"
    assert report["estimate"]["pace_from"] == "incumbent x prior"


def test_on_the_real_log_the_replay_comes_within_ten_percent_early() -> None:
    # docs/evidence/us-h2h-speed/pool-run-replay.txt: from every 600 s poll,
    # the replay's last slice against the actual one. Early by 0.4% to 10.0%;
    # a change to the model that moves this should say why.
    events = projection.parse_log(REAL_LOG.read_text().splitlines()).events
    full = projection._pool(events, "incumbent")
    assert full.opened is not None
    actual = max(full.by_task.values())
    span = actual - full.opened
    prior = projection._prior(WEIGHTS)
    errors = []
    for at in range(int(full.opened) + 600, int(actual), 600):
        pool = projection._pool([e for e in events if e.at <= at], "incumbent")
        history = projection.reconstruct(pool, 3, at)
        timing = projection.timing(pool, history, 3)
        seconds, _ = projection._filled(timing.chunk_seconds, 5, prior.incumbent)
        last, _ = projection.replay(
            now=at,
            workers=3,
            total=145,
            per_chunk=29,
            seconds=seconds,
            start_cost=timing.start_cost or 0.0,
            first_extra=timing.first_extra(seconds) or 0.0,
            history=history,
            finished={t: when for t, when in pool.by_task.items() if when <= at},
        )
        errors.append((last - actual) / span)
    assert len(errors) == 7
    assert all(-0.101 <= error < 0 for error in errors), errors


def test_a_toy_pool_with_steady_state_starts_follows_its_own_timing() -> None:
    # Nine of the toy's twelve starts come after a result, so it exercises
    # the steady-state start rule the real log cannot. A result finishing in
    # the same second as a start once made the rebuild give that submit a
    # token, and every later start went unexplained (review r3, MODEL-4).
    events = projection.parse_log(TOY_LOG.read_text().splitlines()).events
    pool = projection._pool(events, "incumbent")
    end = max(event.at for event in events)
    history = projection.reconstruct(pool, 12, end)
    assert history is not None
    assert len(history.firsts) == len(pool.inits) == 12
    assert len(history.notes) == 1 and "came from earlier submits" in history.notes[0]
    # Whole-second stamps can put a worker's first task one later than its
    # own record says, never earlier.
    assert all(
        0 <= ours - theirs <= 1
        for ours, theirs in zip(history.firsts, TOY_FIRST_TASKS, strict=True)
    )
    timing = projection.timing(pool, history, 3)
    assert set(timing.chunk_seconds) == set(TOY_CHUNK_SECONDS)
    for chunk, seconds in TOY_CHUNK_SECONDS.items():
        assert abs(timing.chunk_seconds[chunk] / seconds - 1) <= 0.10, chunk


def test_the_scorer_prints_after_receiving_after_its_next_submit() -> None:
    # The rebuild dates a token-taking submit by the ``after receiving`` line
    # that follows it, and the opening burst as 2 x workers submits; both
    # are the pinned scorer's order, checked here against its source.
    source = SCORER.read_text()
    loop = source[source.index("def chunk_results()") :]
    loop = loop[: loop.index("yield result")]
    order = [
        loop.index("future.result()"),
        loop.index("submit_next()"),
        loop.index("after receiving"),
    ]
    assert order == sorted(order)
    assert "for _ in range(2 * workers):\n                submit_next()" in source


def test_logged_candidate_starts_count_before_its_first_slice() -> None:
    # Sixteen workers whose candidate starts take 300 s against the
    # incumbent's 40 s. Between the candidate's first start and its first
    # slice, the estimate uses the logged 300 s, not the incumbent's 40 s,
    # which would put the end some 4,000 s early (review r3, Sol 2).
    workers = 16
    incumbent = sim.pool(
        "incumbent",
        start=START + sim.LOAD_SECONDS,
        workers=workers,
        start_cost=40,
        chunk_seconds=[40, 40],
        slices_per_chunk=29,
    )
    candidate = sim.pool(
        "candidate",
        start=incumbent.released + sim.CANDIDATE_LOAD_SECONDS,
        workers=workers,
        start_cost=300,
        chunk_seconds=[40, 40],
        slices_per_chunk=177,
        rss=11.0,
    )
    run = sim.Run(START, incumbent, candidate, [40, 40], [40, 40], 177)
    prior = _prior(run)
    first_slice = min(
        at for at, text in candidate.lines if "after scoring candidate chunk 1/2 " in text
    )
    for at in (candidate.inits[0], first_slice - 1):
        report = _judge(run, _limits(workers=workers, prior=prior), at)
        assert report["measured"]["candidate"]["slices_scored"] == 0
        assert report["estimate"]["start_seconds"]["candidate"] == 300.0
        assert abs(_estimated_end(report) - SURPLUS - run.tool_end) <= 1
    # A budget the run cannot meet is a NO-GO from the first logged start.
    short = _limits(workers=workers, prior=prior, max_wall_seconds=6000)
    assert _judge(run, short, candidate.inits[0])["verdict"] == "NO-GO"


def test_a_running_slice_past_its_pace_is_taken_to_run_as_far_again() -> None:
    # One worker, slices of 10 s. At 80 s slice 3, started at 25 s, should
    # have ended at 35 s. Taking it to end now would free the worker at once
    # and hide what it still holds; the replay takes it to end at 125 s
    # (as far past its pace again), then slice 4 runs to 135 s (review r3,
    # Sol 5).
    def at(seconds: int, text: str) -> str:
        return sim.line(START + seconds, text, 4.0)

    log = [
        at(0, "before starting candidate slice workers"),
        at(5, "after initializing candidate slice worker"),
        at(15, "after scoring candidate chunk 1/1 slice 1/4"),
        at(15, "after receiving candidate slice result"),
        at(25, "after scoring candidate chunk 1/1 slice 2/4"),
        at(25, "after receiving candidate slice result"),
    ]
    pool = projection._pool(projection.parse_log(log).events, "candidate")
    history = projection.reconstruct(pool, 1, START + 80)
    assert history is not None and history.notes == []
    assert history.started[3] == START + 25 and 4 not in history.started
    notes: list[str] = []
    last, _ = projection.replay(
        now=START + 80,
        workers=1,
        total=4,
        per_chunk=4,
        seconds={1: 10.0},
        start_cost=5.0,
        first_extra=0.0,
        history=history,
        finished=dict(pool.by_task),
        notes=notes,
    )
    assert last == START + 135
    assert notes == [
        "1 running slice(s) are past their chunk's pace; each is taken to run "
        "as far past it again"
    ]


def test_a_result_taken_without_its_slice_line_is_a_log_gap() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=8,
        candidate_slices=40,
        start_cost=10,
    )
    at = run.candidate.end - 1
    log = run.log(at)
    lost = [i for i, text in enumerate(log) if "candidate chunk 1/1 slice 3/40" in text]
    assert len(lost) == 1
    del log[lost[0]]
    report = projection.project(projection.parse_log(log), _limits(), at)
    # The rebuild stops at the gap, so the pool behind it is unknown: no
    # estimate, and only the stall check or the runner's budget can end it.
    assert report["verdict"] == "WAIT"
    assert report["log_gap"] == {"candidate": [3]}
    assert "lost lines" in report["reasons"][0]
    stalled = projection.project(projection.parse_log(log), _limits(), at + 3601)
    assert stalled["verdict"] == "NO-GO" and stalled["reasons"][0].startswith("stall:")


def test_the_candidates_fill_carries_the_incumbents_measured_departure() -> None:
    # The prior says chunk 2 takes twice chunk 1 for both artifacts; both
    # run it at three times. Once the candidate has measured only chunk 1,
    # its chunk 2 scales from the incumbent's measured ratio (150 s), not
    # the prior's (100 s) (review r3, ESTIMATE-1).
    run = _run(
        workers=4,
        incumbent_seconds=[100, 300],
        candidate_seconds=[50, 150],
        incumbent_slices=12,
        candidate_slices=40,
        start_cost=10,
    )
    prior = projection.Prior(
        candidate=(50.0, 100.0), incumbent=(100.0, 200.0), candidate_slices=40
    )
    limits = _limits(prior=prior)
    reports = [_judge(run, limits, t) for t in run.times() if t < run.candidate.end]
    early = [
        r
        for r in reports
        if r.get("estimate", {}).get("pace_from") == "candidate"
        and set(r["estimate"]["measured_chunk_seconds"]["candidate"]) == {"1"}
    ]
    assert early
    for report in early:
        assert report["estimate"]["measured_chunk_seconds"]["incumbent"] == {
            "1": 100.0,
            "2": 300.0,
        }
        assert report["estimate"]["chunk_seconds_used"] == {"1": 50.0, "2": 150.0}
        assert any("carries the incumbent" in n for n in report["estimator_notes"])
        assert abs(_estimated_end(report) - SURPLUS - run.tool_end) <= 1


def test_a_prior_off_on_one_chunk_moves_the_first_estimate_by_its_share() -> None:
    # Both artifacts run chunk 1 at 1.1 times the prior and the rest as the
    # prior says. The first estimate sees only the incumbent's chunk 1 and
    # scales every chunk still to come by its departure, so it is late by up
    # to a tenth of the scoring still ahead: a run that fits with less spare
    # than that is stopped at its first estimate (the false-NO-GO band the
    # runbook states), and one with a tenth of its run to spare is not. A
    # chunk 1 faster than the prior errs the other way: early, so a NO-GO
    # comes later than it could (review r3, ESTIMATE-5).
    prior = projection.Prior(
        candidate=(50.0, 50.0, 150.0), incumbent=(100.0, 100.0, 300.0),
        candidate_slices=40,
    )

    def scaled(factor: float) -> sim.Run:
        return _run(
            workers=4,
            incumbent_seconds=[round(100 * factor), 100, 300],
            candidate_seconds=[round(50 * factor), 50, 150],
            incumbent_slices=12,
            candidate_slices=40,
            start_cost=10,
        )

    def first(run: sim.Run, limits: projection.Limits) -> tuple[float, dict]:
        for at in run.times():
            report = _judge(run, limits, at)
            if "estimate" in report:
                return at, report
        raise AssertionError("no estimate")

    slow = scaled(1.1)
    span = slow.tool_end - START
    at, report = first(slow, _limits(prior=prior))
    late = _estimated_end(report) - SURPLUS - slow.tool_end
    assert at < slow.incumbent.end
    assert 0 < late <= 0.1 * (slow.candidate.end - at)
    tight = _limits(prior=prior, max_wall_seconds=span + SURPLUS + late - 1)
    assert first(slow, tight)[1]["verdict"] == "NO-GO"
    roomy = _limits(prior=prior, max_wall_seconds=span * 1.1 + SURPLUS)
    assert all(
        _judge(slow, roomy, t)["verdict"] != "NO-GO"
        for t in slow.times()
        if t < slow.candidate.end
    )
    fast = scaled(1 / 1.1)
    _, report = first(fast, _limits(prior=prior))
    assert _estimated_end(report) - SURPLUS < fast.tool_end


def test_the_properties_reach_an_estimate_in_most_uniform_runs() -> None:
    # The properties above check the estimate only at polls that give one,
    # so a run that never gives one checks nothing. Draws uniform over the
    # same ranges (seed 1) pin how many runs reach an estimate before the
    # candidate's last slice, and how many while the incumbent scores.
    rng = random.Random(1)
    draws = 100
    reached = early = 0
    for _ in range(draws):
        workers, chunks = rng.randint(1, 12), rng.randint(1, 5)
        run = _run(
            workers=workers,
            incumbent_seconds=[rng.randint(20, 300) for _ in range(chunks)],
            candidate_seconds=[rng.randint(20, 300) for _ in range(chunks)],
            incumbent_slices=rng.randint(1, 3 * workers),
            candidate_slices=rng.randint(1, 10 * workers),
            start_cost=rng.randint(1, 400),
            first_slice_extra=rng.randint(0, 200),
        )
        limits = _limits(workers=workers, prior=_prior(run))
        got = [t for t in _polls(run) if "estimate" in _judge(run, limits, t)]
        reached += bool(got)
        early += any(t < run.candidate.opened for t in got)
    assert reached >= 0.9 * draws, reached
    assert early >= 0.7 * draws, early


def test_a_late_receive_line_dates_the_submit_it_follows() -> None:
    # One worker, four slices of 10 s. The parent takes result 1 late (its
    # reducer, or the result's transfer, holds it), submits task 3 and prints
    # ``after receiving`` at once, at 40 s (score_us_release_head_to_head.py:
    # ``future.result()``, ``submit_next()``, then the line). So task 3 cannot
    # start before 40 s although the worker is free at 25 s; dating the submit
    # from slice 1's line (15 s) would start it at 25 s and read its slice as
    # 25 s long.
    def at(seconds: int, text: str) -> str:
        return sim.line(START + seconds, text, 4.0)

    log = [
        at(0, "before starting candidate slice workers"),
        at(5, "after initializing candidate slice worker"),
        at(15, "after scoring candidate chunk 1/1 slice 1/4"),
        at(25, "after scoring candidate chunk 1/1 slice 2/4"),
        at(40, "after receiving candidate slice result"),
        at(40, "after receiving candidate slice result"),
        at(50, "after scoring candidate chunk 1/1 slice 3/4"),
        at(50, "after receiving candidate slice result"),
        at(60, "after scoring candidate chunk 1/1 slice 4/4"),
        at(60, "after receiving candidate slice result"),
    ]
    pool = projection._pool(projection.parse_log(log).events, "candidate")
    history = projection.reconstruct(pool, 1, START + 60)
    assert history is not None and history.notes == []
    assert history.start_costs == [5.0]
    # Submit 1 starts the worker, so submit 2 waits for it.
    assert history.submitted == {1: START, 2: START + 5, 3: START + 40, 4: START + 40}
    assert history.started == {
        1: START + 5,
        2: START + 15,
        3: START + 40,
        4: START + 50,
    }
    timing = projection.timing(pool, history, 3)
    assert timing.chunk_seconds == {1: 10.0}


def test_workers_start_on_demand_so_the_pool_fills_late() -> None:
    # Each start blocks the parent, so the sixteenth worker starts at least
    # fifteen start costs after the first, while results free tokens that
    # later submits take instead of starting workers.
    run = _run(
        workers=16,
        incumbent_seconds=[40],
        candidate_seconds=[40],
        incumbent_slices=29,
        candidate_slices=177,
        start_cost=120,
    )
    inits = run.candidate.inits
    assert len(inits) == 16
    assert inits[-1] - inits[0] >= 15 * 120
    assert sum(at <= inits[-1] for at in run.candidate.starts.values()) > 2 * 16
    limits = _limits(workers=16, prior=_prior(run))
    # Measured long before the pool fills, and already right.
    at = next(t for t in run.times() if "estimate" in _judge(run, limits, t))
    assert at < run.candidate.opened
    assert abs(_estimated_end(_judge(run, limits, at)) - SURPLUS - run.tool_end) <= 61


def test_waits_for_a_pace_and_without_a_prior_for_the_candidates_own() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100, 200],
        candidate_seconds=[200, 400],
        incumbent_slices=6,
        candidate_slices=20,
        start_cost=10,
    )
    limits = _limits()
    for at in (START, run.incumbent.opened + 20, run.candidate.opened + 5):
        report = _judge(run, limits, at)
        assert report["verdict"] == "WAIT", report
        assert report["reasons"]
    at = next(t for t in run.times() if "estimate" in _judge(run, limits, t))
    report = _judge(run, limits, at)
    assert at > run.candidate.opened
    assert report["estimate"]["pace_from"] == "candidate"
    # Without a prior, chunk 2 takes chunk 1's time, and the note says so.
    assert report["estimate"]["chunk_seconds_used"] == {"1": 200.0, "2": 200.0}
    assert any("optimistic" in note for note in report["estimator_notes"])


def test_before_its_pool_the_candidate_needs_the_priors_slice_count() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[50],
        incumbent_slices=12,
        candidate_slices=20,
        start_cost=10,
    )
    full = _prior(run)
    no_count = projection.Prior(candidate=full.candidate, incumbent=full.incumbent)
    at = run.incumbent.end - 1
    report = _judge(run, _limits(prior=no_count), at)
    assert report["verdict"] == "WAIT"
    assert "slices per chunk" in report["reasons"][0]
    report = _judge(run, _limits(prior=full), at)
    assert report["estimate"]["pace_from"] == "incumbent x prior"
    assert report["estimate"]["chunk_seconds_used"] == {"1": 50.0}


def test_unmeasured_chunks_scale_from_the_slowest_measured_result() -> None:
    seconds, notes = projection._filled({1: 200.0, 2: 300.0}, 3, (1.0, 1.0, 2.0))
    # The prior says chunk 2 is as fast as chunk 1; it measures 1.5 times
    # slower, so chunk 3 scales from chunk 2 (600 s), not chunk 1 (400 s).
    assert seconds == {1: 200.0, 2: 300.0, 3: 600.0}
    assert "slowest result" in notes[0]
    seconds, notes = projection._filled({1: 200.0}, 2, ())
    assert seconds == {1: 200.0, 2: 200.0} and "optimistic" in notes[0]


def test_a_log_the_model_cannot_explain_is_reported() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=4,
        candidate_slices=20,
        start_cost=10,
    )
    at = run.candidate.end - 1
    # The opening submits start all four workers before any slice ends; drop
    # the last start from the log.
    log = run.log(at)
    starts = [i for i, text in enumerate(log) if "initializing candidate" in text]
    assert len(starts) == 4
    del log[starts[-1]]
    report = projection.project(projection.parse_log(log), _limits(), at)
    assert any(
        "starts a worker did not" in note for note in report["estimator_notes"]
    ), report


def test_a_scored_candidate_is_never_stopped() -> None:
    run = _run(
        workers=2,
        incumbent_seconds=[50],
        candidate_seconds=[50],
        incumbent_slices=2,
        candidate_slices=8,
        start_cost=5,
    )
    # Long past a budget it never fit; the scorecards are all that is left.
    report = _judge(run, _limits(workers=2, max_wall_seconds=60), run.candidate.end)
    assert report["verdict"] == "GO"
    assert "scored every slice" in report["reasons"][0]


def test_a_stall_is_a_no_go_whether_or_not_the_estimate_is_ready() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=8,
        candidate_slices=40,
        start_cost=10,
    )
    limits = _limits()
    # Loading the candidate and starting its pool print nothing: an hour is
    # allowed after the incumbent's last record.
    incumbent_only = [line for line in run.log() if "candidate" not in line]
    parsed = projection.parse_log(incumbent_only)
    last = run.incumbent.released
    assert projection.project(parsed, limits, last + 3600)["verdict"] == "WAIT"
    stalled = projection.project(parsed, limits, last + 3601)
    assert stalled["verdict"] == "NO-GO"
    assert stalled["reasons"][0].startswith("stall:")
    # Once measured, the allowance is the larger of an hour and four times
    # the slowest slice in use.
    at = next(t for t in run.times() if "estimate" in _judge(run, limits, t))
    parsed = projection.parse_log(run.log(at))
    late = projection.project(parsed, limits, at + 3601)
    assert late["verdict"] == "NO-GO"
    assert late["reasons"][-1].startswith("stall:")
    assert projection.project(parsed, limits, at + 3500)["verdict"] == "GO"
    # Before any record the allowance runs from the launch, so a log that
    # never arrives is not WAIT forever; any scorer line (compiling the
    # yardstick, loading a frame) counts as a record.
    empty = projection.parse_log([])
    assert projection.project(empty, limits, START + 3600)["verdict"] == "WAIT"
    silent = projection.project(empty, limits, START + 3601)
    assert silent["verdict"] == "NO-GO"
    assert "since the launch" in silent["reasons"][0]
    loading = projection.parse_log(
        [sim.line(START + 3000, "after loading populace_us_2024.h5", 3.6)]
    )
    assert projection.project(loading, limits, START + 6600)["verdict"] == "WAIT"
    assert projection.project(loading, limits, START + 6601)["verdict"] == "NO-GO"


def test_the_runners_budget_lines_shorten_the_deadline() -> None:
    lines = [
        "2026-11-16T08:00:01.000Z ATTEMPTS: 1 earlier unfinished attempt(s); "
        "tool budget for this attempt 9405s",
        "BUDGET: the tool may run 9000s (limited by container_timeout)",
    ]
    parsed = projection.parse_log(lines)
    assert parsed.budget_seconds == [9405, 9000]
    report = projection.project(parsed, _limits(max_wall_seconds=14505), START)
    assert report["budget_seconds"] == 9000
    assert report["deadline_utc"] == sim.stamp(START + 9000)


def test_parsing_counts_every_record_and_merges_streams_by_the_most() -> None:
    receive = sim.line(START + 60, "after receiving candidate slice result", 30.0)
    slice_ = sim.line(START + 60, "after scoring candidate chunk 1/5 slice 3/177", 11.4)
    first = [
        "$ /opt/venv/bin/python -B tools/score_us_release_head_to_head.py ...",
        "ap-123 " + sim.line(START, "after initializing candidate slice worker", 11.2),
        # Two processes writing at once: both records count.
        slice_ + receive,
        # The parent prints identical lines within a second; each counts.
        receive,
    ]
    parsed = projection.parse_log(first)
    assert [(e.kind, e.what) for e in parsed.events] == [
        ("init", None),
        ("slice", None),
        ("parent", "after receiving"),
        ("parent", "after receiving"),
    ]
    assert parsed.events[1].chunk == 1 and parsed.events[1].slices == 177
    assert parsed.notes == []
    # A restarted stream replays what the first saw and goes on.
    later = sim.line(START + 99, "after releasing candidate scoring state", 30.0)
    merged = projection.parse_logs([first, [*first[1:], later]])
    assert len(merged.events) == 5
    assert "counted once" in merged.notes[0]


def test_more_worker_starts_than_workers_waits_instead_of_estimating() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=4,
        candidate_slices=40,
        start_cost=5,
    )
    log = run.log(run.candidate.end - 1)
    # A restarted container replays a second pool's starts.
    replayed = [
        line.replace("peak_rss=11.00", "peak_rss=11.01")
        for line in log
        if "initializing candidate" in line
    ]
    parsed = projection.parse_log([*log, *replayed])
    report = projection.project(parsed, _limits(), run.candidate.end - 1)
    assert report["verdict"] == "WAIT"
    assert any("worker starts for 4 workers" in r for r in report["reasons"])


def test_cli_reads_the_d844_plan_and_exits_three_only_on_no_go(
    tmp_path, capsys
) -> None:
    # October's Mac-speed chunk means, about 5 to 10 minutes per slice.
    run = _run(
        workers=16,
        incumbent_seconds=[313, 283, 371, 327, 623],
        candidate_seconds=[313, 283, 371, 327, 623],
        incumbent_slices=29,
        candidate_slices=177,
        start_cost=40,
    )
    limits = _limits(workers=16, prior=_prior(run))
    at = next(t for t in run.times() if "estimate" in _judge(run, limits, t))
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps(
            {
                "slices_per_chunk": {"incumbent": 29, "candidate": 177},
                "candidate_mean_slice_seconds": run.candidate_seconds,
                "incumbent_mean_slice_seconds": run.incumbent_seconds,
            }
        )
    )
    # The log in two streams, the second a restart that replays the first.
    first, second = tmp_path / "app-1.log", tmp_path / "app-2.log"
    first.write_text("\n".join(run.log(at - 300)) + "\n")
    second.write_text("\n".join(run.log(at)) + "\n")
    argv = [
        str(first),
        str(second),
        "--plan",
        str(PLAN),
        "--started-at",
        sim.stamp(START),
        "--chunk-weights",
        str(weights),
        "--now",
        sim.stamp(at),
    ]
    code = projection.main(argv)
    report = json.loads(capsys.readouterr().out)
    # 1,030 such slices on 16 workers cannot fit the plan's 14,505 s.
    assert report["verdict"] == "NO-GO" and code == projection.NO_GO_EXIT == 3
    assert report["workers"] == 16 and report["budget_seconds"] == 14505
    assert report["reasons"][0].startswith("time:")
    first.write_text("\n".join(run.log(START + 60)) + "\n")
    code = projection.main([str(first), *argv[2:-1], sim.stamp(START + 60)])
    assert json.loads(capsys.readouterr().out)["verdict"] == "WAIT"
    assert code == 0


def test_a_log_the_estimate_cannot_read_raises_rather_than_answering() -> None:
    lines = [
        sim.line(START, "after scoring candidate chunk 1/5 slice 1/177", 11.0),
        sim.line(START + 1, "after scoring candidate chunk 1/4 slice 2/177", 11.0),
    ]
    with pytest.raises(ValueError, match="disagree on the chunk shape"):
        projection.project(projection.parse_log(lines), _limits(), START + 2)


def test_limits_and_priors_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="at least 1 worker"):
        _limits(workers=0)
    with pytest.raises(ValueError, match="time_margin"):
        _limits(time_margin=1.0)
    with pytest.raises(ValueError, match="positive"):
        _limits(max_wall_seconds=0)
    with pytest.raises(ValueError, match="min_slices"):
        _limits(min_slices=0)
    with pytest.raises(ValueError, match="positive"):
        projection.Prior(candidate=(1.0, 0.0))
    with pytest.raises(ValueError, match="same chunks"):
        projection.Prior(candidate=(1.0,), incumbent=(1.0, 2.0))
    with pytest.raises(ValueError, match="at least 1 slice"):
        projection.Prior(candidate_slices=0)
