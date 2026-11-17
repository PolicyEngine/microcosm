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
from pathlib import Path

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
#: Review r4's simulated 16-worker pool (seed 27) whose whole-second tie
#: once booked a worker start at 0 s.
TIE_LOG = REAL_LOG.with_name("sim-pool-16-workers-seed27-tie.log")
#: The toy's own per-process records (float clock): its config, then each
#: worker's init time and each task's start and end, from which the truth
#: below is computed (review r4, MODEL-8).
TOY_TRUTH = REAL_LOG.with_name("toy-pool-12-workers-g44-truth.jsonl")


def _toy_truth() -> tuple[dict[int, float], list[int], dict[int, float]]:
    """Each chunk's mean slice time over tasks that were not a worker's first,
    each worker's first task in start order, and each task's true start."""

    records = [json.loads(line) for line in TOY_TRUTH.read_text().splitlines()]
    config = records[0]["config"]
    tasks = {r["task"]: r for r in records if "task" in r}
    order = sorted((r["init_v"], r["pid"]) for r in records if "init_v" in r)
    firsts = [min(t for t, r in tasks.items() if r["pid"] == pid) for _, pid in order]
    per = config["slices"]
    chunks: dict[int, list[float]] = {}
    for task, record in tasks.items():
        if task not in firsts:
            chunk = (task - 1) // per + 1
            chunks.setdefault(chunk, []).append(record["end_v"] - record["start_v"])
    means = {chunk: sum(d) / len(d) for chunk, d in chunks.items()}
    return means, firsts, {task: r["start_v"] for task, r in tasks.items()}


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
@given(_runs(16), st.integers(0, 10**6))
def test_the_rebuilt_schedule_is_the_simulated_one(params, pick) -> None:
    run = _run(**params)
    # Cut at a record time from the incumbent's first task on, so every
    # example compares at least one started task and one start cost; a
    # float cut gave exactly the opening instant in about half the examples
    # (review r4, NUMBERS-6).
    first = min(run.incumbent.starts.values())
    cuts = [t for t in run.times() if first <= t <= run.candidate.end]
    now = cuts[pick % len(cuts)]
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
        if name == "incumbent":
            assert history.started and history.start_costs


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


#: What only a damaged log should raise.
_DAMAGE = (
    "log_gap",
    "stream_holes",
    "hole_suspects",
    "hole_start_suspects",
    "starts_across_holes",
    "newest_stream_empty",
    "restart_overlaps",
    "records_before_start",
)


@settings(max_examples=60, deadline=None)
@given(_runs(16))
def test_an_intact_log_raises_no_damage_flag_at_any_poll(params) -> None:
    # Review r6, N10: the claims that no intact log gives a log gap or a hole
    # flag were tested at one cut per example. Here every record time is a
    # poll, through the whole verdict, on one stream and on the same log
    # replayed whole by a second stream.
    run = _run(**params)
    limits = _limits(
        workers=params["workers"], prior=_prior(run), budget_line_required=False
    )
    times = run.times()
    for at in times[:: max(1, len(times) // 60)]:
        log = run.log(at)
        for parsed in (projection.parse_log(log), projection.parse_logs([log, log])):
            report = projection.project(parsed, limits, at)
            assert not set(_DAMAGE) & set(report), (at, report)


def test_the_fixtures_raise_no_damage_flag_at_any_record_time() -> None:
    for path, workers in ((REAL_LOG, 3), (TOY_LOG, 12), (TIE_LOG, 16)):
        lines = path.read_text().splitlines()
        stamps = [projection.parse_log([line]).record_times for line in lines]
        every = sorted({t for times in stamps for t in times})
        limits = _limits(
            workers=workers, started_at=every[0], budget_line_required=False
        )
        for at in every:
            log = [
                line
                for line, times in zip(lines, stamps, strict=True)
                if not times or max(times) <= at
            ]
            report = projection.project(projection.parse_log(log), limits, at)
            assert not set(_DAMAGE) & set(report), (path.name, at, report)


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
    # A local run of the scorer: no runner, so no BUDGET line. Started at its
    # first record, so no record predates it.
    first = min(projection.parse_log(REAL_LOG.read_text().splitlines()).record_times)
    limits = _limits(
        workers=3, started_at=first, prior=prior, budget_line_required=False
    )
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
    # the steady-state start rule the real log cannot (four of the nine run
    # the consume loop's start branch; the rest the opening loop's). A result
    # finishing in the same second as a start once made the rebuild give
    # that submit a token, and every later start went unexplained (review
    # r3, MODEL-4).
    truth, true_firsts, true_starts = _toy_truth()
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
        for ours, theirs in zip(history.firsts, true_firsts, strict=True)
    )
    # Every start after the first is a spawn the parent waited out, about
    # 40 s here; a model that put the receive line before the next submit
    # reads some as 0 s (review r4, NUMBERS-5).
    assert all(cost >= 30 for cost in history.start_costs), history.start_costs
    assert all(pool.by_task[t] >= history.started[t] for t in pool.by_task)
    # Whole-second ties misdate three slices with no note (review r4,
    # MODEL-6): a completion stamped in a spawning submit's second but
    # landing after it gives that submit a token. Any other misdating fails.
    misdated = {
        task for task, at in history.started.items() if abs(at - true_starts[task]) > 5
    }
    assert misdated == {10, 25, 37}
    # The ties make the first chunks read slow, never fast: chunk 1 +6.3%,
    # chunk 2 +5.5% and chunk 3 -0.1% against the per-process truth.
    timing = projection.timing(pool, history, 3)
    assert set(timing.chunk_seconds) == set(truth)
    for chunk, seconds in truth.items():
        assert -0.01 <= timing.chunk_seconds[chunk] / seconds - 1 <= 0.07, chunk


def test_a_start_logged_in_its_submits_second_is_put_back_on_that_submit() -> None:
    # Review r4, MODEL-7: in this simulated 16-worker pool (seed 27, whole
    # seconds), slice 15 is stamped in the same second as the spawn of submit
    # 55 but lands after it. Taken as a token, it dated submits 55 to 64 late
    # and booked a start at 0 s with no note. A start can take no less than a
    # second, so a start logged in the submit's own second is that submit's.
    log = TIE_LOG.read_text().splitlines()
    events = projection.parse_log(log).events
    pool = projection._pool(events, "incumbent")
    history = projection.reconstruct(pool, 16, max(event.at for event in events))
    assert history is not None
    assert all(cost >= 30 for cost in history.start_costs), history.start_costs
    assert len(history.notes) == 1 and "came from earlier submits" in history.notes[0]


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
        at
        for at, text in candidate.lines
        if "after scoring candidate chunk 1/2 " in text
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


def test_the_carried_fill_scales_by_one_pooled_ratio_not_the_slowest() -> None:
    # Review r4, ESTIMATE-1: the incumbent runs chunk 2 at half its prior
    # and the candidate does not. The carried fill prior is then (50, 50,
    # 150). With chunks 1 and 2 measured at 50 and 100, the slowest result
    # scales chunk 3 to 300 s; one ratio pooled over both gives 225 s. The
    # truth is 150 s: a departure the candidate does not share still makes
    # the fill late, by less.
    run = _run(
        workers=4,
        incumbent_seconds=[100, 100, 300],
        candidate_seconds=[50, 100, 150],
        incumbent_slices=12,
        candidate_slices=40,
        start_cost=10,
    )
    prior = projection.Prior(
        candidate=(50.0, 100.0, 150.0),
        incumbent=(100.0, 200.0, 300.0),
        candidate_slices=40,
    )
    limits = _limits(prior=prior)
    reports = [_judge(run, limits, t) for t in run.times() if t < run.candidate.end]
    two = [
        r
        for r in reports
        if r.get("estimate", {}).get("pace_from") == "candidate"
        and set(r["estimate"]["measured_chunk_seconds"]["candidate"]) == {"1", "2"}
        and set(r["estimate"]["measured_chunk_seconds"]["incumbent"]) == {"1", "2", "3"}
    ]
    assert two
    for report in two:
        assert report["estimate"]["chunk_seconds_used"]["3"] == 225.0
        assert any("pooled" in n for n in report["estimator_notes"])
    seconds, notes = projection._filled({1: 50.0, 2: 100.0}, 3, (50.0, 50.0, 150.0))
    assert seconds[3] == 300.0 and "slowest result" in notes[0]


def _cut_candidate_opening(run: sim.Run, at: float) -> list[str]:
    log = run.log(at)
    lost = [i for i, text in enumerate(log) if "before starting candidate" in text]
    assert len(lost) == 1
    del log[lost[0]]
    return log


def test_a_lost_pool_opening_line_is_a_log_gap() -> None:
    # Review r4, ESTIMATE-3 = WATCHER-2: without the candidate's opening line
    # its pool cannot be rebuilt, and the estimate used to replay the whole
    # pool from now, climbing a poll's worth at every poll into a false NO-GO.
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=8,
        candidate_slices=40,
        start_cost=10,
    )
    limits = _limits(prior=_prior(run))
    at = run.candidate.end - 1
    assert _judge(run, limits, at)["verdict"] == "GO"
    report = projection.project(
        projection.parse_log(_cut_candidate_opening(run, at)), limits, at
    )
    assert report["verdict"] == "WAIT" and "estimate" not in report
    assert report["log_gap"] == {"candidate": "no opening line"}
    assert any("opening line" in reason for reason in report["reasons"])
    # Before the candidate prints anything there is nothing to lose.
    before = run.candidate.opened - 1
    assert "log_gap" not in _judge(run, limits, before)


def test_records_from_before_started_at_mean_an_earlier_attempt() -> None:
    # Review r4, WATCHER-4: a relaunch into the same run folder overwrites
    # started-at, and the earlier attempt's streams were read as this one's.
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=8,
        candidate_slices=40,
        start_cost=10,
    )
    end = run.candidate.end + 1
    later = _limits(prior=_prior(run), started_at=run.candidate.opened)
    report = projection.project(projection.parse_log(run.log(end)), later, end)
    # Even an attempt whose candidate scored every slice is not taken as GO.
    assert report["verdict"] == "WAIT"
    assert report["records_before_start"] > 0
    assert "earlier attempt" in report["reasons"][0]
    own = _judge(run, _limits(prior=_prior(run)), end)
    assert own["verdict"] == "GO" and "records_before_start" not in own


def test_a_restarted_stream_that_does_not_replay_leaves_a_hole() -> None:
    run = _run(
        workers=4,
        incumbent_seconds=[100],
        candidate_seconds=[100],
        incumbent_slices=8,
        candidate_slices=40,
        start_cost=10,
    )
    limits = _limits(prior=_prior(run))
    at = run.candidate.end - 1
    log = run.log(at)
    records = [i for i, text in enumerate(log) if text.startswith("[h2h ")]
    split = records[len(records) // 2]
    first, second = log[:split], log[split + 5 :]
    report = projection.project(projection.parse_logs([first, second]), limits, at)
    holes = report["stream_holes"]
    assert len(holes) == 1 and holes[0]["from_utc"] < holes[0]["to_utc"]
    # A restart that replays what the app printed covers the hole.
    replayed = projection.project(projection.parse_logs([first, log]), limits, at)
    assert "stream_holes" not in replayed
    assert replayed["verdict"] == _judge(run, limits, at)["verdict"]


def _d844(f: float = 1.0, start_cost: int | list[int] = 300) -> sim.Run:
    """A d844-shaped pool: the measured slice times scaled by ``f``, with
    ``start_cost`` per worker start (a list gives each start its own; the
    last repeats)."""

    weights = json.loads(WEIGHTS.read_text())
    return _run(
        workers=16,
        incumbent_seconds=[
            round(s * f) for s in weights["incumbent_mean_slice_seconds"]
        ],
        candidate_seconds=[
            round(s * f) for s in weights["candidate_mean_slice_seconds"]
        ],
        incumbent_slices=29,
        candidate_slices=177,
        start_cost=start_cost,
        first_slice_extra=round(5 * f),
        load_seconds=round(64 * f),
        candidate_load_seconds=round(11 * f),
        scorecard_seconds=600,
    )


def _d844_limits() -> projection.Limits:
    return _limits(
        workers=16, max_wall_seconds=14_505, prior=projection._prior(WEIGHTS)
    )


def _offset(line: str) -> float | None:
    times = projection.parse_log([line]).record_times
    return times[0] - START if times else None


def _lose(log: list[str], start: float, end: float) -> list[list[str]]:
    """Two streams: a restart at ``start`` that does not replay, so the
    records from ``start`` up to ``end`` are lost."""

    offsets = [_offset(line) for line in log]
    first = [
        line for line, t in zip(log, offsets, strict=True) if t is None or t < start
    ]
    second = [
        line for line, t in zip(log, offsets, strict=True) if t is not None and t >= end
    ]
    return [first, second]


def test_a_worker_start_lost_in_a_hole_is_a_log_gap_not_a_no_go() -> None:
    # Review r5: a d844-shaped pool (the measured slice times, 300 s starts)
    # whose stream restarts at +360 s without replaying, losing the
    # incumbent's worker start at +364 s. The parent's later lines do not
    # reveal it, so it was no log gap: the rebuild ran a worker short, dated
    # three slices to end before they start, and the estimate read 1,593 s
    # late, a NO-GO for a run that fits. Such a rebuild is now a log gap: WAIT,
    # and the stall check still stops a hung run.
    run = _d844()
    assert 14_000 < run.tool_end - START < 14_505
    limits = _d844_limits()
    at = START + 1200
    log = run.log(at)
    intact = projection.project(projection.parse_log(log), limits, at)
    assert intact["verdict"] == "GO" and "log_gap" not in intact
    lost = [
        line for line in log if (t := _offset(line)) is not None and 360 <= t <= 370
    ]
    assert len(lost) == 1 and "initializing incumbent" in lost[0]
    damaged = projection.parse_logs(_lose(log, 360, 371))
    report = projection.project(damaged, limits, at)
    assert report["verdict"] == "WAIT", report
    assert report["log_gap"] == {
        "incumbent": "3 slices end before the rebuild starts them"
    }
    assert len(report["stream_holes"]) == 1
    assert any("rebuild starts them" in reason for reason in report["reasons"])
    # The stall check still runs: an hour with no record is a NO-GO.
    hung = projection.project(damaged, limits, at + 3601)
    assert hung["verdict"] == "NO-GO" and hung["reasons"][0].startswith("stall:")


def _cut(
    run: sim.Run, at: float, start: float, end: float
) -> tuple[list[str], list[list[str]]]:
    """The log at ``at`` and the two streams of a restart at ``start`` that
    does not replay, losing the records up to ``end``."""

    log = run.log(at)
    return log, _lose(log, start, end)


def test_lines_lost_before_a_new_stream_delivers_wait_rather_than_read_as_stragglers() -> (
    None
):
    # Before review r6, from a search of random pools: a restart at
    # +6,066 s that does not replay loses a minute of the candidate's
    # chunk-3 lines. The parent has taken those results, but the lines show
    # no gap: their receipts were lost too. The slices read as running far
    # past their pace, which the replay doubles: NO-GO at 7,793 s for a run
    # whose estimate from the intact log is 7,458 s, inside the 7,600 s
    # budget. The new stream has delivered nothing by the poll, which holds
    # the verdict on its own (review r6: this test once credited the slice
    # suspects, which no longer count from an empty stream).
    run = _run(
        workers=8,
        incumbent_seconds=[27, 223, 170],
        candidate_seconds=[94, 25, 198],
        incumbent_slices=9,
        candidate_slices=35,
        start_cost=325,
        first_slice_extra=113,
    )
    limits = _limits(workers=8, max_wall_seconds=7600, prior=_prior(run))
    at = START + 6246
    log, streams = _cut(run, at, 6066, 6126)
    assert len(streams[0]) + len(streams[1]) == len(log) - 16
    intact = projection.project(projection.parse_log(log), limits, at)
    assert intact["verdict"] == "GO"
    # The same lines as one stream, the hole unseen: the straggler misread.
    joined = projection.project(
        projection.parse_log([*streams[0], *streams[1]]), limits, at
    )
    assert joined["verdict"] == "NO-GO" and "log_gap" not in joined
    assert streams[1] == []
    report = projection.project(projection.parse_logs(streams), limits, at)
    assert report["verdict"] == "WAIT", report
    assert report["newest_stream_empty"] is True
    assert "hole_suspects" not in report and "log_gap" not in report
    assert report["reasons"] == [
        reason for reason in report["reasons"] if "delivered nothing yet" in reason
    ]
    # A stream that replays leaves no hole and no hold.
    replayed = projection.parse_logs([streams[0], log[len(streams[0]) :]])
    assert projection.project(replayed, limits, at)["verdict"] == "GO"


def test_a_d844_slice_line_lost_in_a_hole_is_held_twice_over() -> None:
    # The d844 shape at +3,600 s, a restart at +2,850 s losing one incumbent
    # slice line. Commit 6057e6f28 read it from one stream as a straggler,
    # NO-GO at 14,596 s for a run that ends at 14,240 s. The lost line also
    # leaves a submit the rebuild cannot explain, so one stream now gives a
    # log gap with the hole unseen; with it seen, the slice is a hole suspect.
    run, limits, at = _d844(), _d844_limits(), START + 3600
    log, streams = _cut(run, at, 2850, 2860)
    assert len(streams[0]) + len(streams[1]) == len(log) - 1
    assert projection.project(projection.parse_log(log), limits, at)["verdict"] == "GO"
    joined = projection.project(
        projection.parse_log([*streams[0], *streams[1]]), limits, at
    )
    assert (
        joined["verdict"] == "WAIT"
        and "took no token" in joined["log_gap"]["incumbent"]
    )
    report = projection.project(projection.parse_logs(streams), limits, at)
    assert report["verdict"] == "WAIT", report
    assert report["hole_suspects"] == {"incumbent": [54]}
    assert "took no token" in report["log_gap"]["incumbent"]


def test_a_start_whose_span_holds_a_hole_is_left_out_of_the_start_cost() -> None:
    # Before review r6, from a search of random pools: a restart at +2,259 s
    # that does not replay loses one candidate worker start in its opening.
    # The rebuild pairs the next start with the wrong submit and reads one
    # start as taking twice as long: as one stream, NO-GO at 4,112 s against
    # a 4,007 s budget for a run whose estimate from the intact log is
    # 3,672 s. Starts come one at a time, so the merged start's span holds the
    # hole; its cost is left out, and the estimate passes the run. Without the
    # lost worker the rebuild is a worker short; it becomes a log gap at the
    # next poll.
    run = _run(
        workers=9,
        incumbent_seconds=[224],
        candidate_seconds=[43],
        incumbent_slices=15,
        candidate_slices=30,
        start_cost=121,
        first_slice_extra=165,
    )
    limits = _limits(workers=9, max_wall_seconds=4007, prior=_prior(run))
    at = START + 2384
    log, streams = _cut(run, at, 2259, 2264)
    lost = [line for line in log if line not in streams[0] + streams[1]]
    assert len(lost) == 1 and "initializing candidate" in lost[0]
    assert projection.project(projection.parse_log(log), limits, at)["verdict"] == "GO"
    joined = projection.project(
        projection.parse_log([*streams[0], *streams[1]]), limits, at
    )
    assert joined["verdict"] == "NO-GO" and "log_gap" not in joined
    report = projection.project(projection.parse_logs(streams), limits, at)
    assert report["verdict"] == "GO", report
    assert report["starts_across_holes"] == {"candidate": 1}
    assert report["estimate"]["start_seconds"]["candidate"] == 121
    # The rebuild is a start short, so it reads late by about a start
    # (review r6, SF-4): 3,793 s against the intact log's 3,672 s. The
    # margin after a hole, the merged start's span of two starts, holds a
    # budget the late reading alone would fail.
    assert report["estimate"]["estimated_tool_seconds_from_start"] == 3793
    assert report["hole_margin_seconds"] == 242
    for budget, intact_verdict, verdict in (
        (3750, "GO", "WAIT"),
        (3700, "GO", "WAIT"),
        (3550, "NO-GO", "NO-GO"),
    ):
        tight = _limits(workers=9, max_wall_seconds=budget, prior=_prior(run))
        intact = projection.project(projection.parse_log(log), tight, at)
        damaged = projection.project(projection.parse_logs(streams), tight, at)
        assert (intact["verdict"], damaged["verdict"]) == (intact_verdict, verdict)
        if verdict == "WAIT":
            assert "a start for each one lost (242 s here)" in damaged["reasons"][0]
    later = START + 2684
    log, streams = _cut(run, later, 2259, 2264)
    report = projection.project(projection.parse_logs(streams), limits, later)
    assert report["verdict"] == "WAIT" and "candidate" in report["log_gap"]


def test_a_lossless_restart_never_reads_slower_starts_than_it_saw() -> None:
    # Review r6 (Sol, S1): the d844 shape with the first worker start of each
    # pool taking 100 s and every later one 320 s, and a stream that restarts
    # at +64 s, as the incumbent's pool opens, without replaying but losing
    # nothing. The first start's span (+64 s to +164 s) holds that boundary,
    # so it was left out, and the start cost became the later starts' 320 s
    # instead of the logged mean: NO-GO at 14,684 s for a run that fits. The
    # start cost is now the lesser of the two means, which never reads later
    # than the intact log on a restart that lost nothing.
    run, limits = _d844(start_cost=[100] + [320] * 15), _d844_limits()
    for poll in (600, 1200, 2400):
        at = START + poll
        log = run.log(at)
        offsets = [_offset(line) for line in log]
        streams = [
            [
                line
                for line, t in zip(log, offsets, strict=True)
                if t is None or t <= 64
            ],
            [
                line
                for line, t in zip(log, offsets, strict=True)
                if t is not None and t > 64
            ],
        ]
        assert sum(map(len, streams)) == len(log)
        intact = projection.project(projection.parse_log(log), limits, at)
        report = projection.project(projection.parse_logs(streams), limits, at)
        assert intact["verdict"] == report["verdict"] == "GO", (poll, report)
        assert report["estimate"] == intact["estimate"], poll
        assert len(report["stream_holes"]) == 1
        assert report["starts_across_holes"] == {"incumbent": 1}
    assert intact["estimate"]["estimated_tool_seconds_from_start"] < 14_505


def test_a_restart_that_loses_nothing_holds_the_verdict_one_poll() -> None:
    # The d844 shape, a stream that restarts at +4,800 s without replaying
    # but loses nothing. At the +4,800 s poll the new stream is still empty,
    # so the slices running then are suspects and the verdict waits; by the
    # next poll their lines have arrived and it is the intact log's again.
    run, limits = _d844(), _d844_limits()
    for poll, held in ((4800, True), (5400, False), (6000, False)):
        at = START + poll
        log = run.log(at)
        offsets = [_offset(line) for line in log]
        first = [
            line for line, t in zip(log, offsets, strict=True) if t is None or t < 4800
        ]
        second = [
            line
            for line, t in zip(log, offsets, strict=True)
            if t is not None and t >= 4800
        ]
        assert len(first) + len(second) == len(log)
        intact = projection.project(projection.parse_log(log), limits, at)
        report = projection.project(projection.parse_logs([first, second]), limits, at)
        assert intact["verdict"] == "GO"
        if held:
            assert report["verdict"] == "WAIT" and report["newest_stream_empty"]
            assert "hole_suspects" not in report
        else:
            assert report["verdict"] == "GO" and "hole_suspects" not in report
            # Nothing was lost, so nothing reads late; the margin is reported.
            assert report["estimate"] == intact["estimate"]
            assert report["hole_margin_seconds"] == 300


def test_a_restarted_stream_that_has_delivered_nothing_holds_the_estimate() -> None:
    # Before review r6: a restart at +11,820 s loses a minute of chunk-5
    # lines, and the new stream has printed nothing by the +12,000 s poll, so
    # there is no hole to see. The sixteen slices whose lines were lost read
    # as running past their pace: NO-GO at 14,525 s for a run that fits.
    run, limits, at = _d844(), _d844_limits(), START + 12_000
    first, second = _lose(run.log(at), 11_820, 11_880)
    assert second == []
    report = projection.project(projection.parse_logs([first, second]), limits, at)
    assert report["verdict"] == "WAIT", report
    assert report["newest_stream_empty"] is True and "stream_holes" not in report
    assert "hole_suspects" not in report and "estimate" not in report
    # One stream with nothing new is only a quiet scorer.
    quiet = projection.project(projection.parse_log(first), limits, at)
    assert "newest_stream_empty" not in quiet and "hole_suspects" not in quiet
    # An hour on, the silence is still the stream's: the stall check waits
    # with the estimate (review r6), where one quiet stream is a stall.
    later = at + 3700
    held = projection.project(projection.parse_logs([first, second]), limits, later)
    assert held["verdict"] == "WAIT"
    assert not any(reason.startswith("stall:") for reason in held["reasons"])
    stalled = projection.project(projection.parse_log(first), limits, later)
    assert stalled["verdict"] == "NO-GO"
    assert any(reason.startswith("stall:") for reason in stalled["reasons"])


def test_an_empty_new_stream_holds_between_pools_and_during_a_start() -> None:
    # Review r6, SF-2. The d844 shape at 265 s spare. (A) A restart that does
    # not replay loses only the candidate's opening line at +5,104 s, and the
    # new stream holds nothing until the first candidate worker start at
    # +5,404 s: no slice runs, so no suspect held it, and the replay's
    # blocked start pushed the estimate a second a second, NO-GO from 273 s
    # after the opening. (B) A stream ends at P - x and its restart has
    # delivered nothing by the poll P, during a worker start.
    run, limits = _d844(), _d844_limits()
    for poll in (5377, 5386, 5403):
        at = START + poll
        log, streams = _cut(run, at, 5104, 5105)
        assert len(log) - sum(map(len, streams)) == 1 and streams[1] == []
        assert (
            projection.project(projection.parse_log(log), limits, at)["verdict"] == "GO"
        )
        report = projection.project(projection.parse_logs(streams), limits, at)
        assert report["verdict"] == "WAIT" and report["newest_stream_empty"], report
        assert "estimate" not in report
    # Once the new stream delivers, the lost opening line is a log gap.
    at = START + 5450
    report = projection.project(
        projection.parse_logs(_cut(run, at, 5104, 5105)[1]), limits, at
    )
    assert report["verdict"] == "WAIT" and report["log_gap"] == {
        "candidate": "no opening line"
    }
    for poll, x in ((5700, 360), (6000, 480), (3000, 590)):
        at = START + poll
        log = run.log(at)
        first = [line for line in log if (t := _offset(line)) is None or t < poll - x]
        assert (
            projection.project(projection.parse_log(log), limits, at)["verdict"] == "GO"
        )
        report = projection.project(projection.parse_logs([first, []]), limits, at)
        assert report["verdict"] == "WAIT" and report["newest_stream_empty"], (
            poll,
            report,
        )


def test_a_worker_start_lost_before_a_new_stream_delivers_waits() -> None:
    # Review r6 (Sol, S2): a restart at +7,045 s that does not replay loses
    # the candidate's worker start there and 27 of the parent's receive
    # lines, and the new stream has printed nothing by the +7,200 s poll.
    # Before the fix the rebuild ran a worker short: NO-GO at 10,665 s
    # against a 10,600 s budget, where the intact log says 10,510 s.
    run = _run(
        workers=13,
        incumbent_seconds=[297, 73, 128],
        candidate_seconds=[156, 165, 83],
        incumbent_slices=5,
        candidate_slices=124,
        start_cost=328,
        first_slice_extra=123,
        load_seconds=625,
    )
    limits = _limits(workers=13, max_wall_seconds=10_600, prior=_prior(run))
    at = START + 7200
    log, streams = _cut(run, at, 7045, 7050)
    lost = [line for line in log if line not in streams[0]]
    assert streams[1] == [] and len(lost) == 28
    assert "initializing candidate" in lost[0]
    intact = projection.project(projection.parse_log(log), limits, at)
    assert intact["verdict"] == "GO"
    assert intact["estimate"]["estimated_tool_seconds_from_start"] == 10_510
    report = projection.project(projection.parse_logs(streams), limits, at)
    assert report["verdict"] == "WAIT" and report["newest_stream_empty"], report
    assert "estimate" not in report


def test_slice_suspects_after_a_real_hole_hold_on_their_own() -> None:
    # Review r6, SF-5: no test pinned a real hole's slice suspects as the only
    # hold. A restart at +2,068 s loses one candidate line; at +2,148 s the
    # new stream has a record (a hole, not an empty stream), there is no log
    # gap and no start under way, and two candidate slices that began before
    # the hole ended have no line.
    run = _run(
        workers=14,
        incumbent_seconds=[101],
        candidate_seconds=[41],
        incumbent_slices=20,
        candidate_slices=8,
        start_cost=138,
        first_slice_extra=121,
    )
    limits = _limits(workers=14, max_wall_seconds=2950, prior=_prior(run))
    at = START + 2148
    log, streams = _cut(run, at, 2068, 2088)
    assert len(log) - sum(map(len, streams)) == 1 and streams[1]
    intact = projection.project(projection.parse_log(log), limits, at)
    assert intact["verdict"] == "GO"
    assert intact["estimate"]["estimated_tool_seconds_from_start"] == 2920
    report = projection.project(projection.parse_logs(streams), limits, at)
    assert report["verdict"] == "WAIT", report
    assert report["hole_suspects"] == {"candidate": [2, 3]}
    assert len(report["stream_holes"]) == 1
    for other in ("newest_stream_empty", "hole_start_suspects", "log_gap", "estimate"):
        assert other not in report, other
    assert len(report["reasons"]) == 1
    assert "2 of the candidate's slices began before it ended" in report["reasons"][0]


def test_suspects_count_from_the_end_of_the_hole() -> None:
    # Review r6, SF-5 (M3): slices that began in the hole's first second are
    # suspects too. A restart at +4,853 s loses seven candidate slice lines.
    run = _run(
        workers=16,
        incumbent_seconds=[164],
        candidate_seconds=[91],
        incumbent_slices=11,
        candidate_slices=39,
        start_cost=358,
        first_slice_extra=10,
    )
    limits = _limits(workers=16, max_wall_seconds=6145, prior=_prior(run))
    at = START + 4993
    log, streams = _cut(run, at, 4853, 4873)
    assert len(log) - sum(map(len, streams)) == 7
    report = projection.project(projection.parse_logs(streams), limits, at)
    assert report["verdict"] == "WAIT", report
    assert report["hole_suspects"] == {"candidate": [28, 29, 30, 31, 32]}
    assert "7 of the candidate's slices" in report["reasons"][0]


def test_a_start_under_way_across_a_hole_waits_for_its_line() -> None:
    # Review r6, scenario C: a restart that does not replay loses the
    # candidate's worker-start line at +4,477 s. The rebuild still has that
    # start under way and no slice running, so nothing else holds, and the
    # estimate climbs a second a second: NO-GO from +4,700 s against a
    # 9,900 s budget, where the intact log says 9,731 s. A start that began
    # before the hole ended and has no line now holds, until the rebuild
    # stops fitting the log (a log gap) at +4,800 s.
    run = _run(
        workers=14,
        incumbent_seconds=[157],
        candidate_seconds=[262],
        incumbent_slices=39,
        candidate_slices=100,
        start_cost=366,
        first_slice_extra=109,
    )
    limits = _limits(workers=14, max_wall_seconds=9900, prior=_prior(run))
    for poll in (4500, 4700, 4743):
        at = START + poll
        log, streams = _cut(run, at, 4477, 4478)
        lost = [line for line in log if line not in streams[0] + streams[1]]
        assert len(lost) == 1 and "initializing candidate" in lost[0]
        intact = projection.project(projection.parse_log(log), limits, at)
        assert intact["verdict"] == "GO"
        assert intact["estimate"]["estimated_tool_seconds_from_start"] == 9731
        report = projection.project(projection.parse_logs(streams), limits, at)
        assert report["verdict"] == "WAIT", report
        assert set(report["hole_start_suspects"]) == {"candidate"}
        for other in ("newest_stream_empty", "hole_suspects", "log_gap", "estimate"):
            assert other not in report, other
        assert "worker start under way" in report["reasons"][0]
    at = START + 4800
    report = projection.project(
        projection.parse_logs(_cut(run, at, 4477, 4478)[1]), limits, at
    )
    assert report["verdict"] == "WAIT" and "candidate" in report["log_gap"]


def test_a_rebuild_that_cannot_explain_a_submit_is_a_log_gap() -> None:
    # Before review r6: a run that cannot fit (slices 3 times the Mac's,
    # 100 s starts) whose restart at +240 s loses a worker start. The
    # rebuild met a submit with no token and no start, read the pace a
    # quarter fast and said GO where the intact log says NO-GO. That note,
    # like slices ending before their starts, no intact log gives.
    run, limits, at = _d844(f=3, start_cost=100), _d844_limits(), START + 1200
    log = run.log(at)
    assert (
        projection.project(projection.parse_log(log), limits, at)["verdict"] == "NO-GO"
    )
    report = projection.project(projection.parse_logs(_lose(log, 240, 300)), limits, at)
    assert report["verdict"] == "WAIT", report
    assert "took no token and started no worker" in report["log_gap"]["incumbent"]


@pytest.mark.parametrize(
    ("f", "start_cost", "polls"),
    [(1, 300, (3600, 8400, 12_000)), (3, 100, (1200, 3600))],
)
def test_a_lost_stretch_never_turns_the_verdict_around(
    f: float, start_cost: int, polls: tuple[int, ...]
) -> None:
    # Every minute a non-replaying restart could lose, at 60 s steps, in the
    # tightest fitting d844 shape (f = 1, T = 300 s: 265 s spare) and in one
    # that cannot fit: the damaged log never gives NO-GO where the intact one
    # gives GO, or GO where it gives NO-GO.
    run, limits = _d844(f, start_cost), _d844_limits()
    checked = 0
    for poll in polls:
        at = START + poll
        log = run.log(at)
        intact = projection.project(projection.parse_log(log), limits, at)["verdict"]
        assert intact == ("GO" if f == 1 else "NO-GO")
        for start in range(0, poll - 60, 60):
            streams = _lose(log, start, start + 60)
            if sum(map(len, streams)) == len(log):
                continue
            verdict = projection.project(projection.parse_logs(streams), limits, at)
            assert {intact, verdict["verdict"]} != {"GO", "NO-GO"}, (poll, start)
            checked += 1
    assert checked > 50


def test_streams_are_read_in_the_watchers_order() -> None:
    names = [Path(f"app-{n}.log") for n in (10, 2, 1)]
    assert projection._stream_order(names) == [Path(f"app-{n}.log") for n in (1, 2, 10)]
    other = [Path("b.log"), Path("a.log")]
    assert projection._stream_order(other) == other


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
        candidate=(50.0, 50.0, 150.0),
        incumbent=(100.0, 100.0, 300.0),
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
    # same ranges (seed 1) show how many runs reach an estimate before the
    # candidate's last slice, and how many while the incumbent scores. This
    # measures uniform draws, not Hypothesis's own, which favour edge values:
    # it does not pin the properties' reach (review r4, NUMBERS-7).
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


def test_a_late_receive_line_dates_a_full_pool_submit_too() -> None:
    # Two workers, six slices of 10 s (review r4). The second start blocks the
    # parent until 20 s, by when slice 1 is done, so opening submit 3 takes
    # that token and submit 4 finds none in a full pool. Submit 5 then finds
    # no token either (one result, one token taken) and the pool is full, so
    # it starts no worker; the parent takes result 1 late and prints
    # ``after receiving`` at 40 s, straight after the submit. Task 5 cannot
    # start before 40 s although a worker is free at 30 s; dating the submit
    # from the parent's last free moment (20 s) would start it at 30 s and
    # read its slice as 20 s long.
    def at(seconds: int, text: str) -> str:
        return sim.line(START + seconds, text, 4.0)

    log = [
        at(0, "before starting candidate slice workers"),
        at(5, "after initializing candidate slice worker"),
        at(15, "after scoring candidate chunk 1/1 slice 1/6"),
        at(20, "after initializing candidate slice worker"),
        at(25, "after scoring candidate chunk 1/1 slice 2/6"),
        at(30, "after scoring candidate chunk 1/1 slice 3/6"),
        at(35, "after scoring candidate chunk 1/1 slice 4/6"),
        *[at(40, "after receiving candidate slice result")] * 4,
        at(50, "after scoring candidate chunk 1/1 slice 5/6"),
        at(50, "after scoring candidate chunk 1/1 slice 6/6"),
        *[at(50, "after receiving candidate slice result")] * 2,
    ]
    pool = projection._pool(projection.parse_log(log).events, "candidate")
    history = projection.reconstruct(pool, 2, START + 50)
    assert history is not None and history.notes == []
    assert history.start_costs == [5.0, 15.0]
    assert history.submitted == {
        1: START,
        2: START + 5,
        3: START + 20,
        4: START + 20,
        5: START + 40,
        6: START + 40,
    }
    assert history.started[5] == START + 40
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
    # Review r6, SF-5 (M15): no intact log gives it, so it is a log gap and
    # holds the verdict, not only a note.
    assert report["verdict"] == "WAIT"
    assert "starts a worker did not" in report["log_gap"]["candidate"]


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
    assert parsed.budget_seconds == [9405, 9000] and parsed.budget_lines == 1
    report = projection.project(parsed, _limits(max_wall_seconds=14505), START)
    assert report["budget_seconds"] == 9000
    assert report["deadline_utc"] == sim.stamp(START + 9000)


def test_streams_without_the_runners_budget_line_say_go_only_when_all_scored() -> None:
    # Review r4: a stream that begins late can miss the runner's only line
    # shortening the deadline (BUDGET:, or an earlier attempt's ATTEMPTS:) and
    # judge against the plan's whole budget. The runner prints a BUDGET line
    # just before the tool starts, so scorer records without one mean the
    # streams began late or lost it: GO becomes WAIT and the report carries
    # ``log_gap``, which the watcher alarms on. The BUDGET line only shortens
    # the deadline, so a NO-GO against the plan's budget still holds.
    run = _run(
        workers=4,
        incumbent_seconds=[40],
        candidate_seconds=[40],
        incumbent_slices=29,
        candidate_slices=60,
        start_cost=60,
    )
    limits = _limits(prior=_prior(run))
    at = next(t for t in run.times() if "estimate" in _judge(run, limits, t))
    full = _judge(run, limits, at)
    assert full["verdict"] == "GO" and "log_gap" not in full
    late = [line for line in run.log(at) if not line.startswith("BUDGET:")]
    assert len(late) == len(run.log(at)) - 1
    report = projection.project(projection.parse_log(late), limits, at)
    assert report["verdict"] == "WAIT"
    assert report["log_gap"] == {"runner": "no BUDGET line"}
    assert "BUDGET line" in report["reasons"][0]
    tight = _limits(prior=_prior(run), max_wall_seconds=(run.tool_end - START) / 2)
    assert projection.project(projection.parse_log(late), tight, at)["verdict"] == (
        "NO-GO"
    )
    # An ATTEMPTS: line comes before staging and its budget can be longer
    # than BUDGET's (cut to the class timeout), so it does not stand in for
    # the BUDGET line, though it still shortens the deadline (review r5).
    attempts = (
        "ATTEMPTS: 1 earlier unfinished attempt(s); tool budget for this "
        f"attempt {10**8}s"
    )
    kept = projection.project(projection.parse_log([attempts, *late]), limits, at)
    assert kept["verdict"] == "WAIT" and kept["log_gap"] == {"runner": "no BUDGET line"}
    assert kept["budget_seconds"] == 10**8
    # The one GO without it: a candidate that has scored every slice, since
    # stopping then would only throw the scores away.
    done = run.candidate.end
    scored = [line for line in run.log(done) if not line.startswith("BUDGET:")]
    final = projection.project(projection.parse_log(scored), limits, done)
    assert final["verdict"] == "GO" and "every slice" in final["reasons"][0]
    # Before the tool's first output there is nothing to miss yet.
    empty = projection.project(projection.parse_log([]), limits, START + 1)
    assert "log_gap" not in empty


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


def test_a_restart_inside_a_second_of_repeated_lines_is_reported_not_guessed() -> None:
    # Review r6 (Sol, S3): the toy log split between lines 31 and 32, inside a
    # second in which the parent printed the same line five times, one before
    # the split. A restart that does not replay and a replay begun inside that
    # second look the same, so the shared line counts once and one receive is
    # short: the rebuild does not fit, and the verdict waits. The second is
    # reported, so the gap is not read as lines proven lost.
    toy = TOY_LOG.read_text().splitlines()
    assert "after receiving" in toy[30] and "after initializing" in toy[31]
    assert toy[30][:26] == toy[31][:26] == toy[32][:26]
    one = projection.parse_log(toy)
    two = projection.parse_logs([toy[:31], toy[31:]])
    assert (len(one.events), len(two.events)) == (161, 160)
    assert one.overlaps == [] and len(two.overlaps) == 1
    limits = _limits(workers=12, budget_line_required=False)
    end = max(one.record_times)
    assert "log_gap" not in projection.project(one, limits, end)
    report = projection.project(two, limits, end)
    assert report["verdict"] == "WAIT" and "incumbent" in report["log_gap"]
    assert report["restart_overlaps"] == ["2026-11-16T21:27:22Z"]
    assert any("cannot tell apart" in reason for reason in report["reasons"])
    # A replay begun inside that second counts every record once.
    replay = projection.parse_logs([toy[:31], toy[30:]])
    assert sorted(replay.events, key=repr) == sorted(one.events, key=repr)
    assert len(replay.overlaps) == 1


def test_a_replay_from_inside_the_last_second_counts_nothing_twice() -> None:
    # Before the r7 fix a stream that began in the second the last one ended
    # summed that second's records with it. Where the restart replays from
    # inside that second, here only the incumbent's worker start at +664 s,
    # that counts a start twice: NO-GO at +1,200 s for the d844 shape at
    # 265 s spare. Counting each record as often as one stream shows it, the
    # verdict is the intact log's.
    run, limits, at = _d844(), _d844_limits(), START + 1200
    log = run.log(at)
    last = max(i for i, line in enumerate(log) if _offset(line) == 664)
    assert "initializing incumbent" in log[last]
    intact = projection.project(projection.parse_log(log), limits, at)
    report = projection.project(
        projection.parse_logs([log[: last + 1], log[last:]]), limits, at
    )
    assert intact["verdict"] == report["verdict"] == "GO", report
    assert report["estimate"] == intact["estimate"]
    assert report["restart_overlaps"] == [projection._utc(START + 664)]
    assert "log_gap" not in report


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


def test_the_cli_reads_ten_streams_in_the_watchers_order(tmp_path, capsys) -> None:
    # Review r6, N9: the watcher passes its streams in shell-glob order,
    # app-10.log before app-2.log. The d844 shape at +12,000 s, nine streams
    # that each hold the log to a restart at +11,820 s, and the tenth, the
    # newest, restarted and empty: read in the watcher's order the verdict
    # waits for it; read as given, app-9 is the newest, the minute lost
    # before the restart reads as stragglers, and the run would be stopped.
    run, at = _d844(), START + 12_000
    first, _ = _lose(run.log(at), 11_820, 11_880)
    paths = [tmp_path / f"app-{n}.log" for n in range(1, 11)]
    for path in paths[:-1]:
        path.write_text("\n".join(first) + "\n")
    paths[-1].write_text("")
    globbed = sorted(str(path) for path in paths)
    assert Path(globbed[1]).name == "app-10.log"
    argv = [
        *globbed,
        "--plan",
        str(PLAN),
        "--started-at",
        sim.stamp(START),
        "--chunk-weights",
        str(WEIGHTS),
        "--now",
        sim.stamp(at),
    ]
    code = projection.main(argv)
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["verdict"] == "WAIT", report
    assert report["newest_stream_empty"] is True
    as_given = projection.parse_logs(
        [Path(p).read_text().splitlines() for p in globbed]
    )
    assert projection.project(as_given, _d844_limits(), at)["verdict"] == "NO-GO"


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
