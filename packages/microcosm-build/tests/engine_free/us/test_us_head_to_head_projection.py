"""The d844 go/no-go projection, against simulated scorer logs.

A greedy pool simulator stands in for the scorer's spawn pools: each worker
initializes, then takes the next (chunk, slice) task in chunk-major order
whenever it is free, and the log carries the scorer's own line formats.
Slice durations are constant within a chunk and integers, so the projection's
estimates can be checked exactly against the simulated run.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
START = 1_800_000_000  # 2027-01-15T08:00:00Z


def _stamp(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _line(epoch: int, text: str, rss: float) -> str:
    return f"[h2h {_stamp(epoch)}] {text}: peak_rss={rss:.2f} GiB"


@dataclass(frozen=True)
class Pool:
    lines: list[tuple[int, str]]
    inits: list[int]
    end: int
    worker_slices: list[int]


def _pool(
    artifact: str,
    *,
    start: int,
    workers: int,
    init_gaps: list[int],
    chunk_seconds: list[int],
    slices_per_chunk: int,
    base_rss: float,
    growth: float,
) -> Pool:
    """Greedy list scheduling of one artifact's tasks; its log lines in time order."""

    chunks = len(chunk_seconds)
    lines = [(start, _line(start, f"before starting {artifact} slice workers", 3.0))]
    inits, at = [], start
    for gap in init_gaps[:workers]:
        at += gap
        inits.append(at)
        lines.append(
            (at, _line(at, f"after initializing {artifact} slice worker", base_rss))
        )
    free = [(init, worker) for worker, init in enumerate(inits)]
    heapq.heapify(free)
    done = [0] * workers
    end = start
    for chunk in range(chunks):
        for slice_index in range(slices_per_chunk):
            at, worker = heapq.heappop(free)
            finish = at + chunk_seconds[chunk]
            done[worker] += 1
            rss = base_rss + growth * done[worker]
            label = (
                f"after scoring {artifact} chunk {chunk + 1}/{chunks} "
                f"slice {slice_index + 1}/{slices_per_chunk}"
            )
            lines.append((finish, _line(finish, label, rss)))
            lines.append(
                (finish, _line(finish, f"after receiving {artifact} slice result", 4.0))
            )
            end = max(end, finish)
            heapq.heappush(free, (finish, worker))
    return Pool(sorted(lines, key=lambda item: item[0]), inits, end, done)


@dataclass(frozen=True)
class Run:
    incumbent: Pool
    candidate: Pool

    def log(self, until: int | None = None) -> list[str]:
        rows = [*self.incumbent.lines, *self.candidate.lines]
        return [text for at, text in rows if until is None or at <= until]


def _run(
    *,
    workers: int,
    base_seconds: list[int],
    incumbent_factor: int,
    candidate_factor: int,
    incumbent_slices: int,
    candidate_slices: int,
    init_gaps: list[int],
    growth: float = 0.0,
) -> Run:
    incumbent = _pool(
        "incumbent",
        start=START + 600,
        workers=workers,
        init_gaps=init_gaps,
        chunk_seconds=[incumbent_factor * b for b in base_seconds],
        slices_per_chunk=incumbent_slices,
        base_rss=6.0,
        growth=growth,
    )
    candidate = _pool(
        "candidate",
        start=incumbent.end + 300,
        workers=workers,
        init_gaps=init_gaps,
        chunk_seconds=[candidate_factor * b for b in base_seconds],
        slices_per_chunk=candidate_slices,
        base_rss=11.0,
        growth=growth,
    )
    return Run(incumbent, candidate)


def _ready(run: Run, workers: int) -> int:
    """When the candidate's log first holds what the projection needs."""

    times = sorted(at for at, text in run.candidate.lines if "chunk 1/" in text)
    first = times[0]
    pairs = sum(1 for init in run.candidate.inits if init <= first)
    return max(max(run.candidate.inits), times[pairs - 1])


def _limits(**overrides) -> projection.Limits:
    values = {
        "workers": 4,
        "started_at": START,
        "max_wall_seconds": 10**9,
        "memory_limit_gib": 10**6,
    }
    return projection.Limits(**{**values, **overrides})


runs = st.integers(2, 6).flatmap(
    lambda workers: st.fixed_dictionaries(
        {
            "workers": st.just(workers),
            "base_seconds": st.lists(st.integers(20, 300), min_size=1, max_size=5),
            "incumbent_factor": st.integers(1, 4),
            "candidate_factor": st.integers(1, 4),
            "incumbent_slices": st.integers(workers, 3 * workers),
            "candidate_slices": st.integers(workers, 12 * workers),
            # Serial spawns, sometimes slower than a slice.
            "init_gaps": st.lists(
                st.integers(1, 400), min_size=workers, max_size=workers
            ),
        }
    )
)


@settings(max_examples=300, deadline=None)
@given(runs)
def test_projected_end_bounds_the_simulated_end_within_its_allowances(params) -> None:
    run = _run(**params)
    workers = params["workers"]
    limits = _limits(workers=workers, post_scoring_seconds=600.0)
    events, notes = projection.parse_events(run.log(_ready(run, workers)))
    report = projection.project(events, limits)
    assert report["verdict"] == "GO", report
    assert notes == []
    projected = START + report["projection"]["projected_tool_seconds_from_start"]
    true_end = run.candidate.end
    chunk_sum = params["candidate_factor"] * sum(params["base_seconds"])
    # Graham's bound: the candidate ends no later than the projection without
    # the scorecard allowance.
    assert true_end <= projected - 600.0 + 0.05
    # When every worker scored and the incumbent's work was exact, it ends no
    # earlier than its mean start plus its work over the workers: the
    # projection less the tail and the allowance.
    if not report["estimator_notes"] and min(run.candidate.worker_slices) >= 1:
        assert projected - 600.0 - chunk_sum <= true_end + 0.05
    assert report["projection"]["candidate_speed_ratio"] == pytest.approx(
        params["candidate_factor"] / params["incumbent_factor"], abs=5e-5
    )


@settings(max_examples=200, deadline=None)
@given(runs, st.floats(0.0, 0.2))
def test_projected_worker_memory_bounds_every_worker_final_peak(params, growth) -> None:
    run = _run(**params, growth=growth)
    workers = params["workers"]
    events, _ = projection.parse_events(run.log(_ready(run, workers)))
    report = projection.project(events, _limits(workers=workers))
    true_peak = 11.0 + growth * max(run.candidate.worker_slices)
    # The projection already widens for the two-decimal printing.
    assert report["projection"]["worker_peak_gib"] >= true_peak - 0.005
    assert report["projection"]["container_peak_gib"] == pytest.approx(
        workers * report["projection"]["worker_peak_gib"] + 4.0, abs=0.1
    )


def test_waits_until_the_candidate_pool_has_started_and_scored() -> None:
    run = _run(
        workers=4,
        base_seconds=[100, 200],
        incumbent_factor=1,
        candidate_factor=2,
        incumbent_slices=6,
        candidate_slices=20,
        init_gaps=[10, 10, 10, 10],
    )
    limits = _limits()
    for until in (START, run.incumbent.end, run.candidate.inits[-1]):
        events, _ = projection.parse_events(run.log(until))
        report = projection.project(events, limits)
        assert report["verdict"] == "WAIT"
        assert report["reasons"]
    events, _ = projection.parse_events(run.log(_ready(run, 4)))
    assert projection.project(events, limits)["verdict"] == "GO"


def test_no_go_names_each_limit_the_run_would_cross() -> None:
    run = _run(
        workers=4,
        base_seconds=[100, 200],
        incumbent_factor=1,
        candidate_factor=2,
        incumbent_slices=6,
        candidate_slices=40,
        init_gaps=[10, 10, 10, 10],
        growth=0.5,
    )
    events, _ = projection.parse_events(run.log(_ready(run, 4)))
    report = projection.project(
        events,
        _limits(max_wall_seconds=3_600, memory_limit_gib=60.0, process_limit_gib=12.0),
    )
    assert report["verdict"] == "NO-GO"
    kinds = [reason.split(":")[0] for reason in report["reasons"]]
    assert kinds == ["time", "memory", "memory"]
    generous = projection.project(
        events, _limits(memory_limit_gib=10**6, process_limit_gib=10**6)
    )
    assert generous["verdict"] == "GO"


def test_parsing_ignores_other_lines_and_drops_repeated_slice_lines() -> None:
    lines = [
        "2027-01-15T08:00:00.000Z  BUDGET: the tool may run 14505s",
        "$ /opt/venv/bin/python -B tools/score_us_release_head_to_head.py ...",
        "ap-123 " + _line(START, "after initializing candidate slice worker", 11.2),
        _line(START + 60, "after scoring candidate chunk 1/5 slice 3/177", 11.4),
        _line(START + 60, "after scoring candidate chunk 1/5 slice 3/177", 11.4),
        _line(START + 61, "after scoring candidate chunk 1/5", 30.0),
        _line(START + 62, "after releasing candidate scoring state", 30.0),
    ]
    events, notes = projection.parse_events(lines)
    assert [(event.kind, event.what) for event in events] == [
        ("init", None),
        ("slice", None),
        ("parent", "after scoring"),
        ("parent", "after releasing"),
    ]
    assert events[1].chunk == 1 and events[1].slices == 177
    assert notes == ["dropped 1 repeated worker slice lines"]


def test_more_initializations_than_workers_waits_instead_of_projecting() -> None:
    run = _run(
        workers=4,
        base_seconds=[100],
        incumbent_factor=1,
        candidate_factor=1,
        incumbent_slices=4,
        candidate_slices=8,
        init_gaps=[5, 5, 5, 5],
    )
    log = run.log()
    duplicated = [line for line in log if "initializing candidate" in line]
    events, _ = projection.parse_events([*log, *duplicated])
    report = projection.project(events, _limits())
    assert report["verdict"] == "WAIT"
    assert any("8 worker initializations for 4 workers" in r for r in report["reasons"])


def test_cli_reads_the_d844_plan_and_exits_one_only_on_no_go(tmp_path, capsys) -> None:
    run = _run(
        workers=16,
        base_seconds=[300, 280, 370, 330, 620],
        incumbent_factor=1,
        candidate_factor=1,
        incumbent_slices=29,
        candidate_slices=177,
        init_gaps=[20] * 16,
    )
    log = tmp_path / "app.log"
    log.write_text("\n".join(run.log(_ready(run, 16))) + "\n")
    code = projection.main(
        [str(log), "--plan", str(PLAN), "--started-at", _stamp(START)]
    )
    report = json.loads(capsys.readouterr().out)
    # At the $40 cap the plan's 14,505 s budget cannot hold October-like
    # slice times for 1,030 slices on 16 workers.
    assert report["verdict"] == "NO-GO"
    assert code == 1
    assert report["workers"] == 16
    assert any(reason.startswith("time:") for reason in report["reasons"])
    log.write_text("\n".join(run.log(START + 60)) + "\n")
    code = projection.main(
        [str(log), "--plan", str(PLAN), "--started-at", _stamp(START)]
    )
    assert json.loads(capsys.readouterr().out)["verdict"] == "WAIT"
    assert code == 0


def test_limits_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="at least 2 workers"):
        _limits(workers=1)
    with pytest.raises(ValueError, match="time_margin"):
        _limits(time_margin=1.0)
