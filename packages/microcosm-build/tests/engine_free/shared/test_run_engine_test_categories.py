"""Tests for resource-aware country-engine test scheduling and reporting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tools import run_engine_test_categories as runner
from tools.run_engine_test_categories import (
    CategorySummary,
    TaskResult,
    TaskSpec,
    active_seconds,
    batched,
    build_tasks,
    parse_timing_record,
    render_markdown,
    report_payload,
)


def test_batched_preserves_order_and_keeps_the_remainder() -> None:
    paths = tuple(Path(f"test_{index}.py") for index in range(5))

    assert batched(paths, 2) == (
        paths[:2],
        paths[2:4],
        paths[4:],
    )


def test_active_seconds_unions_overlapping_task_intervals() -> None:
    assert active_seconds(((0.0, 5.0), (2.0, 7.0), (10.0, 12.5))) == 9.5


def test_timing_record_accepts_pytest_progress_prefix() -> None:
    assert parse_timing_record('.MICROCOSM_TIMING {"kind":"test"}') == {"kind": "test"}
    assert parse_timing_record("ordinary pytest output") is None


def _result(category: str, index: int, start: float, finish: float) -> TaskResult:
    path = f"packages/example/tests/engine_{category}/us/test_{index}.py"
    return TaskResult(
        category=category,
        index=index,
        paths=(Path(path),),
        returncode=0,
        started_offset=start,
        finished_offset=finish,
        timings=(
            {
                "kind": "test",
                "nodeid": f"{path}::test_example",
                "path": path,
                "wall_seconds": finish - start - 0.5,
                "phase_seconds": finish - start - 0.75,
                "outcome": "passed",
            },
            {
                "collection_seconds": 0.25,
                "kind": "file",
                "path": path,
                "test_count": 1,
                "test_wall_seconds": finish - start - 0.75,
                "wall_seconds": finish - start - 0.5,
                "phase_seconds": finish - start - 0.75,
            },
        ),
    )


def test_report_breaks_time_down_by_category_file_and_test() -> None:
    results = (
        _result("contract", 0, 0.0, 4.0),
        _result("scenario", 0, 0.0, 6.0),
        _result("workflow", 0, 6.0, 15.0),
    )

    payload = report_payload(results, overall_wall_seconds=15.0, exit_code=0)

    assert payload["overall"] == {
        "wall_seconds": 15.0,
        "process_seconds": 19.0,
        "exit_code": 0,
        "task_count": 3,
        "file_count": 3,
        "test_count": 3,
    }
    assert payload["categories"]["contract"]["active_seconds"] == 4.0
    assert payload["categories"]["scenario"]["process_seconds"] == 6.0
    assert payload["categories"]["workflow"]["overhead_seconds"] == 0.5
    assert len(payload["files"]) == 3
    assert len(payload["tests"]) == 3

    markdown = render_markdown(payload)
    assert "## Category timing" in markdown
    assert "## File timing" in markdown
    assert "## Individual test timing" in markdown
    assert "engine_workflow/us/test_0.py" in markdown
    json.dumps(payload, allow_nan=False)


def test_category_summary_handles_a_category_with_no_test_records() -> None:
    result = TaskResult(
        category="workflow",
        index=0,
        paths=(Path("test_failure.py"),),
        returncode=1,
        started_offset=2.0,
        finished_offset=5.0,
        timings=(),
    )

    summary = CategorySummary.from_results((result,))

    assert summary.process_seconds == 3.0
    assert summary.active_seconds == 3.0
    assert summary.overhead_seconds == 3.0
    assert summary.file_count == 1
    assert summary.test_count == 0


def test_repository_plan_batches_light_files_and_isolates_workflows() -> None:
    light, workflows = build_tasks(contract_batch_size=10, scenario_batch_size=8)

    assert {task.category for task in light} == {"contract", "scenario"}
    assert all(
        len(task.paths) <= (10 if task.category == "contract" else 8) for task in light
    )
    assert workflows
    assert all(task.category == "workflow" for task in workflows)
    assert all(len(task.paths) == 1 for task in workflows)


def test_runner_uses_two_fresh_process_slots_for_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    light = (TaskSpec("contract", 0, (Path("contract.py"),)),)
    workflows = (
        TaskSpec("workflow", 0, (Path("workflow_0.py"),)),
        TaskSpec("workflow", 1, (Path("workflow_1.py"),)),
    )
    calls: list[tuple[tuple[TaskSpec, ...], int]] = []

    async def fake_run_bounded(
        specs, *, max_processes: int, run_one
    ) -> tuple[TaskResult, ...]:
        del run_one
        members = tuple(specs)
        calls.append((members, max_processes))
        return tuple(
            _result(spec.category, spec.index, float(spec.index), spec.index + 1.0)
            for spec in members
        )

    monkeypatch.setattr(runner, "run_bounded", fake_run_bounded)

    results, _wall_seconds = asyncio.run(runner.run_suite(light, workflows))

    assert len(results) == 3
    assert calls == [(light, 2), (workflows, 2)]
