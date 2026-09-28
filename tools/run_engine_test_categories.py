#!/usr/bin/env python3
"""Run classified US engine tests with bounded resources and timing reports."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.ci_test_groups import selected_files
from tools.pytest_timing import TIMING_PREFIX

CATEGORY_GROUPS = {
    "contract": "engine-contract-us",
    "scenario": "engine-scenario-us",
    "workflow": "engine-workflow-us",
}
PYTEST_ARGUMENTS = (
    "-v",
    "--tb=short",
    "--maxfail=1",
    "--durations=25",
    "-p",
    "no:cacheprovider",
    "-p",
    "tools.pytest_timing",
)


@dataclass(frozen=True)
class TaskSpec:
    """One pytest process and the files it owns."""

    category: str
    index: int
    paths: tuple[Path, ...]

    @property
    def label(self) -> str:
        return f"{self.category}-{self.index + 1}"


@dataclass(frozen=True)
class TaskResult:
    """Execution result and timing messages for one pytest process."""

    category: str
    index: int
    paths: tuple[Path, ...]
    returncode: int
    started_offset: float
    finished_offset: float
    timings: tuple[dict[str, Any], ...]

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.finished_offset - self.started_offset)


@dataclass(frozen=True)
class CategorySummary:
    """Elapsed-time accounting for one semantic test category."""

    active_seconds: float
    file_count: int
    file_wall_seconds: float
    overhead_seconds: float
    process_seconds: float
    task_count: int
    test_count: int

    @classmethod
    def from_results(cls, results: Sequence[TaskResult]) -> CategorySummary:
        process_seconds = sum(result.elapsed_seconds for result in results)
        file_records = [
            timing
            for result in results
            for timing in result.timings
            if timing.get("kind") == "file"
        ]
        test_records = [
            timing
            for result in results
            for timing in result.timings
            if timing.get("kind") == "test"
        ]
        file_wall_seconds = sum(
            float(record.get("wall_seconds", 0.0)) for record in file_records
        )
        requested_files = {str(path) for result in results for path in result.paths}
        return cls(
            active_seconds=active_seconds(
                (result.started_offset, result.finished_offset) for result in results
            ),
            file_count=len(requested_files),
            file_wall_seconds=file_wall_seconds,
            overhead_seconds=max(0.0, process_seconds - file_wall_seconds),
            process_seconds=process_seconds,
            task_count=len(results),
            test_count=len(test_records),
        )


RunTask = Callable[[TaskSpec], Awaitable[TaskResult]]


def batched(paths: Sequence[Path], size: int) -> tuple[tuple[Path, ...], ...]:
    """Split ordered paths into stable batches of at most ``size`` files."""

    if size < 1:
        raise ValueError("batch size must be at least one")
    return tuple(
        tuple(paths[index : index + size]) for index in range(0, len(paths), size)
    )


def active_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the union length of possibly overlapping task intervals."""

    ordered = sorted((start, finish) for start, finish in intervals if finish > start)
    if not ordered:
        return 0.0
    active = 0.0
    current_start, current_finish = ordered[0]
    for start, finish in ordered[1:]:
        if start <= current_finish:
            current_finish = max(current_finish, finish)
            continue
        active += current_finish - current_start
        current_start, current_finish = start, finish
    return active + current_finish - current_start


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def parse_timing_record(line: str) -> dict[str, Any] | None:
    """Parse a timing record even when pytest prefixes a progress character."""

    marker_offset = line.find(TIMING_PREFIX)
    if marker_offset < 0:
        return None
    return json.loads(line[marker_offset + len(TIMING_PREFIX) :])


def _timing_records(results: Sequence[TaskResult], kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        for timing in result.timings:
            if timing.get("kind") != kind:
                continue
            records.append(
                {
                    **timing,
                    "category": result.category,
                    "task_index": result.index,
                }
            )
    return records


def report_payload(
    results: Sequence[TaskResult],
    *,
    overall_wall_seconds: float,
    exit_code: int,
) -> dict[str, Any]:
    """Build the complete serializable timing report."""

    category_results = {
        category: tuple(result for result in results if result.category == category)
        for category in CATEGORY_GROUPS
    }
    categories = {
        category: {
            key: _round_seconds(value) if isinstance(value, float) else value
            for key, value in asdict(CategorySummary.from_results(members)).items()
        }
        for category, members in category_results.items()
        if members
    }
    files = _timing_records(results, "file")
    tests = _timing_records(results, "test")
    requested_files = {str(path) for result in results for path in result.paths}
    process_seconds = sum(result.elapsed_seconds for result in results)
    tasks = [
        {
            "category": result.category,
            "elapsed_seconds": _round_seconds(result.elapsed_seconds),
            "finished_offset": _round_seconds(result.finished_offset),
            "index": result.index,
            "paths": [str(path) for path in result.paths],
            "returncode": result.returncode,
            "started_offset": _round_seconds(result.started_offset),
        }
        for result in results
    ]
    return {
        "schema_version": 1,
        "overall": {
            "wall_seconds": _round_seconds(overall_wall_seconds),
            "process_seconds": _round_seconds(process_seconds),
            "exit_code": exit_code,
            "task_count": len(results),
            "file_count": len(requested_files),
            "test_count": len(tests),
        },
        "categories": categories,
        "tasks": tasks,
        "files": files,
        "tests": tests,
    }


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a complete human-readable timing report."""

    overall = payload["overall"]
    lines = [
        "# US engine test timing report",
        "",
        (
            f"Overall wall time: **{float(overall['wall_seconds']):.2f}s**; "
            f"process time: **{float(overall['process_seconds']):.2f}s**; "
            f"files: **{overall['file_count']}**; tests: **{overall['test_count']}**; "
            f"exit code: **{overall['exit_code']}**."
        ),
        "",
        "## Category timing",
        "",
        "| Category | Active wall (s) | Process (s) | File wall (s) | Overhead (s) | Processes | Files | Tests |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, summary in payload["categories"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    f"{float(summary['active_seconds']):.2f}",
                    f"{float(summary['process_seconds']):.2f}",
                    f"{float(summary['file_wall_seconds']):.2f}",
                    f"{float(summary['overhead_seconds']):.2f}",
                    str(summary["task_count"]),
                    str(summary["file_count"]),
                    str(summary["test_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Process timing",
            "",
            "| Category | Process | Elapsed (s) | Result | Files |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for task in sorted(
        payload["tasks"], key=lambda item: float(item["elapsed_seconds"]), reverse=True
    ):
        lines.append(
            f"| {task['category']} | {int(task['index']) + 1} | "
            f"{float(task['elapsed_seconds']):.2f} | {task['returncode']} | "
            f"{', '.join(_markdown_cell(path) for path in task['paths'])} |"
        )

    lines.extend(
        [
            "",
            "## File timing",
            "",
            "| Category | File | Total (s) | Collection (s) | Test wall (s) | Pytest phases (s) | Tests |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in sorted(
        payload["files"], key=lambda item: float(item["wall_seconds"]), reverse=True
    ):
        lines.append(
            f"| {record['category']} | `{_markdown_cell(record['path'])}` | "
            f"{float(record['wall_seconds']):.2f} | "
            f"{float(record.get('collection_seconds', 0.0)):.2f} | "
            f"{float(record.get('test_wall_seconds', record['wall_seconds'])):.2f} | "
            f"{float(record['phase_seconds']):.2f} | {record['test_count']} |"
        )

    lines.extend(
        [
            "",
            "## Individual test timing",
            "",
            "| Category | Test | Wall (s) | Pytest phases (s) | Result |",
            "|---|---|---:|---:|---|",
        ]
    )
    for record in sorted(
        payload["tests"], key=lambda item: float(item["wall_seconds"]), reverse=True
    ):
        lines.append(
            f"| {record['category']} | `{_markdown_cell(record['nodeid'])}` | "
            f"{float(record['wall_seconds']):.2f} | "
            f"{float(record['phase_seconds']):.2f} | {record['outcome']} |"
        )
    return "\n".join(lines) + "\n"


def build_tasks(
    *, contract_batch_size: int, scenario_batch_size: int
) -> tuple[tuple[TaskSpec, ...], tuple[TaskSpec, ...]]:
    """Build light batches and isolated workflow process specifications."""

    contract_batches = batched(
        selected_files(CATEGORY_GROUPS["contract"]), contract_batch_size
    )
    scenario_batches = batched(
        selected_files(CATEGORY_GROUPS["scenario"]), scenario_batch_size
    )
    by_category = {
        "contract": tuple(
            TaskSpec("contract", index, paths)
            for index, paths in enumerate(contract_batches)
        ),
        "scenario": tuple(
            TaskSpec("scenario", index, paths)
            for index, paths in enumerate(scenario_batches)
        ),
    }
    light: list[TaskSpec] = []
    for index in range(max(len(contract_batches), len(scenario_batches))):
        for category in ("contract", "scenario"):
            tasks = by_category[category]
            if index < len(tasks):
                light.append(tasks[index])
    workflows = tuple(
        TaskSpec("workflow", index, (path,))
        for index, path in enumerate(selected_files(CATEGORY_GROUPS["workflow"]))
    )
    return tuple(light), workflows


async def run_task(spec: TaskSpec, *, suite_started: float) -> TaskResult:
    """Run one batch, stream output, and collect timing records."""

    label = f"[{spec.label}]"
    print(
        f"{label} START files={len(spec.paths)} "
        + " ".join(str(path) for path in spec.paths),
        flush=True,
    )
    started_offset = time.monotonic() - suite_started
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pytest",
        *(str(path) for path in spec.paths),
        *PYTEST_ARGUMENTS,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    timings: list[dict[str, Any]] = []
    try:
        async for raw_line in process.stdout:
            line = raw_line.decode(errors="replace").rstrip("\n")
            if TIMING_PREFIX in line:
                try:
                    timing = parse_timing_record(line)
                    assert timing is not None
                    timings.append(timing)
                except json.JSONDecodeError:
                    print(f"{label} invalid timing record: {line}", flush=True)
                continue
            print(f"{label} {line}", flush=True)
        returncode = await process.wait()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
        raise
    finished_offset = time.monotonic() - suite_started
    outcome = "PASS" if returncode == 0 else f"FAIL ({returncode})"
    print(
        f"{label} {outcome} in {finished_offset - started_offset:.2f}s",
        flush=True,
    )
    return TaskResult(
        category=spec.category,
        index=spec.index,
        paths=spec.paths,
        returncode=returncode,
        started_offset=started_offset,
        finished_offset=finished_offset,
        timings=tuple(timings),
    )


async def run_bounded(
    specs: Sequence[TaskSpec],
    *,
    max_processes: int,
    run_one: RunTask,
) -> tuple[TaskResult, ...]:
    """Run tasks with bounded concurrency and stop admission after failure."""

    if max_processes < 1:
        raise ValueError("max_processes must be at least one")
    next_index = 0
    failed = False
    pending: set[asyncio.Task[TaskResult]] = set()
    results: list[TaskResult] = []

    def fill() -> None:
        nonlocal next_index
        while not failed and next_index < len(specs) and len(pending) < max_processes:
            pending.add(asyncio.create_task(run_one(specs[next_index])))
            next_index += 1

    fill()
    try:
        while pending:
            done, still_pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            pending = set(still_pending)
            for task in done:
                result = task.result()
                results.append(result)
                failed = failed or result.returncode != 0
            fill()
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    return tuple(results)


async def run_suite(
    light: Sequence[TaskSpec], workflows: Sequence[TaskSpec]
) -> tuple[tuple[TaskResult, ...], float]:
    """Run light work two-at-a-time, then every full workflow alone."""

    suite_started = time.monotonic()

    async def run_one(spec: TaskSpec) -> TaskResult:
        return await run_task(spec, suite_started=suite_started)

    results = list(await run_bounded(light, max_processes=2, run_one=run_one))
    if all(result.returncode == 0 for result in results):
        results.extend(await run_bounded(workflows, max_processes=1, run_one=run_one))
    return tuple(results), time.monotonic() - suite_started


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-batch-size", type=int, default=10)
    parser.add_argument("--scenario-batch-size", type=int, default=8)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    light, workflows = build_tasks(
        contract_batch_size=args.contract_batch_size,
        scenario_batch_size=args.scenario_batch_size,
    )
    try:
        results, wall_seconds = asyncio.run(run_suite(light, workflows))
    except KeyboardInterrupt:
        return 130
    planned_count = len(light) + len(workflows)
    exit_code = (
        1
        if (
            len(results) != planned_count
            or any(result.returncode != 0 for result in results)
        )
        else 0
    )
    payload = report_payload(
        results,
        overall_wall_seconds=wall_seconds,
        exit_code=exit_code,
    )
    markdown = render_markdown(payload)
    if args.json_report is not None:
        args.json_report.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.markdown_report is not None:
        args.markdown_report.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
