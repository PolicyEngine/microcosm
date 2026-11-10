"""Tests for resource-aware country-engine test scheduling and reporting."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from tools import ci_test_plan
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


# The same examples on every run, so a required check never fails by chance.
_PROPERTY = settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
_CATEGORIES = st.sampled_from(["contract", "scenario", "workflow", "parity", None])
_LIGHT = ("contract", "scenario")


@st.composite
def _plans(
    draw: st.DrawFn,
) -> tuple[dict[str, ci_test_plan.TestGroup], tuple[Path, ...]]:
    """A registry of engine-us and engine-uk groups, and each group's files."""

    jobs = [
        *(
            ("engine-us", category)
            for category in draw(st.lists(_CATEGORIES, min_size=1, max_size=5))
        ),
        *(
            ("engine-uk", category)
            for category in draw(st.lists(_CATEGORIES, max_size=2))
        ),
    ]
    groups: dict[str, ci_test_plan.TestGroup] = {}
    files: list[Path] = []
    for index, (job, category) in enumerate(jobs):
        country = job.removeprefix("engine-")
        directory = (f"group_{index}", country)
        groups[f"group-{index}"] = ci_test_plan.TestGroup(
            directory,
            job,
            country,
            report_category=category,
            engine_module=f"policyengine_{country}",
        )
        files.extend(
            Path("/plan/packages/example/tests", *directory, f"test_{index}_{n}.py")
            for n in range(draw(st.integers(1, 4)))
        )
    return groups, tuple(sorted(files))


@_PROPERTY
@given(
    plan=_plans(),
    contract_batch_size=st.integers(1, 5),
    scenario_batch_size=st.integers(1, 5),
)
def test_the_runner_runs_each_listed_file_once_or_refuses_to_start(
    plan: tuple[dict[str, ci_test_plan.TestGroup], tuple[Path, ...]],
    contract_batch_size: int,
    scenario_batch_size: int,
) -> None:
    """Invariant: the runner's tasks hold each file the plan lists for
    engine-us once and no other file, or the runner refuses before it starts
    any task. It refuses exactly when ``verify`` names a group the runner never
    runs, and its refusal names each file of those groups.

    The rule, stated here: for each category it runs, the runner runs the last
    engine-us group that reports the category."""

    groups, files = plan
    last = {
        spec.report_category: name
        for name, spec in groups.items()
        if spec.job == "engine-us"
    }
    runs = {last[category] for category in (*_LIGHT, "workflow") if category in last}
    never_run = [
        name
        for name, spec in groups.items()
        if spec.job == "engine-us" and name not in runs
    ]
    sizes = {"contract": contract_batch_size, "scenario": scenario_batch_size}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ci_test_plan, "TEST_GROUPS", groups)
        patch.setattr(
            ci_test_plan,
            "GROUP_BY_DIRECTORY",
            {spec.directory: name for name, spec in groups.items()},
        )
        patch.setattr(ci_test_plan, "test_files", lambda: files)
        patch.setattr(
            runner,
            "CATEGORY_GROUPS",
            ci_test_plan.reporting_groups_for_job("engine-us"),
        )
        listed = ci_test_plan.selected_files_for_job("engine-us")
        unrun = {name: ci_test_plan.selected_files(name) for name in never_run}
        named = [error.split(":")[0] for error in ci_test_plan.us_runner_errors()]
        try:
            light, workflows = build_tasks(
                contract_batch_size=contract_batch_size,
                scenario_batch_size=scenario_batch_size,
            )
        except SystemExit as refusal:
            light, workflows, refused = (), (), str(refusal)
        else:
            refused = None

    event("refused" if refused is not None else "built")
    assert named == never_run
    if never_run:
        assert refused is not None
        assert refused.splitlines()[1:] == [
            f"  {path}: no task runs it"
            for path in listed
            if any(path in members for members in unrun.values())
        ]
        return
    assert refused is None
    held = [path for task in (*light, *workflows) for path in task.paths]
    assert Counter(held) == Counter(listed)
    # Light batches first, by batch then contract before scenario, each full
    # but the last of its category; then one workflow file to a process.
    assert list(light) == sorted(
        light, key=lambda task: (task.index, task.category != "contract")
    )
    for category in _LIGHT:
        batches = [task for task in light if task.category == category]
        assert [task.index for task in batches] == list(range(len(batches)))
        assert all(len(task.paths) == sizes[category] for task in batches[:-1])
        assert all(0 < len(task.paths) <= sizes[category] for task in batches)
    assert [(task.category, len(task.paths)) for task in workflows] == [
        ("workflow", 1)
    ] * len(workflows)


def test_the_runner_refuses_to_start_when_the_plan_lists_a_group_it_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan's workflow group, given a report category the runner doesn't
    run, makes ``main`` refuse before it starts a test process."""

    group = ci_test_plan.TEST_GROUPS["engine-workflow-us"]
    monkeypatch.setitem(
        ci_test_plan.TEST_GROUPS,
        "engine-workflow-us",
        dataclasses.replace(group, report_category="parity"),
    )
    monkeypatch.setattr(
        runner, "CATEGORY_GROUPS", ci_test_plan.reporting_groups_for_job("engine-us")
    )

    async def no_suite(*_specs: object) -> None:
        raise AssertionError("the runner started a test process")

    monkeypatch.setattr(runner, "run_suite", no_suite)

    with pytest.raises(SystemExit) as refused:
        runner.main([])

    lines = str(refused.value).splitlines()
    assert lines[0] == (
        "the US runner refuses to start: its tasks must run each file the plan "
        "lists for engine-us once, and `python3 tools/ci_test_plan.py verify` "
        "names each group it would not run:"
    )
    assert lines[1:] == [
        f"  {path.relative_to(ci_test_plan.ROOT)}: no task runs it"
        for path in ci_test_plan.selected_files("engine-workflow-us")
    ]


# What the runner imports from ``tools/``.
_RUNNER_MODULES = (
    "ci_test_plan.py",
    "pytest_timing.py",
    "run_engine_test_categories.py",
)


def _checkout(root: Path, failing: str | None) -> None:
    """Copy the runner and the plan into ``root``, with one trivial test in the
    directory of each category the runner runs. The test of ``failing`` fails."""

    (root / "tools").mkdir()
    for name in _RUNNER_MODULES:
        shutil.copyfile(Path(runner.__file__).with_name(name), root / "tools" / name)
    for category, group in ci_test_plan.reporting_groups_for_job("engine-us").items():
        directory = root.joinpath(
            "packages", "example", "tests", *ci_test_plan.TEST_GROUPS[group].directory
        )
        directory.mkdir(parents=True)
        body = "assert False" if category == failing else "pass"
        (directory / f"test_{category}.py").write_text(
            f"def test_{category}():\n    {body}\n", encoding="utf-8"
        )


@pytest.mark.parametrize(
    ("failing", "ran", "exit_code"),
    [
        (None, ("contract", "scenario", "workflow"), 0),
        ("workflow", ("contract", "scenario", "workflow"), 1),
        # A failed light batch stops the run before any workflow file.
        ("contract", ("contract", "scenario"), 1),
    ],
)
def test_the_runner_exits_with_its_tests_result_when_run_as_ci_runs_it(
    tmp_path: Path, failing: str | None, ran: tuple[str, ...], exit_code: int
) -> None:
    """``engine-us`` passes or fails on this module's exit status, which no
    call of ``main`` inside pytest can see. So this runs the module as
    ``tools/run_engine_us_tests.sh`` does, from a copy of the checkout that
    holds the runner, the plan and one trivial test for each US category, so
    no US engine is needed. If the module's last line called ``main()``
    without ``raise SystemExit(...)``, the script would exit 0 after a failed
    test, the job would pass, and the two failing cases here would fail."""

    _checkout(tmp_path, failing)
    # The run starts as CI's does, with no option from the pytest running this.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTEST_") and name != "PYTHONSAFEPATH"
    }

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.run_engine_test_categories",
            "--json-report",
            "engine-us-timings.json",
            "--markdown-report",
            "engine-us-timings.md",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=600,
    )

    assert run.returncode == exit_code, run.stdout + run.stderr
    report = json.loads(
        (tmp_path / "engine-us-timings.json").read_text(encoding="utf-8")
    )
    assert report["overall"]["exit_code"] == exit_code
    assert sorted(task["category"] for task in report["tasks"]) == sorted(ran)
    assert {task["category"]: task["returncode"] for task in report["tasks"]} == {
        category: int(category == failing) for category in ran
    }
    # The copies ran: their plan found the tests below the copy's root.
    assert all(
        Path(path).is_relative_to(tmp_path.resolve())
        for task in report["tasks"]
        for path in task["paths"]
    )
    assert (tmp_path / "engine-us-timings.md").read_text(encoding="utf-8") in run.stdout
