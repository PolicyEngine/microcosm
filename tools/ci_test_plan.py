#!/usr/bin/env python3
"""Define, validate, select, and describe the complete CI test plan."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
API_ROOT = "https://api.github.com"


@dataclass(frozen=True)
class TestGroup:
    """One directory-defined test group and its complete CI ownership."""

    directory: tuple[str, str]
    job: str
    country: str
    report_category: str | None = None
    engine_module: str | None = None
    integration: bool = False


# This registry is the only authority for test directories, country ownership,
# CI job assignment, engine dependencies, integration collection, and US
# timing-report categories. Consumers must query or import it rather than
# maintaining parallel mappings.
TEST_GROUPS = {
    "engine-free-shared": TestGroup(("engine_free", "shared"), "engine-free", "shared"),
    "engine-free-us": TestGroup(("engine_free", "us"), "engine-free", "us"),
    "engine-free-uk": TestGroup(("engine_free", "uk"), "engine-free", "uk"),
    "engine-contract-us": TestGroup(
        ("engine_contract", "us"),
        "engine-us",
        "us",
        report_category="contract",
        engine_module="policyengine_us",
    ),
    "engine-scenario-us": TestGroup(
        ("engine_scenario", "us"),
        "engine-us",
        "us",
        report_category="scenario",
        engine_module="policyengine_us",
    ),
    "engine-workflow-us": TestGroup(
        ("engine_workflow", "us"),
        "engine-us",
        "us",
        report_category="workflow",
        engine_module="policyengine_us",
    ),
    "engine-uk": TestGroup(
        ("engine", "uk"), "engine-uk", "uk", engine_module="policyengine_uk"
    ),
    "integration-uk": TestGroup(
        ("integration", "uk"),
        "integration-uk",
        "uk",
        engine_module="policyengine_uk",
        integration=True,
    ),
    # Tests that start the separately installed microcosm-emitter. Their
    # runner installs each release they cover and passes --run-integration.
    "installed-emitter-shared": TestGroup(
        ("installed_emitter", "shared"),
        "installed-emitter",
        "shared",
        integration=True,
    ),
}
GROUP_BY_DIRECTORY = {spec.directory: name for name, spec in TEST_GROUPS.items()}
JOBS = tuple(dict.fromkeys(spec.job for spec in TEST_GROUPS.values()))
WORKFLOW_INFRASTRUCTURE_JOBS = {
    "select-countries": "selects the country-specific test categories",
    "lint": "runs static checks and verifies this test plan",
    "wheels": "builds and inspects distribution archives",
    "ci": "reports one result for all the other jobs, for the main ruleset to require",
}
# ``ci`` gives the ``main`` ruleset one check that covers the country jobs. It
# fails unless every other job passed or the selection left it out. A skipped
# matrix job reports under its unexpanded name (``engine-uk``, not
# ``engine-uk (3.13)``), so a ruleset cannot require ``engine-us`` or
# ``engine-uk`` by name. The ruleset also names both ``engine-free`` legs, and
# that job runs this plan's tests. A change that breaks ``ci`` therefore fails
# a check the ruleset reads without ``ci``.
AGGREGATE_JOB = "ci"
# The job and step whose outputs say which country jobs a run needs.
SELECTION_JOB = "select-countries"
SELECTION_STEP = "plan"
SELECTION_COMMAND = "python3 tools/ci_test_plan.py select-countries"
# The one ``run`` step of ``ci``, and the context GitHub hands it.
REQUIRE_RESULTS_COMMAND = "python3 tools/ci_test_plan.py require-results"
NEEDS_EXPRESSION = "${{ toJSON(needs) }}"
AGGREGATE_JOB_KEYS = (
    "if",
    "needs",
    "permissions",
    "runs-on",
    "timeout-minutes",
    "steps",
)
# The shell every ``run`` step runs under. GitHub runs an explicit
# ``shell: bash`` as ``bash -eo pipefail``, and a step with no shell as
# ``bash -e``, where a pipe hides the exit status of the command that feeds it.
WORKFLOW_DEFAULTS = {"run": {"shell": "bash"}}
# Step conditions that run a step whenever its job is still passing.
STEP_CONDITIONS = ("always()", "!cancelled()", "success()")
# The tests of this plan, and the job that must run them. The ruleset reads
# that job without ``ci``, so the tests of the code ``ci`` runs stay in it.
PLAN_TESTS = "packages/microcosm-build/tests/engine_free/shared/test_ci_test_plan.py"
PLAN_TESTS_JOB = "engine-free"
# The job that ``tools/run_engine_test_categories.py`` runs, and the only
# report categories it runs: contract and scenario files in batches, then each
# workflow file in a fresh process. The runner refuses to start unless its
# tasks hold each file the plan lists for the job exactly once, and ``verify``
# refuses a group of the job that the runner never runs.
US_RUNNER_JOB = "engine-us"
US_RUNNER_CATEGORIES = ("contract", "scenario", "workflow")
# GitHub lists at most this many files of one pull request.
LISTED_FILES_LIMIT = 3000

DOC_PREFIXES = ("docs/",)
DOC_FILES = {"LICENSE"}
DOC_SUFFIXES = {".md", ".rst"}

RequestJSON = Callable[[str, str], Any]

# A word of a file name: a maximal run of ASCII letters.
_WORD = re.compile(r"[A-Za-z]+")
# Line breaks in YAML 1.1 (PyYAML) that YAML 1.2 reads as ordinary characters.
_YAML_1_1_LINE_BREAKS = "\x85\u2028\u2029"


def plan_jobs() -> tuple[str, ...]:
    """Return every workflow job the plan declares, except the aggregate job."""

    return tuple(
        name for name in (*WORKFLOW_INFRASTRUCTURE_JOBS, *JOBS) if name != AGGREGATE_JOB
    )


def selection_output(job: object) -> str | None:
    """Return the selection output that decides whether one job runs.

    A job whose every test group needs the same country's engine runs only
    when ``select-countries`` reports that country. Every other job always
    runs, and this returns ``None`` for it.
    """

    specs = [spec for spec in TEST_GROUPS.values() if spec.job == job]
    countries = {spec.country for spec in specs}
    if len(countries) == 1 and all(spec.engine_module for spec in specs):
        return f"run_{countries.pop()}"
    return None


def selection_errors() -> tuple[str, ...]:
    """Return errors for jobs that wait on an output the selection never writes."""

    written = set(select_countries(()))
    return tuple(
        f"{job}: the selection writes no {output} output"
        for job in JOBS
        if (output := selection_output(job)) is not None and output not in written
    )


def plan_tests_errors() -> tuple[str, ...]:
    """Return errors unless ``engine-free`` runs this plan's own tests.

    The selection must not decide that job. The root ``conftest.py`` leaves a
    group's files out when the group needs an engine the environment lacks,
    or holds integration tests and pytest runs without ``--run-integration``.
    """

    path = ROOT / PLAN_TESTS
    group = group_name_for_test_path(path)
    spec = TEST_GROUPS.get(group) if group is not None else None
    if (
        path in test_files()
        and spec is not None
        and spec.job == PLAN_TESTS_JOB
        and spec.engine_module is None
        and not spec.integration
        and selection_output(PLAN_TESTS_JOB) is None
    ):
        return ()
    return (
        f"{PLAN_TESTS}: {PLAN_TESTS_JOB} must run this file, the plan's own "
        "tests, on every pull request, because the main ruleset reads that job "
        f"without {AGGREGATE_JOB}. If the tests moved, update PLAN_TESTS",
    )


def us_runner_errors() -> tuple[str, ...]:
    """Return an error for each ``US_RUNNER_JOB`` group the US runner never runs.

    For each category in ``US_RUNNER_CATEGORIES``, the runner runs the one
    group that ``reporting_groups_for_job`` maps the category to. So it never
    runs a group with no report category or with another one, nor a group
    whose category a later group also reports.
    """

    runs = {
        group
        for category, group in reporting_groups_for_job(US_RUNNER_JOB).items()
        if category in US_RUNNER_CATEGORIES
    }
    return tuple(
        f"{name}: the US runner never runs this group, since it runs one "
        f"{US_RUNNER_JOB} group for each report category in "
        f"{', '.join(US_RUNNER_CATEGORIES)}"
        for name, spec in TEST_GROUPS.items()
        if spec.job == US_RUNNER_JOB and name not in runs
    )


def selection_condition(job: object) -> str | None:
    """Return the ``if`` the workflow must give one job, or ``None`` for no ``if``."""

    output = selection_output(job)
    if output is None:
        return None
    return f"needs['{SELECTION_JOB}'].outputs.{output} == 'true'"


def load_workflow(source: str) -> object:
    """Parse workflow YAML, and refuse three things GitHub may read differently.

    PyYAML keeps the last of two keys it reads as one and says nothing,
    applies merge keys (``<<``), which GitHub's docs do not list beside anchors
    and aliases, and reads U+0085, U+2028 and U+2029 as line breaks, which YAML
    1.2 does not. PyYAML reads two keys as one when it builds the same value
    from them, as it does from ``NO`` and ``OFF``. The import sits here because
    ``select-countries`` and ``ci`` run this module on bare ``python3``, where
    PyYAML may not exist.
    """

    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict:
        written: dict[object, str] = {}
        for key, _ in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                problem = "a mapping uses a merge key (`<<`)"
            elif not isinstance(key, yaml.ScalarNode):
                problem = "a mapping uses a list or a mapping as a key"
            else:
                value = loader.construct_object(key)
                if value not in written:
                    written[value] = key.value
                    continue
                problem = (
                    f"a mapping sets `{key.value}` twice"
                    if written[value] == key.value
                    else f"PyYAML reads `{written[value]}` and `{key.value}` as one key"
                )
            raise yaml.constructor.ConstructorError(None, None, problem, key.start_mark)
        return loader.construct_mapping(node, deep=True)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    for number, line in enumerate(source.split("\n"), start=1):
        for character in _YAML_1_1_LINE_BREAKS:
            if character in line:
                raise ValueError(
                    f"workflow line {number} holds U+{ord(character):04X}, which "
                    "PyYAML reads as a line break and YAML 1.2 does not"
                )
    try:
        return yaml.load(source, Loader=Loader)
    except yaml.YAMLError as error:
        raise ValueError(f"workflow fails to load: {error}") from error


def _job_needs(job: Mapping[object, object]) -> tuple[str, ...] | None:
    """Return the jobs one job needs.

    Return ``None`` unless ``needs`` holds a job name or a list of them.
    """

    needs = job.get("needs", [])
    if isinstance(needs, str):
        return (needs,)
    if isinstance(needs, list) and all(isinstance(name, str) for name in needs):
        return tuple(needs)
    return None


def _job_steps(job: Mapping[object, object]) -> list[Mapping[object, object]] | None:
    """Return one job's steps, or ``None`` unless ``steps`` holds a list of mappings."""

    steps = job.get("steps", [])
    if isinstance(steps, list) and all(isinstance(step, Mapping) for step in steps):
        return steps
    return None


def _steps_within(steps: object) -> Iterator[Mapping[object, object]]:
    """Yield every step, and every step that a step groups in a list."""

    for step in steps if isinstance(steps, list) else ():
        if isinstance(step, Mapping):
            yield step
            for value in step.values():
                yield from _steps_within(value)


def _skips_while_passing(step: Mapping[object, object]) -> bool:
    """Say whether a step's ``if`` falls outside ``STEP_CONDITIONS``."""

    condition = step.get("if", "success()")
    if not isinstance(condition, str):
        return True
    text = condition.strip(" ")
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip(" ")
    return text not in STEP_CONDITIONS


def _aggregate_errors(
    job: Mapping[object, object],
    needs: Sequence[str],
    steps: Sequence[Mapping[object, object]],
    others: Sequence[object],
) -> list[str]:
    """Return errors unless ``ci`` always runs, needs every other job, and
    does nothing but check out the repository and run ``require-results``."""

    errors: list[str] = []
    if job.get("if") != "always()":
        errors.append(
            f"{AGGREGATE_JOB}: `if` must read always(), because GitHub reports a "
            "skipped required check as passing"
        )
    errors.extend(
        f"{AGGREGATE_JOB}: needs must list {name}"
        for name in others
        if name not in needs
    )
    errors.extend(
        f"{AGGREGATE_JOB}: needs lists {name}, which the workflow does not declare"
        for name in needs
        if name not in others
    )
    if any(key not in AGGREGATE_JOB_KEYS for key in job):
        errors.append(
            f"{AGGREGATE_JOB}: job may set only {', '.join(AGGREGATE_JOB_KEYS)}"
        )
    checkout, run = steps if len(steps) == 2 else ({}, {})
    if not str(checkout.get("uses")).startswith("actions/checkout@") or any(
        key not in ("name", "uses", "with") for key in checkout
    ):
        errors.append(
            f"{AGGREGATE_JOB}: steps must be a checkout that sets only `name`, "
            "`uses` and `with`, then one `run` step"
        )
    elif (
        run.get("run") != REQUIRE_RESULTS_COMMAND
        or run.get("env") != {"NEEDS": NEEDS_EXPRESSION}
        or any(key not in ("name", "env", "run") for key in run)
    ):
        errors.append(
            f"{AGGREGATE_JOB}: its `run` step must set `run` to "
            f"`{REQUIRE_RESULTS_COMMAND}` on one line, `env` to NEEDS: "
            f"{NEEDS_EXPRESSION} alone, and no other key but `name`"
        )
    return errors


def _selection_job_errors(
    job: Mapping[object, object], steps: Sequence[Mapping[object, object]]
) -> list[str]:
    """Return errors unless the selection job publishes what its step writes."""

    errors: list[str] = []
    wanted = {
        output: f"${{{{ steps.{SELECTION_STEP}.outputs.{output} }}}}"
        for output in select_countries(())
    }
    outputs = job.get("outputs")
    if not isinstance(outputs, Mapping) or any(
        outputs.get(output) != value for output, value in wanted.items()
    ):
        errors.append(
            f"{SELECTION_JOB}: outputs must map each of {', '.join(wanted)} to "
            f"`${{{{ steps.{SELECTION_STEP}.outputs.<its name> }}}}`"
        )
    plans = [step for step in steps if step.get("id") == SELECTION_STEP]
    if len(plans) != 1 or plans[0].get("run") != SELECTION_COMMAND:
        errors.append(
            f"{SELECTION_JOB}: one step with id `{SELECTION_STEP}` must run "
            f"`{SELECTION_COMMAND}` on one line"
        )
    return errors


def workflow_document_errors(document: object) -> tuple[str, ...]:
    """Return an error for each way the parsed workflow breaks the test plan.

    ``document`` holds the workflow as a YAML parser reads it. These rules keep
    a failure from passing. The plan and the workflow declare the same jobs,
    no job or step sets ``continue-on-error``, no step can skip while its job
    passes, every ``run`` step runs under ``bash -eo pipefail``, and no job sets
    ``name``. ``ci`` runs whatever the other jobs report, needs each of them,
    and does nothing but hand their results to ``require-results``. The
    selection job publishes the outputs its ``plan`` step writes. Two more
    rules keep a run from failing for no fault of its
    pull request. Each other job carries the ``if`` the plan gives it, and
    needs ``select-countries`` when that ``if`` reads its outputs. A job needs
    only jobs that run whenever it runs.
    """

    jobs = document.get("jobs") if isinstance(document, Mapping) else None
    if not isinstance(jobs, Mapping):
        return ("workflow has no jobs mapping",)
    expected = (*plan_jobs(), AGGREGATE_JOB)
    errors = [
        f"{name}: the test plan doesn't declare this workflow job"
        for name in jobs
        if name not in expected
    ]
    if document.get("defaults") != WORKFLOW_DEFAULTS:
        errors.append(
            "workflow: `defaults` must set `run: shell: bash` and nothing else, "
            "so every `run` step runs under `bash -eo pipefail` and a pipe cannot "
            "hide a failed command"
        )
    errors.extend(
        f"{name}: the workflow lacks this job of the test plan"
        for name in expected
        if name not in jobs
    )
    for name, job in jobs.items():
        if not isinstance(job, Mapping):
            errors.append(f"{name}: job must be a mapping")
            continue
        needs, steps = _job_needs(job), _job_steps(job)
        if needs is None:
            errors.append(f"{name}: needs must be a job name or a list of job names")
            needs = ()
        if steps is None:
            errors.append(f"{name}: steps must be a list of mappings")
            steps = []
        if name == AGGREGATE_JOB:
            others = [other for other in jobs if other != AGGREGATE_JOB]
            errors.extend(_aggregate_errors(job, needs, steps, others))
            continue
        if "continue-on-error" in job:
            errors.append(
                f"{name}: job sets `continue-on-error`, which can let the run "
                "pass when the job fails"
            )
        if "name" in job:
            errors.append(
                f"{name}: job sets `name`, which would report the job under "
                "another check name"
            )
        within = tuple(_steps_within(steps))
        if any("continue-on-error" in step for step in within):
            errors.append(
                f"{name}: a step sets `continue-on-error`, which can let the job "
                "pass when the step fails"
            )
        if any(map(_skips_while_passing, within)):
            errors.append(
                f"{name}: a step sets an `if` other than `always()`, `success()` "
                "or `!cancelled()`. Write `if: always()`, `if: ${{ !cancelled() }}` "
                "or no `if`, so the step runs whenever its job passes"
            )
        if "defaults" in job or any("shell" in step for step in within):
            errors.append(
                f"{name}: job sets `defaults` or a step sets `shell`, which "
                "can replace the workflow's `bash -eo pipefail`"
            )
        if name == SELECTION_JOB:
            errors.extend(_selection_job_errors(job, steps))
        condition, output = selection_condition(name), selection_output(name)
        if condition is None and "if" in job:
            errors.append(
                f"{name}: job must have no `if`, because the plan runs it on "
                f"every run and {AGGREGATE_JOB} refuses its skip"
            )
        elif condition is not None and job.get("if") != condition:
            errors.append(f"{name}: `if` must read {condition}")
        elif condition is not None and SELECTION_JOB not in needs:
            errors.append(f"{name}: needs must list {SELECTION_JOB}")
        errors.extend(
            f"{name}: needs {needed}, which the selection may leave out when "
            f"it runs {name}, and GitHub would then skip {name} too"
            for needed in needs
            if selection_output(needed) not in (None, output)
        )
    return tuple(errors)


def _job_result(needs: Mapping[str, object], job: str) -> object:
    entry = needs.get(job)
    return entry.get("result") if isinstance(entry, Mapping) else None


def _unplanned_skip(needs: Mapping[str, object], output: str | None) -> str | None:
    """Say why the selection did not ask for this skip, or ``None`` when it did."""

    if output is None:
        return "but the plan runs it on every run"
    if _job_result(needs, SELECTION_JOB) != "success":
        return f"and {SELECTION_JOB} did not succeed"
    entry = needs.get(SELECTION_JOB)
    outputs = entry.get("outputs") if isinstance(entry, Mapping) else None
    selected = outputs.get(output) if isinstance(outputs, Mapping) else None
    if selected == "false":
        return None
    if selected == "true":
        return f"but {SELECTION_JOB} selected it ({output}=true)"
    return f"and {SELECTION_JOB} did not report {output} as true or false"


def aggregate_result_errors(
    needs: Mapping[str, object],
    *,
    jobs: Sequence[str],
    selected_by: Mapping[str, str],
) -> tuple[str, ...]:
    """Return errors unless every job passed, or the selection left it out.

    ``needs`` holds GitHub's ``needs`` context for the aggregate job: one
    entry per needed job with its ``result`` (``success``, ``failure``,
    ``cancelled`` or ``skipped``) and its ``outputs``. ``selected_by`` names
    the selection output that decides each job the selection may leave out. A
    skip passes only when ``selected_by`` names the job, the selection job
    succeeded and that output reads ``false``. Any other result, a missing job
    and a job outside ``jobs`` each add an error, so the aggregate job passes
    only when every job in ``jobs`` reported, none failed, and nothing skipped
    that the selection asked for.
    """

    errors: list[str] = []
    for job in jobs:
        result = _job_result(needs, job)
        if result == "success":
            continue
        if result is None:
            errors.append(f"{job}: reported no result")
        elif result != "skipped":
            errors.append(f"{job}: {result}")
        else:
            reason = _unplanned_skip(needs, selected_by.get(job))
            if reason is not None:
                errors.append(f"{job}: skipped, {reason}")
    errors.extend(
        f"{job}: reported a result, but the plan declares no such job"
        for job in needs
        if job not in jobs
    )
    return tuple(errors)


def engine_guard_lines(source: str) -> tuple[int, ...]:
    """Return lines containing country-engine availability checks."""

    engine_modules = {
        spec.engine_module
        for spec in TEST_GROUPS.values()
        if spec.engine_module is not None
    }
    engine_markers = {
        f"requires_{spec.country}"
        for spec in TEST_GROUPS.values()
        if spec.engine_module is not None
    }
    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and first.value in engine_modules):
                continue
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in {
                "importorskip",
                "find_spec",
            }:
                lines.add(node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                expression = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if (
                    isinstance(expression, ast.Attribute)
                    and expression.attr in engine_markers
                ):
                    lines.add(decorator.lineno)
    return tuple(sorted(lines))


def test_files() -> tuple[Path, ...]:
    """Return every test module below a workspace package's test directory."""

    return tuple(
        sorted(
            path
            for package in PACKAGES.iterdir()
            if package.is_dir()
            for path in (package / "tests").rglob("test_*.py")
            if path.is_file()
        )
    )


def _relative_test_parts(path: Path | PurePosixPath) -> tuple[str, ...] | None:
    parts = path.parts
    try:
        tests_index = max(
            index for index, component in enumerate(parts) if component == "tests"
        )
    except ValueError:
        return None
    return parts[tests_index + 1 :]


def group_name_for_collection_path(path: Path | PurePosixPath) -> str | None:
    """Return the registered group containing a pytest collection path."""

    relative = _relative_test_parts(path)
    if relative is None or len(relative) < 2:
        return None
    return GROUP_BY_DIRECTORY.get((relative[0], relative[1]))


def group_name_for_test_path(path: Path | PurePosixPath) -> str | None:
    """Return the registered group for a test module with a valid layout."""

    relative = _relative_test_parts(path)
    if relative is None:
        return None
    if len(relative) != 3:
        return None
    return GROUP_BY_DIRECTORY.get((relative[0], relative[1]))


def selected_files(group: str) -> tuple[Path, ...]:
    """Return the files belonging to one registered group."""

    if group not in TEST_GROUPS:
        raise SystemExit(f"unknown group: {group}")
    files = tuple(
        path for path in test_files() if group_name_for_test_path(path) == group
    )
    if not files:
        raise SystemExit(f"group is empty: {group}")
    return files


def selected_files_for_job(job: str) -> tuple[Path, ...]:
    """Return all files owned by one CI job."""

    if job not in JOBS:
        raise SystemExit(f"unknown job: {job}")
    files = tuple(
        path
        for group, spec in TEST_GROUPS.items()
        if spec.job == job
        for path in selected_files(group)
    )
    if not files:
        raise SystemExit(f"job has no test files: {job}")
    return files


def reporting_groups_for_job(job: str) -> dict[str, str]:
    """Return report-category to test-group mappings declared for ``job``."""

    if job not in JOBS:
        raise SystemExit(f"unknown job: {job}")
    return {
        spec.report_category: group
        for group, spec in TEST_GROUPS.items()
        if spec.job == job and spec.report_category is not None
    }


def describe() -> str:
    """Return a human-readable description generated from the registry."""

    lines = [
        "group\tdirectory\tjob\tcountry\treport_category\tengine_module\tintegration"
    ]
    for name, spec in TEST_GROUPS.items():
        lines.append(
            "\t".join(
                (
                    name,
                    "/".join(spec.directory),
                    spec.job,
                    spec.country,
                    spec.report_category or "-",
                    spec.engine_module or "-",
                    str(spec.integration).lower(),
                )
            )
        )
    return "\n".join(lines) + "\n"


def verify() -> None:
    """Fail if a test path, an engine dependency or a group disagrees with the plan.

    ``verify`` runs on bare ``python3`` and reads no YAML. The workflow, its
    jobs included, is ``verify-workflow``'s to check.
    """

    files = test_files()
    if not files:
        raise SystemExit("no test files found below packages/*/tests")

    errors = list(selection_errors())
    errors.extend(plan_tests_errors())
    errors.extend(us_runner_errors())
    counts = {name: 0 for name in TEST_GROUPS}
    seen_directories: dict[tuple[str, str], str] = {}
    seen_report_categories: dict[tuple[str, str], str] = {}
    for name, spec in TEST_GROUPS.items():
        if spec.directory in seen_directories:
            errors.append(
                f"{name}: directory {spec.directory!r} is already owned by "
                f"{seen_directories[spec.directory]}"
            )
        seen_directories[spec.directory] = name
        if spec.report_category is not None:
            report_key = (spec.job, spec.report_category)
            if report_key in seen_report_categories:
                errors.append(
                    f"{name}: report category {report_key!r} is already owned by "
                    f"{seen_report_categories[report_key]}"
                )
            seen_report_categories[report_key] = name

    for path in files:
        relative = path.relative_to(ROOT)
        group = group_name_for_test_path(path)
        if group is None:
            errors.append(f"{relative}: expected tests/<environment>/<scope>/test_*.py")
            continue
        counts[group] += 1
        guards = engine_guard_lines(path.read_text(encoding="utf-8"))
        if guards:
            errors.append(
                f"{relative}: engine availability must be expressed by its directory, "
                f"not a pytest skip or import guard (lines {guards})"
            )

    support_tests = sorted((ROOT / "test_support").rglob("test_*.py"))
    errors.extend(
        f"{path.relative_to(ROOT)}: shared support modules must not be test modules"
        for path in support_tests
    )
    errors.extend(
        f"{name}: group is empty" for name, count in counts.items() if count == 0
    )
    if errors:
        raise SystemExit("invalid test plan:\n  " + "\n  ".join(errors))

    print(f"test_files={len(files)}")
    for name, count in counts.items():
        print(f"{name}={count}")
    print("verification=ok")


def scope_for_path(value: str) -> str | None:
    """Return ``shared``, ``us``, ``uk``, or ``None`` for documentation.

    A file below ``tools/`` belongs to a country when that country is a word of
    its file name and the other country's letters appear nowhere in the name.
    A word is a maximal run of ASCII letters, read without case:
    ``run_engine_us_tests.sh`` holds the words run, engine, us, tests and sh.
    ``status.py`` holds no country as a word, and
    ``census_uk_local_targets.py`` holds the letters us inside census, so both
    are shared. A name that holds a country's letters anywhere therefore
    selects that country's tests.
    """

    path = PurePosixPath(value)
    normalized = path.as_posix()
    if (
        normalized in DOC_FILES
        or normalized.startswith(DOC_PREFIXES)
        or path.suffix.lower() in DOC_SUFFIXES
    ):
        return None

    group = group_name_for_test_path(path)
    if group is not None:
        return TEST_GROUPS[group].country
    if normalized.startswith("build/us/"):
        return "us"
    if normalized.startswith("build/uk/"):
        return "uk"
    if "/src/microcosm/build/us_runtime/" in f"/{normalized}":
        return "us"
    if "/src/microcosm/build/uk_runtime/" in f"/{normalized}":
        return "uk"
    if normalized.startswith("tools/"):
        lowered = path.name.lower()
        words = {word.lower() for word in _WORD.findall(path.name)}
        for country, other in (("us", "uk"), ("uk", "us")):
            if country in words and other not in lowered:
                return country
    return "shared"


def select_countries(paths: Iterable[str]) -> dict[str, bool]:
    """Return the expensive country environments affected by changed paths."""

    values = tuple(dict.fromkeys(paths))
    if not values:
        return {"run_us": True, "run_uk": True}
    scopes = {scope_for_path(path) for path in values}
    if "shared" in scopes:
        return {"run_us": True, "run_uk": True}
    return {"run_us": "us" in scopes, "run_uk": "uk" in scopes}


def api_json(url: str, token: str) -> Any:
    """Read one GitHub API response as JSON."""

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def _file_paths(files: Iterable[Mapping[str, object]]) -> list[str]:
    paths: list[str] = []
    for file in files:
        paths.append(str(file["filename"]))
        previous = file.get("previous_filename")
        if previous:
            paths.append(str(previous))
    return paths


def pull_request_paths(
    event: Mapping[str, Any],
    repository: str,
    token: str,
    *,
    request_json: RequestJSON = api_json,
) -> list[str]:
    """Return every current and previous path from a pull request diff.

    GitHub lists at most ``LISTED_FILES_LIMIT`` files. A listing that reaches
    the limit may leave files out, so this returns an empty list for it, and
    an empty list selects both countries.
    """

    number = event["pull_request"]["number"]
    paths: list[str] = []
    listed = 0
    page = 1
    while True:
        files = request_json(
            f"{API_ROOT}/repos/{repository}/pulls/{number}/files"
            f"?per_page=100&page={page}",
            token,
        )
        if not isinstance(files, list):
            raise TypeError("pull request files response must be a list")
        listed += len(files)
        if listed >= LISTED_FILES_LIMIT:
            return []
        paths.extend(_file_paths(files))
        if len(files) < 100:
            return paths
        page += 1


def selection_for_event(
    event_name: str,
    event: Mapping[str, Any],
    repository: str,
    token: str,
    *,
    request_json: RequestJSON = api_json,
) -> tuple[list[str], dict[str, bool]]:
    """Return changed paths and country selection for one workflow event."""

    if event_name == "push":
        return [], {"run_us": True, "run_uk": True}
    if event_name != "pull_request":
        raise ValueError(f"unsupported GitHub event: {event_name}")
    paths = pull_request_paths(event, repository, token, request_json=request_json)
    return paths, select_countries(paths)


def write_country_selection() -> None:
    """Write the selected country jobs to the GitHub Actions output file."""

    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    event_name = os.environ["GITHUB_EVENT_NAME"]
    paths, selected = selection_for_event(
        event_name,
        event,
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_TOKEN"],
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for name, value in selected.items():
            print(f"{name}={str(value).lower()}", file=output)
    if event_name == "push":
        print("main push: all country jobs selected")
    elif paths:
        print("changed paths:")
        for path in paths:
            print(path)
    else:
        print(
            "pull request lists no changed paths, or GitHub's limit of "
            f"{LISTED_FILES_LIMIT:,} files: all country jobs selected"
        )
    print("country selection:", selected)


def verify_workflow() -> None:
    """Fail unless the workflow loads and ``workflow_document_errors`` finds nothing."""

    try:
        document = load_workflow(WORKFLOW.read_text(encoding="utf-8"))
    except ImportError as error:
        raise SystemExit(
            "verify-workflow needs PyYAML: run "
            "`uv run python tools/ci_test_plan.py verify-workflow`"
        ) from error
    except ValueError as error:
        raise SystemExit(str(error)) from error
    errors = workflow_document_errors(document)
    if errors:
        raise SystemExit("workflow fails the test plan:\n  " + "\n  ".join(errors))
    print("workflow=ok")


def require_results() -> None:
    """Fail unless ``NEEDS`` shows each job passed, or the selection left it out."""

    jobs = plan_jobs()
    selected_by = {
        job: output for job in jobs if (output := selection_output(job)) is not None
    }
    try:
        needs = json.loads(os.environ["NEEDS"])
    except (KeyError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"NEEDS must hold the workflow's `needs` context as JSON: {error!r}"
        ) from error
    if not isinstance(needs, Mapping):
        raise SystemExit("NEEDS must hold a JSON object, one entry per needed job")
    for job in jobs:
        print(f"{job}={_job_result(needs, job)}", flush=True)
    errors = aggregate_result_errors(needs, jobs=jobs, selected_by=selected_by)
    if errors:
        raise SystemExit("workflow did not pass:\n  " + "\n  ".join(errors))
    print("results=ok", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("verify-workflow")
    commands.add_parser("describe")
    list_parser = commands.add_parser("list")
    list_parser.add_argument("group", choices=tuple(TEST_GROUPS))
    list_job_parser = commands.add_parser("list-job")
    list_job_parser.add_argument("job", choices=JOBS)
    commands.add_parser("select-countries")
    commands.add_parser("require-results")
    args = parser.parse_args(argv)

    if args.command == "verify":
        verify()
    elif args.command == "verify-workflow":
        verify_workflow()
    elif args.command == "describe":
        print(describe(), end="")
    elif args.command == "list":
        for path in selected_files(args.group):
            print(path.relative_to(ROOT))
    elif args.command == "list-job":
        for path in selected_files_for_job(args.job):
            print(path.relative_to(ROOT))
    elif args.command == "select-countries":
        write_country_selection()
    elif args.command == "require-results":
        require_results()
    return 0


if __name__ == "__main__":
    sys.exit(main())
