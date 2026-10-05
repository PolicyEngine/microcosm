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
from collections.abc import Callable, Iterable, Mapping
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
}
GROUP_BY_DIRECTORY = {spec.directory: name for name, spec in TEST_GROUPS.items()}
JOBS = tuple(dict.fromkeys(spec.job for spec in TEST_GROUPS.values()))
WORKFLOW_INFRASTRUCTURE_JOBS = {
    "select-countries": "selects the country-specific test categories",
    "lint": "runs static checks and verifies this test plan",
    "wheels": "builds and inspects distribution archives",
}

DOC_PREFIXES = ("docs/",)
DOC_FILES = {"LICENSE"}
DOC_SUFFIXES = {".md", ".rst"}

RequestJSON = Callable[[str, str], Any]

_WORKFLOW_JOB = re.compile(r"^  ([A-Za-z0-9][A-Za-z0-9_-]*):(?:\s.*)?$")


def workflow_job_names(source: str) -> tuple[str, ...]:
    """Return top-level job names from the repository's workflow YAML."""

    in_jobs = False
    names: list[str] = []
    for line in source.splitlines():
        if not in_jobs:
            if line == "jobs:":
                in_jobs = True
            continue
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        match = _WORKFLOW_JOB.fullmatch(line)
        if match is not None:
            names.append(match.group(1))
    if not in_jobs:
        raise ValueError("workflow has no top-level jobs mapping")
    return tuple(names)


def workflow_job_errors(source: str) -> tuple[str, ...]:
    """Return errors for workflow jobs outside the declared test plan."""

    try:
        names = workflow_job_names(source)
    except ValueError as error:
        return (str(error),)

    errors: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            errors.append(f"{name}: workflow job is declared more than once")
        seen.add(name)

    expected = set(JOBS) | set(WORKFLOW_INFRASTRUCTURE_JOBS)
    errors.extend(
        f"{name}: workflow job has no registered test category or approved "
        "infrastructure role"
        for name in sorted(seen - expected)
    )
    errors.extend(
        f"{name}: registered workflow job is missing"
        for name in sorted(expected - seen)
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
    """Fail if a test path or engine dependency disagrees with the registry."""

    files = test_files()
    if not files:
        raise SystemExit("no test files found below packages/*/tests")

    errors = list(workflow_job_errors(WORKFLOW.read_text(encoding="utf-8")))
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
    """Return ``shared``, ``us``, ``uk``, or ``None`` for documentation."""

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
        name = path.name.lower()
        if "us" in name and "uk" not in name:
            return "us"
        if "uk" in name and "us" not in name:
            return "uk"
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
    """Return every current and previous path from a pull request diff."""

    number = event["pull_request"]["number"]
    paths: list[str] = []
    page = 1
    while True:
        files = request_json(
            f"{API_ROOT}/repos/{repository}/pulls/{number}/files"
            f"?per_page=100&page={page}",
            token,
        )
        if not isinstance(files, list):
            raise TypeError("pull request files response must be a list")
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
        print("pull request has no changed paths: all country jobs selected")
    print("country selection:", selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("describe")
    list_parser = commands.add_parser("list")
    list_parser.add_argument("group", choices=tuple(TEST_GROUPS))
    list_job_parser = commands.add_parser("list-job")
    list_job_parser.add_argument("job", choices=JOBS)
    commands.add_parser("select-countries")
    args = parser.parse_args(argv)

    if args.command == "verify":
        verify()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
