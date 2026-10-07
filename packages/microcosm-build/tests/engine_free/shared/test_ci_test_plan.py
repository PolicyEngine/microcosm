"""Tests for the single declarative CI test plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import ci_test_plan


@pytest.mark.parametrize(
    (
        "relative",
        "expected_group",
        "expected_job",
        "expected_country",
        "expected_engine",
        "expected_integration",
    ),
    [
        (
            "engine_free/shared/test_one.py",
            "engine-free-shared",
            "engine-free",
            "shared",
            None,
            False,
        ),
        (
            "engine_free/us/test_one.py",
            "engine-free-us",
            "engine-free",
            "us",
            None,
            False,
        ),
        (
            "engine_free/uk/test_one.py",
            "engine-free-uk",
            "engine-free",
            "uk",
            None,
            False,
        ),
        (
            "engine_contract/us/test_one.py",
            "engine-contract-us",
            "engine-us",
            "us",
            "policyengine_us",
            False,
        ),
        (
            "engine_scenario/us/test_one.py",
            "engine-scenario-us",
            "engine-us",
            "us",
            "policyengine_us",
            False,
        ),
        (
            "engine_workflow/us/test_one.py",
            "engine-workflow-us",
            "engine-us",
            "us",
            "policyengine_us",
            False,
        ),
        (
            "engine/uk/test_one.py",
            "engine-uk",
            "engine-uk",
            "uk",
            "policyengine_uk",
            False,
        ),
        (
            "integration/uk/test_one.py",
            "integration-uk",
            "integration-uk",
            "uk",
            "policyengine_uk",
            True,
        ),
    ],
)
def test_registry_is_the_only_test_directory_and_job_authority(
    tmp_path: Path,
    relative: str,
    expected_group: str,
    expected_job: str,
    expected_country: str,
    expected_engine: str | None,
    expected_integration: bool,
) -> None:
    path = tmp_path / "package" / "tests" / relative

    assert ci_test_plan.group_name_for_test_path(path) == expected_group
    spec = ci_test_plan.TEST_GROUPS[expected_group]
    assert spec.job == expected_job
    assert spec.country == expected_country
    assert spec.engine_module == expected_engine
    assert spec.integration is expected_integration


def test_collection_directories_resolve_through_the_same_registry(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "package" / "tests" / "engine_contract" / "us"

    assert ci_test_plan.group_name_for_collection_path(directory) == (
        "engine-contract-us"
    )


def test_group_resolution_uses_the_innermost_tests_directory() -> None:
    path = Path(
        "/temporary/tests/checkout/packages/example/tests/engine_scenario/us/test_one.py"
    )

    assert ci_test_plan.group_name_for_test_path(path) == "engine-scenario-us"


def test_registry_directories_are_unique() -> None:
    directories = [spec.directory for spec in ci_test_plan.TEST_GROUPS.values()]

    assert len(directories) == len(set(directories))


def test_workflow_jobs_match_registered_test_and_infrastructure_jobs() -> None:
    source = ci_test_plan.WORKFLOW.read_text(encoding="utf-8")

    assert ci_test_plan.workflow_job_errors(source) == ()


def test_workflow_rejects_a_separate_behavioral_test_job() -> None:
    source = """\
jobs:
  engine-free:
    steps:
      - run: pytest
  invented-compatibility-test:
    steps:
      - run: node verify.mjs
"""

    assert (
        "invented-compatibility-test: workflow job has no registered test "
        "category or approved infrastructure role"
        in ci_test_plan.workflow_job_errors(source)
    )


def test_workflow_rejects_an_underscore_prefixed_job() -> None:
    source = """\
jobs:
  engine-free:
    steps:
      - run: pytest
  _invented-compatibility-test:
    steps:
      - run: node verify.mjs
"""

    assert (
        "_invented-compatibility-test: workflow job has no registered test "
        "category or approved infrastructure role"
        in ci_test_plan.workflow_job_errors(source)
    )


def test_workflow_job_parser_ignores_nested_yaml_keys() -> None:
    source = """\
jobs:
  engine-free:
    strategy:
      matrix:
        python-version: ["3.13", "3.14"]
    steps:
      - name: Run tests
        run: pytest
"""

    assert ci_test_plan.workflow_job_names(source) == ("engine-free",)


def test_engine_guard_detection_ignores_strings_but_finds_executable_guards() -> None:
    source = """
TEXT = '@pytest.mark.requires_uk'

def test_example():
    pytest.importorskip("policyengine_uk")
"""
    assert ci_test_plan.engine_guard_lines(source) == (5,)


def test_every_repository_test_has_a_valid_execution_category() -> None:
    invalid = [
        path.relative_to(ci_test_plan.ROOT)
        for path in ci_test_plan.test_files()
        if ci_test_plan.group_name_for_test_path(path) is None
    ]
    assert invalid == []


def test_every_test_is_owned_by_exactly_one_job() -> None:
    selected = [
        path
        for job in ci_test_plan.JOBS
        for path in ci_test_plan.selected_files_for_job(job)
    ]

    assert len(selected) == len(ci_test_plan.test_files())
    assert len(selected) == len(set(selected))


def test_us_reporting_categories_come_from_the_registry() -> None:
    assert ci_test_plan.reporting_groups_for_job("engine-us") == {
        "contract": "engine-contract-us",
        "scenario": "engine-scenario-us",
        "workflow": "engine-workflow-us",
    }


def test_unknown_non_documentation_paths_select_both_countries() -> None:
    assert ci_test_plan.scope_for_path("new-area/unrecognized.config") == "shared"
    assert ci_test_plan.select_countries(["new-area/unrecognized.config"]) == {
        "run_us": True,
        "run_uk": True,
    }


@pytest.mark.parametrize(
    ("path", "scope"),
    [
        ("build/us/spec.yaml", "us"),
        ("build/uk/spec.yaml", "uk"),
        ("packages/example/src/microcosm/build/us_runtime/module.py", "us"),
        ("packages/example/src/microcosm/build/uk_runtime/module.py", "uk"),
        ("tools/build_us_candidate.py", "us"),
        ("tools/build_uk_candidate.py", "uk"),
        ("packages/example/src/microcosm/shared.py", "shared"),
    ],
)
def test_country_specific_source_paths_keep_their_existing_scope(
    path: str, scope: str
) -> None:
    assert ci_test_plan.scope_for_path(path) == scope


def test_documentation_only_changes_select_no_country_engine() -> None:
    documentation = [
        "README.md",
        "DESIGN.md",
        "docs/testing.md",
        "changelog.d/change.changed.md",
    ]

    assert ci_test_plan.select_countries(documentation) == {
        "run_us": False,
        "run_uk": False,
    }


def test_country_specific_and_mixed_changes_select_expected_engines() -> None:
    us_test = "packages/example/tests/engine_scenario/us/test_example.py"
    uk_test = "packages/example/tests/integration/uk/test_example.py"

    assert ci_test_plan.select_countries([us_test]) == {
        "run_us": True,
        "run_uk": False,
    }
    assert ci_test_plan.select_countries([uk_test]) == {
        "run_us": False,
        "run_uk": True,
    }
    assert ci_test_plan.select_countries([us_test, uk_test]) == {
        "run_us": True,
        "run_uk": True,
    }
    assert ci_test_plan.select_countries(["README.md", us_test]) == {
        "run_us": True,
        "run_uk": False,
    }
    assert ci_test_plan.select_countries([]) == {"run_us": True, "run_uk": True}


def test_pull_request_paths_fetches_every_page_and_previous_rename_path() -> None:
    calls: list[str] = []
    first_page = [
        {"filename": f"docs/item-{index}.md", "status": "modified"}
        for index in range(100)
    ]
    second_page = [
        {
            "filename": "packages/example/tests/engine_scenario/us/test_new.py",
            "previous_filename": "packages/example/tests/engine/uk/test_old.py",
            "status": "renamed",
        }
    ]

    def request_json(url: str, _token: str):
        calls.append(url)
        return first_page if url.endswith("page=1") else second_page

    paths = ci_test_plan.pull_request_paths(
        {"pull_request": {"number": 42}},
        "PolicyEngine/microcosm",
        "token",
        request_json=request_json,
    )

    assert len(calls) == 2
    assert paths[-2:] == [
        "packages/example/tests/engine_scenario/us/test_new.py",
        "packages/example/tests/engine/uk/test_old.py",
    ]
    assert ci_test_plan.select_countries(paths) == {
        "run_us": True,
        "run_uk": True,
    }


def test_main_push_selects_both_countries_without_requesting_changed_files() -> None:
    def unexpected_request(_url: str, _token: str):
        raise AssertionError("main pushes must not request changed files")

    paths, selected = ci_test_plan.selection_for_event(
        "push",
        {"after": "a" * 40},
        "PolicyEngine/microcosm",
        "token",
        request_json=unexpected_request,
    )

    assert paths == []
    assert selected == {"run_us": True, "run_uk": True}


def test_main_push_writes_both_country_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "output.txt"
    event_path.write_text('{"after": "abc"}', encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REPOSITORY", "PolicyEngine/microcosm")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    ci_test_plan.write_country_selection()

    assert output_path.read_text(encoding="utf-8") == "run_us=true\nrun_uk=true\n"
    assert "main push: all country jobs selected" in capsys.readouterr().out


def test_description_is_generated_from_the_registry() -> None:
    description = ci_test_plan.describe()

    for name, spec in ci_test_plan.TEST_GROUPS.items():
        assert name in description
        assert "/".join(spec.directory) in description
