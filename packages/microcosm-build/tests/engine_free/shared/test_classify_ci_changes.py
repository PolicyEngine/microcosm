"""Tests for the GitHub changed-path classifier."""

from __future__ import annotations

from tools.classify_ci_changes import (
    changed_paths,
    classify_paths,
    pull_request_paths,
    scope_for_path,
)


def test_directory_taxonomy_classifies_country_and_shared_changes() -> None:
    assert (
        scope_for_path("packages/microcosm-build/tests/engine/us/test_engine.py")
        == "us"
    )
    assert (
        scope_for_path("packages/microcosm-build/tests/engine_free/uk/test_contract.py")
        == "uk"
    )
    assert (
        scope_for_path(
            "packages/microcosm-build/tests/engine_free/shared/test_contract.py"
        )
        == "shared"
    )
    assert (
        scope_for_path("packages/microcosm-build/src/microcosm/build/us_runtime/run.py")
        == "us"
    )
    assert (
        scope_for_path("packages/microcosm-build/src/microcosm/build/uk_runtime/run.py")
        == "uk"
    )


def test_unknown_non_documentation_paths_default_to_shared() -> None:
    assert scope_for_path("new-area/unrecognized.config") == "shared"
    assert classify_paths(["new-area/unrecognized.config"]) == {
        "shared": True,
        "us": False,
        "uk": False,
    }


def test_documentation_only_changes_select_no_test_environment() -> None:
    assert classify_paths(["README.md", "docs/testing.md"]) == {
        "shared": False,
        "us": False,
        "uk": False,
    }


def test_empty_and_mixed_change_sets_are_conservative() -> None:
    assert classify_paths([]) == {"shared": True, "us": True, "uk": True}
    assert classify_paths(
        [
            "pyproject.toml",
            "packages/microcosm-build/tests/engine/us/test_engine.py",
            "packages/microcosm-build/tests/engine/uk/test_engine.py",
        ]
    ) == {"shared": True, "us": True, "uk": True}


def test_pull_request_paths_fetches_every_page_and_previous_rename_path() -> None:
    calls: list[str] = []
    first_page = [
        {"filename": f"docs/item-{index}.md", "status": "modified"}
        for index in range(100)
    ]
    second_page = [
        {
            "filename": "packages/example/tests/engine/us/test_new.py",
            "previous_filename": "packages/example/tests/engine/uk/test_old.py",
            "status": "renamed",
        },
        {"filename": "removed.config", "status": "removed"},
    ]

    def request_json(url: str, _token: str):
        calls.append(url)
        return first_page if url.endswith("page=1") else second_page

    paths = pull_request_paths(
        {"pull_request": {"number": 42}},
        "PolicyEngine/microcosm",
        "token",
        request_json=request_json,
    )

    assert len(calls) == 2
    assert paths[-3:] == [
        "packages/example/tests/engine/us/test_new.py",
        "packages/example/tests/engine/uk/test_old.py",
        "removed.config",
    ]


def test_push_comparison_includes_current_and_previous_paths() -> None:
    def request_json(_url: str, _token: str):
        return {
            "files": [
                {
                    "filename": "build/us/new.yaml",
                    "previous_filename": "build/uk/old.yaml",
                    "status": "renamed",
                },
                {"filename": "obsolete.file", "status": "removed"},
            ]
        }

    paths = changed_paths(
        "push",
        {"before": "a" * 40, "after": "b" * 40},
        "PolicyEngine/microcosm",
        "token",
        request_json=request_json,
    )

    assert paths == ["build/us/new.yaml", "build/uk/old.yaml", "obsolete.file"]
