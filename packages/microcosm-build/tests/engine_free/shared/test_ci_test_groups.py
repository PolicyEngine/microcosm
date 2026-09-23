"""Tests for the directory-defined CI test inventory."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import ci_test_groups


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("engine_free/shared/test_one.py", ("engine_free", "shared")),
        ("engine_free/us/test_one.py", ("engine_free", "us")),
        ("engine_free/uk/test_one.py", ("engine_free", "uk")),
        ("engine/us/test_one.py", ("engine", "us")),
        ("engine/uk/test_one.py", ("engine", "uk")),
        ("integration/uk/test_one.py", ("integration", "uk")),
    ],
)
def test_category_is_derived_from_the_directory(
    tmp_path: Path, relative: str, expected: tuple[str, str]
) -> None:
    path = tmp_path / "package" / "tests" / relative
    assert ci_test_groups.category(path) == expected


def test_engine_guard_detection_ignores_strings_but_finds_executable_guards() -> None:
    source = """
TEXT = '@pytest.mark.requires_uk'

def test_example():
    pytest.importorskip("policyengine_uk")
"""
    assert ci_test_groups.engine_guard_lines(source) == (5,)


def test_every_repository_test_has_a_valid_execution_category() -> None:
    invalid = [
        path.relative_to(ci_test_groups.ROOT)
        for path in ci_test_groups.test_files()
        if ci_test_groups.category(path) not in ci_test_groups.VALID_CATEGORIES
    ]
    assert invalid == []
