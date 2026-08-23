from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.spec_engine.errors import SpecParseError
from microcosm.build.spec_engine.yaml12 import load_yaml12, load_yaml12_file


def test_yaml12_core_scalars_are_json_compatible() -> None:
    value = load_yaml12(
        """
truth: true
falsehood: FALSE
yaml_11_words: [yes, No, ON, off]
decimal: 012
octal: 0o12
hexadecimal: 0x12
exponent: 1e3
null_value: null
quoted_date: "2026-08-16"
"""
    )

    assert value == {
        "truth": True,
        "falsehood": False,
        "yaml_11_words": ["yes", "No", "ON", "off"],
        "decimal": 12,
        "octal": 10,
        "hexadecimal": 18,
        "exponent": 1000.0,
        "null_value": None,
        "quoted_date": "2026-08-16",
    }
    json.dumps(value, allow_nan=False)


def test_yaml12_does_not_mutate_pyyaml_safe_loader() -> None:
    import yaml

    assert yaml.safe_load("word: yes") == {"word": True}
    assert load_yaml12("word: yes") == {"word": "yes"}


@pytest.mark.parametrize(
    ("document", "message", "line", "column"),
    [
        ("a: 1\na: 2\n", "duplicate mapping key 'a'", 2, 1),
        (
            "outer:\n  first: 1\n  first: 2\n",
            "duplicate mapping key 'first'",
            3,
            3,
        ),
        (
            "base: &base {a: 1}\ncopy:\n  <<: *base\n",
            "YAML merge keys are not allowed",
            3,
            3,
        ),
        ("value: !!str 1\n", "explicit YAML tags are not allowed", 1, 8),
        ("value: !custom 1\n", "explicit YAML tags are not allowed", 1, 8),
        ("date: 2026-08-16\n", "timestamps and dates are not allowed", 1, 7),
        (
            "time: 2026-08-16T12:34:56Z\n",
            "timestamps and dates are not allowed",
            1,
            7,
        ),
        ("value: .nan\n", "non-finite numbers are not allowed", 1, 8),
        ("value: -.NaN\n", "non-finite numbers are not allowed", 1, 8),
        ("value: +.INF\n", "non-finite numbers are not allowed", 1, 8),
        ("value: 1e9999\n", "non-finite numbers are not allowed", 1, 8),
        ("value: 0o_\n", "invalid YAML 1.2 number", 1, 8),
        ("1: value\n", "mapping keys must be strings", 1, 1),
        ("true: value\n", "mapping keys must be strings", 1, 1),
        ("? [a, b]\n: value\n", "mapping keys must be strings", 1, 3),
        (
            "first: 1\n---\nsecond: 2\n",
            "multiple YAML documents are not allowed",
            2,
            1,
        ),
        (
            "cycle: &cycle\n  - *cycle\n",
            "cyclic YAML aliases are not allowed",
            1,
            8,
        ),
        (
            "%YAML 1.1\n---\nvalue: true\n",
            "only the YAML 1.2 directive is allowed",
            1,
            1,
        ),
    ],
)
def test_rejected_yaml_has_deterministic_source_and_mark(
    document: str,
    message: str,
    line: int,
    column: int,
) -> None:
    with pytest.raises(SpecParseError) as caught:
        load_yaml12(document, source="fixture.yaml")

    error = caught.value
    assert error.message == message
    assert (error.source, error.line, error.column) == (
        "fixture.yaml",
        line,
        column,
    )
    assert str(error) == f"fixture.yaml:{line}:{column}: {message}"


def test_noncyclic_aliases_are_allowed_and_materialized_as_json() -> None:
    value = load_yaml12("first: &values [1, 2]\nsecond: *values\n")

    assert value == {"first": [1, 2], "second": [1, 2]}
    json.dumps(value, allow_nan=False)


def test_quoted_merge_spelling_is_an_ordinary_string_key() -> None:
    assert load_yaml12('"<<": literal\n') == {"<<": "literal"}


def test_single_explicit_yaml_12_document_is_allowed() -> None:
    assert load_yaml12("%YAML 1.2\n---\nvalue: false\n") == {"value": False}


def test_scanner_error_is_normalized() -> None:
    with pytest.raises(
        SpecParseError,
        match=r"^bad.yaml:2:1: invalid YAML:",
    ):
        load_yaml12("value: [\n", source="bad.yaml")


def test_file_loader_reports_the_resource_path(tmp_path: Path) -> None:
    path = tmp_path / "resource.yaml"
    path.write_text("nested:\n  key: 1\n  key: 2\n", encoding="utf-8")

    with pytest.raises(SpecParseError) as caught:
        load_yaml12_file(path)

    assert caught.value.source == str(path)
    assert (caught.value.line, caught.value.column) == (3, 3)
