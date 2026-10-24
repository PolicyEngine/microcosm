"""The F0 pin writer preserves surrounding source and refuses unknown pins."""

import pytest

from tools.repin_us_spec_engine_contracts import (
    _add_count_replacement,
    _replace_assignment,
    _replace_dictionary,
    _replace_length_assertions,
    _replace_literals,
)


def test_repin_replaces_only_exact_literals() -> None:
    source = 'count = 934\nother = 1934\nlabel = "934"\n# 934 is historical\n'
    assert _replace_literals(source, {934: 937}) == (
        'count = 937\nother = 1934\nlabel = "934"\n# 934 is historical\n'
    )


def test_repin_preserves_unchanged_pin_bytes_and_unrelated_comments() -> None:
    source = 'PINS = {\n    "keep": (2, "abc"),  # reviewed\n    "move": 3,\n}\n'
    assert _replace_dictionary(source, "PINS", {"keep": (2, "abc")}) == source
    assert _replace_dictionary(source, "PINS", {"move": 4}) == source.replace(
        '"move": 3', '"move": 4'
    )


def test_repin_refuses_unknown_assignments_and_dictionary_entries() -> None:
    with pytest.raises(ValueError, match="missing assignment"):
        _replace_assignment("KNOWN = 1\n", "UNKNOWN", 2)
    with pytest.raises(ValueError, match="expected one key"):
        _replace_dictionary('PINS = {"known": 1}\n', "PINS", {"unknown": 2})


def test_repin_refuses_colliding_count_transitions() -> None:
    replacements = {}
    _add_count_replacement(replacements, 176, 179)
    _add_count_replacement(replacements, 176, 179)
    with pytest.raises(ValueError, match="ambiguous count replacement"):
        _add_count_replacement(replacements, 176, 180)


def test_inventory_count_updates_only_the_reviewed_length_target() -> None:
    source = "assert len(columns) == 176\nassert len(unrelated) == 176\n"
    assert _replace_length_assertions(source, {"columns": 179}) == (
        "assert len(columns) == 179\nassert len(unrelated) == 176\n"
    )
    with pytest.raises(ValueError, match="targets changed"):
        _replace_length_assertions(source, {"missing": 179})


def test_inventory_count_update_refuses_duplicate_targets() -> None:
    source = "assert len(columns) == 176\nassert len(columns) == 176\n"
    with pytest.raises(ValueError, match="duplicate inventory assertion"):
        _replace_length_assertions(source, {"columns": 179})
