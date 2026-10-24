"""The F0 pin writer preserves surrounding source and refuses unknown pins."""

import ast

import pytest

from tools.repin_us_spec_engine_contracts import (
    _add_count_replacement,
    _replace_asserted_dictionary_value,
    _replace_assignment,
    _replace_dictionary,
    _replace_length_assertions,
    _replace_literal_assertions,
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
    source = (
        "assert len(columns) == 176\nassert len(unrelated) == 176\n"
        "if len(columns) > 176:\n    pass\n"
    )
    assert _replace_length_assertions(source, {"columns": 179}) == (
        "assert len(columns) == 179\nassert len(unrelated) == 176\n"
        "if len(columns) > 176:\n    pass\n"
    )
    with pytest.raises(ValueError, match="targets changed"):
        _replace_length_assertions(source, {"missing": 179})


def test_inventory_count_update_refuses_duplicate_targets() -> None:
    source = "assert len(columns) == 176\nassert len(columns) == 176\n"
    with pytest.raises(ValueError, match="duplicate inventory assertion"):
        _replace_length_assertions(source, {"columns": 179})


def test_receipt_pin_updates_only_one_exact_literal_assertion() -> None:
    source = (
        'assert first.spec_sha256 == ("old")\n'
        'assert other.spec_sha256 == "old"\n'
        "# old is historical\n"
    )
    assert _replace_literal_assertions(source, {"first.spec_sha256": "new"}) == (
        "assert first.spec_sha256 == ('new')\n"
        'assert other.spec_sha256 == "old"\n'
        "# old is historical\n"
    )
    with pytest.raises(ValueError, match="targets changed"):
        _replace_literal_assertions(source, {"missing": "new"})


def test_receipt_pin_refuses_duplicates_and_nonliteral_expectations() -> None:
    source = 'assert receipt == "old"\nassert receipt == "old"\n'
    with pytest.raises(ValueError, match="duplicate receipt assertion"):
        _replace_literal_assertions(source, {"receipt": "new"})
    with pytest.raises(ValueError):
        _replace_literal_assertions("assert receipt == expected\n", {"receipt": "new"})
    with pytest.raises(ValueError, match="not a literal string"):
        _replace_literal_assertions("assert receipt == 123\n", {"receipt": "new"})


def test_nested_receipt_pin_preserves_the_complete_dictionary_assertion() -> None:
    source = (
        'assert run_config == {"spec_binding": {"sha256": "old", "keep": "old"}}\n'
        "# old is historical\n"
    )
    assert _replace_asserted_dictionary_value(
        source, "run_config", ("spec_binding", "sha256"), "new"
    ) == source.replace('"sha256": "old"', "\"sha256\": 'new'")
    with pytest.raises(ValueError, match="expected one receipt dictionary key"):
        _replace_asserted_dictionary_value(
            source, "run_config", ("spec_binding", "missing"), "new"
        )


def test_nested_receipt_pin_refuses_duplicate_assertions_or_dictionary_keys() -> None:
    source = 'assert run_config == {"sha256": "old"}\n'
    with pytest.raises(ValueError, match="expected one asserted dictionary"):
        _replace_asserted_dictionary_value(
            source + source, "run_config", ("sha256",), "new"
        )
    with pytest.raises(ValueError, match="expected one receipt dictionary key"):
        _replace_asserted_dictionary_value(
            'assert run_config == {"sha256": "old", "sha256": "other"}\n',
            "run_config",
            ("sha256",),
            "new",
        )


def test_calibration_pin_hashes_the_fixture_contract_not_the_loaded_bundle() -> None:
    # The loader writes integral floats as integers (5.0 -> 5), so a digest of
    # the compiled bundle's calibration never matches the golden test's pin.
    from microcosm.build.spec_engine import project_legacy_calibration_contract
    from microcosm.build.spec_engine.canonical import sha256_json
    from tools.repin_us_spec_engine_contracts import (
        PACKAGE,
        _assignment,
        fixture_calibration_contract,
    )

    path = (
        PACKAGE / "tests/engine_free/shared/test_spec_engine_calibration_semantics.py"
    )
    pinned = ast.literal_eval(
        _assignment(path.read_text(), "CANONICAL_LEGACY_PROJECTION_SHA256")
    )
    projected = project_legacy_calibration_contract(fixture_calibration_contract())
    assert sha256_json(projected) == pinned
