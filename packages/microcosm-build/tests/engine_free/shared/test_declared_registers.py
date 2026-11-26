"""Reviewed registers declared as data (microcosm.build.declared_registers).

The registers here are made up. They check how a section reads back: one key
field or a tuple of them, one value field, a tuple of them or none, JSON arrays
as tuples, entry order, and the refusals. No country data is read
(microcosm#1728).
"""

from __future__ import annotations

import copy

import pytest

from microcosm.build.declared_registers import (
    DeclaredRegisterError,
    load_register,
)


def _section(**overrides) -> dict:
    section = {
        "description": ["Columns a package reads under the wrong measure."],
        "key": ["measure", "column"],
        "value": "reason",
        "entries": [
            {"measure": "tax_b", "column": "C2", "reason": "Wrong line."},
            {
                "measure": "tax_a",
                "column": "C1",
                "reason": "Wrong form.",
                "note": ["Checked against the guide."],
            },
        ],
    }
    section.update(overrides)
    return section


def test_a_tuple_key_and_one_value_keep_entry_order() -> None:
    register = load_register(_section())
    assert register == {("tax_b", "C2"): "Wrong line.", ("tax_a", "C1"): "Wrong form."}
    assert list(register) == [("tax_b", "C2"), ("tax_a", "C1")]


def test_one_key_field_and_tuple_values() -> None:
    register = load_register(
        {
            "description": "Variables of each concept.",
            "key": "concept",
            "value": ["variables", "family", "metadata"],
            "entries": [
                {
                    "concept": "income",
                    "variables": ["wages", "tips"],
                    "family": "tax",
                    "metadata": {"role": "x"},
                },
                {
                    "concept": "rent",
                    "variables": "rent",
                    "family": "tax",
                    "metadata": {},
                },
            ],
        }
    )
    assert register == {
        "income": (("wages", "tips"), "tax", {"role": "x"}),
        "rent": ("rent", "tax", {}),
    }
    assert isinstance(register["income"][0], tuple)


def test_a_set_register_and_a_null_key_part() -> None:
    assert load_register(
        {
            "description": "Excluded states.",
            "key": "state",
            "value": None,
            "entries": [{"state": "02"}, {"state": "06"}],
        }
    ) == frozenset({"02", "06"})
    assert load_register(
        {
            "description": "Groups.",
            "key": ["source", "group"],
            "value": "variable",
            "entries": [{"source": "cbo", "group": None, "variable": "v"}],
        }
    ) == {("cbo", None): "v"}


def _broken():
    def change(path, value):
        def mutate(section):
            target = section
            for step in path[:-1]:
                target = target[step]
            if value is _DROP:
                del target[path[-1]]
            else:
                target[path[-1]] = value

        return mutate

    return {
        "no description": change(("description",), _DROP),
        "empty description": change(("description",), []),
        "unknown section key": change(("extra",), 1),
        "repeated key field": change(("key",), ["measure", "measure"]),
        "key and value overlap": change(("value",), "measure"),
        "note as a field": change(("value",), "note"),
        "entries not a list": change(("entries",), {}),
        "missing field": change(("entries", 0, "column"), _DROP),
        "unknown field": change(("entries", 0, "extra"), 1),
        "empty note": change(("entries", 1, "note"), ""),
        "repeated key": change(("entries", 1, "column"), "C2"),
    }


_DROP = object()


@pytest.mark.parametrize("case", sorted(_broken()))
def test_a_malformed_register_refuses(case: str) -> None:
    section = copy.deepcopy(_section())
    if case == "repeated key":
        section["entries"][1]["measure"] = "tax_b"
    _broken()[case](section)
    with pytest.raises(DeclaredRegisterError):
        load_register(section)


# ---------------------------------------------------------------------------
# Signed entries: strict reads and review windows (microcosm#1759)
# ---------------------------------------------------------------------------


def test_read_strict_json_refuses_duplicates_non_finite_and_malformed() -> None:
    from microcosm.build.declared_registers import read_strict_json

    assert read_strict_json('{"a": [1, {"b": 2.5}]}', label="f") == {
        "a": [1, {"b": 2.5}]
    }
    with pytest.raises(
        DeclaredRegisterError, match=r"^f: duplicate JSON key\(s\): \['a'\]\.$"
    ):
        read_strict_json('{"a": 1, "a": 2}', label="f")
    with pytest.raises(
        DeclaredRegisterError, match=r"duplicate JSON key\(s\): \['b'\]"
    ):
        read_strict_json('{"a": {"b": 1, "b": 1}}', label="f")
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(
            DeclaredRegisterError,
            match=rf"^f: non-finite JSON value '{constant}' is invalid\.$",
        ):
            read_strict_json(f'{{"a": {constant}}}', label="f")
    with pytest.raises(DeclaredRegisterError, match=r"^f: malformed JSON: .+\.$"):
        read_strict_json('{"a": }', label="f")
    assert issubclass(DeclaredRegisterError, ValueError)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-11-26", "2026-11-26"),
        ("2024-02-29", "2024-02-29"),
        ("20261126", None),
        ("2026-W48-4", None),
        ("2026-11-26T00:00", None),
        ("2026-02-30", None),
        (" 2026-11-26", None),
        ("", None),
        (None, None),
        (20261126, None),
    ],
)
def test_canonical_date_reads_only_the_canonical_spelling(value, expected) -> None:
    from microcosm.build.declared_registers import canonical_date

    parsed = canonical_date(value)
    assert (None if parsed is None else parsed.isoformat()) == expected


def test_a_review_window_includes_both_days() -> None:
    from datetime import date

    from microcosm.build.declared_registers import WINDOW_STATES, ReviewWindow

    window = ReviewWindow(date(2026, 10, 1), date(2026, 12, 31))
    states = {
        day: window.state(date.fromisoformat(day))
        for day in (
            "2026-09-30",
            "2026-10-01",
            "2026-11-26",
            "2026-12-31",
            "2027-01-01",
        )
    }
    assert states == {
        "2026-09-30": "premature",
        "2026-10-01": "in_force",
        "2026-11-26": "in_force",
        "2026-12-31": "in_force",
        "2027-01-01": "expired",
    }
    assert set(states.values()) == set(WINDOW_STATES)
    assert window.premature(date(2026, 9, 30)) and not window.in_force(
        date(2026, 9, 30)
    )
    assert window.expired(date(2027, 1, 1)) and not window.in_force(date(2027, 1, 1))
    assert window.days_left(date(2026, 12, 24)) == 7
    assert window.days_left(date(2027, 1, 2)) == -2


def test_stale_entries_are_the_declared_names_not_found() -> None:
    from microcosm.build.declared_registers import stale_entries

    assert stale_entries(["b", "a", "c", "a"], {"c"}) == ["a", "b"]
    assert stale_entries([], ["x"]) == []
