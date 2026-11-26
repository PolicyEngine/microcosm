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
