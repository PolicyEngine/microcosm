"""Candidate-only eligibility controls; never run against the predecessor."""

import hashlib

import pytest

from microcosm.build.us_runtime import survey_population_preparation as owner


def _value(*, characters=0, ordinal=1, people=()):
    return ((("x" * characters, "1", "0", "0", "h", ordinal, people),), ())


def _person():
    return ("1", "20", "1", "reported", "0", "reported", "1", "p", 1)


@pytest.mark.parametrize(
    "value, expected",
    [
        (((), ()), True),
        (_value(characters=65531), True),
        (_value(characters=65532), True),
        (_value(characters=65533), False),
        (_value(ordinal=-(2**63)), True),
        (_value(ordinal=2**63 - 1), True),
        (_value(ordinal=-(2**63) - 1), False),
        (_value(ordinal=2**63), False),
        (_value(people=(_person(),) * 20), True),
        (_value(people=(_person(),) * 21), False),
        (_value(ordinal=True), False),
        (_value(ordinal=1.0), False),
        (_value(ordinal=None), False),
        (_value(ordinal="é"), False),
        (_value(ordinal="\ud800"), False),
        (([], ()), False),
        ([(), ()], False),
        (((("short",),), ()), False),
    ],
)
def test_exact_eligibility_boundary(value, expected):
    assert owner._catalogue_fast_path(value) is expected


def test_subclass_ineligibility():
    class Text(str):
        pass

    class Integer(int):
        pass

    class Sequence(tuple):
        pass

    for value in [
        _value(ordinal=Text("x")),
        _value(ordinal=Integer(1)),
        Sequence(((), ())),
    ]:
        assert owner._catalogue_fast_path(value) is False


def test_ascii_escaping_uses_eligible_canonical_stream():
    value = ((("a", '"\\\n', "\x00", "0", "h", 1, ()),), ())
    expected = b'[[["a","\\"\\\\\\n","\\u0000","0","h",1,[]]],[]]'
    assert owner._catalogue_fast_path(value) is True
    assert "".join(owner._catalogue_chunks(value)).encode("utf-8") == expected
    assert "".join(owner._chunks(value)).encode("utf-8") == expected
    assert owner._catalogue_digest(value) == hashlib.sha256(expected).hexdigest()
