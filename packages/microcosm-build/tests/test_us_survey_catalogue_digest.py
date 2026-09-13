"""Exact canonical catalogue bytes and unchanged fallback refusals."""

import hashlib

import pytest

from microcosm.build.us_runtime import survey_population_preparation as owner

chunks = owner._catalogue_chunks
digest = owner._catalogue_digest


def _person():
    return ("01", "37", "1", "reported", "0", "reported", "42", "p.csv", 2)


def _record(people=()):
    return ("2024HU0000001", "1", str(len(people)), "12", "h.csv", 1, people)


def _raw(value):
    return "".join(chunks(value)).encode("utf-8")


@pytest.mark.parametrize(
    "value, expected",
    [
        (((), ()), b"[[],[]]"),
        (
            ((), (_record(),)),
            b'[[],[["2024HU0000001","1","0","12","h.csv",1,[]]]]',
        ),
        (
            ((_record((_person(),)),), (_record(),)),
            b'[[["2024HU0000001","1","1","12","h.csv",1,'
            b'[["01","37","1","reported","0","reported","42","p.csv",2]]]],'
            b'[["2024HU0000001","1","0","12","h.csv",1,[]]]]',
        ),
        (
            ((("é", '"\\\n', "\x00", "😀", "z", -1, ()),), ()),
            '[[["é","\\"\\\\\\n","\\u0000","😀","z",-1,[]]],[]]'.encode(),
        ),
    ],
)
def test_frozen_canonical_bytes(value, expected):
    # The literal expected byte sequence is independent of either encoder.
    assert _raw(value) == expected
    assert "".join(owner._chunks(value)).encode("utf-8") == expected
    assert digest(value) == hashlib.sha256(expected).hexdigest()
    assert digest(value) == owner._digest(value)


def test_order_duplicates_and_per_record_boundaries():
    a, b = _record((_person(),)), ("2024GQ0000002", "3", "0", "0", "g", 7, ())
    values = [((a, b, a), ()), ((b, a, a), ()), ((a, a), (b,))]
    actual = [digest(value) for value in values]
    assert len(set(actual)) == 3
    for value in values:
        assert _raw(value) == "".join(owner._chunks(value)).encode("utf-8")


@pytest.mark.parametrize("characters", [65531, 65532, 65533])
def test_fast_eligibility_character_boundary_does_not_change_acceptance(characters):
    # Four remaining one-character strings put 65532 exactly at 65536 total;
    # the ordinal is an int. Acceptance must not depend on eligibility.
    value = ((("x" * characters, "1", "0", "0", "h", 1, ()),), ())
    assert _raw(value) == "".join(owner._chunks(value)).encode("utf-8")
    assert digest(value) == owner._digest(value)


@pytest.mark.parametrize("count", [20, 21])
def test_person_count_fast_boundary_falls_back_without_new_refusal(count):
    value = ((_record((_person(),) * count),), ())
    assert _raw(value) == "".join(owner._chunks(value)).encode("utf-8")


def test_generic_accepted_shapes_and_scalar_types_keep_old_bytes():
    values = [
        ([list(_record())], []),
        (({"z": 1, "a": [None, True, False, 1.0, -0.0, "é"]},), ()),
        (((1 << 90, "1", "0", "0", "h", 1, ()),), ()),
        ((("x" * owner._MAX_SCALAR_BYTES, "1", "0", "0", "h", 1, ()),), ()),
    ]
    for value in values:
        assert _raw(value) == "".join(owner._chunks(value)).encode("utf-8")
        assert digest(value) == owner._digest(value)


def _error(function, value):
    try:
        function(value)
    except Exception as exc:
        return type(exc), str(exc)
    pytest.fail("Malformed corpus member was accepted")


def test_malformed_refusal_and_global_scalar_precedence():
    deep = 0
    for _ in range(65):
        deep = (deep,)
    cycle = []
    cycle.append(cycle)
    oversized = "x" * (owner._MAX_SCALAR_BYTES + 1)
    cases = [
        (((float("nan"),),), ()),
        (((float("inf"),),), ()),
        (((float("-inf"),),), ()),
        (((b"not-json",),), ()),
        ((("\ud800",),), ()),
        ((("\udfff",),), ()),
        ((({1, 2},),), ()),
        ((({1: 1, "a": 2},),), ()),
        ((deep,), ()),
        ((cycle,), ()),
        ((oversized,), ()),
        ((float("nan"), oversized), ()),
        ((b"bad", deep), ()),
    ]
    for value in cases:
        assert _error(digest, value) == _error(owner._digest, value)
    assert _error(digest, ((float("nan"), oversized), ()))[1] == "SCALAR_LIMIT"
    assert _error(digest, ((b"bad", deep), ()))[1] == "VALUE_DEPTH"


def test_subclasses_use_unchanged_dispatch():
    class Text(str):
        pass

    class Integer(int):
        pass

    class Sequence(tuple):
        pass

    for value in [((Text("é"), Integer(2)), ()), Sequence(((), ()))]:
        assert _raw(value) == "".join(owner._chunks(value)).encode("utf-8")


def test_digest_is_independent_of_an_explicit_tiny_transport_bound():
    # This small value does not test the production 64 MiB boundary.
    # It checks digest parity alongside an independently tiny transport limit.
    value = ((_record((_person(),)),) * 1024, ())
    assert digest(value) == owner._digest(value)
    with pytest.raises(owner.SurveyPopulationPreparationError, match="PAYLOAD_LIMIT"):
        owner._encode(value, 30)
