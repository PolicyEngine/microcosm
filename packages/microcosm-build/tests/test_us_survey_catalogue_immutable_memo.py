"""Immutable value reuse plus actual issuer final-borrow mutation controls.

Supplied tuples exercise private value helpers only; they are never treated
as authenticated source catalogues. The last controls use real source
issuance over the maintained invented preparation fixture.
"""

import hashlib
import json
import sys

import pytest
from test_us_survey_population_preparation import fixture

from microcosm.build.us_runtime import survey_population_preparation as owner


def _records():
    person = ("1", 37, "1", "observed", "1", "observed", "100", "p.csv", 1)
    household = ("2024HU0000001", 1, 1, 100, "h.csv", 1, (person,))
    vacancy = ("2024HU0000002", 1, 0, 0, "h.csv", 2, ())
    return ((household,), (vacancy,))


def _oracle(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calls(action):
    calls = []

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is owner._catalogue_digest.__code__:
            calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        result = action()
    finally:
        sys.setprofile(previous)
    return result, len(calls)


def test_same_immutable_roots_reuse_exact_digest_without_another_encode():
    records = _records()
    expected = _oracle(records)
    memo, creation_calls = _calls(lambda: owner._catalogue_memo(records, expected))
    assert creation_calls == 1
    assert type(memo) is tuple
    assert memo[0] is records[0] and memo[1] is records[1]
    assert memo[2] == expected
    result, calls = _calls(
        lambda: [
            owner._memoized_catalogue_digest(
                (records[0], records[1]), memo, expected_digest=expected
            )
            for _ in range(3)
        ]
    )
    assert result == [expected] * 3 and calls == 0


def test_equal_replacement_roots_miss_without_changing_digest_or_retained_memo():
    records = _records()
    expected = _oracle(records)
    memo = owner._catalogue_memo(records, expected)
    replacement = (tuple(list(records[0])), tuple(list(records[1])))
    assert replacement == records
    assert replacement[0] is not records[0] and replacement[1] is not records[1]
    result, calls = _calls(
        lambda: owner._memoized_catalogue_digest(
            replacement, memo, expected_digest=expected
        )
    )
    assert result == expected and calls == 1
    assert memo[0] is records[0] and memo[1] is records[1]


@pytest.mark.parametrize("group", [0, 1])
def test_changed_immutable_record_root_misses_and_changes_digest(group):
    records = _records()
    expected = _oracle(records)
    memo = owner._catalogue_memo(records, expected)
    row = records[group][0]
    changed = list(records)
    changed[group] = ((row[0] + "different", *row[1:]),)
    changed = tuple(changed)
    result, calls = _calls(
        lambda: owner._memoized_catalogue_digest(
            changed, memo, expected_digest=expected
        )
    )
    assert calls == 1 and result == _oracle(changed) and result != expected


@pytest.mark.parametrize(
    "kind",
    [
        "group_list",
        "record_list",
        "people_list",
        "person_list",
        "tuple_subclass",
        "str_subclass",
        "int_subclass",
        "non_ascii",
        "long_ascii",
        "large_integer",
    ],
)
def test_ineligible_shapes_and_subclasses_keep_original_digest_path(kind):
    class TupleSubclass(tuple):
        pass

    class StrSubclass(str):
        pass

    class IntSubclass(int):
        pass

    records = _records()
    row = records[0][0]
    if kind == "group_list":
        records = (list(records[0]), records[1])
    elif kind == "record_list":
        records = ((list(row),), records[1])
    elif kind == "people_list":
        records = (((*row[:6], list(row[6])),), records[1])
    elif kind == "person_list":
        records = (((*row[:6], (list(row[6][0]),)),), records[1])
    elif kind == "tuple_subclass":
        records = (TupleSubclass(records[0]), records[1])
    elif kind == "str_subclass":
        records = (((StrSubclass(row[0]), *row[1:]),), records[1])
    elif kind == "int_subclass":
        records = (((row[0], IntSubclass(row[1]), *row[2:]),), records[1])
    elif kind == "non_ascii":
        records = ((("é", *row[1:]),), records[1])
    elif kind == "long_ascii":
        records = ((("a" * 65_537, *row[1:]),), records[1])
    else:
        records = (((row[0], 2**63, *row[2:]),), records[1])
    expected = _oracle(records)
    memo = owner._catalogue_memo(records, expected)
    assert memo is None
    result, calls = _calls(
        lambda: owner._memoized_catalogue_digest(
            records, memo, expected_digest=expected
        )
    )
    assert result == expected and calls == 1


def test_nested_mutable_value_is_rehashed_after_in_place_change():
    records = _records()
    row = records[0][0]
    person = list(row[6][0])
    records = (((*row[:6], (person,)),), records[1])
    expected = _oracle(records)
    memo = owner._catalogue_memo(records, expected)
    assert memo is None
    person[1] += 1
    result, calls = _calls(
        lambda: owner._memoized_catalogue_digest(
            records, memo, expected_digest=expected
        )
    )
    assert calls == 1 and result == _oracle(records) and result != expected


def test_memo_creation_binds_exact_original_digest():
    records = _records()
    assert owner._catalogue_memo(records, "0" * 64) is None


def test_swapped_complete_memo_entry_misses():
    records = _records()
    expected = _oracle(records)
    row = records[0][0]
    other = (((row[0] + "other", *row[1:]),), records[1])
    memo = owner._catalogue_memo(other, _oracle(other))
    result, calls = _calls(
        lambda: owner._memoized_catalogue_digest(
            records, memo, expected_digest=expected
        )
    )
    assert result == expected and calls == 1


def test_cached_digest_drift_cannot_replace_original_nested_seal():
    records = _records()
    expected = _oracle(records)
    memo = owner._catalogue_memo(records, expected)
    wrong = (*memo[:2], "0" * 64)
    result, calls = _calls(
        lambda: owner._memoized_catalogue_digest(
            records, wrong, expected_digest=expected
        )
    )
    assert result == expected and calls == 1


@pytest.fixture(scope="module")
def actual_preparation(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        arguments = fixture(tmp_path_factory.mktemp("catalogue-memo"), monkeypatch)
        result = owner.prepare_authenticated_survey_population(**arguments)
        state = owner._ISSUED[id(result)][2]
        assert type(state.acs_catalogue_memo) is tuple
        yield result, state
        result.validate()


@pytest.mark.parametrize("field", ["records", "vacancies"])
def test_actual_catalogue_replacement_at_final_producer_return_refuses(
    actual_preparation, field
):
    result, state = actual_preparation
    catalogue = owner.acs_catalogue._lookup(state.catalogues[0])
    original = getattr(catalogue, field)
    assert original
    row = original[0]
    replacement = ((row[0] + "different", *row[1:]), *original[1:])
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is owner._producer.__code__
            and frame.f_back.f_code is owner._validate.__code__
            and not fired
        ):
            fired.append(True)
            object.__setattr__(catalogue, field, replacement)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(
            owner.SurveyPopulationPreparationError, match="NESTED_EVIDENCE_CHANGED"
        ):
            result.checked_view()
    finally:
        sys.setprofile(previous)
        object.__setattr__(catalogue, field, original)
    assert fired == [True]
    assert getattr(catalogue, field) is original
