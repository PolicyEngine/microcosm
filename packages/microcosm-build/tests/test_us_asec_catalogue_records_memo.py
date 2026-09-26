"""Immutable-leaf reuse of the ASEC records identity plus actual issuer controls.

Supplied records exercise private value helpers only; they are never treated
as an authenticated source catalogue. The last controls use real issuance over
the maintained invented preparation fixture.
"""

import sys

import pytest
from test_us_survey_population_preparation import fixture

from microcosm.build.us_runtime import asec_population_catalogue as owner
from microcosm.build.us_runtime import survey_population_domains as domains
from microcosm.build.us_runtime import survey_population_preparation as preparation


def _key(native_id="1"):
    return domains.HouseholdKey(domains.Source.ASEC, 2024, 2025, native_id)


def _records():
    key = _key()
    person = domains.AsecPerson("2024000000000000000001", "1", "37", "1", "1", key)
    household = domains.AsecHousehold(key, "1", "1", "1", "1", "1234567", (person,))
    ledger = owner.UnrepresentedAsecHousehold(_key("2"), "2", "9", "1", "0", "7654321")
    return (household,), (ledger,)


def _calls(action):
    calls = []

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is owner._records_identity.__code__:
            calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        result = action()
    finally:
        sys.setprofile(previous)
    return result, len(calls)


def _issued(households, ledger):
    (identity, memo), calls = _calls(
        lambda: owner._records_identity(households, ledger, memo=True)
    )
    assert calls == 1
    return identity, memo


def _memoized(households, ledger, memo, identity):
    return _calls(
        lambda: owner._memoized_records_identity(
            households, ledger, memo, expected_identity=identity
        )
    )


def test_same_issued_leaves_reuse_exact_identity_without_another_encode():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    assert identity == owner._records_identity(households, ledger)
    assert type(memo) is owner._RecordsMemo
    assert memo.households is households and memo.ledger is ledger
    assert memo.identity == identity
    assert memo.household_leaves == (owner._household_leaves(households[0]),)
    assert memo.ledger_leaves == (owner._ledger_leaves(ledger[0]),)
    result, calls = _calls(
        lambda: [
            owner._memoized_records_identity(
                households, ledger, memo, expected_identity=identity
            )
            for _ in range(3)
        ]
    )
    assert result == [identity] * 3 and calls == 0


def test_equal_replacement_roots_miss_without_changing_identity_or_memo():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    replacement = (tuple(list(households)), tuple(list(ledger)))
    assert replacement == (households, ledger)
    assert replacement[0] is not households and replacement[1] is not ledger
    result, calls = _memoized(*replacement, memo, identity)
    assert result == identity and calls == 1
    assert memo.households is households and memo.ledger is ledger


@pytest.mark.parametrize(
    "target",
    ["household", "person", "household_key", "person_key", "ledger", "reason"],
)
def test_in_place_leaf_change_misses_and_changes_identity(target):
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    household, person = households[0], households[0].persons[0]
    if target == "household":
        object.__setattr__(household, "hsup_wgt", "0000000")
    elif target == "person":
        object.__setattr__(person, "age", "38")
    elif target == "household_key":
        object.__setattr__(household.key, "native_id", "9")
    elif target == "person_key":
        object.__setattr__(person, "household_key", _key("9"))
    elif target == "ledger":
        object.__setattr__(ledger[0], "h_numper", "1")
    else:
        object.__setattr__(ledger[0], "reason", "other")
    result, calls = _memoized(households, ledger, memo, identity)
    assert calls == 1 and result != identity
    assert result == owner._records_identity(households, ledger)


@pytest.mark.parametrize("target", ["persons", "key"])
def test_replaced_container_misses_even_when_equal(target):
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    household = households[0]
    original = getattr(household, target)
    replacement = (
        tuple(list(original)) if target == "persons" else _key(original.native_id)
    )
    assert replacement == original and replacement is not original
    object.__setattr__(household, target, replacement)
    result, calls = _memoized(households, ledger, memo, identity)
    assert result == identity and calls == 1


def test_persons_replaced_by_a_list_takes_the_refusing_path():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    object.__setattr__(households[0], "persons", list(households[0].persons))
    with pytest.raises(owner.AsecSourceCatalogueError, match="^RECORD_TYPE$"):
        owner._memoized_records_identity(
            households, ledger, memo, expected_identity=identity
        )


def test_deleted_leaf_misses_instead_of_raising_inside_the_memo():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    object.__delattr__(households[0].persons[0], "age")
    with pytest.raises(AttributeError):
        owner._memoized_records_identity(
            households, ledger, memo, expected_identity=identity
        )


@pytest.mark.parametrize(
    "kind",
    [
        "str_subclass_literal",
        "str_subclass_person",
        "str_subclass_reason",
        "int_subclass_year",
        "str_subclass_native_id",
    ],
)
def test_subclass_leaves_yield_no_memo_and_keep_the_full_identity_path(kind):
    class StrSubclass(str):
        pass

    class IntSubclass(int):
        pass

    households, ledger = _records()
    household, person = households[0], households[0].persons[0]
    if kind == "str_subclass_literal":
        object.__setattr__(household, "h_hhtype", StrSubclass("1"))
    elif kind == "str_subclass_person":
        object.__setattr__(person, "prpertyp", StrSubclass("1"))
    elif kind == "str_subclass_reason":
        object.__setattr__(ledger[0], "reason", StrSubclass("x"))
    elif kind == "int_subclass_year":
        object.__setattr__(household.key, "source_year", IntSubclass(2024))
    else:
        object.__setattr__(ledger[0].key, "native_id", StrSubclass("2"))
    identity, memo = _issued(households, ledger)
    assert memo is None
    assert identity == owner._records_identity(households, ledger)
    result, calls = _memoized(households, ledger, memo, identity)
    assert result == identity and calls == 1


def test_memo_requires_agreement_with_the_owner_identity():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    wrong = (*identity[:3], identity[3] + 1)
    result, calls = _memoized(households, ledger, memo, wrong)
    assert result == identity and calls == 1
    drifted = memo._replace(identity=wrong)
    result, calls = _memoized(households, ledger, drifted, identity)
    assert result == identity and calls == 1
    plain = tuple(memo)
    assert plain == memo
    result, calls = _memoized(households, ledger, plain, identity)
    assert result == identity and calls == 1


def test_swapped_memo_entry_misses():
    households, ledger = _records()
    identity, _ = _issued(households, ledger)
    other_households, other_ledger = _records()
    object.__setattr__(other_households[0], "hsup_wgt", "0000001")
    other_identity, other_memo = _issued(other_households, other_ledger)
    assert other_identity != identity
    result, calls = _memoized(households, ledger, other_memo, identity)
    assert result == identity and calls == 1
    result, calls = _memoized(households, ledger, other_memo, other_identity)
    assert result == identity and calls == 1


def test_crafted_memo_over_lists_still_refuses():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    rows = list(households)
    crafted = owner._RecordsMemo(
        rows, ledger, memo.household_leaves, memo.ledger_leaves, identity
    )
    with pytest.raises(owner.AsecSourceCatalogueError, match="^RECORD_TYPE$"):
        owner._memoized_records_identity(
            rows, ledger, crafted, expected_identity=identity
        )


def test_wrong_source_member_or_foreign_types_yield_no_memo():
    households, ledger = _records()
    foreign = domains.HouseholdKey(domains.Source.ACS, 2024, 2025, "1")
    object.__setattr__(ledger[0], "key", foreign)
    with pytest.raises(owner.AsecSourceCatalogueError, match="^RECORD_SOURCE$"):
        owner._records_identity(households, ledger, memo=True)
    assert owner._eligible_leaves(ledger[0], "unrepresented") is None
    assert owner._eligible_leaves(households[0], "unrepresented") is None
    assert owner._eligible_leaves(ledger[0], "household") is None


def test_source_member_value_tampering_misses_and_the_full_path_reflects_it():
    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    member = domains.Source.ASEC
    original = member._value_
    object.__setattr__(member, "_value_", "tampered")
    try:
        assert member.value == "tampered"
        result, calls = _memoized(households, ledger, memo, identity)
        assert calls == 1 and result != identity
        assert result == owner._records_identity(households, ledger)
    finally:
        object.__setattr__(member, "_value_", original)
    assert member.value == original
    result, calls = _memoized(households, ledger, memo, identity)
    assert result == identity and calls == 0


@pytest.mark.parametrize("target", ["household", "person", "key", "ledger"])
def test_class_reassignment_misses_and_the_full_path_refuses(target):
    class SwappedHousehold(domains.AsecHousehold):
        __slots__ = ()

    class SwappedPerson(domains.AsecPerson):
        __slots__ = ()

    class SwappedKey(domains.HouseholdKey):
        __slots__ = ()

    class SwappedLedger(owner.UnrepresentedAsecHousehold):
        __slots__ = ()

    households, ledger = _records()
    identity, memo = _issued(households, ledger)
    victim, swapped = {
        "household": (households[0], SwappedHousehold),
        "person": (households[0].persons[0], SwappedPerson),
        "key": (households[0].key, SwappedKey),
        "ledger": (ledger[0], SwappedLedger),
    }[target]
    original = type(victim)
    object.__setattr__(victim, "__class__", swapped)
    try:
        assert type(victim) is swapped
        with pytest.raises(owner.AsecSourceCatalogueError, match="^RECORD_TYPE$"):
            owner._memoized_records_identity(
                households, ledger, memo, expected_identity=identity
            )
    finally:
        object.__setattr__(victim, "__class__", original)
    result, calls = _memoized(households, ledger, memo, identity)
    assert result == identity and calls == 0


@pytest.fixture(scope="module")
def actual_preparation(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        arguments = fixture(tmp_path_factory.mktemp("asec-records-memo"), monkeypatch)
        result = preparation.prepare_authenticated_survey_population(**arguments)
        state = preparation._ISSUED[id(result)][2]
        catalogue = state.catalogues[1]
        catalogue_state = owner._ISSUED[id(catalogue)][2]
        assert type(catalogue_state.records_memo) is owner._RecordsMemo
        assert catalogue_state.records_memo.identity == catalogue_state.records_identity
        yield result, catalogue, catalogue_state
        result.validate()
        catalogue.validate()


def test_actual_borrows_reuse_the_issued_identity(actual_preparation):
    result, catalogue, _ = actual_preparation
    _, calls = _calls(catalogue.validate)
    assert calls == 0
    _, calls = _calls(result.checked_view)
    assert calls == 0


def test_actual_household_leaf_change_refuses_every_borrow(actual_preparation):
    result, catalogue, catalogue_state = actual_preparation
    row = catalogue_state.households[0]
    original = row.hsup_wgt
    object.__setattr__(row, "hsup_wgt", original + "0")
    try:
        with pytest.raises(owner.AsecSourceCatalogueError, match="RECORDS_CHANGED"):
            catalogue.validate()
        with pytest.raises(
            preparation.SurveyPopulationPreparationError,
            match="NESTED_EVIDENCE_CHANGED",
        ):
            result.checked_view()
    finally:
        object.__setattr__(row, "hsup_wgt", original)
    _, calls = _calls(catalogue.validate)
    assert calls == 0


def test_actual_person_key_change_refuses_every_borrow(actual_preparation):
    result, catalogue, catalogue_state = actual_preparation
    person = catalogue_state.households[0].persons[0]
    original = person.household_key
    object.__setattr__(person, "household_key", _key(original.native_id + "0"))
    try:
        with pytest.raises(owner.AsecSourceCatalogueError, match="RECORDS_CHANGED"):
            catalogue.validate()
        with pytest.raises(
            preparation.SurveyPopulationPreparationError,
            match="NESTED_EVIDENCE_CHANGED",
        ):
            result.checked_view()
    finally:
        object.__setattr__(person, "household_key", original)
    result.validate()
