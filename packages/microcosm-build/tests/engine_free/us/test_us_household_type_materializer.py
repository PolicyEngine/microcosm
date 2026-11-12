"""Household roster classification, independent pandas oracle and partitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.household_type_targets import (
    HOUSEHOLD_TYPE_MATERIALIZER,
    household_type_household_values,
)
from test_support.microcosm_build.us_batched_target_materialization import (
    _install_fake_engine,
    _load_builder_module,
    _nested_frame,
    _spec,
)


@dataclass
class _Frame:
    tables: dict[str, pd.DataFrame]

    def table(self, entity: str) -> pd.DataFrame:
        return self.tables[entity]


def _frame(
    rosters: list[list[tuple[int, int, float, bool]]],
    *,
    kinds: list[float] | None = None,
) -> _Frame:
    households = pd.DataFrame(
        {
            "household_id": [10 + 11 * i for i in range(len(rosters))],
            "state_fips": [6 if i % 2 else 48 for i in range(len(rosters))],
        }
    )
    if kinds is not None:
        households["TYPEHUGQ"] = kinds
    people = []
    for i, roster in enumerate(rosters):
        for relationship, marital, age, female in roster:
            people.append(
                {
                    "person_household_id": int(households.iloc[i].household_id),
                    "A_EXPRRP": relationship,
                    "A_MARITL": marital,
                    "age": age,
                    "is_female": female,
                }
            )
    return _Frame({"person": pd.DataFrame(people), "household": households})


def _values(frame: _Frame, household_type: str, own_children: str = "all", **extra):
    return household_type_household_values(
        frame,
        {"household_type": household_type, "own_children": own_children, **extra},
    )


def test_classifies_each_family_type_and_living_alone() -> None:
    frame = _frame(
        [
            [(1, 1, 40, False), (4, 1, 40, True), (5, 7, 8, False)],
            [(1, 5, 40, False), (5, 7, 5, True)],
            [(1, 5, 40, True), (5, 7, 17, True)],
            [(2, 7, 70, True)],
            [(2, 7, 40, False), (12, 7, 35, True)],
            [(1, 1, 50, True), (3, 1, 50, False)],
        ]
    )
    assert HOUSEHOLD_TYPE_MATERIALIZER == "household_type"
    np.testing.assert_array_equal(_values(frame, "all"), [1, 1, 1, 1, 1, 1])
    for kind, expected in (
        ("married_couple_family", [1, 0, 0, 0, 0, 1]),
        ("male_householder_no_spouse", [0, 1, 0, 0, 0, 0]),
        ("female_householder_no_spouse", [0, 0, 1, 0, 0, 0]),
        ("householder_living_alone", [0, 0, 0, 1, 0, 0]),
        ("nonfamily", [0, 0, 0, 1, 1, 0]),
    ):
        np.testing.assert_array_equal(_values(frame, kind), expected)
    np.testing.assert_array_equal(
        _values(frame, "married_couple_family", "with_own_children_under_18"),
        [1, 0, 0, 0, 0, 0],
    )


def test_own_children_excludes_adults_ever_married_and_other_relations() -> None:
    frame = _frame(
        [
            [(1, 5, 40, False), (5, 7, 18, True)],
            [(1, 5, 40, False), (5, 5, 17, True)],
            [(1, 5, 40, False), (9, 7, 5, True)],
            [(1, 5, 40, False), (11, 7, 5, True)],
            [(1, 5, 40, False), (5, 7, 17.99, True)],
        ]
    )
    np.testing.assert_array_equal(
        _values(frame, "male_householder_no_spouse", "with_own_children_under_18"),
        [0, 0, 0, 0, 1],
    )


def test_group_quarters_stay_outside_every_household_indicator() -> None:
    frame = _frame(
        [[(2, 7, 30, False)], [(14, 7, 30, True)], [(14, 7, 70, False)]],
        kinds=[1, 2, 3],
    )
    np.testing.assert_array_equal(_values(frame, "all"), [1, 0, 0])
    np.testing.assert_array_equal(_values(frame, "householder_living_alone"), [1, 0, 0])
    np.testing.assert_array_equal(_values(frame, "nonfamily"), [1, 0, 0])
    # Mixed support leaves TYPEHUGQ absent on ASEC rows.
    frame.tables["household"].loc[0, "TYPEHUGQ"] = np.nan
    np.testing.assert_array_equal(_values(frame, "all"), [1, 0, 0])


@pytest.mark.parametrize("heads", [0, 2])
def test_requires_exactly_one_householder(heads: int) -> None:
    frame = _frame([[(2 if i < heads else 12, 7, 30, False) for i in range(2)]])
    with pytest.raises(ValueError, match="exactly one householder"):
        _values(frame, "all")


@pytest.mark.parametrize(
    ("entity", "column"),
    [
        ("person", "A_EXPRRP"),
        ("person", "A_MARITL"),
        ("person", "age"),
        ("person", "is_female"),
        ("person", "person_household_id"),
        ("household", "household_id"),
    ],
)
def test_missing_required_column_refuses(entity: str, column: str) -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables[entity] = frame.tables[entity].drop(columns=column)
    with pytest.raises(ValueError, match="require.*column"):
        _values(frame, "all")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("A_EXPRRP", np.nan, "A_EXPRRP codes"),
        ("A_EXPRRP", 15, "A_EXPRRP codes"),
        ("A_MARITL", 8, "A_MARITL codes"),
        ("age", np.nan, "finite nonnegative age"),
        ("age", -1, "finite nonnegative age"),
        ("is_female", None, "observed boolean is_female"),
        ("person_household_id", 99, "household is missing"),
    ],
)
def test_invalid_person_values_refuse(column, value, message) -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables["person"][column] = value
    with pytest.raises(ValueError, match=message):
        _values(frame, "all")


@pytest.mark.parametrize("kind", [0, 4, "unknown"])
def test_invalid_group_quarters_kind_refuses(kind) -> None:
    frame = _frame([[(2, 7, 30, False)]], kinds=[kind])
    with pytest.raises(ValueError, match="TYPEHUGQ"):
        _values(frame, "all")


def test_acs_provenance_requires_group_quarters_classification() -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables["household"]["household_spine"] = "acs_2024_1yr"
    with pytest.raises(ValueError, match="observed TYPEHUGQ"):
        _values(frame, "all")


def test_group_quarters_cannot_contain_multiple_people() -> None:
    frame = _frame([[(14, 7, 30, False), (14, 7, 40, True)]], kinds=[2])
    with pytest.raises(ValueError, match="exactly one person"):
        _values(frame, "all")


def test_unknown_metadata_and_missing_state_refuse() -> None:
    frame = _frame([[(2, 7, 30, False)]])
    with pytest.raises(ValueError, match="Unknown ACS household type"):
        _values(frame, "new_type")
    with pytest.raises(ValueError, match="Unknown ACS own-children filter"):
        _values(frame, "all", "new_children")
    frame.tables["household"] = frame.tables["household"].drop(columns="state_fips")
    with pytest.raises(ValueError, match="state_fips"):
        _values(frame, "all", state_fips="06")


@pytest.mark.parametrize("field", ["household_type", "own_children", "statistic"])
def test_release_filter_guard_checks_household_filter_attestation(field: str) -> None:
    builder = _load_builder_module()
    metadata = {
        "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
        "household_type": "married_couple_family",
        "own_children": "with_own_children_under_18",
        "statistic": "estimate",
    }
    metadata[f"ledger_filter_{field}"] = metadata[field]
    spec = SimpleNamespace(name="households", metadata=metadata)
    builder._assert_supported_ledger_filter_metadata((spec,))
    metadata[f"ledger_filter_{field}"] = "different"
    with pytest.raises(RuntimeError, match="disagrees with compiled"):
        builder._assert_supported_ledger_filter_metadata((spec,))
    metadata[f"ledger_filter_{field}"] = metadata[field]
    metadata["materializer"] = "policyengine_variable"
    with pytest.raises(RuntimeError, match=f"ledger_filter_{field}"):
        builder._assert_supported_ledger_filter_metadata((spec,))


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_release_materializes_required_roster_indicators(
    monkeypatch, batch_size
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    person = frame.table("person")
    person["A_EXPRRP"] = [1, 4, 5, 2, 2, 12, 1, 5, 7, 12, 14]
    person["A_MARITL"] = 7
    person["age"] = [40, 40, 5, 40, 40, 40, 40, 5, 40, 40, 40]
    person["is_female"] = False
    frame.table("household")["TYPEHUGQ"] = [1, 1, 1, 1, 2]
    targets = (
        _spec(
            "households",
            "census_households",
            {
                "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
                "household_type": "all",
            },
        ),
        _spec(
            "married_with_children",
            "census_households",
            {
                "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
                "household_type": "married_couple_family",
                "own_children": "with_own_children_under_18",
            },
        ),
    )
    _install_fake_engine(builder, monkeypatch, reform_specs=())
    target_frame, _, compilation = builder._materialize_target_frame(
        frame, targets, maximum_microsim_batch_size=batch_size
    )
    assert compilation["dropped_target_names"] == []
    np.testing.assert_array_equal(
        target_frame.table("household")["households"], [1, 1, 1, 1, 0]
    )
    np.testing.assert_array_equal(
        target_frame.table("household")["married_with_children"], [1, 0, 0, 0, 0]
    )


@st.composite
def _frames(draw):
    count = draw(st.integers(min_value=1, max_value=8))
    rosters = []
    kinds = []
    for _ in range(count):
        kind = draw(st.sampled_from(["married", "other", "nonfamily", "gq"]))
        female = draw(st.booleans())
        if kind == "gq":
            rosters.append([(14, 7, draw(st.integers(0, 105)), female)])
            kinds.append(draw(st.sampled_from([2, 3])))
            continue
        members = [(2 if kind == "nonfamily" else 1, 7, 40, female)]
        if kind == "married":
            members.append((3 if female else 4, 1, 40, not female))
        extras = draw(st.integers(min_value=0 if kind != "other" else 1, max_value=4))
        for _ in range(extras):
            relation = draw(
                st.sampled_from(
                    [12, 13, 14] if kind == "nonfamily" else [5, 7, 9, 11, 12]
                )
            )
            members.append(
                (
                    relation,
                    draw(st.integers(1, 7)),
                    draw(st.integers(0, 105)),
                    draw(st.booleans()),
                )
            )
        rosters.append(members)
        kinds.append(draw(st.sampled_from([1, np.nan])))
    frame = _frame(rosters, kinds=kinds)
    # Neither implementation can depend on sorted persons or household ids.
    frame.tables["household"] = frame.tables["household"].iloc[::-1]
    frame.tables["person"] = frame.tables["person"].iloc[::-1]
    return frame


@given(_frames())
@settings(max_examples=70, deadline=None)
def test_indicator_partitions_hold_for_every_valid_roster(frame: _Frame) -> None:
    household_types = (
        "all",
        "family",
        "married_couple_family",
        "male_householder_no_spouse",
        "female_householder_no_spouse",
        "nonfamily",
        "householder_living_alone",
        "householder_not_living_alone",
    )
    values = {kind: _values(frame, kind) for kind in household_types}
    for kind, national in values.items():
        assert np.isin(national, [0.0, 1.0]).all()
        np.testing.assert_array_equal(
            _values(frame, kind, state_fips="06")
            + _values(frame, kind, state_fips="48"),
            national,
        )
    np.testing.assert_array_equal(
        values["married_couple_family"]
        + values["male_householder_no_spouse"]
        + values["female_householder_no_spouse"]
        + values["nonfamily"],
        values["all"],
    )
    for kind in (
        "married_couple_family",
        "male_householder_no_spouse",
        "female_householder_no_spouse",
    ):
        with_children = _values(frame, kind, "with_own_children_under_18")
        without_children = _values(frame, kind, "no_own_children_under_18")
        assert np.isin(with_children, [0.0, 1.0]).all()
        assert np.isin(without_children, [0.0, 1.0]).all()
        np.testing.assert_array_equal(with_children + without_children, values[kind])
        for own_children in ("with_own_children_under_18", "no_own_children_under_18"):
            np.testing.assert_array_equal(
                _values(frame, kind, own_children, state_fips="06")
                + _values(frame, kind, own_children, state_fips="48"),
                _values(frame, kind, own_children),
            )


@given(_frames(), st.sampled_from([0, 2]))
@settings(max_examples=50, deadline=None)
def test_any_roster_with_zero_or_two_householders_refuses(
    frame: _Frame, heads: int
) -> None:
    person = frame.table("person").copy()
    references = person.loc[person.A_EXPRRP.isin([1, 2])]
    assume(not references.empty)
    household_id = references.iloc[0].person_household_id
    member_indices = person.index[person.person_household_id == household_id]
    person.loc[member_indices, "A_EXPRRP"] = 12
    if heads == 2:
        person.loc[member_indices[0], "A_EXPRRP"] = 2
        duplicate = person.loc[[member_indices[0]]].copy()
        person = pd.concat([person, duplicate], ignore_index=True)
    frame.tables["person"] = person
    with pytest.raises(ValueError, match="exactly one householder"):
        _values(frame, "all")


def _pandas_indicators(frame: _Frame) -> np.ndarray:
    """Independent row/group oracle for the eight bound national ACS cells."""
    rows = []
    people = frame.table("person")
    for _, household in frame.table("household").iterrows():
        if household.TYPEHUGQ in (2, 3):
            rows.append([0] * 8)
            continue
        members = people.loc[people.person_household_id == household.household_id]
        reference = members.loc[members.A_EXPRRP.isin([1, 2])].iloc[0]
        family = reference.A_EXPRRP == 1
        married = family and members.A_EXPRRP.isin([3, 4]).any()
        other_male = family and not married and not reference.is_female
        other_female = family and not married and reference.is_female
        child = not members.loc[
            members.A_EXPRRP.eq(5) & members.age.lt(18) & members.A_MARITL.eq(7)
        ].empty
        rows.append(
            [
                1,
                married,
                other_male,
                other_female,
                len(members) == 1,
                married and child,
                other_male and child,
                other_female and child,
            ]
        )
    return np.asarray(rows, dtype=float)


@given(_frames())
@settings(max_examples=70, deadline=None)
def test_eight_indicators_match_independent_pandas_oracle(frame: _Frame) -> None:
    actual = np.column_stack(
        [
            _values(frame, "all"),
            _values(frame, "married_couple_family"),
            _values(frame, "male_householder_no_spouse"),
            _values(frame, "female_householder_no_spouse"),
            _values(frame, "householder_living_alone"),
            _values(frame, "married_couple_family", "with_own_children_under_18"),
            _values(frame, "male_householder_no_spouse", "with_own_children_under_18"),
            _values(
                frame, "female_householder_no_spouse", "with_own_children_under_18"
            ),
        ]
    )
    np.testing.assert_array_equal(actual, _pandas_indicators(frame))
