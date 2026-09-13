"""Invented values through household-role classification and exact clone binding.

Private fixture projections assert no source authority. Actual owners have a
separate test file and separately bounded runtime proposal.
"""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_household_roles as roles
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

DEFAULT = (
    (10, 2, "asec", 1, 1, 0),
    (20, 3, "acs", 21, 2, 0),
    (30, 5, "asec", 6, 0, 1),
)


def _table(specs=DEFAULT):
    rows = []
    for _original, native, survey, code, state, reason in specs:
        observation = (
            roles.asec_household_role_observation(state, reason)
            if survey == "asec"
            else roles.acs_household_role_observation(state, code)
        )
        rows.append(
            {
                roles.SURVEY_COLUMN: survey,
                roles.NATIVE_ID_COLUMN: native,
                roles.CODE_COLUMN: code,
                roles.CODE_KNOWN_COLUMN: code
                in (
                    roles.demographic.A_EXPRRP.named_codes
                    if survey == "asec"
                    else roles.ACS_PRINTED_RELATIONSHIP_DOMAIN
                ),
                roles.ALLOCATION_KNOWN_COLUMN: False,
                **observation,
            }
        )
    return roles._observation_table(rows, [row[0] for row in specs])


def _frame(people):
    people = people.copy(deep=True)
    tables = {"person": people}
    for entity in US_SCHEMA.group_entities:
        people[US_SCHEMA.membership_column(entity)] = people.person_id.to_numpy()
        tables[entity] = pd.DataFrame(
            {US_SCHEMA.entity_id_column(entity): np.sort(people.person_id.to_numpy())}
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(people)), WeightKind.DESIGN)},
    )


def _qualified(table=None):
    table = _table() if table is None else table
    frame = _frame(pd.DataFrame({"person_id": table.index.to_numpy()}))
    origins = table[[roles.SURVEY_COLUMN, roles.NATIVE_ID_COLUMN]].copy(deep=True)
    projection = roles._projection_bytes(table)
    receipt = roles.source._encode(
        {
            "protocol": roles.PROTOCOL,
            "contract_sha256": roles.contract_sha256(),
            "projection_sha256": roles._sha(projection),
            **roles._counts(table),
            "source_admission_issued": False,
            "release_eligible": False,
        },
        maximum=roles.MAX_RECEIPT_BYTES,
    )
    return roles.QualifiedCurrentSurveyHouseholdRoles(
        frame, origins, table, projection, receipt
    )


def _receiving(table, *, incumbent=None, base=0, dtype="boolean"):
    pairs = [(int(original), clone) for original in table.index for clone in (0, 1)][
        ::-1
    ]
    people = pd.DataFrame(
        {
            "person_id": np.arange(base + 1, base + 1 + len(pairs), dtype=np.int64),
            roles.support_source_id_column("person"): np.array(
                [p for p, _ in pairs], dtype=np.int64
            ),
            roles.support_clone_index_column("person"): np.array(
                [c for _, c in pairs], dtype=np.int64
            ),
            roles.spine_source_id_column("person"): np.array(
                [table.at[p, roles.NATIVE_ID_COLUMN] for p, _ in pairs], dtype=np.int64
            ),
            roles.support_channel_column("person"): pd.array(
                [table.at[p, roles.SURVEY_COLUMN] for p, _ in pairs], dtype="string"
            ),
            "unrelated_amount": np.arange(len(pairs), dtype=np.float64),
            "census_block_geoid": pd.array(
                ["060750101001001"] * len(pairs), dtype="string"
            ),
        }
    )
    if incumbent is not None:
        values = [incumbent.get(original, pd.NA) for original, _ in pairs]
        people[roles.CANONICAL_COLUMN] = pd.array(values, dtype=dtype)
    return SimpleNamespace(person=people)


@pytest.mark.parametrize(
    "state,reason,head",
    [
        (1, 0, True),
        (2, 0, False),
        (0, 1, None),
        (0, 2, None),
        (0, 3, None),
        (0, 4, None),
    ],
)
def test_asec_preserves_authenticated_owner_state_and_unknownness(state, reason, head):
    result = roles.asec_household_role_observation(state, reason)
    assert result[roles.VALUE_COLUMN] is head
    assert result[roles.VALUE_KNOWN_COLUMN] is (head is not None)
    assert result[roles.UNBOUND_REASON_COLUMN] == reason
    assert result[roles.ORIGIN_COLUMN] == roles.ASEC_ORIGIN


@pytest.mark.parametrize("state,reason", [(1, 2), (2, 3), (0, 0), (3, 0), (-1, 1)])
def test_asec_inconsistent_or_unnamed_owner_states_refuse(state, reason):
    with pytest.raises(ValueError, match="ASEC_STATE"):
        roles.asec_household_role_observation(state, reason)


@pytest.mark.parametrize("state,code", [(1, 21), (2, 20), (2, 37), (2, 38), (3, 20)])
def test_acs_state_must_agree_with_named_reference_and_universe(state, code):
    with pytest.raises(ValueError, match="CODE_DISAGREE"):
        roles.acs_household_role_observation(state, code)


@pytest.mark.parametrize(
    "codes,expected",
    [
        ((21, 20), (2, 1)),
        ((21, 22), (0, 0)),
        ((20, 20), (0, 0)),
        ((20, 99), (0, 0)),
        ((37, 38), (3, 3)),
        ((20, 37), (0, 0)),
    ],
)
def test_acs_uses_whole_household_named_role_and_universe(codes, expected):
    serial = "2024HU0000001"
    states, _ = roles._acs_states(
        {serial: {2}}, {(serial, 1): str(codes[0]), (serial, 2): str(codes[1])}
    )
    # A retained subset may not discard the other published household member.
    assert tuple(states[serial, line][0] for line in (1, 2)) == expected
    for line, state in enumerate(expected, 1):
        value = roles.acs_household_role_observation(state, codes[line - 1])
        assert value[roles.VALUE_COLUMN] is (None if state in (0, 3) else state == 1)


@pytest.mark.parametrize(
    "token,code,known",
    [
        ("20", 20, True),
        ("37", 37, True),
        ("38", 38, True),
        ("99", 99, False),
        ("", -1, False),
        ("20.0", -1, False),
        ("020", -1, False),
        (" 20", -1, False),
    ],
)
def test_relationship_literal_knownness_does_not_invent_a_role(token, code, known):
    assert roles.acs_relationship_token(token) == (code, known)


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_owner_state_is_an_exact_integer(bad):
    with pytest.raises(ValueError, match="ASEC_STATE_TYPE"):
        roles.asec_household_role_observation(bad, 0)


def test_group_quarters_codes_remain_descriptive_without_headship():
    table = _table(((10, 2, "acs", 37, 3, 5), (20, 3, "acs", 38, 3, 5)))
    assert table[roles.CODE_KNOWN_COLUMN].all()
    assert not table[roles.VALUE_KNOWN_COLUMN].any()
    assert table[roles.VALUE_COLUMN].isna().all()
    qualified = _qualified(table)
    receiving = _receiving(table)
    result = roles.household_role_columns_for_population(qualified, receiving)
    assert result["person", roles.CANONICAL_COLUMN].isna().all()


def test_exact_integer_identity_fans_to_both_clones_and_preserves_unowned_cells():
    maximum = np.iinfo(np.int64).max
    table = _table(
        (
            (maximum - 2, 2, "asec", 1, 1, 0),
            (maximum - 1, 3, "acs", 21, 2, 0),
            (maximum, 5, "asec", 6, 0, 1),
        )
    )
    qualified = _qualified(table)
    receiving = _receiving(table, incumbent={maximum - 1: False}, base=2**53 + 1)
    before = receiving.person.copy(deep=True)
    result = roles.household_role_columns_for_population(qualified, receiving)[
        "person", roles.CANONICAL_COLUMN
    ]
    original = before[roles.support_source_id_column("person")].to_numpy()
    assert result.index.equals(pd.Index(before.person_id, name="person_id"))
    assert result[original == maximum - 2].all()
    assert not result[original == maximum - 1].any()
    assert result[original == maximum].isna().all()
    pd.testing.assert_frame_equal(before, receiving.person, check_exact=True)


@pytest.mark.parametrize("known", [False, True])
def test_unsupported_incumbent_refuses_without_erasing_the_original(known):
    qualified = _qualified()
    receiving = _receiving(qualified.rows, incumbent={30: known})
    before = receiving.person.copy(deep=True)
    with pytest.raises(ValueError, match="UNBOUND_INCUMBENT"):
        roles.household_role_columns_for_population(qualified, receiving)
    pd.testing.assert_frame_equal(before, receiving.person, check_exact=True)
    audit = roles.household_role_reconciliation(qualified, receiving)
    assert audit["incumbent_known_qualified_unbound"] == 2
    assert audit["binding_would_refuse"]


def test_conflicting_incumbent_refuses_and_remains_visible_to_reconciliation():
    qualified = _qualified()
    receiving = _receiving(qualified.rows, incumbent={10: False})
    before = receiving.person.copy(deep=True)
    with pytest.raises(ValueError, match="CANONICAL_CONFLICT"):
        roles.household_role_columns_for_population(qualified, receiving)
    audit = roles.household_role_reconciliation(qualified, receiving)
    assert audit["conflicting_cells"] == 2 and audit["binding_would_refuse"]
    pd.testing.assert_frame_equal(before, receiving.person, check_exact=True)


@pytest.mark.parametrize(
    "column",
    [
        "support_source_id_person",
        "support_clone_index_person",
        "spine_source_id_person",
    ],
)
def test_clone_identity_cannot_be_float_even_when_values_are_exact(column):
    # Resolve actual shared names explicitly; no hand-maintained identity aliases.
    columns = {
        "support_source_id_person": roles.support_source_id_column("person"),
        "support_clone_index_person": roles.support_clone_index_column("person"),
        "spine_source_id_person": roles.spine_source_id_column("person"),
    }
    qualified = _qualified()
    receiving = _receiving(qualified.rows)
    receiving.person[columns[column]] = receiving.person[columns[column]].astype(float)
    with pytest.raises(ValueError, match="CLONE_AXIS_DTYPE"):
        roles.household_role_columns_for_population(qualified, receiving)


def test_public_receipt_contains_only_closed_aggregate_fields():
    qualified = _qualified()
    receiving = _receiving(qualified.rows)
    receipt = roles.household_role_binding_receipt(
        qualified,
        receiving,
        receiving_version="invented.receiving",
        declared_rewrite=False,
    )
    assert set(receipt) == roles.PUBLIC_RECEIPT_KEYS
    assert receipt["filled_cells"] == 4 and receipt["unresolved_cells"] == 2
    assert not receipt["source_admission_issued"]
    assert not receipt["unallocated_observation_claim"]
    assert receipt["relationship_allocation_provenance"] == "unresolved"


@pytest.mark.parametrize(
    "name",
    ["VALUE_COLUMN", "CODE_COLUMN", "ACS_OBSERVATION_YEAR", "MAX_HOUSEHOLD_MEMBERS"],
)
def test_live_contract_seals_individual_column_and_period_bindings(monkeypatch, name):
    before = roles._live()
    monkeypatch.setattr(
        roles, name, "changed" if type(getattr(roles, name)) is str else 2099
    )
    assert roles._live() != before


def test_caller_projection_cannot_substitute_for_the_retained_preparation():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        roles.qualify_current_survey_household_roles(_qualified())


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("header", "SOURCE_HEADER"),
        ("duplicate", "DUPLICATE_SELECTED_ACS_ROW"),
        ("line", "SOURCE_LINE"),
        ("width", "SOURCE_ROW_SHAPE"),
        ("count", "SOURCE_ROW_SHAPE"),
        ("token", "SOURCE_TOKEN_BOUND"),
    ],
)
def test_literal_acs_roster_requires_a_complete_well_formed_member(defect, reason):
    serial = "2024HU0000001"
    header = "SERIALNO,SPORDER,RELSHIPP"
    rows = [f"{serial},1,20", f"{serial},2,25"]
    maximum = 2
    if defect == "header":
        header = "SERIALNO,SPORDER,SPORDER"
    elif defect == "duplicate":
        rows[1] = rows[0]
    elif defect == "line":
        rows[1] = f"{serial},0,25"
    elif defect == "width":
        rows[1] += ",unrequested"
    elif defect == "count":
        maximum = 1
    else:
        rows[1] = f"{serial},2," + "2" * 65
    stream = BytesIO((header + "\n" + "\n".join(rows) + "\n").encode())
    with pytest.raises(ValueError, match=reason):
        roles._scan_acs_relationships(
            stream, serials={serial: {2}}, selected={}, maximum=maximum
        )


@pytest.mark.parametrize("defect", ["native", "survey", "clone", "original"])
def test_clone_join_refuses_a_changed_identity_without_mutating_receiving(defect):
    qualified = _qualified()
    receiving = _receiving(qualified.rows)
    if defect == "native":
        receiving.person.loc[0, roles.spine_source_id_column("person")] = 99
    elif defect == "survey":
        receiving.person.loc[0, roles.support_channel_column("person")] = "acs"
    elif defect == "clone":
        receiving.person.loc[0, roles.support_clone_index_column("person")] = 0
    else:
        receiving.person.loc[0, roles.support_source_id_column("person")] = 99
    before = receiving.person.copy(deep=True)
    with pytest.raises(ValueError, match="CLONE_"):
        roles.household_role_columns_for_population(qualified, receiving)
    pd.testing.assert_frame_equal(before, receiving.person, check_exact=True)
