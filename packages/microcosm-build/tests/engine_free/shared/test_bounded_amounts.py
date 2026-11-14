"""Whether amounts on a record, or summed from its members, stay within the
bound the record carries. Every table below is invented."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.bounded_amounts import FINITE_NONNEGATIVE, amounts_within_bound

TOLERANCE = 1e-9


def _records(bill=(100.0, 40.0, 0.0), rebate=(30.0, 40.0, 0.0)) -> pd.DataFrame:
    """Three records, each with a bill and a rebate on the record itself."""
    return pd.DataFrame(
        {
            "record_id": np.asarray([101, 102, 103], dtype=np.int64),
            "bill": np.asarray(bill, dtype=np.float64),
            "rebate": np.asarray(rebate, dtype=np.float64),
        }
    )


def _members(reduction=(60.0, 40.0, 25.0, 0.0), owner=(101, 101, 102, 101)):
    """Four members of the first two records; the third record has none."""
    return pd.DataFrame(
        {
            "member_record_id": np.asarray(owner, dtype=np.int64),
            "reduction": np.asarray(reduction, dtype=np.float64),
        }
    )


def _check(records=None, members=None, **overrides) -> dict[str, bool]:
    arguments = {
        "entity": "record",
        "id_column": "record_id",
        "bound_column": "bill",
        "same_entity_columns": ("rebate",),
        "member_columns": {"member": ("reduction",)},
        "membership_columns": {"member": "member_record_id"},
        "absolute_tolerance": TOLERANCE,
        **overrides,
    }
    tables = {
        "record": _records() if records is None else records,
        "member": _members() if members is None else members,
    }
    return amounts_within_bound(tables, **arguments)


def test_amounts_at_their_bound_pass_with_one_boolean_per_column():
    # The first record's members sum to its bill (60 + 40 + 0), the second's
    # rebate equals its bill, and the third has a zero bill and no member.
    result = _check()
    assert result == {"rebate": True, "reduction": True, FINITE_NONNEGATIVE: True}
    assert list(result) == ["rebate", "reduction", FINITE_NONNEGATIVE]
    assert all(type(value) is bool for value in result.values())


@pytest.mark.parametrize("above", [2e-9, 1e-6, 1.0])
def test_an_amount_just_above_its_bound_fails(above):
    result = _check(_records(rebate=(30.0, 40.0 + above, 0.0)))
    assert result == {"rebate": False, "reduction": True, FINITE_NONNEGATIVE: True}


def test_the_tolerance_is_honoured():
    # Within 1e-9 of the bill passes, on the record and summed from members.
    within = _check(
        _records(rebate=(30.0, 40.0 + 5e-10, 0.0)),
        _members(reduction=(60.0, 40.0 + 5e-10, 25.0, 0.0)),
    )
    assert within == {"rebate": True, "reduction": True, FINITE_NONNEGATIVE: True}
    # Binary fractions show the edge exactly: 100.25 is the bill plus a
    # quarter, the tolerance; 100.5 is past it.
    at_edge = _records(bill=(100.0, 40.0, 0.0), rebate=(100.25, 0.0, 0.0))
    assert _check(at_edge, absolute_tolerance=0.25)["rebate"] is True
    past = _records(bill=(100.0, 40.0, 0.0), rebate=(100.5, 0.0, 0.0))
    assert _check(past, absolute_tolerance=0.25)["rebate"] is False
    # Without a tolerance the bound itself passes and the next float fails.
    exact = _records(rebate=(30.0, 40.0, 0.0))
    assert _check(exact, absolute_tolerance=0)["rebate"] is True
    nudged = _records(rebate=(30.0, float(np.nextafter(40.0, np.inf)), 0.0))
    assert _check(nudged, absolute_tolerance=0.0)["rebate"] is False


def test_a_two_member_sum_above_the_bound_fails():
    # Each of the second record's two members is within its bill of 40; their
    # sum is not.
    members = _members(reduction=(60.0, 40.0, 25.0, 25.0), owner=(101, 101, 102, 102))
    result = _check(members=members)
    assert result == {"rebate": True, "reduction": False, FINITE_NONNEGATIVE: True}


def test_members_count_toward_their_own_record_only():
    # 75 on the first record's bill of 100 passes; on the second's 40 it would
    # not, and the third has no member at all.
    members = _members(reduction=(75.0, 0.0, 40.0, 0.0), owner=(101, 103, 102, 101))
    assert _check(members=members)["reduction"] is True
    moved = _members(reduction=(75.0, 0.0, 40.0, 0.0), owner=(102, 103, 101, 101))
    assert _check(members=moved)["reduction"] is False


@pytest.mark.parametrize(
    ("table", "column", "value", "expected"),
    [
        ("record", "rebate", np.nan, (False, True)),
        ("record", "rebate", np.inf, (False, True)),
        ("record", "rebate", -1.0, (True, True)),
        ("record", "rebate", -np.inf, (True, True)),
        ("member", "reduction", np.nan, (True, False)),
        ("member", "reduction", np.inf, (True, False)),
        ("member", "reduction", -1.0, (True, True)),
        ("member", "reduction", -np.inf, (True, True)),
        ("record", "bill", np.nan, (False, False)),
        ("record", "bill", np.inf, (True, True)),
        ("record", "bill", -np.inf, (False, False)),
        ("record", "bill", -1.0, (False, False)),
    ],
)
def test_nan_infinite_and_negative_values_fail_where_they_are_read(
    table, column, value, expected
):
    """A NaN or plus-infinite amount fails its own column, a negative one,
    minus infinity included, does not, and each fails finite_nonnegative; a
    NaN, negative or minus-infinite bill fails both columns of its record."""
    records, members = _records(), _members()
    if table == "record":
        records.loc[0, column] = value
    else:
        members.loc[1, column] = value
    result = _check(records, members)
    assert (result["rebate"], result["reduction"]) == expected
    assert result[FINITE_NONNEGATIVE] is False


def test_a_missing_value_reads_as_nan():
    records = _records().astype({"rebate": "Float64"})
    records.loc[1, "rebate"] = pd.NA
    members = _members().astype({"reduction": "Float64"})
    members.loc[2, "reduction"] = pd.NA
    assert _check(records, members) == {
        "rebate": False,
        "reduction": False,
        FINITE_NONNEGATIVE: False,
    }


def test_integer_amounts_and_tables_without_rows_are_read():
    records = _records().astype({"bill": np.int64, "rebate": np.int64})
    members = _members().astype({"reduction": np.int32})
    assert all(_check(records, members).values())
    assert all(_check(_records().iloc[:0], _members().iloc[:0]).values())


def test_exclusions_are_the_callers():
    """Every record and member given is read; a record the caller leaves out
    goes with its members before the call."""
    records = _records(bill=(100.0, 40.0, -5.0))
    members = _members(owner=(101, 101, 102, 103))
    result = _check(records, members)
    assert result == {"rebate": False, "reduction": False, FINITE_NONNEGATIVE: False}
    kept = records[records["record_id"] != 103]
    assert _check(kept, members[members["member_record_id"] != 103]) == {
        "rebate": True,
        "reduction": True,
        FINITE_NONNEGATIVE: True,
    }


def test_the_verdicts_do_not_depend_on_row_order():
    rng = np.random.default_rng(931)
    records = pd.DataFrame(
        {
            "record_id": np.arange(500, dtype=np.int64) * 7 + 3,
            "bill": rng.integers(0, 50, 500).astype(np.float64),
            "rebate": rng.integers(0, 50, 500).astype(np.float64),
        }
    )
    owner = rng.choice(records["record_id"].to_numpy(), 1500)
    members = pd.DataFrame(
        {"member_record_id": owner, "reduction": rng.integers(0, 5, 1500) * 1.0}
    )
    expected = _check(records, members)
    assert expected["rebate"] is False and expected["reduction"] is False
    shuffled = _check(
        records.iloc[rng.permutation(500)], members.iloc[rng.permutation(1500)]
    )
    assert shuffled == expected
    bills = members.groupby("member_record_id")["reduction"].sum()
    records["bill"] = (
        records["record_id"].map(bills).fillna(0.0) + records["rebate"] + 1.0
    )
    assert all(_check(records, members).values())


def test_several_member_tables_report_in_the_order_named():
    groups = pd.DataFrame(
        {
            "group_record_id": np.asarray([103, 102], dtype=np.int64),
            "credit": [0.0, 1.0],
        }
    )
    result = amounts_within_bound(
        {"record": _records(), "member": _members(), "group": groups},
        entity="record",
        id_column="record_id",
        bound_column="bill",
        same_entity_columns=(),
        member_columns={"group": ["credit"], "member": ("reduction",)},
        membership_columns={"member": "member_record_id", "group": "group_record_id"},
        absolute_tolerance=TOLERANCE,
    )
    assert list(result) == ["credit", "reduction", FINITE_NONNEGATIVE]
    assert all(result.values())


def _refused(**overrides) -> str:
    records = overrides.pop("records", None)
    members = overrides.pop("members", None)
    with pytest.raises(ValueError) as refused:
        _check(records, members, **overrides)
    text = str(refused.value)
    assert text.startswith("Bounded amounts: ")
    assert not any(character.isdigit() for character in text)
    return text


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"entity": ""}, "must be names"),
        ({"bound_column": " bill"}, "must be names"),
        ({"bound_column": "record_id"}, "must not be the id column"),
        ({"same_entity_columns": "rebate"}, "must list columns"),
        ({"same_entity_columns": ("rebate", None)}, "must list columns"),
        ({"member_columns": {"member": "reduction"}}, "lists of columns"),
        (
            {
                "member_columns": {"member": ("reduction",), "record": ("bill",)},
                "membership_columns": {"member": "member_record_id", "record": "x"},
            },
            "must not be the record's own table",
        ),
        ({"membership_columns": {}}, "one column for each member table"),
        ({"membership_columns": {"member": ""}}, "one column for each member table"),
        (
            {"same_entity_columns": (), "member_columns": {}, "membership_columns": {}},
            "no amount column",
        ),
        ({"member_columns": {"member": ("rebate",)}}, "named twice"),
        ({"same_entity_columns": ("rebate", "rebate")}, "named twice"),
        ({"same_entity_columns": (FINITE_NONNEGATIVE,)}, "may be named"),
        ({"same_entity_columns": ("bill",)}, "the id or the bound column"),
        ({"same_entity_columns": ("record_id",)}, "the id or the bound column"),
        (
            {"member_columns": {"member": ("member_record_id",)}},
            "its table's membership column",
        ),
        ({"absolute_tolerance": -1e-9}, "finite nonnegative number"),
        ({"absolute_tolerance": float("nan")}, "finite nonnegative number"),
        ({"absolute_tolerance": float("inf")}, "finite nonnegative number"),
        ({"absolute_tolerance": True}, "finite nonnegative number"),
        ({"absolute_tolerance": "0"}, "finite nonnegative number"),
        ({"entity": "household"}, "not a DataFrame"),
        (
            {
                "member_columns": {"person": ("reduction",)},
                "membership_columns": {"person": "member_record_id"},
            },
            "not a DataFrame",
        ),
        ({"id_column": "household_id"}, "has no household_id column"),
        ({"bound_column": "council"}, "has no council column"),
        ({"same_entity_columns": ("credit",)}, "has no credit column"),
        ({"membership_columns": {"member": "owner"}}, "has no owner column"),
    ],
)
def test_a_declaration_it_cannot_read_is_refused_naming_no_data(overrides, message):
    assert message in _refused(**overrides)


@pytest.mark.parametrize(
    ("records", "members", "message"),
    [
        (_records().assign(rebate=["a", "b", "c"]), None, "not a numeric column"),
        (_records().assign(rebate=[True, False, True]), None, "not a numeric column"),
        (
            _records().assign(bill=np.asarray([1, 2, 3], dtype=object)),
            None,
            "not a numeric",
        ),
        (None, _members().assign(reduction=["a"] * 4), "not a numeric column"),
        (_records().assign(record_id=[101, 101, 103]), None, "name each row once"),
        (
            _records().assign(record_id=[101.0, np.nan, 103.0]),
            None,
            "name each row once",
        ),
        (None, _members(owner=(101, 101, 102, 104)), "names no record row"),
        (
            None,
            _members().assign(member_record_id=[101.0, np.nan, 102.0, 101.0]),
            "names no record row",
        ),
        (
            pd.concat([_records(), _records()[["rebate"]]], axis=1),
            None,
            "repeats its rebate column",
        ),
    ],
)
def test_tables_it_cannot_read_are_refused_naming_no_data(records, members, message):
    assert message in _refused(records=records, members=members)


def test_tables_must_be_a_mapping_naming_no_data():
    with pytest.raises(ValueError) as refused:
        amounts_within_bound(
            [("record", _records())],
            entity="record",
            id_column="record_id",
            bound_column="bill",
            same_entity_columns=("rebate",),
            member_columns={},
            membership_columns={},
            absolute_tolerance=TOLERANCE,
        )
    assert str(refused.value) == (
        "Bounded amounts: tables must map entities to DataFrames"
    )
