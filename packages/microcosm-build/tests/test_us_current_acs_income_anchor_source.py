"""Actual retained ACS source owners over privately pinned invented bytes."""

import copy
import io

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_acs_income_anchor_source as owner


@pytest.mark.parametrize(
    "field,token,age,status,known",
    [
        ("INTP", "-10000", "30", "observed", True),
        ("INTP", "-4", "30", "observed", True),
        ("INTP", "0", "30", "observed", True),
        ("RETP", "0", "30", "observed", True),
        ("RETP", "999999", "30", "observed", True),
        ("INTP", "-3", "30", "outside_published_domain", False),
        ("RETP", "-4", "30", "outside_published_domain", False),
        ("INTP", "1000000", "30", "outside_published_domain", False),
        ("INTP", "", "14", "outside_universe_blank", False),
        ("INTP", "", "15", "missing_source_amount", False),
        ("RETP", "4", "14", "outside_universe_observation", False),
        ("INTP", "4.0", "30", "malformed_source_amount", False),
        ("INTP", " 4", "30", "malformed_source_amount", False),
        ("INTP", "NA", "30", "malformed_source_amount", False),
    ],
)
def test_published_anchor_domain_is_distinct_from_storage_and_universe(
    field, token, age, status, known
):
    value = owner.parse_anchor(
        token, age=age, adjustment="1000133", allocation="1", field=field
    )
    assert value["status"] == status and value["known"] is known
    assert value["allocation_status"] == "allocated"
    if known:
        expected = np.float64(int(token)) * (np.float64(1000133) / 1_000_000.0)
        assert np.float64(value["amount"]).view("uint64") == expected.view("uint64")
    else:
        assert value["amount"] is None


@pytest.mark.parametrize(
    "flag,status",
    [
        ("0", "not_allocated"),
        ("1", "allocated"),
        ("", "allocation_missing"),
        ("x", "allocation_unrecognized"),
    ],
)
def test_allocation_does_not_decide_amount_knownness(flag, status):
    value = owner.parse_anchor(
        "4", age="15", adjustment="1000133", allocation=flag, field="RETP"
    )
    assert value["known"] and value["allocation_status"] == status


@pytest.mark.parametrize("adjustment", ["", "0", "-1", "1.1", "inf"])
def test_unknown_or_invalid_adjustment_never_supplies_an_amount(adjustment):
    value = owner.parse_anchor(
        "4", age="15", adjustment=adjustment, allocation="0", field="INTP"
    )
    assert not value["known"] and value["amount"] is None
    assert value["status"] == "invalid_adjustment"


def test_column_numeric_identity_keeps_scalar_mapper_bits_and_unknowns():
    tokens = ["-800", "0", "bad", "", "4.0", "-0.0", "NA", "999999", "1000133"]
    expected = np.array(
        [
            pd.to_numeric(pd.Series([v], dtype=object), errors="coerce").iloc[0]
            for v in tokens
        ],
        dtype=np.float64,
    )
    actual = owner._native_numbers(tokens)
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(
        actual[~np.isnan(actual)].view("uint64"),
        expected[~np.isnan(expected)].view("uint64"),
    )


def _literal_rows():
    return [
        {
            "SERIALNO": "2024HU0000001",
            "SPORDER": "1",
            "INTP": "-800",
            "RETP": "444",
            "ADJINC": "1000133",
            "AGEP": "30",
            "FINTP": "0",
            "FRETP": "1",
        },
        {
            "SERIALNO": "2024HU0000001",
            "SPORDER": "2",
            "INTP": "0",
            "RETP": "",
            "ADJINC": "1000133",
            "AGEP": "15",
            "FINTP": "1",
            "FRETP": "",
        },
        {
            "SERIALNO": "2024HU0000002",
            "SPORDER": "1",
            "INTP": "bad",
            "RETP": "4",
            "ADJINC": "1000133",
            "AGEP": "80",
            "FINTP": "",
            "FRETP": "x",
        },
        *(
            {
                "SERIALNO": serial,
                "SPORDER": "1",
                "INTP": "0",
                "RETP": "0",
                "ADJINC": "1000000",
                "AGEP": age,
                "FINTP": "0",
                "FRETP": "0",
            }
            for serial, age in (("2024GQ0000001", "40"), ("2024GQ0000002", "50"))
        ),
    ]


def _scan(rows, *, wanted=None):
    import csv

    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=owner.COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    selected = {}
    wanted = (
        wanted
        if wanted is not None
        else {(r["SERIALNO"], int(r["SPORDER"])) for r in rows}
    )
    count = owner._scan(
        io.BytesIO(text.getvalue().encode()), wanted, selected, maximum=100
    )
    return count, selected


def _records_raw_table(origins, selected):
    """Pre-iteration implementation, retained only as a parity oracle."""
    owner.require(
        set(selected) == set(origins.anchor_source_key), "SELECTED_SOURCE_ROSTER"
    )
    raw = pd.DataFrame(
        [selected[k] for k in origins.anchor_source_key],
        index=origins.index.copy(),
        columns=owner.COLUMNS,
        dtype=object,
    )
    owner.require(
        all(
            owner._key(row) == key
            for row, key in zip(
                raw.to_dict("records"), origins.anchor_source_key, strict=True
            )
        ),
        "SOURCE_COORDINATE_CHANGED",
    )
    return raw


def _records_parsed_table(raw, origins):
    """Old anchor order, pandas dtypes and record boxing for exact comparison."""
    result = raw.copy(deep=True)
    result["native_person_id"] = origins.selected_receiving_person_id.to_numpy(
        copy=True
    )
    result["source_year"] = 2024
    result["dollar_year"] = 2024
    for column, flag, prefix, _output in owner.ANCHORS:
        values = [
            owner.parse_anchor(
                row[column],
                age=row["AGEP"],
                adjustment=row["ADJINC"],
                allocation=row[flag],
                field=column,
            )
            for row in raw.to_dict("records")
        ]
        parsed = pd.DataFrame(values, index=raw.index)
        for name in parsed:
            dtype = (
                "Float64"
                if name == "amount"
                else (
                    bool
                    if name in ("known", "adjustment_known", "in_income_universe")
                    else owner.STRING_DTYPE
                )
            )
            result[prefix + "_" + name] = pd.array(parsed[name], dtype=dtype)
    return result


def _iteration_fixture():
    rows = _literal_rows()
    for i, overrides in enumerate(
        (
            {"INTP": "", "RETP": "4", "AGEP": "14"},
            {"INTP": "１２", "RETP": "−4", "FINTP": "é", "FRETP": "未"},
            {"INTP": "-10000", "RETP": "999999", "ADJINC": "9999999"},
            {"INTP": "-4", "RETP": "4", "ADJINC": "1"},
            {"INTP": "0", "RETP": "0", "ADJINC": ""},
            {"INTP": "9" * 64, "RETP": "-4", "ADJINC": "0"},
        ),
        start=3,
    ):
        rows.append({**rows[0], "SERIALNO": f"2024HU{i:07d}", **overrides})
    _, selected = _scan(rows)
    origins = pd.DataFrame(
        {
            "anchor_source_key": list(reversed(selected)),
            "selected_receiving_person_id": np.arange(len(rows), dtype="int64") + 2**60,
        },
        index=pd.Index(
            np.iinfo("int64").max - np.arange(len(rows), dtype="int64") * 3,
            name="person_id",
        ),
    )
    return origins, selected


def _assert_exact_anchor_parity(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for column in ("property_income_amount", "retirement_income_amount"):
        assert (
            actual[column].array._data.tobytes()
            == expected[column].array._data.tobytes()
        )
        assert (
            actual[column].array._mask.tobytes()
            == expected[column].array._mask.tobytes()
        )
    projections = [
        table.reset_index().to_json(orient="table", index=False).encode()
        for table in (actual, expected)
    ]
    assert projections[0] == projections[1]
    qualified = [
        owner.QualifiedAcsIncomeAnchors(
            table,
            projection,
            {"projection_sha256": owner.hashlib.sha256(projection).hexdigest()},
        )
        for table, projection in zip((actual, expected), projections, strict=True)
    ]
    assert owner.income_anchor_seal(qualified[0]) == owner.income_anchor_seal(
        qualified[1]
    )


def test_column_iteration_matches_records_projection_bits_and_digest():
    origins, selected = _iteration_fixture()
    raw = owner._raw_table(origins, selected)
    expected_raw = _records_raw_table(origins, selected)
    pd.testing.assert_frame_equal(raw, expected_raw, check_exact=True)
    _assert_exact_anchor_parity(
        owner._parsed_table(raw, origins), _records_parsed_table(expected_raw, origins)
    )


def test_source_tables_do_not_expand_a_full_record_dictionary_list(monkeypatch):
    origins, selected = _iteration_fixture()
    original = pd.DataFrame.to_dict

    def refuse_records(self, orient="dict", *args, **kwargs):
        assert orient != "records", "whole-roster records allocation"
        return original(self, orient, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", refuse_records)
    parsed = owner._parsed_table(owner._raw_table(origins, selected), origins)
    assert len(parsed) == len(origins)


@pytest.mark.parametrize(
    "column,value",
    [
        ("INTP", None),
        ("INTP", pd.NA),
        ("INTP", np.nan),
        ("INTP", np.int64(4)),
        ("RETP", np.float64(4)),
        ("AGEP", "１５"),
        ("ADJINC", 1000133),
        ("FRETP", pd.NA),
    ],
)
def test_iteration_preserves_invalid_literal_refusals(column, value):
    origins, selected = _iteration_fixture()
    raw = owner._raw_table(origins, selected)
    raw.loc[raw.index[1], column] = value
    with pytest.raises(ValueError) as previous:
        _records_parsed_table(raw, origins)
    with pytest.raises(ValueError) as current:
        owner._parsed_table(raw, origins)
    assert str(current.value) == str(previous.value)


@pytest.mark.parametrize("defect", ["missing", "coordinate", "household", "person"])
def test_iteration_preserves_roster_and_coordinate_refusals(defect):
    origins, selected = _iteration_fixture()
    key = origins.anchor_source_key.iloc[0]
    if defect == "missing":
        del selected[key]
    elif defect == "coordinate":
        selected[key]["SPORDER"] = "20"
    elif defect == "household":
        selected[key]["SERIALNO"] = "bad"
    else:
        selected[key]["SPORDER"] = "21"
    with pytest.raises(ValueError) as previous:
        _records_raw_table(origins, selected)
    with pytest.raises(ValueError) as current:
        owner._raw_table(origins, selected)
    assert str(current.value) == str(previous.value)


def test_iteration_preserves_anchor_then_row_validation_order(monkeypatch):
    origins, selected = _iteration_fixture()
    raw = owner._raw_table(origins, selected)
    # A RETP failure in the first row must not precede a later INTP failure.
    raw.loc[raw.index[0], "RETP"] = None
    raw.loc[raw.index[2], "INTP"] = None
    parse = owner.parse_anchor
    calls = []

    def record_parse(token, **kwargs):
        calls.append((token, kwargs))
        return parse(token, **kwargs)

    monkeypatch.setattr(owner, "parse_anchor", record_parse)
    with pytest.raises(ValueError):
        _records_parsed_table(raw, origins)
    previous = calls.copy()
    calls.clear()
    with pytest.raises(ValueError):
        owner._parsed_table(raw, origins)
    assert calls == previous
    assert [kwargs["field"] for _, kwargs in calls] == ["INTP"] * 3


def test_literal_scan_is_keyed_and_keeps_every_original_token():
    count, values = _scan(_literal_rows()[::-1])
    assert count == 5
    assert values["2024HU0000001", 2]["RETP"] == ""
    assert values["2024HU0000002", 1]["INTP"] == "bad"


@pytest.mark.parametrize("defect", ["duplicate", "foreign_bad_key", "short_row"])
def test_literal_reader_refuses_duplicate_selected_and_invalid_tail(defect):
    rows = _literal_rows()
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "foreign_bad_key":
        rows.append({**rows[0], "SERIALNO": "bad"})
    else:
        stream = io.BytesIO((",".join(owner.COLUMNS) + "\n1,2\n").encode())
        with pytest.raises(ValueError, match="ACS_INCOME_ANCHOR_"):
            owner._scan(stream, set(), {}, maximum=100)
        return
    with pytest.raises(ValueError, match="ACS_INCOME_ANCHOR_"):
        _scan(rows)


def _arguments(tmp_path, monkeypatch):
    import test_us_survey_population_preparation as fixture

    original = fixture._person
    lookup = {(r["SERIALNO"], int(r["SPORDER"])): r for r in _literal_rows()}

    def person(*args, **kwargs):
        row = original(*args, **kwargs)
        key = (row["SERIALNO"], int(row["SPORDER"]))
        row.update(
            {
                k: v
                for k, v in lookup.get(key, {}).items()
                if k not in ("SERIALNO", "SPORDER")
            }
        )
        row.setdefault("FINTP", "0")
        row.setdefault("FRETP", "0")
        return row

    monkeypatch.setattr(fixture, "_person", person)
    return fixture.fixture(tmp_path, monkeypatch, zero=False)


def test_actual_qualifier_binds_originals_and_adjusted_bits(tmp_path, monkeypatch):
    prepared = owner.preparation.prepare_authenticated_survey_population(
        **_arguments(tmp_path, monkeypatch)
    )
    original = prepared.checked_view().frame
    before = owner.preparation._frame_identity(original)
    qualified = owner.qualify_current_acs_income_anchors(prepared)
    assert qualified.evidence["source_admission_issued"] is False
    assert qualified.evidence["decomposition_performed"] is False
    assert len(qualified.anchors) == 5
    first = qualified.anchors.set_index(["SERIALNO", "SPORDER"])
    assert first.loc[("2024HU0000001", "1"), "property_income_amount"] < 0
    assert first.loc[("2024HU0000001", "2"), "property_income_amount"] == 0
    assert (
        first.loc[("2024HU0000001", "2"), "retirement_income_status"]
        == "missing_source_amount"
    )
    assert (
        first.loc[("2024HU0000002", "1"), "property_income_status"]
        == "malformed_source_amount"
    )
    assert pd.isna(first.loc[("2024HU0000002", "1"), "property_income_amount"])
    entry = prepared._checked()
    origins = owner._origins(entry[2], owner.json.loads(entry[1]))
    _, selected = _scan(_literal_rows())
    expected = _records_parsed_table(_records_raw_table(origins, selected), origins)
    _assert_exact_anchor_parity(qualified.anchors, expected)
    assert (
        qualified.projection
        == expected.reset_index().to_json(orient="table", index=False).encode()
    )
    assert owner.preparation._frame_identity(original) == before
    fresh = owner.qualify_current_acs_income_anchors(prepared)
    assert owner.income_anchor_seal(fresh) == owner.income_anchor_seal(qualified)
    with pytest.raises(ValueError):
        owner.qualify_current_acs_income_anchors(copy.copy(prepared))


@pytest.mark.parametrize(
    "defect", ["row_order", "person_id", "raw_amount", "adjusted_bit"]
)
def test_exact_retained_comparison_refuses_source_or_amount_mutation(
    tmp_path, monkeypatch, defect
):
    prepared = owner.preparation.prepare_authenticated_survey_population(
        **_arguments(tmp_path, monkeypatch)
    )
    entry = prepared._checked()
    document = owner.json.loads(entry[1])
    origins = owner._origins(entry[2], document)
    _, selected = _scan(_literal_rows())
    raw = owner._raw_table(origins, selected)
    native = owner.preparation._copy_source(entry[2].source_frames[0])
    if defect == "row_order":
        origins = origins.iloc[::-1].copy()
    elif defect == "person_id":
        native.person.loc[native.person.index[0], "person_id"] += 100
    elif defect == "raw_amount":
        native.person.loc[native.person.index[0], "RETP"] = 888
    else:
        column = "acs_interest_dividend_rental_income"
        native.person.loc[native.person.index[0], column] = np.nextafter(
            native.person[column].iloc[0], np.inf
        )
    with pytest.raises(ValueError, match="ACS_INCOME_ANCHOR_"):
        owner._compare_retained(raw, origins, native)


def test_final_source_io_mutation_cannot_return_qualified_value(tmp_path, monkeypatch):
    prepared = owner.preparation.prepare_authenticated_survey_population(
        **_arguments(tmp_path, monkeypatch)
    )
    original = prepared.checked_view().frame
    capture = owner._capture_person
    mutated = []

    def mutate_after_io(*args):
        value = capture(*args)
        original.person.loc[original.person.index[0], "age"] += 1
        mutated.append(True)
        return value

    monkeypatch.setattr(owner, "_capture_person", mutate_after_io)
    with pytest.raises(ValueError):
        owner.qualify_current_acs_income_anchors(prepared)
    assert mutated == [True]


def test_final_source_fence_seals_exact_returned_amount_bits(tmp_path, monkeypatch):
    prepared = owner.preparation.prepare_authenticated_survey_population(
        **_arguments(tmp_path, monkeypatch)
    )
    original_seal = owner.income_anchor_seal
    changed = []

    def change_after_seal(qualified):
        seal = original_seal(qualified)
        if not changed:
            changed.append(True)
            table = qualified.anchors
            index = table.index[
                table.SERIALNO.eq("2024HU0000001") & table.SPORDER.eq("1")
            ][0]
            table.loc[index, "property_income_amount"] = np.nextafter(
                table.loc[index, "property_income_amount"], np.inf
            )
            # Portable JSON's precision is not an exact physical mutation seal.
            assert (
                qualified.projection
                == table.reset_index().to_json(orient="table", index=False).encode()
            )
        return seal

    monkeypatch.setattr(owner, "income_anchor_seal", change_after_seal)
    with pytest.raises(ValueError, match="FINAL_VALUES_CHANGED"):
        owner.qualify_current_acs_income_anchors(prepared)
    assert changed == [True]
