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
