"""ACS literal qualification and genuine invented-source owner controls."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import (
    current_acs_immigration_source_projection as owner,
)


def raw_rows(rows):
    return pd.DataFrame(
        [
            dict(SERIALNO="2024HU0000001", SPORDER=str(i + 1), **r)
            for i, r in enumerate(rows)
        ],
        columns=owner.literals.ACS_COLUMNS,
        dtype=owner.literals.STRING,
    )


@pytest.mark.parametrize(
    "cit,year,precision,lower,upper",
    [
        ("1", "", "not_in_universe", None, None),
        ("2", "1938", "bottom_coded", None, 1938),
        ("3", "1939", "interval", 1939, 1944),
        ("4", "1945", "calendar_year", 1945, 1945),
        ("5", "2024", "calendar_year", 2024, 2024),
    ],
)
def test_published_entry_universe_and_precision(cit, year, precision, lower, upper):
    raw = raw_rows([dict(CIT=cit, POBP="464", YOEP=year, AGEP="99")])
    typed = owner._numeric_literals(raw)
    assert typed.at[0, "CIT"] == int(cit)
    assert typed.at[0, "YOEP_precision"] == precision
    assert typed.at[0, "YOEP_is_niu"] == (year == "")
    assert typed.at[0, "POBP_precision"] == "Tunisia_Western_Sahara_South_Sudan_group"
    for name, expected in (("YOEP_lower", lower), ("YOEP_upper", upper)):
        assert (
            pd.isna(typed.at[0, name])
            if expected is None
            else typed.at[0, name] == expected
        )
    assert raw.at[0, "YOEP"] == year


@pytest.mark.parametrize("cit", ["2", "3", "4", "5"])
def test_only_cit_one_allows_entry_niu(cit):
    with pytest.raises(ValueError, match="YOEP_REQUIRED"):
        owner._numeric_literals(
            raw_rows([dict(CIT=cit, POBP="451", YOEP="", AGEP="45")])
        )


@pytest.mark.parametrize(
    "field,token",
    [
        ("CIT", "0"),
        ("CIT", "6"),
        ("CIT", "True"),
        ("CIT", "1.0"),
        ("POBP", "057"),
        ("POBP", "073"),
        ("POBP", "463"),
        ("POBP", "999"),
        ("POBP", ""),
        ("YOEP", "0"),
        ("YOEP", "1900"),
        ("YOEP", "1937"),
        ("YOEP", "1940"),
        ("YOEP", "1944"),
        ("YOEP", "2025"),
        ("YOEP", "2000.5"),
        ("YOEP", "True"),
        ("AGEP", "100"),
        ("AGEP", "-1"),
    ],
)
def test_literal_domains_refuse_unresolved_codes(field, token):
    values = dict(CIT="5", POBP="451", YOEP="2001", AGEP="45")
    values[field] = token
    with pytest.raises(ValueError):
        owner._numeric_literals(raw_rows([values]))


def source_arguments(path, patch, *, missing_entry=False):
    import test_us_current_survey_immigration_source as fixture
    import test_us_survey_population_preparation as preparation_fixture

    old = preparation_fixture._person
    from test_us_acs_person_coverage_authentication import _csv as csv

    def person(*args, **kwargs):
        row = old(*args, **kwargs)
        row["SEX"] = "2" if int(row["SPORDER"]) == 2 else "1"
        return row

    def source_csv(rows):
        if rows and "SPORDER" in rows[0]:
            rows = [dict(r) for r in rows]
            for row, cit, birthplace, entry in zip(
                rows,
                ("1", "2", "3", "4", "5"),
                ("001", "072", "200", "451", "464"),
                ("", "2020", "2000", "2010", "2024"),
                strict=True,
            ):
                row.update(
                    CIT=cit, POBP=birthplace, YOEP="" if missing_entry else entry
                )
        return csv(rows)

    patch.setattr(preparation_fixture, "_person", person)
    patch.setattr(preparation_fixture, "_csv", source_csv)
    return fixture.source_arguments(path, patch)


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        full_args, partial_args, _ = source_arguments(
            tmp_path_factory.mktemp("acs-immigration"), patch
        )
        full = owner.source.prepare_authenticated_survey_population(**full_args)
        partial = owner.source.prepare_authenticated_survey_population(**partial_args)
        yield SimpleNamespace(
            full=full,
            partial=partial,
            view=owner.borrow_current_acs_immigration_projection(full),
            smaller=owner.borrow_current_acs_immigration_projection(partial),
        )


def test_real_selected_source_projection_has_no_stock_or_status_authority(actual):
    view = actual.view
    view.validate()
    assert view.raw.index.equals(view.person.index)
    assert view.person.source.eq("acs").all()
    assert view.person.observation_year.eq(2024).all()
    assert view.person.source_year.eq(2024).all()
    assert sorted(view.person.CIT) == [1, 2, 3, 4, 5]
    assert view.raw.YOEP.eq("").sum() == 1
    assert view.person.YOEP.isna().sum() == 1
    assert view.person.YOEP_is_niu.eq(view.person.CIT.eq(1)).all()
    assert view.person.sort_values("CIT").POBP.tolist() == [1, 72, 200, 451, 464]
    assert (
        view.person.loc[view.person.POBP.eq(464), "POBP_precision"].iloc[0]
        == "Tunisia_Western_Sahara_South_Sudan_group"
    )
    assert view.person.is_female.eq(view.person.SEX.eq(2)).all()
    assert view.person.state_fips.eq(6).all()
    assert not {
        "person_weight",
        "household_weight",
        "WSAL_VAL",
        "immigration_status",
    } & set(view.person)
    receipt = json.loads(view.receipt)
    assert receipt["projection_scope"] == "selected_original_ACS_people"
    for key in (
        "status_assignment_performed",
        "national_stock_alignment_qualified",
        "source_admission_issued",
    ):
        assert receipt[key] is False
    assert receipt["person_weight_authority"] == "none"
    assert receipt["unallocated_observation_claim"] is False
    assert "2024GQ" in " ".join(view.raw.SERIALNO)
    # Re-selection cannot become a full-country denominator claim.
    assert set(actual.smaller.raw.index) <= set(view.raw.index)


@pytest.mark.parametrize("field", ["person", "raw", "households"])
def test_copy_and_equal_detached_tables_refuse(actual, field):
    with pytest.raises(ValueError, match="RETAINED_VIEW"):
        copy.copy(actual.view).validate()
    with pytest.raises(ValueError, match="RETAINED_VIEW"):
        replace(actual.view).validate()
    original = getattr(actual.view, field)
    try:
        object.__setattr__(actual.view, field, original.copy(deep=True))
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.view.validate()
    finally:
        object.__setattr__(actual.view, field, original)


@pytest.mark.parametrize(
    "field,column,value",
    [
        ("person", "AGEP", 99),
        ("person", "SEX", 2),
        ("person", "observation_year", 2025),
        ("person", "native_person_id", 999),
        ("person", "person_household_id", 999),
        ("raw", "CIT", "5"),
        ("households", "state_fips", 1),
    ],
)
def test_mutated_projection_refuses(actual, field, column, value):
    table = getattr(actual.view, field)
    row, old = table.index[0], table.iloc[0][column]
    try:
        table.at[row, column] = value
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.view.validate()
    finally:
        table.at[row, column] = old
    actual.view.validate()


def test_unissued_preparation_and_wrong_owner_refuse(actual):
    with pytest.raises(ValueError):
        owner.borrow_current_acs_immigration_projection(copy.copy(actual.full))
    with pytest.raises(ValueError):
        owner.borrow_current_acs_immigration_projection(actual.view)
    old = actual.view.receipt
    try:
        object.__setattr__(actual.view, "receipt", actual.smaller.receipt)
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.view.validate()
    finally:
        object.__setattr__(actual.view, "receipt", old)


def test_final_file_io_cannot_mutate_output(actual, monkeypatch):
    table = actual.view.person
    row, old = table.index[0], table.iloc[0].AGEP
    read, triggered = Path.read_bytes, []

    def mutate(path):
        result = read(path)
        if not triggered:
            triggered.append(True)
            table.at[row, "AGEP"] = 99
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", mutate)
            with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
                actual.view.validate()
        assert triggered
    finally:
        table.at[row, "AGEP"] = old


def test_literal_closure_and_value_mutation_cannot_move_retained_seal(actual):
    retained = owner._RETAINED[id(actual.view)][2]
    qualified = retained.qualified
    raw, callback = qualified.acs_raw, qualified._revalidate
    row, old = raw.index[0], raw.iloc[0].YOEP
    cell = dict(zip(callback.__code__.co_freevars, callback.__closure__, strict=True))[
        "seals"
    ]
    previous = cell.cell_contents
    try:
        raw.at[row, "YOEP"] = "2001"
        tables = (
            qualified.origins,
            qualified.asec_full_raw,
            qualified.asec_selected_raw,
            qualified.acs_raw,
            qualified.person_evidence,
        )
        cell.cell_contents = tuple(owner.literals._table_seal(t) for t in tables)
        qualified.validate()  # Its editable closure is deliberately insufficient.
        with pytest.raises(ValueError, match="LITERAL_CHANGED"):
            actual.view.validate()
    finally:
        raw.at[row, "YOEP"] = old
        cell.cell_contents = previous
    actual.view.validate()


def test_genuine_source_missing_entry_refuses_before_projection(tmp_path, monkeypatch):
    full, _, _ = source_arguments(tmp_path, monkeypatch, missing_entry=True)
    preparation = owner.source.prepare_authenticated_survey_population(**full)
    with pytest.raises(ValueError, match="YOEP_REQUIRED"):
        owner.borrow_current_acs_immigration_projection(preparation)


def projection_inputs(actual):
    retained = owner._RETAINED[id(actual.view)][2]
    native = retained.preparation_entry[2].source_frames[0]
    return [
        actual.view.raw.copy(deep=True),
        retained.qualified.origins.copy(deep=True),
        copy.deepcopy(
            json.loads(retained.preparation_entry[1])["origins"]["households"]
        ),
        native.person.copy(deep=True),
        native.table("household").copy(deep=True),
        retained.preparation_entry[2].frame.person.copy(deep=True),
    ]


@pytest.mark.parametrize(
    "defect",
    [
        "age",
        "sex",
        "sex_boolean",
        "state",
        "membership",
        "line",
        "literal_household",
        "year",
        "household_year",
        "household_source",
        "household_key",
    ],
)
def test_pure_source_coordinate_checks(actual, defect):
    raw, origins, h_origins, people, houses, receiving = args = projection_inputs(
        actual
    )
    if defect == "age":
        people.loc[0, "A_AGE"] += 1
    elif defect in ("sex", "sex_boolean"):
        people["SEX"] = people.SEX.astype(object)
        people.loc[0, "SEX"] = True if defect == "sex_boolean" else 2
    elif defect == "state":
        houses.loc[0, "ST"] = "72"
    elif defect == "membership":
        people.loc[0, "person_household_id"] = houses.household_id.iloc[-1]
    elif defect == "line":
        people.loc[0, "SPORDER"] = 2
    elif defect == "literal_household":
        raw.loc[raw.index[0], "SERIALNO"] = "2024HU9999999"
    elif defect == "year":
        origins.loc[raw.index[0], "survey_year"] = 2025
    else:
        target = next(r for r in h_origins if r["source"] == "acs")
        target[
            {
                "household_year": "survey_year",
                "household_source": "source",
                "household_key": "raw_native_id",
            }[defect]
        ] = {
            "household_year": 2025,
            "household_source": "asec",
            "household_key": "2024HU9999999",
        }[defect]
    with pytest.raises(ValueError):
        owner._project(*args)


def test_pure_projection_reorders_by_exact_keys_and_preserves_large_ids(actual):
    raw, origins, h_origins, people, houses, receiving = args = projection_inputs(
        actual
    )
    offset = 2**90
    ids = {old: offset + int(old) for old in receiving.person_id}
    raw.index = pd.Index([ids[v] for v in raw.index], name=raw.index.name)
    origins.index = pd.Index([ids[v] for v in origins.index], name=origins.index.name)
    receiving["person_id"] = receiving.person_id.map(ids)
    hids = {old: 2 * offset + int(old) for old in receiving.person_household_id}
    receiving["person_household_id"] = receiving.person_household_id.map(hids)
    for row in h_origins:
        row["household_id"] = hids[row["household_id"]]
    args[3] = people.iloc[::-1].copy()
    args[4] = houses.iloc[::-1].copy()
    args[5] = receiving.iloc[::-1].copy()
    person, household = owner._project(*args)
    assert person.index.equals(raw.index)
    assert person.person_household_id.tolist() == [
        hids[v] for v in actual.view.person.person_household_id
    ]
    assert set(household.household_id) == set(person.person_household_id)
    pd.testing.assert_series_equal(
        person.AGEP.reset_index(drop=True),
        actual.view.person.AGEP.reset_index(drop=True),
    )


def test_literal_line_zero_padding_preserves_the_same_coordinate(actual):
    args = projection_inputs(actual)
    args[0]["SPORDER"] = args[0].SPORDER.str.zfill(2)
    person, households = owner._project(*args)
    pd.testing.assert_frame_equal(person, actual.view.person)
    pd.testing.assert_frame_equal(households, actual.view.households)


@pytest.mark.parametrize("which", ["SEX", "AGEP", "source_year"])
def test_authentic_native_table_mutation_is_refused(actual, which):
    retained = owner._RETAINED[id(actual.view)][2]
    table = retained.preparation_entry[2].source_frames[0].person
    row, old = table.index[0], table.iloc[0][which]
    try:
        table.at[row, which] = old + 1
        with pytest.raises(ValueError):
            actual.view.validate()
    finally:
        table.at[row, which] = old
    actual.view.validate()


def test_late_io_cannot_mutate_literal_even_if_callback_was_already_checked(
    actual, monkeypatch
):
    retained = owner._RETAINED[id(actual.view)][2]
    raw = retained.qualified.acs_raw
    row, old = raw.index[0], raw.iloc[0].CIT
    read, triggered = Path.read_bytes, []

    def mutate(path):
        result = read(path)
        if path == Path(owner.__file__) and not triggered:
            triggered.append(True)
            raw.at[row, "CIT"] = "5"
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", mutate)
            with pytest.raises(ValueError, match="LITERAL_CHANGED"):
                actual.view.validate()
        assert triggered
    finally:
        raw.at[row, "CIT"] = old
    actual.view.validate()
