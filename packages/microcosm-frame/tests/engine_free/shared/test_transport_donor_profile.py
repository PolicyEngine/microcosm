"""Donor-decoding profiles: the closed document and the reader's staging.

A synthetic donor carries the raw survey answers a profile reads. Without a
profile the reader reads exactly what it did before; with one it stages
tenure, marital status and full-time study, reconciles housing costs with
tenure, and records what it did. The property tests drive the staging
against an independent oracle. No test opens a real donor.
"""

import ast
import itertools
import json
import re
from hashlib import sha256
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import microcosm.frame.transport as transport_module
from microcosm.frame.concepts import (
    MARITAL_STATUS_DOMAIN,
    RENTED_TENURES,
    TENURE_DOMAIN,
    validate_concept_tables,
)
from microcosm.frame.transport import (
    DONOR_PROFILE_CHOICES,
    DONOR_PROFILE_FORMAT,
    DonorProfile,
    parse_donor_profile,
    read_populace_us_donor,
)
from test_support.paths import paths_for

FIXTURE = paths_for("microcosm-frame").tests / "fixtures/transport/synthetic_donor.json"
PROPERTY = settings(
    max_examples=60,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
NAME, VERSION = "populace-us-cps-asec", 1
OFFERED = {
    "assisted_renter_tenure": ("rented_public_authority", "rented_private"),
    "blank_full_time_study": ("refuse", "read_false"),
    "housing_cost_contradicting_tenure": ("tenure_wins", "refuse"),
}
CHOICES = {
    "assisted_renter_tenure": "rented_public_authority",
    "blank_full_time_study": "read_false",
    "housing_cost_contradicting_tenure": "tenure_wins",
}
STAGED = {
    "person": ("legal_marital_status", "enrolled_full_time"),
    "household": ("tenure",),
}
#: The codes the oracle reads, written out independently of the module. The
#: official 2025 CPS ASEC data dictionary, printed page 6C-3
#: (https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf),
#: maps codes 1 to 3 to married, 4 widowed, 5 divorced, 6 separated and 7
#: never married.
A_MARITL = {
    1: "married",
    2: "married",
    3: "married",
    4: "widowed",
    5: "divorced",
    6: "separated",
    7: "never_married",
}


def _canonical(document: dict) -> str:
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _document(donor_sha256: str, **choices) -> dict:
    return {
        "format": DONOR_PROFILE_FORMAT,
        "profile": NAME,
        "version": VERSION,
        "donor_sha256": donor_sha256,
        "choices": {**CHOICES, **choices},
    }


def _profile(donor_sha256: str, **choices) -> DonorProfile:
    return parse_donor_profile(_canonical(_document(donor_sha256, **choices)))


def _raw_tables() -> dict[str, pd.DataFrame]:
    """The synthetic donor plus the raw columns a profile reads.

    Household 101 owns with a mortgage and carries rent; household 202 rents,
    with one assisted and one unassisted SPM unit, and carries mortgage
    interest. Person 4 (source year 2024) has blank enrollment answers.
    """
    document = json.loads(FIXTURE.read_text())
    tables = {name: pd.DataFrame(rows) for name, rows in document["tables"].items()}
    person = tables["person"]
    person["home_mortgage_interest"] = [12.0, 0.0, 0.0, 7.0, 0.0, 0.0]
    person["person_spm_unit_id"] = [11, 11, 11, 21, 22, 22]
    person["A_MARITL"] = [1, 1, 7, 1, 1, 7]
    person["A_ENRLW"] = [2.0, 1.0, 1.0, np.nan, 0.0, 1.0]
    person["A_FTPT"] = [0.0, 1.0, 1.0, np.nan, 0.0, 2.0]
    household = tables["household"]
    household["tenure_type"] = ["OWNED_WITH_MORTGAGE", "RENTED"]
    household["H_TENURE"] = [1, 2]
    tables["spm_unit"] = pd.DataFrame(
        {
            "spm_unit_id": [11, 21, 22],
            "spm_unit_tenure_type": ["OWNER_WITH_MORTGAGE", "RENTER", "RENTER"],
            "receives_housing_assistance": [False, False, True],
        }
    )
    return tables


def _write(path: Path, tables: dict[str, pd.DataFrame]) -> dict:
    with pd.HDFStore(path, mode="w") as store:
        for name, table in tables.items():
            store.put(name, table, format="table")
    return {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def _read(tmp_path, tables=None, **choices):
    path = tmp_path / "donor.h5"
    pin = _write(path, _raw_tables() if tables is None else tables)
    return read_populace_us_donor(
        path, **pin, profile=_profile(pin["sha256"], **choices)
    )


# --- The document -----------------------------------------------------------


def test_the_value_sets_live_in_one_place_and_name_the_donor_family():
    assert DONOR_PROFILE_FORMAT == "microcosm.donor-profile/1"
    assert {key: dict(value) for key, value in DONOR_PROFILE_CHOICES.items()} == {
        (NAME, VERSION): OFFERED
    }
    # Every tenure a choice can write is a schema tenure.
    assert set(OFFERED["assisted_renter_tenure"]) <= set(TENURE_DOMAIN)


def test_the_document_parses_once_into_its_canonical_text_and_digest():
    text = _canonical(_document("a" * 64))
    profile = parse_donor_profile(text)
    assert (profile.name, profile.version, profile.donor_sha256) == (
        NAME,
        VERSION,
        "a" * 64,
    )
    assert dict(profile.choices) == CHOICES
    assert profile.canonical_text == text
    assert profile.sha256 == sha256(text.encode("utf-8")).hexdigest()
    assert parse_donor_profile(text) == profile
    # The document carries no target_references key, which composition trims.
    assert "target_references" not in json.loads(text)


@pytest.mark.parametrize(
    "values", list(itertools.product(*OFFERED.values())), ids=lambda v: "-".join(v)
)
def test_every_offered_choice_value_parses(values):
    choices = dict(zip(OFFERED, values, strict=True))
    profile = _profile("b" * 64, **choices)
    assert dict(profile.choices) == choices


def _without(document, *path):
    document = json.loads(json.dumps(document))
    target = document
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return document


def _with(document, value, *path):
    document = json.loads(json.dumps(document))
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return document


BASE = _document("c" * 64)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([BASE], "must be a JSON object"),
        ("text", "must be a JSON object"),
        (_with(BASE, "x", "target_references"), r"unknown field\(s\) \['target_"),
        (_with(BASE, "nz", "country"), r"unknown field\(s\) \['country'\]"),
        (_with(BASE, "d" * 64, "donor_profile_sha256"), "unknown field"),
        (_without(BASE, "format"), r"lacks field\(s\) \['format'\]"),
        (_without(BASE, "profile"), r"lacks field\(s\) \['profile'\]"),
        (_without(BASE, "version"), r"lacks field\(s\) \['version'\]"),
        (_without(BASE, "donor_sha256"), r"lacks field\(s\) \['donor_sha256'\]"),
        (_without(BASE, "choices"), r"lacks field\(s\) \['choices'\]"),
        (
            _with(BASE, "microcosm.donor-profile/2", "format"),
            "Unknown donor profile format",
        ),
        (_with(BASE, "populace-us", "profile"), "Unknown donor profile"),
        (_with(BASE, 2, "version"), "Unknown donor profile"),
        (_with(BASE, "1", "version"), "int version"),
        (_with(BASE, True, "version"), "int version"),
        (_with(BASE, 1.0, "version"), "int version"),
        (_with(BASE, "C" * 64, "donor_sha256"), "64 lowercase hexadecimal"),
        (_with(BASE, "c" * 63, "donor_sha256"), "64 lowercase hexadecimal"),
        (_with(BASE, None, "donor_sha256"), "64 lowercase hexadecimal"),
        (_with(BASE, [], "choices"), "choices must be an object"),
        (_with(BASE, "x", "choices", "rent_rule"), r"declares no choice field\(s\)"),
        (
            _without(BASE, "choices", "blank_full_time_study"),
            r"requires choice field\(s\) \['blank_full_time_study'\]",
        ),
        (
            _with(BASE, "rented_nonprofit_social", "choices", "assisted_renter_tenure"),
            "must be one of",
        ),
        (_with(BASE, "impute", "choices", "blank_full_time_study"), "must be one of"),
        (_with(BASE, "keep", "choices", "housing_cost_contradicting_tenure"), "one of"),
        (_with(BASE, 1, "choices", "blank_full_time_study"), "must be one of"),
    ],
)
def test_an_unknown_or_missing_part_of_the_document_refuses(document, message):
    with pytest.raises(ValueError, match=message):
        parse_donor_profile(_canonical(document))


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(BASE, sort_keys=True),
        json.dumps(BASE, separators=(",", ":")),
        _canonical(BASE) + "\n",
        _canonical(BASE).replace('"profile"', '"\\u0070rofile"'),
    ],
)
def test_a_non_canonical_spelling_refuses(text):
    assert json.loads(text) == BASE
    with pytest.raises(ValueError, match="must be canonical JSON"):
        parse_donor_profile(text)


@pytest.mark.parametrize(
    ("text", "error", "message"),
    [
        (None, TypeError, "must be JSON text"),
        (b"{}", TypeError, "must be JSON text"),
        ("{", ValueError, "is not JSON"),
        ('{"format":NaN}', ValueError, "cannot hold NaN"),
        (
            _canonical(BASE)[:-1] + ',"version":1}',
            ValueError,
            "repeats a key",
        ),
    ],
)
def test_a_document_that_is_not_one_json_object_refuses(text, error, message):
    with pytest.raises(error, match=message):
        parse_donor_profile(text)


def test_a_profile_built_directly_is_held_to_the_same_closed_sets():
    with pytest.raises(ValueError, match="must be one of"):
        DonorProfile(NAME, VERSION, "a" * 64, {**CHOICES, "blank_full_time_study": "x"})
    with pytest.raises(ValueError, match="Unknown donor profile"):
        DonorProfile("populace-us-cps-asec-nz", VERSION, "a" * 64, CHOICES)


@PROPERTY
@given(
    values=st.tuples(*(st.sampled_from(values) for values in OFFERED.values())),
    digest=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
    position=st.integers(min_value=0, max_value=300),
)
def test_canonical_text_round_trips_and_any_inserted_space_refuses(
    values, digest, position
):
    choices = dict(zip(OFFERED, values, strict=True))
    text = _canonical(_document(digest, **choices))
    profile = parse_donor_profile(text)
    assert profile.canonical_text == text
    assert dict(profile.choices) == choices
    # A space anywhere either breaks the JSON, changes a name or value, or
    # respells the same document; each refuses.
    cut = position % (len(text) + 1)
    with pytest.raises(ValueError):
        parse_donor_profile(text[:cut] + " " + text[cut:])


# --- The reader without a profile -------------------------------------------


def test_without_a_profile_the_reader_opens_no_spm_unit_and_stages_nothing(
    tmp_path, monkeypatch
):
    path = tmp_path / "donor.h5"
    pin = _write(path, _raw_tables())
    opened = []
    real = transport_module.read_frame_table

    def spy(store, key):
        opened.append(key)
        return real(store, key)

    monkeypatch.setattr(transport_module, "read_frame_table", spy)
    plain = read_populace_us_donor(path, **pin)
    explicit = read_populace_us_donor(path, **pin, profile=None)
    assert opened == ["person", "household"] * 2
    for donor in (plain, explicit):
        assert donor.donor_profile is None
        for entity, columns in STAGED.items():
            assert not set(columns) & set(donor.tables[entity].columns)
    for entity in ("person", "household"):
        pd.testing.assert_frame_equal(plain.tables[entity], explicit.tables[entity])
    # The unprofiled frame keeps the donor's rent on the owner household.
    assert plain.tables["household"]["rent"].tolist() == [25.0, 0.0]
    assert plain.tables["household"]["mortgage_interest"].tolist() == [12.0, 7.0]
    assert validate_concept_tables(plain.tables) == ()


# --- The reader with a profile ----------------------------------------------


def test_a_profile_stages_reconciles_and_records(tmp_path):
    donor = _read(tmp_path)
    household = donor.tables["household"]
    person = donor.tables["person"]
    assert household["tenure"].tolist() == [
        "owned_with_mortgage",
        "rented_public_authority",
    ]
    assert household["rent"].tolist() == [0.0, 0.0]
    assert household["mortgage_interest"].tolist() == [12.0, 0.0]
    assert person["legal_marital_status"].tolist() == [
        "married",
        "married",
        "never_married",
        "married",
        "married",
        "never_married",
    ]
    assert person["enrolled_full_time"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    # Staged text has the dtype decode gives its own text concepts.
    assert household["tenure"].dtype == person["sex"].dtype
    assert person["legal_marital_status"].dtype == person["sex"].dtype
    assert validate_concept_tables(donor.tables) == ()
    record = donor.donor_profile
    assert record["profile"] == NAME and record["version"] == VERSION
    assert record["choices"] == CHOICES
    assert record["document_sha256"] == _profile(record["donor_sha256"]).sha256
    assert set(record["tenure"]) == set(TENURE_DOMAIN)
    assert record["tenure"]["owned_with_mortgage"] == {"households": 1, "weight": 1.0}
    assert record["tenure"]["rented_public_authority"] == {
        "households": 1,
        "weight": 3.0,
    }
    assert record["household_tenure_type"] == {
        "OWNED_WITH_MORTGAGE": 1,
        "RENTED": 1,
        "NONE": 0,
    }
    assert record["spm_units"] == {
        "units": 3,
        "units_without_members": 0,
        "assistance_rule": "any_spm_unit_in_household",
        "households_with_mixed_assistance": 1,
        "assisted_renter_households": 1,
        "assisted_other_households": 0,
        "rent_free_households_with_spm_owner_tenure": 0,
        "renter_households_with_spm_owner_tenure": 0,
    }
    assert record["reconciliation"] == {
        "rent": {
            "column": "present",
            "contradicting_households": 1,
            "contradicting_amount": 25.0,
            "contradicting_weighted_amount": 25.0,
            "total_amount": 25.0,
            "weighted_total_amount": 25.0,
            "zeroed": True,
        },
        "mortgage_interest": {
            "column": "present",
            "contradicting_households": 1,
            "contradicting_amount": 7.0,
            "contradicting_weighted_amount": 21.0,
            "total_amount": 19.0,
            "weighted_total_amount": 33.0,
            "zeroed": True,
        },
        # Absent, not unreconciled: the donor carries no principal.
        "mortgage_principal": {"column": "absent"},
    }
    assert record["full_time_study"] == {
        "enrolled_full_time": 2,
        "blank_a_enrlw_by_source_year": {"2024": 1},
        "blank_a_ftpt_by_source_year": {"2024": 1},
        "read_false_by_source_year": {"2024": 1},
    }
    assert record["legal_marital_status"] == {
        "never_married": 2,
        "married": 4,
        "separated": 0,
        "divorced": 0,
        "widowed": 0,
    }
    assert set(record["legal_marital_status"]) == set(MARITAL_STATUS_DOMAIN)
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("value", OFFERED["assisted_renter_tenure"])
def test_assisted_renter_tenure_takes_each_offered_value(tmp_path, value):
    donor = _read(tmp_path, assisted_renter_tenure=value)
    assert donor.tables["household"]["tenure"].tolist() == [
        "owned_with_mortgage",
        value,
    ]
    # Both values are rented tenures, so the renter's rent survives either.
    assert value in RENTED_TENURES
    assert donor.donor_profile["choices"]["assisted_renter_tenure"] == value
    assert donor.donor_profile["tenure"][value]["households"] == 1


def test_an_unassisted_renter_is_a_private_renter_under_either_value(tmp_path):
    tables = _raw_tables()
    tables["spm_unit"]["receives_housing_assistance"] = [False, False, False]
    for value in OFFERED["assisted_renter_tenure"]:
        donor = _read(tmp_path, tables, assisted_renter_tenure=value)
        assert donor.tables["household"]["tenure"].tolist()[1] == "rented_private"


def test_read_false_reads_blanks_as_not_studying_and_counts_them_by_year(tmp_path):
    tables = _raw_tables()
    tables["person"]["A_ENRLW"] = [np.nan, 1.0, 1.0, np.nan, 0.0, 1.0]
    tables["person"]["A_FTPT"] = [np.nan, 1.0, np.nan, 1.0, 0.0, 2.0]
    # One source year per household, as lineage requires.
    tables["person"]["source_year"] = [2022, 2022, 2022, 2023, 2023, 2023]
    donor = _read(tmp_path, tables, blank_full_time_study="read_false")
    assert donor.tables["person"]["enrolled_full_time"].tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert donor.donor_profile["full_time_study"] == {
        "enrolled_full_time": 1,
        "blank_a_enrlw_by_source_year": {"2022": 1, "2023": 1},
        "blank_a_ftpt_by_source_year": {"2022": 2, "2023": 0},
        "read_false_by_source_year": {"2022": 2, "2023": 1},
    }


def test_refuse_refuses_a_blank_full_time_study_answer_with_its_counts(tmp_path):
    with pytest.raises(ValueError, match=r"blank .*\{'2024': 1\}.*refuses"):
        _read(tmp_path, blank_full_time_study="refuse")


def test_refuse_accepts_a_donor_with_no_blank_answers(tmp_path):
    tables = _raw_tables()
    tables["person"]["A_ENRLW"] = [2.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    tables["person"]["A_FTPT"] = [0.0, 1.0, 1.0, 0.0, 0.0, 2.0]
    donor = _read(tmp_path, tables, blank_full_time_study="refuse")
    assert donor.donor_profile["full_time_study"]["read_false_by_source_year"] == {
        "2024": 0
    }


def test_tenure_wins_zeroes_contradicting_costs_and_refuse_refuses_them(tmp_path):
    donor = _read(tmp_path, housing_cost_contradicting_tenure="tenure_wins")
    assert donor.tables["household"]["rent"].tolist() == [0.0, 0.0]
    with pytest.raises(
        ValueError,
        match=r"\{'rent': 1, 'mortgage_interest': 1\}, which this donor profile refuses",
    ):
        _read(tmp_path, housing_cost_contradicting_tenure="refuse")


def test_refuse_keeps_costs_that_agree_with_tenure(tmp_path):
    tables = _raw_tables()
    tables["person"]["pre_subsidy_rent"] = 0.0
    tables["person"].loc[3, "pre_subsidy_rent"] = 40.0
    tables["person"]["home_mortgage_interest"] = [12.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for value in OFFERED["housing_cost_contradicting_tenure"]:
        donor = _read(tmp_path, tables, housing_cost_contradicting_tenure=value)
        assert donor.tables["household"]["rent"].tolist() == [0.0, 40.0]
        assert donor.tables["household"]["mortgage_interest"].tolist() == [12.0, 0.0]
        rent = donor.donor_profile["reconciliation"]["rent"]
        assert rent["column"] == "present"
        assert rent["contradicting_households"] == 0
        assert rent["zeroed"] is False


def test_a_profile_naming_other_bytes_refuses_before_the_file_is_opened(
    tmp_path, monkeypatch
):
    path = tmp_path / "donor.h5"
    pin = _write(path, _raw_tables())

    def never(*_, **__):
        raise AssertionError("A mismatched profile must refuse before any read.")

    monkeypatch.setattr(Path, "open", never)
    monkeypatch.setattr(pd, "HDFStore", never)
    with pytest.raises(ValueError, match="names donor bytes other than"):
        read_populace_us_donor(path, **pin, profile=_profile("e" * 64))
    with pytest.raises(TypeError, match="must be a parsed DonorProfile"):
        read_populace_us_donor(
            path, **pin, profile=_canonical(_document(pin["sha256"]))
        )


def _edited(edit):
    tables = _raw_tables()
    edit(tables)
    return tables


def _set(table, column, values):
    def edit(tables):
        tables[table][column] = values

    return edit


def _drop(table, column):
    def edit(tables):
        tables[table].pop(column)

    return edit


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda t: t.pop("spm_unit"), "needs the donor's spm_unit table"),
        (_drop("household", "H_TENURE"), r"\['household.H_TENURE'\]"),
        (_drop("household", "tenure_type"), r"\['household.tenure_type'\]"),
        (_drop("person", "person_spm_unit_id"), r"\['person.person_spm_unit_id'\]"),
        (_drop("person", "A_FTPT"), r"\['person.A_FTPT'\]"),
        (_drop("person", "A_MARITL"), r"\['person.A_MARITL'\]"),
        (
            _drop("spm_unit", "receives_housing_assistance"),
            r"\['spm_unit.receives_housing_assistance'\]",
        ),
        (_set("person", "A_MARITL", [1, 1, 7, 1, 1, 8]), "A_MARITL holds codes"),
        (_set("person", "A_MARITL", [0, 1, 7, 1, 1, 7]), "A_MARITL holds codes"),
        (_set("person", "A_MARITL", [1.0, 1, 7, 1, 1, np.nan]), "A_MARITL has 1 blank"),
        (_set("person", "A_MARITL", [1.5, 1, 7, 1, 1, 7]), "A_MARITL holds codes"),
        (_set("person", "A_ENRLW", [2.0, 1.0, 1.0, np.nan, 0.0, 3.0]), "A_ENRLW holds"),
        (_set("person", "A_ENRLW", [np.inf, 1, 1, np.nan, 0, 1]), "A_ENRLW holds"),
        (_set("person", "A_FTPT", [0.0, 1.5, 1.0, np.nan, 0.0, 2.0]), "A_FTPT holds"),
        (_set("household", "tenure_type", ["OWNED_WITH_MORTGAGE", "LEASED"]), "labels"),
        (_set("household", "tenure_type", ["OWNED_OUTRIGHT", "RENTED"]), "labels"),
        (_set("household", "tenure_type", ["NONE", None]), "labels outside"),
        (_set("household", "H_TENURE", [1, 3]), "disagrees with H_TENURE in 1"),
        (_set("household", "H_TENURE", [1.0, np.nan]), "H_TENURE has 1 blank"),
        (_set("household", "H_TENURE", [1, 4]), "H_TENURE holds codes"),
        (
            _set("spm_unit", "spm_unit_tenure_type", ["OWNER", "RENTER", "RENTER"]),
            "spm_unit_tenure_type holds labels",
        ),
        (
            _set("spm_unit", "receives_housing_assistance", [0, 0, 1]),
            "must be boolean",
        ),
        (_set("spm_unit", "spm_unit_id", [11, 21, 21]), "must be unique"),
        (_set("spm_unit", "spm_unit_id", [11.0, 21.0, np.nan]), "integer ids"),
        (_set("person", "person_spm_unit_id", [11.0] * 3 + [21.0] + [22.0] * 2), "ids"),
        (_set("spm_unit", "spm_unit_id", [11, 21, 23]), "2 donor persons belong"),
        (
            _set("person", "person_spm_unit_id", [11, 11, 11, 11, 22, 22]),
            "1 donor SPM units span more than one household",
        ),
        (
            _set("person", "person_spm_unit_id", [11, 11, 21, 21, 22, 22]),
            "span more than one household",
        ),
        (
            _set(
                "spm_unit",
                "spm_unit_tenure_type",
                ["OWNER_WITH_MORTGAGE", "RENTER", "OWNER_WITH_MORTGAGE"],
            ),
            "1 donor households have SPM units that disagree on tenure",
        ),
        (
            _set("spm_unit", "spm_unit_tenure_type", ["RENTER", "RENTER", "RENTER"]),
            "owner households have SPM renter tenure",
        ),
        (
            _set("person", "pre_subsidy_rent", [np.inf, 0, 0, 0, 0, 0]),
            "rent must be finite before reconciliation",
        ),
    ],
)
def test_unexpected_values_and_failed_joins_refuse(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, _edited(edit))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        # A blank H_TENURE refuses as a blank, not as a disagreement.
        (_set("household", "H_TENURE", [np.nan, 3.0]), "H_TENURE has 1 blank"),
        # An unknown label refuses before H_TENURE is compared with it.
        (_set("household", "tenure_type", ["NONE", "LEASED"]), "labels outside"),
        # An invalid code refuses before any join or tenure is built.
        (
            lambda t: (
                _set("person", "A_MARITL", [9, 1, 7, 1, 1, 7])(t),
                _set("spm_unit", "spm_unit_id", [11, 21, 21])(t),
            ),
            "A_MARITL holds codes",
        ),
    ],
)
def test_raw_checks_fire_before_any_comparison_reads_the_value(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, _edited(edit))


def test_a_memberless_spm_unit_is_counted_and_stages_nothing(tmp_path):
    """A unit with no member reaches no household, so it cannot change a
    staged value: the reader counts it rather than refusing it."""
    (tmp_path / "plain").mkdir()
    (tmp_path / "extra").mkdir()
    plain = _read(tmp_path / "plain")
    tables = _raw_tables()
    tables["spm_unit"] = pd.concat(
        [
            tables["spm_unit"],
            pd.DataFrame(
                {
                    "spm_unit_id": [99],
                    "spm_unit_tenure_type": ["OWNER_WITHOUT_MORTGAGE"],
                    "receives_housing_assistance": [True],
                }
            ),
        ],
        ignore_index=True,
    )
    extra = _read(tmp_path / "extra", tables)
    for entity in ("person", "household"):
        pd.testing.assert_frame_equal(plain.tables[entity], extra.tables[entity])
    assert extra.donor_profile["spm_units"]["units"] == 4
    assert extra.donor_profile["spm_units"]["units_without_members"] == 1
    # The new bytes change the donor and document digests, and nothing else.
    differ = {"spm_units", "donor_sha256", "document_sha256"}
    assert {k: v for k, v in extra.donor_profile.items() if k not in differ} == {
        k: v for k, v in plain.donor_profile.items() if k not in differ
    }


def test_validation_still_runs_on_the_staged_frame(tmp_path, monkeypatch):
    seen = []
    real = transport_module.validate_concept_tables

    def spy(tables):
        seen.append({entity: set(table.columns) for entity, table in tables.items()})
        return real(tables)

    monkeypatch.setattr(transport_module, "validate_concept_tables", spy)
    _read(tmp_path)
    assert any("tenure" in columns["household"] for columns in seen)
    monkeypatch.setattr(transport_module, "validate_concept_tables", real)
    tables = _raw_tables()
    tables["person"]["age"] = [40, 38, -1, 45, 42, 12]
    with pytest.raises(ValueError, match="Invalid donor concepts"):
        _read(tmp_path, tables)


def _copy_household(tables, source, copy, weight):
    """Append household ``source`` and its persons again as household ``copy``."""
    person, household = tables["person"], tables["household"]
    members = person.loc[person["person_household_id"] == source].copy()
    first = len(person) + 1
    members["person_id"] = person["person_id"].max() + np.arange(1, len(members) + 1)
    members["person_household_id"] = copy
    members["source_household_id"] = copy
    numbers = [f"{n:022d}" for n in range(first, first + len(members))]
    members["source_person_id"] = numbers
    members["person_source_id"] = numbers
    row = household.loc[household["household_id"] == source].copy()
    row["household_id"] = copy
    row["household_weight"] = weight
    tables["person"] = pd.concat([person, members], ignore_index=True)
    tables["household"] = pd.concat([household, row], ignore_index=True)


def _five_households() -> dict[str, pd.DataFrame]:
    """Five households that give every SPM counter of the record a value.

    101 (weight 1) owns with a mortgage; its two SPM units differ in
    assistance. 202 (weight 3) rents; its two SPM units differ in
    assistance. 303 (weight 2, 101's persons again) lives rent free in an
    unassisted SPM unit that owns without a mortgage. 404 (weight 5, 202's
    persons again) lives rent free in an assisted SPM unit that owns with a
    mortgage. 505 (weight 4, 202's persons again) rents, in an unassisted SPM
    unit that owns without a mortgage.
    """
    tables = _raw_tables()
    _copy_household(tables, 101, 303, weight=2.0)
    _copy_household(tables, 202, 404, weight=5.0)
    _copy_household(tables, 202, 505, weight=4.0)
    household = tables["household"]
    household["tenure_type"] = [
        "OWNED_WITH_MORTGAGE",
        "RENTED",
        "NONE",
        "NONE",
        "RENTED",
    ]
    household["H_TENURE"] = [1, 2, 3, 3, 2]
    tables["person"]["person_spm_unit_id"] = [
        *(11, 11, 12),
        *(21, 22, 22),
        *(31, 31, 31),
        *(41, 41, 41),
        *(51, 51, 51),
    ]
    tables["spm_unit"] = pd.DataFrame(
        {
            "spm_unit_id": [11, 12, 21, 22, 31, 41, 51],
            "spm_unit_tenure_type": [
                "OWNER_WITH_MORTGAGE",
                "OWNER_WITH_MORTGAGE",
                "RENTER",
                "RENTER",
                "OWNER_WITHOUT_MORTGAGE",
                "OWNER_WITH_MORTGAGE",
                "OWNER_WITHOUT_MORTGAGE",
            ],
            "receives_housing_assistance": [
                True,
                False,
                False,
                True,
                False,
                True,
                False,
            ],
        }
    )
    return tables


def test_rent_free_assisted_owner_and_mixed_households_give_the_whole_record(
    tmp_path,
):
    path = tmp_path / "donor.h5"
    pin = _write(path, _five_households())
    profile = _profile(pin["sha256"])
    donor = read_populace_us_donor(path, **pin, profile=profile)
    household = donor.tables["household"]
    assert household["household_id"].tolist() == [101, 202, 303, 404, 505]
    assert household["tenure"].tolist() == [
        "owned_with_mortgage",
        "rented_public_authority",
        "rent_free",
        "rent_free",
        "rented_private",
    ]
    # Rent survives only for renters, interest only for the mortgaged owner.
    assert household["rent"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert household["mortgage_interest"].tolist() == [12.0, 0.0, 0.0, 0.0, 0.0]
    assert validate_concept_tables(donor.tables) == ()
    empty = {"households": 0, "weight": 0.0}
    assert donor.donor_profile == {
        "profile": NAME,
        "version": VERSION,
        "document_sha256": profile.sha256,
        "donor_sha256": pin["sha256"],
        "choices": CHOICES,
        "tenure": {
            "owned_with_mortgage": {"households": 1, "weight": 1.0},
            "owned_outright": empty,
            "rented_private": {"households": 1, "weight": 4.0},
            "rented_public_authority": {"households": 1, "weight": 3.0},
            "rented_nonprofit_social": empty,
            "rent_free": {"households": 2, "weight": 7.0},
        },
        "household_tenure_type": {
            "OWNED_WITH_MORTGAGE": 1,
            "RENTED": 2,
            "NONE": 2,
        },
        "spm_units": {
            "units": 7,
            "units_without_members": 0,
            "assistance_rule": "any_spm_unit_in_household",
            # 101 and 202.
            "households_with_mixed_assistance": 2,
            # 202.
            "assisted_renter_households": 1,
            # 101, an owner, and 404, rent free.
            "assisted_other_households": 2,
            # 303 and 404.
            "rent_free_households_with_spm_owner_tenure": 2,
            # 505.
            "renter_households_with_spm_owner_tenure": 1,
        },
        "full_time_study": {
            "enrolled_full_time": 4,
            "blank_a_enrlw_by_source_year": {"2024": 3},
            "blank_a_ftpt_by_source_year": {"2024": 3},
            "read_false_by_source_year": {"2024": 3},
        },
        "legal_marital_status": {
            "married": 10,
            "widowed": 0,
            "divorced": 0,
            "separated": 0,
            "never_married": 5,
        },
        "reconciliation": {
            # 25 each on owner 101 (weight 1) and rent-free 303 (weight 2).
            "rent": {
                "column": "present",
                "contradicting_households": 2,
                "contradicting_amount": 50.0,
                "contradicting_weighted_amount": 75.0,
                "total_amount": 50.0,
                "weighted_total_amount": 75.0,
                "zeroed": True,
            },
            # 7 on 202 (weight 3), 12 on 303 (weight 2), 7 on 404 (weight 5),
            # 7 on 505 (weight 4); 101 keeps its 12.
            "mortgage_interest": {
                "column": "present",
                "contradicting_households": 4,
                "contradicting_amount": 33.0,
                "contradicting_weighted_amount": 108.0,
                "total_amount": 45.0,
                "weighted_total_amount": 120.0,
                "zeroed": True,
            },
            "mortgage_principal": {"column": "absent"},
        },
    }
    assert set(donor.donor_profile["tenure"]) == set(TENURE_DOMAIN)


# Each A_MARITL code against the status the data dictionary cited at
# A_MARITL above gives it.
@pytest.mark.parametrize(
    ("code", "status"),
    [
        (1, "married"),
        (2, "married"),
        (3, "married"),
        (4, "widowed"),
        (5, "divorced"),
        (6, "separated"),
        (7, "never_married"),
    ],
)
def test_each_a_maritl_code_reads_as_its_codebook_status(tmp_path, code, status):
    assert A_MARITL[code] == status
    tables = _raw_tables()
    tables["person"]["A_MARITL"] = [code] * 6
    donor = _read(tmp_path, tables)
    assert donor.tables["person"]["legal_marital_status"].tolist() == [status] * 6
    assert donor.donor_profile["legal_marital_status"] == {
        value: 6 if value == status else 0 for value in MARITAL_STATUS_DOMAIN
    }


# --- Staging properties against an independent oracle ---------------------


@st.composite
def _donors(draw, *, blanks=True, costs_agree=False):
    """Raw households, SPM units and persons, and the decoded household rows.

    With ``blanks=False`` no enrollment answer is blank. With
    ``costs_agree=True`` only renters carry rent and only owners whose SPM
    units hold a mortgage carry mortgage interest, so no cost contradicts
    the staged tenure.
    """
    enrollment = [0.0, 1.0, 2.0, np.nan] if blanks else [0.0, 1.0, 2.0]
    n_households = draw(st.integers(1, 6))
    households, units, persons = [], [], []
    for h in range(n_households):
        label = draw(st.sampled_from(["OWNED_WITH_MORTGAGE", "RENTED", "NONE"]))
        owner = label == "OWNED_WITH_MORTGAGE"
        spm_tenure = draw(
            st.sampled_from(
                ["OWNER_WITH_MORTGAGE", "OWNER_WITHOUT_MORTGAGE"]
                if owner
                else ["OWNER_WITH_MORTGAGE", "OWNER_WITHOUT_MORTGAGE", "RENTER"]
            )
        )
        pays_rent = label == "RENTED" or not costs_agree
        pays_interest = (
            owner and spm_tenure == "OWNER_WITH_MORTGAGE"
        ) or not costs_agree
        households.append(
            {
                "household_id": 100 + h,
                "tenure_type": label,
                "H_TENURE": {"OWNED_WITH_MORTGAGE": 1, "RENTED": 2, "NONE": 3}[label],
                "rent": draw(st.sampled_from([0.0, 1.0, 250.5])) if pays_rent else 0.0,
                "mortgage_interest": (
                    draw(st.sampled_from([0.0, 3.0, 99.25])) if pays_interest else 0.0
                ),
                "weight": draw(st.sampled_from([0.5, 1.0, 4.0])),
            }
        )
        for u in range(draw(st.integers(1, 2))):
            unit_id = 1000 + 10 * h + u
            units.append(
                {
                    "spm_unit_id": unit_id,
                    "spm_unit_tenure_type": spm_tenure,
                    "receives_housing_assistance": draw(st.booleans()),
                }
            )
            for _ in range(draw(st.integers(1, 2))):
                persons.append(
                    {
                        "person_id": len(persons) + 1,
                        "person_household_id": 100 + h,
                        "person_spm_unit_id": unit_id,
                        "source_year": draw(st.sampled_from([2022, 2023, 2024])),
                        "A_MARITL": draw(st.integers(1, 7)),
                        "A_ENRLW": draw(st.sampled_from(enrollment)),
                        "A_FTPT": draw(st.sampled_from(enrollment)),
                    }
                )
    person_order = draw(st.permutations(range(len(persons))))
    household_order = draw(st.permutations(range(n_households)))
    return (
        pd.DataFrame(households),
        pd.DataFrame(units),
        pd.DataFrame([persons[i] for i in person_order]),
        list(household_order),
    )


def _stage(households, units, persons, profile, decoded_order=None):
    """Run the raw staging, then attach to decoded tables in another order."""
    raw_household = households[["household_id", "tenure_type", "H_TENURE"]].copy()
    staged = transport_module._stage_donor_profile(
        profile, persons, raw_household, units
    )
    order = list(range(len(households))) if decoded_order is None else decoded_order
    decoded = households.iloc[order].reset_index(drop=True)
    tables = {
        "person": persons[["person_id", "person_household_id"]]
        .iloc[::-1]
        .reset_index(drop=True),
        "household": decoded[["household_id", "rent", "mortgage_interest"]].copy(),
    }
    weights = households["weight"].to_numpy(dtype=np.float64)
    record = transport_module._attach_donor_profile(staged, tables, weights)
    return tables, record


def _oracle_tenure(row, units, assisted_renter_tenure):
    own = units.loc[units.spm_unit_id // 10 == row.household_id]
    if row.tenure_type == "NONE":
        return "rent_free"
    if row.tenure_type == "RENTED":
        assisted = own.receives_housing_assistance.any()
        return assisted_renter_tenure if assisted else "rented_private"
    if own.spm_unit_tenure_type.iloc[0] == "OWNER_WITH_MORTGAGE":
        return "owned_with_mortgage"
    return "owned_outright"


def _pays(tenure: str, column: str) -> bool:
    if column == "rent":
        return tenure in RENTED_TENURES
    return tenure == "owned_with_mortgage"


def test_weighted_record_values_follow_decoded_rows_not_raw_rows():
    """Decoded rows in another order than the raw rows, unequal weights."""
    households = pd.DataFrame(
        {
            "household_id": [100, 101, 102],
            "tenure_type": ["OWNED_WITH_MORTGAGE", "RENTED", "NONE"],
            "H_TENURE": [1, 2, 3],
            "rent": [10.0, 250.5, 1.0],
            "mortgage_interest": [3.0, 99.25, 0.0],
            "weight": [0.5, 4.0, 1.0],
        }
    )
    units = pd.DataFrame(
        {
            "spm_unit_id": [1000, 1010, 1020],
            "spm_unit_tenure_type": [
                "OWNER_WITH_MORTGAGE",
                "RENTER",
                "OWNER_WITHOUT_MORTGAGE",
            ],
            "receives_housing_assistance": [False, True, False],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [100, 101, 102],
            "person_spm_unit_id": [1000, 1010, 1020],
            "source_year": [2024, 2024, 2024],
            "A_MARITL": [1, 4, 7],
            "A_ENRLW": [1.0, 1.0, 2.0],
            "A_FTPT": [1.0, 2.0, 1.0],
        }
    )
    # Decoded rows 102, 100, 101 against raw weights 0.5, 4.0, 1.0.
    tables, record = _stage(households, units, persons, _profile("f" * 64), [2, 0, 1])
    household = tables["household"]
    assert household["household_id"].tolist() == [102, 100, 101]
    assert household["tenure"].tolist() == [
        "rent_free",
        "owned_with_mortgage",
        "rented_public_authority",
    ]
    assert household["rent"].tolist() == [0.0, 0.0, 250.5]
    assert household["mortgage_interest"].tolist() == [0.0, 3.0, 0.0]
    empty = {"households": 0, "weight": 0.0}
    assert record["tenure"] == {
        "owned_with_mortgage": {"households": 1, "weight": 0.5},
        "owned_outright": empty,
        "rented_private": empty,
        "rented_public_authority": {"households": 1, "weight": 4.0},
        "rented_nonprofit_social": empty,
        "rent_free": {"households": 1, "weight": 1.0},
    }
    assert record["reconciliation"] == {
        # 1.0 on rent-free 102 (weight 1) and 10.0 on owner 100 (weight 0.5).
        "rent": {
            "column": "present",
            "contradicting_households": 2,
            "contradicting_amount": 11.0,
            "contradicting_weighted_amount": 6.0,
            "total_amount": 261.5,
            "weighted_total_amount": 1008.0,
            "zeroed": True,
        },
        # 99.25 on renter 101 (weight 4); owner 100 keeps its 3.0.
        "mortgage_interest": {
            "column": "present",
            "contradicting_households": 1,
            "contradicting_amount": 99.25,
            "contradicting_weighted_amount": 397.0,
            "total_amount": 102.25,
            "weighted_total_amount": 398.5,
            "zeroed": True,
        },
        "mortgage_principal": {"column": "absent"},
    }
    # The helper hands persons to the attach step in reverse order.
    person = tables["person"]
    assert person["person_id"].tolist() == [3, 2, 1]
    assert person["legal_marital_status"].tolist() == [
        "never_married",
        "widowed",
        "married",
    ]
    assert person["enrolled_full_time"].tolist() == [False, False, True]


@PROPERTY
@given(
    values=st.tuples(*(st.sampled_from(values) for values in OFFERED.values())),
    data=st.data(),
)
def test_staging_follows_the_rules_and_leaves_a_consistent_frame(values, data):
    """Each drawn donor stages: under a refusing choice the donor has no
    blank answer, or no cost that contradicts its tenure."""
    choices = dict(zip(OFFERED, values, strict=True))
    households, units, persons, order = data.draw(
        _donors(
            blanks=choices["blank_full_time_study"] == "read_false",
            costs_agree=choices["housing_cost_contradicting_tenure"] == "refuse",
        )
    )
    profile = _profile("f" * 64, **choices)
    blank = (persons.A_ENRLW.isna() | persons.A_FTPT.isna()).to_numpy()
    tenure_by_id = {
        row.household_id: _oracle_tenure(row, units, choices["assisted_renter_tenure"])
        for row in households.itertuples()
    }
    rented = np.asarray(
        [tenure_by_id[i] in RENTED_TENURES for i in households.household_id]
    )
    mortgaged = np.asarray(
        [tenure_by_id[i] == "owned_with_mortgage" for i in households.household_id]
    )
    rent = households["rent"].to_numpy()
    interest = households["mortgage_interest"].to_numpy()
    contradicts = ((~rented) & (rent != 0)).any() or (
        (~mortgaged) & (interest != 0)
    ).any()
    assert not (blank.any() and choices["blank_full_time_study"] == "refuse")
    assert not (
        contradicts and choices["housing_cost_contradicting_tenure"] == "refuse"
    )
    tables, record = _stage(households, units, persons, profile, order)
    household = tables["household"]
    assert household["tenure"].tolist() == [
        tenure_by_id[i] for i in household["household_id"]
    ]
    assert "rented_nonprofit_social" not in household["tenure"].tolist()
    by_id = households.set_index("household_id")
    expected_rent = [
        by_id.rent[i] if tenure_by_id[i] in RENTED_TENURES else 0.0
        for i in household["household_id"]
    ]
    expected_interest = [
        by_id.mortgage_interest[i] if tenure_by_id[i] == "owned_with_mortgage" else 0.0
        for i in household["household_id"]
    ]
    assert household["rent"].tolist() == expected_rent
    assert household["mortgage_interest"].tolist() == expected_interest
    weights = households["weight"].to_numpy()
    for column, values_, keep in (
        ("rent", rent, rented),
        ("mortgage_interest", interest, mortgaged),
    ):
        zeroed = ~keep & (values_ != 0)
        assert record["reconciliation"][column] == {
            "column": "present",
            "contradicting_households": int(zeroed.sum()),
            "contradicting_amount": fsum(values_[zeroed]),
            "contradicting_weighted_amount": fsum((values_ * weights)[zeroed]),
            "total_amount": fsum(values_),
            "weighted_total_amount": fsum(values_ * weights),
            "zeroed": bool(zeroed.any()),
        }
    assert record["reconciliation"]["mortgage_principal"] == {"column": "absent"}
    assert sum(item["households"] for item in record["tenure"].values()) == len(
        households
    )
    person = tables["person"]
    by_person = persons.set_index("person_id")
    assert person["legal_marital_status"].tolist() == [
        A_MARITL[by_person.A_MARITL[i]] for i in person["person_id"]
    ]
    assert person["enrolled_full_time"].tolist() == [
        bool(by_person.A_ENRLW[i] == 1 and by_person.A_FTPT[i] == 1)
        for i in person["person_id"]
    ]
    by_year = record["full_time_study"]["read_false_by_source_year"]
    for year in (2022, 2023, 2024):
        in_year = (persons.source_year == year).to_numpy()
        assert by_year.get(str(year), 0) == int((blank & in_year).sum())
    staged = {"tenure", "rent", "mortgage_interest", *STAGED["person"]}
    assert [v for v in validate_concept_tables(tables) if v.column in staged] == []


@PROPERTY
@given(donor=_donors(), data=st.data())
def test_a_refusing_choice_refuses_with_its_counts(donor, data):
    """A blank answer under ``blank_full_time_study=refuse``, or a cost that
    contradicts the staged tenure under ``refuse``, refuses with its counts.
    A blank refuses first, because it refuses during staging."""
    households, units, persons, order = donor
    households, persons = households.copy(), persons.copy()
    refusing = data.draw(st.sampled_from([{"blank"}, {"cost"}, {"blank", "cost"}]))
    if "blank" in refusing:
        who = data.draw(st.integers(0, len(persons) - 1))
        column = data.draw(st.sampled_from(["A_ENRLW", "A_FTPT"]))
        persons.loc[who, column] = np.nan
    if "cost" in refusing:
        # Every tenure contradicts rent or mortgage interest, so this does.
        where = data.draw(st.integers(0, len(households) - 1))
        households.loc[where, ["rent", "mortgage_interest"]] = [1.0, 3.0]
    choices = {
        "assisted_renter_tenure": data.draw(
            st.sampled_from(OFFERED["assisted_renter_tenure"])
        ),
        "blank_full_time_study": "refuse" if "blank" in refusing else "read_false",
        "housing_cost_contradicting_tenure": (
            "refuse" if "cost" in refusing else "tenure_wins"
        ),
    }
    if "blank" in refusing:
        blank = (persons.A_ENRLW.isna() | persons.A_FTPT.isna()).to_numpy()
        by_year = {
            str(year): int((blank & (persons.source_year == year).to_numpy()).sum())
            for year in sorted(set(persons.source_year))
        }
        message = f"blank for persons by source year {by_year}, which"
    else:
        tenure_by_id = {
            row.household_id: _oracle_tenure(
                row, units, choices["assisted_renter_tenure"]
            )
            for row in households.itertuples()
        }
        counts = {
            column: sum(
                int(
                    bool(getattr(row, column) != 0)
                    and not _pays(tenure_by_id[row.household_id], column)
                )
                for row in households.itertuples()
            )
            for column in ("rent", "mortgage_interest")
        }
        contradicted = {column: n for column, n in counts.items() if n}
        message = f"contradict the staged tenure in households {contradicted}, which"
    with pytest.raises(ValueError, match=re.escape(message)):
        _stage(households, units, persons, _profile("f" * 64, **choices), order)


@PROPERTY
@given(donor=_donors(), data=st.data())
def test_a_missing_spm_join_refuses_and_never_falls_through(donor, data):
    """A household the join cannot reach would otherwise read as
    rented_private (no assistance) or owned_outright (no mortgage flag)."""
    households, units, persons, _ = donor
    profile = _profile("f" * 64)
    target = data.draw(st.sampled_from(households.household_id.tolist()))
    detached = persons.copy()
    detached.loc[detached.person_household_id == target, "person_spm_unit_id"] = 999
    with pytest.raises(ValueError, match="belong to no known SPM unit"):
        _stage(households, units, detached, profile)
    label = data.draw(st.sampled_from(["OWNED_WITH_MORTGAGE", "RENTED", "NONE"]))
    orphan = pd.DataFrame(
        [
            {
                "household_id": 999,
                "tenure_type": label,
                "H_TENURE": {"OWNED_WITH_MORTGAGE": 1, "RENTED": 2, "NONE": 3}[label],
                "rent": 0.0,
                "mortgage_interest": 0.0,
                "weight": 1.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="1 donor households have no SPM unit"):
        _stage(
            pd.concat([households, orphan], ignore_index=True), units, persons, profile
        )


@PROPERTY
@given(
    values=st.lists(
        st.one_of(
            st.integers(-2, 9).map(float),
            st.sampled_from([0.5, 1.5, np.nan, np.inf, -np.inf, 2.0000001]),
        ),
        min_size=1,
        max_size=8,
    ),
    missing_ok=st.booleans(),
)
def test_survey_codes_refuse_anything_outside_the_codebook(values, missing_ok):
    series = pd.Series(np.asarray(values, dtype=np.float64), name="A_TEST")
    numbers = series.to_numpy()
    blank = np.isnan(numbers)
    invalid = ~blank & ~np.isin(numbers, [1.0, 2.0, 3.0])
    if invalid.any() or (blank.any() and not missing_ok):
        with pytest.raises(ValueError, match="A_TEST"):
            transport_module._profile_codes(
                series, frozenset({1, 2, 3}), missing_ok=missing_ok
            )
        return
    codes, mask = transport_module._profile_codes(
        series, frozenset({1, 2, 3}), missing_ok=missing_ok
    )
    assert mask.tolist() == blank.tolist()
    assert codes.tolist() == [
        -1 if m else int(v) for v, m in zip(numbers, blank, strict=True)
    ]


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, 3])
def test_a_missing_or_non_text_label_refuses(missing):
    values = pd.Series(["RENTED", missing], name="tenure_type", dtype=object)
    with pytest.raises(ValueError, match="labels outside"):
        transport_module._profile_labels(values, {"RENTED"})


@pytest.mark.parametrize(
    "assistance",
    [
        pd.Series([False, None, True], dtype="boolean"),
        pd.Series([False, None, True], dtype=object),
        pd.Series([0.0, 0.0, 1.0]),
    ],
)
def test_assistance_must_be_boolean_with_no_nulls(assistance):
    tables = _raw_tables()
    tables["spm_unit"]["receives_housing_assistance"] = assistance
    with pytest.raises(ValueError, match="must be boolean with no nulls"):
        transport_module._stage_donor_profile(
            _profile("a" * 64),
            tables["person"],
            tables["household"],
            tables["spm_unit"],
        )


def test_the_shared_reader_names_no_country_and_no_donor_digest():
    """No country name, node id or donor digest literal in the shared module."""
    source = Path(transport_module.__file__).read_text()
    strings = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not [s for s in strings if re.search(r"[0-9a-f]{64}", s)]
    assert not [s for s in strings if re.search(r"(?i)zealand|belgi", s)]
    # A country-scoped node id such as "nz.create".
    assert not [s for s in strings if re.search(r"\b(nz|be|uk|us)\.[a-z_]", s)]
    for name, _ in DONOR_PROFILE_CHOICES:
        assert not re.search(r"(?i)\b(nz|be|uk)\b", name)
