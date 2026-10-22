"""The processed CPS ASEC builder on small synthetic Census archives.

``microcosm.build.us_runtime.asec_processed_source`` rebuilds the
``census_cps_YYYY.h5`` inputs from a pinned Census public-use archive. Its
differential check against the real pinned 2024 and 2025 files needs gated
data and lives in ``experiments/us-asec-processed-source-builder``. These
tests build synthetic archives shaped like Census's and pin:

* archive verification (bytes, person member, row count) before any read;
* the five tables' schema, column order, row selection and key order;
* the pandas 2.3 column types the loader wrote (identifiers as text);
* tax-unit invariants, as Hypothesis properties: every person in exactly one
  tax unit, units nested in households, dense ids, one head per unit,
  married pairs kept together, dependents meeting a dependency test, and a
  partition that is local to each household and independent of row order;
* agreement with the archived construction on golden cases;
* independence from ``PYTHONHASHSEED``;
* the store comparator, which must catch each class of difference;
* the command-line tool.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import asec_processed_source as aps
from microcosm.build.us_runtime import asec_sources
from microcosm.build.us_runtime.asec_processed_source import (
    ASEC_PROCESSED_TABLE_KEYS,
    OPTIONAL_PERSON_COLUMNS,
    PERSON_COLUMNS,
    SPM_BBSUBVAL_INCOME_YEARS,
    SPM_UNIT_COLUMNS,
    TAX_UNIT_COLUMNS,
    asec_archive_members,
    asec_processed_archive_pins,
    build_asec_processed_tables,
    compare_asec_processed_stores,
    construct_asec_tax_units,
    dependent_gross_income_limit_for_year,
    read_asec_processed_store,
    resolve_asec_archive,
    spm_unit_columns_for_year,
    verify_asec_archive,
    write_asec_processed_store,
)
from microcosm.build.us_runtime.asec_sources import ASEC_SOURCE_ARTIFACTS
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
    AsecEducationArchive,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_ROOT = _TEST_PATHS.repository
_GOLDEN_CASES = (
    _ROOT
    / "packages"
    / "microcosm-build"
    / "tests"
    / "fixtures"
    / "us_asec_tax_unit_golden_cases.csv"
)

_INCOME_COLUMNS = (
    "WSAL_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "INT_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "CAP_VAL",
    "UC_VAL",
    "OI_VAL",
    "ANN_VAL",
    "PNSN_VAL",
)
_DISABILITY_COLUMNS = (
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
)


# ---------------------------------------------------------------------------
# Synthetic Census archives
# ---------------------------------------------------------------------------


def _peridnum(household: int, line: int) -> str:
    # 22 digits with leading zeros: small enough to fit in 64 bits, so only
    # the text rule keeps the zeros.
    return f"{household:016d}{line:06d}"


def _cases() -> pd.DataFrame:
    """Four households: a married couple with three children and a
    grandparent, a single parent, a lone adult, and adult roommates with a
    dependent."""

    rows = [
        # household, line, age, marital, spouse, par1, par2, relationship,
        # wages, enrolled, full time
        (11, 1, 44, 1, 2, -1, -1, 1, 60_000, 0, 0),
        (11, 2, 42, 1, 1, -1, -1, 4, 30_000, 0, 0),
        (11, 3, 12, 7, 0, 1, 2, 5, 0, 1, 1),
        (11, 4, 20, 7, 0, 1, 2, 5, 3_000, 1, 1),
        (11, 5, 70, 4, 0, -1, -1, 8, 2_000, 0, 0),
        (11, 6, 21, 7, 0, 1, 2, 5, 3_000, 0, 0),
        (22, 1, 35, 5, 0, -1, -1, 1, 25_000, 0, 0),
        (22, 2, 8, 7, 0, 1, -1, 5, 0, 1, 1),
        (33, 1, 29, 7, 0, -1, -1, 2, 40_000, 0, 0),
        (44, 1, 51, 7, 0, -1, -1, 1, 45_000, 0, 0),
        (44, 2, 49, 7, 0, -1, -1, 13, 38_000, 0, 0),
        (44, 3, 16, 7, 0, 2, -1, 12, 0, 1, 1),
    ]
    columns = [
        "PH_SEQ",
        "A_LINENO",
        "A_AGE",
        "A_MARITL",
        "A_SPOUSE",
        "PEPAR1",
        "PEPAR2",
        "A_EXPRRP",
        "WSAL_VAL",
        "A_ENRLW",
        "A_FTPT",
    ]
    return pd.DataFrame(rows, columns=columns)


def _person_file(
    cases: pd.DataFrame,
    income_year: int,
    *,
    drop: tuple[str, ...] = (),
    extra: dict[str, object] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """A Census-shaped person file: every requested column, some unrequested
    ones, and a shuffled column order."""

    frame = pd.DataFrame(index=range(len(cases)))
    requested = dict.fromkeys(
        (*PERSON_COLUMNS, *spm_unit_columns_for_year(income_year), *TAX_UNIT_COLUMNS)
    )
    for column in requested:
        frame[column] = 0
    for column in cases.columns:
        frame[column] = cases[column].to_numpy()
    frame["P_SEQ"] = np.arange(1, len(cases) + 1)
    frame["PF_SEQ"] = 1 + (frame["A_AGE"] >= 65).astype(int)
    frame["PERIDNUM"] = [
        _peridnum(household, line)
        for household, line in zip(frame["PH_SEQ"], frame["A_LINENO"], strict=True)
    ]
    frame["TAX_ID"] = frame["PH_SEQ"] * 100 + frame["A_LINENO"].clip(upper=2)
    frame["SPM_ID"] = frame["PH_SEQ"] * 10 + 1
    for position, column in enumerate(spm_unit_columns_for_year(income_year)):
        if column != "SPM_ID":
            # Varies within an SPM unit, so the table must take the first row.
            frame[column] = frame["P_SEQ"] * 1_000 + position
    frame["SPM_EQUIVSCALE"] = frame["P_SEQ"] / 4
    frame["A_FNLWGT"] = frame["P_SEQ"] * 1_000 + 55
    frame["AGI"] = frame["WSAL_VAL"]
    frame["M5G_DSCP"] = 3  # published by Census, not kept
    frame["ZZ_UNREQUESTED"] = 9
    if income_year not in SPM_BBSUBVAL_INCOME_YEARS:
        frame["SPM_BBSUBVAL"] = 77  # present but not requested for this year
    for column, value in (extra or {}).items():
        frame[column] = value
    frame = frame.drop(columns=list(drop))
    order = np.random.default_rng(seed).permutation(frame.columns.to_numpy())
    return frame[list(order)]


def _family_file(person: pd.DataFrame) -> pd.DataFrame:
    keys = person[["PH_SEQ", "PF_SEQ"]].drop_duplicates()
    family = pd.DataFrame(
        {
            "FH_SEQ": keys["PH_SEQ"].to_numpy(),
            "FFPOS": keys["PF_SEQ"].to_numpy(),
            "FPERSONS": 1,
            "FTOT_R": np.arange(len(keys)) * 1_000,
        }
    )
    orphan = pd.DataFrame(
        {"FH_SEQ": [99], "FFPOS": [1], "FPERSONS": [0], "FTOT_R": [5]}
    )
    return pd.concat([family, orphan], ignore_index=True)


def _household_file(person: pd.DataFrame) -> pd.DataFrame:
    households = sorted(person["PH_SEQ"].unique())
    present = pd.DataFrame(
        {
            "H_IDNUM": [f"{household:020d}" for household in households],
            "H_SEQ": households,
            "GESTFIPS": 6,
            "H_NUMPER": person.groupby("PH_SEQ").size().loc[households].to_numpy(),
            "HWIDE": [f"{household:025d}" for household in households],
            "HSUP_WGT": [1_234.5] * len(households),
        }
    )
    # A non-interview household (no person rows) and an empty numeric field.
    absent = pd.DataFrame(
        {
            "H_IDNUM": ["98765432109876543210"],
            "H_SEQ": [99],
            "GESTFIPS": [6],
            "H_NUMPER": [0],
            "HWIDE": ["1234567890123456789012345"],
            "HSUP_WGT": [np.nan],
        }
    )
    return pd.concat([present, absent], ignore_index=True)


def _write_archive(
    path: Path,
    income_year: int,
    person: pd.DataFrame,
    *,
    family: pd.DataFrame | None = None,
    household: pd.DataFrame | None = None,
    omit: tuple[str, ...] = (),
) -> AsecEducationArchive:
    survey_year = income_year + 1
    members = asec_archive_members(survey_year)
    texts = {
        "person": person.to_csv(index=False),
        "family": (_family_file(person) if family is None else family).to_csv(
            index=False
        ),
        "household": (
            _household_file(person) if household is None else household
        ).to_csv(index=False),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in ("household", "family", "person"):
            if key not in omit:
                archive.writestr(members[key], texts[key])
        archive.writestr(f"asec_csv_repwgt_{survey_year}.csv", "PERIDNUM\n1\n")
    data = path.read_bytes()
    person_bytes = texts["person"].encode()
    with zipfile.ZipFile(path) as archive:
        crc = archive.getinfo(members["person"]).CRC if "person" not in omit else 0
    return AsecEducationArchive(
        survey_year=survey_year,
        income_year=income_year,
        zip_url=(
            "https://www2.census.gov/programs-surveys/cps/datasets/"
            f"{survey_year}/march/asecpub{survey_year % 100:02d}csv.zip"
        ),
        zip_size_bytes=len(data),
        zip_sha256=hashlib.sha256(data).hexdigest(),
        member=members["person"],
        member_size_bytes=len(person_bytes),
        member_crc32=f"{crc:08x}",
        member_sha256=hashlib.sha256(person_bytes).hexdigest(),
        rows=len(person),
        positive_rows=0,
        weighted_positive_share=0.0,
        weighted_total=0.0,
    )


@pytest.fixture
def archive_2024(tmp_path: Path) -> tuple[Path, AsecEducationArchive, pd.DataFrame]:
    person = _person_file(_cases(), 2024)
    path = tmp_path / "asecpub25csv.zip"
    return path, _write_archive(path, 2024, person), person


# ---------------------------------------------------------------------------
# Registry and year rules
# ---------------------------------------------------------------------------


def test_every_pinned_archive_year_is_buildable() -> None:
    """Each pinned archive names its survey year's person member, and
    microunit publishes a dependent income limit for its income year."""

    for income_year, pins in ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items():
        assert asec_processed_archive_pins(income_year) is pins
        assert pins.survey_year == income_year + 1
        assert pins.member == asec_archive_members(pins.survey_year)["person"]
        code = f"{pins.survey_year % 100:02d}"
        assert pins.zip_url.endswith(f"/{pins.survey_year}/march/asecpub{code}csv.zip")
        assert dependent_gross_income_limit_for_year(income_year) > 0


def test_every_pinned_processed_file_has_a_pinned_archive() -> None:
    """A pinned census_cps file can always be rebuilt from a pinned archive."""

    assert set(ASEC_SOURCE_ARTIFACTS) <= set(ASEC_EDUCATION_ASSISTANCE_ARCHIVES)


def test_unpinned_income_year_refuses() -> None:
    with pytest.raises(ValueError, match="No pinned Census ASEC archive"):
        asec_processed_archive_pins(1999)


def test_spm_bbsubval_is_requested_exactly_for_income_years_2021_to_2024() -> None:
    for year in range(1990, 2101):
        columns = spm_unit_columns_for_year(year)
        assert ("SPM_BBSUBVAL" in columns) == (2021 <= year <= 2024)
        assert [c for c in SPM_UNIT_COLUMNS if c != "SPM_BBSUBVAL"] == [
            c for c in columns if c != "SPM_BBSUBVAL"
        ]
        assert columns[-1] == "SPM_ID"


def test_archive_member_names_follow_the_survey_year() -> None:
    assert asec_archive_members(2026) == {
        "person": "pppub26.csv",
        "family": "ffpub26.csv",
        "household": "hhpub26.csv",
    }
    assert asec_archive_members(2100)["person"] == "pppub00.csv"


def test_dependent_income_limits_are_the_irs_exemption_amounts() -> None:
    # Rev. Proc. 2021-45, 2022-38, 2023-34, 2024-40 and 2025-32.
    assert {
        year: dependent_gross_income_limit_for_year(year)
        for year in (2022, 2023, 2024, 2025, 2026)
    } == {2022: 4_400.0, 2023: 4_700.0, 2024: 5_050.0, 2025: 5_200.0, 2026: 5_300.0}


def test_a_year_past_the_published_limits_refuses() -> None:
    with pytest.raises(ValueError, match="publishes no dependent gross income limit"):
        dependent_gross_income_limit_for_year(2100)


def test_column_rosters_have_no_duplicates_and_optional_columns_are_requested() -> None:
    for roster in (PERSON_COLUMNS, SPM_UNIT_COLUMNS, TAX_UNIT_COLUMNS):
        assert len(roster) == len(set(roster))
    assert set(OPTIONAL_PERSON_COLUMNS) <= set(PERSON_COLUMNS)
    assert len(SPM_UNIT_COLUMNS) == 39
    assert len(TAX_UNIT_COLUMNS) == 10


# ---------------------------------------------------------------------------
# Archive verification
# ---------------------------------------------------------------------------


def test_verify_accepts_the_pinned_archive(archive_2024) -> None:
    path, pins, _ = archive_2024
    assert verify_asec_archive(path, pins) == path


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("zip_size_bytes", 1, "byte length mismatch"),
        ("zip_sha256", "0" * 64, "SHA-256 mismatch"),
        ("member", "pppub99.csv", "exactly one"),
        ("member_size_bytes", 1, "person member byte length"),
        ("member_crc32", "00000000", "CRC-32"),
    ],
)
def test_verify_refuses_any_pin_mismatch(archive_2024, field, value, message) -> None:
    path, pins, _ = archive_2024
    with pytest.raises(ValueError, match=message):
        verify_asec_archive(path, replace(pins, **{field: value}))


def test_verify_refuses_a_missing_archive(tmp_path: Path, archive_2024) -> None:
    _, pins, _ = archive_2024
    with pytest.raises(FileNotFoundError):
        verify_asec_archive(tmp_path / "absent.zip", pins)


def test_build_verifies_before_reading(archive_2024, monkeypatch) -> None:
    path, pins, _ = archive_2024
    monkeypatch.setattr(
        aps, "_read_census_csv", lambda *a, **k: pytest.fail("read before verify")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_asec_processed_tables(path, 2024, pins=replace(pins, zip_sha256="f" * 64))


def test_build_refuses_a_row_count_that_differs_from_the_pin(archive_2024) -> None:
    path, pins, _ = archive_2024
    with pytest.raises(ValueError, match="the pin records"):
        build_asec_processed_tables(path, 2024, pins=replace(pins, rows=pins.rows + 1))


def test_build_refuses_a_pin_for_another_year(archive_2024) -> None:
    path, pins, _ = archive_2024
    with pytest.raises(ValueError, match="not 2025"):
        build_asec_processed_tables(path, 2025, pins=pins)


def test_build_refuses_an_archive_missing_a_member(tmp_path: Path) -> None:
    person = _person_file(_cases(), 2024)
    path = tmp_path / "asecpub25csv.zip"
    pins = _write_archive(path, 2024, person, omit=("family",))
    with pytest.raises(ValueError, match="lacks members: ffpub25.csv"):
        build_asec_processed_tables(path, 2024, pins=pins)


def test_resolve_uses_the_education_cache(archive_2024, monkeypatch) -> None:
    path, pins, _ = archive_2024
    monkeypatch.setitem(ASEC_EDUCATION_ASSISTANCE_ARCHIVES, 2024, pins)
    calls = []

    def fake_fetch(income_year, cache_dir=None):
        calls.append((income_year, cache_dir))
        return path.parent / pins.member

    monkeypatch.setattr(aps, "fetch_asec_education_assistance_source", fake_fetch)
    assert resolve_asec_archive(2024, cache_dir="cache") == path
    assert calls == [(2024, "cache")]
    assert resolve_asec_archive(2024, path) == path


# ---------------------------------------------------------------------------
# The five tables
# ---------------------------------------------------------------------------


def test_tables_follow_the_loader_schema(archive_2024) -> None:
    path, pins, person_file = archive_2024
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    assert [key for key, _ in tables.items()] == list(ASEC_PROCESSED_TABLE_KEYS)

    person = tables.person
    requested = set(PERSON_COLUMNS) | set(SPM_UNIT_COLUMNS) | set(TAX_UNIT_COLUMNS)
    in_file_order = [column for column in person_file.columns if column in requested]
    assert list(person.columns) == [*in_file_order, "CENSUS_TAX_ID"]
    assert "M5G_DSCP" not in person and "ZZ_UNREQUESTED" not in person
    assert len(person) == len(person_file)
    assert person["CENSUS_TAX_ID"].tolist() == person_file["TAX_ID"].tolist()
    assert person["PERIDNUM"].tolist() == person_file["PERIDNUM"].tolist()
    assert tables.zero_filled_columns == ()

    # Only families and households with a person row, all their columns.
    assert tables.family.columns.tolist() == ["FH_SEQ", "FFPOS", "FPERSONS", "FTOT_R"]
    assert 99 not in set(tables.family["FH_SEQ"])
    assert set(
        zip(tables.family["FH_SEQ"], tables.family["FFPOS"], strict=True)
    ) == set(zip(person["PH_SEQ"], person["PF_SEQ"], strict=True))
    assert tables.household["H_SEQ"].tolist() == [11, 22, 33, 44]
    assert tables.tax_unit_input_audit == {
        "married_spouse_present_with_pointer": 2,
        "pairs": 1,
        "spouse_line_absent": 0,
        "self_spouse_pointer": 0,
        "spouse_not_married_spouse_present": 0,
        "one_sided_spouse_pointer": 0,
        "parent_line_absent": 0,
        "persons_in_several_pairs": 0,
        "self_parent_pointer": 0,
        "duplicate_person_lines": 0,
        "refused_households": [],
    }
    assert tables.household.index.tolist() == [0, 1, 2, 3]

    # The SPM table is the first member row of each unit, SPM_ID kept.
    spm = tables.spm_unit
    assert spm.index.name == "SPM_ID"
    assert spm.index.tolist() == [111, 221, 331, 441]
    assert list(spm.columns) == list(SPM_UNIT_COLUMNS)
    first_rows = person.groupby("SPM_ID", sort=True).head(1).set_index("SPM_ID")
    pd.testing.assert_frame_equal(
        spm, first_rows[list(SPM_UNIT_COLUMNS[:-1])].assign(SPM_ID=spm.index.to_numpy())
    )

    # Tax units: ids 1..K, ascending, one row each.
    assert tables.tax_unit.columns.tolist() == ["TAX_ID"]
    assert tables.tax_unit["TAX_ID"].tolist() == list(
        range(1, person["TAX_ID"].max() + 1)
    )
    assert tables.tax_unit.index.equals(pd.RangeIndex(len(tables.tax_unit)))


def test_constructed_tax_units_for_the_worked_households(archive_2024) -> None:
    path, pins, _ = archive_2024
    person = build_asec_processed_tables(path, 2024, pins=pins).person
    units = {
        (household, line): tax_id
        for household, line, tax_id in zip(
            person["PH_SEQ"], person["A_LINENO"], person["TAX_ID"], strict=True
        )
    }
    # The couple claims the 12-year-old and the 20-year-old full-time
    # student (qualifying children). Adults are claimed only as qualifying
    # children, so the 21-year-old non-student and the grandparent file
    # alone, although each earns under the $5,050 limit.
    assert len({units[(11, line)] for line in (1, 2, 3, 4)}) == 1
    assert len({units[(11, line)] for line in (1, 5, 6)}) == 3
    assert units[(22, 1)] == units[(22, 2)]
    assert units[(33, 1)] not in {units[key] for key in units if key[0] != 33}
    # Roommates file separately; the child joins its parent's unit.
    assert units[(44, 1)] != units[(44, 2)]
    assert units[(44, 3)] == units[(44, 2)]


def test_a_2024_person_file_without_spm_bbsubval_refuses(tmp_path: Path) -> None:
    person = _person_file(_cases(), 2024, drop=("SPM_BBSUBVAL", "PEIOOCC"))
    path = tmp_path / "asecpub25csv.zip"
    pins = _write_archive(path, 2024, person)
    with pytest.raises(KeyError, match="PEIOOCC, SPM_BBSUBVAL"):
        build_asec_processed_tables(path, 2024, pins=pins)


def test_income_year_2025_omits_spm_bbsubval_even_when_published(tmp_path) -> None:
    person = _person_file(_cases(), 2025)
    assert "SPM_BBSUBVAL" in person
    path = tmp_path / "asecpub26csv.zip"
    tables = build_asec_processed_tables(
        path, 2025, pins=_write_archive(path, 2025, person)
    )
    assert "SPM_BBSUBVAL" not in tables.person
    assert "SPM_BBSUBVAL" not in tables.spm_unit
    assert tables.spm_unit.shape[1] == 38


def test_missing_optional_columns_are_zero_filled_in_declared_order(tmp_path) -> None:
    dropped = ("PTOTVAL", "NOW_MIL", "NOW_COV")
    person = _person_file(_cases(), 2024, drop=dropped)
    path = tmp_path / "asecpub25csv.zip"
    tables = build_asec_processed_tables(
        path, 2024, pins=_write_archive(path, 2024, person)
    )
    assert tables.zero_filled_columns == ("NOW_COV", "NOW_MIL", "PTOTVAL")
    assert list(tables.person.columns[-4:]) == [
        "NOW_COV",
        "NOW_MIL",
        "PTOTVAL",
        "CENSUS_TAX_ID",
    ]
    for column in dropped:
        assert tables.person[column].dtype == np.int64
        assert (tables.person[column] == 0).all()


def test_identifiers_keep_their_text_and_numbers_keep_their_types(archive_2024) -> None:
    path, pins, person_file = archive_2024
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    peridnum = tables.person["PERIDNUM"]
    assert peridnum.dtype == object
    assert all(isinstance(value, str) and len(value) == 22 for value in peridnum)
    assert peridnum.iloc[0].startswith("0000000000000011")
    household = tables.household
    assert household["H_IDNUM"].dtype == object
    assert household["H_IDNUM"].iloc[0] == "00000000000000000011"
    # A column too wide for 64 bits keeps its text too, as under pandas 2.3.
    assert household["HWIDE"].dtype == object
    assert household["HWIDE"].iloc[0] == f"{11:025d}"
    assert household["HSUP_WGT"].dtype == np.float64
    numeric = tables.person.drop(columns=["PERIDNUM"])
    assert {str(dtype) for dtype in numeric.dtypes} <= {"int64", "float64"}


def test_empty_fields_are_filled_with_zero(tmp_path: Path) -> None:
    person = _person_file(_cases(), 2024)
    household = _household_file(person)
    household.loc[household["H_SEQ"] == 22, "HSUP_WGT"] = np.nan
    path = tmp_path / "asecpub25csv.zip"
    pins = _write_archive(path, 2024, person, household=household)
    built = build_asec_processed_tables(path, 2024, pins=pins).household
    assert built.set_index("H_SEQ").loc[22, "HSUP_WGT"] == 0.0


# ---------------------------------------------------------------------------
# Tax-unit invariants (Hypothesis)
# ---------------------------------------------------------------------------

_INCOMES = st.sampled_from(
    [0, 0, 0, 1, 4_699, 4_700, 5_049, 5_050, 5_199, 5_200, 12_000, 90_000, -2_500]
)
_RELATIONSHIPS = st.sampled_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 99])
_FLAGS = st.sampled_from([0, 1, 2])


def _married_spouse_present(person: dict[str, int]) -> bool:
    return person["A_MARITL"] in (1, 2) and person["A_SPOUSE"] > 0


def _repair(people: list[dict[str, int]]) -> None:
    """Remove the structures the construction cannot take: self parent
    pointers, and one-sided spouse pointers that put someone in two married
    pairs."""

    for person in people:
        for column in ("PEPAR1", "PEPAR2"):
            if person[column] == person["A_LINENO"]:
                person[column] = -1
    by_line = {person["A_LINENO"]: person for person in people}
    while True:
        counts: dict[int, int] = {}
        pairs = set()
        for person in people:
            spouse = by_line.get(person["A_SPOUSE"])
            if (
                _married_spouse_present(person)
                and spouse is not None
                and spouse is not person
                and _married_spouse_present(spouse)
            ):
                pairs.add(tuple(sorted((person["A_LINENO"], spouse["A_LINENO"]))))
        for pair in pairs:
            for line in pair:
                counts[line] = counts.get(line, 0) + 1
        over = {line for line, count in counts.items() if count > 1}
        if not over:
            return
        offender = next(
            person
            for person in people
            if _married_spouse_present(person)
            and person["A_SPOUSE"] in over
            and by_line[person["A_SPOUSE"]]["A_SPOUSE"] != person["A_LINENO"]
        )
        offender["A_SPOUSE"] = 0


@st.composite
def _households(draw, *, repair: bool = True) -> pd.DataFrame:
    rows = []
    for household in range(draw(st.integers(1, 5))):
        size = draw(st.integers(1, 7))
        lines = draw(
            st.lists(st.integers(1, 16), min_size=size, max_size=size, unique=True)
        )
        pointers = st.sampled_from([-1, 0, 17, *lines])
        people = []
        for line in lines:
            person = {
                "PH_SEQ": 3 * household + 5,
                "A_LINENO": line,
                "A_AGE": draw(st.integers(0, 90)),
                "A_MARITL": draw(st.integers(1, 7)),
                "A_SPOUSE": draw(pointers),
                "PEPAR1": draw(pointers),
                "PEPAR2": draw(pointers),
                "A_EXPRRP": draw(_RELATIONSHIPS),
                "SS_VAL": draw(_INCOMES),
                "PTOTVAL": draw(st.integers(-5_000, 100_000)),
                "A_ENRLW": draw(_FLAGS),
                "A_FTPT": draw(_FLAGS),
                "A_HSCOL": draw(_FLAGS),
            }
            for column in _INCOME_COLUMNS:
                person[column] = draw(_INCOMES) if draw(st.booleans()) else 0
            for column in _DISABILITY_COLUMNS:
                person[column] = draw(st.sampled_from([0, 2, 2, 1]))
            people.append(person)
        # Random pointers rarely pair two people, so pair some directly.
        for _ in range(draw(st.integers(0, size // 2))):
            first, second = draw(
                st.lists(
                    st.sampled_from(range(size)), min_size=2, max_size=2, unique=True
                )
            )
            people[first]["A_SPOUSE"] = people[second]["A_LINENO"]
            people[second]["A_SPOUSE"] = people[first]["A_LINENO"]
            people[first]["A_MARITL"] = people[second]["A_MARITL"] = draw(
                st.sampled_from([1, 2])
            )
        if repair:
            _repair(people)
        rows.extend(people)
    return pd.DataFrame(rows).astype("int64")


def _partition(frame: pd.DataFrame, tax_ids) -> frozenset[frozenset[tuple[int, int]]]:
    units: dict[int, set[tuple[int, int]]] = {}
    for household, line, tax_id in zip(
        frame["PH_SEQ"], frame["A_LINENO"], tax_ids, strict=True
    ):
        units.setdefault(int(tax_id), set()).add((int(household), int(line)))
    return frozenset(frozenset(members) for members in units.values())


def _married_pairs(frame: pd.DataFrame) -> set[tuple[int, int, int]]:
    """(household, first line, second line) of each married pair: a married
    spouse-present person naming a present, married spouse-present member.
    The pointer need not be returned."""

    by_key = {(row.PH_SEQ, row.A_LINENO): row for row in frame.itertuples(index=False)}

    def married(row) -> bool:
        return row.A_MARITL in (1, 2) and row.A_SPOUSE > 0

    pairs = set()
    for (household, line), row in by_key.items():
        spouse = by_key.get((household, row.A_SPOUSE))
        if (
            married(row)
            and spouse is not None
            and spouse is not row
            and married(spouse)
        ):
            pairs.add((household, min(line, row.A_SPOUSE), max(line, row.A_SPOUSE)))
    return pairs


_PROPERTY_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_PROPERTY_SETTINGS
@given(frame=_households(), year=st.sampled_from([2022, 2023, 2024, 2025]))
def test_tax_units_partition_people_within_households(frame, year) -> None:
    tax_ids, tax_unit, assignments = construct_asec_tax_units(frame.copy(), year)

    # Every person is in exactly one tax unit.
    assert len(tax_ids) == len(frame)
    assert tax_ids.dtype == np.int64
    assert (tax_ids >= 1).all()
    # Ids are dense, numbered by first appearance.
    assert pd.unique(tax_ids).tolist() == list(range(1, tax_ids.max() + 1))
    # The table holds each unit once, ascending.
    assert tax_unit["TAX_ID"].tolist() == list(range(1, tax_ids.max() + 1))
    # Units nest in households.
    assert (pd.Series(frame["PH_SEQ"].to_numpy()).groupby(tax_ids).nunique() == 1).all()

    roles = pd.Series([value.decode() for value in assignments["tax_unit_role_input"]])
    assert set(roles) <= {"HEAD", "SPOUSE", "DEPENDENT"}
    counts = pd.crosstab(tax_ids, roles)
    # One head per unit, at most one spouse.
    assert "HEAD" in counts and (counts["HEAD"] == 1).all()
    if "SPOUSE" in counts:
        assert (counts["SPOUSE"] <= 1).all()


@_PROPERTY_SETTINGS
@given(frame=_households(), year=st.sampled_from([2024, 2025]))
def test_married_pairs_file_together_and_dependents_pass_a_test(frame, year) -> None:
    tax_ids, _, assignments = construct_asec_tax_units(frame.copy(), year)
    roles = [value.decode() for value in assignments["tax_unit_role_input"]]
    position = {
        (household, line): index
        for index, (household, line) in enumerate(
            zip(frame["PH_SEQ"], frame["A_LINENO"], strict=True)
        )
    }
    paired = set()
    for household, first, second in _married_pairs(frame):
        a, b = position[(household, first)], position[(household, second)]
        assert tax_ids[a] == tax_ids[b]
        assert {roles[a], roles[b]} == {"HEAD", "SPOUSE"}
        paired.update({a, b})
    spouses = {index for index, role in enumerate(roles) if role == "SPOUSE"}
    assert spouses <= paired

    limit = dependent_gross_income_limit_for_year(year)
    gross = frame[list(_INCOME_COLUMNS)].clip(lower=0).sum(axis=1).to_numpy()
    student = (frame["A_ENRLW"] == 1) & (
        (frame["A_FTPT"] == 1) | frame["A_HSCOL"].isin([1, 2])
    )
    disabled = (frame[list(_DISABILITY_COLUMNS)] == 1).any(axis=1)
    age_limit = np.where(student, 24, 19)
    qualifying_child_age = disabled | (frame["A_AGE"] < age_limit)
    for index, role in enumerate(roles):
        if role != "DEPENDENT":
            continue
        assert index not in paired
        # A claim passes the qualifying-child age test or the
        # qualifying-relative gross income test.
        assert qualifying_child_age.iloc[index] or gross[index] < limit


@_PROPERTY_SETTINGS
@given(frame=_households(), data=st.data())
def test_partition_is_local_and_independent_of_row_order(frame, data) -> None:
    tax_ids, _, _ = construct_asec_tax_units(frame.copy(), 2024)
    expected = _partition(frame, tax_ids)

    # Each household's units depend on that household's rows only.
    local = set()
    for _, household in frame.groupby("PH_SEQ", sort=False):
        household = household.reset_index(drop=True)
        household_ids, _, _ = construct_asec_tax_units(household.copy(), 2024)
        local |= _partition(household, household_ids)
    assert frozenset(local) == expected

    # Shuffling rows relabels units but keeps them.
    order = data.draw(st.permutations(range(len(frame))))
    shuffled = frame.iloc[list(order)].reset_index(drop=True)
    shuffled_ids, _, _ = construct_asec_tax_units(shuffled.copy(), 2024)
    assert _partition(shuffled, shuffled_ids) == expected


@_PROPERTY_SETTINGS
@given(frame=_households(repair=False))
def test_construction_refuses_exactly_the_structures_it_would_mishandle(frame) -> None:
    audit = aps.audit_tax_unit_inputs(frame)
    if audit["persons_in_several_pairs"] or audit["self_parent_pointer"]:
        with pytest.raises(ValueError, match="cannot be constructed"):
            construct_asec_tax_units(frame.copy(), 2024)
    else:
        tax_ids, _, _ = construct_asec_tax_units(frame.copy(), 2024)
        assert len(tax_ids) == len(frame)


def test_a_spouse_chain_is_refused_not_silently_split() -> None:
    """Line 1 names 2, 2 names 3 and 3 names 2: line 2 would sit in two
    pairs. The archived construction raises StopIteration here."""

    frame = _cases().iloc[:3].copy()
    frame["A_MARITL"] = 1
    frame["A_SPOUSE"] = [2, 3, 2]
    audit = aps.audit_tax_unit_inputs(frame)
    assert audit["pairs"] == 2
    assert audit["one_sided_spouse_pointer"] == 1
    assert audit["persons_in_several_pairs"] == 1
    assert audit["refused_households"] == [11]
    with pytest.raises(ValueError, match=r"1 person\(s\) in several married pairs"):
        construct_asec_tax_units(frame, 2024)


def test_a_self_parent_pointer_is_refused() -> None:
    """An adult naming their own line as a parent who passes the
    qualifying-child age test (here, disabled) would claim their own unit."""

    frame = _cases()
    grandparent = frame.index[4]
    frame.loc[grandparent, "PEPAR1"] = 5
    frame["PEDISPHY"] = 0
    frame.loc[grandparent, "PEDISPHY"] = 1
    audit = aps.audit_tax_unit_inputs(frame)
    assert audit["self_parent_pointer"] == 1
    assert audit["refused_households"] == [11]
    with pytest.raises(ValueError, match=r"1 self parent pointer\(s\)"):
        construct_asec_tax_units(frame, 2024)


def test_a_repeated_line_number_is_refused() -> None:
    frame = _cases()
    frame.loc[frame.index[1], "A_LINENO"] = 1
    with pytest.raises(ValueError, match=r"2 row\(s\) repeating"):
        construct_asec_tax_units(frame, 2024)


def test_benign_spouse_pointer_anomalies_are_counted_not_refused() -> None:
    frame = _cases()
    rows = frame.index
    frame.loc[rows[5], ["A_MARITL", "A_SPOUSE"]] = [1, 9]  # absent line
    frame.loc[rows[6], ["A_MARITL", "A_SPOUSE"]] = [2, 1]  # self (line 1)
    frame.loc[rows[8], ["A_MARITL", "A_SPOUSE"]] = [1, 1]  # self in 33
    # 44/1 names 2, who names 3, who is not married: one pair, (1, 2), with
    # a one-sided pointer.
    frame.loc[rows[9], ["A_MARITL", "A_SPOUSE"]] = [1, 2]
    frame.loc[rows[10], ["A_MARITL", "A_SPOUSE"]] = [1, 3]
    frame.loc[rows[11], "PEPAR2"] = 9  # absent parent line
    audit = aps.audit_tax_unit_inputs(frame)
    assert audit["spouse_line_absent"] == 1
    assert audit["self_spouse_pointer"] == 2
    assert audit["spouse_not_married_spouse_present"] == 1
    assert audit["one_sided_spouse_pointer"] == 1
    assert audit["parent_line_absent"] == 1
    assert audit["pairs"] == 2
    assert audit["persons_in_several_pairs"] == 0
    tax_ids, _, _ = construct_asec_tax_units(frame, 2024)
    assert len(tax_ids) == len(frame)


def test_construction_does_not_modify_the_person_frame() -> None:
    frame = _cases()
    for column in ("SS_VAL", "PTOTVAL", *_DISABILITY_COLUMNS, "A_ENRLW", "A_FTPT"):
        frame[column] = 0
    before = frame.copy()
    construct_asec_tax_units(frame, 2024)
    pd.testing.assert_frame_equal(frame, before)


def test_golden_cases_match_the_archived_construction() -> None:
    """Households whose tax units the archived construction (loader commit
    40314a75) assigned, in its own environment: every household of the
    20,000-household differential whose units change with the income limit
    or that carries a tolerated spouse-pointer anomaly, then the
    lowest-numbered households up to 400. See
    experiments/us-asec-processed-source-builder."""

    golden = pd.read_csv(_GOLDEN_CASES)
    expected_columns = [c for c in golden.columns if c.startswith("EXPECTED_")]
    cases = golden.drop(columns=expected_columns)
    years = sorted({int(c.rsplit("_", 1)[1]) for c in expected_columns})
    assert years == [2022, 2023, 2024, 2025]
    assert golden["PH_SEQ"].nunique() == 400
    for year in years:
        tax_ids, _, assignments = construct_asec_tax_units(cases.copy(), year)
        roles = [value.decode() for value in assignments["tax_unit_role_input"]]
        assert tax_ids.tolist() == golden[f"EXPECTED_TAX_ID_{year}"].tolist()
        assert roles == golden[f"EXPECTED_ROLE_{year}"].tolist()
    # The cases reach the income test: some households' units differ between
    # years, whose only difference is the dependent income limit.
    limit_sensitive = sum(
        _partition(rows, rows["EXPECTED_TAX_ID_2024"])
        != _partition(rows, rows["EXPECTED_TAX_ID_2025"])
        for _, rows in golden.groupby("PH_SEQ")
    )
    assert limit_sensitive > 0


# ---------------------------------------------------------------------------
# Writing, reading and comparing stores
# ---------------------------------------------------------------------------


def test_store_round_trips_every_table(archive_2024, tmp_path: Path) -> None:
    path, pins, _ = archive_2024
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    store = write_asec_processed_store(tables, tmp_path / "census_cps_2024.h5")
    read = read_asec_processed_store(store)
    assert sorted(read) == sorted(ASEC_PROCESSED_TABLE_KEYS)
    for key, table in tables.items():
        # pandas reads a stored text column in its own text dtype.
        expected = table.astype(
            {
                column: read[key][column].dtype
                for column in table
                if table[column].dtype == object
            }
        )
        pd.testing.assert_frame_equal(
            read[key], expected, check_exact=True, check_index_type=False
        )
        assert read[key].index.dtype == np.int64
    assert read["person"]["PERIDNUM"].map(type).eq(str).all()

    report = compare_asec_processed_stores(store, store)
    assert report["equal"]
    assert all(t["hdf5"]["block_layout_equal"] for t in report["tables"].values())

    import tables as pytables

    with pytables.open_file(store) as handle:
        assert sorted(node._v_name for node in handle.root) == sorted(
            ASEC_PROCESSED_TABLE_KEYS
        )
        text_blocks = [
            node
            for node in handle.walk_nodes("/person")
            if isinstance(node, pytables.VLArray)
        ]
        assert len(text_blocks) == 1  # PERIDNUM, a pickled object block as before


def test_write_refuses_to_overwrite_without_permission(archive_2024, tmp_path) -> None:
    path, pins, _ = archive_2024
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    output = tmp_path / "out.h5"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_asec_processed_store(tables, output)
    assert output.read_bytes() == b"existing"
    write_asec_processed_store(tables, output, overwrite=True)
    assert read_asec_processed_store(output)["tax_unit"].shape == tables.tax_unit.shape
    assert not (tmp_path / "out.h5.part").exists()


def test_write_never_replaces_a_pinned_inputs_bytes(
    archive_2024, tmp_path, monkeypatch
) -> None:
    path, pins, _ = archive_2024
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    pinned = tmp_path / "census_cps_2024.h5"
    pinned.write_bytes(b"pinned bytes")
    artifact = asec_sources.AsecSourceArtifact(
        income_year=2024,
        filename=pinned.name,
        sha256=hashlib.sha256(b"pinned bytes").hexdigest(),
        size_bytes=len(b"pinned bytes"),
    )
    monkeypatch.setattr(asec_sources, "ASEC_SOURCE_ARTIFACTS", {2024: artifact})
    with pytest.raises(FileExistsError, match="pinned bytes"):
        write_asec_processed_store(tables, pinned, overwrite=True)
    assert pinned.read_bytes() == b"pinned bytes"


def _store_pair(tmp_path: Path, archive_2024) -> tuple[dict[str, pd.DataFrame], Path]:
    path, pins, _ = archive_2024
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    reference = write_asec_processed_store(tables, tmp_path / "reference.h5")
    return read_asec_processed_store(reference), reference


def _write_tables(path: Path, tables: dict[str, pd.DataFrame]) -> Path:
    return write_asec_processed_store(tables, path, overwrite=True)


def _mutations():
    def value(tables):
        tables["person"].loc[0, "WSAL_VAL"] += 1

    def order(tables):
        columns = list(tables["household"].columns)
        columns[0], columns[1] = columns[1], columns[0]
        tables["household"] = tables["household"][columns]

    def dtype(tables):
        tables["family"]["FPERSONS"] = tables["family"]["FPERSONS"].astype("int32")

    def identifier_type(tables):
        tables["person"]["PERIDNUM"] = tables["person"]["PERIDNUM"].map(int)

    def negative_zero(tables):
        # Equal as numbers, so pandas agrees; the stored bits differ.
        tables["person"]["SPM_EQUIVSCALE"] = tables["person"]["SPM_EQUIVSCALE"] * 0.0
        tables["person"].loc[0, "SPM_EQUIVSCALE"] = -0.0

    def index(tables):
        tables["household"].index = tables["household"].index + 1

    def missing_table(tables):
        del tables["spm_unit"]

    def extra_column(tables):
        tables["tax_unit"]["filing_status"] = 1

    return {
        "value": (value, "person"),
        "order": (order, "household"),
        "dtype": (dtype, "family"),
        "identifier_type": (identifier_type, "person"),
        "negative_zero": (negative_zero, "person"),
        "index": (index, "household"),
        "missing_table": (missing_table, None),
        "extra_column": (extra_column, "tax_unit"),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_comparator_catches_each_class_of_difference(
    name, tmp_path: Path, archive_2024
) -> None:
    tables, reference = _store_pair(tmp_path, archive_2024)
    mutate, key = _mutations()[name]
    if name == "negative_zero":
        # Make the reference hold +0.0 in the same column first.
        tables["person"]["SPM_EQUIVSCALE"] = tables["person"]["SPM_EQUIVSCALE"] * 0.0
        reference = _write_tables(tmp_path / "zeros.h5", tables)
        tables = read_asec_processed_store(reference)
    mutate(tables)
    candidate = _write_tables(tmp_path / "candidate.h5", tables)
    report = compare_asec_processed_stores(candidate, reference)
    assert not report["equal"]
    if key is None:
        assert report["keys_only_reference"] == ["spm_unit"]
        return
    table = report["tables"][key]
    assert not table["equal"]
    if name == "negative_zero":
        assert table["frame_equal_exact"]  # pandas cannot see it
        assert table["hdf5"]["mismatched_columns"] == ["SPM_EQUIVSCALE"]
    else:
        assert not table["frame_equal_exact"]
        assert not table["hdf5"]["equal"]
    others = [other for other in report["tables"] if other != key]
    assert all(report["tables"][other]["equal"] for other in others)


def test_comparator_ignores_only_how_columns_are_grouped_into_blocks(
    tmp_path: Path, archive_2024
) -> None:
    import shutil

    import tables as pytables

    _, reference = _store_pair(tmp_path, archive_2024)
    candidate = tmp_path / "candidate.h5"
    shutil.copyfile(reference, candidate)
    # Renumber the household frame's first two blocks: the same columns,
    # labels and values, grouped in another order.
    with pytables.open_file(candidate, mode="a") as handle:
        group = handle.get_node("/household")
        assert int(group._v_attrs.nblocks) >= 2
        for part in ("items", "values"):
            handle.rename_node(group, f"swap_{part}", name=f"block0_{part}")
            handle.rename_node(group, f"block0_{part}", name=f"block1_{part}")
            handle.rename_node(group, f"block1_{part}", name=f"swap_{part}")
    for key, table in read_asec_processed_store(reference).items():
        pd.testing.assert_frame_equal(read_asec_processed_store(candidate)[key], table)
    report = compare_asec_processed_stores(candidate, reference)
    storage = report["tables"]["household"]["hdf5"]
    assert report["equal"]
    assert not storage["block_layout_equal"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

_HASH_SEED_RUNNER = """
import dataclasses, json, sys
from microcosm.build.us_runtime.asec_processed_source import (
    build_asec_processed_tables, write_asec_processed_store,
)
from microcosm.build.us_runtime.education_assistance_source import AsecEducationArchive
pins = AsecEducationArchive(**json.loads(sys.argv[2]))
tables = build_asec_processed_tables(sys.argv[1], pins.income_year, pins=pins)
write_asec_processed_store(tables, sys.argv[3])
"""


def test_output_does_not_depend_on_the_hash_seed(tmp_path: Path) -> None:
    """Set iteration order changes with PYTHONHASHSEED; the tables must not."""

    person = _person_file(_cases(), 2024, drop=("NOW_DIR", "NOW_PUB", "PTOTVAL"))
    path = tmp_path / "asecpub25csv.zip"
    pins = _write_archive(path, 2024, person)
    stores = []
    for seed in ("0", "4242"):
        output = tmp_path / f"seed_{seed}.h5"
        subprocess.run(
            [
                sys.executable,
                "-c",
                _HASH_SEED_RUNNER,
                str(path),
                json.dumps(pins.__dict__),
                str(output),
            ],
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        stores.append(output)
    report = compare_asec_processed_stores(*stores)
    assert report["equal"], report
    assert all(t["hdf5"]["block_layout_equal"] for t in report["tables"].values())


# ---------------------------------------------------------------------------
# The command-line tool
# ---------------------------------------------------------------------------


def _load_tool():
    path = _ROOT / "tools" / "build_us_asec_processed_source.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_asec_processed_source", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tool_builds_compares_and_writes_a_receipt(
    archive_2024, tmp_path: Path, monkeypatch, capsys
) -> None:
    path, pins, _ = archive_2024
    monkeypatch.setitem(ASEC_EDUCATION_ASSISTANCE_ARCHIVES, 2024, pins)
    tool = _load_tool()
    reference = tmp_path / "reference.h5"
    assert (
        tool.main(
            [
                "--income-year=2024",
                f"--archive={path}",
                f"--output={reference}",
            ]
        )
        == 0
    )
    capsys.readouterr()

    output = tmp_path / "rebuilt.h5"
    receipt_path = tmp_path / "receipt.json"
    assert (
        tool.main(
            [
                "--income-year=2024",
                f"--archive={path}",
                f"--output={output}",
                f"--compare-to={reference}",
                f"--receipt={receipt_path}",
            ]
        )
        == 0
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == tool.RECEIPT_SCHEMA
    assert receipt["income_year"] == 2024 and receipt["survey_year"] == 2025
    assert receipt["archive"]["sha256"] == pins.zip_sha256
    assert receipt["rows"]["person"] == len(_cases())
    assert (
        receipt["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert receipt["tax_unit_construction"]["mode"] == "policyengine"
    assert receipt["tax_unit_construction"]["dependent_gross_income_limit"] == 5_050.0
    assert receipt["comparison"]["equal"] is True
    assert set(receipt["comparison"]["tables"]) == set(ASEC_PROCESSED_TABLE_KEYS)
    assert receipt["environment"]["builder_sha256"]
    assert json.loads(capsys.readouterr().out) == receipt


def test_tool_exits_nonzero_when_the_reference_differs(
    archive_2024, tmp_path: Path, monkeypatch, capsys
) -> None:
    path, pins, person = archive_2024
    monkeypatch.setitem(ASEC_EDUCATION_ASSISTANCE_ARCHIVES, 2024, pins)
    tool = _load_tool()
    tables = build_asec_processed_tables(path, 2024, pins=pins)
    reference = write_asec_processed_store(tables, tmp_path / "reference.h5")
    frames = read_asec_processed_store(reference)
    frames["person"].loc[0, "A_AGE"] += 1
    reference = _write_tables(tmp_path / "reference_changed.h5", frames)
    receipt_path = tmp_path / "receipt.json"
    status = tool.main(
        [
            "--income-year=2024",
            f"--archive={path}",
            f"--output={tmp_path / 'rebuilt.h5'}",
            f"--compare-to={reference}",
            f"--receipt={receipt_path}",
        ]
    )
    assert status == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["comparison"]["equal"] is False
    assert receipt["comparison"]["tables"]["person"]["value_mismatches"] == ["A_AGE"]
    assert "differs from" in capsys.readouterr().err


def test_tool_refuses_an_existing_output_and_an_unpinned_year(
    archive_2024, tmp_path: Path, monkeypatch
) -> None:
    path, pins, _ = archive_2024
    monkeypatch.setitem(ASEC_EDUCATION_ASSISTANCE_ARCHIVES, 2024, pins)
    tool = _load_tool()
    output = tmp_path / "exists.h5"
    output.write_bytes(b"x")
    with pytest.raises(SystemExit, match="overwrite"):
        tool.main(["--income-year=2024", f"--archive={path}", f"--output={output}"])
    with pytest.raises(SystemExit, match="No pinned Census ASEC archive"):
        tool.main(["--income-year=1999", f"--output={tmp_path / 'y.h5'}"])
