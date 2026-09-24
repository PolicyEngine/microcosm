"""The #720 Census person-column restoration: pinned, one-to-one, never overwriting."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import re
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.asec_census_person_columns import (
    ASEC_CENSUS_PERSON_COLUMN_NAMES,
    ASEC_CENSUS_PERSON_COLUMNS,
    ASEC_CENSUS_PERSON_COLUMNS_BEYOND_OFFLINE_FIX,
    ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED,
    ASEC_CENSUS_PERSON_IDENTITY_COLUMNS,
    WEIND_TO_WEMIND,
    AsecCensusPersonColumnsError,
    restore_asec_census_person_columns,
)
from microcosm.build.us_runtime.spm_role_source import (
    _OPTIONAL_RAW_CHECKS,
    ASEC_SPM_ROLE_SOURCES,
    AsecSpmRoleSource,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MODULE = (
    _REPOSITORY_ROOT
    / "packages/microcosm-build/src/microcosm/build/us_runtime"
    / "asec_census_person_columns.py"
)

#: The 24 person columns the 2026-08-23 offline fix appended to the 2022 input
#: (``_buildo-runtime/inputs/asec-720/receipt_720.json``; the 2023 input
#: already carried LKWEEKS). The review must place every one of them.
_RECEIPT_720_ADDED = frozenset(
    {
        "PECOHAB",
        "A_ENRLW",
        "A_FTPT",
        "A_FAMREL",
        "A_FAMTYP",
        "A_EXPRRP",
        "LKWEEKS",
        "NOW_CAID",
        "NOW_CHAMPVA",
        "NOW_COV",
        "NOW_DIR",
        "NOW_IHSFLG",
        "NOW_MCAID",
        "NOW_MCARE",
        "NOW_MIL",
        "NOW_MRKS",
        "NOW_MRKUN",
        "NOW_NONM",
        "NOW_OTHMT",
        "NOW_PCHIP",
        "NOW_PRIV",
        "NOW_PUB",
        "NOW_VACARE",
        "PTOTVAL",
    }
)

#: H5 row order differs from the member's, so a positional copy would fail.
_H5_ORDER = [3, 0, 7, 5, 1, 6, 2, 4]

#: Member-order industry codes: six workers (one in the Armed Forces) and two
#: people 15+ who did not work last year. Every member person is 16+.
_WEIND = [7, 16, 23, 22, 23, 9, 1, 21]
#: Weeks worked last year consistent with ``_WEIND`` (positive iff 1--22).
_WKSWORK = [52, 40, 0, 52, 0, 12, 26, 3]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member() -> pd.DataFrame:
    """A complete synthetic Census person member for eight people."""

    member = pd.DataFrame(
        {
            # Leading zeros: a numeric parse would change the key.
            "PERIDNUM": [f"{number:022}" for number in range(101, 109)],
            "PH_SEQ": [1, 1, 2, 2, 2, 3, 4, 4],
            "P_SEQ": [1, 2, 1, 2, 3, 1, 1, 2],
            "A_LINENO": [1, 2, 1, 2, 3, 1, 1, 2],
            "A_AGE": [44, 41, 70, 16, 19, 30, 52, 23],
            "A_SEX": [1, 2, 2, 1, 2, 1, 1, 2],
        }
    )
    for spec in ASEC_CENSUS_PERSON_COLUMNS:
        if spec.domain is None:
            member[spec.name] = [-1_200, 0, 18_500, 0, 3_000, 61_000, 0, 9_999]
        else:
            codes = sorted(spec.domain)
            member[spec.name] = [codes[(row * 3) % len(codes)] for row in range(8)]
    # The industry pair is rigid: WEMIND is the major group of WEIND.
    member["WEIND"] = _WEIND
    member["WEMIND"] = [WEIND_TO_WEMIND[code] for code in _WEIND]
    return member


def _write(tmp_path: Path, member: pd.DataFrame, *, name: str = "pppub23.csv"):
    """Write ``member`` as a CSV and as a Census-style archive; return their pin."""

    csv = tmp_path / name
    member.to_csv(csv, index=False)
    archive = tmp_path / "asecpub23csv.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
        out.write(csv, arcname=name)
        out.writestr("hhpub23.csv", "H_SEQ\n1\n")
    pin = AsecSpmRoleSource(
        income_year=2022,
        survey_year=2023,
        csv_sha256=_digest(csv),
        csv_size_bytes=csv.stat().st_size,
        persons=len(member),
        units=0,
        official_archive_url="https://example.invalid/asecpub23csv.zip",
        archive_sha256=_digest(archive),
        member=name,
    )
    return csv, archive, pin


def _h5_person(member: pd.DataFrame, *, carry: tuple[str, ...] = ()) -> pd.DataFrame:
    """The H5 side: the same people in another order, lacking reviewed columns."""

    rows = member.iloc[_H5_ORDER].reset_index(drop=True)
    person = rows[["PERIDNUM", *ASEC_CENSUS_PERSON_IDENTITY_COLUMNS]].copy()
    person["PERIDNUM"] = person["PERIDNUM"].astype("string")
    person["NOW_GRP"] = np.array([1, 2, 2, 1, 1, 2, 1, 2], dtype=np.int64)
    # An existing float column with a missing value must survive untouched.
    person["WSAL_VAL"] = [0.0, np.nan, 5.5, 0.0, 1.0, 2.0, 3.0, 4.0]
    for column in carry:
        person[column] = rows[column].to_numpy()
    return person


@pytest.fixture
def sources(tmp_path):
    member = _member()
    csv, archive, pin = _write(tmp_path, member)
    return member, csv, archive, pin


def _restore(person, path, pin):
    return restore_asec_census_person_columns(
        person, income_year=2022, source_path=path, pin=pin
    )


@pytest.mark.parametrize("form", ["archive", "csv"])
def test_appends_every_reviewed_column_the_h5_lacks_by_peridnum(sources, form):
    member, csv, archive, pin = sources
    person = _h5_person(member)
    before = person.copy(deep=True)

    result, record = _restore(person, archive if form == "archive" else csv, pin)

    pd.testing.assert_frame_equal(person, before)  # input not mutated
    width = len(before.columns)
    pd.testing.assert_frame_equal(result.iloc[:, :width], before)
    assert list(result.columns[width:]) == list(ASEC_CENSUS_PERSON_COLUMN_NAMES)
    expected = member.iloc[_H5_ORDER].reset_index(drop=True)
    for column in ASEC_CENSUS_PERSON_COLUMN_NAMES:
        assert result[column].dtype == np.int64
        np.testing.assert_array_equal(
            result[column].to_numpy(), expected[column].to_numpy()
        )
    assert "A_SEX" not in result  # an unreviewed member column is never read in
    assert record["source_form"] == form
    assert record["columns_added"] == list(ASEC_CENSUS_PERSON_COLUMN_NAMES)
    assert record["columns_verified_equal"] == []
    assert record["h5_person_rows"] == record["member_person_rows"] == 8
    assert record["joined_person_rows"] == 8
    assert record["archive_sha256"] == pin.archive_sha256
    assert record["member_sha256"] == pin.csv_sha256
    assert record["member"] == "pppub23.csv"
    assert record["identity_columns_verified_equal"] == list(
        ASEC_CENSUS_PERSON_IDENTITY_COLUMNS
    )
    assert record["member_values"]["NOW_MCAID"] == {
        "1": int((member["NOW_MCAID"] == 1).sum()),
        "2": int((member["NOW_MCAID"] == 2).sum()),
    }
    assert record["member_values"]["PTOTVAL"] == {
        "nonzero_rows": 5,
        "negative_rows": 1,
    }


def test_carried_columns_are_verified_against_the_member_never_overwritten(sources):
    """The 2024 case: the H5 carries every reviewed column, so all are checked."""

    member, _, archive, pin = sources
    person = _h5_person(member, carry=ASEC_CENSUS_PERSON_COLUMN_NAMES)

    result, record = _restore(person, archive, pin)

    pd.testing.assert_frame_equal(result, person)
    assert record["columns_added"] == []
    assert record["columns_verified_equal"] == list(ASEC_CENSUS_PERSON_COLUMN_NAMES)


def test_partially_carried_h5_gets_only_the_missing_columns(sources):
    member, _, archive, pin = sources
    person = _h5_person(member, carry=("NOW_MCAID", "A_EXPRRP"))

    result, record = _restore(person, archive, pin)

    assert list(result.columns[: len(person.columns)]) == list(person.columns)
    assert record["columns_verified_equal"] == ["NOW_MCAID", "A_EXPRRP"]
    assert record["columns_added"] == [
        name
        for name in ASEC_CENSUS_PERSON_COLUMN_NAMES
        if name not in {"NOW_MCAID", "A_EXPRRP"}
    ]


def test_refuses_a_carried_column_that_disagrees_with_the_member(sources):
    member, _, archive, pin = sources
    person = _h5_person(member, carry=ASEC_CENSUS_PERSON_COLUMN_NAMES)
    person.loc[2, "NOW_MCAID"] = 3 - person.loc[2, "NOW_MCAID"]

    with pytest.raises(
        AsecCensusPersonColumnsError, match=r"NOW_MCAID disagrees .* on 1 "
    ):
        _restore(person, archive, pin)


def test_refuses_a_carried_column_with_missing_values(sources):
    member, _, archive, pin = sources
    person = _h5_person(member, carry=("NOW_MCAID",))
    person["NOW_MCAID"] = person["NOW_MCAID"].astype(float)
    person.loc[0, "NOW_MCAID"] = np.nan

    with pytest.raises(AsecCensusPersonColumnsError, match="never overwritten"):
        _restore(person, archive, pin)


def test_refuses_duplicate_h5_keys(sources):
    member, _, archive, pin = sources
    person = _h5_person(member)
    person.loc[1, "PERIDNUM"] = person.loc[0, "PERIDNUM"]

    with pytest.raises(AsecCensusPersonColumnsError, match="repeats 2 PERIDNUM"):
        _restore(person, archive, pin)


def test_refuses_duplicate_member_keys(tmp_path):
    member = _member()
    member.loc[7, "PERIDNUM"] = member.loc[6, "PERIDNUM"]
    _, archive, pin = _write(tmp_path, member)
    person = _h5_person(_member())

    with pytest.raises(AsecCensusPersonColumnsError, match="repeats 2 PERIDNUM"):
        _restore(person, archive, pin)


def test_refuses_an_h5_person_missing_from_the_member(sources):
    member, _, archive, pin = sources
    person = _h5_person(member)
    person.loc[4, "PERIDNUM"] = "9" * 22

    with pytest.raises(AsecCensusPersonColumnsError, match="join must be total"):
        _restore(person, archive, pin)


def test_refuses_a_member_person_missing_from_the_h5(sources):
    member, _, archive, pin = sources
    person = _h5_person(member).iloc[:-1]

    with pytest.raises(AsecCensusPersonColumnsError, match="absent from"):
        _restore(person, archive, pin)


def test_refuses_a_non_census_key(sources):
    member, _, archive, pin = sources
    person = _h5_person(member)
    person.loc[0, "PERIDNUM"] = person.loc[0, "PERIDNUM"][1:]

    with pytest.raises(AsecCensusPersonColumnsError, match="22-digit"):
        _restore(person, archive, pin)


@pytest.mark.parametrize("column", ASEC_CENSUS_PERSON_IDENTITY_COLUMNS)
def test_refuses_identity_disagreement_after_the_join(sources, column):
    member, _, archive, pin = sources
    person = _h5_person(member)
    person.loc[5, column] = person.loc[5, column] + 1

    with pytest.raises(AsecCensusPersonColumnsError, match=f"{column} disagrees"):
        _restore(person, archive, pin)


def test_refuses_an_h5_without_the_join_or_identity_columns(sources):
    member, _, archive, pin = sources
    with pytest.raises(AsecCensusPersonColumnsError, match="P_SEQ"):
        _restore(_h5_person(member).drop(columns=["P_SEQ"]), archive, pin)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "parsed as float64"),
        ("yes", "parsed as"),
        (3, "outside its reviewed Census codes"),
    ],
)
def test_refuses_member_dtype_and_code_surprises(tmp_path, value, match):
    member = _member().astype({"NOW_MCAID": object})
    member.loc[2, "NOW_MCAID"] = value
    _, archive, pin = _write(tmp_path, member)

    with pytest.raises(AsecCensusPersonColumnsError, match=match):
        _restore(_h5_person(_member()), archive, pin)


def test_refuses_a_member_without_a_reviewed_column(tmp_path):
    _, archive, pin = _write(tmp_path, _member().drop(columns=["NOW_IHSFLG"]))

    with pytest.raises(AsecCensusPersonColumnsError, match="NOW_IHSFLG"):
        _restore(_h5_person(_member()), archive, pin)


def test_refuses_sources_that_fail_their_pins(sources):
    member, csv, archive, pin = sources
    person = _h5_person(member)
    with pytest.raises(AsecCensusPersonColumnsError, match="archive SHA-256"):
        _restore(person, archive, replace(pin, archive_sha256="0" * 64))
    with pytest.raises(AsecCensusPersonColumnsError, match="CSV SHA-256"):
        _restore(person, archive, replace(pin, csv_sha256="0" * 64))
    with pytest.raises(AsecCensusPersonColumnsError, match="CSV SHA-256"):
        _restore(person, csv, replace(pin, csv_sha256="0" * 64))
    with pytest.raises(AsecCensusPersonColumnsError, match="exactly one"):
        _restore(person, archive, replace(pin, member="pppub99.csv"))
    with pytest.raises(AsecCensusPersonColumnsError, match="the pin records 9"):
        _restore(person, archive, replace(pin, persons=9))


def test_refuses_an_unpinned_or_mismatched_income_year(sources):
    member, _, archive, pin = sources
    person = _h5_person(member)
    with pytest.raises(AsecCensusPersonColumnsError, match="No pinned"):
        restore_asec_census_person_columns(
            person, income_year=2019, source_path=archive
        )
    with pytest.raises(AsecCensusPersonColumnsError, match="income year 2023"):
        restore_asec_census_person_columns(
            person, income_year=2023, source_path=archive, pin=pin
        )


def test_default_pins_are_the_shared_census_person_pins(sources):
    """Without an explicit pin the production table applies (and refuses here)."""

    member, _, archive, _ = sources
    assert set(ASEC_SPM_ROLE_SOURCES) == {2022, 2023, 2024}
    with pytest.raises(AsecCensusPersonColumnsError, match="archive SHA-256"):
        restore_asec_census_person_columns(
            _h5_person(member), income_year=2022, source_path=archive
        )


def test_the_review_places_every_offline_fix_column_exactly_once():
    restored = set(ASEC_CENSUS_PERSON_COLUMN_NAMES)
    not_restored = set(ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED)
    beyond = set(ASEC_CENSUS_PERSON_COLUMNS_BEYOND_OFFLINE_FIX)
    assert len(restored) == len(ASEC_CENSUS_PERSON_COLUMN_NAMES)
    assert restored.isdisjoint(not_restored)
    assert beyond == {"WEIND", "WEMIND"}
    assert beyond <= restored
    assert beyond.isdisjoint(_RECEIPT_720_ADDED)
    assert (restored - beyond) | not_restored == _RECEIPT_720_ADDED
    assert not restored & set(ASEC_CENSUS_PERSON_IDENTITY_COLUMNS)


def test_weind_to_wemind_is_the_codebook_nesting():
    """Pinned from all three members; every major group nests detailed ones."""

    assert dict(WEIND_TO_WEMIND) == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 5,
        7: 5,
        8: 6,
        9: 6,
        10: 7,
        11: 8,
        12: 8,
        13: 9,
        14: 9,
        15: 10,
        16: 10,
        17: 11,
        18: 11,
        19: 12,
        20: 12,
        21: 13,
        22: 14,
        23: 15,
    }
    specs = {spec.name: spec for spec in ASEC_CENSUS_PERSON_COLUMNS}
    assert set(WEIND_TO_WEMIND) == specs["WEIND"].domain
    assert set(WEIND_TO_WEMIND.values()) == specs["WEMIND"].domain
    majors = list(WEIND_TO_WEMIND.values())
    assert majors == sorted(majors)


def test_restores_industry_and_verifies_the_worker_universe(sources):
    member, csv, _, pin = sources
    person = _h5_person(member)
    person["WKSWORK"] = np.asarray(_WKSWORK, dtype=np.int64)[_H5_ORDER]
    result, record = _restore(person, csv, pin)
    joined = member.set_index("PERIDNUM").loc[person["PERIDNUM"].astype(str)]
    assert result["WEIND"].tolist() == joined["WEIND"].tolist()
    assert result["WEMIND"].tolist() == joined["WEMIND"].tolist()
    assert record["work_experience_universe"] == {
        "weind_to_wemind_rows_verified": 8,
        "worker_code_iff_weeks_worked": "verified",
        "worker_rows": 6,
    }


def test_records_that_the_worker_universe_needs_wkswork(sources):
    member, csv, _, pin = sources
    _, record = _restore(_h5_person(member), csv, pin)
    assert record["work_experience_universe"] == {
        "weind_to_wemind_rows_verified": 8,
        "worker_code_iff_weeks_worked": "not checked: H5 lacks WKSWORK",
    }


@pytest.mark.parametrize("weeks", [0, 30])
def test_refuses_a_worker_code_that_disagrees_with_weeks_worked(sources, weeks):
    member, csv, _, pin = sources
    person = _h5_person(member)
    wkswork = np.asarray(_WKSWORK, dtype=np.int64)[_H5_ORDER]
    # H5 row 1 is member row 0 (a worker); H5 row 6 is member row 2 (no work).
    wkswork[1 if weeks == 0 else 6] = weeks
    person["WKSWORK"] = wkswork
    with pytest.raises(AsecCensusPersonColumnsError, match="worker code"):
        _restore(person, csv, pin)


def test_refuses_a_major_group_that_is_not_the_detailed_groups_parent(tmp_path):
    member = _member()
    member.loc[0, "WEMIND"] = 6  # WEIND 7 nests in major group 5
    csv, _, pin = _write(tmp_path, member)
    with pytest.raises(AsecCensusPersonColumnsError, match="major group of WEIND"):
        _restore(_h5_person(member), csv, pin)


def _cited_objects(reader: str):
    for dotted in re.findall(r"\b(?:microcosm|microunit)(?:\.\w+)+", reader):
        parts = dotted.split(".")
        for split in range(len(parts), 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:split]))
            except ImportError:
                continue
            target = module
            for attribute in parts[split:]:
                target = getattr(target, attribute)
            yield dotted, module, target
            break
        else:
            raise AssertionError(f"cited reader {dotted!r} does not import")


def test_every_restored_column_cites_readers_that_really_read_it():
    """Each cited function or constant must resolve and itself name the column."""

    pytest.importorskip("microunit")
    for spec in ASEC_CENSUS_PERSON_COLUMNS:
        assert spec.readers, spec.name
        for reader in spec.readers:
            cited = list(_cited_objects(reader))
            assert cited, reader
            for dotted, _module, target in cited:
                if inspect.isfunction(target):
                    names_it = bool(
                        re.search(rf"\b{spec.name}\b", inspect.getsource(target))
                    )
                else:
                    names_it = spec.name in target
                assert names_it, (
                    f"{dotted} is cited as a reader of {spec.name} but does not "
                    "name it."
                )


def test_columns_left_unrestored_for_want_of_a_reader_still_have_none():
    """If code starts reading one, it must join the reviewed restoration set."""

    unread = sorted(
        column
        for column, reason in ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED.items()
        if reason.startswith("No build reader")
    )
    assert unread == [
        "NOW_CAID",
        "NOW_COV",
        "NOW_DIR",
        "NOW_MCARE",
        "NOW_MRKS",
        "NOW_MRKUN",
        "NOW_PCHIP",
        "NOW_PRIV",
        "NOW_PUB",
    ]
    pattern = re.compile(rf"\b({'|'.join(unread)})\b")
    roots = [*(_REPOSITORY_ROOT / "packages").glob("*/src"), _REPOSITORY_ROOT / "tools"]
    readers = {
        str(path.relative_to(_REPOSITORY_ROOT)): sorted(set(pattern.findall(text)))
        for root in roots
        for path in root.rglob("*.py")
        if path.resolve() != _MODULE.resolve()
        and pattern.search(text := path.read_text(encoding="utf-8"))
    }
    assert readers == {}


def test_other_unrestored_columns_keep_their_recorded_reasons():
    from microcosm.build.us_runtime.weeks_unemployed import (
        ASEC_2023_WEEKS_UNEMPLOYED_MEMBER,
        ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256,
        ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR,
        ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
    )

    for column in ("A_FAMTYP", "A_FAMREL", "PECOHAB"):
        assert column in _OPTIONAL_RAW_CHECKS
    # LKWEEKS: the existing 2022 repair reads the very member pinned here.
    pin = ASEC_SPM_ROLE_SOURCES[ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR]
    assert ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256 == pin.archive_sha256
    assert ASEC_2023_WEEKS_UNEMPLOYED_MEMBER == pin.member
    assert ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256 == pin.csv_sha256


# --- the pool: every vintage is restored before anything reads it -----------


def _write_asec_h5(path: Path, member: pd.DataFrame, *, carry: bool) -> None:
    person = member[["PERIDNUM", *ASEC_CENSUS_PERSON_IDENTITY_COLUMNS, "A_SEX"]].copy()
    person["A_MARITL"] = [1, 1, 7, 7, 7, 7, 7, 7]
    person["A_SPOUSE"] = [2, 1, 0, 0, 0, 0, 0, 0]
    person["PEPAR1"] = [0, 0, 0, 1, 1, 0, 0, 0]
    person["PEPAR2"] = 0
    person["PF_SEQ"] = 1
    person["SPM_ID"] = person["PH_SEQ"]
    person["NOW_GRP"] = 2
    person["NOW_MRK"] = 2
    if carry:
        for column in ASEC_CENSUS_PERSON_COLUMN_NAMES:
            person[column] = member[column].to_numpy()
    household = pd.DataFrame(
        {
            "H_SEQ": [1, 2, 3, 4],
            "HSUP_WGT": [10_000, 20_000, 15_000, 5_000],
            "GESTFIPS": [6, 36, 48, 12],
            "H_TENURE": [1, 2, 1, 2],
        }
    )
    person.to_hdf(path, key="person", mode="w")
    household.to_hdf(path, key="household", mode="a")


@pytest.fixture
def pool_inputs(tmp_path):
    pytest.importorskip("tables")
    pytest.importorskip("microunit")
    from microcosm.build.us_runtime.asec_pool import AsecSource

    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    member = _member()
    # Person 4 is a partner the derived fallback cannot identify (it codes 12).
    member.loc[4, "A_EXPRRP"] = 13
    _, old_archive, old_pin = _write(old, member)
    _, new_archive, new_pin = _write(new, member, name="pppub25.csv")
    new_pin = replace(new_pin, income_year=2024, survey_year=2025)
    _write_asec_h5(old / "census_cps_2022.h5", member, carry=False)
    _write_asec_h5(new / "census_cps_2024.h5", member, carry=True)
    return member, [
        AsecSource(
            year=2024,
            path=new / "census_cps_2024.h5",
            census_person_source=new_archive,
            census_person_pin=new_pin,
        ),
        AsecSource(
            year=2022,
            path=old / "census_cps_2022.h5",
            census_person_source=old_archive,
            census_person_pin=old_pin,
        ),
    ]


def test_pool_restores_every_vintage_before_the_relationship_recode(pool_inputs):
    from microcosm.build.us_runtime.asec_pool import pool_asec_sources

    member, sources = pool_inputs
    pooled = pool_asec_sources(sources)
    person = pooled.person

    for column in ASEC_CENSUS_PERSON_COLUMN_NAMES:
        assert person[column].notna().all(), column
        assert person[column].dtype == np.int64, column
        for year in (2022, 2024):
            rows = person.loc[person["source_year"] == year]
            expected = member.set_index("PERIDNUM").loc[rows["PERIDNUM"], column]
            np.testing.assert_array_equal(rows[column].to_numpy(), expected)
    by_year = {item["year"]: item for item in pooled.metadata["sources"]}
    assert {
        year: item["relationship_recode_source"] for year, item in by_year.items()
    } == {
        2022: "source:A_EXPRRP",
        2024: "source:A_EXPRRP",
    }
    old, new = (
        by_year[2022]["census_person_columns"],
        by_year[2024]["census_person_columns"],
    )
    assert old["columns_added"] == list(ASEC_CENSUS_PERSON_COLUMN_NAMES)
    assert new["columns_added"] == []
    assert new["columns_verified_equal"] == list(ASEC_CENSUS_PERSON_COLUMN_NAMES)
    partner = person["PERIDNUM"].eq(member.loc[4, "PERIDNUM"])
    assert set(person.loc[partner, "A_EXPRRP"]) == {13}


def test_pool_without_a_census_person_source_keeps_the_720_hole(pool_inputs):
    """The library default documents the defect the build tools now close."""

    from microcosm.build.us_runtime.asec_pool import pool_asec_sources

    _, sources = pool_inputs
    bare = [
        replace(source, census_person_source=None, census_person_pin=None)
        for source in sources
    ]
    pooled = pool_asec_sources(bare)
    old = pooled.person["source_year"] == 2022
    assert pooled.person.loc[old, "NOW_MCAID"].isna().all()
    assert {
        item["year"]: item["relationship_recode_source"]
        for item in pooled.metadata["sources"]
    } == {2022: "derived:line_spouse_parent", 2024: "source:A_EXPRRP"}
    assert all(
        item["census_person_columns"] is None for item in pooled.metadata["sources"]
    )


def test_pool_smoke_limit_still_checks_the_complete_join(pool_inputs):
    from microcosm.build.us_runtime.asec_pool import pool_asec_sources

    _, sources = pool_inputs
    smoke = [replace(source, max_households=1) for source in sources]
    pooled = pool_asec_sources(smoke)
    assert set(pooled.person["source_household_id"]) == {1}
    for item in pooled.metadata["sources"]:
        assert item["census_person_columns"]["joined_person_rows"] == 8


def test_pool_refuses_a_source_that_fails_the_restoration(pool_inputs, tmp_path):
    from microcosm.build.us_runtime.asec_pool import pool_asec_sources

    _, sources = pool_inputs
    broken = list(sources)
    broken[1] = replace(
        broken[1],
        census_person_pin=replace(broken[1].census_person_pin, archive_sha256="0" * 64),
    )
    with pytest.raises(AsecCensusPersonColumnsError, match="archive SHA-256"):
        pool_asec_sources(broken)


def test_pooled_unit_frame_builds_on_the_restored_columns(pool_inputs):
    from microcosm.build.us_runtime.asec_pool import build_pooled_asec_unit_frame

    _, sources = pool_inputs
    frame, metadata = build_pooled_asec_unit_frame(sources, target_year=2024)
    person = frame.table("person")
    assert person["NOW_MCAID"].notna().all()
    assert frame.n("tax_unit") > 0
    assert all(item["census_person_columns"] for item in metadata["sources"])
