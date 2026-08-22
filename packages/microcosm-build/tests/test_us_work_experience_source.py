"""The pinned ASEC work-experience sidecar: load, verify, and fill.

The frozen census_cps inputs never carried the ASEC work-experience industry
recodes ``WEIND``/``WEMIND``, so the work-experience stage restores them from
the official survey archives via an exact per-income-year ``PERIDNUM`` join —
the education-assistance pattern applied to two columns.  These tests run
the real loader and fill against synthetic archives with overridden pins,
exercising every fail-closed path including the official universe identity
(``WEIND`` in 1..22 iff ``WKSWORK > 0``) the loader enforces.
"""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.work_experience_source import (
    ASEC_WORK_EXPERIENCE_ARCHIVES,
    ASEC_WORK_EXPERIENCE_INCOME_YEARS,
    ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS,
    AsecWorkExperienceArchive,
    fill_asec_work_experience_source,
    load_asec_work_experience_sources,
)

_YEAR = 2023  # income year; survey archive 2024


def _peridnum(index: int) -> str:
    return f"{index:022d}"


def _source_frame(rows: int = 6) -> pd.DataFrame:
    """Six people: three workers with industries, two nonworkers, one child."""

    return pd.DataFrame(
        {
            "PH_SEQ": np.arange(1, rows + 1, dtype=np.int64),
            "P_SEQ": np.ones(rows, dtype=np.int64),
            "A_LINENO": np.ones(rows, dtype=np.int64),
            "PERIDNUM": [_peridnum(index) for index in range(rows)],
            "WEIND": [7, 0, 21, 23, 16, 0][:rows],
            "WEMIND": [5, 0, 13, 15, 10, 0][:rows],
            "WKSWORK": [52, 0, 48, 0, 10, 0][:rows],
            "WORKYN": [1, 2, 1, 2, 2, 0][:rows],
            "A_FNLWGT": np.full(rows, 100.0),
        }
    )


def _write_archive(path: Path, frame: pd.DataFrame, member: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, frame.to_csv(index=False))


def _pins_for(
    path: Path, frame: pd.DataFrame, member: str
) -> AsecWorkExperienceArchive:
    zip_bytes = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        member_bytes = archive.read(member)
    weights = frame["A_FNLWGT"].to_numpy(dtype=np.float64) / 100.0
    worked = frame["WKSWORK"].to_numpy() > 0
    recode = frame["WEIND"].to_numpy() != 0
    return AsecWorkExperienceArchive(
        survey_year=_YEAR + 1,
        income_year=_YEAR,
        zip_url="https://example.invalid/asec.zip",
        zip_size_bytes=len(zip_bytes),
        zip_sha256=hashlib.sha256(zip_bytes).hexdigest(),
        member=member,
        member_size_bytes=info.file_size,
        member_crc32=f"{info.CRC:08x}",
        member_sha256=hashlib.sha256(member_bytes).hexdigest(),
        rows=len(frame),
        worked_rows=int(worked.sum()),
        weighted_worked_share=float(weights[worked].sum() / weights.sum()),
        recode_rows=int(recode.sum()),
        weighted_recode_share=float(weights[recode].sum() / weights.sum()),
    )


def _pin(tmp_path: Path, monkeypatch, frame: pd.DataFrame) -> Path:
    member = f"pppub{str(_YEAR + 1)[2:]}.csv"
    path = tmp_path / f"asecpub{str(_YEAR + 1)[2:]}csv.zip"
    _write_archive(path, frame, member)
    monkeypatch.setitem(
        ASEC_WORK_EXPERIENCE_ARCHIVES, _YEAR, _pins_for(path, frame, member)
    )
    return path


@pytest.fixture
def pinned_archive(tmp_path, monkeypatch):
    frame = _source_frame()
    return _pin(tmp_path, monkeypatch, frame), frame


def test_income_years_map_to_next_survey_year() -> None:
    """Every pinned archive is the survey published the year after income."""

    for income_year, pins in ASEC_WORK_EXPERIENCE_ARCHIVES.items():
        assert pins.survey_year == income_year + 1
        assert str(pins.survey_year) in pins.zip_url
        assert pins.member == f"pppub{str(pins.survey_year)[2:]}.csv"
        assert 0.0 < pins.weighted_worked_share < pins.weighted_recode_share < 1.0
        assert 0 < pins.worked_rows < pins.recode_rows < pins.rows
    assert ASEC_WORK_EXPERIENCE_INCOME_YEARS == (2022, 2023, 2024)


def test_source_columns_carry_only_identity_and_industry_recodes() -> None:
    assert ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS == (
        "PH_SEQ",
        "P_SEQ",
        "A_LINENO",
        "PERIDNUM",
        "WEIND",
        "WEMIND",
    )


def test_loader_reads_pinned_zip_and_audits(pinned_archive) -> None:
    path, frame = pinned_archive
    source = load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))
    assert list(source["source_year"].unique()) == [_YEAR]
    assert len(source) == len(frame)
    assert list(source.columns) == ["source_year", *ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS]
    audit = source.attrs["source_audit"][_YEAR]
    assert audit["worked_rows"] == 3
    assert audit["recode_rows"] == 4
    assert audit["weighted_worked_share"] == pytest.approx(0.5)
    assert audit["weighted_recode_share"] == pytest.approx(4 / 6)


def test_loader_rejects_wrong_zip_bytes(pinned_archive, tmp_path) -> None:
    path, frame = pinned_archive
    tampered = tmp_path / "tampered.zip"
    member = f"pppub{str(_YEAR + 1)[2:]}.csv"
    corrupted = frame.copy()
    corrupted.loc[0, "WEIND"] = 8
    _write_archive(tampered, corrupted, member)
    with pytest.raises(ValueError, match="mismatch"):
        load_asec_work_experience_sources({_YEAR: tampered}, income_years=(_YEAR,))


def test_loader_rejects_unpinned_income_year(pinned_archive) -> None:
    path, _ = pinned_archive
    with pytest.raises(ValueError, match="covers income year"):
        load_asec_work_experience_sources({1999: path}, income_years=(1999,))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"WEIND": 24}, r"WEIND must be an integer in \[0, 23\]"),
        ({"WEMIND": 16}, r"WEMIND must be an integer in \[0, 15\]"),
        ({"WEIND": 0, "WEMIND": 0}, "universe identity"),
        ({"WEMIND": 0}, "disagree on the not-in-universe rows"),
        ({"WORKYN": 1, "WKSWORK": 0, "WEIND": 23}, "WORKYN = 1 without positive"),
    ),
)
def test_loader_enforces_official_universe_identities(
    tmp_path, monkeypatch, mutation: dict, message: str
) -> None:
    frame = _source_frame()
    for column, value in mutation.items():
        frame.loc[0, column] = value
    path = _pin(tmp_path, monkeypatch, frame)
    with pytest.raises(ValueError, match=message):
        load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))


def test_pins_reject_audit_drift(pinned_archive, monkeypatch) -> None:
    path, _ = pinned_archive
    pins = ASEC_WORK_EXPERIENCE_ARCHIVES[_YEAR]
    monkeypatch.setitem(
        ASEC_WORK_EXPERIENCE_ARCHIVES,
        _YEAR,
        dataclasses.replace(pins, worked_rows=pins.worked_rows + 1),
    )
    with pytest.raises(ValueError, match="audit drifted"):
        load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))


def _person_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_year": np.full(rows, _YEAR, dtype=np.int64),
            "PERIDNUM": [_peridnum(index) for index in range(rows)],
            "PH_SEQ": np.arange(1, rows + 1, dtype=np.int64),
            "P_SEQ": np.ones(rows, dtype=np.int64),
            "A_LINENO": np.ones(rows, dtype=np.int64),
        }
    )


def test_fill_joins_both_recodes_by_identity(pinned_archive) -> None:
    path, frame = pinned_archive
    source = load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))
    filled = fill_asec_work_experience_source(_person_frame(), source)
    expected = frame.set_index("PERIDNUM")
    for _, row in filled.iterrows():
        assert row["WEIND"] == expected.loc[row["PERIDNUM"], "WEIND"]
        assert row["WEMIND"] == expected.loc[row["PERIDNUM"], "WEMIND"]
    assert filled["WEIND"].dtype == np.int64
    assert filled["WEMIND"].dtype == np.int64
    assert filled.attrs["work_experience_source_audit"][_YEAR]["worked_rows"] == 3


def test_fill_fails_closed_on_uncovered_key(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))
    person = _person_frame()
    person.loc[0, "PERIDNUM"] = _peridnum(999)
    with pytest.raises(ValueError, match="does not cover frame PERIDNUM"):
        fill_asec_work_experience_source(person, source)


def test_fill_fails_closed_on_uncovered_year(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))
    person = _person_frame()
    person["source_year"] = _YEAR - 1
    with pytest.raises(ValueError, match="does not cover pooled income"):
        fill_asec_work_experience_source(person, source)


def test_fill_fails_closed_on_identity_mismatch(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))
    person = _person_frame()
    person.loc[1, "A_LINENO"] = 9
    with pytest.raises(ValueError, match="redundant identity mismatch"):
        fill_asec_work_experience_source(person, source)


def test_fill_refuses_to_overwrite_measured_values(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_work_experience_sources({_YEAR: path}, income_years=(_YEAR,))
    person = _person_frame()
    person["WEMIND"] = 1
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fill_asec_work_experience_source(person, source)
