"""The pinned ASEC education-assistance sidecar: load, verify, and fill.

The frozen census_cps inputs never carried raw ASEC ``ED_VAL``
(microcosm#417's sibling gap), so the education-inputs stage restores it from
the official survey archives via an exact per-income-year ``PERIDNUM`` join —
the ``weeks_unemployed``/``LKWEEKS`` repair pattern extended to every pooled
year.  These tests run the real loader and fill against synthetic archives
with overridden pins, exercising every fail-closed path.
"""

from __future__ import annotations

import dataclasses
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
    ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS,
    AsecEducationArchive,
    fill_asec_education_assistance_source,
    load_asec_education_assistance_sources,
)

_YEAR = 2023  # income year; survey archive 2024


def _peridnum(index: int) -> str:
    return f"{index:022d}"


def _source_frame(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PH_SEQ": np.arange(1, rows + 1, dtype=np.int64),
            "P_SEQ": np.ones(rows, dtype=np.int64),
            "A_LINENO": np.ones(rows, dtype=np.int64),
            "PERIDNUM": [_peridnum(index) for index in range(rows)],
            "ED_VAL": [0.0, 0.0, 2_500.0, 0.0, 12_000.0, 0.0][:rows],
            "A_FNLWGT": np.full(rows, 100.0),
        }
    )


def _write_archive(path: Path, frame: pd.DataFrame, member: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, frame.to_csv(index=False))


def _pins_for(path: Path, frame: pd.DataFrame, member: str) -> AsecEducationArchive:
    import hashlib

    zip_bytes = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        member_bytes = archive.read(member)
    weights = frame["A_FNLWGT"].to_numpy(dtype=np.float64) / 100.0
    values = frame["ED_VAL"].to_numpy(dtype=np.float64)
    positive = values > 0.0
    return AsecEducationArchive(
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
        positive_rows=int(positive.sum()),
        weighted_positive_share=float(weights[positive].sum() / weights.sum()),
        weighted_total=float(np.dot(weights, values)),
    )


@pytest.fixture
def pinned_archive(tmp_path, monkeypatch):
    frame = _source_frame()
    member = f"pppub{str(_YEAR + 1)[2:]}.csv"
    path = tmp_path / f"asecpub{str(_YEAR + 1)[2:]}csv.zip"
    _write_archive(path, frame, member)
    pins = _pins_for(path, frame, member)
    monkeypatch.setitem(ASEC_EDUCATION_ASSISTANCE_ARCHIVES, _YEAR, pins)
    return path, frame


def test_income_years_map_to_next_survey_year() -> None:
    """Every pinned archive is the survey published the year after income.

    The census_cps income-year universe is the survey-year+1 person file
    (verified upstream by 100.0% PERIDNUM coverage per year); pinning any
    other vintage only overlaps the ~33% rotation group and the exact-join
    fill fails loudly.
    """

    for income_year, pins in ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items():
        assert pins.survey_year == income_year + 1
        assert str(pins.survey_year) in pins.zip_url
        assert pins.member == f"pppub{str(pins.survey_year)[2:]}.csv"
    assert ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS == (2022, 2023, 2024)


def test_loader_reads_pinned_zip_and_audits(pinned_archive) -> None:
    path, frame = pinned_archive
    source = load_asec_education_assistance_sources(
        {_YEAR: path}, income_years=(_YEAR,)
    )
    assert list(source["source_year"].unique()) == [_YEAR]
    assert len(source) == len(frame)
    assert source.attrs["source_audit"][_YEAR]["positive_rows"] == 2


def test_loader_rejects_wrong_zip_bytes(pinned_archive, tmp_path) -> None:
    path, frame = pinned_archive
    tampered = tmp_path / "tampered.zip"
    member = f"pppub{str(_YEAR + 1)[2:]}.csv"
    corrupted = frame.copy()
    corrupted.loc[0, "ED_VAL"] = 999.0
    _write_archive(tampered, corrupted, member)
    with pytest.raises(ValueError, match="mismatch"):
        load_asec_education_assistance_sources({_YEAR: tampered}, income_years=(_YEAR,))


def test_loader_rejects_unpinned_income_year(pinned_archive) -> None:
    path, _ = pinned_archive
    with pytest.raises(ValueError, match="covers income year"):
        load_asec_education_assistance_sources({1999: path}, income_years=(1999,))


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


def test_fill_joins_ed_val_by_identity(pinned_archive) -> None:
    path, frame = pinned_archive
    source = load_asec_education_assistance_sources(
        {_YEAR: path}, income_years=(_YEAR,)
    )
    filled = fill_asec_education_assistance_source(_person_frame(), source)
    expected = frame.set_index("PERIDNUM")["ED_VAL"]
    for _, row in filled.iterrows():
        assert row["ED_VAL"] == expected[row["PERIDNUM"]]


def test_fill_fails_closed_on_uncovered_key(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_education_assistance_sources(
        {_YEAR: path}, income_years=(_YEAR,)
    )
    person = _person_frame()
    person.loc[0, "PERIDNUM"] = _peridnum(999)
    with pytest.raises(ValueError, match="does not cover frame PERIDNUM"):
        fill_asec_education_assistance_source(person, source)


def test_fill_fails_closed_on_uncovered_year(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_education_assistance_sources(
        {_YEAR: path}, income_years=(_YEAR,)
    )
    person = _person_frame()
    person["source_year"] = _YEAR - 1
    with pytest.raises(ValueError, match="does not cover pooled income"):
        fill_asec_education_assistance_source(person, source)


def test_fill_fails_closed_on_identity_mismatch(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_education_assistance_sources(
        {_YEAR: path}, income_years=(_YEAR,)
    )
    person = _person_frame()
    person.loc[1, "A_LINENO"] = 9
    with pytest.raises(ValueError, match="redundant identity mismatch"):
        fill_asec_education_assistance_source(person, source)


def test_fill_refuses_to_overwrite_measured_values(pinned_archive) -> None:
    path, _ = pinned_archive
    source = load_asec_education_assistance_sources(
        {_YEAR: path}, income_years=(_YEAR,)
    )
    person = _person_frame()
    person["ED_VAL"] = 1.0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        fill_asec_education_assistance_source(person, source)


def test_pins_reject_audit_drift(pinned_archive, monkeypatch) -> None:
    path, frame = pinned_archive
    pins = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[_YEAR]
    monkeypatch.setitem(
        ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
        _YEAR,
        dataclasses.replace(pins, positive_rows=pins.positive_rows + 1),
    )
    with pytest.raises(ValueError, match="audit drifted"):
        load_asec_education_assistance_sources({_YEAR: path}, income_years=(_YEAR,))
