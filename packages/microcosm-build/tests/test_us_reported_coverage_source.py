from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.us_runtime import reported_coverage_source as source_module
from microcosm.build.us_runtime.education_assistance_source import (
    AsecEducationArchive,
)
from microcosm.build.us_runtime.reported_coverage_source import (
    ASEC_REPORTED_COVERAGE_RAW_COLUMNS,
    fill_asec_reported_coverage_source,
    load_asec_reported_coverage_sources,
)


def _peridnum(value: int) -> str:
    return f"{value:022d}"


def _source() -> pd.DataFrame:
    source = pd.DataFrame(
        {
            "source_year": [2022, 2022, 2023, 2023],
            "PH_SEQ": [101, 101, 202, 303],
            "P_SEQ": [1, 2, 1, 1],
            "A_LINENO": [1, 2, 1, 1],
            "PERIDNUM": [_peridnum(value) for value in (1, 2, 3, 4)],
            "NOW_NONM": [1, 2, 2, 1],
            "NOW_MCAID": [2, 1, 1, 2],
            "NOW_CHAMPVA": [2, 2, 1, 2],
            "NOW_MIL": [1, 2, 2, 1],
            "NOW_VACARE": [2, 1, 2, 1],
            "NOW_OTHMT": [2, 2, 1, 2],
            "NOW_IHSFLG": [1, 2, 2, 1],
        }
    )
    source.attrs["source_audit"] = {
        2022: {"rows": 2},
        2023: {"rows": 2},
    }
    return source


def _person() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_year": [2023, 2022, 2023, 2022],
            "source_household_id": [303, 101, 202, 101],
            "P_SEQ": [1, 2, 1, 1],
            "A_LINENO": [1, 2, 1, 1],
            "PERIDNUM": [_peridnum(value) for value in (4, 2, 3, 1)],
        }
    )


def test_exact_join_restores_measured_2022_and_2023_coverage_recodes() -> None:
    source = _source()

    result = fill_asec_reported_coverage_source(_person(), source)

    expected = source.set_index(["source_year", "PERIDNUM"])
    keys = pd.MultiIndex.from_frame(result[["source_year", "PERIDNUM"]])
    for column in ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        assert result[column].tolist() == expected.reindex(keys)[column].tolist()
        assert str(result[column].dtype) == "int64"
    assert (
        result.attrs["reported_coverage_source_audit"] == source.attrs["source_audit"]
    )


def test_exact_join_refuses_missing_person_key() -> None:
    source = _source().iloc[:-1].copy()

    with pytest.raises(ValueError, match="does not cover frame PERIDNUM key"):
        fill_asec_reported_coverage_source(_person(), source)


def test_exact_join_refuses_duplicate_source_key() -> None:
    source = pd.concat([_source(), _source().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="keys must be unique"):
        fill_asec_reported_coverage_source(_person(), source)


def test_exact_join_refuses_missing_source_column() -> None:
    source = _source().drop(columns="NOW_OTHMT")

    with pytest.raises(ValueError, match="missing column.*NOW_OTHMT"):
        fill_asec_reported_coverage_source(_person(), source)


@pytest.mark.parametrize(
    "bad_value", (0, 3, None, "not-a-code", True, 1.5, float("inf"))
)
def test_exact_join_refuses_bad_source_domain(bad_value: object) -> None:
    source = _source()
    source["NOW_MCAID"] = source["NOW_MCAID"].astype(object)
    source.loc[0, "NOW_MCAID"] = bad_value

    with pytest.raises(ValueError, match="NOW_MCAID must be complete integers"):
        fill_asec_reported_coverage_source(_person(), source)


@pytest.mark.parametrize("field", ("P_SEQ", "A_LINENO", "source_household_id"))
def test_redundant_identity_cannot_disagree(field: str) -> None:
    person = _person()
    person.loc[0, field] = 999
    with pytest.raises(ValueError, match="redundant identity mismatch"):
        fill_asec_reported_coverage_source(person, _source())


@pytest.mark.parametrize("value", (2, 0, True))
def test_existing_observations_cannot_be_silently_replaced(value: object) -> None:
    person = _person()
    person["NOW_NONM"] = [value, None, None, None]
    with pytest.raises(ValueError, match="existing NOW_NONM disagrees"):
        fill_asec_reported_coverage_source(person, _source())


def test_native_recodes_and_row_order_are_preserved_even_with_duplicate_index() -> None:
    person = _person()
    person["NOW_GRP"] = [2, 1, 2, 1]
    person["NOW_MRK"] = [1, 2, 2, 1]
    person.index = [0, 0, 1, 1]
    original = person.copy(deep=True)
    result = fill_asec_reported_coverage_source(person, _source())
    pd.testing.assert_frame_equal(person, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result["NOW_NONM"].tolist() == [1, 2, 2, 1]


def test_exact_join_refuses_numeric_peridnum_without_string_coercion() -> None:
    source = _source()
    source["PERIDNUM"] = source["PERIDNUM"].astype(object)
    source.loc[0, "PERIDNUM"] = 1234567890123456789012
    with pytest.raises(ValueError, match="exact 22-digit string"):
        fill_asec_reported_coverage_source(_person(), source)


def test_exact_join_decodes_valid_peridnum_bytes() -> None:
    person = _person()
    person["PERIDNUM"] = person["PERIDNUM"].map(str.encode)
    result = fill_asec_reported_coverage_source(person, _source())
    assert result["PERIDNUM"].tolist() == _person()["PERIDNUM"].tolist()


def _pinned_member_fixture(path: Path) -> AsecEducationArchive:
    raw = (
        _source()
        .loc[
            lambda frame: frame["source_year"].eq(2022),
            [
                "PH_SEQ",
                "P_SEQ",
                "A_LINENO",
                "PERIDNUM",
                *ASEC_REPORTED_COVERAGE_RAW_COLUMNS,
            ],
        ]
        .copy()
    )
    raw["A_FNLWGT"] = [100.0, 300.0]
    raw.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return AsecEducationArchive(
        survey_year=2023,
        income_year=2022,
        zip_url="https://example.test/asecpub23csv.zip",
        zip_size_bytes=1,
        zip_sha256="0" * 64,
        member=path.name,
        member_size_bytes=path.stat().st_size,
        member_crc32="00000000",
        member_sha256=digest,
        rows=2,
        positive_rows=0,
        weighted_positive_share=0.0,
        weighted_total=0.0,
    )


def test_loader_enforces_pinned_member_identity_and_records_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "pppub23.csv"
    pins = _pinned_member_fixture(path)
    monkeypatch.setattr(
        source_module,
        "ASEC_EDUCATION_ASSISTANCE_ARCHIVES",
        {2022: pins},
    )

    loaded = load_asec_reported_coverage_sources(
        {2022: path},
        income_years=(2022,),
    )

    assert loaded["source_year"].tolist() == [2022, 2022]
    assert loaded["NOW_NONM"].tolist() == [1, 2]
    assert loaded.attrs["source_audit"][2022]["columns"]["NOW_NONM"] == {
        "no_rows": 1,
        "yes_rows": 1,
        "weighted_yes_share": 0.25,
    }

    monkeypatch.setattr(
        source_module,
        "ASEC_EDUCATION_ASSISTANCE_ARCHIVES",
        {2022: replace(pins, member_sha256="f" * 64)},
    )
    with pytest.raises(ValueError, match="person member SHA-256 mismatch"):
        load_asec_reported_coverage_sources(
            {2022: path},
            income_years=(2022,),
        )
