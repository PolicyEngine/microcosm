"""Measured CPS ASEC educational assistance for the education-inputs stage.

The retired eCPS build carried person-level educational assistance directly
from ASEC ``ED_VAL`` (archived ``datasets/cps/cps.py`` line 1493:
``cps["educational_assistance"] = person.ED_VAL`` — a direct carry with no
recode, filter, or universe restriction), and the archived
``datasets/cps/census_cps.py`` listed ``ED_VAL`` among the raw ASEC columns it
kept.  The frozen ``census_cps_*.h5`` inputs populace builds from never
carried that column, so this module restores it from the official, immutable
Census ASEC public-use archives — the same repair class as
:mod:`.weeks_unemployed`'s ``LKWEEKS`` sidecar, extended to every pooled
income year.

Income years map to survey archives one year later: the ``census_cps_YYYY``
income-year universe is exactly the ASEC survey-year ``YYYY+1`` person file
(verified per year by 100.0% ``PERIDNUM`` coverage and equal row counts).
The repair is an exact ``PERIDNUM`` join within each income year with
redundant Census identity checks; it never predicts a source value.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

__all__ = [
    "ASEC_EDUCATION_ASSISTANCE_ARCHIVES",
    "ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS",
    "ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS",
    "EDUCATION_ASSISTANCE_ARCHIVED_DERIVATION_URL",
    "EDUCATION_ASSISTANCE_ARCHIVED_SOURCE_URL",
    "AsecEducationArchive",
    "fetch_asec_education_assistance_source",
    "fill_asec_education_assistance_source",
    "load_asec_education_assistance_sources",
]

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_PACKAGE = "policyengine_" + "us_data"
_ARCHIVED_ROOT = (
    f"https://github.com/PolicyEngine/{_ARCHIVED_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_ARCHIVED_PACKAGE}"
)
EDUCATION_ASSISTANCE_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "/datasets/cps/cps.py#L1493"
)
EDUCATION_ASSISTANCE_ARCHIVED_SOURCE_URL = (
    _ARCHIVED_ROOT + "/datasets/cps/census_cps.py#L357"
)

_SOURCE = "ED_VAL"
ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "PERIDNUM",
    _SOURCE,
)
_AUDIT_WEIGHT_COLUMN = "A_FNLWGT"


@dataclass(frozen=True)
class AsecEducationArchive:
    """Pinned identity of one official ASEC survey-year person archive."""

    survey_year: int
    income_year: int
    zip_url: str
    zip_size_bytes: int
    zip_sha256: str
    member: str
    member_size_bytes: int
    member_crc32: str
    member_sha256: str
    rows: int
    positive_rows: int
    weighted_positive_share: float
    weighted_total: float


#: One pinned archive per pooled income year. The survey-year file published
#: the March after each income year carries that income year's person
#: universe: row counts equal the pooled cohorts exactly and PERIDNUM
#: coverage is 100.0% per year (and only ~33% against any adjacent survey
#: year, the CPS rotation-group overlap — pinning the wrong vintage fails the
#: full-coverage join loudly).
ASEC_EDUCATION_ASSISTANCE_ARCHIVES: dict[int, AsecEducationArchive] = {
    archive.income_year: archive
    for archive in (
        AsecEducationArchive(
            survey_year=2023,
            income_year=2022,
            zip_url=(
                "https://www2.census.gov/programs-surveys/cps/datasets/2023/"
                "march/asecpub23csv.zip"
            ),
            zip_size_bytes=150_165_063,
            zip_sha256=(
                "d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11"
            ),
            member="pppub23.csv",
            member_size_bytes=281_065_733,
            member_crc32="49c09e5f",
            member_sha256=(
                "19b56537e50e7663f954361ef2bb5ce9cef8d9d45f156fe1a69a99b654198ffe"
            ),
            rows=146_133,
            positive_rows=2_588,
            weighted_positive_share=0.019982647283099255,
            weighted_total=68_484_835_026.81,
        ),
        AsecEducationArchive(
            survey_year=2024,
            income_year=2023,
            zip_url=(
                "https://www2.census.gov/programs-surveys/cps/datasets/2024/"
                "march/asecpub24csv.zip"
            ),
            zip_size_bytes=148_664_101,
            zip_sha256=(
                "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09"
            ),
            member="pppub24.csv",
            member_size_bytes=277_250_415,
            member_crc32="87950ece",
            member_sha256=(
                "21a2b9e0e4b08534563578a45acad77868af4ae9a7d46f23776b707d4a559aa7"
            ),
            rows=144_265,
            positive_rows=2_971,
            weighted_positive_share=0.022189613802856226,
            weighted_total=72_008_253_866.15,
        ),
        AsecEducationArchive(
            survey_year=2025,
            income_year=2024,
            zip_url=(
                "https://www2.census.gov/programs-surveys/cps/datasets/2025/"
                "march/asecpub25csv.zip"
            ),
            zip_size_bytes=147_271_429,
            zip_sha256=(
                "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b"
            ),
            member="pppub25.csv",
            member_size_bytes=277_882_549,
            member_crc32="7dc2878f",
            member_sha256=(
                "06921fe83fc66c907e6c7b86b82255dc70458ee7d76258fc48297cb34f0c06b5"
            ),
            rows=142_125,
            positive_rows=2_948,
            weighted_positive_share=0.023247141039884213,
            weighted_total=83_164_893_924.73,
        ),
    )
}

ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS: tuple[int, ...] = tuple(
    sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES)
)


def _sha256_stream(stream: BinaryIO, *, chunk_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _verify_file(
    path: Path,
    *,
    label: str,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
    chunk_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise ValueError(
            f"{label} byte length mismatch: expected {expected_size_bytes}, got {size}."
        )
    if expected_sha256 is not None:
        with path.open("rb") as stream:
            digest, _ = _sha256_stream(stream, chunk_size=chunk_size)
        if digest != expected_sha256:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}."
            )


def _verified_member_info(
    archive_file: zipfile.ZipFile,
    pins: AsecEducationArchive,
) -> zipfile.ZipInfo:
    members = [info for info in archive_file.infolist() if info.filename == pins.member]
    if len(members) != 1:
        raise ValueError(
            f"ASEC {pins.survey_year} archive must contain exactly one "
            f"{pins.member!r} member; found {len(members)}."
        )
    info = members[0]
    if info.file_size != pins.member_size_bytes:
        raise ValueError(
            f"ASEC {pins.survey_year} person member byte length mismatch: "
            f"expected {pins.member_size_bytes}, got {info.file_size}."
        )
    actual_crc = f"{info.CRC:08x}"
    if actual_crc != pins.member_crc32:
        raise ValueError(
            f"ASEC {pins.survey_year} person member CRC32 mismatch: expected "
            f"{pins.member_crc32}, got {actual_crc}."
        )
    return info


def fetch_asec_education_assistance_source(
    income_year: int,
    cache_dir: str | Path | None = None,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download, verify, extract, and cache one income year's person member."""

    if income_year not in ASEC_EDUCATION_ASSISTANCE_ARCHIVES:
        raise ValueError(
            f"No pinned ASEC education-assistance archive covers income year "
            f"{income_year}; pinned income years: "
            f"{list(ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS)}."
        )
    pins = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[income_year]
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "populace" / "cps" / "asec_education"
    )
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"asecpub{str(pins.survey_year)[2:]}csv.zip"
    member_path = root / pins.member

    if not archive_path.exists():
        temporary = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(
            pins.zip_url,
            headers={"User-Agent": "populace-build/asec-education-assistance"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                with temporary.open("wb") as destination:
                    digest = hashlib.sha256()
                    size = 0
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size != pins.zip_size_bytes:
                raise ValueError(
                    f"ASEC {pins.survey_year} archive download byte length "
                    f"mismatch: expected {pins.zip_size_bytes}, got {size}."
                )
            actual_digest = digest.hexdigest()
            if actual_digest != pins.zip_sha256:
                raise ValueError(
                    f"ASEC {pins.survey_year} archive download SHA-256 mismatch: "
                    f"expected {pins.zip_sha256}, got {actual_digest}."
                )
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)

    _verify_file(
        archive_path,
        label=f"ASEC {pins.survey_year} archive",
        expected_size_bytes=pins.zip_size_bytes,
        expected_sha256=pins.zip_sha256,
        chunk_size=chunk_size,
    )
    with zipfile.ZipFile(archive_path) as archive_file:
        info = _verified_member_info(archive_file, pins)
        if member_path.exists():
            _verify_file(
                member_path,
                label=f"ASEC {pins.survey_year} person member",
                expected_size_bytes=pins.member_size_bytes,
                expected_sha256=pins.member_sha256,
                chunk_size=chunk_size,
            )
            return member_path
        temporary = member_path.with_suffix(".csv.part")
        try:
            with archive_file.open(info) as source, temporary.open("wb") as out:
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: source.read(chunk_size), b""):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != pins.member_size_bytes:
                raise ValueError(
                    f"ASEC {pins.survey_year} extracted person member byte "
                    f"length mismatch: expected {pins.member_size_bytes}, "
                    f"got {size}."
                )
            actual_digest = digest.hexdigest()
            if actual_digest != pins.member_sha256:
                raise ValueError(
                    f"ASEC {pins.survey_year} extracted person member SHA-256 "
                    f"mismatch: expected {pins.member_sha256}, got {actual_digest}."
                )
            os.replace(temporary, member_path)
        finally:
            temporary.unlink(missing_ok=True)
    return member_path


def _fixed_width_peridnum(values: pd.Series, *, label: str) -> pd.Series:
    if values.isna().any():
        rows = values.index[values.isna()].tolist()[:5]
        raise ValueError(f"{label} PERIDNUM is missing at row(s): {rows}.")
    decoded = values.map(
        lambda value: (
            value.decode()
            if isinstance(value, (bytes, bytearray, np.bytes_))
            else value
        )
    ).astype(str)
    valid = decoded.str.fullmatch(r"[0-9]{22}", na=False)
    if not valid.all():
        rows = decoded.index[~valid].tolist()[:5]
        raise ValueError(
            f"{label} PERIDNUM must be an exact 22-digit string at row(s): {rows}."
        )
    return decoded


def _load_one_source(
    path: Path, pins: AsecEducationArchive, chunk_size: int
) -> pd.DataFrame:
    usecols = [*ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS, _AUDIT_WEIGHT_COLUMN]
    if zipfile.is_zipfile(path):
        _verify_file(
            path,
            label=f"ASEC {pins.survey_year} archive",
            expected_size_bytes=pins.zip_size_bytes,
            expected_sha256=pins.zip_sha256,
            chunk_size=chunk_size,
        )
        with zipfile.ZipFile(path) as archive_file:
            info = _verified_member_info(archive_file, pins)
            with archive_file.open(info) as member:
                digest, size = _sha256_stream(member, chunk_size=chunk_size)
            if size != pins.member_size_bytes or digest != pins.member_sha256:
                raise ValueError(
                    f"ASEC {pins.survey_year} person member identity mismatch."
                )
            with archive_file.open(info) as member:
                return pd.read_csv(
                    member,
                    usecols=usecols,
                    dtype={"PERIDNUM": "string"},
                    low_memory=False,
                )
    _verify_file(
        path,
        label=f"ASEC {pins.survey_year} person member",
        expected_size_bytes=pins.member_size_bytes,
        expected_sha256=pins.member_sha256,
        chunk_size=chunk_size,
    )
    return pd.read_csv(
        path, usecols=usecols, dtype={"PERIDNUM": "string"}, low_memory=False
    )


def load_asec_education_assistance_sources(
    paths: Mapping[int, str | Path] | None = None,
    *,
    income_years: tuple[int, ...] = ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS,
    chunk_size: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Load and pin-verify the pooled education-assistance sidecar.

    ``paths`` maps INCOME years (the ``--asec-h5`` vocabulary) to local copies
    of the pinned survey archives (zip or extracted member); any income year
    without a path is fetched from the official Census archive and verified
    against the same pins.
    """

    unknown = sorted(set(paths or ()) - set(ASEC_EDUCATION_ASSISTANCE_ARCHIVES))
    if unknown:
        raise ValueError(
            f"No pinned ASEC education-assistance archive covers income "
            f"year(s) {unknown}; pinned income years: "
            f"{list(ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS)}."
        )
    if not income_years:
        empty = pd.DataFrame(
            columns=["source_year", *ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS]
        )
        empty.attrs["source_audit"] = {}
        return empty
    parts: list[pd.DataFrame] = []
    audits: dict[int, dict[str, float | int]] = {}
    for income_year in income_years:
        pins = ASEC_EDUCATION_ASSISTANCE_ARCHIVES.get(income_year)
        if pins is None:
            raise ValueError(
                f"No pinned ASEC education-assistance archive covers income "
                f"year {income_year}."
            )
        provided = None if paths is None else paths.get(income_year)
        path = (
            Path(provided).expanduser()
            if provided is not None
            else fetch_asec_education_assistance_source(income_year)
        )
        raw = _load_one_source(path, pins, chunk_size)
        missing = sorted(
            {*ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS, _AUDIT_WEIGHT_COLUMN}
            - set(raw.columns)
        )
        if missing:
            raise ValueError(
                f"ASEC {pins.survey_year} education source missing column(s): "
                f"{missing}."
            )
        if len(raw) != pins.rows:
            raise ValueError(
                f"ASEC {pins.survey_year} education source row count mismatch: "
                f"expected {pins.rows}, got {len(raw)}."
            )
        raw["PERIDNUM"] = _fixed_width_peridnum(
            raw["PERIDNUM"], label=f"ASEC {pins.survey_year} source"
        )
        if raw["PERIDNUM"].duplicated(keep=False).any():
            raise ValueError(
                f"ASEC {pins.survey_year} education source PERIDNUM must be unique."
            )
        values = pd.to_numeric(raw[_SOURCE], errors="coerce").to_numpy(dtype=np.float64)
        if not (np.isfinite(values) & (values >= 0.0)).all():
            rows = np.flatnonzero(~(np.isfinite(values) & (values >= 0.0)))[:5].tolist()
            raise ValueError(
                f"ASEC {pins.survey_year} ED_VAL must be finite and nonnegative "
                f"at row(s): {rows}."
            )
        weights = pd.to_numeric(raw[_AUDIT_WEIGHT_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if (
            not np.isfinite(weights).all()
            or (weights < 0.0).any()
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError(
                f"ASEC {pins.survey_year} A_FNLWGT must be finite and "
                "nonnegative with positive total mass."
            )
        positive = values > 0.0
        scaled = weights / 100.0
        audit = {
            "rows": int(len(raw)),
            "positive_rows": int(np.count_nonzero(positive)),
            "weighted_positive_share": float(scaled[positive].sum() / scaled.sum()),
            "weighted_total": float(np.dot(scaled, values)),
        }
        pinned_input = provided is None or (
            len(raw) == pins.rows
            and Path(path).stat().st_size
            in (pins.zip_size_bytes, pins.member_size_bytes)
        )
        if pinned_input:
            if audit["positive_rows"] != pins.positive_rows:
                raise ValueError(
                    f"ASEC {pins.survey_year} education audit drifted: expected "
                    f"{pins.positive_rows} positive rows, got "
                    f"{audit['positive_rows']}."
                )
            if not np.isclose(
                audit["weighted_positive_share"],
                pins.weighted_positive_share,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"ASEC {pins.survey_year} education audit drifted for "
                    "weighted_positive_share."
                )
            if not np.isclose(
                audit["weighted_total"], pins.weighted_total, rtol=0.0, atol=1.0
            ):
                raise ValueError(
                    f"ASEC {pins.survey_year} education audit drifted for "
                    "weighted_total."
                )
        audits[income_year] = audit
        part = raw.loc[:, list(ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS)].copy()
        part.insert(0, "source_year", np.int64(income_year))
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result.attrs["source_audit"] = audits
    return result


def fill_asec_education_assistance_source(
    person: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Fill ``ED_VAL`` for every pooled person via exact Census identity."""

    required_person = ("source_year", "PERIDNUM")
    missing_person = [column for column in required_person if column not in person]
    if missing_person:
        raise ValueError(
            "ASEC education-assistance repair requires person column(s): "
            f"{missing_person}."
        )
    required_source = ("source_year", *ASEC_EDUCATION_ASSISTANCE_SOURCE_COLUMNS)
    missing_source = [column for column in required_source if column not in source]
    if missing_source:
        raise ValueError(
            f"ASEC education-assistance sidecar missing column(s): {missing_source}."
        )
    result = person.copy(deep=True)
    person_years = pd.to_numeric(result["source_year"], errors="coerce")
    if person_years.isna().any() or (person_years != np.floor(person_years)).any():
        raise ValueError("ASEC education-assistance person source_year is invalid.")
    needed_years = sorted(int(year) for year in person_years.unique())
    covered = set(
        pd.to_numeric(source["source_year"], errors="coerce").astype(int).unique()
    )
    uncovered = [year for year in needed_years if year not in covered]
    if uncovered:
        raise ValueError(
            "ASEC education-assistance sidecar does not cover pooled income "
            f"year(s): {uncovered}."
        )

    donor = source.copy(deep=True)
    donor["PERIDNUM"] = _fixed_width_peridnum(
        donor["PERIDNUM"], label="ASEC education sidecar"
    )
    if _SOURCE in result.columns:
        existing = pd.to_numeric(result[_SOURCE], errors="coerce")
        if existing.notna().any():
            raise ValueError(
                "ASEC education-assistance fill found a preexisting ED_VAL "
                "column with values; refusing to overwrite measured data."
            )
    result[_SOURCE] = np.nan

    for year in needed_years:
        year_mask = person_years.eq(year).to_numpy()
        year_donor = donor.loc[
            pd.to_numeric(donor["source_year"], errors="coerce").eq(year)
        ].set_index("PERIDNUM")
        keys = _fixed_width_peridnum(
            result.loc[year_mask, "PERIDNUM"], label="ASEC education frame"
        )
        missing_keys = keys[~keys.isin(year_donor.index)].drop_duplicates()
        if not missing_keys.empty:
            raise ValueError(
                "ASEC education-assistance sidecar does not cover frame "
                f"PERIDNUM key(s) for income year {year}: "
                f"{missing_keys.tolist()[:5]}."
            )
        aligned = year_donor.reindex(keys.to_numpy())
        aligned.index = result.index[year_mask]
        identity_pairs = [
            (
                "PH_SEQ",
                "source_household_id" if "source_household_id" in result else "PH_SEQ",
            ),
            ("P_SEQ", "P_SEQ"),
            ("A_LINENO", "A_LINENO"),
        ]
        for donor_column, frame_column in identity_pairs:
            if frame_column not in result:
                continue
            observed = pd.to_numeric(
                result.loc[year_mask, frame_column], errors="coerce"
            ).to_numpy(dtype=np.float64)
            expected = pd.to_numeric(aligned[donor_column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            mismatch = (
                ~np.isfinite(observed) | ~np.isfinite(expected) | (observed != expected)
            )
            if mismatch.any():
                rows = result.index[year_mask].to_numpy()[mismatch][:5].tolist()
                raise ValueError(
                    "ASEC education-assistance redundant identity mismatch for "
                    f"{frame_column} against sidecar {donor_column} in income "
                    f"year {year} at row(s): {rows}."
                )
        values = pd.to_numeric(aligned[_SOURCE], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not (np.isfinite(values) & (values >= 0.0)).all():
            raise ValueError(
                f"ASEC education-assistance sidecar ED_VAL is invalid for "
                f"income year {year}."
            )
        result.loc[result.index[year_mask], _SOURCE] = values
    result.attrs["education_assistance_source_audit"] = source.attrs.get(
        "source_audit", {}
    )
    return result
