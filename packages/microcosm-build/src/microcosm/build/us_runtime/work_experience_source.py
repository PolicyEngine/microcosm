"""Pinned ASEC work-experience industry recodes for the pooled person years.

The frozen census_cps person inputs never carried the ASEC work-experience
industry recodes, so this sidecar restores ``WEIND`` (industry of longest job
by detailed groups, 0--23) and ``WEMIND`` (industry of longest job by major
industry groups, 0--15) for every pooled income year from the official,
immutable ASEC public-use archives.  The restore is an exact per-income-year
``PERIDNUM`` join; it never predicts an ASEC source value.

The loader also enforces the official universe identity of the
work-experience recode block against each archive's own ``WKSWORK`` and
``WORKYN`` columns: ``WEIND`` is a worker code (1--22) exactly where
``WKSWORK > 0``, ``WEIND`` and ``WEMIND`` are zero on exactly the same rows,
and ``WORKYN = 1`` never appears without positive ``WKSWORK``.  Those audit
columns are consumed at load time only and are not part of the sidecar
payload.
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
    "ASEC_WORK_EXPERIENCE_ARCHIVES",
    "ASEC_WORK_EXPERIENCE_INCOME_YEARS",
    "ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS",
    "WORK_EXPERIENCE_OFFICIAL_DICTIONARY_URL",
    "AsecWorkExperienceArchive",
    "fetch_asec_work_experience_source",
    "fill_asec_work_experience_source",
    "load_asec_work_experience_sources",
]

#: Official data dictionary naming the carried entries: person WEIND
#: ("IND. OF LONGEST JOB BY DETAILED GROUPS", position 326 length 2) and
#: WEMIND ("IND. OF LONGEST JOB BY MAJOR IND. GROUPS", position 329 length
#: 2), both with universe "All Persons aged 15+".
WORK_EXPERIENCE_OFFICIAL_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/"
    "asec2024_ddl_pub_full.pdf"
)

_DETAILED_SOURCE = "WEIND"
_MAJOR_SOURCE = "WEMIND"
ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "PERIDNUM",
    _DETAILED_SOURCE,
    _MAJOR_SOURCE,
)
_AUDIT_WEIGHT_COLUMN = "A_FNLWGT"
_AUDIT_WEEKS_COLUMN = "WKSWORK"
_AUDIT_WORKED_COLUMN = "WORKYN"
_AUDIT_COLUMNS: tuple[str, ...] = (
    _AUDIT_WEIGHT_COLUMN,
    _AUDIT_WEEKS_COLUMN,
    _AUDIT_WORKED_COLUMN,
)
_DETAILED_MAX = 23
_DETAILED_WORKER_MAX = 22
_MAJOR_MAX = 15


@dataclass(frozen=True)
class AsecWorkExperienceArchive:
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
    worked_rows: int
    weighted_worked_share: float
    recode_rows: int
    weighted_recode_share: float


#: One pinned archive per pooled income year. The survey-year file published
#: the March after each income year carries that income year's person
#: universe: row counts equal the pooled cohorts exactly and PERIDNUM
#: coverage is 100.0% per year (and only ~33% against any adjacent survey
#: year, the CPS rotation-group overlap — pinning the wrong vintage fails the
#: full-coverage join loudly).  The zip and member identities equal the
#: education-assistance pins of the same archives; the audit statistics are
#: this sidecar's own work-experience measurements.
ASEC_WORK_EXPERIENCE_ARCHIVES: dict[int, AsecWorkExperienceArchive] = {
    archive.income_year: archive
    for archive in (
        AsecWorkExperienceArchive(
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
            worked_rows=73_186,
            weighted_worked_share=0.519967970757664,
            recode_rows=116_650,
            weighted_recode_share=0.821736897883986,
        ),
        AsecWorkExperienceArchive(
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
            worked_rows=73_471,
            weighted_worked_share=0.523171279944173,
            recode_rows=115_836,
            weighted_recode_share=0.822002159972951,
        ),
        AsecWorkExperienceArchive(
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
            worked_rows=72_460,
            weighted_worked_share=0.522793674686251,
            recode_rows=114_446,
            weighted_recode_share=0.824376426736239,
        ),
    )
}

ASEC_WORK_EXPERIENCE_INCOME_YEARS: tuple[int, ...] = tuple(
    sorted(ASEC_WORK_EXPERIENCE_ARCHIVES)
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
    pins: AsecWorkExperienceArchive,
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


def fetch_asec_work_experience_source(
    income_year: int,
    cache_dir: str | Path | None = None,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download, verify, extract, and cache one income year's person member."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if income_year not in ASEC_WORK_EXPERIENCE_ARCHIVES:
        raise ValueError(
            f"No pinned ASEC work-experience archive covers income year "
            f"{income_year}; pinned income years: "
            f"{list(ASEC_WORK_EXPERIENCE_INCOME_YEARS)}."
        )
    pins = ASEC_WORK_EXPERIENCE_ARCHIVES[income_year]
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "microcosm" / "cps" / "asec_work_experience"
    )
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"asecpub{str(pins.survey_year)[2:]}csv.zip"
    member_path = root / pins.member

    if not archive_path.exists():
        temporary = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(
            pins.zip_url,
            headers={"User-Agent": "microcosm-build/asec-work-experience"},
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


def _integer_bounded(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
    upper: int,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & (values == np.floor(values))
    valid &= (values >= 0.0) & (values <= float(upper))
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            f"{label} {column} must be an integer in [0, {upper}] at row(s): {rows}."
        )
    return values.astype(np.int64)


def _load_one_source(
    path: Path, pins: AsecWorkExperienceArchive, chunk_size: int
) -> pd.DataFrame:
    usecols = [*ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS, *_AUDIT_COLUMNS]
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


def load_asec_work_experience_sources(
    paths: Mapping[int, str | Path] | None = None,
    *,
    income_years: tuple[int, ...] = ASEC_WORK_EXPERIENCE_INCOME_YEARS,
    chunk_size: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Load and pin-verify the pooled work-experience sidecar.

    ``paths`` maps INCOME years (the ``--asec-h5`` vocabulary) to local copies
    of the pinned survey archives (zip or extracted member); any income year
    without a path is fetched from the official Census archive and verified
    against the same pins.
    """

    unknown = sorted(set(paths or ()) - set(ASEC_WORK_EXPERIENCE_ARCHIVES))
    if unknown:
        raise ValueError(
            f"No pinned ASEC work-experience archive covers income year(s) "
            f"{unknown}; pinned income years: "
            f"{list(ASEC_WORK_EXPERIENCE_INCOME_YEARS)}."
        )
    if not income_years:
        empty = pd.DataFrame(
            columns=["source_year", *ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS]
        )
        empty.attrs["source_audit"] = {}
        return empty
    parts: list[pd.DataFrame] = []
    audits: dict[int, dict[str, float | int]] = {}
    for income_year in income_years:
        pins = ASEC_WORK_EXPERIENCE_ARCHIVES.get(income_year)
        if pins is None:
            raise ValueError(
                f"No pinned ASEC work-experience archive covers income year "
                f"{income_year}."
            )
        provided = None if paths is None else paths.get(income_year)
        path = (
            Path(provided).expanduser()
            if provided is not None
            else fetch_asec_work_experience_source(income_year)
        )
        raw = _load_one_source(path, pins, chunk_size)
        missing = sorted(
            {*ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS, *_AUDIT_COLUMNS}
            - set(raw.columns)
        )
        if missing:
            raise ValueError(
                f"ASEC {pins.survey_year} work-experience source missing "
                f"column(s): {missing}."
            )
        if len(raw) != pins.rows:
            raise ValueError(
                f"ASEC {pins.survey_year} work-experience source row count "
                f"mismatch: expected {pins.rows}, got {len(raw)}."
            )
        raw["PERIDNUM"] = _fixed_width_peridnum(
            raw["PERIDNUM"], label=f"ASEC {pins.survey_year} source"
        )
        if raw["PERIDNUM"].duplicated(keep=False).any():
            raise ValueError(
                f"ASEC {pins.survey_year} work-experience source PERIDNUM "
                "must be unique."
            )
        label = f"ASEC {pins.survey_year} work-experience source"
        detailed = _integer_bounded(
            raw, _DETAILED_SOURCE, label=label, upper=_DETAILED_MAX
        )
        major = _integer_bounded(raw, _MAJOR_SOURCE, label=label, upper=_MAJOR_MAX)
        weeks = _integer_bounded(raw, _AUDIT_WEEKS_COLUMN, label=label, upper=52)
        worked_yn = _integer_bounded(raw, _AUDIT_WORKED_COLUMN, label=label, upper=2)
        worked = weeks > 0
        worker_code = (detailed >= 1) & (detailed <= _DETAILED_WORKER_MAX)
        universe_breaks = int(np.count_nonzero(worker_code != worked))
        if universe_breaks:
            raise ValueError(
                f"{label} breaks the work-experience universe identity "
                f"(WEIND in 1..{_DETAILED_WORKER_MAX} iff WKSWORK > 0) on "
                f"{universe_breaks} row(s)."
            )
        recode_zero_breaks = int(np.count_nonzero((detailed == 0) != (major == 0)))
        if recode_zero_breaks:
            raise ValueError(
                f"{label} detailed and major recodes disagree on the "
                f"not-in-universe rows: {recode_zero_breaks} row(s)."
            )
        affirm_without_weeks = int(np.count_nonzero((worked_yn == 1) & ~worked))
        if affirm_without_weeks:
            raise ValueError(
                f"{label} reports WORKYN = 1 without positive WKSWORK on "
                f"{affirm_without_weeks} row(s)."
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
        scaled = weights / 100.0
        recode_positive = detailed != 0
        audit = {
            "rows": int(len(raw)),
            "worked_rows": int(np.count_nonzero(worked)),
            "weighted_worked_share": float(scaled[worked].sum() / scaled.sum()),
            "recode_rows": int(np.count_nonzero(recode_positive)),
            "weighted_recode_share": float(
                scaled[recode_positive].sum() / scaled.sum()
            ),
        }
        pinned_input = provided is None or (
            len(raw) == pins.rows
            and Path(path).stat().st_size
            in (pins.zip_size_bytes, pins.member_size_bytes)
        )
        if pinned_input:
            if audit["worked_rows"] != pins.worked_rows:
                raise ValueError(
                    f"ASEC {pins.survey_year} work-experience audit drifted: "
                    f"expected {pins.worked_rows} worked rows, got "
                    f"{audit['worked_rows']}."
                )
            if audit["recode_rows"] != pins.recode_rows:
                raise ValueError(
                    f"ASEC {pins.survey_year} work-experience audit drifted: "
                    f"expected {pins.recode_rows} recode rows, got "
                    f"{audit['recode_rows']}."
                )
            for key, expected in (
                ("weighted_worked_share", pins.weighted_worked_share),
                ("weighted_recode_share", pins.weighted_recode_share),
            ):
                if not np.isclose(audit[key], expected, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"ASEC {pins.survey_year} work-experience audit "
                        f"drifted for {key}."
                    )
        audits[income_year] = audit
        part = raw.loc[:, list(ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS)].copy()
        part.insert(0, "source_year", np.int64(income_year))
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result.attrs["source_audit"] = audits
    return result


def fill_asec_work_experience_source(
    person: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Fill ``WEIND``/``WEMIND`` for every pooled person via Census identity."""

    required_person = ("source_year", "PERIDNUM")
    missing_person = [column for column in required_person if column not in person]
    if missing_person:
        raise ValueError(
            "ASEC work-experience repair requires person column(s): "
            f"{missing_person}."
        )
    required_source = ("source_year", *ASEC_WORK_EXPERIENCE_SOURCE_COLUMNS)
    missing_source = [column for column in required_source if column not in source]
    if missing_source:
        raise ValueError(
            f"ASEC work-experience sidecar missing column(s): {missing_source}."
        )
    result = person.copy(deep=True)
    person_years = pd.to_numeric(result["source_year"], errors="coerce")
    if person_years.isna().any() or (person_years != np.floor(person_years)).any():
        raise ValueError("ASEC work-experience person source_year is invalid.")
    needed_years = sorted(int(year) for year in person_years.unique())
    covered = set(
        pd.to_numeric(source["source_year"], errors="coerce").astype(int).unique()
    )
    uncovered = [year for year in needed_years if year not in covered]
    if uncovered:
        raise ValueError(
            "ASEC work-experience sidecar does not cover pooled income "
            f"year(s): {uncovered}."
        )

    donor = source.copy(deep=True)
    donor["PERIDNUM"] = _fixed_width_peridnum(
        donor["PERIDNUM"], label="ASEC work-experience sidecar"
    )
    for column in (_DETAILED_SOURCE, _MAJOR_SOURCE):
        if column in result.columns:
            existing = pd.to_numeric(result[column], errors="coerce")
            if existing.notna().any():
                raise ValueError(
                    f"ASEC work-experience repair must not overwrite an "
                    f"existing {column} surface."
                )
            result = result.drop(columns=[column])
    person_identity = _fixed_width_peridnum(
        result["PERIDNUM"], label="ASEC work-experience person"
    )
    keys = pd.MultiIndex.from_arrays(
        [person_years.astype(np.int64), person_identity],
        names=("source_year", "PERIDNUM"),
    )
    donor_index = pd.MultiIndex.from_arrays(
        [
            pd.to_numeric(donor["source_year"], errors="coerce").astype(np.int64),
            donor["PERIDNUM"],
        ],
        names=("source_year", "PERIDNUM"),
    )
    if donor_index.duplicated().any():
        raise ValueError(
            "ASEC work-experience sidecar (source_year, PERIDNUM) keys must "
            "be unique."
        )
    lookup = donor.set_index(donor_index)
    for column in (_DETAILED_SOURCE, _MAJOR_SOURCE):
        joined = lookup[column].reindex(keys)
        if joined.isna().any():
            missing_rows = int(joined.isna().sum())
            raise ValueError(
                f"ASEC work-experience sidecar does not cover {missing_rows} "
                f"pooled person(s) for {column}; the exact Census identity "
                "join must be total."
            )
        result[column] = joined.to_numpy(dtype=np.int64)
    return result
