"""Measured CPS ASEC public-assistance type for the TANF enrollment gate.

``PAW_VAL`` alone cannot identify TANF: the ASEC public-assistance amount
covers TANF/AFDC *and* other cash welfare (general assistance), and the
official dictionaries split the program with ``PAW_TYP`` (1 = TANF/AFDC,
2 = other, 3 = both).  In the 2023 ASEC (income year 2022), 271 of 682
PAW-positive SPM units — 39.7% unweighted, 42.6% SPM-weighted — have no
member reporting a TANF type, so gating ``is_tanf_enrolled`` on
``PAW_VAL > 0`` alone would mark non-TANF cash welfare as TANF enrollment.

The frozen ``census_cps_*.h5`` inputs microcosm builds from never carried
``PAW_TYP``, so this module restores it from the official, immutable Census
ASEC public-use archives — the same repair class as
:mod:`.education_assistance_source`'s ``ED_VAL`` sidecar, sharing that
module's per-income-year archive pins because both columns live in the same
pinned survey-year person members.  The repair is an exact per-income-year
``PERIDNUM`` join with redundant Census identity checks; it never predicts
a source value.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
    AsecEducationArchive,
    fetch_asec_education_assistance_source,
)

__all__ = [
    "ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS",
    "ASEC_PUBLIC_ASSISTANCE_TYPE_INCOME_YEARS",
    "ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS",
    "PAW_TYPE_TANF_CODES",
    "PAW_TYPE_VALID_CODES",
    "AsecPublicAssistanceTypeAudit",
    "fill_asec_public_assistance_type_source",
    "load_asec_public_assistance_type_sources",
]

_SOURCE = "PAW_TYP"
_AUDIT_WEIGHT_COLUMN = "A_FNLWGT"
_AMOUNT_COLUMN = "PAW_VAL"

#: Official ASEC ``PAW_TYP`` codes: 0 not in universe, 1 TANF/AFDC,
#: 2 other cash welfare, 3 both.
PAW_TYPE_VALID_CODES = frozenset({0, 1, 2, 3})

#: ``PAW_TYP`` codes that identify TANF/AFDC receipt (1 = TANF/AFDC,
#: 3 = both TANF/AFDC and other assistance).
PAW_TYPE_TANF_CODES = frozenset({1, 3})

ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "PERIDNUM",
    _AMOUNT_COLUMN,
    _SOURCE,
)


@dataclass(frozen=True)
class AsecPublicAssistanceTypeAudit:
    """Exact measured ``PAW_TYP`` composition of one pinned person archive."""

    income_year: int
    rows: int
    #: Full-column ``PAW_TYP`` counts for codes 0..3, in code order.
    paw_type_counts: tuple[int, int, int, int]
    paw_positive_rows: int
    paw_positive_tanf_rows: int


#: One exact audit pin per pooled income year, measured from the archives
#: pinned in :data:`ASEC_EDUCATION_ASSISTANCE_ARCHIVES`.  All values are
#: integers, so drift detection is exact.
ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS: dict[int, AsecPublicAssistanceTypeAudit] = {
    audit.income_year: audit
    for audit in (
        AsecPublicAssistanceTypeAudit(
            income_year=2022,
            rows=146_133,
            paw_type_counts=(145_377, 428, 306, 22),
            paw_positive_rows=756,
            paw_positive_tanf_rows=450,
        ),
        AsecPublicAssistanceTypeAudit(
            income_year=2023,
            rows=144_265,
            paw_type_counts=(143_555, 398, 298, 14),
            paw_positive_rows=710,
            paw_positive_tanf_rows=412,
        ),
        AsecPublicAssistanceTypeAudit(
            income_year=2024,
            rows=142_125,
            paw_type_counts=(141_415, 406, 289, 15),
            paw_positive_rows=710,
            paw_positive_tanf_rows=421,
        ),
    )
}

ASEC_PUBLIC_ASSISTANCE_TYPE_INCOME_YEARS: tuple[int, ...] = tuple(
    sorted(ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS)
)


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
    expected_size_bytes: int,
    expected_sha256: str,
    chunk_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_size_bytes:
        raise ValueError(
            f"{label} byte length mismatch: expected {expected_size_bytes}, got {size}."
        )
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


def _load_one_source(
    path: Path, pins: AsecEducationArchive, chunk_size: int = 8 * 1024 * 1024
) -> pd.DataFrame:
    usecols = [
        *ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS,
        _AUDIT_WEIGHT_COLUMN,
    ]
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


def load_asec_public_assistance_type_sources(
    paths: Mapping[int, str | Path] | None = None,
    *,
    income_years: tuple[int, ...] = ASEC_PUBLIC_ASSISTANCE_TYPE_INCOME_YEARS,
) -> pd.DataFrame:
    """Load and audit-verify the pooled public-assistance-type sidecar.

    ``paths`` maps INCOME years (the ``--asec-h5`` vocabulary) to local
    copies of the pinned survey archives (zip or extracted member); any
    income year without a path is fetched from the official Census archive
    and verified against the shared education-sidecar pins.  Every loaded
    year must reproduce its exact pinned ``PAW_TYP`` composition.
    """

    unknown = sorted(set(paths or ()) - set(ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS))
    if unknown:
        raise ValueError(
            f"No pinned ASEC public-assistance-type audit covers income "
            f"year(s) {unknown}; pinned income years: "
            f"{list(ASEC_PUBLIC_ASSISTANCE_TYPE_INCOME_YEARS)}."
        )
    if not income_years:
        empty = pd.DataFrame(
            columns=["source_year", *ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS]
        )
        empty.attrs["source_audit"] = {}
        return empty
    parts: list[pd.DataFrame] = []
    audits: dict[int, dict[str, object]] = {}
    for income_year in income_years:
        audit_pins = ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS.get(income_year)
        archive_pins = ASEC_EDUCATION_ASSISTANCE_ARCHIVES.get(income_year)
        if audit_pins is None or archive_pins is None:
            raise ValueError(
                f"No pinned ASEC public-assistance-type source covers income "
                f"year {income_year}."
            )
        provided = None if paths is None else paths.get(income_year)
        path = (
            Path(provided).expanduser()
            if provided is not None
            else fetch_asec_education_assistance_source(income_year)
        )
        raw = _load_one_source(path, archive_pins)
        missing = sorted(
            {*ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS, _AUDIT_WEIGHT_COLUMN}
            - set(raw.columns)
        )
        if missing:
            raise ValueError(
                f"ASEC {archive_pins.survey_year} public-assistance-type "
                f"source missing column(s): {missing}."
            )
        if len(raw) != audit_pins.rows:
            raise ValueError(
                f"ASEC {archive_pins.survey_year} public-assistance-type "
                f"source row count mismatch: expected {audit_pins.rows}, "
                f"got {len(raw)}."
            )
        raw["PERIDNUM"] = _fixed_width_peridnum(
            raw["PERIDNUM"], label=f"ASEC {archive_pins.survey_year} source"
        )
        if raw["PERIDNUM"].duplicated(keep=False).any():
            raise ValueError(
                f"ASEC {archive_pins.survey_year} public-assistance-type "
                "source PERIDNUM must be unique."
            )
        types = pd.to_numeric(raw[_SOURCE], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(types) & np.isin(types, sorted(PAW_TYPE_VALID_CODES))
        if not valid.all():
            rows = np.flatnonzero(~valid)[:5].tolist()
            raise ValueError(
                f"ASEC {archive_pins.survey_year} PAW_TYP must be one of "
                f"{sorted(PAW_TYPE_VALID_CODES)} at row(s): {rows}."
            )
        amounts = pd.to_numeric(raw[_AMOUNT_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not (np.isfinite(amounts) & (amounts >= 0.0)).all():
            rows = np.flatnonzero(~(np.isfinite(amounts) & (amounts >= 0.0)))[
                :5
            ].tolist()
            raise ValueError(
                f"ASEC {archive_pins.survey_year} PAW_VAL must be finite and "
                f"nonnegative at row(s): {rows}."
            )
        type_counts = tuple(int(np.count_nonzero(types == code)) for code in range(4))
        positive = amounts > 0.0
        tanf = positive & np.isin(types, sorted(PAW_TYPE_TANF_CODES))
        audit = {
            "rows": int(len(raw)),
            "paw_type_counts": list(type_counts),
            "paw_positive_rows": int(np.count_nonzero(positive)),
            "paw_positive_tanf_rows": int(np.count_nonzero(tanf)),
        }
        if type_counts != audit_pins.paw_type_counts:
            raise ValueError(
                f"ASEC {archive_pins.survey_year} public-assistance-type audit "
                f"drifted: expected PAW_TYP counts "
                f"{list(audit_pins.paw_type_counts)}, got {list(type_counts)}."
            )
        if audit["paw_positive_rows"] != audit_pins.paw_positive_rows:
            raise ValueError(
                f"ASEC {archive_pins.survey_year} public-assistance-type audit "
                f"drifted: expected {audit_pins.paw_positive_rows} PAW-positive "
                f"rows, got {audit['paw_positive_rows']}."
            )
        if audit["paw_positive_tanf_rows"] != audit_pins.paw_positive_tanf_rows:
            raise ValueError(
                f"ASEC {archive_pins.survey_year} public-assistance-type audit "
                f"drifted: expected {audit_pins.paw_positive_tanf_rows} "
                f"TANF-typed PAW-positive rows, got "
                f"{audit['paw_positive_tanf_rows']}."
            )
        audits[income_year] = audit
        part = raw.loc[:, list(ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS)].copy()
        part[_SOURCE] = pd.to_numeric(part[_SOURCE], errors="raise").astype("int64")
        part.insert(0, "source_year", np.int64(income_year))
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result.attrs["source_audit"] = audits
    return result


def fill_asec_public_assistance_type_source(
    person: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Fill ``PAW_TYP`` for every pooled person via exact Census identity.

    The join mirrors :func:`fill_asec_education_assistance_source`: an exact
    per-income-year ``PERIDNUM`` join with redundant household/line identity
    checks.  The restored column is ``int64``, matching the frozen inputs'
    ``PAW_VAL`` treatment.
    """

    required_person = ("source_year", "PERIDNUM")
    missing_person = [column for column in required_person if column not in person]
    if missing_person:
        raise ValueError(
            "ASEC public-assistance-type repair requires person column(s): "
            f"{missing_person}."
        )
    required_source = (
        "source_year",
        *ASEC_PUBLIC_ASSISTANCE_TYPE_SOURCE_COLUMNS,
    )
    missing_source = [column for column in required_source if column not in source]
    if missing_source:
        raise ValueError(
            f"ASEC public-assistance-type sidecar missing column(s): {missing_source}."
        )
    result = person.copy(deep=True)
    person_years = pd.to_numeric(result["source_year"], errors="coerce")
    if person_years.isna().any() or (person_years != np.floor(person_years)).any():
        raise ValueError("ASEC public-assistance-type person source_year is invalid.")
    needed_years = sorted(int(year) for year in person_years.unique())
    covered = set(
        pd.to_numeric(source["source_year"], errors="coerce").astype(int).unique()
    )
    uncovered = [year for year in needed_years if year not in covered]
    if uncovered:
        raise ValueError(
            "ASEC public-assistance-type sidecar does not cover pooled income "
            f"year(s): {uncovered}."
        )

    donor = source.copy(deep=True)
    donor["PERIDNUM"] = _fixed_width_peridnum(
        donor["PERIDNUM"], label="ASEC public-assistance-type sidecar"
    )
    if _SOURCE in result.columns:
        existing = pd.to_numeric(result[_SOURCE], errors="coerce")
        if existing.notna().any():
            raise ValueError(
                "ASEC public-assistance-type fill found a preexisting PAW_TYP "
                "column with values; refusing to overwrite measured data."
            )
    filled = np.zeros(len(result), dtype=np.int64)

    for year in needed_years:
        year_mask = person_years.eq(year).to_numpy()
        year_donor = donor.loc[
            pd.to_numeric(donor["source_year"], errors="coerce").eq(year)
        ].set_index("PERIDNUM")
        keys = _fixed_width_peridnum(
            result.loc[year_mask, "PERIDNUM"],
            label="ASEC public-assistance-type frame",
        )
        missing_keys = keys[~keys.isin(year_donor.index)].drop_duplicates()
        if not missing_keys.empty:
            raise ValueError(
                "ASEC public-assistance-type sidecar does not cover frame "
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
                    "ASEC public-assistance-type redundant identity mismatch "
                    f"for {frame_column} against sidecar {donor_column} in "
                    f"income year {year} at row(s): {rows}."
                )
        values = pd.to_numeric(aligned[_SOURCE], errors="coerce").to_numpy(
            dtype=np.float64
        )
        valid = np.isfinite(values) & np.isin(values, sorted(PAW_TYPE_VALID_CODES))
        if not valid.all():
            raise ValueError(
                f"ASEC public-assistance-type sidecar PAW_TYP is invalid for "
                f"income year {year}."
            )
        filled[year_mask] = values.astype(np.int64)
    result[_SOURCE] = filled
    result.attrs["public_assistance_type_source_audit"] = source.attrs.get(
        "source_audit", {}
    )
    return result
