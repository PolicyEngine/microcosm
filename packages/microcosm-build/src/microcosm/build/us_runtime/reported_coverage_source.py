"""Measured ASEC at-interview coverage recodes omitted by frozen H5 inputs.

The pooled ``census_cps`` inputs carry ``NOW_GRP`` and ``NOW_MRK`` for every
income year, but their 2022 and 2023 person tables omit seven other recodes
consumed by :func:`derive_us_cps_carried_inputs`.  Those omissions otherwise
turn measured coverage into structural false values for two pooled vintages.

This module restores the seven fields from the same SHA-pinned official ASEC
person members already used for ``ED_VAL`` and ``PAW_TYP``.  Restoration is an
exact ``(source_year, PERIDNUM)`` join with redundant Census identity checks;
it does not predict, infer, or substitute any coverage value.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
    ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS,
    AsecEducationArchive,
    fetch_asec_education_assistance_source,
)

__all__ = [
    "ASEC_REPORTED_COVERAGE_INCOME_YEARS",
    "ASEC_REPORTED_COVERAGE_RAW_COLUMNS",
    "ASEC_REPORTED_COVERAGE_SOURCE_COLUMNS",
    "fill_asec_reported_coverage_source",
    "load_asec_reported_coverage_sources",
]

# ``NOW_GRP`` and ``NOW_MRK`` are deliberately absent: they already survive
# in every frozen pooled source and remain native rather than sidecar-restored.
ASEC_REPORTED_COVERAGE_RAW_COLUMNS: tuple[str, ...] = (
    "NOW_NONM",
    "NOW_MCAID",
    "NOW_CHAMPVA",
    "NOW_MIL",
    "NOW_VACARE",
    "NOW_OTHMT",
    "NOW_IHSFLG",
)
ASEC_REPORTED_COVERAGE_SOURCE_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "PERIDNUM",
    *ASEC_REPORTED_COVERAGE_RAW_COLUMNS,
)
ASEC_REPORTED_COVERAGE_INCOME_YEARS = ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS
_AUDIT_WEIGHT_COLUMN = "A_FNLWGT"
_VALID_CODES = frozenset({1, 2})


def _sha256_stream(stream: BinaryIO, *, chunk_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _verify_member_path(
    path: Path,
    pins: AsecEducationArchive,
    *,
    chunk_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != pins.member_size_bytes:
        raise ValueError(
            f"ASEC {pins.survey_year} person member byte length mismatch: "
            f"expected {pins.member_size_bytes}, got {path.stat().st_size}."
        )
    with path.open("rb") as stream:
        digest, _ = _sha256_stream(stream, chunk_size=chunk_size)
    if digest != pins.member_sha256:
        raise ValueError(
            f"ASEC {pins.survey_year} person member SHA-256 mismatch: "
            f"expected {pins.member_sha256}, got {digest}."
        )


def _verified_zip_member(
    archive: zipfile.ZipFile,
    pins: AsecEducationArchive,
) -> zipfile.ZipInfo:
    members = [info for info in archive.infolist() if info.filename == pins.member]
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


def _read_one_source(
    path: Path,
    pins: AsecEducationArchive,
    *,
    chunk_size: int,
) -> pd.DataFrame:
    usecols = [*ASEC_REPORTED_COVERAGE_SOURCE_COLUMNS, _AUDIT_WEIGHT_COLUMN]
    if zipfile.is_zipfile(path):
        if path.stat().st_size != pins.zip_size_bytes:
            raise ValueError(
                f"ASEC {pins.survey_year} archive byte length mismatch: expected "
                f"{pins.zip_size_bytes}, got {path.stat().st_size}."
            )
        with path.open("rb") as stream:
            archive_digest, _ = _sha256_stream(stream, chunk_size=chunk_size)
        if archive_digest != pins.zip_sha256:
            raise ValueError(
                f"ASEC {pins.survey_year} archive SHA-256 mismatch: expected "
                f"{pins.zip_sha256}, got {archive_digest}."
            )
        with zipfile.ZipFile(path) as archive:
            info = _verified_zip_member(archive, pins)
            with archive.open(info) as member:
                member_digest, member_size = _sha256_stream(
                    member,
                    chunk_size=chunk_size,
                )
            if (
                member_size != pins.member_size_bytes
                or member_digest != pins.member_sha256
            ):
                raise ValueError(
                    f"ASEC {pins.survey_year} person member identity mismatch."
                )
            with archive.open(info) as member:
                try:
                    return pd.read_csv(
                        member,
                        usecols=usecols,
                        dtype={"PERIDNUM": "string"},
                        low_memory=False,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"ASEC {pins.survey_year} reported-coverage source is "
                        f"missing required column(s): {error}."
                    ) from error

    _verify_member_path(path, pins, chunk_size=chunk_size)
    try:
        return pd.read_csv(
            path,
            usecols=usecols,
            dtype={"PERIDNUM": "string"},
            low_memory=False,
        )
    except ValueError as error:
        raise ValueError(
            f"ASEC {pins.survey_year} reported-coverage source is missing "
            f"required column(s): {error}."
        ) from error


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
    )
    valid = decoded.map(lambda value: isinstance(value, str))
    valid &= decoded.astype("string").str.fullmatch(r"[0-9]{22}", na=False)
    if not valid.all():
        rows = decoded.index[~valid].tolist()[:5]
        raise ValueError(
            f"{label} PERIDNUM must be an exact 22-digit string at row(s): {rows}."
        )
    return decoded.astype(str)


def _validated_source_year(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.eq(np.floor(numeric))
    if not valid.all():
        rows = values.index[~valid].tolist()[:5]
        raise ValueError(f"{label} source_year is invalid at row(s): {rows}.")
    return numeric.astype(np.int64)


def _validated_codes(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    boolean = values.map(lambda value: isinstance(value, (bool, np.bool_))).to_numpy()
    valid = np.isfinite(numeric) & np.isin(numeric, sorted(_VALID_CODES)) & ~boolean
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise ValueError(
            f"{label} must be complete integers in {sorted(_VALID_CODES)} at "
            f"row(s): {rows}."
        )
    return numeric.astype(np.int64)


def load_asec_reported_coverage_sources(
    paths: Mapping[int, str | Path] | None = None,
    *,
    income_years: tuple[int, ...] = ASEC_REPORTED_COVERAGE_INCOME_YEARS,
    chunk_size: int = 8 * 1024 * 1024,
) -> pd.DataFrame:
    """Load the seven measured coverage recodes from pinned ASEC members."""

    unknown = sorted(set(paths or ()) - set(ASEC_EDUCATION_ASSISTANCE_ARCHIVES))
    if unknown:
        raise ValueError(
            "No pinned ASEC reported-coverage archive covers income year(s) "
            f"{unknown}; pinned income years: "
            f"{list(ASEC_REPORTED_COVERAGE_INCOME_YEARS)}."
        )
    if not income_years:
        empty = pd.DataFrame(
            columns=["source_year", *ASEC_REPORTED_COVERAGE_SOURCE_COLUMNS]
        )
        empty.attrs["source_audit"] = {}
        return empty

    parts: list[pd.DataFrame] = []
    audits: dict[int, dict[str, object]] = {}
    for income_year in income_years:
        pins = ASEC_EDUCATION_ASSISTANCE_ARCHIVES.get(income_year)
        if pins is None:
            raise ValueError(
                "No pinned ASEC reported-coverage archive covers income year "
                f"{income_year}."
            )
        provided = None if paths is None else paths.get(income_year)
        path = (
            Path(provided).expanduser()
            if provided is not None
            else fetch_asec_education_assistance_source(income_year)
        )
        raw = _read_one_source(path, pins, chunk_size=chunk_size)
        if len(raw) != pins.rows:
            raise ValueError(
                f"ASEC {pins.survey_year} reported-coverage source row count "
                f"mismatch: expected {pins.rows}, got {len(raw)}."
            )
        raw["PERIDNUM"] = _fixed_width_peridnum(
            raw["PERIDNUM"],
            label=f"ASEC {pins.survey_year} reported-coverage source",
        )
        if raw["PERIDNUM"].duplicated(keep=False).any():
            raise ValueError(
                f"ASEC {pins.survey_year} reported-coverage source PERIDNUM "
                "must be unique."
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

        column_audits: dict[str, dict[str, float | int]] = {}
        scaled_weights = weights / 100.0
        for column in ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
            values = _validated_codes(
                raw[column],
                label=f"ASEC {pins.survey_year} {column}",
            )
            raw[column] = values
            yes = values == 1
            column_audits[column] = {
                "no_rows": int(np.count_nonzero(values == 2)),
                "yes_rows": int(np.count_nonzero(yes)),
                "weighted_yes_share": float(
                    scaled_weights[yes].sum() / scaled_weights.sum()
                ),
            }
        audits[income_year] = {
            "rows": int(len(raw)),
            "columns": column_audits,
        }
        part = raw.loc[:, list(ASEC_REPORTED_COVERAGE_SOURCE_COLUMNS)].copy()
        part.insert(0, "source_year", np.int64(income_year))
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    result.attrs["source_audit"] = audits
    return result


def fill_asec_reported_coverage_source(
    person: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Restore all seven recodes by exact ``(source_year, PERIDNUM)`` join."""

    required_person = ("source_year", "PERIDNUM")
    missing_person = [column for column in required_person if column not in person]
    if missing_person:
        raise ValueError(
            "ASEC reported-coverage repair requires person column(s): "
            f"{missing_person}."
        )
    required_source = ("source_year", *ASEC_REPORTED_COVERAGE_SOURCE_COLUMNS)
    missing_source = [column for column in required_source if column not in source]
    if missing_source:
        raise ValueError(
            f"ASEC reported-coverage sidecar missing column(s): {missing_source}."
        )

    donor = source.copy(deep=True)
    donor["source_year"] = _validated_source_year(
        donor["source_year"], label="ASEC reported-coverage sidecar"
    )
    donor["PERIDNUM"] = _fixed_width_peridnum(
        donor["PERIDNUM"], label="ASEC reported-coverage sidecar"
    )
    duplicate = donor.duplicated(["source_year", "PERIDNUM"], keep=False)
    if duplicate.any():
        keys = donor.loc[duplicate, ["source_year", "PERIDNUM"]].head(5)
        raise ValueError(
            "ASEC reported-coverage sidecar (source_year, PERIDNUM) keys must "
            f"be unique; duplicate key(s): {keys.to_dict(orient='records')}."
        )
    for column in ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        donor[column] = _validated_codes(
            donor[column], label=f"ASEC reported-coverage sidecar {column}"
        )

    result = person.copy(deep=True)
    person_years = _validated_source_year(
        result["source_year"], label="ASEC reported-coverage person"
    )
    result["PERIDNUM"] = _fixed_width_peridnum(
        result["PERIDNUM"], label="ASEC reported-coverage frame"
    )
    needed_years = sorted(int(year) for year in person_years.unique())
    covered_years = set(int(year) for year in donor["source_year"].unique())
    uncovered = [year for year in needed_years if year not in covered_years]
    if uncovered:
        raise ValueError(
            "ASEC reported-coverage sidecar does not cover pooled income "
            f"year(s): {uncovered}."
        )

    for year in needed_years:
        year_mask = person_years.eq(year).to_numpy()
        year_donor = donor.loc[donor["source_year"].eq(year)].set_index("PERIDNUM")
        keys = result.loc[year_mask, "PERIDNUM"]
        missing_keys = keys[~keys.isin(year_donor.index)].drop_duplicates()
        if not missing_keys.empty:
            raise ValueError(
                "ASEC reported-coverage sidecar does not cover frame PERIDNUM "
                f"key(s) for income year {year}: {missing_keys.tolist()[:5]}."
            )
        aligned = year_donor.reindex(keys.to_numpy())
        aligned.index = result.index[year_mask]

        identity_pairs = (
            (
                "PH_SEQ",
                "source_household_id" if "source_household_id" in result else "PH_SEQ",
            ),
            ("P_SEQ", "P_SEQ"),
            ("A_LINENO", "A_LINENO"),
        )
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
                    "ASEC reported-coverage redundant identity mismatch for "
                    f"{frame_column} against sidecar {donor_column} in income "
                    f"year {year} at row(s): {rows}."
                )

        for column in ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
            expected = aligned[column].to_numpy(dtype=np.int64)
            if column in result:
                current = result.loc[year_mask, column]
                present = current.notna().to_numpy()
                numeric = pd.to_numeric(current, errors="coerce").to_numpy(
                    dtype=np.float64
                )
                invalid = present & (
                    ~np.isfinite(numeric)
                    | ~np.isin(numeric, sorted(_VALID_CODES))
                    | current.map(
                        lambda value: isinstance(value, (bool, np.bool_))
                    ).to_numpy()
                )
                mismatch = present & ~invalid & (numeric != expected)
                if invalid.any() or mismatch.any():
                    bad = invalid | mismatch
                    rows = result.index[year_mask].to_numpy()[bad][:5].tolist()
                    raise ValueError(
                        f"ASEC reported-coverage existing {column} disagrees "
                        f"with the pinned sidecar in income year {year} at "
                        f"row(s): {rows}."
                    )
            else:
                result[column] = np.nan
            result.loc[year_mask, column] = expected

    for column in ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result.attrs["reported_coverage_source_audit"] = source.attrs.get(
        "source_audit", {}
    )
    return result
