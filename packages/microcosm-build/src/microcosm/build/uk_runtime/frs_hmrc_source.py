"""Source-faithful raw FRS extraction for the canonical HMRC spine stages.

This module reads raw survey tables and preserves named partial income concepts.
It does not restore or accept a pre-existing candidate population.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.spi_support import (
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
)

FRS_WEEKS_IN_YEAR = 365.25 / 7


FRS_HMRC_PAY_COLUMN = SPI_HMRC_PAY_COLUMN


FRS_HMRC_UBISJA_COLUMN = SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN


FRS_HMRC_INCPBEN_COLUMN = SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN


FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN = "ossben_identifiable_subset"


FRS_HMRC_SRP_REGULAR_CODE5_COLUMN = "srp_regular_code5"


FRS_HMRC_RETAINED_LEAF_COLUMNS = (
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
)


FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE: dict[str, dict[str, object]] = {
    FRS_HMRC_PAY_COLUMN: {
        "spi_concept": "PAY",
        "scope": "full",
        "raw_sources": ["ADULT.INEARNS"],
        "formula": "max(0, ADULT.INEARNS) * (365.25 / 7)",
    },
    FRS_HMRC_UBISJA_COLUMN: {
        "spi_concept": "UBISJA",
        "scope": "full",
        "raw_sources": [
            "BENEFITS.BENEFIT=14:BENAMT",
            "BENEFITS.BENEFIT=19:BENAMT",
        ],
        "formula": "sum(BENAMT where BENEFIT in {14, 19}) * (365.25 / 7)",
    },
    FRS_HMRC_INCPBEN_COLUMN: {
        "spi_concept": "INCPBEN",
        "scope": "full",
        "raw_sources": ["BENEFITS.BENEFIT=17:BENAMT"],
        "formula": "sum(BENAMT where BENEFIT == 17) * (365.25 / 7)",
    },
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: {
        "spi_concept": "OSSBEN",
        "scope": "identifiable_subset",
        "raw_sources": [
            "BENEFITS.BENEFIT=13:BENAMT",
            "BENEFITS.BENEFIT=16,VAR2 in {1,3}:BENAMT",
        ],
        "formula": (
            "sum(BENAMT where BENEFIT == 13 or "
            "(BENEFIT == 16 and VAR2 in {1, 3})) * (365.25 / 7)"
        ),
    },
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: {
        "spi_concept": "SRP",
        "scope": "regular_code5_subset",
        "raw_sources": ["BENEFITS.BENEFIT=5:BENAMT"],
        "formula": "sum(BENAMT where BENEFIT == 5) * (365.25 / 7)",
    },
}


@dataclass(frozen=True)
class UKFRSRawTableIdentity:
    """Stable identity and extraction surface for one raw FRS table."""

    path: Path
    filename: str
    source_vintage: str
    sha256: str
    size_bytes: int
    rows: int
    extracted_columns: tuple[str, ...]

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe source evidence."""

        return {
            "path": str(self.path),
            "filename": self.filename,
            "source_vintage": self.source_vintage,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "extracted_columns": list(self.extracted_columns),
        }


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


def _read_raw_frs_table(
    path: str | Path,
    *,
    expected_filename: str,
    required_columns: tuple[str, ...],
    source_vintage: str = "unspecified",
) -> tuple[pd.DataFrame, UKFRSRawTableIdentity]:
    source_path = Path(path).expanduser().resolve()
    if source_path.name.lower() != expected_filename:
        raise ValueError(
            f"Expected raw FRS table {expected_filename!r}, got {source_path.name!r}."
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw FRS table not found: {source_path}.")
    before = _file_fingerprint(source_path)
    digest = _sha256(source_path)
    after_hash = _file_fingerprint(source_path)
    if after_hash != before:
        raise RuntimeError(f"Raw FRS table changed while hashing: {source_path}.")
    required = set(required_columns)
    frame = pd.read_csv(
        source_path,
        sep="\t",
        usecols=lambda column: str(column).strip().lower() in required,
    )
    after_read = _file_fingerprint(source_path)
    if after_read != before:
        raise RuntimeError(f"Raw FRS table changed while reading: {source_path}.")
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(
            f"Raw FRS {expected_filename} has duplicate normalized columns: "
            f"{duplicates}."
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Raw FRS {expected_filename} is missing required column(s): {missing}."
        )
    frame = frame.loc[:, list(required_columns)]
    identity = UKFRSRawTableIdentity(
        path=source_path,
        filename=expected_filename,
        source_vintage=source_vintage,
        sha256=digest,
        size_bytes=before.size_bytes,
        rows=len(frame),
        extracted_columns=required_columns,
    )
    return frame, identity


def _materialize_source_leaves(
    adult: pd.DataFrame,
    benefits: pd.DataFrame,
) -> pd.DataFrame:
    adult_ids = _raw_source_person_ids(adult, label="ADULT")
    if pd.Index(adult_ids).duplicated().any():
        duplicates = pd.Index(adult_ids)[pd.Index(adult_ids).duplicated()].unique()
        raise ValueError(
            "Raw FRS ADULT person identities must be unique; duplicate "
            f"value(s): {duplicates[:5].tolist()}."
        )
    earnings = _finite_numeric(adult["inearns"], label="ADULT.INEARNS")
    pay = np.maximum(earnings, 0.0) * FRS_WEEKS_IN_YEAR
    adult_leaf = pd.DataFrame(
        {FRS_HMRC_PAY_COLUMN: pay},
        index=pd.Index(adult_ids, name="source_person_id"),
    )

    benefit_ids = _raw_source_person_ids(benefits, label="BENEFITS")
    benefit_codes = _strict_integer_values(
        benefits["benefit"],
        label="BENEFITS.BENEFIT",
        minimum=0,
    )
    relevant = np.isin(benefit_codes, (5, 13, 14, 16, 17, 19))
    amounts = np.zeros(len(benefits), dtype=float)
    if relevant.any():
        relevant_amounts = _finite_numeric(
            benefits.loc[relevant, "benamt"],
            label="relevant BENEFITS.BENAMT",
        )
        if (relevant_amounts < 0.0).any():
            raise ValueError("Relevant BENEFITS.BENAMT values must be non-negative.")
        amounts[relevant] = relevant_amounts

    code16 = benefit_codes == 16
    contribution_based_esa = np.zeros(len(benefits), dtype=bool)
    if code16.any():
        var2 = _strict_integer_values(
            benefits.loc[code16, "var2"],
            label="BENEFITS.VAR2 for BENEFIT=16",
        )
        contribution_based_esa[code16] = np.isin(var2, (1, 3))

    benefit_leaf = pd.DataFrame(
        {
            FRS_HMRC_UBISJA_COLUMN: amounts * np.isin(benefit_codes, (14, 19)),
            FRS_HMRC_INCPBEN_COLUMN: amounts * (benefit_codes == 17),
            FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: amounts
            * ((benefit_codes == 13) | contribution_based_esa),
            FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: amounts * (benefit_codes == 5),
        },
        index=pd.Index(benefit_ids, name="source_person_id"),
    )
    benefit_leaf = benefit_leaf.groupby(level=0, sort=False).sum()
    benefit_leaf *= FRS_WEEKS_IN_YEAR

    source_ids = adult_leaf.index.union(benefit_leaf.index, sort=False)
    result = pd.DataFrame(
        0.0,
        index=source_ids,
        columns=FRS_HMRC_RETAINED_LEAF_COLUMNS,
    )
    result.loc[adult_leaf.index, FRS_HMRC_PAY_COLUMN] = adult_leaf[FRS_HMRC_PAY_COLUMN]
    for column in benefit_leaf.columns:
        result.loc[benefit_leaf.index, column] = benefit_leaf[column]
    numeric = result.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0.0).any():
        raise RuntimeError("Raw FRS source-leaf materialization is invalid.")
    return result


def _raw_source_person_ids(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    households = _strict_integer_values(
        frame["sernum"], label=f"{label}.SERNUM", minimum=1
    )
    people = _strict_integer_values(frame["person"], label=f"{label}.PERSON", minimum=1)
    if (people >= 1000).any():
        raise ValueError(f"{label}.PERSON must be less than 1000.")
    maximum_household = (np.iinfo(np.int64).max - people) // 1000
    if (households > maximum_household).any():
        raise ValueError(f"{label} source person identity exceeds int64 range.")
    return households * 1000 + people


def _strict_integer_values(
    values: pd.Series,
    *,
    label: str,
    minimum: int | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain finite numeric values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain integer values.")
    if (np.abs(numeric) > np.iinfo(np.int64).max).any():
        raise ValueError(f"{label} exceeds int64 range.")
    result = numeric.astype(np.int64)
    if minimum is not None and (result < minimum).any():
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _finite_numeric(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain finite numeric values.")
    return numeric


def _file_fingerprint(path: Path) -> _FileFingerprint:
    stat = path.stat()
    return _FileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
