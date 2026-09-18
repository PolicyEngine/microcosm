"""Reconstruct the native SPM role from pinned, complete Census ASEC files.

This is a source enrichment of an existing population. The raw relationship
rule is a documented-source inference reconciled against Census unit counts;
it is not a claim to have retrieved the Census production program. No model,
weight, calibration, or period-aging operation is performed here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

NATIVE_SPM_ROLE = "is_spm_independent_minor_role"
EVIDENCE_SPM_ROLE = "is_spm_independence_role"
SPM_ROLE_RULE = "SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})"
SPM_ADULT_RULE = "age >= 18 OR (age >= 15 AND is_spm_independent_minor_role)"
_COUNTS = ("SPM_NUMADULTS", "SPM_NUMKIDS", "SPM_NUMPER")
_REQUIRED_RAW_CHECKS = ("A_AGE", "A_LINENO", "P_SEQ", "SPM_HAGE", *_COUNTS)
_OPTIONAL_RAW_CHECKS = ("SPM_HEAD", "A_FAMTYP", "A_FAMREL", "A_SPOUSE", "PECOHAB")
_SOURCE_COLUMNS = (
    "PERIDNUM",
    "SPM_ID",
    "PH_SEQ",
    *_REQUIRED_RAW_CHECKS,
    *_OPTIONAL_RAW_CHECKS,
)


@dataclass(frozen=True)
class AsecSpmRoleSource:
    """Reviewed identity and population coverage of a complete person CSV."""

    income_year: int
    survey_year: int
    csv_sha256: str
    csv_size_bytes: int
    persons: int
    units: int
    official_archive_url: str
    archive_sha256: str
    member: str


ASEC_SPM_ROLE_SOURCES: dict[int, AsecSpmRoleSource] = {
    year: AsecSpmRoleSource(
        income_year=year,
        survey_year=pin.survey_year,
        csv_sha256=pin.member_sha256,
        csv_size_bytes=pin.member_size_bytes,
        persons=pin.rows,
        units={2022: 59_181, 2023: 58_711, 2024: 58_147}[year],
        official_archive_url=pin.zip_url,
        archive_sha256=pin.zip_sha256,
        member=pin.member,
    )
    for year, pin in ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items()
}


@dataclass(frozen=True)
class SpmRoleSourceResult:
    """The primitive in parent row order, keyed evidence, and public counts."""

    role: np.ndarray
    evidence: pd.DataFrame
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    _require(not missing, f"{label} missing required columns: {missing}.")


def _exact_person_keys(values: pd.Series, label: str) -> pd.Series:
    # Converting numeric IDs to text would hide lossy earlier coercions.
    decoded = values.map(
        lambda value: value.decode("ascii") if isinstance(value, bytes) else value
    )
    valid = decoded.map(
        lambda value: (
            isinstance(value, str)
            and len(value) == 22
            and value.isascii()
            and value.isdigit()
        )
    )
    _require(bool(valid.all()), f"{label} PERIDNUM must be exact 22-digit strings.")
    return decoded


def independent_minor_role(persons: pd.DataFrame) -> pd.Series:
    """Return the source role BEFORE any age gate (including for adults)."""
    _require_columns(persons, ("SPM_HEAD", "A_FAMTYP", "A_FAMREL"), "ASEC role")
    return persons.SPM_HEAD.eq(1) | (
        persons.A_FAMTYP.isin((1, 4)) & persons.A_FAMREL.isin((1, 2))
    )


def _reconcile_units(persons: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    groups = persons.groupby(key, sort=False, dropna=False)
    _require(
        bool(groups[list(_COUNTS)].nunique(dropna=False).eq(1).all().all()),
        f"{label} source counts are not constant within SPM units.",
    )
    units = groups.agg(
        adults=("_adult", "sum"),
        expected_adults=("SPM_NUMADULTS", "first"),
        expected_children=("SPM_NUMKIDS", "first"),
        expected_persons=("SPM_NUMPER", "first"),
        persons=("PERIDNUM", "size"),
        heads=("SPM_HEAD", "sum"),
        age_only_adults=("_age_only_adult", "sum"),
    )
    _require(
        bool(units.adults.eq(units.expected_adults).all())
        and bool((units.persons - units.adults).eq(units.expected_children).all())
        and bool(units.persons.eq(units.expected_persons).all()),
        f"{label} adult/child/person count reconciliation failed.",
    )
    _require(
        bool(units.heads.eq(1).all()),
        f"{label} must have exactly one SPM head per unit.",
    )
    _require(
        bool(units.adults.ge(1).all()),
        f"{label} has an unresolved zero-adult SPM unit.",
    )
    heads = persons.SPM_HEAD.eq(1)
    _require(
        bool(persons.loc[heads, "A_AGE"].eq(persons.loc[heads, "SPM_HAGE"]).all()),
        f"{label} source SPM head age disagrees with SPM_HAGE.",
    )
    return units


def _load_source(
    path: Path, pin: AsecSpmRoleSource
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label = f"ASEC {pin.survey_year}"
    _require(
        pin.survey_year == pin.income_year + 1, f"{label} income/survey year mismatch."
    )
    _require(
        path.stat().st_size == pin.csv_size_bytes, f"{label} CSV byte length mismatch."
    )
    _require(_sha256(path) == pin.csv_sha256, f"{label} CSV SHA-256 mismatch.")
    source = pd.read_csv(
        path,
        usecols=list(_SOURCE_COLUMNS),
        dtype={"PERIDNUM": str, "SPM_ID": str},
        low_memory=False,
    )
    _require(_sha256(path) == pin.csv_sha256, f"{label} CSV changed while reading.")
    _require(len(source) == pin.persons, f"{label} CSV person count mismatch.")
    source["PERIDNUM"] = _exact_person_keys(source.PERIDNUM, label)
    _require(bool(source.PERIDNUM.is_unique), f"{label} has duplicate PERIDNUM keys.")
    _require(
        bool(source.notna().all().all()),
        f"{label} source columns contain missing values.",
    )
    _require(
        bool(source.SPM_ID.str.fullmatch(r"[0-9]+").all()),
        f"{label} SPM_ID must retain exact digit strings.",
    )
    numeric = source.drop(columns=["PERIDNUM", "SPM_ID"]).to_numpy(dtype=float)
    _require(
        bool((np.isfinite(numeric) & (numeric == np.floor(numeric))).all()),
        f"{label} raw role/count/identity fields must be finite integers.",
    )
    _require(
        bool(source.SPM_HEAD.isin((0, 1)).all()), f"{label} SPM_HEAD must be binary."
    )
    _require(
        bool(source.A_AGE.ge(0).all()), f"{label} source ages must be nonnegative."
    )
    source["_role"] = independent_minor_role(source)
    source["_age_only_adult"] = source.A_AGE.ge(18)
    source["_adult"] = source._age_only_adult | (source.A_AGE.ge(15) & source._role)
    units = _reconcile_units(source, "SPM_ID", label)
    _require(len(units) == pin.units, f"{label} CSV SPM unit count mismatch.")
    source["source_year"] = pin.income_year
    source["_source_row_id"] = np.arange(len(source), dtype=np.int64)
    extra_minor = source.A_AGE.between(15, 17) & source._role & source.SPM_HEAD.eq(0)
    return source, {
        "income_year": pin.income_year,
        "survey_year": pin.survey_year,
        "csv_sha256": pin.csv_sha256,
        "csv_size_bytes": pin.csv_size_bytes,
        "official_archive_url": pin.official_archive_url,
        "archive_sha256": pin.archive_sha256,
        "member": pin.member,
        "persons": len(source),
        "units": len(units),
        "adult_child_person_count_mismatch_units": 0,
        "extra_independent_minor_people_vs_head_only": int(extra_minor.sum()),
        "extra_independent_minor_units_vs_head_only": int(
            source.loc[extra_minor, "SPM_ID"].nunique()
        ),
    }


def derive_spm_role_source(
    parent_h5: str | Path,
    source_paths: Mapping[int, str | Path],
    *,
    expected_parent_sha256: str,
    source_pins: Mapping[int, AsecSpmRoleSource] | None = None,
) -> SpmRoleSourceResult:
    """Derive and reconcile from complete pinned CSVs; never trust a sidecar.

    Local CSV paths are mandatory; the existing education-assistance Census
    fetcher can obtain their pinned archives separately. The default pins cover
    all three certified BuildP source years. Alternate pins support explicit
    review of other populations and small synthetic tests.

    Support clones may repeat a source person across native units. Within every
    native unit, the source members must be unique and exhaust one complete
    Census unit. Original membership and every existing input remain untouched.
    """
    path = Path(parent_h5)
    _require(_sha256(path) == expected_parent_sha256, "Parent H5 SHA-256 mismatch.")
    pins = ASEC_SPM_ROLE_SOURCES if source_pins is None else source_pins
    parent = pd.read_hdf(path, "person")
    spm = pd.read_hdf(path, "spm_unit")
    required = (
        "person_id",
        "person_spm_unit_id",
        "source_year",
        "PERIDNUM",
        "age",
        "source_household_id",
        "source_person_id",
        "source_row_id",
        *_REQUIRED_RAW_CHECKS,
    )
    _require_columns(parent, required, "Parent person table")
    _require_columns(spm, ("spm_unit_id",), "Parent SPM table")
    _require(len(parent) > 0, "Parent person table is empty.")
    _require(bool(parent.person_id.is_unique), "Parent person_id must be unique.")
    _require(bool(spm.spm_unit_id.is_unique), "Parent spm_unit_id must be unique.")
    _require(
        bool(parent[list(required)].notna().all().all()),
        "Parent required identity/age/count fields contain missing values.",
    )
    _require(
        bool(spm.spm_unit_id.notna().all())
        and set(parent.person_spm_unit_id) == set(spm.spm_unit_id),
        "Parent SPM membership does not exactly cover its SPM table.",
    )
    years = pd.to_numeric(parent.source_year, errors="coerce")
    _require(
        bool((np.isfinite(years) & years.eq(np.floor(years))).all()),
        "Parent source_year must be finite integers.",
    )
    _require(
        set(years) == set(pins) == set(source_paths),
        "Parent income years, pinned sources, and explicit CSV paths must match exactly.",
    )
    parent = parent.copy()
    parent["PERIDNUM"] = _exact_person_keys(parent.PERIDNUM, "Parent")
    sources, source_checks = [], []
    for year in sorted(pins):
        _require(pins[year].income_year == year, "Source pin income-year key mismatch.")
        source, check = _load_source(Path(source_paths[year]), pins[year])
        sources.append(source)
        source_checks.append(check)
    source = pd.concat(sources, ignore_index=True)
    # Raw SPM_ID was globalized by the pooled-source producer; reconstruct its
    # source membership through (income year, PERIDNUM), never compare remapped IDs.
    joined = parent.merge(
        source,
        on=["source_year", "PERIDNUM"],
        how="left",
        sort=False,
        validate="many_to_one",
        suffixes=("_parent", ""),
        indicator=True,
    )
    _require(
        bool(joined._merge.eq("both").all()),
        "Source join has unmatched parent persons.",
    )
    _require(
        np.array_equal(joined.person_id.to_numpy(), parent.person_id.to_numpy()),
        "Source join changed parent person order.",
    )
    for column in _REQUIRED_RAW_CHECKS:
        _require(
            bool(joined[f"{column}_parent"].eq(joined[column]).all()),
            f"Parent raw field {column} disagrees with the pinned Census source.",
        )
    optional_coverage = {}
    for column in _OPTIONAL_RAW_CHECKS:
        if column in parent:
            available = joined[f"{column}_parent"].notna()
            _require(
                bool(
                    joined.loc[available, f"{column}_parent"]
                    .eq(joined.loc[available, column])
                    .all()
                ),
                f"Parent raw field {column} disagrees with the pinned Census source.",
            )
            optional_coverage[column] = int(available.sum())
    for native, raw in (
        ("age", "A_AGE"),
        ("source_household_id", "PH_SEQ"),
        ("source_person_id", "PERIDNUM"),
        ("source_row_id", "_source_row_id"),
    ):
        _require(
            bool(joined[native].eq(joined[raw]).all()),
            f"Parent {native} disagrees with the pinned Census source.",
        )
    _require(
        not bool(
            joined.duplicated(["person_spm_unit_id", "source_year", "PERIDNUM"]).any()
        ),
        "Native SPM unit repeats a source person.",
    )
    native_groups = joined.groupby("person_spm_unit_id", sort=False)
    _require(
        bool(native_groups[["source_year", "SPM_ID"]].nunique().eq(1).all().all()),
        "Native SPM membership combines distinct source units.",
    )
    units = _reconcile_units(joined, "person_spm_unit_id", "Parent native SPM")
    role = joined._role.to_numpy(dtype=np.bool_, copy=True)
    evidence = parent[["person_id", "person_spm_unit_id"]].copy()
    evidence[EVIDENCE_SPM_ROLE] = role
    minor = joined.A_AGE.between(15, 17) & joined._role
    _require(
        _sha256(path) == expected_parent_sha256,
        "Parent H5 changed while deriving roles.",
    )
    return SpmRoleSourceResult(
        role=role,
        evidence=evidence,
        provenance={
            "dataset_sha256": expected_parent_sha256,
            "primitive_column": NATIVE_SPM_ROLE,
            "evidence_column": EVIDENCE_SPM_ROLE,
            "primitive_rule": SPM_ROLE_RULE,
            "adult_rule": SPM_ADULT_RULE,
            "evidence_status": (
                "Inference from documented primitive relationships, reconciled against "
                "every SPM count in pinned complete Census ASEC sources; "
                "Census production program not retrieved."
            ),
            "source_join": [
                "source_year (income year)",
                "PERIDNUM (exact 22-digit string)",
            ],
            "source_checks": source_checks,
            "total_source_people": len(source),
            "total_source_units": sum(check["units"] for check in source_checks),
            "persons_joined": len(joined),
            "unmatched_persons": 0,
            "all_rows_equal_source_columns": list(_REQUIRED_RAW_CHECKS),
            "nonmissing_optional_raw_fields_checked": optional_coverage,
            "native_spm_units": len(units),
            "complete_source_membership_units": len(units),
            "adult_child_person_count_mismatch_units": 0,
            "independent_minor_persons": int(minor.sum()),
            "classification_changed_units_vs_age_only": int(
                units.adults.ne(units.age_only_adults).sum()
            ),
            "minor_only_units_resolved": int(units.age_only_adults.eq(0).sum()),
            "true_role_persons": int(role.sum()),
            "table_rows": len(evidence),
            "table_key": ["person_id", "person_spm_unit_id"],
            "table_order": "exact original HDF person order; verified",
            "weights_used": False,
            "ages_changed": False,
            "period_handling": "Store the role before the age gate; model age supplies the requested period.",
        },
    )
