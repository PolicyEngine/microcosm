"""Build a processed CPS ASEC ``census_cps_YYYY.h5`` from a pinned Census archive.

The US base stage reads one processed CPS ASEC HDF5 file per pooled income
year; :mod:`.asec_sources` pins them. The pinned files were written by the
archived US data package's ``CensusCPS`` loader, which takes no changes. This
module is Microcosm's own builder for them. It reads the official Census
public-use archive for survey year ``income_year + 1``, verifies its bytes
against :data:`.education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES`,
and writes the same five tables the loader wrote.

Semantics are those of loader commit ``40314a75`` (#824, "Construct CPS tax
units from household records") with the income-year 2025 patch (fork commit
``9f177ddd``). That revision rebuilds every table of the pinned
``census_cps_2024.h5`` exactly; ``experiments/us-asec-2025-source`` holds the
receipts, and ``experiments/us-asec-processed-source-builder`` holds this
module's own differential receipts against the pinned 2024 and 2025 files.

The five tables, written with pandas ``HDFStore`` in fixed format and in this
order:

``person``
    Every column of :data:`PERSON_COLUMNS`, :func:`spm_unit_columns_for_year`
    and :data:`TAX_UNIT_COLUMNS` that the Census person file carries, in the
    file's own column order, with missing values filled with 0. A column of
    :data:`OPTIONAL_PERSON_COLUMNS` that the file lacks is appended as zeros,
    in that tuple's order; any other missing column refuses the build.
    ``CENSUS_TAX_ID`` is appended with Census's own ``TAX_ID``, and ``TAX_ID``
    is replaced by the constructed tax unit.
``family``
    The Census family rows whose ``FH_SEQ * 10 + FFPOS`` matches some person's
    ``PH_SEQ * 10 + PF_SEQ``. All columns, missing values filled with 0.
``household``
    The Census household rows whose ``H_SEQ`` matches some person's
    ``PH_SEQ``. All columns, missing values filled with 0.
``tax_unit``
    One row per constructed tax unit, ``TAX_ID`` ascending from 1.
``spm_unit``
    The first person row of each SPM unit, restricted to the SPM columns
    (``SPM_ID`` included) and indexed by ``SPM_ID`` ascending.

Tax units come from :func:`microunit.construct_tax_units` in its
``policyengine`` mode: the constructor :mod:`microcosm.frame.units` already
uses, and a behavior-preserving refactor of the construction the loader
called. Within each household it forms married spouse-present pairs and a
unit for each other person 18 or older or ever married, then assigns
dependents to those units. The qualifying-relative gross income limit is the
IRS exemption amount for the income year. Records that break the
construction's preconditions are refused (:func:`audit_tax_unit_inputs`).

Two pandas behaviors changed between the loader's pandas 2.3 and Microcosm's
pandas 3, and the reader pins the loader's results:

* A digit-only column too wide for 64-bit integers (``PERIDNUM``, 22 digits;
  ``H_IDNUM``, 20) is kept as its CSV text. pandas 3 parses it into Python
  integers, which drops leading zeros; about 30% of ``PERIDNUM`` values start
  with ``0``.
* Text columns are stored as NumPy ``object`` arrays, as pandas 2 stored
  them, not in the pandas 3 string dtype, so each is stored as the pinned
  files store it: a pickled two-dimensional object block.

HDF5 bytes are not deterministic, so a rebuild matches the pinned tables but
not the pinned SHA-256. :func:`compare_asec_processed_stores` compares two
stores table by table, as pandas frames and column by column in HDF5
storage.
"""

from __future__ import annotations

import hashlib
import os
import warnings
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
    AsecEducationArchive,
    fetch_asec_education_assistance_source,
)

__all__ = [
    "ASEC_PROCESSED_TABLE_KEYS",
    "OPTIONAL_PERSON_COLUMNS",
    "PERSON_COLUMNS",
    "SPM_BBSUBVAL_INCOME_YEARS",
    "SPM_UNIT_COLUMNS",
    "TAX_UNIT_COLUMNS",
    "TAX_UNIT_CONSTRUCTION_MODE",
    "TEXT_IDENTIFIER_COLUMNS",
    "AsecProcessedTables",
    "asec_archive_members",
    "audit_tax_unit_inputs",
    "asec_processed_archive_pins",
    "build_asec_processed_source",
    "build_asec_processed_tables",
    "compare_asec_processed_stores",
    "construct_asec_tax_units",
    "dependent_gross_income_limit_for_year",
    "read_asec_processed_store",
    "resolve_asec_archive",
    "spm_unit_columns_for_year",
    "verify_asec_archive",
    "write_asec_processed_store",
]

#: HDF5 keys, in the order the loader wrote them.
ASEC_PROCESSED_TABLE_KEYS: tuple[str, ...] = (
    "person",
    "family",
    "household",
    "tax_unit",
    "spm_unit",
)

#: The constructor mode the loader used for every pinned file.
TAX_UNIT_CONSTRUCTION_MODE = "policyengine"

#: Census publishes the SPM unit's broadband subsidy, ``SPM_BBSUBVAL``, only
#: for these income years. ASEC 2026 (income year 2025) dropped it, and the
#: builder omits it there rather than zero-filling it.
SPM_BBSUBVAL_INCOME_YEARS = range(2021, 2025)

#: Tax-model outputs Census attaches to each person record.
TAX_UNIT_COLUMNS: tuple[str, ...] = (
    "ACTC_CRD",
    "AGI",
    "CTC_CRD",
    "EIT_CRED",
    "FEDTAX_AC",
    "FEDTAX_BC",
    "MARG_TAX",
    "STATETAX_A",
    "STATETAX_B",
    "TAX_INC",
)

#: SPM unit columns, each carried on every member's person record.
SPM_UNIT_COLUMNS: tuple[str, ...] = tuple(
    "SPM_" + column
    for column in (
        "ACTC",
        "BBSUBVAL",
        "CAPHOUSESUB",
        "CAPWKCCXPNS",
        "CHILDCAREXPNS",
        "CHILDSUPPD",
        "EITC",
        "ENGVAL",
        "EQUIVSCALE",
        "FAMTYPE",
        "FEDTAX",
        "FEDTAXBC",
        "FICA",
        "GEOADJ",
        "HAGE",
        "HHISP",
        "HMARITALSTATUS",
        "HRACE",
        "MEDXPNS",
        "NUMADULTS",
        "NUMKIDS",
        "NUMPER",
        "POOR",
        "POVTHRESHOLD",
        "RESOURCES",
        "SCHLUNCH",
        "SNAPSUB",
        "STTAX",
        "TENMORTSTATUS",
        "TOTVAL",
        "WCOHABIT",
        "WEIGHT",
        "WFOSTER22",
        "WICVAL",
        "WKXPNS",
        "WNEWHEAD",
        "WNEWPARENT",
        "WUI_LT15",
        "ID",
    )
)

#: Person-file columns the processed file keeps. The loader's list named
#: ``PHIP_VAL`` twice; it is named once here, which selects the same columns.
#: Selection keeps the file's column order, so this tuple's order is only the
#: order refusals report in.
PERSON_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "PF_SEQ",
    "P_SEQ",
    "TAX_ID",
    "PECOHAB",
    "SPM_ID",
    "A_FNLWGT",
    "A_LINENO",
    "A_SPOUSE",
    "A_EXPRRP",
    "A_FAMREL",
    "A_FAMTYP",
    "A_AGE",
    "A_SEX",
    "A_ENRLW",
    "A_FTPT",
    "PEDISEYE",
    "NOW_COV",
    "NOW_DIR",
    "NOW_MRK",
    "NOW_MRKS",
    "NOW_MRKUN",
    "NOW_NONM",
    "NOW_PRIV",
    "NOW_PUB",
    "NOW_GRP",
    "NOW_CAID",
    "NOW_MCAID",
    "NOW_PCHIP",
    "NOW_OTHMT",
    "NOW_MCARE",
    "NOW_MIL",
    "NOW_CHAMPVA",
    "NOW_VACARE",
    "NOW_IHSFLG",
    "WSAL_VAL",
    "INT_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "SS_VAL",
    "UC_VAL",
    "LKWEEKS",
    "ANN_VAL",
    "PNSN_VAL",
    "PTOTVAL",
    "OI_OFF",
    "OI_VAL",
    "CSP_VAL",
    "PAW_VAL",
    "SSI_VAL",
    "RETCB_VAL",
    "CAP_VAL",
    "WICYN",
    "VET_VAL",
    "WC_VAL",
    "DIS_VAL1",
    "DIS_VAL2",
    "CHSP_VAL",
    "PHIP_VAL",
    "MOOP",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "PEPAR1",
    "PEPAR2",
    "DIS_SC1",
    "DIS_SC2",
    "DST_SC1",
    "DST_SC2",
    "DST_SC1_YNG",
    "DST_SC2_YNG",
    "DST_VAL1",
    "DST_VAL2",
    "DST_VAL1_YNG",
    "DST_VAL2_YNG",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "PERIDNUM",
    "I_ERNVAL",
    "I_SEVAL",
    "A_HSCOL",
    "HRSWK",
    "WKSWORK",
    "POTC_VAL",
    "PMED_VAL",
    "PEMCPREM",
    "PRCITSHP",
    "POCCU2",
    "PEINUSYR",
    "MCARE",
    "PEN_SC1",
    "PEN_SC2",
    "RESNSS1",
    "RESNSS2",
    "IHSFLG",
    "CAID",
    "CHAMPVA",
    "PEIO1COW",
    "A_MJOCC",
    "SS_YN",
    "PEAFEVER",
    "SSI_YN",
    "RESNSSI1",
    "RESNSSI2",
    "PENATVTY",
    "PEIOOCC",
    "MIL",
    "A_HRS1",
)

#: Person columns an older person file may lack; each is then zero-filled.
#: The loader kept these in a ``set`` and appended the missing ones in set
#: iteration order, which varies with ``PYTHONHASHSEED``. This tuple fixes
#: the order to the loader's declaration order. The 2024 and 2025 files carry
#: every one, so their tables are unaffected.
OPTIONAL_PERSON_COLUMNS: tuple[str, ...] = (
    "NOW_COV",
    "NOW_DIR",
    "NOW_MRK",
    "NOW_MRKS",
    "NOW_MRKUN",
    "NOW_NONM",
    "NOW_PRIV",
    "NOW_PUB",
    "NOW_GRP",
    "NOW_CAID",
    "NOW_MCAID",
    "NOW_PCHIP",
    "NOW_OTHMT",
    "NOW_MCARE",
    "NOW_MIL",
    "NOW_CHAMPVA",
    "NOW_VACARE",
    "NOW_IHSFLG",
    "PTOTVAL",
)

#: Census record identifiers, published as digit strings (``PERIDNUM`` has 22
#: digits, ``H_IDNUM`` 20). They are always read as text. The loader relied
#: on pandas falling back to text because the values overflow 64-bit
#: integers; a file whose identifiers all fit would have been parsed as
#: integers, dropping leading zeros.
TEXT_IDENTIFIER_COLUMNS: tuple[str, ...] = ("PERIDNUM", "H_IDNUM")

_CENSUS_TAX_ID = "CENSUS_TAX_ID"
_CHUNK_SIZE = 8 * 1024 * 1024


def spm_unit_columns_for_year(income_year: int) -> tuple[str, ...]:
    """Return the SPM unit columns the Census person file carries that year."""

    if int(income_year) in SPM_BBSUBVAL_INCOME_YEARS:
        return SPM_UNIT_COLUMNS
    return tuple(column for column in SPM_UNIT_COLUMNS if column != "SPM_BBSUBVAL")


def asec_archive_members(survey_year: int) -> dict[str, str]:
    """Return the person, family and household CSV member names in a survey archive."""

    code = f"{int(survey_year) % 100:02d}"
    return {
        "person": f"pppub{code}.csv",
        "family": f"ffpub{code}.csv",
        "household": f"hhpub{code}.csv",
    }


def asec_processed_archive_pins(income_year: int) -> AsecEducationArchive:
    """Return the pinned Census archive for an income year, or refuse."""

    pins = ASEC_EDUCATION_ASSISTANCE_ARCHIVES.get(int(income_year))
    if pins is None:
        raise ValueError(
            f"No pinned Census ASEC archive covers income year {income_year!r}; "
            "pin its asecpubYYcsv.zip in education_assistance_source."
            "ASEC_EDUCATION_ASSISTANCE_ARCHIVES first. Pinned income years: "
            f"{sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES)}."
        )
    if pins.income_year != int(income_year) or pins.survey_year != pins.income_year + 1:
        raise ValueError(
            f"Archive pin for income year {income_year} names income year "
            f"{pins.income_year} and survey year {pins.survey_year}; the survey "
            "year must be the income year plus one."
        )
    return pins


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asec_archive(path: str | Path, pins: AsecEducationArchive) -> Path:
    """Refuse an archive whose bytes or person member differ from ``pins``.

    Checks the archive's byte length and SHA-256, then that it holds exactly
    one person member of the pinned name, byte length and CRC-32.
    """

    archive = Path(path).expanduser()
    if not archive.is_file():
        raise FileNotFoundError(f"ASEC {pins.survey_year} archive not found: {archive}")
    size = archive.stat().st_size
    if size != pins.zip_size_bytes:
        raise ValueError(
            f"ASEC {pins.survey_year} archive byte length mismatch: expected "
            f"{pins.zip_size_bytes}, got {size} ({archive})."
        )
    digest = _sha256_file(archive)
    if digest != pins.zip_sha256:
        raise ValueError(
            f"ASEC {pins.survey_year} archive SHA-256 mismatch: expected "
            f"{pins.zip_sha256}, got {digest} ({archive})."
        )
    with zipfile.ZipFile(archive) as archive_file:
        members = [
            info for info in archive_file.infolist() if info.filename == pins.member
        ]
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
        if f"{info.CRC:08x}" != pins.member_crc32:
            raise ValueError(
                f"ASEC {pins.survey_year} person member CRC-32 mismatch: "
                f"expected {pins.member_crc32}, got {info.CRC:08x}."
            )
    return archive


def resolve_asec_archive(
    income_year: int,
    archive: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a verified local copy of an income year's pinned Census archive.

    An explicit ``archive`` is verified and returned. Otherwise the archive is
    taken from the education-assistance cache (``cache_dir`` or
    ``~/.cache/microcosm/cps/asec_education``), where
    :func:`.fetch_asec_education_assistance_source` downloads and verifies it
    if it is absent.
    """

    pins = asec_processed_archive_pins(income_year)
    if archive is not None:
        return verify_asec_archive(archive, pins)
    person_member = fetch_asec_education_assistance_source(
        int(income_year), cache_dir=cache_dir
    )
    return verify_asec_archive(
        person_member.parent / f"asecpub{pins.survey_year % 100:02d}csv.zip",
        pins,
    )


def _holds_wide_integers(values: pd.Series) -> bool:
    return values.dtype == object and any(
        isinstance(value, int) and not isinstance(value, bool)
        for value in values.to_numpy()
    )


def _read_census_csv(
    archive_file: zipfile.ZipFile,
    member: str,
    *,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one Census CSV member with pandas 2.3's column types.

    :data:`TEXT_IDENTIFIER_COLUMNS` and any other digit-only column too wide
    for 64-bit integers keep their CSV text, and every text column is an
    ``object`` column.
    """

    with archive_file.open(member) as stream:
        header = pd.read_csv(stream, nrows=0).columns
    selected = set(header if usecols is None else usecols)
    text = {column: str for column in TEXT_IDENTIFIER_COLUMNS if column in selected}
    with archive_file.open(member) as stream:
        frame = pd.read_csv(
            stream,
            usecols=None if usecols is None else list(usecols),
            dtype=text or None,
        )
    wide = [column for column in frame.columns if _holds_wide_integers(frame[column])]
    if wide:
        with archive_file.open(member) as stream:
            wide_text = pd.read_csv(stream, usecols=wide, dtype=str)
        for column in wide:
            frame[column] = wide_text[column].astype(object)
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = frame[column].astype(object)
    return frame.fillna(0)


def _person_usecols(available: Sequence[str], income_year: int) -> list[str]:
    requested = dict.fromkeys(
        (*PERSON_COLUMNS, *spm_unit_columns_for_year(income_year), *TAX_UNIT_COLUMNS)
    )
    present = set(available)
    optional = set(OPTIONAL_PERSON_COLUMNS)
    missing = sorted(
        column
        for column in requested
        if column not in present and column not in optional
    )
    if missing:
        raise KeyError(
            f"Census person file for income year {income_year} lacks required "
            f"columns: {', '.join(missing)}."
        )
    return [column for column in requested if column in present]


def _published_dependent_income_limits() -> dict[int, float]:
    import yaml

    path = resources.files("microunit") / "data" / "dependent_gross_income_limit.yaml"
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)["values"]
    return {
        int(str(period)[:4]): float(entry["value"]) for period, entry in values.items()
    }


def dependent_gross_income_limit_for_year(income_year: int) -> float:
    """Return the qualifying-relative gross income limit the construction uses.

    microunit carries its last published IRS amount forward to later years.
    The builder refuses such a year instead, so a new income year cannot
    silently reuse the previous year's limit.
    """

    from microunit import dependent_gross_income_limit

    published = _published_dependent_income_limits()
    if int(income_year) not in published:
        raise ValueError(
            f"microunit publishes no dependent gross income limit for income "
            f"year {income_year}; its table covers {min(published)}-"
            f"{max(published)}. Add the IRS exemption amount for "
            f"{income_year} to microunit before building that year."
        )
    limit = float(dependent_gross_income_limit(int(income_year)))
    if limit != published[int(income_year)]:
        raise ValueError(
            f"microunit's dependent gross income limit for {income_year} is "
            f"{limit}, but its table says {published[int(income_year)]}."
        )
    return limit


def audit_tax_unit_inputs(person: pd.DataFrame) -> dict[str, Any]:
    """Count the record structures tax-unit construction depends on.

    The construction (microunit's ``policyengine`` mode, as in the loader)
    has three preconditions that it does not check:

    * Line numbers are unique within a household.
    * No person is in two married pairs. A person is married spouse-present
      when ``A_MARITL`` is 1 or 2 and ``A_SPOUSE`` is positive, and is paired
      with the member on line ``A_SPOUSE`` if that member is married
      spouse-present too. The pointer need not be returned, so a chain (A
      names B, B names C) puts B in two pairs.
    * No person names their own line as a parent (``PEPAR1``/``PEPAR2``). An
      adult who does, and who passes the qualifying-child age test, claims
      their own unit.

    Each violation leaves a unit without a head: the construction then drops
    a person from a unit or raises ``StopIteration`` partway through. Every
    pinned Census year satisfies all three, and has no one-sided spouse
    pointer and no pointer to an absent line either. The other counts are
    diagnostics the construction tolerates.
    """

    frame = person[["PH_SEQ", "A_LINENO", "A_MARITL", "A_SPOUSE", "PEPAR1", "PEPAR2"]]
    duplicated = frame.duplicated(["PH_SEQ", "A_LINENO"], keep=False)
    lines = frame[["PH_SEQ", "A_LINENO"]].drop_duplicates()
    married = frame[frame["A_MARITL"].isin([1, 2]) & (frame["A_SPOUSE"] > 0)]
    spouses = married.merge(
        frame.drop_duplicates(["PH_SEQ", "A_LINENO"], keep="last")[
            ["PH_SEQ", "A_LINENO", "A_MARITL", "A_SPOUSE"]
        ].rename(
            columns={
                "A_LINENO": "A_SPOUSE",
                "A_MARITL": "SPOUSE_MARITL",
                "A_SPOUSE": "SPOUSE_SPOUSE",
            }
        ),
        on=["PH_SEQ", "A_SPOUSE"],
        how="left",
    )
    present = spouses["SPOUSE_MARITL"].notna()
    self_spouse = spouses["A_SPOUSE"] == spouses["A_LINENO"]
    spouse_married = (
        present & spouses["SPOUSE_MARITL"].isin([1, 2]) & (spouses["SPOUSE_SPOUSE"] > 0)
    )
    paired = spouses[spouse_married & ~self_spouse]
    pairs = pd.DataFrame(
        {
            "PH_SEQ": paired["PH_SEQ"].to_numpy(),
            "first": np.minimum(paired["A_LINENO"], paired["A_SPOUSE"]).to_numpy(),
            "second": np.maximum(paired["A_LINENO"], paired["A_SPOUSE"]).to_numpy(),
        }
    ).drop_duplicates()
    members = pd.concat(
        [
            pairs[["PH_SEQ", "first"]].set_axis(["PH_SEQ", "A_LINENO"], axis=1),
            pairs[["PH_SEQ", "second"]].set_axis(["PH_SEQ", "A_LINENO"], axis=1),
        ]
    )
    several_pairs = members[members.duplicated(keep=False)].drop_duplicates()

    parents = pd.concat(
        [
            frame[["PH_SEQ", "A_LINENO", column]].set_axis(
                ["PH_SEQ", "A_LINENO", "parent"], axis=1
            )
            for column in ("PEPAR1", "PEPAR2")
        ]
    )
    parents = parents[parents["parent"] > 0]
    self_parent = parents["parent"] == parents["A_LINENO"]
    parent_present = parents.merge(
        lines.rename(columns={"A_LINENO": "parent"}),
        on=["PH_SEQ", "parent"],
        how="left",
        indicator=True,
    )["_merge"].eq("both")

    refused = {
        *map(int, several_pairs["PH_SEQ"]),
        *map(int, parents.loc[self_parent, "PH_SEQ"]),
        *map(int, frame.loc[duplicated, "PH_SEQ"]),
    }
    return {
        "married_spouse_present_with_pointer": int(len(married)),
        "pairs": int(len(pairs)),
        "spouse_line_absent": int((~present).sum()),
        "self_spouse_pointer": int(self_spouse.sum()),
        "spouse_not_married_spouse_present": int(
            (present & ~self_spouse & ~spouse_married).sum()
        ),
        "one_sided_spouse_pointer": int(
            (
                spouse_married
                & ~self_spouse
                & (spouses["SPOUSE_SPOUSE"] != spouses["A_LINENO"])
            ).sum()
        ),
        "parent_line_absent": int((~parent_present.to_numpy()).sum()),
        "persons_in_several_pairs": int(len(several_pairs)),
        "self_parent_pointer": int(self_parent.sum()),
        "duplicate_person_lines": int(duplicated.sum()),
        "refused_households": sorted(refused)[:10],
    }


_REFUSED_STRUCTURES = {
    "persons_in_several_pairs": "person(s) in several married pairs",
    "self_parent_pointer": "self parent pointer(s)",
    "duplicate_person_lines": "row(s) repeating a household line number",
}


def construct_asec_tax_units(
    person: pd.DataFrame,
    income_year: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Construct tax units from Census person records.

    Returns the per-person ``TAX_ID`` (in row order), the ``tax_unit`` table
    and microunit's full per-person assignment frame (``TAX_ID``,
    ``tax_unit_role_input``, ``is_related_to_head_or_spouse``). Refuses
    records that break the construction's preconditions (see
    :func:`audit_tax_unit_inputs`) rather than let it drop a person from a
    unit or fail partway through.
    """

    try:
        from microunit import construct_tax_units
    except ImportError as error:  # pragma: no cover - the us extra provides it
        raise ImportError(
            "Building a processed ASEC file needs microunit, Microcosm's tax-unit "
            "constructor; install microcosm-build[us]."
        ) from error

    dependent_gross_income_limit_for_year(income_year)
    audit = audit_tax_unit_inputs(person)
    violations = [
        f"{audit[key]} {label}"
        for key, label in _REFUSED_STRUCTURES.items()
        if audit[key]
    ]
    if violations:
        raise ValueError(
            "Tax units cannot be constructed from these records: "
            f"{', '.join(violations)} (households {audit['refused_households']})."
        )
    assignments, tax_unit = construct_tax_units(
        person=person,
        year=int(income_year),
        mode=TAX_UNIT_CONSTRUCTION_MODE,
    )
    tax_ids = assignments["TAX_ID"].to_numpy(dtype=np.int64)
    return tax_ids, tax_unit[["TAX_ID"]].reset_index(drop=True), assignments


@dataclass(frozen=True)
class AsecProcessedTables:
    """The five tables of one processed ASEC file, in memory."""

    income_year: int
    person: pd.DataFrame
    family: pd.DataFrame
    household: pd.DataFrame
    tax_unit: pd.DataFrame
    spm_unit: pd.DataFrame
    #: Optional person columns the Census file lacked, zero-filled in order.
    zero_filled_columns: tuple[str, ...]
    #: :func:`audit_tax_unit_inputs` of the person records.
    tax_unit_input_audit: dict[str, Any]
    #: microunit's per-person assignment frame: ``TAX_ID``,
    #: ``tax_unit_role_input`` and ``is_related_to_head_or_spouse``.
    tax_unit_assignments: pd.DataFrame = field(repr=False)

    def items(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield ``(key, table)`` in write order."""

        for key in ASEC_PROCESSED_TABLE_KEYS:
            yield key, getattr(self, key)


def _spm_unit_table(person: pd.DataFrame, income_year: int) -> pd.DataFrame:
    columns = list(spm_unit_columns_for_year(income_year))
    # Group by a named Index, not the person column, so SPM_ID stays a column
    # under every pandas version. pandas 3 drops a grouping column it can
    # trace back to the frame; pandas 2.3, which the loader ran, kept it.
    keys = pd.Index(person["SPM_ID"].to_numpy(), name="SPM_ID")
    return person[columns].groupby(keys, sort=True).first()


def build_asec_processed_tables(
    archive: str | Path,
    income_year: int,
    *,
    pins: AsecEducationArchive | None = None,
) -> AsecProcessedTables:
    """Build the five processed tables from a verified Census archive.

    ``pins`` defaults to the income year's entry in
    :data:`.education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES`.
    The archive is verified against it before any member is read, and the
    person file's row count must equal the pinned count.
    """

    income_year = int(income_year)
    pins = asec_processed_archive_pins(income_year) if pins is None else pins
    if pins.income_year != income_year:
        raise ValueError(
            f"Archive pin is for income year {pins.income_year}, not {income_year}."
        )
    archive_path = verify_asec_archive(archive, pins)
    members = asec_archive_members(pins.survey_year)
    if members["person"] != pins.member:
        raise ValueError(
            f"Archive pin names person member {pins.member!r}; survey year "
            f"{pins.survey_year} implies {members['person']!r}."
        )

    with zipfile.ZipFile(archive_path) as archive_file:
        names = set(archive_file.namelist())
        absent = sorted(name for name in members.values() if name not in names)
        if absent:
            raise ValueError(
                f"ASEC {pins.survey_year} archive lacks members: {', '.join(absent)}."
            )
        with archive_file.open(members["person"]) as stream:
            header = pd.read_csv(stream, nrows=0).columns
        person = _read_census_csv(
            archive_file,
            members["person"],
            usecols=_person_usecols(header, income_year),
        )
        if len(person) != pins.rows:
            raise ValueError(
                f"ASEC {pins.survey_year} person file has {len(person)} rows; "
                f"the pin records {pins.rows}."
            )
        zero_filled = tuple(
            column for column in OPTIONAL_PERSON_COLUMNS if column not in person
        )
        for column in zero_filled:
            person[column] = 0
        person[_CENSUS_TAX_ID] = person["TAX_ID"]
        tax_ids, tax_unit, assignments = construct_asec_tax_units(person, income_year)
        person["TAX_ID"] = tax_ids

        family = _read_census_csv(archive_file, members["family"])
        person_family_id = person["PH_SEQ"] * 10 + person["PF_SEQ"]
        family = family[
            (family["FH_SEQ"] * 10 + family["FFPOS"]).isin(person_family_id)
        ]

        household = _read_census_csv(archive_file, members["household"])
        household = household[household["H_SEQ"].isin(person["PH_SEQ"])]

    return AsecProcessedTables(
        income_year=income_year,
        person=person,
        family=family,
        household=household,
        tax_unit=tax_unit,
        spm_unit=_spm_unit_table(person, income_year),
        zero_filled_columns=zero_filled,
        tax_unit_input_audit=audit_tax_unit_inputs(person),
        tax_unit_assignments=assignments,
    )


def _refuse_pinned_overwrite(path: Path) -> None:
    """Refuse to replace a file whose bytes are a pinned processed ASEC input."""

    from microcosm.build.us_runtime.asec_sources import ASEC_SOURCE_ARTIFACTS

    size = path.stat().st_size
    candidates = [
        artifact
        for artifact in ASEC_SOURCE_ARTIFACTS.values()
        if artifact.size_bytes == size
    ]
    if candidates and _sha256_file(path) in {item.sha256 for item in candidates}:
        raise FileExistsError(
            f"{path} holds the pinned bytes of a processed ASEC input; a rebuild "
            "matches its tables, not its bytes, so it must not replace it. "
            "Write the rebuild elsewhere."
        )


def _loader_text_columns(table: pd.DataFrame) -> pd.DataFrame:
    text = [
        column
        for column in table.columns
        if isinstance(table[column].dtype, pd.StringDtype)
    ]
    return table.astype({column: object for column in text}) if text else table


def write_asec_processed_store(
    tables: AsecProcessedTables | Mapping[str, pd.DataFrame],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the tables to ``path`` atomically, in the loader's key order.

    Text columns are written as ``object`` columns, as the loader wrote them:
    pandas 3 would store a column in its string dtype as a one-dimensional
    block rather than the loader's two-dimensional object block.
    """

    output = Path(path).expanduser()
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output} exists; pass overwrite=True to replace it."
            )
        _refuse_pinned_overwrite(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with warnings.catch_warnings():
            # The loader stored text columns the same way: pickled object
            # arrays, which PyTables warns about.
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            items = dict(tables.items())
            with pd.HDFStore(temporary, mode="w") as store:
                for key in ASEC_PROCESSED_TABLE_KEYS:
                    if key in items:
                        store.put(key, _loader_text_columns(items[key]), format="fixed")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def read_asec_processed_store(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read every table of a processed ASEC store."""

    with pd.HDFStore(Path(path).expanduser(), mode="r") as store:
        return {key.lstrip("/"): store[key] for key in store.keys()}


def _attributes(node: Any, *, skip: frozenset[str] = frozenset()) -> dict[str, str]:
    attrs = node._v_attrs
    return {
        name: f"{type(attrs[name]).__name__}:{attrs[name]!r}"
        for name in sorted(attrs._f_list("user"))
        if name not in skip
    }


def _labels(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values.tolist()
    ]


def _column_bits(values: np.ndarray) -> tuple[str, Any]:
    column = np.ascontiguousarray(values)
    if column.dtype == object:
        return "object", [(type(value).__name__, value) for value in column.tolist()]
    return column.dtype.str, column.tobytes()


def _block_structure_attribute(name: str) -> bool:
    return name == "nblocks" or (
        name.startswith("block") and name.endswith("_items_variety")
    )


def _stored_table(file: Any, key: str) -> dict[str, Any]:
    """Read one fixed-format frame's HDF5 storage, keyed by column."""

    import tables

    group = file.get_node(f"/{key}")
    group_attrs = _attributes(group)
    axes = {}
    for axis in ("axis0", "axis1"):
        node = file.get_node(f"/{key}/{axis}")
        values = node.read()
        axes[axis] = {
            "atom": repr(node.atom),
            "attrs": _attributes(node),
            "values": _column_bits(values),
        }
    rows = int(file.get_node(f"/{key}/axis1").shape[0])
    columns: dict[str, Any] = {}
    layout: list[list[Any]] = []
    for index in range(int(group._v_attrs.nblocks)):
        items = _labels(file.get_node(f"/{key}/block{index}_items").read())
        node = file.get_node(f"/{key}/block{index}_values")
        values = node.read()
        if isinstance(node, tables.VLArray):
            values = np.asarray(values[0]) if len(values) == 1 else np.asarray(values)
        stored_shape = list(values.shape)
        if values.shape == (rows,) and len(items) == 1:
            # pandas 3 stores a string-dtype column as a 1-D block; the
            # recorded shape still tells it apart from a 2-D object block.
            values = values.reshape(rows, 1)
        if values.shape != (rows, len(items)):
            raise ValueError(
                f"/{key}/block{index}_values has shape {values.shape}; expected "
                f"({rows}, {len(items)})."
            )
        layout.append([repr(node.atom), items])
        for position, name in enumerate(items):
            columns[name] = {
                "atom": repr(node.atom),
                "attrs": _attributes(node),
                "block_dimensions": len(stored_shape),
                "bits": _column_bits(values[:, position]),
            }
    return {
        "group_attrs": {
            name: value
            for name, value in group_attrs.items()
            if not _block_structure_attribute(name)
        },
        "axes": axes,
        "columns": columns,
        "layout": layout,
    }


def _compare_hdf5_storage(
    left_path: Path, right_path: Path, key: str
) -> dict[str, Any]:
    """Compare two stores' HDF5 storage of one frame, column by column.

    Every column must be stored with the same atom, the same dataset
    attributes and bit-identical values, and the two axes (column labels and
    index) and the frame's group attributes must be identical. How pandas
    groups same-dtype columns into blocks is reported, not required: it is an
    internal memory layout that differs between pandas versions and that no
    reader depends on, since every reader places columns by label.
    """

    import tables

    with (
        tables.open_file(left_path, mode="r") as left_file,
        tables.open_file(right_path, mode="r") as right_file,
    ):
        left = _stored_table(left_file, key)
        right = _stored_table(right_file, key)
    shared = sorted(set(left["columns"]) & set(right["columns"]))
    mismatched = [
        name for name in shared if left["columns"][name] != right["columns"][name]
    ]
    only_left = sorted(set(left["columns"]) - set(right["columns"]))
    only_right = sorted(set(right["columns"]) - set(left["columns"]))
    axes_equal = {
        axis: left["axes"][axis] == right["axes"][axis] for axis in left["axes"]
    }
    group_attrs_equal = left["group_attrs"] == right["group_attrs"]
    return {
        "equal": not (mismatched or only_left or only_right)
        and all(axes_equal.values())
        and group_attrs_equal,
        "columns": len(left["columns"]),
        "mismatched_columns": mismatched,
        "only_candidate": only_left,
        "only_reference": only_right,
        "axes_equal": axes_equal,
        "group_attrs_equal": group_attrs_equal,
        "block_layout_equal": left["layout"] == right["layout"],
        "block_layout": {"candidate": left["layout"], "reference": right["layout"]},
    }


def _compare_frames(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    shared = [
        column for column in candidate.columns if column in set(reference.columns)
    ]
    dtype_mismatches = {
        column: [str(candidate[column].dtype), str(reference[column].dtype)]
        for column in shared
        if candidate[column].dtype != reference[column].dtype
    }
    value_mismatches: list[str] = []
    if len(candidate) == len(reference) and candidate.index.equals(reference.index):
        for column in shared:
            if column in dtype_mismatches:
                continue
            if not candidate[column].equals(reference[column]):
                value_mismatches.append(column)
    try:
        pd.testing.assert_frame_equal(
            candidate,
            reference,
            check_exact=True,
            check_index_type=True,
            check_column_type=True,
            check_names=True,
        )
        frame_equal = True
        detail = None
    except AssertionError as error:
        frame_equal = False
        detail = str(error).splitlines()[0][:500]
    return {
        "frame_equal_exact": frame_equal,
        "detail": detail,
        "shape": [list(candidate.shape), list(reference.shape)],
        "column_order_equal": list(candidate.columns) == list(reference.columns),
        "only_candidate": [
            c for c in candidate.columns if c not in set(reference.columns)
        ],
        "only_reference": [
            c for c in reference.columns if c not in set(candidate.columns)
        ],
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
    }


def compare_asec_processed_stores(
    candidate: str | Path,
    reference: str | Path,
) -> dict[str, Any]:
    """Compare two processed ASEC stores table by table.

    Each table is compared twice: as pandas frames, with
    ``assert_frame_equal(check_exact=True)`` plus per-column dtype and value
    checks, and as HDF5 storage, column by column (see
    :func:`_compare_hdf5_storage`). ``equal`` is true only if both views agree
    for every table and the stores hold the same keys.
    """

    candidate_path = Path(candidate).expanduser()
    reference_path = Path(reference).expanduser()
    candidate_tables = read_asec_processed_store(candidate_path)
    reference_tables = read_asec_processed_store(reference_path)
    tables_report: dict[str, Any] = {}
    for key in sorted(set(candidate_tables) & set(reference_tables)):
        report = _compare_frames(candidate_tables[key], reference_tables[key])
        report["hdf5"] = _compare_hdf5_storage(candidate_path, reference_path, key)
        report["equal"] = report["frame_equal_exact"] and report["hdf5"]["equal"]
        tables_report[key] = report
    only_candidate = sorted(set(candidate_tables) - set(reference_tables))
    only_reference = sorted(set(reference_tables) - set(candidate_tables))
    return {
        "equal": (
            not only_candidate
            and not only_reference
            and all(report["equal"] for report in tables_report.values())
        ),
        "keys_only_candidate": only_candidate,
        "keys_only_reference": only_reference,
        "tables": tables_report,
    }


def build_asec_processed_source(
    income_year: int,
    output: str | Path,
    *,
    archive: str | Path | None = None,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Resolve, verify and read an income year's archive, then write its store.

    Returns a summary: the verified archive, the output's SHA-256 and size,
    per-table rows and columns, the zero-filled optional columns and the
    tax-unit construction settings.
    """

    income_year = int(income_year)
    pins = asec_processed_archive_pins(income_year)
    archive_path = resolve_asec_archive(income_year, archive, cache_dir=cache_dir)
    tables = build_asec_processed_tables(archive_path, income_year, pins=pins)
    output_path = write_asec_processed_store(tables, output, overwrite=overwrite)
    import microunit

    return {
        "income_year": income_year,
        "survey_year": pins.survey_year,
        "archive": {
            "path": str(archive_path),
            "url": pins.zip_url,
            "sha256": pins.zip_sha256,
            "size_bytes": pins.zip_size_bytes,
            "person_member": pins.member,
            "person_rows": pins.rows,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "rows": {key: int(len(table)) for key, table in tables.items()},
        "columns": {key: int(table.shape[1]) for key, table in tables.items()},
        "spm_bbsubval_requested": income_year in SPM_BBSUBVAL_INCOME_YEARS,
        "zero_filled_columns": list(tables.zero_filled_columns),
        "tax_unit_input_audit": tables.tax_unit_input_audit,
        "tax_unit_construction": {
            "constructor": "microunit.construct_tax_units",
            "mode": TAX_UNIT_CONSTRUCTION_MODE,
            "microunit_version": microunit.__version__,
            "dependent_gross_income_limit": dependent_gross_income_limit_for_year(
                income_year
            ),
        },
    }
