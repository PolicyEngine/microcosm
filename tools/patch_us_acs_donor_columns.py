"""Write a patched copy of a US national donor H5 for the ACS staging (R1).

The County Files proof of concept (microcosm#922, plan microcosm#1647) stages
its ACS rows from the national donor of build 766b7c7
(``populace-us-2024-spm-receipts-20260923``). The ACS staging tool
(``tools/_legacy/build_us_acs_multispine_base.py``, R1) refuses that donor:
it predates three later donor contracts. This tool writes a patched copy of
the donor for the proof of concept only. It reads the published file and
never changes it. The real County Files donor will be a new national build
from ``main``.

The copy keeps every table, column, dtype and HDF5 root attribute of the
input. Each patch step only adds named columns to the entity tables, in this
order:

- ``asec_work_columns`` (:func:`asec_work_column_step`) restores
  ``detailed_industry_recode``, ``major_industry_recode`` and
  ``worked_last_year`` on ``person`` from the pinned ASEC person files
  (``pppub23``-``pppub26``), matched on source year and ``PERIDNUM``, as the
  #720 restoration and the fiscal ``org_wages`` carry value them (int16,
  int16, bool). It refuses a file whose SHA-256 is not pinned and a donor
  person with no match or more than one. The command looks the files up in
  ``--asec-source-cache`` (default ``~/.cache/microcosm/cps/asec_education``)
  and downloads a missing pinned archive only with ``--asec-fetch``.
- ``reviewed_zero_columns`` adds reviewed all-zero columns. It looks up each
  column's entity, dtype and default in the US engine and refuses a column
  whose engine default is not 0. ``REVIEWED_ZERO_COLUMNS`` holds the reason
  for each column.

A step never overwrites data: when a column is already present, the step
must give exactly the stored values, or the tool refuses the input. The tool
writes the copy atomically, verifies it against the input, and writes a JSON
receipt with both SHA-256 digests, the row counts per entity, each added
column with its entity and reason or source item, each step's own record
(for the work columns: the files read, their pinned SHA-256 digests and the
match counts), and the tool's commit.

Two options record lineage (both optional):

- ``--input-release-manifest`` is the published donor's
  ``release_manifest.json``. The input's SHA-256 must match its one microdata
  artifact, and the receipt records the donor build it names.
- ``--release-manifest`` with ``--manifest-release`` and ``--manifest-build``
  writes a release manifest in the build-folder layout that names the copy
  and its receipt. Upload the copy, the receipt and this file (as
  ``release_manifest.json``) to ``releases/<release>/<build>/`` in
  ``--manifest-repo-id``, where ``referenced_build`` reads a build. R1's
  ``--donor-release-manifest`` then verifies the copy against it and records
  the copy's build as ``donor_release``, which the local-area release
  contract requires before publication.

Usage::

    python tools/patch_us_acs_donor_columns.py \\
        --input-h5 populace_us_2024_receipt_qualified.h5 \\
        --input-release-manifest published_release_manifest.json \\
        --output-h5 run/donor_patched_poc.h5 \\
        --release-manifest run/release_manifest.json \\
        --manifest-release RELEASE --manifest-build BUILD

R1 then reads the copy with ``--base-h5 run/donor_patched_poc.h5`` and
``--donor-release-manifest run/release_manifest.json``, or without a manifest
(``donor_release`` is then null, and the publisher refuses the release).
"""

# No ``from __future__ import annotations``: test loaders run this file
# without registering it in ``sys.modules``, where string annotations break
# the dataclasses below.
import argparse
import gc
import hashlib
import json
import numbers
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import education_assistance_source, org_wages

# The #720 restoration's pin resolver, domain specs and work-experience
# identities, and the pinned reader's exact-key parser, are reused rather than
# re-derived, so the work-column step checks what the restoration checks.
from microcosm.build.us_runtime.asec_census_person_columns import (  # noqa: PLC2701
    ASEC_CENSUS_PERSON_COLUMNS,
    WORKER_INDUSTRY_CODES,
    _check_work_experience_universe,
    _resolve_pin,
)
from microcosm.build.us_runtime.h5_io import (
    assert_h5_unchanged,
    refuse_denied_pool_h5,
)
from microcosm.build.us_runtime.org_wages import (
    derive_us_org_occupation_inputs,
    require_us_occupation_and_work_experience_inputs,
)
from microcosm.build.us_runtime.spm_role_source import (  # noqa: PLC2701
    AsecSpmRoleSource,
    _exact_person_keys,
    read_pinned_asec_person_columns,
)
from microcosm.data.build_folder_manifests import referenced_build
from microcosm.frame import put_frame_table, read_frame_table
from microcosm.frame.materialize import _PYTABLES_FRAME_TABLE_CODEC_ATTR
from microcosm.frame.units import US_SCHEMA

TOOL = "tools/patch_us_acs_donor_columns.py"
RECEIPT_VERSION = 1
ISSUE = "microcosm#922"
PLAN = "microcosm#1647"
PURPOSE = (
    "County Files proof of concept only: a patched copy of the published "
    "national donor that the ACS staging (R1) accepts. The real County Files "
    "donor is a new national build from main."
)
_CONSUMER = f"US ACS donor column patch ({TOOL})"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: The repository the amended ruling names for the copy (microcosm#1647).
DEFAULT_MANIFEST_REPO_ID = "policyengine/populace-us-staging"
#: The copy's dataset role: not a release role, so the release contract
#: refuses the copy's folder as a release of its own.
COPY_DATASET_ROLE = "proof_of_concept_donor_copy"
#: A release or build name is one folder name (``releases/<release>/<build>``).
_FOLDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")

#: Each reviewed all-zero column and the reason its true values are unknown
#: or zero on the donor. The fill must equal the engine default (0).
REVIEWED_ZERO_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "strike_benefits": "no donor or ASEC pppub23–26 row has OI_OFF == 12",
        "passive_partnership_s_corp_income": (
            "the donor holds partnership and S-corp income as one combined "
            "amount in `partnership_income`; the split is unknown"
        ),
    }
)
ZERO_STEP = "reviewed_zero_columns"

#: The ASEC work columns that R1's ``_require_donor_work_experience`` needs
#: and that the restore step adds (see :func:`asec_work_column_step`).
ASEC_WORK_COLUMNS: tuple[str, ...] = (
    "detailed_industry_recode",
    "major_industry_recode",
    "worked_last_year",
)
ASEC_WORK_STEP = "asec_work_columns"

#: Each work column's source item in the pinned ASEC person member and its
#: dtype: those of ``org_wages.derive_us_org_occupation_inputs`` (#719).
ASEC_WORK_SOURCE_ITEMS: Mapping[str, str] = MappingProxyType(
    {
        "detailed_industry_recode": "WEIND",
        "major_industry_recode": "WEMIND",
        "worked_last_year": "WKSWORK > 0",
    }
)
ASEC_WORK_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "detailed_industry_recode": "int16",
        "major_industry_recode": "int16",
        "worked_last_year": "bool",
    }
)
#: Where the pinned members are looked up by default (the cache of
#: ``education_assistance_source.fetch_asec_education_assistance_source``).
DEFAULT_ASEC_SOURCE_CACHE = Path.home() / ".cache/microcosm/cps/asec_education"

#: Donor person column -> member column, compared on every joined row. The
#: donor's ``PH_SEQ`` is the pooled household id; the Census ``PH_SEQ``
#: survives as ``source_household_id`` (``asec_pool._remap_source_year``).
ASEC_IDENTITY_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "source_household_id": "PH_SEQ",
        "P_SEQ": "P_SEQ",
        "A_LINENO": "A_LINENO",
        "A_AGE": "A_AGE",
        "WKSWORK": "WKSWORK",
        "POCCU2": "POCCU2",
        "PRDTRACE": "PRDTRACE",
        "PRDTHSP": "PRDTHSP",
    }
)
#: Member columns read: the join key, the identity and the two restored codes.
_ASEC_MEMBER_COLUMNS: tuple[str, ...] = (
    "PERIDNUM",
    *dict.fromkeys(ASEC_IDENTITY_COLUMNS.values()),
    "WEIND",
    "WEMIND",
)
#: The restoration's reviewed domains for the two restored codes.
_ASEC_RESTORED_DOMAINS: Mapping[str, Any] = MappingProxyType(
    {
        spec.name: spec.domain
        for spec in ASEC_CENSUS_PERSON_COLUMNS
        if spec.name in ("WEIND", "WEMIND")
    }
)
_ASEC_REQUIRED_PERSON_COLUMNS: tuple[str, ...] = (
    "PERIDNUM",
    "source_year",
    *ASEC_IDENTITY_COLUMNS,
    "detailed_occupation_recode",
)
_SUPPORT_CHANNEL_COLUMN = "person_support_channel"

_NUMERIC_DTYPES: Mapping[str, type[np.generic]] = MappingProxyType(
    {"float": np.float64, "int": np.int64}
)


@dataclass(frozen=True)
class ColumnPatch:
    """One column that a step adds to one entity table.

    ``values`` must have the entity table's index. ``record`` goes into the
    receipt with the column, entity, dtype, step and action.
    """

    entity: str
    column: str
    values: pd.Series
    record: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchStepResult:
    """A step's column patches and its step-level receipt (sources, counts)."""

    patches: tuple[ColumnPatch, ...]
    receipt: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchStep:
    """A named step. ``run`` gets the entity tables after earlier steps.

    ``run`` must not change the tables it gets; it returns column patches.
    """

    name: str
    run: Callable[[Mapping[str, pd.DataFrame]], PatchStepResult]


def reviewed_zero_column_step(columns: Sequence[str], *, engine: Any) -> PatchStep:
    """Add the named reviewed columns as zeros on their engine entity.

    ``engine`` is the US engine adapter (or a stand-in with the same
    ``variable_metadata`` and ``default_values``). Each column must be in
    ``REVIEWED_ZERO_COLUMNS``, be a numeric engine input with a default, and
    have the engine default 0, so the fill is the value the engine would use.
    """

    unknown = sorted(set(columns) - set(REVIEWED_ZERO_COLUMNS))
    if unknown:
        raise ValueError(
            f"No reviewed reason for zero column(s) {unknown}; reviewed columns "
            f"are {sorted(REVIEWED_ZERO_COLUMNS)}."
        )
    if len(set(columns)) != len(columns) or not columns:
        raise ValueError(
            f"Zero columns must be distinct and non-empty; got {list(columns)}."
        )
    specs: list[tuple[str, str, str, object]] = []
    for column in columns:
        metadata = engine.variable_metadata(column)
        if metadata.dtype not in _NUMERIC_DTYPES:
            raise ValueError(
                f"{column}: the engine dtype is {metadata.dtype!r}; a zero fill "
                "needs a numeric input."
            )
        defaults = engine.default_values([column])
        if column not in defaults:
            raise ValueError(
                f"{column}: the engine reports no default for it (an unknown, "
                "formula-owned or declared source input); a zero fill must "
                "equal the engine default."
            )
        default = defaults[column]
        if isinstance(default, np.generic):
            default = default.item()
        if isinstance(default, bool) or not isinstance(default, numbers.Real):
            raise ValueError(f"{column}: engine default {default!r} is not numeric.")
        if default != 0:
            raise ValueError(
                f"{column}: the engine default is {default!r}, not 0; a zero "
                "fill would differ from the engine default."
            )
        specs.append((column, metadata.entity, metadata.dtype, default))

    def run(tables: Mapping[str, pd.DataFrame]) -> PatchStepResult:
        patches = []
        for column, entity, dtype, default in specs:
            if entity not in tables:
                raise ValueError(
                    f"{column}: the engine entity {entity!r} has no table in "
                    "the donor H5."
                )
            index = tables[entity].index
            patches.append(
                ColumnPatch(
                    entity=entity,
                    column=column,
                    values=pd.Series(
                        np.zeros(len(index), dtype=_NUMERIC_DTYPES[dtype]),
                        index=index,
                        name=column,
                    ),
                    record={
                        "engine_dtype": dtype,
                        "engine_default": default,
                        "value": 0,
                        "reason": REVIEWED_ZERO_COLUMNS[column],
                    },
                )
            )
        return PatchStepResult(patches=tuple(patches))

    return PatchStep(name=ZERO_STEP, run=run)


@dataclass(frozen=True)
class AsecPersonSources:
    """Where :func:`asec_work_column_step` finds the pinned ASEC person files.

    ``paths`` names each income year's file (the official archive or its
    person CSV). Otherwise each is looked up in ``cache`` (default
    :data:`DEFAULT_ASEC_SOURCE_CACHE`) as the pinned member CSV, then as the
    official archive, and downloaded through the pin-verifying
    ``fetch_asec_education_assistance_source`` only with ``fetch``. ``pins``
    overrides ``spm_role_source.ASEC_SPM_ROLE_SOURCES`` per income year, for
    synthetic tests; the restoration's resolver still refuses a pin whose
    survey year is not the income year plus one. Every file is read through
    ``read_pinned_asec_person_columns``, which refuses a size or SHA-256 that
    is not the pin's.
    """

    cache: Path | None = None
    paths: Mapping[int, Path] | None = None
    fetch: bool = False
    pins: Mapping[int, AsecSpmRoleSource] | None = None


def _asec_income_years(person: pd.DataFrame) -> tuple[np.ndarray, tuple[int, ...]]:
    years = pd.to_numeric(person["source_year"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not np.isfinite(years).all() or not np.equal(years, np.floor(years)).all():
        raise ValueError("Donor person source_year must be finite integers.")
    years = years.astype(np.int64)
    return years, tuple(sorted({int(year) for year in years}))


def resolve_asec_person_files(
    pins: Mapping[int, AsecSpmRoleSource], sources: AsecPersonSources
) -> dict[int, Path]:
    """Locate each income year's pinned member; pins are checked when read."""

    if sources.paths is not None:
        missing = sorted(
            str(year)
            for year in pins
            if year not in sources.paths or not Path(sources.paths[year]).is_file()
        )
        if missing:
            raise ValueError(
                f"No ASEC person file for income year(s) {missing}; every "
                "donor income year needs its pinned member."
            )
        return {year: Path(sources.paths[year]) for year in pins}
    cache = Path(sources.cache or DEFAULT_ASEC_SOURCE_CACHE).expanduser()
    resolved: dict[int, Path] = {}
    absent: list[str] = []
    for year, pin in pins.items():
        member = cache / pin.member
        archive = cache / Path(urlparse(pin.official_archive_url).path).name
        if member.is_file():
            resolved[year] = member
        elif archive.is_file():
            resolved[year] = archive
        elif sources.fetch:
            resolved[year] = (
                education_assistance_source.fetch_asec_education_assistance_source(
                    year, cache_dir=cache
                )
            )
        else:
            absent.append(str(year))
    if absent:
        raise ValueError(
            f"The pinned ASEC person member(s) for income year(s) {absent} are "
            f"not in {cache}; pass --asec-source-cache DIR, or --asec-fetch to "
            "download the pinned archives."
        )
    return resolved


def read_asec_work_member(
    path: Path, pin: AsecSpmRoleSource
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read and check one pinned member's join key, identity and codes."""

    # The label names the file actually read beside the pin's member, so a
    # wrong-year file is reported as itself.
    label = (
        f"ASEC {pin.survey_year} Census person member {pin.member} "
        f"(read from {Path(path).name})"
    )
    source, form = read_pinned_asec_person_columns(
        Path(path).expanduser(), pin, label, _ASEC_MEMBER_COLUMNS
    )
    if len(source) != pin.persons:
        raise ValueError(
            f"{label} holds {len(source)} persons; the pin records {pin.persons}."
        )
    for column in _ASEC_MEMBER_COLUMNS[1:]:
        values = source[column]
        if not pd.api.types.is_integer_dtype(values) or pd.api.types.is_bool_dtype(
            values
        ):
            raise ValueError(
                f"{label} column {column} parsed as {values.dtype}, not integer."
            )
    for column, domain in _ASEC_RESTORED_DOMAINS.items():
        outside = int((~source[column].isin(domain)).sum())
        if outside:
            raise ValueError(
                f"{label} column {column} has {outside} values outside its "
                f"reviewed Census codes {sorted(domain)}."
            )
    source = source.copy()
    source["PERIDNUM"] = _exact_person_keys(source["PERIDNUM"], label)
    repeated = int(source["PERIDNUM"].duplicated(keep=False).sum())
    if repeated:
        raise ValueError(
            f"{label} repeats PERIDNUM on {repeated} rows, so a donor person "
            "could match more than one member row."
        )
    universe = _check_work_experience_universe(source, label)
    return source.reset_index(drop=True), {
        "path": str(Path(path).expanduser().resolve()),
        "form": form,
        "income_year": pin.income_year,
        "survey_year": pin.survey_year,
        "member": pin.member,
        "member_sha256": pin.csv_sha256,
        "member_size_bytes": pin.csv_size_bytes,
        "archive_sha256": pin.archive_sha256 if form == "archive" else None,
        "official_archive_url": pin.official_archive_url,
        "persons": len(source),
        "work_experience_universe": universe,
    }


def _integer_values(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    invalid = ~np.isfinite(numeric) | (numeric != np.floor(numeric))
    if invalid.any():
        raise ValueError(
            f"{label} has {int(invalid.sum())} missing or non-integer rows."
        )
    return numeric.astype(np.int64)


def _member_positions(member_keys: pd.Series, keys: np.ndarray) -> np.ndarray:
    """Each donor key's member row by exact text, or -1."""

    return pd.Index(member_keys).get_indexer(keys)


def join_asec_work_codes(
    person: pd.DataFrame,
    years: np.ndarray,
    keys: np.ndarray,
    members: Mapping[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, object]]]:
    """``WEIND``/``WEMIND`` for every donor row, by exact (year, PERIDNUM).

    Many donor rows may join one member row (an ASEC row and its PUF-support
    clones); every donor row must match exactly one member row, and agree
    with it on every :data:`ASEC_IDENTITY_COLUMNS` pair.
    """

    rows = len(person)
    weind = np.full(rows, -1, dtype=np.int64)
    wemind = np.full(rows, -1, dtype=np.int64)
    per_year: dict[str, dict[str, object]] = {}
    for year in sorted(members):
        mask = years == year
        member = members[year]
        if not member["PERIDNUM"].is_unique:
            raise ValueError(
                f"Income year {year} member repeats PERIDNUM; a donor person "
                "could match more than one member row."
            )
        positions = _member_positions(member["PERIDNUM"], keys[mask])
        missing = int((positions < 0).sum())
        if missing:
            raise ValueError(
                f"{missing} donor person rows of income year {year} have no row "
                "in its pinned ASEC person member; every donor person must match."
            )
        aligned = member.iloc[positions]
        if not np.array_equal(aligned["PERIDNUM"].to_numpy(), keys[mask]):
            raise ValueError(
                f"Income year {year} member rows did not align by PERIDNUM."
            )
        mismatches: dict[str, int] = {}
        for donor_column, member_column in ASEC_IDENTITY_COLUMNS.items():
            donor_values = _integer_values(
                person.loc[mask, donor_column],
                f"Donor person {donor_column} (income year {year})",
            )
            disagree = donor_values != aligned[member_column].to_numpy(dtype=np.int64)
            if disagree.any():
                mismatches[donor_column] = int(disagree.sum())
        if mismatches:
            raise ValueError(
                f"Income year {year} donor rows disagree with their ASEC person "
                f"member after the (source_year, PERIDNUM) join: {mismatches} rows "
                "by column; the join would pair the wrong people."
            )
        weind[mask] = aligned["WEIND"].to_numpy(dtype=np.int64)
        wemind[mask] = aligned["WEMIND"].to_numpy(dtype=np.int64)
        distinct = int(len(np.unique(positions)))
        per_year[str(year)] = {
            "donor_rows": int(mask.sum()),
            "donor_rows_matched_once": int(mask.sum()),
            "distinct_keys_matched": distinct,
            "member_rows": int(len(member)),
            "member_rows_not_in_donor": int(len(member) - distinct),
            "identity_mismatches": 0,
        }
    if (weind < 0).any() or (wemind < 0).any():
        raise ValueError("A donor person row was left without a joined ASEC code.")
    return weind, wemind, per_year


def asec_clone_summary(
    person: pd.DataFrame, years: np.ndarray, keys: np.ndarray
) -> dict[str, object]:
    """Counts of the donor rows that share a (source_year, PERIDNUM) key."""

    channels = (
        person[_SUPPORT_CHANNEL_COLUMN].astype(str).to_numpy()
        if _SUPPORT_CHANNEL_COLUMN in person
        else np.full(len(person), "unlabelled", dtype=object)
    )
    frame = pd.DataFrame({"year": years, "key": keys, "channel": channels})
    grouped = frame.groupby(["year", "key"], sort=False)["channel"]
    sizes = grouped.size()
    composition = grouped.agg(lambda values: "+".join(sorted(values)))
    repeated = sizes.gt(1)
    return {
        "rows": int(len(frame)),
        "distinct_keys": int(len(sizes)),
        "keys_with_one_row": int(sizes.eq(1).sum()),
        "keys_with_two_rows": int(sizes.eq(2).sum()),
        "keys_with_more_rows": int(sizes.gt(2).sum()),
        "repeated_key_channel_composition": {
            str(label): int(count)
            for label, count in composition[repeated]
            .value_counts()
            .sort_index()
            .items()
        },
        "rows_by_support_channel": {
            str(label): int(count)
            for label, count in frame["channel"].value_counts().sort_index().items()
        },
        "handling": (
            "Every row sharing a key joins the same member row and must match "
            "it on every identity column, so clones get identical codes."
        ),
    }


def carry_asec_work_columns(
    person: pd.DataFrame, weind: np.ndarray, wemind: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """The three columns, from ``derive_us_org_occupation_inputs`` itself.

    The carry's other outputs must equal the donor's own columns (value and
    dtype), so the donor is known to come from the same carry.
    """

    restored = person[["PRDTRACE", "PRDTHSP", "POCCU2", "WKSWORK"]].copy()
    restored["WEIND"] = weind
    restored["WEMIND"] = wemind
    universe = _check_work_experience_universe(restored, "Donor person rows")
    carried = derive_us_org_occupation_inputs(restored)
    derived: dict[str, np.ndarray] = {}
    for column, dtype in ASEC_WORK_DTYPES.items():
        values = carried[column].to_numpy()
        if values.dtype != np.dtype(dtype) or values.shape != (len(person),):
            raise ValueError(
                f"The carry did not emit {column} as a person-grain {dtype}."
            )
        derived[column] = values
    verified, absent = [], []
    for column in carried.columns:
        if column in ASEC_WORK_DTYPES:
            continue
        if column not in person:
            absent.append(column)
            continue
        if (
            not np.array_equal(person[column].to_numpy(), carried[column].to_numpy())
            or person[column].dtype != carried[column].dtype
        ):
            raise ValueError(
                f"Donor {column} differs from the carry's value; the donor was "
                "not built by the carry these columns come from."
            )
        verified.append(column)
    candidate = pd.DataFrame(
        {"detailed_occupation_recode": person["detailed_occupation_recode"]},
        index=person.index,
    )
    for column, values in derived.items():
        candidate[column] = values
    require_us_occupation_and_work_experience_inputs(
        candidate, label="Donor with restored work columns"
    )
    return derived, {
        "function": (
            "microcosm.build.us_runtime.org_wages.derive_us_org_occupation_inputs"
        ),
        "inputs": ["PRDTRACE", "PRDTHSP", "POCCU2", "WKSWORK", "WEIND", "WEMIND"],
        "donor_work_experience_universe": universe,
        "reemitted_columns_verified_equal": verified,
        "reemitted_columns_absent_from_donor": absent,
        "require_us_occupation_and_work_experience_inputs": "passed",
    }


def _person_weights(tables: Mapping[str, pd.DataFrame]) -> np.ndarray | None:
    """``household_weight`` resolved to persons, or None without one."""

    household, person = tables["household"], tables["person"]
    if "household_weight" not in household or "person_household_id" not in person:
        return None
    weights = pd.Series(
        household["household_weight"].to_numpy(dtype=np.float64),
        index=household["household_id"].to_numpy(),
    )
    resolved = person["person_household_id"].map(weights)
    if resolved.isna().any():
        return None
    return resolved.to_numpy(dtype=np.float64)


def asec_work_summary(
    weights: np.ndarray | None,
    years: np.ndarray,
    derived: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Nonzero shares beside the carry's gate bands, and code counts."""

    bands = org_wages._SHARE_BANDS  # noqa: SLF001 - the carry's measured bands
    detailed = derived["detailed_industry_recode"].astype(np.int64)
    major = derived["major_industry_recode"].astype(np.int64)
    worked = derived["worked_last_year"].astype(bool)
    worker = np.isin(detailed, WORKER_INDUSTRY_CODES)
    shares: dict[str, object] = {}
    for column, values in derived.items():
        nonzero = np.asarray(values).astype(np.int64) != 0
        low, high = bands[column]
        entry: dict[str, object] = {
            "unweighted_nonzero_share": float(nonzero.mean()) if nonzero.size else 0.0,
            "gate_band": [low, high],
        }
        if weights is not None and float(weights.sum()):
            weighted = float(weights[nonzero].sum() / weights.sum())
            entry["weighted_nonzero_share"] = weighted
            entry["within_band"] = bool(low <= weighted <= high)
            entry["by_source_year_weighted"] = {
                str(year): (
                    float(
                        weights[(years == year) & nonzero].sum()
                        / weights[years == year].sum()
                    )
                    if weights[years == year].sum()
                    else 0.0
                )
                for year in sorted(set(years.tolist()))
            }
        shares[column] = entry
    return {
        "weights": (
            "household_weight resolved to persons" if weights is not None else None
        ),
        "nonzero_shares": shares,
        "worked_last_year_rows": int(worked.sum()),
        "worked_rows_with_worker_industry": int((worked & worker).sum()),
        "worker_industry_rows_without_weeks_worked": int((worker & ~worked).sum()),
        "detailed_industry_recode_counts": {
            str(code): int((detailed == code).sum()) for code in range(24)
        },
        "major_industry_recode_counts": {
            str(code): int((major == code).sum()) for code in range(16)
        },
    }


def asec_work_column_step(sources: AsecPersonSources | None = None) -> PatchStep:
    """Restore ``ASEC_WORK_COLUMNS`` from the pinned ASEC person files.

    The #719 work columns, valued as the #720 restoration and the fiscal
    ``org_wages`` carry value them (the route 2 step reviewed on #1647):

    - Each donor person's ``source_year`` is its ASEC income year; its member
      is the survey-year ``source_year + 1`` file (``_resolve_pin`` refuses
      any other pairing). Only the donor's income years are read, through
      ``read_pinned_asec_person_columns``, which refuses a file whose size or
      SHA-256 is not the pin's.
    - ``WEIND`` and ``WEMIND`` join by exact ``(source_year, PERIDNUM)`` text.
      A donor repeats a key within a year for an ASEC row and its PUF-support
      clone, so many donor rows join one member row. A donor person with no
      member row is refused, and so is a member that repeats a ``PERIDNUM``
      (a donor person would match more than one row). On every row the
      :data:`ASEC_IDENTITY_COLUMNS` must equal the member's.
    - The restoration's domains and work-experience identities are checked on
      each member and on the joined rows. The columns are the output of
      ``derive_us_org_occupation_inputs`` (int16, int16, bool); its other
      outputs must equal the donor's own columns.
    - A work column already on another entity is refused; one already on
      ``person`` must equal the restored values (the tool's rule).

    ``run`` returns one :class:`ColumnPatch` per column on ``person``, with
    its source item in ``record``, and the files, their digests, the match
    counts, the clone counts, the carry checks and the nonzero shares in the
    step ``receipt``. R1's ``_require_donor_work_experience`` then passes on
    the copy.
    """

    sources = sources or AsecPersonSources()

    def run(tables: Mapping[str, pd.DataFrame]) -> PatchStepResult:
        for entity, table in tables.items():
            elsewhere = sorted(
                column for column in ASEC_WORK_COLUMNS if column in table.columns
            )
            if entity != "person" and elsewhere:
                raise ValueError(
                    f"The donor already carries work column(s) {elsewhere} on "
                    f"{entity}; refusing to duplicate them on person."
                )
        person = tables["person"]
        missing = [c for c in _ASEC_REQUIRED_PERSON_COLUMNS if c not in person]
        if missing:
            raise ValueError(
                f"The donor person table lacks {missing}, which the ASEC work "
                "column join needs."
            )
        years, income_years = _asec_income_years(person)
        pins = {
            year: _resolve_pin(year, (sources.pins or {}).get(year))
            for year in income_years
        }
        paths = resolve_asec_person_files(pins, sources)
        keys = _exact_person_keys(person["PERIDNUM"], "Donor person table").to_numpy()
        members: dict[int, pd.DataFrame] = {}
        files: dict[str, dict[str, object]] = {}
        for year in income_years:
            members[year], files[str(year)] = read_asec_work_member(
                paths[year], pins[year]
            )
        weind, wemind, per_year = join_asec_work_codes(person, years, keys, members)
        del members
        gc.collect()
        clones = asec_clone_summary(person, years, keys)
        derived, carry = carry_asec_work_columns(person, weind, wemind)
        summary = asec_work_summary(_person_weights(tables), years, derived)
        patches = tuple(
            ColumnPatch(
                entity="person",
                column=column,
                values=pd.Series(derived[column], index=person.index, name=column),
                record={
                    "source_item": ASEC_WORK_SOURCE_ITEMS[column],
                    "source": (
                        "pinned ASEC person member of the row's income year, "
                        "joined on (source_year, PERIDNUM)"
                    ),
                    "derivation": carry["function"],
                },
            )
            for column in ASEC_WORK_COLUMNS
        )
        receipt = {
            "source": (
                "pinned Census ASEC person members "
                "(spm_role_source.ASEC_SPM_ROLE_SOURCES), read through "
                "spm_role_source.read_pinned_asec_person_columns"
            ),
            "vintage_rule": (
                "source_year is the ASEC income year; its member is survey year "
                "source_year + 1 (asec_census_person_columns._resolve_pin)"
            ),
            "files": files,
            "join": {
                "key": "(source_year, PERIDNUM as exact 22-digit text)",
                "identity_columns": dict(ASEC_IDENTITY_COLUMNS),
                "donor_rows": int(len(person)),
                "donor_rows_matched_once": int(len(person)),
                "unmatched_donor_rows": 0,
                "multiply_matched_donor_rows": 0,
                "by_income_year": per_year,
            },
            "clones": clones,
            "carry": carry,
            "work_experience": summary,
        }
        return PatchStepResult(patches=patches, receipt=receipt)

    return PatchStep(name=ASEC_WORK_STEP, run=run)


@dataclass(frozen=True)
class CopyReleaseManifest:
    """Where to write the copy's release manifest, and the build it names."""

    path: Path
    release: str
    build: str
    repo_id: str = DEFAULT_MANIFEST_REPO_ID

    def __post_init__(self) -> None:
        for name in ("release", "build"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _FOLDER_NAME.fullmatch(value):
                raise ValueError(
                    f"The copy's manifest {name} must be one folder name "
                    f"(letters, digits, '.', '_', '-'); got {value!r}."
                )
        if not isinstance(self.repo_id, str) or not _REPO_ID.fullmatch(self.repo_id):
            raise ValueError(
                f"The copy's manifest repo_id must be OWNER/NAME; got {self.repo_id!r}."
            )

    def identity(self) -> dict[str, str]:
        return {
            "path": str(Path(self.path).resolve()),
            "repo_id": self.repo_id,
            "release": self.release,
            "build": self.build,
            "folder": referenced_build(
                {"repo_id": self.repo_id, "release": self.release, "build": self.build}
            ).folder,
        }


def input_release_identity(manifest_path: Path, input_sha256: str) -> dict[str, object]:
    """Verify the input against its published release manifest; name its build.

    The manifest must carry exactly one microdata artifact, whose SHA-256 is
    the input's, and name a build that ``referenced_build`` accepts: a build
    folder (``release`` and ``build``) or a tag (``build.build_id`` and the
    artifact's ``revision``).
    """

    manifest_path = Path(manifest_path)
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read --input-release-manifest {manifest_path}: {error}."
        ) from error
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict):
        raise ValueError(f"{manifest_path} carries no artifacts map.")
    microdata = [
        (name, entry)
        for name, entry in artifacts.items()
        if isinstance(entry, dict) and entry.get("kind") == "microdata"
    ]
    if len(microdata) != 1:
        raise ValueError(
            f"{manifest_path} must carry exactly one microdata artifact; found "
            f"{sorted(name for name, _ in microdata)}."
        )
    name, entry = microdata[0]
    if entry.get("sha256") != input_sha256:
        raise ValueError(
            f"--input-h5 has SHA-256 {input_sha256}, but {manifest_path} pins "
            f"{entry.get('sha256')!r} for {name!r}."
        )
    build = manifest.get("build")
    if isinstance(build, dict):
        reference = {
            "repo_id": entry.get("repo_id"),
            "release_id": build.get("build_id"),
            "revision": entry.get("revision"),
        }
    else:
        reference = {
            "repo_id": entry.get("repo_id"),
            "release": manifest.get("release"),
            "build": build,
        }
    try:
        location = referenced_build(reference)
    except ValueError as error:
        raise ValueError(f"{manifest_path} names no readable build: {error}") from error
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact": name,
        **{key: value for key, value in reference.items() if value is not None},
        "folder": location.folder,
        "sha256": input_sha256,
    }


def copy_release_manifest(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    receipt_sha256: str,
    target: CopyReleaseManifest,
    built_at: str,
) -> dict[str, object]:
    """The copy's release manifest in the build-folder layout.

    Its one microdata artifact is the copy, so R1's
    ``--donor-release-manifest`` check (the base H5's SHA-256 against it)
    passes on the copy only. ``derived_from`` names the input; ``patch``
    names the tool, its commit, the receipt and the added columns.
    """

    output, source = receipt["output"], receipt["input"]
    assert isinstance(output, Mapping) and isinstance(source, Mapping)
    output_path = Path(str(output["path"]))
    added = receipt["added_columns"]
    assert isinstance(added, list)
    return {
        "schema_version": 1,
        "release": target.release,
        "build": target.build,
        "built_at": built_at,
        "dataset_role": COPY_DATASET_ROLE,
        "is_default": False,
        "issue": ISSUE,
        "plan": PLAN,
        "purpose": PURPOSE,
        "derived_from": {
            "file": Path(str(source["path"])).name,
            "sha256": source["sha256"],
            "rows": source["rows"],
            "release": source.get("release"),
        },
        "patch": {
            "tool": TOOL,
            "tool_commit": receipt.get("tool_commit"),
            "engine": receipt.get("engine"),
            "receipt": Path(receipt_path).name,
            "added_columns": [f"{r['entity']}.{r['column']}" for r in added],
        },
        "artifacts": {
            output_path.stem: {
                "kind": "microdata",
                "path": output_path.name,
                "repo_id": target.repo_id,
                "sha256": output["sha256"],
                "role": "dataset",
            },
            "patch_receipt": {
                "kind": "receipt",
                "path": Path(receipt_path).name,
                "repo_id": target.repo_id,
                "sha256": receipt_sha256,
                "role": "patch_receipt",
            },
        },
    }


def build_patch_steps(
    *,
    zero_columns: Sequence[str],
    engine: Any,
    asec_sources: AsecPersonSources | None = None,
) -> list[PatchStep]:
    """The steps the command runs, in order.

    With ``asec_sources``, the ASEC work-column restore step
    (:func:`asec_work_column_step`) goes first, then the reviewed zero
    columns. The command always passes ``asec_sources``.
    """

    steps = [] if asec_sources is None else [asec_work_column_step(asec_sources)]
    steps.append(reviewed_zero_column_step(zero_columns, engine=engine))
    return steps


def patch_donor_h5(
    input_h5: Path,
    output_h5: Path,
    *,
    steps: Sequence[PatchStep],
    receipt_path: Path,
    tool_identity: Mapping[str, object] | None = None,
    engine_identity: Mapping[str, object] | None = None,
    input_release_manifest: Path | None = None,
    release_manifest: CopyReleaseManifest | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write the patched copy and its receipt; return the receipt.

    The input must have every US entity table that R1's ``_load_base_frame``
    reads. The copy starts as a byte copy of the input; only the entity
    tables a step changes are rewritten, through the shared frame-table
    codec, in their stored HDF format. The copy is verified before it
    replaces ``output_h5``. ``input_release_manifest`` is checked before any
    work; ``release_manifest`` is written last, after the receipt. An
    existing ``output_h5``, receipt or ``release_manifest`` is refused unless
    ``overwrite``: a re-run writes a copy with another SHA-256.
    """

    input_h5, output_h5 = Path(input_h5), Path(output_h5)
    receipt_path = Path(receipt_path)
    _validate_paths(
        input_h5,
        output_h5,
        receipt_path,
        *(() if input_release_manifest is None else (Path(input_release_manifest),)),
        *(() if release_manifest is None else (Path(release_manifest.path),)),
    )
    _refuse_existing_outputs(
        (
            output_h5,
            receipt_path,
            *(() if release_manifest is None else (Path(release_manifest.path),)),
        ),
        overwrite=overwrite,
    )
    input_sha256 = refuse_denied_pool_h5(input_h5, consumer=_CONSUMER)
    input_release = (
        None
        if input_release_manifest is None
        else input_release_identity(Path(input_release_manifest), input_sha256)
    )
    with pd.HDFStore(input_h5, mode="r") as store:
        keys = sorted(store.keys())
        absent = [entity for entity in US_SCHEMA.entities if f"/{entity}" not in keys]
        if absent:
            raise ValueError(
                f"{input_h5} lacks entity table(s) {absent}; R1 reads every "
                f"US entity table ({list(US_SCHEMA.entities)})."
            )
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
        layouts = {
            entity: _stored_layout(store, entity, tables[entity])
            for entity in US_SCHEMA.entities
        }
        other_keys = [key for key in keys if key.lstrip("/") not in tables]
        others = {key: store[key] for key in other_keys}
        root_attributes = _root_attributes(store)
    assert_h5_unchanged(input_h5, input_sha256, consumer=_CONSUMER)

    patched, columns, step_receipts = _apply_steps(tables, steps)
    changed = sorted(
        entity for entity in patched if patched[entity] is not tables[entity]
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_name(f".{output_h5.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(input_h5, temporary)
        _rewrite_entity_tables(
            temporary, {entity: patched[entity] for entity in changed}, layouts
        )
        _verify_copy(
            temporary,
            original=tables,
            patched=patched,
            formats={entity: layout.format for entity, layout in layouts.items()},
            keys=keys,
            others=others,
            root_attributes=root_attributes,
        )
        os.replace(temporary, output_h5)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    assert_h5_unchanged(input_h5, input_sha256, consumer=_CONSUMER)

    receipt: dict[str, object] = {
        "version": RECEIPT_VERSION,
        "tool": TOOL,
        "tool_commit": dict(tool_identity) if tool_identity is not None else None,
        "issue": ISSUE,
        "plan": PLAN,
        "purpose": PURPOSE,
        "engine": dict(engine_identity) if engine_identity is not None else None,
        "input": {
            "path": str(input_h5.resolve()),
            "sha256": input_sha256,
            "rows": {entity: len(tables[entity]) for entity in US_SCHEMA.entities},
            "release": input_release,
        },
        "output": {
            "path": str(output_h5.resolve()),
            "sha256": _sha256(output_h5),
            "rows": {entity: len(patched[entity]) for entity in US_SCHEMA.entities},
        },
        "rewritten_tables": changed,
        "preserved": {
            "tables": keys,
            "root_attributes": sorted(root_attributes),
        },
        "added_columns": [r for r in columns if r["action"] == "added"],
        "kept_columns": [r for r in columns if r["action"] == "kept_existing"],
        "steps": step_receipts,
        "release_manifest": (
            None if release_manifest is None else release_manifest.identity()
        ),
    }
    receipt_sha256 = _write_json(receipt_path, receipt)
    if release_manifest is not None:
        _write_json(
            Path(release_manifest.path),
            copy_release_manifest(
                receipt,
                receipt_path=receipt_path,
                receipt_sha256=receipt_sha256,
                target=release_manifest,
                built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
    return receipt


def _write_json(path: Path, document: Mapping[str, object]) -> str:
    """Write ``document`` as sorted, indented JSON; return the bytes' SHA-256."""

    data = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _StoredLayout:
    """How the input stored one entity table, so the rewrite can keep it."""

    format: str
    data_columns: bool | tuple[str, ...] | None
    nullable_boolean_columns: tuple[str, ...]


def _stored_layout(
    store: pd.HDFStore, entity: str, table: pd.DataFrame
) -> _StoredLayout:
    storer = store.get_storer(entity)
    raw_codec = getattr(storer.attrs, _PYTABLES_FRAME_TABLE_CODEC_ATTR, None)
    nullable = (
        tuple(json.loads(raw_codec)["nullable_boolean_columns"])
        if raw_codec is not None
        else ()
    )
    if not storer.is_table:
        return _StoredLayout("fixed", None, nullable)
    data_columns = tuple(getattr(storer, "data_columns", None) or ())
    if data_columns and set(data_columns) == set(table.columns):
        return _StoredLayout("table", True, nullable)
    return _StoredLayout("table", data_columns or None, nullable)


def _rewrite_entity_tables(
    path: Path,
    tables: Mapping[str, pd.DataFrame],
    layouts: Mapping[str, _StoredLayout],
) -> None:
    """Replace the changed entity tables in the copy, in their stored layout.

    Columns the input's frame-table codec declared nullable Boolean go back
    to ``boolean`` dtype first, so the shared codec writes them, and its
    table attribute, as the input had them.
    """

    with pd.HDFStore(path, mode="a") as store:
        for entity, table in tables.items():
            layout = layouts[entity]
            if layout.nullable_boolean_columns:
                table = table.copy(deep=False)
                for column in layout.nullable_boolean_columns:
                    table[column] = table[column].astype("boolean")
            put_frame_table(
                store,
                entity,
                table,
                preferred_format=layout.format,
                data_columns=(
                    list(layout.data_columns)
                    if isinstance(layout.data_columns, tuple)
                    else layout.data_columns
                ),
            )


def _apply_steps(
    tables: Mapping[str, pd.DataFrame],
    steps: Sequence[PatchStep],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    patched = dict(tables)
    added: list[dict[str, object]] = []
    step_receipts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    names = [step.name for step in steps]
    if len(set(names)) != len(names):
        raise ValueError(f"Patch step names must be distinct; got {names}.")
    for step in steps:
        result = step.run(MappingProxyType(dict(patched)))
        records = []
        for patch in result.patches:
            key = (patch.entity, patch.column)
            if key in seen:
                raise ValueError(
                    f"{patch.entity}.{patch.column}: two patches name the column."
                )
            seen.add(key)
            if patch.entity not in patched:
                raise ValueError(
                    f"{step.name}: no {patch.entity!r} table for {patch.column}."
                )
            table = patched[patch.entity]
            values = patch.values
            if not isinstance(values, pd.Series) or not values.index.equals(
                table.index
            ):
                raise ValueError(
                    f"{step.name}: {patch.entity}.{patch.column} values must be "
                    "a Series on the table's index."
                )
            action = "added"
            if patch.column in table.columns:
                stored = table[patch.column]
                if not _same_values(stored, values):
                    raise ValueError(
                        f"{step.name}: {patch.entity}.{patch.column} is already "
                        "in the donor with other values (a nonzero, null or "
                        "different value); refusing to overwrite it."
                    )
                action = "kept_existing"
            else:
                table = table.copy(deep=False)
                table[patch.column] = values.rename(patch.column)
                patched[patch.entity] = table
            record = {
                **dict(patch.record),
                "column": patch.column,
                "entity": patch.entity,
                "dtype": str(patched[patch.entity][patch.column].dtype),
                "step": step.name,
                "action": action,
            }
            records.append(record)
            added.append(record)
        step_receipts.append(
            {
                **dict(result.receipt),
                "name": step.name,
                "columns": [f"{r['entity']}.{r['column']}" for r in records],
            }
        )
    return patched, added, step_receipts


def _same_values(stored: pd.Series, values: pd.Series) -> bool:
    """Whether a stored column already holds exactly the patch values."""

    if stored.isna().any() or values.isna().any():
        return False
    try:
        left = pd.to_numeric(stored, errors="raise").to_numpy(dtype=np.float64)
        right = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        return bool(stored.astype(object).equals(values.astype(object)))
    return bool(np.array_equal(left, right))


def _verify_copy(
    path: Path,
    *,
    original: Mapping[str, pd.DataFrame],
    patched: Mapping[str, pd.DataFrame],
    formats: Mapping[str, str],
    keys: Sequence[str],
    others: Mapping[str, Any],
    root_attributes: Mapping[str, object],
) -> None:
    """Refuse a copy that lost or changed anything but the added columns."""

    with pd.HDFStore(path, mode="r") as store:
        if sorted(store.keys()) != list(keys):
            raise RuntimeError(
                f"Patched copy changed the table list: {sorted(store.keys())} "
                f"against {list(keys)}."
            )
        if _root_attributes(store) != dict(root_attributes):
            raise RuntimeError("Patched copy changed the HDF5 root attributes.")
        for key, expected in others.items():
            stored = store[key]
            if type(stored) is not type(expected) or not stored.equals(expected):
                raise RuntimeError(f"Patched copy changed {key!r}.")
        for entity, before in original.items():
            stored = read_frame_table(store, entity)
            after = patched[entity]
            if list(stored.columns) != list(after.columns):
                raise RuntimeError(
                    f"Patched copy has {entity!r} columns {list(stored.columns)}; "
                    f"expected {list(after.columns)}."
                )
            try:
                pd.testing.assert_frame_equal(
                    stored[list(before.columns)], before, check_exact=True
                )
                pd.testing.assert_frame_equal(stored, after, check_exact=True)
            except AssertionError as error:
                raise RuntimeError(
                    f"Patched copy changed {entity!r}: {error}"
                ) from error
            stored_format = "table" if store.get_storer(entity).is_table else "fixed"
            if stored_format != formats[entity]:
                raise RuntimeError(
                    f"Patched copy stored {entity!r} in {stored_format} format; "
                    f"the input used {formats[entity]}."
                )


def _root_attributes(store: pd.HDFStore) -> dict[str, object]:
    attributes = store.get_node("/")._v_attrs
    return {
        name: _attribute_value(attributes[name])
        for name in sorted(attributes._v_attrnames)
    }


def _attribute_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_paths(
    input_h5: Path, output_h5: Path, receipt_path: Path, *manifests: Path
) -> None:
    if not input_h5.is_file():
        raise ValueError(f"--input-h5 is not a file: {input_h5}.")
    if output_h5.suffix != ".h5":
        raise ValueError(f"--output-h5 must end with .h5, got {output_h5.name!r}.")
    resolved = {input_h5.resolve(), output_h5.resolve()}
    if len(resolved) != 2:
        raise ValueError("--output-h5 must differ from --input-h5.")
    if receipt_path.resolve() in resolved:
        raise ValueError("--receipt must differ from --input-h5 and --output-h5.")
    paths = [*resolved, receipt_path.resolve(), *(path.resolve() for path in manifests)]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "--input-h5, --output-h5, --receipt, --input-release-manifest and "
            "--release-manifest must be different files."
        )


def _refuse_existing_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    """Refuse to replace an existing output unless asked.

    A re-run writes a copy with another SHA-256 (the HDF5 object headers
    record write times), so a staging summary that pinned the first copy
    (``base.sha256``) would name bytes no longer at its path.
    """

    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            f"Output(s) already exist: {existing}. A re-run writes a copy with "
            "another SHA-256, so a staging summary built on the first copy "
            "would no longer match it. Write to a new path, or pass --overwrite "
            "and re-run R1 on the new copy."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def current_tool_identity(repository: Path = _REPOSITORY_ROOT) -> dict[str, object]:
    """The commit this tool runs from, and whether its file is edited."""

    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", TOOL)
    return {
        "commit": commit,
        "tool_dirty": None if status is None else bool(status),
    }


def _engine_identity() -> dict[str, object]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("policyengine-us")
    except PackageNotFoundError:
        installed = None
    return {"package": "policyengine-us", "version": installed}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a patched copy of a US national donor H5 for the ACS "
            "staging, for the County Files proof of concept only."
        )
    )
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="JSON receipt path (default: OUTPUT-H5 with a .receipt.json suffix).",
    )
    parser.add_argument(
        "--zero-column",
        action="append",
        choices=sorted(REVIEWED_ZERO_COLUMNS),
        help=(
            "Reviewed column to add as zeros on its engine entity; repeat for "
            "more than one (default: every reviewed column)."
        ),
    )
    parser.add_argument(
        "--asec-source-cache",
        type=Path,
        help=(
            "Directory holding the pinned ASEC person members (pppubYY.csv) or "
            "their official archives (asecpubYYcsv.zip), for the work-column "
            f"restore (default: {DEFAULT_ASEC_SOURCE_CACHE})."
        ),
    )
    parser.add_argument(
        "--asec-fetch",
        action="store_true",
        help=(
            "Download a pinned ASEC archive absent from the cache (verified "
            "against its pin) instead of refusing."
        ),
    )
    parser.add_argument(
        "--input-release-manifest",
        type=Path,
        help=(
            "The published donor's release_manifest.json. The input's SHA-256 "
            "must match its one microdata artifact; the receipt records the "
            "donor build it names."
        ),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help=(
            "Write a build-folder release manifest for the copy here, for R1's "
            "--donor-release-manifest. Needs --manifest-release and "
            "--manifest-build. Upload the copy, the receipt and this file (as "
            "release_manifest.json) to releases/<release>/<build>/ in "
            "--manifest-repo-id."
        ),
    )
    parser.add_argument(
        "--manifest-repo-id",
        default=DEFAULT_MANIFEST_REPO_ID,
        help=f"Repository the copy is uploaded to (default: {DEFAULT_MANIFEST_REPO_ID}).",
    )
    parser.add_argument("--manifest-release", help="Release folder name of the copy.")
    parser.add_argument("--manifest-build", help="Build folder name of the copy.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing --output-h5, receipt and --release-manifest "
            "(refused by default). The new copy has another SHA-256 (the HDF5 "
            "object headers record write times), so re-run R1 on it: a staging "
            "summary built on the old copy no longer matches."
        ),
    )
    args = parser.parse_args(argv)
    names = (args.manifest_release, args.manifest_build)
    if args.release_manifest is None and any(name is not None for name in names):
        parser.error("--manifest-release and --manifest-build need --release-manifest.")
    if args.release_manifest is not None and any(name is None for name in names):
        parser.error(
            "--release-manifest needs --manifest-release and --manifest-build."
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt_path = args.receipt or args.output_h5.with_suffix(".receipt.json")
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    try:
        steps = build_patch_steps(
            zero_columns=args.zero_column or list(REVIEWED_ZERO_COLUMNS),
            engine=PolicyEngineUSEngine(),
            asec_sources=AsecPersonSources(
                cache=args.asec_source_cache, fetch=args.asec_fetch
            ),
        )
        release_manifest = (
            None
            if args.release_manifest is None
            else CopyReleaseManifest(
                path=args.release_manifest,
                release=args.manifest_release,
                build=args.manifest_build,
                repo_id=args.manifest_repo_id,
            )
        )
        receipt = patch_donor_h5(
            args.input_h5,
            args.output_h5,
            steps=steps,
            receipt_path=receipt_path,
            tool_identity=current_tool_identity(),
            engine_identity=_engine_identity(),
            input_release_manifest=args.input_release_manifest,
            release_manifest=release_manifest,
            overwrite=args.overwrite,
        )
    except ValueError as error:
        raise SystemExit(f"Donor patch refused: {error}") from error
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
