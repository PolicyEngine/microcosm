"""Restore the reviewed Census person columns the pinned ASEC H5 inputs lack.

Microcosm #720. The base pools three processed ASEC HDF5 inputs (pins in
:mod:`.asec_sources`: ``census_cps_2022.h5`` 7ccca976…, ``census_cps_2023.h5``
cb578173…, ``census_cps_2024.h5`` ec36604c…). The income-year 2022 and 2023
files were extracted with an older column list: they carry 2 of the 18
``NOW_*`` at-interview coverage recodes (``NOW_GRP``, ``NOW_MRK``) and lack
``A_EXPRRP``, ``PTOTVAL``, ``A_ENRLW`` and ``A_FTPT``, which the 2024 file
carries. :func:`.asec_pool.pool_asec_sources` concatenates per-year person
tables, so every 2022/2023 person gets NaN in a column only the 2024 file
carries, and each reader treats NaN as "no" or zero: seven reported-coverage
flags are False for two-thirds of the pool (the 14 failures of the #744
``us_reported_coverage_vintage_signal_gate``), the 2022/2023 relationship
recode falls back to a line/spouse/parent derivation, and tax-unit
construction sees no enrollment or total-income signal for those vintages.

The complete Census person files those inputs were extracted from are already
pinned build inputs: the base stage's ``--asec-education-source`` names the
official archives (``asecpub23csv.zip`` d2e00025…, ``asecpub24csv.zip``,
``asecpub25csv.zip``), pinned in
:data:`.spm_role_source.ASEC_SPM_ROLE_SOURCES` (archive SHA-256, member name,
member size and member SHA-256) and read through
:func:`.spm_role_source.read_pinned_asec_person_columns`, the reader the SPM
role stage uses. :func:`restore_asec_census_person_columns` appends, per
vintage and before any pooling, every reviewed column the H5 lacks, read from
that vintage's pinned person member by exact ``PERIDNUM`` identity. It never
predicts, imputes or defaults a value, and it never overwrites a column the H5
carries: such a column is instead checked equal to the archive, which on the
2024 vintage (where the H5 carries all of them) validates the join on real
data. See ``docs/us-asec-census-person-columns.md`` for the review.

Reviewed columns and their build readers (grep of ``packages/*/src`` and
``tools/`` on 2026-09-23; a column is restored only if a build reader exists):

======================  ======================================================
Column                  Build reader
======================  ======================================================
``NOW_MCAID``           ``cps_carried._fill_health_coverage_inputs`` ->
                        ``has_medicaid_health_coverage_at_interview``
``NOW_NONM``            same -> ``has_non_marketplace_direct_purchase_…``
``NOW_CHAMPVA``         same -> ``has_champva_health_coverage_at_interview``
``NOW_MIL``             same -> ``has_tricare_health_coverage_at_interview``
``NOW_VACARE``          same -> ``has_va_health_coverage_at_interview``
``NOW_OTHMT``           same -> ``has_other_means_tested_…_at_interview``
``NOW_IHSFLG``          same -> ``has_indian_health_service_coverage_…``
``A_EXPRRP``            ``asec_pool._with_relationship_recode`` (the Census
                        recode instead of the derived fallback);
                        ``microcosm.frame.units.MICROUNIT_REQUIRED_COLUMNS``
                        -> microunit tax-unit head choice;
                        ``acs_transfer._person_head_feature``; the
                        ``optional_household_head`` requirement in
                        ``us_late_producer_registry._transfer_input_inventory``
``PTOTVAL``             ``microcosm.frame.units._is_microunit_optional_column``
                        -> microunit ``_precompute_tax_unit_inputs`` (total
                        money income; a pooled NaN reads as 0)
``A_ENRLW``             same -> microunit ``_is_full_time_student``
``A_FTPT``              same; ``eligibility_inputs``
                        ``derive_us_eligibility_inputs_from_manifest``
                        (``is_full_time_college_student``)
``A_LFSR``              ``immigration._assign_ssn_card_codes`` (the
                        ``immigration_status`` stage's worker EAD spill; the
                        stage refuses without it)
======================  ======================================================

The other thirteen columns the 2026-08-23 offline fix appended are not
restored; :data:`ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED` records why.

``A_LFSR`` is not one of #720's columns. None of the three H5 inputs carries
it, and the native line's immigration method binds its undocumented-worker
spill to actual ASEC labor-force status (codes 1-4 at ages 16+) instead of
prior-year earnings. :data:`ASEC_CENSUS_PERSON_COLUMNS_BEYOND_720` records it
separately, so the #720 review stays exactly the offline fix's column list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .spm_role_source import (
    ASEC_SPM_ROLE_SOURCES,
    AsecSpmRoleSource,
    _exact_person_keys,
    read_pinned_asec_person_columns,
)

__all__ = [
    "ASEC_CENSUS_PERSON_COLUMNS",
    "ASEC_CENSUS_PERSON_COLUMNS_BEYOND_720",
    "ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED",
    "ASEC_CENSUS_PERSON_COLUMN_NAMES",
    "ASEC_CENSUS_PERSON_IDENTITY_COLUMNS",
    "AsecCensusPersonColumn",
    "AsecCensusPersonColumnsError",
    "restore_asec_census_person_columns",
]


class AsecCensusPersonColumnsError(ValueError):
    """The #720 Census person-column restoration refused its inputs."""


@dataclass(frozen=True)
class AsecCensusPersonColumn:
    """One reviewed Census person column and the build code that reads it."""

    name: str
    #: Census codes the column may take; ``None`` admits any integer
    #: (``PTOTVAL`` is a signed dollar amount).
    domain: frozenset[int] | None
    readers: tuple[str, ...]


_NOW_YES_NO = frozenset({1, 2})
_ENROLLMENT = frozenset({0, 1, 2})
_HEALTH_COVERAGE_READER = (
    "microcosm.build.us_runtime.cps_carried._fill_health_coverage_inputs"
)
_MICROUNIT_OPTIONAL_READER = (
    "microcosm.frame.units._is_microunit_optional_column (passes it to microunit)"
)

#: The reviewed restoration set, in the order columns are appended. Domains are
#: the Census codes observed in all three pinned person members (pppub23/24/25,
#: 2026-09-23); ``A_EXPRRP`` has no code 6 in the Census codebook or the data.
ASEC_CENSUS_PERSON_COLUMNS: tuple[AsecCensusPersonColumn, ...] = (
    AsecCensusPersonColumn(
        "NOW_MCAID",
        _NOW_YES_NO,
        (f"{_HEALTH_COVERAGE_READER} -> has_medicaid_health_coverage_at_interview",),
    ),
    AsecCensusPersonColumn(
        "NOW_NONM",
        _NOW_YES_NO,
        (
            f"{_HEALTH_COVERAGE_READER} -> "
            "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        ),
    ),
    AsecCensusPersonColumn(
        "NOW_CHAMPVA",
        _NOW_YES_NO,
        (f"{_HEALTH_COVERAGE_READER} -> has_champva_health_coverage_at_interview",),
    ),
    AsecCensusPersonColumn(
        "NOW_MIL",
        _NOW_YES_NO,
        (f"{_HEALTH_COVERAGE_READER} -> has_tricare_health_coverage_at_interview",),
    ),
    AsecCensusPersonColumn(
        "NOW_VACARE",
        _NOW_YES_NO,
        (f"{_HEALTH_COVERAGE_READER} -> has_va_health_coverage_at_interview",),
    ),
    AsecCensusPersonColumn(
        "NOW_OTHMT",
        _NOW_YES_NO,
        (
            f"{_HEALTH_COVERAGE_READER} -> "
            "has_other_means_tested_health_coverage_at_interview",
        ),
    ),
    AsecCensusPersonColumn(
        "NOW_IHSFLG",
        _NOW_YES_NO,
        (
            f"{_HEALTH_COVERAGE_READER} -> "
            "has_indian_health_service_coverage_at_interview",
        ),
    ),
    AsecCensusPersonColumn(
        "A_EXPRRP",
        frozenset({1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14}),
        (
            "microcosm.build.us_runtime.asec_pool._with_relationship_recode "
            "(source recode instead of the line/spouse/parent fallback)",
            "microcosm.frame.units.MICROUNIT_REQUIRED_COLUMNS -> "
            "microunit.tax_unit_construction._prepare_household_people "
            "(relationship code for the tax-unit head choice)",
            "microcosm.build.us_runtime.acs_transfer._person_head_feature",
            "microcosm.build.us_runtime.us_late_producer_registry."
            "_transfer_input_inventory (optional_household_head)",
        ),
    ),
    AsecCensusPersonColumn(
        "PTOTVAL",
        None,
        (
            _MICROUNIT_OPTIONAL_READER,
            "microunit.tax_unit_construction._precompute_tax_unit_inputs "
            "(total money income; a pooled NaN reads as 0)",
        ),
    ),
    AsecCensusPersonColumn(
        "A_ENRLW",
        _ENROLLMENT,
        (
            _MICROUNIT_OPTIONAL_READER,
            "microunit.tax_unit_construction._is_full_time_student",
        ),
    ),
    AsecCensusPersonColumn(
        "A_FTPT",
        _ENROLLMENT,
        (
            _MICROUNIT_OPTIONAL_READER,
            "microunit.tax_unit_construction._is_full_time_student",
            "microcosm.build.us_runtime.eligibility_inputs."
            "derive_us_eligibility_inputs_from_manifest "
            "(is_full_time_college_student <- A_HSCOL == 2 & A_FTPT == 1)",
        ),
    ),
    # Observed complete in all three pinned members (pppub23/24/25,
    # 2026-09-26): codes 0 (not in universe, every person under 15), 1-4 (in
    # the labor force) and 7 (not in the labor force).
    AsecCensusPersonColumn(
        "A_LFSR",
        frozenset({0, 1, 2, 3, 4, 7}),
        (
            "microcosm.build.us_runtime.immigration."
            "US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS (the immigration_status "
            "stage refuses a person table without it)",
            "microcosm.build.us_runtime.immigration._assign_ssn_card_codes "
            "(worker EAD spill: A_LFSR 1-4 at ages 16+)",
        ),
    ),
)

#: Restored columns that are not among #720's offline-fix columns. Each has a
#: build reader that arrived after #720; see the module docstring.
ASEC_CENSUS_PERSON_COLUMNS_BEYOND_720: frozenset[str] = frozenset({"A_LFSR"})
ASEC_CENSUS_PERSON_COLUMN_NAMES: tuple[str, ...] = tuple(
    column.name for column in ASEC_CENSUS_PERSON_COLUMNS
)

_UNREAD = (
    "No build reader: no code under packages/*/src or tools/ reads it "
    "(2026-09-23), so restoring it would only widen the exported person table."
)
_SELF_CHECK = (
    "Read only as an optional cross-check by the SPM role derivation "
    "(spm_role_source._OPTIONAL_RAW_CHECKS), which reads this same pinned member "
    "itself and derives the role from it, not from the frame; restoring it from "
    "that member would make the check compare the member with itself. #38's "
    "unmarried-partner derivation, if built, should restore it with its reader."
)

_NATIVE_MEMBER_READER = (
    "Read only by the native current-survey health projection "
    "(current_survey_health_coverage via current_survey_health_source), which "
    "captures its own pinned ASEC person member and never reads the pooled H5 "
    "frame; restoring it into the H5 would add a column no frame reader uses."
)
#: Columns the 2026-08-23 offline fix (``receipt_720.json``) appended that this
#: restoration deliberately leaves out, and why. ``CENSUS_TAX_ID`` was never
#: appended: it is extractor-derived and absent from the Census files.
ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED: MappingProxyType[str, str] = MappingProxyType(
    {
        "NOW_CAID": _NATIVE_MEMBER_READER,
        "NOW_COV": _UNREAD,
        "NOW_DIR": _UNREAD,
        "NOW_MCARE": _UNREAD,
        "NOW_MRKS": _UNREAD,
        "NOW_MRKUN": _UNREAD,
        "NOW_PCHIP": _UNREAD,
        "NOW_PRIV": _UNREAD,
        "NOW_PUB": _UNREAD,
        "A_FAMTYP": _SELF_CHECK,
        "A_FAMREL": _SELF_CHECK,
        "PECOHAB": _SELF_CHECK,
        "LKWEEKS": (
            "Already restored for income year 2022, the only vintage lacking it, "
            "by weeks_unemployed.fill_asec_2022_weeks_unemployed_source from the "
            "same pinned pppub23.csv (asecpub23csv.zip d2e00025…) before its only "
            "reader; census_cps_2023.h5 and census_cps_2024.h5 carry it."
        ),
    }
)

#: Census identity carried by both the H5 and the person member. After the
#: PERIDNUM join each must agree row for row, so a join that pairs the wrong
#: people cannot pass on a coincidence of keys.
ASEC_CENSUS_PERSON_IDENTITY_COLUMNS: tuple[str, ...] = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "A_AGE",
)

_JOIN_KEY = "PERIDNUM"
_EXAMPLES = 5


def _refuse(message: str) -> None:
    raise AsecCensusPersonColumnsError(message)


def _is_integer(values: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(values) and not pd.api.types.is_bool_dtype(
        values
    )


def _examples(values: pd.Series | pd.Index) -> list[str]:
    return [str(value) for value in list(values)[:_EXAMPLES]]


def _resolve_pin(income_year: int, pin: AsecSpmRoleSource | None) -> AsecSpmRoleSource:
    if pin is None:
        pin = ASEC_SPM_ROLE_SOURCES.get(income_year)
        if pin is None:
            _refuse(
                "No pinned Census ASEC person source covers income year "
                f"{income_year}; pinned income years: "
                f"{sorted(ASEC_SPM_ROLE_SOURCES)}."
            )
    if pin.income_year != income_year or pin.survey_year != income_year + 1:
        _refuse(
            f"Census ASEC person pin is for income year {pin.income_year} (survey "
            f"{pin.survey_year}); the H5 input is income year {income_year}."
        )
    return pin


def _read_member(
    path: Path, pin: AsecSpmRoleSource, label: str
) -> tuple[pd.DataFrame, str]:
    columns = (_JOIN_KEY, *ASEC_CENSUS_PERSON_IDENTITY_COLUMNS)
    columns += ASEC_CENSUS_PERSON_COLUMN_NAMES
    try:
        source, form = read_pinned_asec_person_columns(path, pin, label, columns)
    except ValueError as error:
        # Pin mismatches (already labelled) and a member header missing a
        # reviewed column (pandas' usecols message, unlabelled).
        message = str(error)
        raise AsecCensusPersonColumnsError(
            message if message.startswith(label) else f"{label}: {message}"
        ) from error
    if len(source) != pin.persons:
        _refuse(f"{label} holds {len(source)} persons; the pin records {pin.persons}.")
    for column in columns[1:]:
        if not _is_integer(source[column]):
            _refuse(
                f"{label} column {column} parsed as {source[column].dtype}, not "
                "integer: the member has blank, fractional or text values there."
            )
    for spec in ASEC_CENSUS_PERSON_COLUMNS:
        if spec.domain is None:
            continue
        outside = ~source[spec.name].isin(spec.domain)
        if outside.any():
            _refuse(
                f"{label} column {spec.name} has {int(outside.sum())} values "
                f"outside its reviewed Census codes {sorted(spec.domain)}, e.g. "
                f"{_examples(source.loc[outside, spec.name].drop_duplicates())}."
            )
    return source, form


def _one_to_one_positions(
    h5_keys: pd.Series, source_keys: pd.Series, *, h5_label: str, label: str
) -> np.ndarray:
    """Positions of each H5 person's archive row; refuse anything not a bijection."""

    for side, keys in ((h5_label, h5_keys), (label, source_keys)):
        duplicated = keys.duplicated(keep=False)
        if duplicated.any():
            _refuse(
                f"{side} repeats {int(duplicated.sum())} PERIDNUM rows, e.g. "
                f"{_examples(keys[duplicated].drop_duplicates())}; the join must "
                "be one-to-one."
            )
    source_index = pd.Index(source_keys)
    positions = source_index.get_indexer(h5_keys)
    missing = positions < 0
    if missing.any():
        _refuse(
            f"{int(missing.sum())} {h5_label} persons have no row in {label}, e.g. "
            f"PERIDNUM {_examples(h5_keys[missing])}; the join must be total."
        )
    extra = ~source_index.isin(h5_keys)
    if extra.any():
        _refuse(
            f"{int(extra.sum())} {label} persons are absent from {h5_label}, e.g. "
            f"PERIDNUM {_examples(source_keys[extra])}; the H5 must be the complete "
            "person universe of its Census member."
        )
    return positions


def _value_summary(spec: AsecCensusPersonColumn, values: pd.Series) -> dict[str, int]:
    if spec.domain is None:
        return {
            "nonzero_rows": int(values.ne(0).sum()),
            "negative_rows": int(values.lt(0).sum()),
        }
    counts = values.value_counts()
    return {str(code): int(counts.get(code, 0)) for code in sorted(spec.domain)}


def restore_asec_census_person_columns(
    person: pd.DataFrame,
    *,
    income_year: int,
    source_path: str | Path,
    pin: AsecSpmRoleSource | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Append the reviewed Census person columns one ASEC H5 input lacks.

    ``person`` is one vintage's complete raw person table as read from its
    pinned H5 (before pooling or any smoke-household limit). ``source_path``
    is that income year's pinned Census person file: the official archive or
    its extracted CSV, verified against ``pin`` (default:
    :data:`.spm_role_source.ASEC_SPM_ROLE_SOURCES` for ``income_year``; an
    explicit pin is for synthetic tests).

    Refuses, with :class:`AsecCensusPersonColumnsError`, when the H5 lacks the
    join key or an identity column; when the archive or member fails its pins
    or its person count; when a member column is not integer or leaves its
    reviewed Census codes; when either side repeats a ``PERIDNUM``, an H5
    person has no member row, or a member person is absent from the H5; when
    ``PH_SEQ``/``P_SEQ``/``A_LINENO``/``A_AGE`` disagree after the join; and
    when a reviewed column the H5 already carries is not integer or differs
    from the member on any row. A column the H5 carries is never overwritten.
    New columns are appended as ``int64`` after the H5's own, in reviewed
    order; every H5 column keeps its position, dtype and values (checked).

    Returns the widened table and a JSON-ready provenance record.
    """

    pin = _resolve_pin(int(income_year), pin)
    label = f"ASEC {pin.survey_year} Census person member {pin.member}"
    h5_label = f"income-year {income_year} ASEC H5 person table"
    required = (_JOIN_KEY, *ASEC_CENSUS_PERSON_IDENTITY_COLUMNS)
    missing_columns = [column for column in required if column not in person.columns]
    if missing_columns:
        _refuse(f"{h5_label} lacks join/identity column(s) {missing_columns}.")
    if person.columns.duplicated().any():
        _refuse(f"{h5_label} repeats column names; cannot append by name.")
    if not person.index.is_unique:
        _refuse(f"{h5_label} has a repeated row index; cannot append by row.")

    path = Path(source_path).expanduser()
    source, form = _read_member(path, pin, label)
    try:
        h5_keys = _exact_person_keys(person[_JOIN_KEY], h5_label).reset_index(drop=True)
        source_keys = _exact_person_keys(source[_JOIN_KEY], label).reset_index(
            drop=True
        )
    except ValueError as error:
        raise AsecCensusPersonColumnsError(str(error)) from error
    positions = _one_to_one_positions(
        h5_keys, source_keys, h5_label=h5_label, label=label
    )
    aligned = source.iloc[positions].reset_index(drop=True)
    if not np.array_equal(source_keys.to_numpy()[positions], h5_keys.to_numpy()):
        _refuse(f"{label} rows did not align to {h5_label} by PERIDNUM.")

    for column in ASEC_CENSUS_PERSON_IDENTITY_COLUMNS:
        if not _is_integer(person[column]) or person[column].isna().any():
            _refuse(
                f"{h5_label} identity column {column} is {person[column].dtype}, "
                "not integer."
            )
        disagree = person[column].to_numpy() != aligned[column].to_numpy()
        if disagree.any():
            _refuse(
                f"{h5_label} {column} disagrees with {label} on "
                f"{int(disagree.sum())} PERIDNUM-joined rows, e.g. PERIDNUM "
                f"{_examples(h5_keys[disagree])}."
            )

    added: dict[str, np.ndarray] = {}
    verified: list[str] = []
    values: dict[str, dict[str, int]] = {}
    for spec in ASEC_CENSUS_PERSON_COLUMNS:
        member_values = aligned[spec.name].to_numpy(dtype=np.int64)
        if spec.name in person.columns:
            carried = person[spec.name]
            if not _is_integer(carried) or carried.isna().any():
                _refuse(
                    f"{h5_label} already carries {spec.name} as {carried.dtype} "
                    "(missing or non-integer values); it cannot be verified "
                    "against the Census member and is never overwritten."
                )
            disagree = carried.to_numpy() != member_values
            if disagree.any():
                _refuse(
                    f"{h5_label} {spec.name} disagrees with {label} on "
                    f"{int(disagree.sum())} PERIDNUM-joined rows, e.g. PERIDNUM "
                    f"{_examples(h5_keys[disagree])}."
                )
            verified.append(spec.name)
        else:
            added[spec.name] = member_values
        values[spec.name] = _value_summary(spec, pd.Series(member_values))

    result = pd.concat(
        [person, pd.DataFrame(added, index=person.index)],
        axis=1,
    )
    width = len(person.columns)
    unchanged = (
        result.index.equals(person.index)
        and list(result.columns[:width]) == list(person.columns)
        and list(result.columns[width:]) == list(added)
        and all(
            result[column].dtype == person[column].dtype
            and result[column].equals(person[column])
            for column in person.columns
        )
    )
    if not unchanged:
        _refuse(f"Restoring Census columns changed an existing {h5_label} column.")

    return result, {
        "operation": "exact_source_join",
        "join_key": "PERIDNUM (exact 22-digit string), one-to-one and total",
        "income_year": pin.income_year,
        "survey_year": pin.survey_year,
        "source_path": str(path.resolve()),
        "source_form": form,
        "official_archive_url": pin.official_archive_url,
        "archive_sha256": pin.archive_sha256,
        "member": pin.member,
        "member_sha256": pin.csv_sha256,
        "member_size_bytes": pin.csv_size_bytes,
        "h5_person_rows": len(person),
        "member_person_rows": len(source),
        "joined_person_rows": len(aligned),
        "identity_columns_verified_equal": list(ASEC_CENSUS_PERSON_IDENTITY_COLUMNS),
        "reviewed_columns": list(ASEC_CENSUS_PERSON_COLUMN_NAMES),
        "columns_added": list(added),
        "columns_verified_equal": verified,
        "member_values": values,
    }
