"""Declared ACS PUMS earnings-source universes for stacked builds.

The 2024 ACS PUMS ``WAGP`` and ``SEMP`` fields apply to people age 15 and
older.  Census encodes younger people as not applicable (blank), not as an
observed zero.  Stacked builds preserve those person-grain blanks and apply
the documented universe only where an additive tax-unit feature needs them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from populace.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_PUMS_2024_DATA_DICTIONARY_URL",
    "ACS_PUMS_EARNINGS_MINIMUM_AGE",
    "ACS_PUMS_EARNINGS_SOURCE_COLUMNS",
    "AcsPumsEarningsUniverse",
    "resolve_acs_pums_earnings_universe",
]

ACS_PUMS_2024_DATA_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
    "data_dict/PUMS_Data_Dictionary_2024.pdf"
)
ACS_PUMS_EARNINGS_MINIMUM_AGE = 15
ACS_PUMS_EARNINGS_SOURCE_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "employment_income_before_lsr": "WAGP",
        "self_employment_income_before_lsr": "SEMP",
    }
)

_ACS_SUPPORT_CHANNEL = "acs"
_RULE_VERSION = 1
_UNIVERSE_DESCRIPTION = "ACS persons age 15 and older"
_AGGREGATION_SEMANTICS = (
    "sum eligible person values; an empty eligible set is the derived "
    "additive identity 0.0"
)


@dataclass(frozen=True)
class AcsPumsEarningsUniverse:
    """Exact structural-absence masks plus their JSON-ready receipt."""

    structural_absence_masks: Mapping[str, pd.Series]
    receipt: Mapping[str, object]


def resolve_acs_pums_earnings_universe(
    frame: Frame,
    *,
    columns: Sequence[str],
    person_scope: Sequence[bool] | pd.Series | np.ndarray | None = None,
    boundary: str,
) -> AcsPumsEarningsUniverse:
    """Resolve and validate the exact age-based ACS earnings universe.

    ``person_scope`` identifies the rows whose source role is being consumed.
    Primary-QRF preparation passes its clone-recipient rows; post-PUF QBI
    reconciliation passes native rows only.  The function never mutates the
    person table.  A null is authorized only on an ACS row below age 15, and
    an observed value on such a row is also rejected so the declaration is an
    exact source equation rather than a one-way tolerance.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError(f"{boundary}: ACS earnings universes require the US schema.")
    requested = tuple(dict.fromkeys(columns))
    unsupported = sorted(set(requested) - set(ACS_PUMS_EARNINGS_SOURCE_COLUMNS))
    if unsupported:
        raise ValueError(
            f"{boundary}: unsupported ACS earnings-universe column(s) {unsupported}."
        )

    person = frame.table("person")
    required = {
        "age",
        "person_tax_unit_id",
        support_channel_column("person"),
        support_clone_index_column("person"),
        *requested,
    }
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"{boundary}: ACS earnings universe cannot resolve required person "
            f"column(s) {missing}."
        )

    scope = _aligned_scope(person, person_scope, boundary=boundary)
    channel = person[support_channel_column("person")].astype(str)
    acs_scope = scope & channel.eq(_ACS_SUPPORT_CHANNEL)
    age = pd.to_numeric(person["age"], errors="coerce")
    invalid_age = acs_scope & (~np.isfinite(age.to_numpy(dtype=np.float64)))
    if invalid_age.any():
        raise ValueError(
            f"{boundary}: ACS earnings universe has {int(invalid_age.sum())} "
            "missing or nonfinite age value(s)."
        )
    structurally_absent = acs_scope & age.lt(ACS_PUMS_EARNINGS_MINIMUM_AGE)
    eligible_acs = acs_scope & ~structurally_absent

    masks: dict[str, pd.Series] = {}
    rule_receipts: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for column in requested:
        values = person[column]
        null = values.isna()
        in_universe_null = int((eligible_acs & null).sum())
        out_of_universe_nonnull = int((structurally_absent & ~null).sum())
        if in_universe_null or out_of_universe_nonnull:
            failures.append(
                f"{column}: in_universe_null_rows={in_universe_null}, "
                "out_of_universe_nonnull_rows="
                f"{out_of_universe_nonnull}"
            )
        source_column = ACS_PUMS_EARNINGS_SOURCE_COLUMNS[column]
        raw_source_present = source_column in person.columns
        raw_source_mismatch_rows = 0
        if not raw_source_present and acs_scope.any():
            failures.append(
                f"{source_column}: raw_source_authority_missing_for_scoped_acs_rows"
            )
        elif raw_source_present:
            raw_null = person[source_column].isna()
            raw_source_mismatch_rows = int(
                (
                    acs_scope
                    & (raw_null.to_numpy(dtype=bool) != structurally_absent.to_numpy())
                ).sum()
            )
            if raw_source_mismatch_rows:
                failures.append(
                    f"{source_column}: raw_source_universe_mismatch_rows="
                    f"{raw_source_mismatch_rows}"
                )
        masks[column] = structurally_absent.copy()
        rule_receipts[column] = {
            "rule_id": f"acs_2024_pums_{source_column.lower()}_age_15_plus",
            "source_column": source_column,
            "mapped_column": column,
            "source_universe": _UNIVERSE_DESCRIPTION,
            "structurally_absent_person_rows": int(structurally_absent.sum()),
            "eligible_acs_person_rows": int(eligible_acs.sum()),
            "in_universe_null_rows": in_universe_null,
            "out_of_universe_nonnull_rows": out_of_universe_nonnull,
            "raw_source_column_present": raw_source_present,
            "raw_source_universe_mismatch_rows": raw_source_mismatch_rows,
            "source_cells_sha256": _source_cells_sha256(
                person,
                column=column,
                scope=scope,
            ),
        }
    if failures:
        raise ValueError(
            f"{boundary}: ACS earnings source-universe equation failed:\n  "
            + "\n  ".join(failures)
        )

    affected_ids = person.loc[structurally_absent, "person_tax_unit_id"]
    affected_unique = pd.Index(affected_ids.drop_duplicates())
    acs_scoped_people = person.loc[acs_scope, ["person_tax_unit_id"]].copy()
    acs_scoped_people["eligible"] = eligible_acs.loc[acs_scope].to_numpy()
    eligible_by_unit = acs_scoped_people.groupby("person_tax_unit_id", sort=False)[
        "eligible"
    ].any()
    empty_ids = pd.Index(eligible_by_unit.index[~eligible_by_unit.to_numpy()])
    mixed_ids = affected_unique.difference(empty_ids, sort=False)
    lineage_column = support_source_id_column("person")
    if lineage_column not in person:
        lineage_column = frame.schema.entity_id_column("person")
    structural_lineages = person.loc[
        structurally_absent, lineage_column
    ].drop_duplicates()
    clone_index = pd.to_numeric(
        person[support_clone_index_column("person")], errors="raise"
    ).astype("int64")
    by_origin_role = {
        f"{origin}/clone_{int(role)}": int(count)
        for (origin, role), count in (
            pd.DataFrame(
                {
                    "origin": channel.loc[structurally_absent],
                    "clone_index": clone_index.loc[structurally_absent],
                }
            )
            .groupby(["origin", "clone_index"], sort=True)
            .size()
            .items()
        )
    }
    receipt: dict[str, object] = {
        "version": _RULE_VERSION,
        "policy": "exact_documented_source_universe",
        "source_dataset": "2024 ACS 1-year PUMS",
        "source_document": "2024 ACS PUMS Data Dictionary (WAGP and SEMP)",
        "source_url": ACS_PUMS_2024_DATA_DICTIONARY_URL,
        "source_channel": _ACS_SUPPORT_CHANNEL,
        "age_column": "age (mapped from AGEP)",
        "minimum_age": ACS_PUMS_EARNINGS_MINIMUM_AGE,
        "aggregation": _AGGREGATION_SEMANTICS,
        "person_source_cells_mutated": False,
        "scoped_person_rows": int(scope.sum()),
        "scoped_acs_person_rows": int(acs_scope.sum()),
        "structurally_absent_person_rows": int(structurally_absent.sum()),
        "affected_tax_unit_rows": int(len(affected_unique)),
        "mixed_universe_tax_unit_rows": int(len(mixed_ids)),
        "empty_universe_tax_unit_rows": int(len(empty_ids)),
        "structurally_absent_person_lineages_sha256": _values_sha256(
            structural_lineages
        ),
        "affected_tax_unit_ids_sha256": _values_sha256(affected_unique),
        "empty_universe_tax_unit_ids_sha256": _values_sha256(empty_ids),
        "by_origin_role": by_origin_role,
        "rules": rule_receipts,
    }
    receipt["sha256"] = _mapping_sha256(receipt)
    return AcsPumsEarningsUniverse(
        structural_absence_masks=MappingProxyType(masks),
        receipt=MappingProxyType(receipt),
    )


def _aligned_scope(
    person: pd.DataFrame,
    scope: Sequence[bool] | pd.Series | np.ndarray | None,
    *,
    boundary: str,
) -> pd.Series:
    if scope is None:
        return pd.Series(True, index=person.index, dtype=bool)
    if isinstance(scope, pd.Series):
        if not scope.index.equals(person.index):
            raise ValueError(f"{boundary}: person scope index is not aligned.")
        result = scope.astype(bool)
    else:
        values = np.asarray(scope)
        if values.ndim != 1 or len(values) != len(person):
            raise ValueError(f"{boundary}: person scope has the wrong shape.")
        result = pd.Series(values, index=person.index, dtype=bool)
    return result


def _source_cells_sha256(
    person: pd.DataFrame,
    *,
    column: str,
    scope: pd.Series,
) -> str:
    lineage = support_source_id_column("person")
    columns = ["person_tax_unit_id", column]
    if lineage in person:
        columns.insert(0, lineage)
    cells = person.loc[scope, columns]
    header = {
        "columns": columns,
        "dtypes": [str(cells[name].dtype) for name in columns],
    }
    digest = hashlib.sha256(_canonical_json(header).encode())
    digest.update(
        pd.util.hash_pandas_object(cells, index=False).to_numpy(dtype="<u8").tobytes()
    )
    return digest.hexdigest()


def _values_sha256(values: Sequence[object] | pd.Index | pd.Series) -> str:
    normalized = sorted(str(value) for value in list(values))
    return hashlib.sha256(_canonical_json(normalized).encode()).hexdigest()


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
