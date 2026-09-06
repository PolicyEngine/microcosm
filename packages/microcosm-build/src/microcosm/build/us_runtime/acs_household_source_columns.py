"""Explicitly versioned ACS household source-column contracts.

The executed 2024 ACS spine reads one fixed household roster, declared here as
version 1 exactly as :mod:`microcosm.build.us_runtime.acs_pums` executes it.
Version 2 adds the two publisher columns the household reported-income
measurement needs: ``HINCP`` and the *housing-record* ``ADJINC``.

Version 2 is a separate additive read. It never widens the version 1 roster,
never rewrites the version 1 frame, and does not participate in the executed
loader's implementation identity, so an existing ACS source cache stays valid.
The two reads are aligned by native ``SERIALNO``, never by row position: the
version 1 loader may apply its smoke subset, and the caller owns which of its
households it asks about.

This module reads archives. It applies no adjustment factor, decides no
universe policy and produces no measurement; the pure measurement module
:mod:`microcosm.build.us_runtime.household_reported_income` owns all of that.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_2024_1YR_VINTAGE,
    DEFAULT_CHUNKSIZE,
    AcsPumsSource,
    _occupied_households,
    _read_archive,
)

__all__ = [
    "ACS_HOUSEHOLD_SOURCE_COLUMNS_V1",
    "ACS_HOUSEHOLD_SOURCE_COLUMNS_V2",
    "AcsHouseholdSourceColumns",
    "acs_household_source_column_contract",
    "read_acs_household_income_columns",
]

# The version 1 roster is a declaration of already-executed behavior. The
# contract test asserts it against the loader's own private tuples, so an
# unnoticed drift fails rather than silently redefining what version 1 means.
ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_REQUIRED = (
    "SERIALNO",
    "PUMA",
    "WGTP",
    "NP",
    "ADJHSG",
    "TEN",
    "RNTP",
    "GRNTP",
    "TAXAMT",
    "TYPEHUGQ",
)
ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_OPTIONAL = ("ST", "STATE")
ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_FRAME = (
    "NP",
    "ADJHSG",
    "TEN",
    "RNTP",
    "GRNTP",
    "TAXAMT",
    "TYPEHUGQ",
)
_V2_ADDED = ("HINCP", "ADJINC")
# Columns the additive version 2 read needs: the join key, the two added
# publisher columns, and the three columns the shared occupancy rule checks.
_V2_READ_COLUMNS = ("SERIALNO", "WGTP", "NP", "TYPEHUGQ", *_V2_ADDED)
# HINCP is a 7-wide numeric field and ADJINC a 7-wide character field in the
# pinned dictionary; both stay well inside exact float64 integers, so the
# loader's float parse is losslessly recovered as pandas nullable Int64.
_INT_COLUMNS = ("NP", "TYPEHUGQ", *_V2_ADDED)
_MEMBER_PREFIX = "psam_hus"


@dataclass(frozen=True)
class AcsHouseholdSourceColumns:
    """One explicitly versioned household source-column roster."""

    version: int
    required: tuple[str, ...]
    optional: tuple[str, ...]
    frame_columns: tuple[str, ...]
    added_since_previous: tuple[str, ...]
    note: str

    def identity(self) -> str:
        """Return this roster's own digest, independent of any loader build."""

        return hashlib.sha256(
            json.dumps(
                {
                    "spine": ACS_2024_1YR_SPINE,
                    "vintage": ACS_2024_1YR_VINTAGE,
                    "version": self.version,
                    "required": list(self.required),
                    "optional": list(self.optional),
                    "frame_columns": list(self.frame_columns),
                    "added_since_previous": list(self.added_since_previous),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


ACS_HOUSEHOLD_SOURCE_COLUMNS_V1 = AcsHouseholdSourceColumns(
    version=1,
    required=ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_REQUIRED,
    optional=ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_OPTIONAL,
    frame_columns=ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_FRAME,
    added_since_previous=(),
    note=(
        "The executed 2024 ACS spine roster. Declared here for comparison only; "
        "this module never reads or rewrites it."
    ),
)
ACS_HOUSEHOLD_SOURCE_COLUMNS_V2 = AcsHouseholdSourceColumns(
    version=2,
    required=tuple(
        dict.fromkeys((*ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_REQUIRED, *_V2_ADDED))
    ),
    optional=ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_OPTIONAL,
    frame_columns=tuple(
        dict.fromkeys((*ACS_HOUSEHOLD_SOURCE_COLUMNS_V1_FRAME, *_V2_ADDED))
    ),
    added_since_previous=_V2_ADDED,
    note=(
        "Adds the publisher household total HINCP and the housing-record ADJINC "
        "it requires. Read additively and joined on native SERIALNO."
    ),
)
_CONTRACTS = {
    contract.version: contract
    for contract in (ACS_HOUSEHOLD_SOURCE_COLUMNS_V1, ACS_HOUSEHOLD_SOURCE_COLUMNS_V2)
}


def acs_household_source_column_contract(version: int) -> AcsHouseholdSourceColumns:
    """Return the declared household source-column contract for ``version``."""

    if type(version) is not int or version not in _CONTRACTS:
        raise ValueError(
            f"Unknown ACS household source-column contract version {version!r}."
        )
    return _CONTRACTS[version]


def read_acs_household_income_columns(
    source: AcsPumsSource,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the version 2 household income columns from a pinned ACS archive.

    Returns occupied household rows keyed by native ``SERIALNO`` with literal
    ``HINCP``/``ADJINC`` values retained as pandas nullable ``Int64``. Census
    blanks stay missing: this read never turns an out-of-universe blank into a
    zero, and never applies the adjustment factor.
    """

    # Defense in depth. AcsPumsSource already refuses any other vintage in its
    # own constructor, so this branch is unreachable through the public API; it
    # exists so a future alternate construction path cannot silently reuse the
    # 2024-pinned column contract for a different file.
    if source.vintage != ACS_2024_1YR_VINTAGE:
        raise ValueError(
            "The version 2 household source-column contract is pinned to the "
            f"2024 1-year PUMS; got vintage {source.vintage}."
        )
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    household, members = _read_archive(
        source.household_zip,
        member_prefix=_MEMBER_PREFIX,
        required=_V2_READ_COLUMNS,
        optional=(),
        chunksize=chunksize,
    )
    duplicates = household["SERIALNO"].duplicated(keep=False)
    if duplicates.any():
        examples = household.loc[duplicates, "SERIALNO"].head().tolist()
        raise ValueError(f"ACS duplicate household SERIALNO value(s): {examples}.")
    household = household.sort_values("SERIALNO", kind="stable").reset_index(drop=True)
    household, vacant_rows = _occupied_households(household)
    if household.empty:
        raise ValueError(
            "ACS household income read selected no occupied household records."
        )
    table = pd.DataFrame({"SERIALNO": household["SERIALNO"].astype("string")})
    for column in _INT_COLUMNS:
        table[column] = _nullable_integer(household[column], column)
    metadata = {
        "spine": ACS_2024_1YR_SPINE,
        "vintage": source.vintage,
        "household_source_column_version": ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.version,
        "household_source_columns": list(ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.required),
        "household_source_column_identity": (
            ACS_HOUSEHOLD_SOURCE_COLUMNS_V2.identity()
        ),
        "read_columns": list(_V2_READ_COLUMNS),
        "household_csv_members": members,
        "household_rows": int(len(table)),
        "vacant_household_rows_dropped": vacant_rows,
        "join_key": "SERIALNO",
        "adjustment_applied": False,
        # This read is deliberately whole-file: the version 1 loader's smoke
        # subset is a hash-ranked selection this read does not replicate, and
        # the caller joins on SERIALNO, so a superset is harmless. Recording it
        # keeps a smoke caller from reading the row count as its own selection.
        "max_households_requested": source.max_households,
        "max_households_applied": False,
    }
    return table, metadata


def _nullable_integer(values: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    observed = numeric.notna()
    if observed.any():
        amounts = numeric.loc[observed]
        if not (amounts % 1 == 0).all():
            raise ValueError(f"ACS household column {column!r} must be integral.")
    return numeric.astype("Int64")
