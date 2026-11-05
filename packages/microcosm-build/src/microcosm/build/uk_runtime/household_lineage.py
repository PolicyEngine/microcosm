"""The ordered clone path of every post-clone UK household, from its lineage columns.

The geography identity kernel keys each pool household on its source id and
its explicit clone path (microcosm#931); the post-geography property node
(microcosm#1240) needs the same path without the geographic clone branch, so
that every pool clone of one spine household shares one key. Both read the
path through this module, which validates the lineage columns once and never
infers a branch from ids, weights or values. The UK's copy lineage is declared
here (:data:`UK_COPY_LINEAGE`) for the shared readers in
``microcosm.build.source_lineage``: the receipts' true sources and the
FRS-trained fits' collapse of geographic clones read it too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from microcosm.build.source_lineage import (
    CopyBranch,
    CopyLineage,
    collapse_replicas,
    copy_paths,
    replica_keys,
)
from microcosm.build.uk_runtime.national_frame import UK_NATIONAL_SCHEMA
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column

__all__ = [
    "GEOGRAPHIC_SUPPORT_BRANCH",
    "HIGH_VALUE_SUPPORT_COPY_COLUMN",
    "HIGH_VALUE_SUPPORT_INDEX_COLUMN",
    "UK_COPY_LINEAGE",
    "UK_HOUSEHOLD_LINEAGE_COLUMNS",
    "collapse_geographic_clones",
    "uk_household_clone_paths",
]

#: Lineage columns the post-clone household identity is keyed on. The support
#: channel, the CGT support split's copy flag and index (microcosm#1045), the
#: high-value home support copy flag and index (microcosm#1240) and the clone
#: flags come from the spine stages; the geographic clone index from
#: ``uk.full.expand``. Calibration K, weights and remapped ids are not inputs.
HIGH_VALUE_SUPPORT_COPY_COLUMN = "household_is_high_value_support_copy"
HIGH_VALUE_SUPPORT_INDEX_COLUMN = "high_value_support_copy_index"
UK_HOUSEHOLD_LINEAGE_COLUMNS: tuple[str, ...] = (
    "source_household_id",
    "household_support_channel",
    "household_support_clone_index",
    "household_is_cgt_support_copy",
    "cgt_support_copy_index",
    "household_is_capital_gains_clone",
    "household_is_cgt_residential_clone",
    "cgt_residential_clone_index",
    ladder_clone_index_column("household"),
)
#: The two optional columns the high-value support stage adds (microcosm#1240);
#: a spine built before that stage carries neither and keys as before.
UK_OPTIONAL_LINEAGE_COLUMNS: tuple[str, ...] = (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)
GEOGRAPHIC_SUPPORT_BRANCH = "geographic_support"
_BASE_SUPPORT_CHANNEL = "frs"
_SPI_SUPPORT_CHANNEL = "spi"
#: The UK's copy lineage for the shared readers in
#: ``microcosm.build.source_lineage``: every copy's true source is
#: ``source_household_id``, and its copying steps are the clone path's
#: branches below, in path order.
UK_COPY_LINEAGE = CopyLineage(
    entity="household",
    branches=(
        CopyBranch(
            "spi_support_channel",
            index="household_support_clone_index",
            marker="household_support_channel",
            copy=_SPI_SUPPORT_CHANNEL,
            original=_BASE_SUPPORT_CHANNEL,
        ),
        CopyBranch(
            "cgt_support_split",
            index="cgt_support_copy_index",
            marker="household_is_cgt_support_copy",
        ),
        CopyBranch(
            "high_value_support",
            index=HIGH_VALUE_SUPPORT_INDEX_COLUMN,
            marker=HIGH_VALUE_SUPPORT_COPY_COLUMN,
        ),
        CopyBranch("cgt_incidence_clone", marker="household_is_capital_gains_clone"),
        CopyBranch(
            "cgt_residential_split",
            index="cgt_residential_clone_index",
            marker="household_is_cgt_residential_clone",
        ),
        CopyBranch(
            GEOGRAPHIC_SUPPORT_BRANCH, index=ladder_clone_index_column("household")
        ),
    ),
)


#: The branches every post-clone household carries: the SPI support channel
#: and the CGT steps. High-value support (a spine built before that stage has
#: neither of its columns) and the geographic clone index (a spine frame has
#: none yet) are read where present.
_REQUIRED_BRANCHES = (
    "spi_support_channel",
    "cgt_support_split",
    "cgt_incidence_clone",
    "cgt_residential_split",
)
_INT64_LINEAGE_COLUMNS = (
    "source_household_id",
    "cgt_support_copy_index",
    "cgt_residential_clone_index",
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)


def uk_household_clone_paths(
    household: pd.DataFrame,
) -> tuple[list[int], list[tuple[tuple[str, int], ...]]]:
    """Return every household's source id and ordered clone path.

    The path lists, in branch order: ``spi_support_channel`` with the support
    clone index when the household is an SPI support copy, ``cgt_support_split``
    with the copy index when it is a CGT support copy, ``high_value_support``
    with the copy index when it is a high-value home support copy,
    ``cgt_incidence_clone`` (ordinal 1) from its flag, ``cgt_residential_split``
    with the arm index, and ``geographic_support`` with the pool clone index
    when it is not the original copy. The shared reader
    (``microcosm.build.source_lineage.copy_paths``) reads it on
    :data:`UK_COPY_LINEAGE`; malformed lineage fails by name.
    """
    clone_column = ladder_clone_index_column("household")
    required = [c for c in UK_HOUSEHOLD_LINEAGE_COLUMNS if c != clone_column]
    missing = [c for c in required if c not in household]
    if missing:
        raise ValueError(f"Geography identity lacks lineage column(s) {missing}.")
    present = [
        column
        for column in (UK_COPY_LINEAGE.source, *UK_COPY_LINEAGE.copy_columns)
        if column in household
    ]
    if household.loc[:, present].isna().any().any():
        raise ValueError("Geography identity refuses null lineage values.")
    for column in _INT64_LINEAGE_COLUMNS:
        if column in household and household[column].dtype != np.dtype("int64"):
            raise ValueError(f"Geography identity requires an int64 {column}.")
    sources, paths = copy_paths(household, UK_COPY_LINEAGE, require=_REQUIRED_BRANCHES)
    return [int(source) for source in sources], paths


def collapse_geographic_clones(
    tables: Mapping[str, pd.DataFrame],
    entity: str,
    rows: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """A fit's training rows with each record's geographic clones folded in.

    A survey record a fit trains on keeps one row, its first among ``rows``,
    with the weight of all its geographic clones; every other copy (SPI
    support, high-value support and the CGT steps) stays a row of its own
    (microcosm#931). The record's key is its true source and its clone path
    without the geographic branch, through the shared reader
    (``microcosm.build.source_lineage``); a frame without geographic clones,
    as every spine is today, keeps ``rows`` and ``weights`` value for value.
    Returns the kept rows and the weights over every row of ``entity``.
    """

    keys = replica_keys(
        tables,
        UK_NATIONAL_SCHEMA,
        entity,
        UK_COPY_LINEAGE,
        merge=(GEOGRAPHIC_SUPPORT_BRANCH,),
    )
    return collapse_replicas(keys, rows, weights)


def without_geographic_branch(
    paths: Sequence[tuple[tuple[str, int], ...]],
) -> list[tuple[tuple[str, int], ...]]:
    """The same paths with the pool clone branch removed (one per spine household)."""
    return [
        tuple(branch for branch in path if branch[0] != GEOGRAPHIC_SUPPORT_BRANCH)
        for path in paths
    ]
