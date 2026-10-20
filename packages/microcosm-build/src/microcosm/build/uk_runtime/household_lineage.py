"""The ordered clone path of every post-clone UK household, from its lineage columns.

The geography identity kernel keys each pool household on its source id and
its explicit clone path (microcosm#931); the post-geography property node
(microcosm#1240) needs the same path without the geographic clone branch, so
that every pool clone of one spine household shares one key. Both read the
path through this module, which validates the lineage columns once and never
infers a branch from ids, weights or values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column

__all__ = [
    "GEOGRAPHIC_SUPPORT_BRANCH",
    "HIGH_VALUE_SUPPORT_COPY_COLUMN",
    "HIGH_VALUE_SUPPORT_INDEX_COLUMN",
    "UK_HOUSEHOLD_LINEAGE_COLUMNS",
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


def _require_bool(series: pd.Series, column: str) -> None:
    dtype = series.dtype
    if dtype != np.dtype("bool") and not isinstance(dtype, pd.BooleanDtype):
        raise ValueError(f"Geography identity requires a boolean {column}.")


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
    when it is not the original copy. Malformed lineage fails by name.
    """
    clone_column = ladder_clone_index_column("household")
    required = [c for c in UK_HOUSEHOLD_LINEAGE_COLUMNS if c != clone_column]
    missing = [c for c in required if c not in household]
    if missing:
        raise ValueError(f"Geography identity lacks lineage column(s) {missing}.")
    # A spine frame carries no pool clone index yet: every row is clone 0.
    columns: list[str] = list(required) + (
        [clone_column] if clone_column in household else []
    )
    has_high_value = all(c in household for c in UK_OPTIONAL_LINEAGE_COLUMNS)
    if any(c in household for c in UK_OPTIONAL_LINEAGE_COLUMNS) and not has_high_value:
        raise ValueError(
            "Geography identity needs both high-value support lineage columns or neither."
        )
    if has_high_value:
        columns.extend(UK_OPTIONAL_LINEAGE_COLUMNS)
    lineage = household.loc[:, columns]
    if lineage.isna().any().any():
        raise ValueError("Geography identity refuses null lineage values.")
    if lineage["source_household_id"].dtype != np.dtype("int64"):
        raise ValueError("Geography identity requires int64 source household ids.")
    if lineage["cgt_support_copy_index"].dtype != np.dtype("int64"):
        raise ValueError("Geography identity requires an int64 support copy index.")
    if lineage["cgt_residential_clone_index"].dtype != np.dtype("int64"):
        raise ValueError("Geography identity requires an int64 residential arm index.")
    if has_high_value and lineage[HIGH_VALUE_SUPPORT_INDEX_COLUMN].dtype != np.dtype(
        "int64"
    ):
        raise ValueError(
            "Geography identity requires an int64 high-value support copy index."
        )
    for column in (
        "household_is_cgt_support_copy",
        "household_is_capital_gains_clone",
        "household_is_cgt_residential_clone",
        *((HIGH_VALUE_SUPPORT_COPY_COLUMN,) if has_high_value else ()),
    ):
        _require_bool(lineage[column], column)
    high_value_copy = (
        lineage[HIGH_VALUE_SUPPORT_COPY_COLUMN].to_numpy()
        if has_high_value
        else np.zeros(len(lineage), dtype=bool)
    )
    high_value_index = (
        lineage[HIGH_VALUE_SUPPORT_INDEX_COLUMN].to_numpy()
        if has_high_value
        else np.zeros(len(lineage), dtype=np.int64)
    )
    source_ids: list[int] = []
    paths: list[tuple[tuple[str, int], ...]] = []
    for (
        source_id,
        channel,
        support_index,
        support_copy,
        copy_index,
        cgt,
        residential_arm,
        residential_index,
        clone_index,
        hv_copy,
        hv_index,
    ) in zip(
        lineage["source_household_id"].to_numpy(),
        lineage["household_support_channel"].to_numpy(),
        lineage["household_support_clone_index"].to_numpy(),
        lineage["household_is_cgt_support_copy"].to_numpy(),
        lineage["cgt_support_copy_index"].to_numpy(),
        lineage["household_is_capital_gains_clone"].to_numpy(),
        lineage["household_is_cgt_residential_clone"].to_numpy(),
        lineage["cgt_residential_clone_index"].to_numpy(),
        (
            lineage[clone_column].to_numpy()
            if clone_column in lineage
            else np.zeros(len(lineage), dtype=np.int64)
        ),
        high_value_copy,
        high_value_index,
        strict=True,
    ):
        channel = str(channel)
        support_index, clone_index = int(support_index), int(clone_index)
        copy_index = int(copy_index)
        residential_index = int(residential_index)
        hv_index = int(hv_index)
        if residential_index < 0 or hv_index < 0:
            raise ValueError("Geography identity refuses negative clone indices.")
        if bool(residential_arm) != (residential_index != 0):
            raise ValueError("Household residential-arm flag and arm index disagree.")
        if channel not in {_BASE_SUPPORT_CHANNEL, _SPI_SUPPORT_CHANNEL}:
            raise ValueError(f"Unknown household support channel {channel!r}.")
        if support_index < 0 or clone_index < 0 or copy_index < 0:
            raise ValueError("Geography identity refuses negative clone indices.")
        if (channel == _SPI_SUPPORT_CHANNEL) != (support_index != 0):
            raise ValueError(
                "Household support channel and support clone index disagree."
            )
        if bool(support_copy) != (copy_index != 0):
            raise ValueError(
                "Household support-copy flag and support copy index disagree."
            )
        if bool(hv_copy) != (hv_index != 0):
            raise ValueError(
                "Household high-value support-copy flag and copy index disagree."
            )
        path: list[tuple[str, int]] = []
        if support_index:
            path.append(("spi_support_channel", support_index))
        if copy_index:
            path.append(("cgt_support_split", copy_index))
        if hv_index:
            path.append(("high_value_support", hv_index))
        if bool(cgt):
            path.append(("cgt_incidence_clone", 1))
        if residential_index:
            path.append(("cgt_residential_split", residential_index))
        if clone_index:
            path.append((GEOGRAPHIC_SUPPORT_BRANCH, clone_index))
        source_ids.append(int(source_id))
        paths.append(tuple(path))
    return source_ids, paths


def without_geographic_branch(
    paths: Sequence[tuple[tuple[str, int], ...]],
) -> list[tuple[tuple[str, int], ...]]:
    """The same paths with the pool clone branch removed (one per spine household)."""
    return [
        tuple(branch for branch in path if branch[0] != GEOGRAPHIC_SUPPORT_BRANCH)
        for path in paths
    ]
