"""US support-source provenance and operator-role helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "BASE_ASEC_SUPPORT_CHANNEL",
    "PUF_TAX_DETAIL_CLONE_INDEX",
    "PUF_TAX_DETAIL_SUPPORT_CHANNEL",
    "has_support_role_metadata",
    "puf_tax_detail_clone_mask",
    "spine_source_id_column",
    "support_channel_column",
    "support_clone_index_column",
    "support_role_series",
    "support_source_id_column",
    "without_support_role_metadata",
]

BASE_ASEC_SUPPORT_CHANNEL = "asec"
PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
PUF_TAX_DETAIL_CLONE_INDEX = 1


def has_support_role_metadata(
    table: pd.DataFrame,
    *,
    entity: str,
) -> bool:
    """Return whether clone-role or legacy support-role metadata is present."""

    return (
        support_clone_index_column(entity) in table
        or support_channel_column(entity) in table
    )


def spine_source_id_column(entity: str) -> str:
    """Return the entity-prefixed raw spine-record ID metadata column."""

    _require_entity_name(entity)
    return f"{entity}_spine_source_id"


def support_channel_column(entity: str) -> str:
    """Return the entity-prefixed source-support metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_channel"


def support_clone_index_column(entity: str) -> str:
    """Return the entity-prefixed clone-index metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_clone_index"


def support_source_id_column(entity: str) -> str:
    """Return the entity-prefixed source-record ID metadata column."""

    _require_entity_name(entity)
    return f"{entity}_source_id"


def without_support_role_metadata(
    table: pd.DataFrame,
    *,
    entity: str,
) -> pd.DataFrame:
    """Copy a table without source-channel or clone-role metadata."""

    return table.drop(
        columns=[
            support_channel_column(entity),
            support_clone_index_column(entity),
        ],
        errors="ignore",
    ).copy(deep=True)


def support_role_series(
    table: pd.DataFrame,
    *,
    entity: str,
) -> pd.Series:
    """Return legacy-compatible operator roles derived from clone provenance.

    Native records have the ASEC-compatible role and every donor-detail clone
    has the PUF-compatible role. Clone provenance takes precedence. Legacy
    fixtures without clone indices may use the two historical role labels in
    their support-channel column.
    """

    clone_index_column = support_clone_index_column(entity)
    if clone_index_column not in table:
        channel_column = support_channel_column(entity)
        if channel_column not in table:
            raise ValueError(
                "PUF support metadata is missing both "
                f"{clone_index_column!r} and {channel_column!r}."
            )
        channels = table[channel_column]
        if channels.isna().any():
            raise ValueError(
                f"Legacy support metadata column {channel_column!r} requires "
                "complete support provenance."
            )
        valid = channels.isin(
            (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        )
        if not valid.all():
            invalid = sorted(set(channels.loc[~valid].astype(str)))
            raise ValueError(
                f"Legacy support metadata column {channel_column!r} must contain "
                "the exact ASEC/PUF roles; unsupported value(s): "
                f"{invalid}."
            )
        return pd.Series(
            channels.to_numpy(dtype=object, copy=True),
            index=table.index,
            name=f"{entity}_support_role",
        )
    numeric = pd.to_numeric(table[clone_index_column], errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"PUF support metadata column {clone_index_column!r} must be integral."
        )
    clone_indices = numeric.to_numpy(dtype=np.float64)
    if (clone_indices < 0).any() or not np.equal(
        clone_indices, np.floor(clone_indices)
    ).all():
        raise ValueError(
            f"PUF support metadata column {clone_index_column!r} must contain "
            "nonnegative integers."
        )
    return pd.Series(
        np.where(
            clone_indices == 0,
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ),
        index=table.index,
        name=f"{entity}_support_role",
    )


def puf_tax_detail_clone_mask(
    table: pd.DataFrame,
    *,
    entity: str,
) -> np.ndarray:
    """Select primary PUF-detail clones without reading their source spine."""

    roles = support_role_series(table, entity=entity)
    clone_index_column = support_clone_index_column(entity)
    if clone_index_column not in table:
        return roles.eq(PUF_TAX_DETAIL_SUPPORT_CHANNEL).to_numpy()
    clone_indices = pd.to_numeric(
        table[clone_index_column],
        errors="raise",
    ).to_numpy(dtype=np.int64)
    return clone_indices == PUF_TAX_DETAIL_CLONE_INDEX


def _require_entity_name(entity: str) -> None:
    if not isinstance(entity, str) or not entity:
        raise ValueError("entity must be a non-empty string.")
