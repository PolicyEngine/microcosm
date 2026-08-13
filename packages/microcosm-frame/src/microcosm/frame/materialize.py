"""Materialize a frame's entity tables for a rules-engine dataset.

Rules engines consume flat per-entity tables with weights carried as a
``{entity}_weight`` column; the frame carries them as typed
:class:`~microcosm.frame.weights.Weights`. This module is the one shared
materializer for that boundary: the engine adapters delegate to it, and
country build runtimes that write engine-readable artifacts without an
adapter class call it directly.

The typed weights are authoritative. Any ``{entity}_weight`` column already
on a table is overwritten — never trusted — so a stale or leftover column
can never override calibrated weights on export. Overwriting assigns in
place, which preserves the column's existing position; a weight column
that is *added* (the entity table carried none) is appended last.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame.bundle import Frame

__all__ = [
    "PyTablesBooleanMaterialization",
    "engine_tables",
    "materialize_nullable_booleans_for_pytables",
    "nullable_boolean_values_and_mask",
    "put_frame_table",
]

_WEIGHT_COLUMN_SUFFIX = "_weight"


@dataclass(frozen=True)
class PyTablesBooleanMaterialization:
    """A table made safe for pandas' two PyTables storage formats.

    ``nullable_columns`` records every pandas ``BooleanDtype`` input;
    ``missing_columns`` is the subset represented as object-backed Python
    booleans plus ``pd.NA``. A table with any such column requires fixed HDF
    format because PyTables table format has no nullable-boolean column type.
    """

    table: pd.DataFrame
    nullable_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]

    def hdf_format(self, preferred: str) -> str:
        """Return ``preferred`` unless missing booleans require fixed format."""

        if preferred not in {"fixed", "table"}:
            raise ValueError("preferred HDF format must be 'fixed' or 'table'.")
        return "fixed" if self.missing_columns else preferred


def nullable_boolean_values_and_mask(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical bool values and the exact NA mask for BooleanDtype.

    Missing positions always carry a false value bit. This makes the value
    bytes deterministic and prevents data hidden underneath the mask from
    changing serialized artifacts.
    """

    if not isinstance(series, pd.Series):
        raise TypeError(f"series must be a pandas Series, got {type(series).__name__}.")
    if not isinstance(series.dtype, pd.BooleanDtype):
        raise TypeError(
            "nullable_boolean_values_and_mask requires pandas BooleanDtype, "
            f"got {series.dtype!s}."
        )
    values = series.to_numpy(
        dtype=np.bool_,
        na_value=False,
        copy=False,
    )
    mask = series.isna().to_numpy(dtype=np.bool_, copy=False)
    if values.ndim != 1 or mask.ndim != 1 or len(values) != len(mask):
        raise RuntimeError("Nullable-boolean materialization produced invalid shape.")
    if values[mask].any():  # pragma: no cover - defensive canonicality assertion
        raise RuntimeError("Nullable-boolean null positions must carry false bits.")
    return values, mask


def materialize_nullable_booleans_for_pytables(
    table: pd.DataFrame,
) -> PyTablesBooleanMaterialization:
    """Materialize pandas nullable booleans without changing their semantics.

    Complete columns become native NumPy bool with byte-identical logical
    values. Columns containing NA become object-backed Python ``bool`` plus
    the explicit ``pd.NA`` sentinel; pandas fixed-format HDF preserves that
    representation losslessly. The source table is never mutated, and a
    shallow boundary copy is allocated only when a nullable boolean exists.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be a pandas DataFrame, got {type(table).__name__}."
        )
    result = table
    nullable_columns: list[str] = []
    missing_columns: list[str] = []
    for column in table.columns:
        series = table[column]
        if not isinstance(series.dtype, pd.BooleanDtype):
            continue
        if not isinstance(column, str):
            raise TypeError("Nullable-boolean table columns must have string names.")
        if result is table:
            result = table.copy(deep=False)
        nullable_columns.append(column)
        values, mask = nullable_boolean_values_and_mask(series)
        if mask.any():
            object_values = values.astype(object)
            object_values[mask] = pd.NA
            result[column] = pd.Series(
                object_values,
                index=series.index,
                name=series.name,
                dtype=object,
                copy=False,
            )
            missing_columns.append(column)
        else:
            result[column] = pd.Series(
                values,
                index=series.index,
                name=series.name,
                dtype=np.bool_,
                copy=False,
            )
    return PyTablesBooleanMaterialization(
        table=result,
        nullable_columns=tuple(nullable_columns),
        missing_columns=tuple(missing_columns),
    )


def put_frame_table(
    store: Any,
    key: str,
    table: pd.DataFrame,
    *,
    preferred_format: str,
    data_columns: bool | list[str] | None = None,
) -> PyTablesBooleanMaterialization:
    """Write one Frame table through the shared nullable-boolean boundary."""

    materialized = materialize_nullable_booleans_for_pytables(table)
    hdf_format = materialized.hdf_format(preferred_format)
    options: dict[str, object] = {}
    if hdf_format == "table" and data_columns is not None:
        options["data_columns"] = data_columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        store.put(
            key,
            materialized.table,
            format=hdf_format,
            **options,
        )
    return materialized


def engine_tables(
    frame: Frame,
    *,
    weighted_entities: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Copy the frame's tables and materialize typed weights as columns.

    Args:
        frame: The frame to materialize. Every entity table is copied, so
            mutating the result never touches the frame.
        weighted_entities: Entities whose typed weights to materialize as
            ``{entity}_weight`` columns. Defaults to
            :attr:`Frame.weighted_entities` — every entity carrying its own
            explicit weights. Pass a subset to pin an export contract that
            materializes fewer (an entity outside the frame's weighted set
            raises, exactly as :meth:`Frame.weights_for` does; inherited
            weights are never materialized implicitly).

    Returns:
        Entity name -> copied table, in :attr:`Frame.entities` order, with
        each selected entity's ``{entity}_weight`` column overwritten from
        its typed weights.
    """

    tables = {name: frame.table(name).copy() for name in frame.entities}
    selected = (
        frame.weighted_entities
        if weighted_entities is None
        else tuple(weighted_entities)
    )
    for entity in selected:
        tables[entity][f"{entity}{_WEIGHT_COLUMN_SUFFIX}"] = frame.weights_for(
            entity
        ).values
    return tables
