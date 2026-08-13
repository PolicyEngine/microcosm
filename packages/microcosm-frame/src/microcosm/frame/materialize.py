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

import json
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
    "read_frame_table",
]

_WEIGHT_COLUMN_SUFFIX = "_weight"
_PYTABLES_FRAME_TABLE_CODEC_ATTR = "_microcosm_frame_table_codec"
_PYTABLES_FRAME_TABLE_CODEC_VERSION = 1


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
    if materialized.missing_columns:
        codec = {
            "codec_version": _PYTABLES_FRAME_TABLE_CODEC_VERSION,
            "nullable_boolean_columns": list(materialized.missing_columns),
        }
        setattr(
            store.get_storer(key).attrs,
            _PYTABLES_FRAME_TABLE_CODEC_ATTR,
            json.dumps(codec, sort_keys=True, separators=(",", ":")),
        )
    return materialized


def read_frame_table(store: Any, key: str) -> pd.DataFrame:
    """Read one Frame table and restore its versioned PyTables dtypes.

    The table-local codec attribute is absent from older artifacts and from
    tables that needed no explicit nullable representation. When present it
    identifies object-backed BooleanDtype columns whose exact null positions
    PyTables preserves but which pandas may otherwise infer as strings when
    every value is missing.
    """

    table = store[key]
    raw_codec = getattr(
        store.get_storer(key).attrs,
        _PYTABLES_FRAME_TABLE_CODEC_ATTR,
        None,
    )
    if raw_codec is None:
        return table
    try:
        codec = json.loads(raw_codec)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame table {key!r} has malformed dtype codec metadata."
        ) from exc
    expected_fields = {"codec_version", "nullable_boolean_columns"}
    if not isinstance(codec, dict) or set(codec) != expected_fields:
        raise ValueError(
            f"Frame table {key!r} dtype codec must contain exactly "
            f"{sorted(expected_fields)!r}."
        )
    if type(codec["codec_version"]) is not int or (
        codec["codec_version"] != _PYTABLES_FRAME_TABLE_CODEC_VERSION
    ):
        raise ValueError(
            f"Frame table {key!r} has unsupported dtype codec version "
            f"{codec['codec_version']!r}."
        )
    columns = codec["nullable_boolean_columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError(
            f"Frame table {key!r} has invalid nullable-boolean column metadata."
        )
    missing_columns = sorted(set(columns) - set(table.columns))
    if missing_columns:
        raise ValueError(
            f"Frame table {key!r} dtype codec names absent column(s): "
            f"{missing_columns!r}."
        )

    restored = table.copy(deep=False)
    for column in columns:
        series = table[column]
        invalid = series.notna() & ~series.map(lambda value: isinstance(value, bool))
        if invalid.any():
            raise ValueError(
                f"Frame table {key!r} nullable-boolean column {column!r} "
                "contains a non-boolean value."
            )
        values = series.to_numpy(dtype=object, copy=True)
        null_mask = series.isna().to_numpy(dtype=np.bool_, copy=False)
        values[null_mask] = pd.NA
        restored[column] = pd.Series(
            values,
            index=series.index,
            name=series.name,
            dtype=object,
            copy=False,
        )
    return restored


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
