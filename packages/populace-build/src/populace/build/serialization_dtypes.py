"""Canonical pandas dtype policy for serialized :class:`populace.frame.Frame`s."""

from __future__ import annotations

import numpy as np
import pandas as pd

from populace.frame import Frame

__all__ = [
    "CANONICAL_STRING_DTYPE",
    "canonicalize_frame_string_dtypes",
    "canonicalize_table_string_dtypes",
]

CANONICAL_STRING_DTYPE = pd.StringDtype(storage="python", na_value=np.nan)
"""One physical dtype for semantic strings at build serialization boundaries."""


def canonicalize_table_string_dtypes(
    table: pd.DataFrame,
    *,
    boundary: str,
    table_name: str,
    reject_untyped_all_missing_object: bool = False,
) -> pd.DataFrame:
    """Return ``table`` with every unambiguous string column canonicalized.

    Object columns are semantic strings only when every observed scalar is a
    Python or NumPy string. Mixed string/non-string object columns fail closed
    before pandas can stringify values and erase the ambiguity.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"table must be a pandas DataFrame, got {type(table).__name__}."
        )
    if not isinstance(boundary, str) or not boundary:
        raise ValueError("boundary must be a non-empty string.")
    if not isinstance(table_name, str) or not table_name:
        raise ValueError("table_name must be a non-empty string.")
    if not isinstance(reject_untyped_all_missing_object, bool):
        raise TypeError("reject_untyped_all_missing_object must be a bool.")

    result = table
    for column in table.columns:
        series = table[column]
        dtype = series.dtype
        if isinstance(dtype, pd.StringDtype):
            if dtype != CANONICAL_STRING_DTYPE:
                if result is table:
                    result = table.copy(deep=False)
                result[column] = series.astype(CANONICAL_STRING_DTYPE)
            continue
        if not pd.api.types.is_object_dtype(dtype):
            continue

        observed = series.dropna().to_numpy(dtype=object, copy=False)
        if not len(observed):
            if reject_untyped_all_missing_object and len(series):
                raise TypeError(
                    f"String dtype boundary {boundary!r} found ambiguous "
                    f"all-missing object column {table_name}.{column}: no "
                    "observed values establish its semantic type; declare an "
                    "explicit dtype before serialization."
                )
            continue
        string_observed = np.fromiter(
            (isinstance(value, (str, np.str_)) for value in observed),
            dtype=bool,
            count=len(observed),
        )
        if string_observed.any() and not string_observed.all():
            offending_types = sorted(
                {
                    f"{type(value).__module__}.{type(value).__qualname__}"
                    for value, is_string in zip(
                        observed,
                        string_observed,
                        strict=True,
                    )
                    if not is_string
                }
            )
            raise TypeError(
                f"String dtype boundary {boundary!r} found ambiguous object "
                f"column {table_name}.{column}: semantic strings cannot mix "
                "with non-string values; offending value types: "
                f"{offending_types}."
            )

        if not string_observed.any():
            continue
        if result is table:
            result = table.copy(deep=False)
        result[column] = series.astype(CANONICAL_STRING_DTYPE)
    return result


def canonicalize_frame_string_dtypes(
    frame: Frame,
    *,
    boundary: str,
) -> Frame:
    """Canonicalize string columns across every registered entity and link."""

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")

    tables: dict[str, pd.DataFrame] = {}
    changed = False
    for name in (*frame.entities, *frame.links):
        source = frame.table(name) if name in frame.entities else frame.link(name)
        canonical = canonicalize_table_string_dtypes(
            source,
            boundary=boundary,
            table_name=name,
        )
        tables[name] = canonical
        changed |= canonical is not source
    if not changed:
        return frame
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
