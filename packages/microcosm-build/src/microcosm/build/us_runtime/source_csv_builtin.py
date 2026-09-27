"""Bind source CSV readers to the real stdlib builtin, including caller aliases."""

import _csv
import csv
from types import BuiltinFunctionType

_NATIVE_CSV = _csv
_CSV = csv
_READER = _csv.reader


def csv_reader_bound(module) -> bool:
    """Refuse live rebinding, including wrappers installed before this import."""
    return (
        type(_READER) is BuiltinFunctionType
        and _READER.__module__ == "_csv"
        and _READER.__name__ == "reader"
        and _READER.__self__ is _NATIVE_CSV
        and module is _CSV
        and getattr(module, "reader", None) is _READER
        and getattr(_NATIVE_CSV, "reader", None) is _READER
    )


def capture_csv_reader(module):
    """Return the checked builtin itself, never a later mutable alias lookup."""
    return _READER if csv_reader_bound(module) else None


if not csv_reader_bound(csv):
    raise ValueError("SOURCE_CSV_READER_CHANGED")
