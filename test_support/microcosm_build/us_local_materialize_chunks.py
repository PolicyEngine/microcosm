"""Compare the ACS local release's target matrices across household chunk sizes.

``materialize_chunked`` (``tools/build_us_acs_local_release.py``) returns a
CSR float32 matrix of targets by households, built one household chunk at a
time. Two runs give the same matrix when these are bit-for-bit equal, compared
through :func:`canonical_result_bytes`:

* the compiled target specs' names, in order;
* the matrix's type, shape and the dtype and raw bytes of its ``data``,
  ``indices`` and ``indptr`` arrays.

Those arrays and the shape are what ``save_target_matrix`` writes for these
rows. ``do_materialize`` writes ``target_matrix.npz`` from them after adding
the county and population rows, which this comparison does not cover. On a
difference, :func:`assert_same_surface` names the targets and household
positions that differ and by how much.
"""

from __future__ import annotations

import numpy as np

from test_support.microcosm_build.worker_pool_identity import canonical_result_bytes

__all__ = ["assert_same_surface", "surface_bytes"]

#: How many differing targets, and households per target, a report lists.
_REPORTED_TARGETS = 10
_REPORTED_HOUSEHOLDS = 5


def _target_names(materialized) -> list[str]:
    """The compiled target specs' names, in matrix row order.

    ``MaterializedSurface.names`` gives their measures, which can differ.
    """

    return [spec.name for spec in materialized.compiled_specs]


def surface_bytes(materialized) -> bytes:
    """The canonical bytes of a ``MaterializedSurface``'s target names and matrix."""

    matrix = materialized.matrix
    return canonical_result_bytes(
        {
            "names": _target_names(materialized),
            "matrix_type": type(matrix),
            "shape": tuple(int(size) for size in matrix.shape),
            "data": matrix.data,
            "indices": matrix.indices,
            "indptr": matrix.indptr,
        }
    )


def _value_differences(reference, materialized) -> list[str]:
    """Report lines for the targets whose values differ; empty if none."""

    expected = reference.matrix.toarray()
    actual = materialized.matrix.toarray()
    # Bitwise, so -0.0 against 0.0 and NaN payloads count as differences.
    bits = np.dtype(f"u{expected.dtype.itemsize}")
    differs = expected.view(bits) != actual.view(bits)
    rows = np.flatnonzero(differs.any(axis=1))
    if not len(rows):
        return []
    lines = [f"{len(rows)} target(s) hold different values:"]
    for row in rows[:_REPORTED_TARGETS]:
        households = np.flatnonzero(differs[row])
        shown = ", ".join(
            f"household {int(position)}: {float(expected[row, position])!r} -> "
            f"{float(actual[row, position])!r}"
            for position in households[:_REPORTED_HOUSEHOLDS]
        )
        largest = float(
            np.max(
                np.abs(
                    expected[row, households].astype(np.float64)
                    - actual[row, households].astype(np.float64)
                )
            )
        )
        lines.append(
            f"  {_target_names(reference)[row]}: {len(households)} household(s), "
            f"largest absolute difference {largest!r} ({shown})"
        )
    return lines


def assert_same_surface(reference, materialized, *, label: str) -> None:
    """Refuse ``materialized`` unless it is bit-for-bit ``reference``."""

    if surface_bytes(materialized) == surface_bytes(reference):
        return
    lines = [f"{label}: the target matrix differs from the reference run."]
    if _target_names(materialized) != _target_names(reference):
        lines.append(
            f"Compiled target names differ: {_target_names(reference)} -> "
            f"{_target_names(materialized)}."
        )
    else:
        expected, actual = reference.matrix, materialized.matrix
        layout = {
            "type": (type(expected), type(actual)),
            "shape": (expected.shape, actual.shape),
            "data dtype": (expected.data.dtype, actual.data.dtype),
            "indices dtype": (expected.indices.dtype, actual.indices.dtype),
            "indptr dtype": (expected.indptr.dtype, actual.indptr.dtype),
        }
        changed = {key: pair for key, pair in layout.items() if pair[0] != pair[1]}
        if changed:
            lines.append(f"Matrix layout differs: {changed}.")
        else:
            lines += _value_differences(reference, materialized) or [
                "Every value is equal, but the stored entries differ "
                "(an explicit zero, or indices in another order)."
            ]
    raise AssertionError("\n".join(lines))
