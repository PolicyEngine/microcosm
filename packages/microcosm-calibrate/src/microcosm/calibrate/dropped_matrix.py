"""Free a calibration result's constraint matrices once only their size is read.

A :class:`~microcosm.calibrate.solve.CalibrationResult` keeps its compiled
:class:`~microcosm.calibrate.matrix.CalibrationProblem`, whose CSR matrix is
the largest object a solve returns. A long build that, after its solves, reads
only each matrix's shape and nonzero count, as
:func:`~microcosm.calibrate.diagnostics.target_surface_payload` does, can call
:func:`without_constraint_matrix` to free the matrices.

The returned result's problem holds a :class:`DroppedConstraintMatrix` in the
``matrix`` field: the same ``shape`` and ``nnz``, so such a reader reports the
matrix it describes, and nothing else. Every use that needs the entries raises
:class:`ConstraintMatrixDroppedError`, so no caller computes with an empty or
missing matrix. The solver modules are untouched: the placeholder only ever
reaches a problem after its solve.
"""

from __future__ import annotations

from dataclasses import replace

from microcosm.calibrate.solve import CalibrationResult, L0RefitResult

__all__ = [
    "ConstraintMatrixDroppedError",
    "DroppedConstraintMatrix",
    "without_constraint_matrix",
]


class ConstraintMatrixDroppedError(RuntimeError):
    """A use of a constraint matrix that was dropped after its solve."""


class DroppedConstraintMatrix:
    """The shape and nonzero count of a constraint matrix that was freed.

    ``shape`` and ``nnz`` keep the dropped matrix's values. Every other use of
    the matrix (a product, a row slice, a conversion to an array, its length,
    iteration, its stored arrays or any other attribute, including a
    ``hasattr`` probe for one) raises :class:`ConstraintMatrixDroppedError`.
    Copying and pickling work.
    """

    __slots__ = ("nnz", "shape")

    def __init__(self, shape: tuple[int, int], nnz: int) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.nnz = int(nnz)

    @classmethod
    def of(cls, matrix) -> DroppedConstraintMatrix:
        """Record ``matrix``'s shape and nonzero count; a record passes through."""
        if isinstance(matrix, cls):
            return matrix
        return cls(matrix.shape, matrix.nnz)

    def _refuse(self, use: str):
        raise ConstraintMatrixDroppedError(
            f"The {self.shape[0]} x {self.shape[1]} constraint matrix was dropped "
            f"after calibration; only its shape and nonzero count ({self.nnz}) "
            f"were kept, so {use} is unavailable."
        )

    def __getattr__(self, name: str):
        # Called only for attributes the class lacks. Special-method lookups
        # (copy, pickle, numpy's array protocols) see a plain missing
        # attribute; so does an unset slot, which only partial construction
        # leaves.
        if name in type(self).__slots__ or (
            name.startswith("__") and name.endswith("__")
        ):
            raise AttributeError(name)
        self._refuse(f"its {name!r}")

    def __array__(self, *args, **kwargs):
        self._refuse("its dense array")

    def __matmul__(self, other):
        self._refuse("a matrix product")

    def __rmatmul__(self, other):
        self._refuse("a matrix product")

    def __mul__(self, other):
        self._refuse("a product")

    def __rmul__(self, other):
        self._refuse("a product")

    def __getitem__(self, key):
        self._refuse("a row or entry")

    def __iter__(self):
        self._refuse("iteration over its rows")

    def __len__(self) -> int:
        self._refuse("its length")

    def __repr__(self) -> str:
        return f"DroppedConstraintMatrix(shape={self.shape!r}, nnz={self.nnz!r})"


def without_constraint_matrix(result):
    """Free a result's constraint matrices, keeping their shapes and nonzero counts.

    For a :class:`~microcosm.calibrate.solve.CalibrationResult`, the returned
    result's problem holds a :class:`DroppedConstraintMatrix` in place of its
    CSR matrix; any later use that needs the entries, such as
    :meth:`~microcosm.calibrate.matrix.CalibrationProblem.estimates`, raises
    :class:`ConstraintMatrixDroppedError`. Every other field is the same object
    as before.

    An :class:`~microcosm.calibrate.solve.L0RefitResult` has two problems, the
    selection's over the full frame and the refit's over the selected support;
    both are dropped. Any other object holds no compiled problem and is
    returned unchanged.
    """
    if isinstance(result, L0RefitResult):
        return replace(
            result,
            selection=without_constraint_matrix(result.selection),
            refit=without_constraint_matrix(result.refit),
        )
    if isinstance(result, CalibrationResult):
        problem = result.problem
        return replace(
            result,
            problem=replace(problem, matrix=DroppedConstraintMatrix.of(problem.matrix)),
        )
    return result
