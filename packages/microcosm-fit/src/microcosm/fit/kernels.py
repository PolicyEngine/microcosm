"""Graph kernels backed by :mod:`microcosm.fit`'s public QRF API.

The model artifact emitted by :class:`QRFKernel` is a pickle of the fitted
object *before* its first draw.  Pickle loading executes code: consumers must
only load bytes obtained from a trusted, content-verified graph store and must
never unpickle attacker-controlled data.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

import microcosm.fit.model as fit_model_module
import microcosm.fit.qrf as qrf_module
from microcosm.fit import fit as fit_qrf
from microcosm.fit.qrf import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    SeedSource,
    Tolerance,
    source_hash,
)

__all__ = [
    "FIT_QRF_DEPENDENCIES",
    "QRF_EXECUTOR_SEED_HIGH",
    "QRF_EXECUTOR_KERNEL",
    "QRF_KERNEL",
    "QRF_PARAM_KERNEL",
    "QRFKernel",
]


FIT_QRF_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "quantile-forest",
)
"""Distributions whose versions form part of ``fit.qrf@1``'s identity."""

#: How far ``fit.qrf@1`` numbers may move between machines. The forest stack
#: promises no cross-platform bit stability (charter H1 records the claim as
#: ``tolerance_bound``); this bound is provisional until measured on the H1
#: fixture across arm64 and x86_64 (amendment 13 follow-up), and parity in
#: the locked environment is still asserted byte for byte.
FIT_QRF_TOLERANCE = Tolerance(rtol=1e-6)


QRF_EXECUTOR_SEED_HIGH = 2**31 - 1
"""Exclusive upper bound for the one seed drawn from ``KernelContext.rng``."""


_DONOR_MASK = "is_donor"
_PARAMS = frozenset(
    {
        "donor_target",
        "max_samples_leaf",
        "min_samples_leaf",
        "n_estimators",
        "seed",
        "zero_atol",
    }
)


@dataclass(frozen=True)
class _QRFRunSpec:
    """Validated declaration and parameter values for one QRF run."""

    entity: str
    target: str
    donor_target: str
    predictors: tuple[str, ...]
    recipient_mask: str
    n_estimators: int
    zero_atol: float
    max_samples_leaf: int | float | None
    seed: int


class QRFKernel(KernelBase):
    """Fit one donor target and draw it for the declared recipient rows.

    A legal graph declaration cannot both consume and own the same column, even
    under disjoint row masks.  ``donor_target`` therefore optionally names the
    observed donor input column; the kernel renames it to the sole declared
    output target in a private donor DataFrame before calling
    :func:`microcosm.fit.fit`.  Omitting it retains a same-name fallback for
    direct protocol contexts.

    Seed provenance is an instance-level contract because
    :class:`~microcosm.graph.Capabilities` is context-free.  A ``PARAM`` instance
    requires a literal ``seed`` parameter and never consumes ``context.rng``; an
    ``EXECUTOR`` instance forbids that parameter and consumes exactly one integer
    from ``context.rng``.

    The ``model`` artifact is a pickle of the fitted object before prediction.
    Loading it is safe only when its bytes came from a trusted, content-verified
    store; never unpickle untrusted input.
    """

    ref = "fit.qrf@1"

    def __init__(self, seed_source: SeedSource = SeedSource.EXECUTOR) -> None:
        if seed_source not in (SeedSource.PARAM, SeedSource.EXECUTOR):
            raise ValueError(
                "QRFKernel seed_source must be SeedSource.PARAM or SeedSource.EXECUTOR."
            )
        self.capabilities = Capabilities(
            determinism=Determinism.SEEDED,
            numeric=Numeric.TOLERANCE_BOUND,
            seed_source=seed_source,
            dependencies=FIT_QRF_DEPENDENCIES,
            tolerance=FIT_QRF_TOLERANCE,
        )

    def implementation_hash(self) -> str:
        """Hash the adapter and every local module implementing the wrapped fit."""
        return source_hash(
            type(self),
            fit_qrf,
            fit_model_module,
            qrf_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        """Call the public QRF fit/predict API on donor/recipient row subsets."""
        spec = self._run_spec(context)
        table = self._table(context, spec)
        donor_mask = self._boolean_mask(table, _DONOR_MASK)
        recipient_mask = self._boolean_mask(table, spec.recipient_mask)
        if not np.array_equal(recipient_mask, ~donor_mask):
            raise ValueError(
                f"Owned-row mask {spec.recipient_mask!r} must be the exact "
                f"complement of {_DONOR_MASK!r}."
            )
        donor_rows = int(donor_mask.sum())
        recipient_rows = int(recipient_mask.sum())
        if donor_rows == 0:
            raise ValueError("fit.qrf@1 requires at least one donor row.")
        if recipient_rows == 0:
            raise ValueError("fit.qrf@1 requires at least one recipient row.")

        try:
            weights = context.weights[spec.entity]
        except KeyError as error:
            raise ValueError(
                f"fit.qrf@1 received no weights for entity {spec.entity!r}."
            ) from error
        if len(weights) != len(table):
            raise ValueError(
                f"Weights for entity {spec.entity!r} have length {len(weights)}, "
                f"but its kernel table has {len(table)} rows."
            )
        donor_weights = weights.values[donor_mask]

        donor_columns = [*spec.predictors, spec.donor_target]
        donor = table.loc[donor_mask, donor_columns].copy()
        if spec.donor_target != spec.target:
            donor = donor.rename(columns={spec.donor_target: spec.target})
        recipient = table.loc[recipient_mask, list(spec.predictors)].copy()

        fitted = fit_qrf(
            donor,
            list(spec.predictors),
            [spec.target],
            weights=donor_weights,
            n_estimators=spec.n_estimators,
            zero_atol=spec.zero_atol,
            max_samples_leaf=spec.max_samples_leaf,
            seed=spec.seed,
        )
        # Serialize before predict so the artifact can reproduce this first draw.
        model_bytes = pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL)
        drawn = fitted.predict(recipient)[spec.target]

        id_column = f"{spec.entity}_id"
        ids = table.loc[recipient_mask, id_column]
        index = pd.Index(ids.to_numpy(copy=True), name=id_column)
        output = pd.Series(
            drawn.to_numpy(dtype=np.float64, copy=True),
            index=index,
            name=spec.target,
            dtype=np.float64,
        )

        target_values = donor[spec.target].to_numpy(dtype=np.float64, copy=False)
        receipt = {
            "entity": spec.entity,
            "target": spec.target,
            "donor_target": spec.donor_target,
            "predictors": spec.predictors,
            "donor_rows": donor_rows,
            "recipient_rows": recipient_rows,
            "seed": spec.seed,
            "seed_source": self.capabilities.seed_source.value,
            "weight_kind": weights.kind.value,
            "fit_weight_kind": fitted.weight_kind,
            "donor_weight_sum": float(donor_weights.sum()),
            "regime": fitted.regimes()[spec.target],
            "n_estimators": spec.n_estimators,
            "min_samples_leaf": 1,
            "zero_atol": spec.zero_atol,
            "max_samples_leaf": spec.max_samples_leaf,
            "donor_target_min": float(target_values.min()),
            "donor_target_max": float(target_values.max()),
            "donor_target_mean": float(target_values.mean()),
            "donor_target_weighted_mean": float(
                np.average(target_values, weights=donor_weights)
            ),
        }
        return KernelResult(
            columns={(spec.entity, spec.target): output},
            artifacts={"model": model_bytes},
            receipt=receipt,
        )

    def _run_spec(self, context: KernelContext) -> _QRFRunSpec:
        """Validate the declaration and resolve parameters without touching data."""
        node = context.node
        if node.kernel != self.ref:
            raise ValueError(
                f"QRFKernel handles {self.ref!r}, not node kernel {node.kernel!r}."
            )
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise ValueError(
                "fit.qrf@1 requires exactly one Slice and one Owned output."
            )
        input_slice = node.inputs[0]
        owned = node.outputs[0]
        if input_slice.entity != owned.entity:
            raise ValueError("fit.qrf@1 input and output must use the same entity.")
        if input_slice.rows != ROWS_ALL:
            raise ValueError(
                "fit.qrf@1's input Slice must use rows='all' so it receives both "
                "donors and recipients."
            )
        if owned.rows == ROWS_ALL:
            raise ValueError(
                "fit.qrf@1 output must declare a recipient row-mask column."
            )
        if owned.dtype != "float64":
            raise ValueError("fit.qrf@1's sole output must declare dtype='float64'.")

        unknown = sorted(set(context.params) - _PARAMS)
        if unknown:
            raise ValueError(f"fit.qrf@1 received unknown parameter(s): {unknown}.")
        donor_target = context.params.get("donor_target", owned.column)
        if not isinstance(donor_target, str) or not donor_target:
            raise TypeError("fit.qrf@1 donor_target must be a non-empty string.")
        excluded = {donor_target, _DONOR_MASK, owned.rows}
        predictors = tuple(c for c in input_slice.columns if c not in excluded)
        if not predictors:
            raise ValueError("fit.qrf@1 requires at least one declared predictor.")
        required = {donor_target, _DONOR_MASK, owned.rows}
        missing = sorted(required - set(input_slice.columns))
        if missing:
            raise ValueError(
                f"fit.qrf@1's Slice is missing required column(s) {missing}."
            )

        min_samples_leaf = context.params.get("min_samples_leaf", 1)
        if (
            not isinstance(min_samples_leaf, int)
            or isinstance(min_samples_leaf, bool)
            or min_samples_leaf != 1
        ):
            raise ValueError(
                "microcosm.fit's public QRF API does not expose "
                "min_samples_leaf; fit.qrf@1 supports only its existing default "
                "value of 1."
            )

        n_estimators = context.params.get("n_estimators", DEFAULT_N_ESTIMATORS)
        if not isinstance(n_estimators, int) or isinstance(n_estimators, bool):
            raise TypeError("fit.qrf@1 n_estimators must be an integer.")
        zero_atol = context.params.get("zero_atol", DEFAULT_ZERO_ATOL)
        if not isinstance(zero_atol, int | float) or isinstance(zero_atol, bool):
            raise TypeError("fit.qrf@1 zero_atol must be numeric.")
        max_samples_leaf = context.params.get("max_samples_leaf")
        if max_samples_leaf is not None and (
            not isinstance(max_samples_leaf, int | float)
            or isinstance(max_samples_leaf, bool)
        ):
            raise TypeError(
                "fit.qrf@1 max_samples_leaf must be an int, float, or None."
            )

        seed = self._seed(context)
        return _QRFRunSpec(
            entity=input_slice.entity,
            target=owned.column,
            donor_target=donor_target,
            predictors=predictors,
            recipient_mask=owned.rows,
            n_estimators=n_estimators,
            zero_atol=float(zero_atol),
            max_samples_leaf=max_samples_leaf,
            seed=seed,
        )

    def _seed(self, context: KernelContext) -> int:
        """Resolve the one model seed under this instance's declared provenance."""
        if self.capabilities.seed_source is SeedSource.PARAM:
            if "seed" not in context.params:
                raise ValueError(
                    "A PARAM-seeded fit.qrf@1 node requires params['seed']."
                )
            seed = context.params["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise TypeError("fit.qrf@1 seed must be an integer.")
            return seed
        if "seed" in context.params:
            raise ValueError(
                "An EXECUTOR-seeded fit.qrf@1 node must omit params['seed']; "
                "its seed comes from KernelContext.rng."
            )
        return int(context.rng.integers(0, QRF_EXECUTOR_SEED_HIGH))

    @staticmethod
    def _table(context: KernelContext, spec: _QRFRunSpec) -> pd.DataFrame:
        """Return the entity table after validating its declared columns and ids."""
        try:
            table = context.tables[spec.entity]
        except KeyError as error:
            raise ValueError(
                f"fit.qrf@1 received no table for entity {spec.entity!r}."
            ) from error
        if not isinstance(table, pd.DataFrame):
            raise TypeError("KernelContext.tables values must be pandas DataFrames.")
        id_column = f"{spec.entity}_id"
        required = {
            id_column,
            spec.donor_target,
            *spec.predictors,
            _DONOR_MASK,
            spec.recipient_mask,
        }
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"fit.qrf@1's {spec.entity!r} table is missing column(s) {missing}."
            )
        ids = table[id_column]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(
                f"fit.qrf@1 requires non-null, unique values in {id_column!r}."
            )
        return table

    @staticmethod
    def _boolean_mask(table: pd.DataFrame, name: str) -> np.ndarray:
        """Read one non-null boolean mask without coercing truthy values."""
        values = table[name]
        if not pd.api.types.is_bool_dtype(values.dtype):
            raise TypeError(
                f"fit.qrf@1 mask {name!r} must have a boolean dtype, got "
                f"{values.dtype}."
            )
        if values.isna().any():
            raise ValueError(f"fit.qrf@1 mask {name!r} contains null values.")
        return values.to_numpy(dtype=np.bool_, copy=True)


QRF_EXECUTOR_KERNEL = QRFKernel(SeedSource.EXECUTOR)
"""Production graph kernel whose seed is derived from the node-key RNG."""

QRF_PARAM_KERNEL = QRFKernel(SeedSource.PARAM)
"""Legacy-parity kernel whose seed is a literal node parameter."""

QRF_KERNEL = QRF_EXECUTOR_KERNEL
"""Default graph kernel; new graphs derive their seed from node identity."""
