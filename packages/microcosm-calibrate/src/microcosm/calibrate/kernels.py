"""Graph kernels for the legacy calibration computation.

``calibrate.adam@1`` is deliberately a thin adapter around
:func:`microcosm.calibrate.calibrate`.  It reconstructs the minimal
:class:`~microcosm.frame.Frame` that the public API expects from the one entity
table visible through :class:`~microcosm.graph.KernelContext`, compiles the
declared target tuples, and returns the public call's typed calibrated weights.

The target tuple's standard error is transported unchanged in the receipt but
is not passed to the solver.  Today's public calibration loss does not consume
standard errors, which is reflected by ``capabilities.consumes_se=False``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import microcosm.calibrate.diagnostics as diagnostics_module
import microcosm.calibrate.matrix as matrix_module
import microcosm.calibrate.solve as solve_module
import microcosm.calibrate.target as target_module
import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.schema as frame_schema_module
import microcosm.frame.weights as frame_weights_module
from microcosm.calibrate import Target, TargetSet, calibrate, diagnostics_payload
from microcosm.frame import EntitySchema, Frame
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    SeedSource,
    StructuralDelta,
    source_hash,
)

__all__ = ["CALIBRATE_ADAM", "CalibrateAdamKernel"]


_PARAMS = frozenset(
    {
        "epochs",
        "learning_rate",
        "mass",
        "max_weight_ratio",
        "targets",
        # Which weights the cap is measured against. The executor enforces the
        # anchor (charter D3); the kernel records it so the receipt says what
        # the cap meant.
        "weight_anchor",
    }
)


def _dummy_group_columns(entity: str, columns: pd.Index) -> tuple[str, str, str]:
    """Choose deterministic synthetic linkage names absent from ``columns``."""

    index = 0
    while True:
        group = f"__calibration_kernel_group_{index}"
        group_id = f"{group}_id"
        membership = f"{entity}_{group}_id"
        if group != entity and group_id not in columns and membership not in columns:
            return group, group_id, membership
        index += 1


def _frame_from_context(context: KernelContext, entity: str) -> Frame:
    """Build a row-order-preserving minimal frame for one calibrated entity.

    ``EntitySchema`` requires a person entity and at least one group.  Treating
    the calibrated entity as the synthetic person entity preserves its exact
    table and weight order; a one-row dummy group supplies the required linkage
    without affecting any calibration constraint.
    """

    table = context.tables[entity].copy(deep=True)
    group, group_id, membership = _dummy_group_columns(entity, table.columns)
    table[membership] = np.zeros(len(table), dtype=np.int64)
    group_table = pd.DataFrame({group_id: np.zeros(1, dtype=np.int64)})
    schema = EntitySchema(person_entity=entity, group_entities=(group,))
    return Frame(
        {entity: table, group: group_table},
        schema,
        {entity: context.weights[entity]},
    )


def _declared_targets(
    context: KernelContext,
    entity: str,
) -> tuple[tuple[object, ...], TargetSet]:
    """Return the untouched declaration plus its solver ``TargetSet``."""

    raw_targets = context.params.get("targets")
    if not isinstance(raw_targets, tuple):
        raise TypeError(
            "calibrate.adam@1 parameter 'targets' must be a tuple of "
            "(name, measure_column, filter_column, value, se) tuples."
        )

    targets: list[Target] = []
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, tuple) or len(raw_target) != 5:
            raise TypeError(
                f"calibrate.adam@1 target {index} must be a five-item tuple "
                "(name, measure_column, filter_column, value, se)."
            )
        name, measure_column, filter_column, value, se = raw_target
        if not isinstance(name, str) or not isinstance(measure_column, str):
            raise TypeError(
                f"calibrate.adam@1 target {index} name and measure column "
                "must be strings."
            )
        if filter_column is not None and not isinstance(filter_column, str):
            raise TypeError(
                f"calibrate.adam@1 target {index} filter column must be a "
                "string or None."
            )
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"calibrate.adam@1 target {index} value must be numeric.")
        if se is not None:
            if isinstance(se, bool) or not isinstance(se, int | float):
                raise TypeError(
                    f"calibrate.adam@1 target {index} standard error must be "
                    "numeric or None."
                )
            if not np.isfinite(se) or se <= 0:
                raise ValueError(
                    f"calibrate.adam@1 target {index} standard error must be "
                    "positive and finite."
                )
        targets.append(
            Target(
                name=name,
                entity=entity,
                measure=measure_column,
                value=value,
                filter=filter_column,
            )
        )
    return raw_targets, TargetSet(targets)


def _integer_param(context: KernelContext, name: str) -> int:
    value = context.params.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"calibrate.adam@1 parameter {name!r} must be an integer.")
    return value


def _numeric_param(context: KernelContext, name: str) -> int | float:
    value = context.params.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"calibrate.adam@1 parameter {name!r} must be numeric.")
    return value


class CalibrateAdamKernel(KernelBase):
    """Expose the public Adam calibrator through the graph kernel protocol."""

    ref = "calibrate.adam@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        # A weight transition is a new population version: every later node
        # reads the calibrated weights, so the node that produces them is
        # structural (charter D1; interface amendment 6).
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )

    def implementation_hash(self) -> str:
        """Bind both adapter and wrapped public diagnostic/solver sources."""

        return source_hash(
            type(self),
            calibrate,
            diagnostics_payload,
            solve_module,
            diagnostics_module,
            matrix_module,
            target_module,
            frame_bundle_module,
            frame_schema_module,
            frame_weights_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        """Calibrate the node's declared entity and return its typed weights."""

        unknown = sorted(set(context.params) - _PARAMS)
        if unknown:
            raise ValueError(
                f"calibrate.adam@1 received unknown parameter(s): {unknown}."
            )

        transition = context.node.weights
        if transition is None or transition.to_kind != "calibrated":
            raise ValueError(
                "calibrate.adam@1 requires a WeightTransition whose to_kind is "
                "'calibrated'."
            )
        entity = transition.entity

        mass = context.params.get("mass")
        if not isinstance(mass, str):
            raise TypeError("calibrate.adam@1 parameter 'mass' must be a string.")
        if mass != transition.mass:
            raise ValueError(
                "calibrate.adam@1 mass parameter must match the declared weight "
                f"transition: parameter {mass!r}, transition {transition.mass!r}."
            )

        max_weight_ratio = context.params.get("max_weight_ratio")
        if max_weight_ratio is not None and (
            isinstance(max_weight_ratio, bool)
            or not isinstance(max_weight_ratio, int | float)
        ):
            raise TypeError(
                "calibrate.adam@1 parameter 'max_weight_ratio' must be numeric or None."
            )
        epochs = _integer_param(context, "epochs")
        learning_rate = _numeric_param(context, "learning_rate")
        declared_targets, targets = _declared_targets(context, entity)
        frame = _frame_from_context(context, entity)

        result = calibrate(
            frame,
            targets,
            weight_entity=entity,
            method="adam",
            max_weight_ratio=max_weight_ratio,
            epochs=epochs,
            learning_rate=learning_rate,
            mass=mass,
            seed=0,
        )
        return KernelResult(
            weights=result.frame.weights_for(entity),
            receipt={
                "declared_targets": declared_targets,
                "diagnostics": diagnostics_payload(result),
            },
        )


CALIBRATE_ADAM = CalibrateAdamKernel()
