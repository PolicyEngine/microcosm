"""UK temporary-measure and frame adapters shared by full builds and scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from microcosm.build.target_materialization import (
    resolve_target_measures,
)
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    materialize_uk_ledger_targets,
)
from microcosm.calibrate import (
    TargetRegistry,
)
from microcosm.frame import Frame, Weights

__all__ = [
    "CalibrationFrameAdapter",
    "drop_injected_measure_inputs",
    "inject_measure_inputs",
]


def inject_measure_inputs(
    adapter: UKFrameTargetAdapter,
    measure_inputs: Mapping[tuple[str, str], np.ndarray],
) -> None:
    for (entity, variable), values in measure_inputs.items():
        adapter.tables[entity][variable] = values


def drop_injected_measure_inputs(
    adapter: UKFrameTargetAdapter,
    measure_inputs: Mapping[tuple[str, str], np.ndarray],
    original_columns: Mapping[str, set[str]],
) -> None:
    for entity, variable in measure_inputs:
        if variable not in original_columns[entity]:
            adapter.tables[entity].drop(columns=[variable], inplace=True)


class CalibrationFrameAdapter(UKFrameTargetAdapter):
    """The shared UK adapter plus the prepared-frame/restore lifecycle.

    Prepared measure columns are scratch state: they exist for constraint
    compilation only, and ``restore`` rebuilds pristine entity tables around
    the calibrated weights so the staged frame survives the HDFStore writer
    (slash-named scratch columns crash it).
    """

    def __init__(self, frame: Frame) -> None:
        super().__init__(frame)
        self._source_frame = frame
        self._original_tables = {
            name: table.copy() for name, table in self.tables.items()
        }

    def prepared_frame(self) -> Frame:
        return Frame(
            {**self.tables, **self.link_tables},
            self._source_frame.schema,
            {
                entity: self._source_frame.weights_for(entity)
                for entity in self._source_frame.weighted_entities
            },
            self._source_frame.strata,
            mass_log=self._source_frame.mass_log,
            metadata=self._source_frame.metadata,
        )

    def restore(self, calibrated: Frame) -> Frame:
        tables = {name: table.copy() for name, table in self._original_tables.items()}
        tables.update({name: table.copy() for name, table in self.link_tables.items()})
        return Frame(
            tables,
            calibrated.schema,
            {
                entity: Weights(
                    calibrated.weights_for(entity).values,
                    calibrated.weights_for(entity).kind,
                )
                for entity in calibrated.weighted_entities
            },
            calibrated.strata,
            mass_log=calibrated.mass_log,
            metadata=calibrated.metadata,
        )


def prepare_uk_target_frame(
    frame: Frame,
    registry: TargetRegistry,
    *,
    period: int | str,
    measure_resolver: object | None,
) -> tuple[Frame, Mapping[str, Any] | None]:
    """Materialize a registry's measures onto a frame, for scoring.

    The same resolve-inject-materialize route the full build takes,
    without the solve: scoring a UK register against a raw exported H5 cannot
    work, because every packaged reference binds a slash-named prepared
    measure that calibration deliberately strips before export. A skipped
    target refuses rather than quietly shrinking the surface both sides are
    compared on.
    """

    resolution = None
    if measure_resolver is not None:
        resolution = resolve_target_measures(
            lambda: CalibrationFrameAdapter(frame),
            registry,
            measure_resolver,
            period=period,
        )
    adapter = CalibrationFrameAdapter(frame)
    if resolution is not None:
        inject_measure_inputs(adapter, resolution.measure_inputs)
    materialized = materialize_uk_ledger_targets(adapter, registry, period=period)
    if materialized.skipped:
        raise RuntimeError(
            "target measures did not materialize for scoring: "
            f"{[skip.__dict__ for skip in materialized.skipped]}."
        )
    return adapter.prepared_frame(), (
        None if resolution is None else dict(resolution.receipt)
    )


# Compatibility aliases for the established internal call sites.
_CalibrationFrameAdapter = CalibrationFrameAdapter
_inject_measure_inputs = inject_measure_inputs
_drop_injected_measure_inputs = drop_injected_measure_inputs
