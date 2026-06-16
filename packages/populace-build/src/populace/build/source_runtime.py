"""Shared interpreter for declarative source-stage manifests.

Country manifests describe source operations as data. This module executes the
generic envelope of that plan: table reads, operation dispatch, and explicit
stop points for staged/cached builds. Operation implementations are injected by
shared runtimes, not named inside manifests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from populace.build.source_manifest import SourceOperationSpec, SourceStageSpec

__all__ = [
    "SourceOperationHandler",
    "SourceRuntimeConfig",
    "SourceRuntimeContext",
    "SourceRuntimeError",
    "UnsupportedSourceOperationError",
    "run_source_stage",
]


@dataclass(frozen=True)
class SourceRuntimeConfig:
    """Build knobs available to manifest operation handlers."""

    seed: int = 0
    target_year: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRuntimeContext:
    """Runtime context passed to injected source operation handlers."""

    config: SourceRuntimeConfig
    tables: Mapping[str, pd.DataFrame]

    def read_table(self, name: str) -> pd.DataFrame:
        """Return a defensive copy of a declared source table."""

        try:
            table = self.tables[name]
        except KeyError as exc:
            available = sorted(self.tables)
            raise SourceRuntimeError(
                f"Source table {name!r} was not provided; available tables: "
                f"{available}."
            ) from exc
        if not isinstance(table, pd.DataFrame):
            raise SourceRuntimeError(
                f"Source table {name!r} must be a pandas DataFrame, got "
                f"{type(table).__name__}."
            )
        return table.copy(deep=True)


SourceOperationHandler = Callable[
    [pd.DataFrame | None, SourceOperationSpec, SourceRuntimeContext],
    pd.DataFrame,
]


class SourceRuntimeError(RuntimeError):
    """Base error for source manifest execution failures."""


class UnsupportedSourceOperationError(SourceRuntimeError):
    """Raised when a manifest operation has no injected runtime handler."""


def run_source_stage(
    stage: SourceStageSpec,
    *,
    tables: Mapping[str, pd.DataFrame],
    operation_handlers: Mapping[str, SourceOperationHandler] | None = None,
    config: SourceRuntimeConfig | None = None,
    stop_after: str | None = None,
) -> pd.DataFrame:
    """Execute a source-stage manifest against explicit source tables.

    Args:
        stage: Declarative source-stage contract.
        tables: Table name -> DataFrame inputs. The runtime never discovers
            source data by importing country packages.
        operation_handlers: Injected implementations keyed by operation kind.
        config: Build knobs such as seed and target year.
        stop_after: Optional operation kind at which to return the intermediate
            frame. This lets release builds cache stage prefixes such as
            ``read_table -> disaggregate_aggregate_records`` before later
            uprating/fitting work exists.
    """

    handlers = dict(operation_handlers or {})
    context = SourceRuntimeContext(
        config=config or SourceRuntimeConfig(),
        tables=tables,
    )
    current: pd.DataFrame | None = None

    for operation in stage.operations:
        if operation.kind == "read_table":
            if current is not None:
                raise SourceRuntimeError(
                    f"Stage {stage.stage!r} attempted to read a second primary "
                    "table into an existing frame."
                )
            current = _run_read_table(operation, context)
        else:
            handler = handlers.get(operation.kind)
            if handler is None:
                raise UnsupportedSourceOperationError(
                    f"Stage {stage.stage!r} operation {operation.kind!r} has no "
                    "injected source runtime handler."
                )
            current = handler(current, operation, context)
            if not isinstance(current, pd.DataFrame):
                raise SourceRuntimeError(
                    f"Stage {stage.stage!r} operation {operation.kind!r} returned "
                    f"{type(current).__name__}, expected pandas DataFrame."
                )

        if stop_after == operation.kind:
            if current is None:  # pragma: no cover - defensive
                raise SourceRuntimeError(
                    f"Stage {stage.stage!r} stop point {stop_after!r} produced no "
                    "frame."
                )
            return current.copy(deep=True)

    if current is None:
        raise SourceRuntimeError(f"Stage {stage.stage!r} produced no source frame.")
    if stop_after is not None:
        raise SourceRuntimeError(
            f"Stage {stage.stage!r} did not contain stop point {stop_after!r}."
        )
    return current.copy(deep=True)


def _run_read_table(
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    table = operation.parameters.get("table")
    if not isinstance(table, str) or not table:
        raise SourceRuntimeError("read_table operation requires a non-empty table.")
    return context.read_table(table)
