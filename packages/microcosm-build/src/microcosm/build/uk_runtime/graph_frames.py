"""The owned cells and created frames the UK graph kernels hand the executor.

A kernel returns each cell its node owns at the dtype the node declares for it,
and a root kernel returns its created frame with those cells cast and the build
period stamped. The spine's create, claim and stage kernels and the national
line's bound spine root share these helpers, so each hashes this module rather
than the module of another kernel family (microcosm#931).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from microcosm.frame import Frame
from microcosm.graph.population import dtype_for_token

if TYPE_CHECKING:
    from microcosm.graph import KernelContext

__all__ = ["cast_owned", "normalize_create_frame"]


def cast_owned(series: pd.Series, dtype: str) -> pd.Series:
    """``series`` at the dtype its node declares for it (the token ``dtype``)."""

    target = dtype_for_token(dtype)
    if series.dtype != target:
        series = series.astype(target)
    return series


def normalize_create_frame(frame: Frame, context: KernelContext) -> Frame:
    """A created frame with its node's owned cells at their declared dtypes
    and the build period of the node's ``time_period`` parameter."""

    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    for owned in context.node.outputs:
        tables[owned.entity][owned.column] = cast_owned(
            tables[owned.entity][owned.column], owned.dtype
        ).array
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            "time_period": str(context.params.get("time_period", "2024")),
        },
    )
