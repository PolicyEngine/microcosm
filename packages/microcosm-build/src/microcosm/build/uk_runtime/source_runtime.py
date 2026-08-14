"""UK source runtime seams for the shared source-stage manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import (
    SourceOperationHandler,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.frame import Frame
from microcosm.frame.rules import materialize_rules_engine_predictors

__all__ = [
    "materialize_uk_rules_engine_predictors_from_manifest",
    "uk_source_operation_handlers",
    "uk_stage_implementations",
]


def uk_stage_implementations(
    *,
    retained_leaves_transform: Callable[[Frame], Frame],
    hmrc_income_transform: Callable[[Frame], Frame],
) -> dict[str, Callable[[Frame], Frame]]:
    """Return the whole-stage implementation map for the UK source plan."""

    return {
        "frs_hmrc_retained_leaves": retained_leaves_transform,
        "hmrc_spi_income": hmrc_income_transform,
    }


def uk_source_operation_handlers() -> Mapping[str, SourceOperationHandler]:
    """Return UK operation handlers available to declarative source runtimes."""

    return {
        "materialize_rules_engine_predictors": (
            materialize_uk_rules_engine_predictors_from_manifest
        )
    }


def materialize_uk_rules_engine_predictors_from_manifest(
    current: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> Frame:
    """Materialize declared rules-engine predictors onto a runtime Frame."""

    if current is not None:
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors operates on the runtime Frame, "
            "not an intermediate source table."
        )
    frame = _extra(context, "frame", Frame)
    engine = _extra(context, "rules_engine", object)
    country = _optional_country(context)
    period = context.config.extra.get("period", context.config.target_year)
    if period is None:
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors requires a period in the "
            "runtime context."
        )
    predictors = operation.parameters.get("predictors")
    if not isinstance(predictors, list) or not all(
        isinstance(name, str) and name for name in predictors
    ):
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors requires a non-empty "
            "'predictors' list."
        )
    try:
        return materialize_rules_engine_predictors(
            frame,
            variables=tuple(predictors),
            period=period,
            engine=engine,
            country=country,
        )
    except ValueError as error:
        raise SourceRuntimeError(str(error)) from error


def _extra(
    context: SourceRuntimeContext,
    key: str,
    expected_type: type,
) -> Any:
    value = context.config.extra.get(key)
    if not isinstance(value, expected_type):
        raise SourceRuntimeError(
            f"materialize_rules_engine_predictors requires context.config.extra"
            f"[{key!r}] to be a {expected_type.__name__}."
        )
    return value


def _optional_country(context: SourceRuntimeContext) -> str | None:
    country = context.config.extra.get("country")
    if country is None:
        return None
    if not isinstance(country, str) or not country:
        raise SourceRuntimeError(
            "materialize_rules_engine_predictors context country must be a "
            "non-empty string."
        )
    return country
