"""US source operation handlers for declarative manifests."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from populace.build.source_manifest import SourceOperationSpec
from populace.build.source_runtime import (
    SourceOperationHandler,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from populace.build.us.puf_aggregate_records import (
    disaggregate_puf_aggregate_records,
    load_default_puf_aggregate_disaggregation_spec,
)

__all__ = [
    "disaggregate_us_puf_aggregate_records_from_manifest",
    "us_source_operation_handlers",
]

_PUF_AGGREGATE_DISAGGREGATION_PARAMETER_KEYS = frozenset(
    {
        "method",
        "spec",
        "replace_records",
        "weight",
        "amount_columns",
        "seed_from_build_config",
    }
)


def us_source_operation_handlers() -> Mapping[str, SourceOperationHandler]:
    """Return US handlers keyed by manifest operation kind."""

    return {
        "disaggregate_aggregate_records": (
            disaggregate_us_puf_aggregate_records_from_manifest
        ),
    }


def disaggregate_us_puf_aggregate_records_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Execute the US PUF aggregate-row disaggregation manifest operation."""

    if operation.kind != "disaggregate_aggregate_records":
        raise SourceRuntimeError(
            "PUF aggregate-row handler received unexpected operation "
            f"{operation.kind!r}."
        )
    params = operation.parameters
    unexpected = sorted(set(params) - _PUF_AGGREGATE_DISAGGREGATION_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation received unsupported "
            f"parameter(s): {unexpected}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires a current source frame."
        )
    if params.get("spec") != "puf_aggregate_record_disaggregation":
        raise SourceRuntimeError(
            "US disaggregate_aggregate_records currently supports only "
            "spec='puf_aggregate_record_disaggregation'."
        )
    if params.get("method") != "donor_template_calibration":
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires "
            "method='donor_template_calibration'."
        )

    spec = load_default_puf_aggregate_disaggregation_spec()
    replace_records = tuple(int(value) for value in params.get("replace_records", ()))
    if replace_records != spec.aggregate_recids:
        raise SourceRuntimeError(
            "PUF aggregate-record manifest replace_records do not match the "
            f"packaged spec: {replace_records} != {spec.aggregate_recids}."
        )

    weight = params.get("weight")
    if weight not in {"s006", "S006"}:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires weight='s006'."
        )
    if params.get("amount_columns") != "irs_puf_amount_columns":
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires "
            "amount_columns='irs_puf_amount_columns'."
        )
    if params.get("seed_from_build_config") is not True:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires seed_from_build_config=true."
        )

    return disaggregate_puf_aggregate_records(
        frame,
        seed=int(context.config.seed),
        spec=spec,
    )
