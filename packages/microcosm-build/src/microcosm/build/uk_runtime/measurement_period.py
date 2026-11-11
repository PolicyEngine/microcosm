"""How each UK plain-column row's value was measured (microcosm#1324).

A plain-column binding's value variable is measured at the binding's period:
through the measure provider at the calibration period, or, by declaration,
as the frame's stored column at the frame's own period. This module rebuilds
that route for every row of a finished build from its contract, the frame's
stored columns and the resolver's attached routes, and names any row whose
stored value was read at a period it does not hold. The receipt carries
target names, periods and route labels only: no figures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from microcosm.build.target_materialization import expression_names

UK_MEASUREMENT_PERIOD_RECEIPT_SCHEMA = 1

_COUNT_VARIABLES = frozenset({"person_count", "household_count", "benunit_count"})
#: Provider routes that read the frame's stored columns. Only the high-value
#: home masks do so at any period on purpose (their values are calendar-2024
#: Land Registry valuations); any other such route at a period the frame does
#: not hold is an off-period read.
_FRAME_COLUMN_ROUTE = "frame_columns:"
_DELIBERATE_FRAME_ROUTES = ("frame_columns:high_value_home_mask",)


def _value_variables(binding: Mapping[str, Any]) -> tuple[str, ...]:
    if "value_expression" in binding:
        # The binding interpreter's own reading of the expression: a split on
        # "+" alone would read a subtraction as one name.
        return expression_names(str(binding["value_expression"]))
    if "value_variable" in binding:
        return (str(binding["value_variable"]),)
    return ()


def uk_value_read_routes(
    contract_targets: Mapping[str, Mapping[str, Any]],
    rows: Iterable[tuple[str, str, str]],
    *,
    stored_columns: Mapping[str, Iterable[str]],
    frame_period: str,
    calibration_period: int | str,
    attached: Mapping[str, str],
) -> dict[str, Any]:
    """Each plain-column row's value route, and the rows read off-period.

    ``rows`` are ``(target name, contract target id, entity)`` triples, as a
    calibration diagnostics row names them. ``attached`` is the resolver's
    ``entity.variable`` → route map from the build record.
    """

    stored = {entity: frozenset(names) for entity, names in stored_columns.items()}
    out_rows: list[dict[str, Any]] = []
    off_period: list[str] = []
    for name, contract_id, row_entity in rows:
        target = contract_targets.get(str(contract_id))
        if target is None:
            continue
        binding = (target.get("bindings") or {}).get("policyengine") or {}
        if binding.get("kind"):
            continue
        entity = str(binding.get("from_entity") or row_entity)
        period = str(binding.get("measurement_period", calibration_period))
        declared = "measurement_period" in binding
        routes: dict[str, str] = {}
        for variable in _value_variables(binding):
            if variable in _COUNT_VARIABLES or variable == f"{entity}_count":
                routes[variable] = "count"
                continue
            is_stored = variable in stored.get(entity, frozenset())
            if is_stored and period == str(frame_period):
                routes[variable] = (
                    "frame_period_declared" if declared else "frame_period"
                )
            elif f"{entity}.{variable}" in attached:
                route = str(attached[f"{entity}.{variable}"])
                routes[variable] = (
                    "stored_read_off_period:" + route
                    if route.startswith(_FRAME_COLUMN_ROUTE)
                    and not route.startswith(_DELIBERATE_FRAME_ROUTES)
                    and period != str(frame_period)
                    else "provider:" + route
                )
            elif is_stored:
                routes[variable] = "stored_read_off_period"
            else:
                routes[variable] = "unresolved"
        if any(route.startswith("stored_read_off_period") for route in routes.values()):
            off_period.append(name)
        out_rows.append(
            {"target": name, "measurement_period": period, "value_routes": routes}
        )
    return {
        "schema_version": UK_MEASUREMENT_PERIOD_RECEIPT_SCHEMA,
        "frame_period": str(frame_period),
        "calibration_period": str(calibration_period),
        "rows": out_rows,
        "stored_reads_off_period": sorted(off_period),
    }


__all__ = [
    "UK_MEASUREMENT_PERIOD_RECEIPT_SCHEMA",
    "uk_value_read_routes",
]
