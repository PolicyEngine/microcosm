"""Declared uprating of UK imputation donors to the base year (microcosm#890 U).

The LCFS 2023-24 and ETB FYE-2024 donors are one year behind the FRS 2024-25
spine. Each stage declares an ``uprate_donor_columns`` operation naming, per
column, the basis that moves it from ``from_period`` to ``to_period``:

* ``engine_parameter`` — the ratio of a PolicyEngine UK parameter at the two
  January-first instants (the SPI-donor precedent in ``spi_income``), read from
  the installed engine so the factor stays lockstepped with what the engine
  itself applies afterwards;
* ``vendored_litre_proxy`` — publisher outturn for road fuel: the DESNZ annual
  pump-price ratio times the HMRC fiscal-year litres ratio over the ONS
  population ratio, all read from the vendored Chronicle facts (María's ruling
  of 2026-09-14: the engine's own litre-proxy indices reproduce its 2023 pump
  prices, which disagree with DESNZ, so they are audited, not applied);
* ``vendored_ratio`` — one vendored concept's value at two periods.

Every factor and its inputs are returned as a receipt the stage records in its
evidence; nothing is derived silently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

UPRATE_DONOR_COLUMNS_KIND = "uprate_donor_columns"
BASIS_ENGINE_PARAMETER = "engine_parameter"
BASIS_VENDORED_LITRE_PROXY = "vendored_litre_proxy"
BASIS_VENDORED_RATIO = "vendored_ratio"

ParameterReader = Callable[[str, int], float]


def uprating_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``uprate_donor_columns`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == UPRATE_DONOR_COLUMNS_KIND:
            return dict(operation.parameters)
    return None


def engine_parameter_reader() -> ParameterReader:
    """Read a parameter path at a January-first instant from the installed engine."""

    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()

    def read(path: str, year: int) -> float:
        node = system.parameters(f"{int(year)}-01-01")
        try:
            for part in path.split("."):
                node = getattr(node, part)
        except AttributeError as error:
            raise ValueError(f"Missing engine parameter {path!r}.") from error
        return float(node)

    return read


def donor_uprating_factors(
    parameters: Mapping[str, Any],
    *,
    parameter_reader: ParameterReader | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Resolve every declared column factor and return (factors, receipt)."""

    from_period = int(parameters["from_period"])
    to_period = int(parameters["to_period"])
    if to_period < from_period:
        raise ValueError("uprate_donor_columns cannot move a donor backwards.")
    columns = parameters.get("columns")
    if not isinstance(columns, Mapping) or not columns:
        raise ValueError("uprate_donor_columns declares no columns.")
    exempt = tuple(str(column) for column in parameters.get("exempt", ()))
    overlap = sorted(set(exempt) & set(columns))
    if overlap:
        raise ValueError(
            f"uprate_donor_columns lists exempt columns as uprated: {overlap}."
        )
    reader = parameter_reader
    factors: dict[str, float] = {}
    receipt_columns: dict[str, dict[str, Any]] = {}
    for column, spec in columns.items():
        basis = str(spec.get("basis") or "")
        if basis == BASIS_ENGINE_PARAMETER:
            if reader is None:
                reader = engine_parameter_reader()
            path = str(spec["parameter_path"])
            before = reader(path, from_period)
            after = reader(path, to_period)
            _require_positive(column, before, after)
            factor = after / before
            detail = {"parameter_path": path, "before": before, "after": after}
        elif basis == BASIS_VENDORED_LITRE_PROXY:
            factor, detail = _litre_proxy_factor(spec, from_period, to_period)
            audit_path = spec.get("audit_parameter_path")
            if audit_path:
                if reader is None:
                    reader = engine_parameter_reader()
                detail["engine_audit"] = _engine_index_audit(
                    reader, str(audit_path), from_period, to_period, factor
                )
        elif basis == BASIS_VENDORED_RATIO:
            factor, detail = _vendored_ratio_factor(spec, from_period, to_period)
        else:
            raise ValueError(
                f"uprate_donor_columns column {column!r} has unknown basis {basis!r}."
            )
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(
                f"uprate_donor_columns factor for {column!r} is not positive."
            )
        factors[str(column)] = float(factor)
        receipt_columns[str(column)] = {
            "basis": basis,
            "factor": float(factor),
            **detail,
        }
    receipt = {
        "operation": UPRATE_DONOR_COLUMNS_KIND,
        "from_period": from_period,
        "to_period": to_period,
        "instant": str(parameters.get("instant", "january_first")),
        "columns": receipt_columns,
        "exempt": list(exempt),
    }
    return factors, receipt


def apply_donor_uprating(
    table: pd.DataFrame, factors: Mapping[str, float]
) -> pd.DataFrame:
    """Multiply each declared column by its factor; refuse a column the donor lacks."""

    missing = [column for column in factors if column not in table]
    if missing:
        raise KeyError(f"donor table lacks uprated column(s): {missing}.")
    result = table.copy()
    for column, factor in factors.items():
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(
            0.0
        ) * float(factor)
    return result


def _litre_proxy_factor(
    spec: Mapping[str, Any], from_period: int, to_period: int
) -> tuple[float, dict[str, Any]]:
    resource = str(spec["resource"])
    price_from = _single_value(
        resource,
        concept=str(spec["price_concept"]),
        period_type="calendar_year",
        period_value=from_period,
    )
    price_to = _single_value(
        resource,
        concept=str(spec["price_concept"]),
        period_type="calendar_year",
        period_value=to_period,
    )
    litres_from = _single_value(
        resource,
        concept=str(spec["litres_concept"]),
        period_type="fiscal_year",
        fiscal_start=f"{from_period}-04-01",
    )
    litres_to = _single_value(
        resource,
        concept=str(spec["litres_concept"]),
        period_type="fiscal_year",
        fiscal_start=f"{to_period}-04-01",
    )
    population_from = _single_value(
        resource,
        concept=str(spec["population_concept"]),
        period_type="calendar_year",
        period_value=from_period,
    )
    population_to = _single_value(
        resource,
        concept=str(spec["population_concept"]),
        period_type="calendar_year",
        period_value=to_period,
    )
    for name, value in (
        ("price", price_from),
        ("price", price_to),
        ("litres", litres_from),
        ("litres", litres_to),
        ("population", population_from),
        ("population", population_to),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"vendored litre proxy {name} input must be positive.")
    price_ratio = price_to / price_from
    litres_ratio = litres_to / litres_from
    population_ratio = population_to / population_from
    factor = price_ratio * litres_ratio / population_ratio
    detail = {
        "resource": resource,
        "price_concept": spec["price_concept"],
        "price": {
            str(from_period): price_from,
            str(to_period): price_to,
            "ratio": price_ratio,
        },
        "litres_concept": spec["litres_concept"],
        "litres": {
            f"fy{from_period}": litres_from,
            f"fy{to_period}": litres_to,
            "ratio": litres_ratio,
        },
        "population_concept": spec["population_concept"],
        "population": {
            str(from_period): population_from,
            str(to_period): population_to,
            "ratio": population_ratio,
        },
        "formula": "price_ratio * litres_ratio / population_ratio",
    }
    return float(factor), detail


def _vendored_ratio_factor(
    spec: Mapping[str, Any], from_period: int, to_period: int
) -> tuple[float, dict[str, Any]]:
    resource = str(spec["resource"])
    concept = str(spec["concept"])
    criteria: dict[str, Any] = {}
    if spec.get("dimension_values"):
        criteria["dimensions"] = dict(spec["dimension_values"])
    if spec.get("geography_id"):
        criteria["geography_id"] = str(spec["geography_id"])
    numerator_period = str(spec.get("numerator_period") or f"{to_period}-04-01")
    denominator_period = str(spec.get("denominator_period") or f"{from_period}-04-01")
    numerator = _single_value(
        resource, concept=concept, fiscal_start=numerator_period, **criteria
    )
    denominator = _single_value(
        resource, concept=concept, fiscal_start=denominator_period, **criteria
    )
    if (
        not np.isfinite([numerator, denominator]).all()
        or denominator <= 0
        or numerator <= 0
    ):
        raise ValueError(f"vendored ratio for {concept!r} needs two positive values.")
    detail = {
        "resource": resource,
        "concept": concept,
        "numerator": {"fiscal_start": numerator_period, "value": numerator},
        "denominator": {"fiscal_start": denominator_period, "value": denominator},
        **(
            {"dimension_values": dict(spec["dimension_values"])}
            if spec.get("dimension_values")
            else {}
        ),
        **(
            {"geography_id": str(spec["geography_id"])}
            if spec.get("geography_id")
            else {}
        ),
    }
    return float(numerator / denominator), detail


def _engine_index_audit(
    reader: ParameterReader,
    path: str,
    from_period: int,
    to_period: int,
    outturn_factor: float,
) -> dict[str, Any]:
    """Record the engine's own index ratio beside the applied outturn factor.

    The engine's ``*_spending_litre_proxy`` indices are not applied (they
    reproduce its 2023 pump-price parameters, which disagree with DESNZ);
    the ratio is recorded so the divergence stays visible in the evidence.
    """

    before = reader(path, from_period)
    after = reader(path, to_period)
    _require_positive(path, before, after)
    engine_ratio = after / before
    return {
        "parameter_path": path,
        "before": before,
        "after": after,
        "engine_ratio": engine_ratio,
        "outturn_over_engine": float(outturn_factor) / engine_ratio,
        "applied": "outturn",
    }


def _single_value(resource: str, **criteria: Any) -> float:
    rows = vendored_rows(resource, **criteria)
    if len(rows) != 1:
        raise ValueError(
            f"{resource}: expected exactly one row for {criteria}, found {len(rows)}."
        )
    return float(rows[0]["value"])


def _require_positive(column: str, before: float, after: float) -> None:
    if not np.isfinite([before, after]).all() or before <= 0 or after <= 0:
        raise ValueError(f"uprate_donor_columns index must be positive for {column!r}.")
