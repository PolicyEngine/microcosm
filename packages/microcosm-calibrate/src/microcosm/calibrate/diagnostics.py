"""Serialize a calibration's diagnostics so they travel with the artifact.

A :class:`~microcosm.calibrate.solve.CalibrationResult` carries everything a
reviewer needs to audit what calibration did — per-target estimates before
and after, the per-epoch loss trajectory, the targets that failed to compile
*and why*, and the solver options actually used. Until now none of it left
the build machine: the build pushed the diagnostics to telemetry and dropped
them, and the published ``.npz`` kept only closing scalars. "Skipped and
reported, never dropped silently" is only true if the report ships.

:func:`diagnostics_payload` renders the result as a JSON-stable dict, and
:func:`write_calibration_diagnostics` writes it as
``calibration_diagnostics.json`` — the artifact a release publishes next to
its manifests (charter rule: artifacts carry their environment; a published
dataset's calibration evidence belongs with the dataset, not in telemetry).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.calibrate._target_loss_attribution import (
    TARGET_LOSS_ATTRIBUTION_WARNING_CODES,
    TargetLossAttributionError,
    assemble_target_loss_attribution,
)
from microcosm.calibrate.provider_labels import calibration_provider_label
from microcosm.calibrate.solve import CalibrationResult
from microcosm.calibrate.variable_labels import calibration_variable_label

__all__ = [
    "CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION",
    "diagnostics_payload",
    "past_cap_census",
    "write_calibration_diagnostics",
]

#: Version of the diagnostics payload. Consumers (dashboards, scorers) key
#: their readers on it; bump it with any shape change.
#: v4 added the weight-concentration scalars (``effective_sample_size``,
#: ``realized_max_weight_ratio``, ``top_1pct_weight_share``).
#: v5 added the ``past_cap_census`` block (rows past the loss cap at
#: initialization and at final, escaped/frozen/pushed-out counts, and the
#: pushed-out row list).
#: v6 added authoritative final per-target loss attribution and an explicit
#: warning-only degradation state when that supplementary attribution cannot
#: be validated.
#: v7 adds producer-defined source, variable, and dimension identity for
#: registry-backed release diagnostics. Sources include country-owned display
#: labels when registered. Geography is represented as a typed dimension with
#: stable identifiers and display labels.
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 7

_LOGGER = logging.getLogger(__name__)

_UK_GEOGRAPHY_LABELS = {
    "K02000001": "United Kingdom",
    "K03000001": "Great Britain",
    "E92000001": "England",
    "W92000004": "Wales",
    "S92000003": "Scotland",
    "N92000002": "Northern Ireland",
}

_US_STATE_POSTAL = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

_COUNT_UNITS = frozenset(
    {
        "count",
        "households",
        "people",
        "persons",
        "returns",
        "claims",
    }
)
_TOTAL_UNITS = frozenset({"gbp", "usd", "dollars", "pounds"})
_MEAN_UNITS = frozenset({"percent", "percentage", "rate", "ratio"})


def _finite(value: float) -> float | None:
    """JSON has no NaN/inf; a non-finite diagnostic serializes as null."""
    value = float(value)
    return value if math.isfinite(value) else None


def _jsonable(value: object) -> object:
    """An option value as strict JSON: non-finite floats become null."""
    if isinstance(value, float):
        return _finite(value)
    return value


def _selector_payload(selector: object) -> dict[str, str] | None:
    """A JSON-stable target measure/filter selector."""
    if selector is None:
        return None
    if isinstance(selector, str):
        return {"kind": "column", "name": selector}
    name = getattr(selector, "__qualname__", None) or getattr(
        selector, "__name__", None
    )
    if name is None:
        name = type(selector).__name__
    module = getattr(selector, "__module__", None)
    qualified = f"{module}.{name}" if module else str(name)
    return {"kind": "callable", "name": qualified}


def _strict_json_bytes(value: object) -> bytes:
    """Canonical bytes for hashes embedded in diagnostics artifacts."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _registry_payload(target_registry: object | None) -> dict[str, object] | None:
    """The target-registry identity, if the build supplied one."""
    if target_registry is None:
        return None
    country = getattr(target_registry, "country", None)
    version = getattr(target_registry, "version", None)
    specs = getattr(target_registry, "specs", ())
    return {
        "country": str(country) if country is not None else "",
        "version": str(version) if version is not None else "",
        "n_specs": len(tuple(specs)),
    }


def _registry_spec_lookup(target_registry: object | None) -> dict[str, object]:
    """Registry specs keyed by their compiled row labels."""
    if target_registry is None:
        return {}
    lookup: dict[str, object] = {}
    for spec in getattr(target_registry, "specs", ()):
        name = getattr(spec, "name", None)
        period = getattr(spec, "period", None)
        if name is not None and period is not None:
            lookup[f"{name}@{period}"] = spec
    return lookup


def _metadata_string(metadata: Mapping[str, object], key: str) -> str:
    """Return a stripped metadata string, or an empty string."""

    value = metadata.get(key)
    return value.strip() if isinstance(value, str) else ""


def _source_id(spec: object, metadata: Mapping[str, object]) -> str:
    """Return the publisher identifier declared by a registry-backed target."""

    explicit = _metadata_string(metadata, "diagnostic_source_id")
    if explicit:
        return explicit
    selector_source = _metadata_string(metadata, "ledger_selector_source_name")
    if selector_source:
        return selector_source
    source_record = _metadata_string(metadata, "ledger_source_record_id")
    if source_record:
        return source_record.split(".", 1)[0]
    family = str(getattr(spec, "family", "")).strip()
    if family:
        return family
    name = str(getattr(spec, "name", "")).strip()
    return re.split(r"[./]", name, maxsplit=1)[0] or "other"


def _variable_id(
    spec: object,
    metadata: Mapping[str, object],
    *,
    source_id: str,
) -> str:
    """Return a stable statistic identifier for a registry-backed target.

    Explicit diagnostic identifiers are already producer declarations and are
    preserved verbatim. Ledger measure concepts are older compound identifiers:
    some append ``_count`` or ``_amount`` even though schema 7 represents that
    distinction separately in ``variable.measure``. Remove only the suffix that
    agrees with the declared unit so count and amount rows remain one dashboard
    category without conflating their measurements.
    """

    def without_source_prefix(value: str) -> str:
        for prefix in (f"{source_id}.", f"{source_id}:"):
            if value.startswith(prefix):
                return value[len(prefix) :]
        return value

    for key in ("diagnostic_variable_id", "variable"):
        value = _metadata_string(metadata, key)
        if value:
            return without_source_prefix(value)

    measure = _variable_measure(metadata)
    measure_suffix = {"count": "_count", "total": "_amount"}.get(measure, "")
    for key in ("ledger_measure_concept", "ledger_source_concept"):
        value = _metadata_string(metadata, key)
        if value:
            identifier = without_source_prefix(value)
            if measure_suffix and identifier.endswith(measure_suffix):
                identifier = identifier[: -len(measure_suffix)]
            return identifier
    contract_id = _metadata_string(metadata, "contract_target_id")
    if contract_id:
        prefix = re.split(r"[./]", contract_id, maxsplit=1)[0]
        remainder = contract_id[len(prefix) :].lstrip("./")
        return remainder or contract_id
    measure = str(getattr(spec, "measure", "")).strip()
    return measure or str(getattr(spec, "name", "")).strip() or "unknown"


def _variable_measure(metadata: Mapping[str, object]) -> str:
    """Classify a declared Ledger unit into the dashboard's measure vocabulary."""

    unit = _metadata_string(metadata, "ledger_measure_unit").lower()
    if unit in _COUNT_UNITS:
        return "count"
    if unit in _TOTAL_UNITS:
        return "total"
    if unit in _MEAN_UNITS:
        return "mean"
    return ""


def _humanize_identifier(value: str) -> str:
    """Turn a machine identifier into a concise dimension label."""

    tail = value.rsplit("#", 1)[-1].rsplit(".", 1)[-1]
    return " ".join(part.capitalize() for part in re.split(r"[_:/-]+", tail) if part)


def _dimension_label(dimension_id: str) -> str:
    """Return the established display label for a Ledger dimension id."""

    if dimension_id == "us:statutes/26/62#adjusted_gross_income":
        return "Income Band"
    if dimension_id == "census_stc.item":
        return "Item"
    if dimension_id == "hhs_acf_tanf.spending_category":
        return "Spending Category"
    if dimension_id == "income_range":
        return "Income Band"
    if dimension_id == "filing_status":
        return "Filing Status"
    if dimension_id == "eitc_child_count":
        return "Qualifying Children"
    return _humanize_identifier(dimension_id)


def _normalized_geography_level(value: str) -> str:
    """Normalize producer aliases used by the existing country contracts."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return "local_authority" if normalized == "la" else normalized


def _geography_label(
    *,
    country: str,
    level: str,
    geography_id: str,
    metadata: Mapping[str, object],
) -> str:
    """Resolve a producer-owned geography label without consumer name parsing."""

    explicit = _metadata_string(metadata, "ledger_geography_name")
    if explicit:
        return explicit
    if country == "uk":
        return _UK_GEOGRAPHY_LABELS.get(geography_id, geography_id)
    if country == "us":
        if geography_id == "0100000US" or level in {"country", "national"}:
            return "United States"
        match = re.search(r"US(\d{2})(\d{2})$", geography_id)
        if level == "congressional_district" and match:
            postal = _US_STATE_POSTAL.get(match.group(1))
            if postal:
                return f"{postal}-{match.group(2)}"
        match = re.search(r"US(\d{2})$", geography_id)
        if level == "state" and match:
            return _US_STATE_POSTAL.get(match.group(1), geography_id)
    return geography_id


def _structured_dimensions(
    metadata: Mapping[str, object],
    *,
    country: str,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Build schema-7 row dimensions and their producer definitions."""

    values: dict[str, str] = {}
    definitions: dict[str, dict[str, object]] = {}

    geography_id = _metadata_string(metadata, "ledger_geography_id")
    geography_level = _normalized_geography_level(
        _metadata_string(metadata, "ledger_geography_level")
    )
    if geography_id and geography_level:
        dimension_id = f"geography_{geography_level}"
        values[dimension_id] = geography_id
        definitions[dimension_id] = {
            "label": _humanize_identifier(geography_level),
            "role": "geography",
            "level": geography_level,
            "values": {
                geography_id: _geography_label(
                    country=country,
                    level=geography_level,
                    geography_id=geography_id,
                    metadata=metadata,
                )
            },
            "order": [geography_id],
        }

    filter_dimensions = [
        (key.removeprefix("ledger_filter_"), raw_value.strip())
        for key, raw_value in metadata.items()
        if key.startswith("ledger_filter_")
        and isinstance(raw_value, str)
        and key.removeprefix("ledger_filter_")
        and raw_value.strip()
    ]
    layout_dimension = _metadata_string(metadata, "ledger_layout_groupby_dimension")
    layout_value = _metadata_string(metadata, "ledger_layout_groupby_value_id")
    layout_label = _dimension_label(layout_dimension)
    duplicate_filter = any(
        _dimension_label(dimension_id) == layout_label and value == layout_value
        for dimension_id, value in filter_dimensions
    )
    resolved_geography_label = (
        _geography_label(
            country=country,
            level=geography_level,
            geography_id=geography_id,
            metadata=metadata,
        )
        if geography_id and geography_level
        else ""
    )
    geography_layout = layout_dimension in {
        "geography",
        "state",
        "cms_medicaid.state_abbreviation",
    }
    redundant_geography = layout_value.lower() in {
        geography_id.lower(),
        resolved_geography_label.lower(),
    }
    if (
        layout_dimension
        and layout_value
        and not duplicate_filter
        and not geography_layout
        and not redundant_geography
    ):
        values[layout_dimension] = layout_value
        definitions[layout_dimension] = {
            "label": layout_label,
            "values": {layout_value: _humanize_identifier(layout_value)},
            "order": [layout_value],
        }

    for dimension_id, value in filter_dimensions:
        values[dimension_id] = value
        definitions[dimension_id] = {
            "label": _dimension_label(dimension_id),
            "values": {value: _humanize_identifier(value)},
            "order": [value],
        }
    return values, definitions


def _merge_dimension_definitions(
    destination: dict[str, dict[str, object]],
    additions: Mapping[str, Mapping[str, object]],
) -> None:
    """Merge per-row dimension declarations into one deterministic dictionary."""

    for dimension_id, addition in additions.items():
        current = destination.get(dimension_id)
        if current is None:
            destination[dimension_id] = {
                **addition,
                "values": dict(addition.get("values", {})),
                "order": list(addition.get("order", [])),
            }
            continue
        for key in ("label", "role", "level"):
            incoming = addition.get(key)
            if incoming is not None and current.get(key) != incoming:
                raise ValueError(
                    f"Diagnostics dimension {dimension_id!r} has conflicting "
                    f"{key} declarations {current.get(key)!r} and {incoming!r}."
                )
        current_values = current.setdefault("values", {})
        current_order = current.setdefault("order", [])
        if not isinstance(current_values, dict) or not isinstance(current_order, list):
            raise TypeError("Diagnostics dimension aggregation state is malformed.")
        for raw_value, label in addition.get("values", {}).items():
            existing = current_values.get(raw_value)
            if existing is not None and existing != label:
                raise ValueError(
                    f"Diagnostics dimension {dimension_id!r} value "
                    f"{raw_value!r} has conflicting labels {existing!r} and {label!r}."
                )
            current_values[raw_value] = label
        for raw_value in addition.get("order", []):
            if raw_value not in current_order:
                current_order.append(raw_value)


def _structured_target_fields(
    target: object,
    spec: object,
    *,
    country: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Serialize complete schema-7 identity for one registry-backed target."""

    metadata = dict(getattr(spec, "metadata", {}) or {})
    source_id = _source_id(spec, metadata)
    variable_id = _variable_id(spec, metadata, source_id=source_id)
    dimensions, definitions = _structured_dimensions(metadata, country=country)
    citation = str(getattr(target, "source", "")).strip()
    source: dict[str, str] = {"id": source_id}
    source_label = calibration_provider_label(country, source_id)
    if source_label:
        source["label"] = source_label
    if citation:
        source["citation"] = citation
        source_url = next(
            (
                part.strip()
                for part in citation.split("|")
                if part.strip().startswith(("https://", "http://"))
            ),
            "",
        )
        if source_url:
            source["url"] = source_url
    variable: dict[str, str] = {"id": variable_id}
    variable_label = calibration_variable_label(country, source_id, variable_id)
    if variable_label:
        variable["label"] = variable_label
    measure = _variable_measure(metadata)
    if measure:
        variable["measure"] = measure
    return {
        "source": source,
        "variable": variable,
        "dimensions": dimensions,
    }, definitions


def _target_identity_rows(result: CalibrationResult) -> list[dict[str, object]]:
    """The target surface as structured rows suitable for hashing."""
    rows: list[dict[str, object]] = []
    for index, (diagnostic, target) in enumerate(
        zip(result.diagnostics, result.problem.targets, strict=True)
    ):
        rows.append(
            {
                "row_name": diagnostic.name,
                "target_name": target.name,
                "period": target.period,
                "entity": target.entity,
                "measure": _selector_payload(target.measure),
                "filter": _selector_payload(target.filter),
                "source": target.source,
                "metadata": dict(target.metadata),
                "target": _finite(diagnostic.target),
                "compiled_target": _finite(result.problem.target_vector[index]),
            }
        )
    return rows


def _target_surface_payload(result: CalibrationResult) -> dict[str, object]:
    """Content-address the exact target surface the calibration solved."""
    rows = _target_identity_rows(result)
    names = [row["row_name"] for row in rows]
    values = [{"row_name": row["row_name"], "target": row["target"]} for row in rows]
    matrix = result.problem.matrix
    return {
        "schema_version": 1,
        "weight_entity": result.weight_entity,
        "n_targets": len(rows),
        "n_records": int(result.weights.shape[0]),
        "constraint_matrix": {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "nnz": int(matrix.nnz),
        },
        "sha256": hashlib.sha256(_strict_json_bytes(rows)).hexdigest(),
        "names_sha256": hashlib.sha256(_strict_json_bytes(names)).hexdigest(),
        "values_sha256": hashlib.sha256(_strict_json_bytes(values)).hexdigest(),
    }


def _target_row(
    diagnostic,
    target,
    *,
    compiled_target: float,
    spec: object | None,
) -> dict[str, object]:
    """A diagnostics target row with both fit metrics and structured identity."""
    row: dict[str, object] = {
        "name": diagnostic.name,
        "target_name": target.name,
        "period": target.period,
        "entity": target.entity,
        "measure": _selector_payload(target.measure),
        "filter": _selector_payload(target.filter),
        "source": target.source,
        "metadata": dict(target.metadata),
        "target": _finite(diagnostic.target),
        "compiled_target": _finite(compiled_target),
        "initial_estimate": _finite(diagnostic.initial_estimate),
        "final_estimate": _finite(diagnostic.final_estimate),
        "relative_error": _finite(diagnostic.relative_error),
        "within_tolerance": diagnostic.within_tolerance,
    }
    if spec is not None:
        row["registry"] = {
            "family": getattr(spec, "family", ""),
            "se": _finite(getattr(spec, "se", None))
            if getattr(spec, "se", None) is not None
            else None,
            "signed": bool(getattr(spec, "signed", False)),
            "notes": getattr(spec, "notes", ""),
        }
    return row


def _result_target_loss_cap(options: object) -> float | None:
    """The per-row loss cap recorded in a result's options, if any.

    :func:`~microcosm.calibrate.solve.calibrate` records the cap inside its
    ``target_loss_scales`` options summary;
    :func:`~microcosm.calibrate.score.score_targets` records a top-level
    ``target_loss_cap``. Accept both shapes so every result that knows its
    cap can be censused.
    """
    if not isinstance(options, Mapping):
        return None
    scales = options.get("target_loss_scales")
    if isinstance(scales, Mapping):
        cap = scales.get("cap")
        if isinstance(cap, (int, float)) and math.isfinite(float(cap)):
            return float(cap)
    cap = options.get("target_loss_cap")
    if isinstance(cap, (int, float)) and math.isfinite(float(cap)):
        return float(cap)
    return None


def past_cap_census(result: CalibrationResult) -> dict[str, object] | None:
    """Census of target rows past the loss cap at the solve's start and end.

    Under the capped weighted-MAPE objective, a row whose scaled absolute
    miss ``abs(estimate - target) / max(abs(target), 1)`` reaches
    ``target_loss_cap`` contributes a constant to the loss: its gradient is
    zero, so the solver is neither rewarded for improving it nor charged for
    making it worse. Past-cap rows are therefore potential dumping grounds —
    mass moved to satisfy live targets can push an in-cap row past the cap
    and abandon it there at zero marginal cost. The census makes that triage
    first-class release evidence:

    - ``initial_past_cap`` / ``final_past_cap``: rows at or past the cap
      under the initial / final estimates.
    - ``escaped``: past at initialization, back inside at final (recovered
      via shared carriers despite carrying no gradient of their own).
    - ``frozen``: past at initialization and still past at final.
    - ``pushed_out``: inside at initialization, past the cap at final — the
      rows the solve wrote off. Each is listed in ``pushed_out_rows`` with
      its scaled misses, worst final miss first.

    ``init_rel`` / ``final_rel`` are scaled absolute misses on the default
    scale rule ``max(abs(target), 1)``. This block intentionally preserves its
    schema-version-5 semantics: a run that supplied custom scales is still
    censused on the default rule even though schema version 6 separately
    retains and reports its actual aligned loss scales. Rows with a non-finite
    target or estimate are excluded from every count. Returns ``None`` when
    the result's options record no cap — there is nothing to census against.
    """
    cap = _result_target_loss_cap(getattr(result, "options", None))
    if cap is None:
        return None
    initial_past = 0
    final_past = 0
    escaped = 0
    frozen = 0
    pushed_out_rows: list[dict[str, object]] = []
    for diagnostic in result.diagnostics:
        target = float(diagnostic.target)
        initial = float(diagnostic.initial_estimate)
        final = float(diagnostic.final_estimate)
        if not (
            math.isfinite(target) and math.isfinite(initial) and math.isfinite(final)
        ):
            continue
        scale = max(abs(target), 1.0)
        init_rel = abs(initial - target) / scale
        final_rel = abs(final - target) / scale
        # Strictly past the cap: torch.clamp keeps gradient AT the boundary
        # (verified: d/dx clamp(x, max=cap) at x == cap is 1), so a row
        # exactly at the cap still pulls. "Past cap" == zero gradient.
        init_past = init_rel > cap
        fin_past = final_rel > cap
        if init_past:
            initial_past += 1
        if fin_past:
            final_past += 1
        if init_past and not fin_past:
            escaped += 1
        elif init_past and fin_past:
            frozen += 1
        elif fin_past:
            pushed_out_rows.append(
                {
                    "name": diagnostic.name,
                    "init_rel": init_rel,
                    "final_rel": final_rel,
                }
            )
    pushed_out_rows.sort(key=lambda row: (-row["final_rel"], row["name"]))
    return {
        "cap": cap,
        "scale_basis": "max(abs(target), 1)",
        "n_targets": len(result.diagnostics),
        "initial_past_cap": initial_past,
        "final_past_cap": final_past,
        "escaped": escaped,
        "frozen": frozen,
        "pushed_out": len(pushed_out_rows),
        "pushed_out_rows": pushed_out_rows,
    }


def diagnostics_payload(
    result: CalibrationResult,
    *,
    target_registry: object | None = None,
    build: dict[str, Any] | None = None,
) -> dict:
    """Render a calibration result as a JSON-stable diagnostics payload.

    The payload carries the full evidence, not summaries: every per-target
    row, the whole loss trajectory, and every skipped target with its
    reason. Summary scalars (``final_loss``, ``fraction_within_10pct``) are
    included so a consumer need not recompute them, but they are derived
    from the rows, never a substitute.

    Args:
        result: The :func:`~microcosm.calibrate.solve.calibrate` output.

    Returns:
        A dict that round-trips through ``json`` unchanged (non-finite
        floats become ``null``).
    """
    registry_specs = _registry_spec_lookup(target_registry)
    registry_country = str(getattr(target_registry, "country", "")).strip()
    dimension_definitions: dict[str, dict[str, object]] = {}
    target_rows: list[dict[str, object]] = []
    for index, (diagnostic, target) in enumerate(
        zip(result.diagnostics, result.problem.targets, strict=True)
    ):
        spec = registry_specs.get(diagnostic.name)
        if target_registry is not None and spec is None:
            raise ValueError(
                "The supplied target registry does not contain compiled target "
                f"row {diagnostic.name!r}."
            )
        row = _target_row(
            diagnostic,
            target,
            compiled_target=result.problem.target_vector[index],
            spec=spec,
        )
        if spec is not None:
            structured_fields, row_definitions = _structured_target_fields(
                target,
                spec,
                country=registry_country,
            )
            row.update(structured_fields)
            _merge_dimension_definitions(dimension_definitions, row_definitions)
        target_rows.append(row)
    payload = {
        "schema_version": CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
        "weight_entity": result.weight_entity,
        "options": {key: _jsonable(value) for key, value in result.options.items()},
        "target_surface": _target_surface_payload(result),
        "l0_lambda": _finite(result.l0_lambda),
        "n_nonzero": int(result.n_nonzero),
        "n_records": int(result.weights.shape[0]),
        "initial_loss": _finite(result.initial_loss),
        "final_loss": _finite(result.final_loss),
        "fraction_within_10pct": _finite(result.fraction_within_10pct),
        "effective_sample_size": _finite(result.effective_sample_size),
        "realized_max_weight_ratio": _finite(result.realized_max_weight_ratio),
        "top_1pct_weight_share": _finite(result.top_1pct_weight_share),
        "loss_trajectory": [_finite(loss) for loss in result.loss_trajectory],
        "skipped": [
            {"name": skip.target.name, "reason": skip.reason} for skip in result.skipped
        ],
        "past_cap_census": past_cap_census(result),
        "diagnostic_warnings": [],
        "targets": target_rows,
    }
    if target_registry is not None:
        payload["dimensions"] = dimension_definitions
    try:
        attribution = assemble_target_loss_attribution(result)
    except TargetLossAttributionError as error:
        _LOGGER.warning(
            "TARGET LOSS ATTRIBUTION UNAVAILABLE [%s]: %s",
            error.code,
            error,
        )
        payload["diagnostic_warnings"].append(
            {
                "code": error.code,
                "severity": "warning",
                "message": str(error),
            }
        )
    except Exception as error:  # pragma: no cover - defensive build protection
        code = TARGET_LOSS_ATTRIBUTION_WARNING_CODES["assembly_error"]
        _LOGGER.exception(
            "TARGET LOSS ATTRIBUTION UNAVAILABLE [%s]: unexpected attribution "
            "assembly error",
            code,
        )
        payload["diagnostic_warnings"].append(
            {
                "code": code,
                "severity": "warning",
                "message": (
                    "Unexpected target-loss attribution assembly error: "
                    f"{type(error).__name__}: {error}"
                ),
            }
        )
    else:
        payload["target_loss_basis"] = attribution.basis
        for row, attribution_row in zip(target_rows, attribution.rows, strict=True):
            row.update(attribution_row)
    registry = _registry_payload(target_registry)
    if registry is not None:
        payload["target_registry"] = registry
    if build is not None:
        payload["build"] = build
    return payload


def write_calibration_diagnostics(
    result: CalibrationResult,
    path: Path | str,
    *,
    target_registry: object | None = None,
    build: dict[str, Any] | None = None,
) -> Path:
    """Write the diagnostics payload to ``path`` as JSON.

    The conventional filename is ``calibration_diagnostics.json`` inside a
    release directory, alongside ``build_manifest.json``.

    Args:
        result: The :func:`~microcosm.calibrate.solve.calibrate` output.
        path: Destination file path; parent directories must exist.

    Returns:
        The path written.
    """
    path = Path(path)
    # allow_nan=False is the guard: a non-finite value that escaped the
    # scrub is a bug here, not something to smuggle out as invalid JSON.
    path.write_text(
        json.dumps(
            diagnostics_payload(
                result,
                target_registry=target_registry,
                build=build,
            ),
            indent=1,
            allow_nan=False,
        )
    )
    return path
