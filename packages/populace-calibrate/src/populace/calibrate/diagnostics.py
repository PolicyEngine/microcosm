"""Serialize a calibration's diagnostics so they travel with the artifact.

A :class:`~populace.calibrate.solve.CalibrationResult` carries everything a
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
import math
from pathlib import Path
from typing import Any

from populace.calibrate.solve import CalibrationResult

__all__ = [
    "CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION",
    "diagnostics_payload",
    "write_calibration_diagnostics",
]

#: Version of the diagnostics payload. Consumers (dashboards, scorers) key
#: their readers on it; bump it with any shape change.
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 2


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
                "aggregation": target.aggregation,
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
        "aggregation": target.aggregation,
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
        result: The :func:`~populace.calibrate.solve.calibrate` output.

    Returns:
        A dict that round-trips through ``json`` unchanged (non-finite
        floats become ``null``).
    """
    registry_specs = _registry_spec_lookup(target_registry)
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
        "loss_trajectory": [_finite(loss) for loss in result.loss_trajectory],
        "skipped": [
            {"name": skip.target.name, "reason": skip.reason} for skip in result.skipped
        ],
        "targets": [
            _target_row(
                diagnostic,
                target,
                compiled_target=result.problem.target_vector[index],
                spec=registry_specs.get(diagnostic.name),
            )
            for index, (diagnostic, target) in enumerate(
                zip(result.diagnostics, result.problem.targets, strict=True)
            )
        ],
    }
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
        result: The :func:`~populace.calibrate.solve.calibrate` output.
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
