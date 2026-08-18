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
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.calibrate.solve import CalibrationResult

__all__ = [
    "CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION",
    "TARGET_LOSS_ATTRIBUTION_ABS_TOLERANCE",
    "TARGET_LOSS_ATTRIBUTION_REL_TOLERANCE",
    "TARGET_LOSS_ATTRIBUTION_WARNING_CODES",
    "TARGET_LOSS_BASIS_HASH_ALGORITHM",
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
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 6

#: Cross-language validation constants for schema-version-6 attribution.
TARGET_LOSS_ATTRIBUTION_ABS_TOLERANCE = 1e-12
TARGET_LOSS_ATTRIBUTION_REL_TOLERANCE = 1e-12
TARGET_LOSS_BASIS_HASH_ALGORITHM = "sha256_utf8len32_f64be_v1"
TARGET_LOSS_ATTRIBUTION_WARNING_CODES = {
    "alignment": "target_loss_attribution_alignment_error",
    "invalid_basis": "target_loss_attribution_invalid_basis",
    "contribution_mismatch": "target_loss_attribution_contribution_mismatch",
    "assembly_error": "target_loss_attribution_assembly_error",
}

_TARGET_LOSS_FORMULA = "weighted_mean(min(abs((estimate - target) / scale), cap))"
_LOGGER = logging.getLogger(__name__)


class _TargetLossAttributionError(ValueError):
    """A validation failure that degrades only supplementary attribution."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def _loss_basis_kind(
    result: CalibrationResult,
    option_name: str,
    *,
    default: str,
) -> str:
    """Read a result's recorded loss-basis kind without rebuilding its values."""
    options = getattr(result, "options", None)
    if not isinstance(options, Mapping):
        return default
    value = options.get(option_name)
    if isinstance(value, Mapping):
        kind = value.get("kind")
        return str(kind) if isinstance(kind, str) and kind else default
    if isinstance(value, str) and value:
        return value
    return default


def _target_loss_basis_hash(
    names: list[str],
    weights: np.ndarray,
    scales: np.ndarray,
) -> str:
    """Hash ordered UTF-8 names and IEEE-754 values without JSON float ambiguity.

    Each name is encoded as a four-byte unsigned big-endian byte length followed
    by its UTF-8 bytes, then its raw weight and scale as big-endian float64. The
    versioned algorithm identifier travels with the digest so non-Python
    consumers can reproduce it exactly.
    """
    digest = hashlib.sha256()
    digest.update(b"microcosm-target-loss-basis-v1\x00")
    for name, weight, scale in zip(names, weights, scales, strict=True):
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">d", float(weight)))
        digest.update(struct.pack(">d", float(scale)))
    return digest.hexdigest()


def _target_loss_attribution(
    result: CalibrationResult,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    """Validate and assemble the complete final target-loss attribution block."""
    diagnostics = tuple(result.diagnostics)
    target_count = len(diagnostics)
    weights = np.asarray(result.target_loss_weights, dtype=np.float64)
    scales = np.asarray(result.target_loss_scales, dtype=np.float64)
    expected_shape = (target_count,)
    if weights.shape != expected_shape or scales.shape != expected_shape:
        raise _TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["alignment"],
            "Final target-loss weights and scales must each have one value per "
            f"target diagnostic; got weights {weights.shape}, scales {scales.shape}, "
            f"and {target_count} target rows.",
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise _TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "Final target-loss weights must be finite and non-negative.",
        )
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        raise _TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "Final target-loss scales must be finite and strictly positive.",
        )
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise _TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "Final target-loss weights must have positive finite total weight.",
        )
    cap = float(result.target_loss_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise _TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "The final target-loss cap must be finite and strictly positive.",
        )

    weight_shares = weights / total_weight
    attribution_rows: list[dict[str, float]] = []
    names: list[str] = []
    for diagnostic, weight, weight_share, scale in zip(
        diagnostics,
        weights,
        weight_shares,
        scales,
        strict=True,
    ):
        target = float(diagnostic.target)
        estimate = float(diagnostic.final_estimate)
        if not math.isfinite(target) or not math.isfinite(estimate):
            raise _TargetLossAttributionError(
                TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
                f"Target {diagnostic.name!r} has a non-finite target or final estimate.",
            )
        capped_error = min(abs(estimate - target) / float(scale), cap)
        contribution = float(weight_share) * capped_error
        if not math.isfinite(capped_error) or not math.isfinite(contribution):
            raise _TargetLossAttributionError(
                TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
                f"Target {diagnostic.name!r} produced non-finite attribution values.",
            )
        names.append(str(diagnostic.name))
        attribution_rows.append(
            {
                "target_loss_weight": float(weight),
                "target_loss_weight_share": float(weight_share),
                "target_loss_scale": float(scale),
                "final_capped_scaled_error": float(capped_error),
                "final_loss_contribution": float(contribution),
            }
        )

    contribution_sum = float(
        math.fsum(row["final_loss_contribution"] for row in attribution_rows)
    )
    final_loss = float(result.final_loss)
    if not math.isfinite(final_loss) or not math.isclose(
        contribution_sum,
        final_loss,
        rel_tol=TARGET_LOSS_ATTRIBUTION_REL_TOLERANCE,
        abs_tol=TARGET_LOSS_ATTRIBUTION_ABS_TOLERANCE,
    ):
        raise _TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["contribution_mismatch"],
            "Final target-loss contributions do not reproduce final_loss within "
            "the schema-version-6 tolerance: "
            f"contributions={contribution_sum!r}, final_loss={final_loss!r}.",
        )

    basis = {
        "formula": _TARGET_LOSS_FORMULA,
        "cap": cap,
        "target_count": target_count,
        "total_target_weight": total_weight,
        "weight_kind": _loss_basis_kind(
            result,
            "target_loss_weights",
            default="unknown",
        ),
        "scale_kind": _loss_basis_kind(
            result,
            "target_loss_scales",
            default="unknown",
        ),
        "hash_algorithm": TARGET_LOSS_BASIS_HASH_ALGORITHM,
        "sha256": _target_loss_basis_hash(names, weights, scales),
    }
    return basis, attribution_rows


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
    scale rule ``max(abs(target), 1)`` — the units the cap applies to. A run
    that supplied custom ``target_loss_scales`` is censused on the default
    rule (the custom scales do not travel with the result); its options
    record that the scales were provided. Rows with a non-finite target or
    estimate are excluded from every count. Returns ``None`` when the
    result's options record no cap — there is nothing to census against.
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
    target_rows = [
        _target_row(
            diagnostic,
            target,
            compiled_target=result.problem.target_vector[index],
            spec=registry_specs.get(diagnostic.name),
        )
        for index, (diagnostic, target) in enumerate(
            zip(result.diagnostics, result.problem.targets, strict=True)
        )
    ]
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
    try:
        target_loss_basis, attribution_rows = _target_loss_attribution(result)
    except _TargetLossAttributionError as error:
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
        payload["target_loss_basis"] = target_loss_basis
        for row, attribution in zip(target_rows, attribution_rows, strict=True):
            row.update(attribution)
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
