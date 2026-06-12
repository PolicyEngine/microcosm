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

import json
import math
from pathlib import Path

from populace.calibrate.solve import CalibrationResult

__all__ = [
    "CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION",
    "diagnostics_payload",
    "write_calibration_diagnostics",
]

#: Version of the diagnostics payload. Consumers (dashboards, scorers) key
#: their readers on it; bump it with any shape change.
CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION = 1


def _finite(value: float) -> float | None:
    """JSON has no NaN/inf; a non-finite diagnostic serializes as null."""
    value = float(value)
    return value if math.isfinite(value) else None


def _jsonable(value: object) -> object:
    """An option value as strict JSON: non-finite floats become null."""
    if isinstance(value, float):
        return _finite(value)
    return value


def diagnostics_payload(result: CalibrationResult) -> dict:
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
    return {
        "schema_version": CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
        "weight_entity": result.weight_entity,
        "options": {key: _jsonable(value) for key, value in result.options.items()},
        "l0_lambda": _finite(result.l0_lambda),
        "n_nonzero": int(result.n_nonzero),
        "n_records": int(result.weights.shape[0]),
        "initial_loss": _finite(result.initial_loss),
        "final_loss": _finite(result.final_loss),
        "fraction_within_10pct": _finite(result.fraction_within_10pct),
        "loss_trajectory": [_finite(loss) for loss in result.loss_trajectory],
        "skipped": [
            {"name": skip.target.name, "reason": skip.reason}
            for skip in result.skipped
        ],
        "targets": [
            {
                "name": diagnostic.name,
                "target": _finite(diagnostic.target),
                "initial_estimate": _finite(diagnostic.initial_estimate),
                "final_estimate": _finite(diagnostic.final_estimate),
                "relative_error": _finite(diagnostic.relative_error),
                "within_tolerance": diagnostic.within_tolerance,
            }
            for diagnostic in result.diagnostics
        ],
    }


def write_calibration_diagnostics(
    result: CalibrationResult, path: Path | str
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
        json.dumps(diagnostics_payload(result), indent=1, allow_nan=False)
    )
    return path
