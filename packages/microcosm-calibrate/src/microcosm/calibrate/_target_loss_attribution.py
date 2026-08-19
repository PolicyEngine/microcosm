"""Build and validate schema-version-6 final target-loss attribution.

This module owns the loss-specific contract: aligned basis validation,
per-target contribution calculation, and deterministic basis hashing.
``diagnostics`` remains responsible for embedding a complete result in the
artifact or translating an attribution-only failure into a structured warning.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from microcosm.calibrate.solve import CalibrationResult

__all__ = [
    "TARGET_LOSS_ATTRIBUTION_ABS_TOLERANCE",
    "TARGET_LOSS_ATTRIBUTION_REL_TOLERANCE",
    "TARGET_LOSS_ATTRIBUTION_WARNING_CODES",
    "TARGET_LOSS_BASIS_HASH_ALGORITHM",
    "TargetLossAttribution",
    "TargetLossAttributionError",
    "assemble_target_loss_attribution",
    "target_loss_basis_hash",
]

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


@dataclass(frozen=True)
class TargetLossAttribution:
    """A complete attribution block ready to attach to diagnostics."""

    basis: dict[str, object]
    rows: tuple[dict[str, float], ...]


class TargetLossAttributionError(ValueError):
    """A validation failure that degrades only supplementary attribution."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def target_loss_basis_hash(
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


def assemble_target_loss_attribution(
    result: CalibrationResult,
) -> TargetLossAttribution:
    """Validate and assemble a complete final target-loss attribution block."""
    diagnostics = tuple(result.diagnostics)
    target_count = len(diagnostics)
    weights = np.asarray(result.target_loss_weights, dtype=np.float64)
    scales = np.asarray(result.target_loss_scales, dtype=np.float64)
    expected_shape = (target_count,)
    if weights.shape != expected_shape or scales.shape != expected_shape:
        raise TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["alignment"],
            "Final target-loss weights and scales must each have one value per "
            f"target diagnostic; got weights {weights.shape}, scales {scales.shape}, "
            f"and {target_count} target rows.",
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "Final target-loss weights must be finite and non-negative.",
        )
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        raise TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "Final target-loss scales must be finite and strictly positive.",
        )
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise TargetLossAttributionError(
            TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
            "Final target-loss weights must have positive finite total weight.",
        )
    cap = float(result.target_loss_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise TargetLossAttributionError(
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
            raise TargetLossAttributionError(
                TARGET_LOSS_ATTRIBUTION_WARNING_CODES["invalid_basis"],
                f"Target {diagnostic.name!r} has a non-finite target or final estimate.",
            )
        capped_error = min(abs(estimate - target) / float(scale), cap)
        contribution = float(weight_share) * capped_error
        if not math.isfinite(capped_error) or not math.isfinite(contribution):
            raise TargetLossAttributionError(
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
        raise TargetLossAttributionError(
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
        "sha256": target_loss_basis_hash(names, weights, scales),
    }
    return TargetLossAttribution(basis=basis, rows=tuple(attribution_rows))
