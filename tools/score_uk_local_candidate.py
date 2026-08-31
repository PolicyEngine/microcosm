"""Score a UK local candidate against incumbent wide-format area weights.

The candidate side is read from its schema-v6 calibration diagnostics.  The
incumbent side is deliberately explicit: a household-grain metric table and a
wide weight table with one column per local area.  Both are evaluated on the
same frozen UK TargetRegistry; no fitted row is allowed to disappear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.calibrate import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    TargetRegistry,
)

UK_LOCAL_ACTIVE_REFERENCE_COUNT = 17_077
UK_LOCAL_SCORE_TARGET_PERIOD = 2025
UK_LOCAL_SCORE_LOSS_CAP = 10.0
UK_LOCAL_SCORE_HOLDOUT_BASIS = "none_declared"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(path: str | Path, expected_sha256: str) -> dict[str, object]:
    artifact = Path(path)
    measured = _sha256_file(artifact)
    if measured != expected_sha256:
        raise ValueError(
            f"artifact sha mismatch for {artifact}: measured {measured}, "
            f"pinned {expected_sha256}"
        )
    return {
        "path": str(artifact),
        "sha256": measured,
        "size_bytes": artifact.stat().st_size,
    }


def _candidate_estimates(
    diagnostics: Mapping[str, object],
    registry: TargetRegistry,
) -> dict[str, float]:
    if diagnostics.get("schema_version") != CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("UK local scoring requires schema-v6 candidate diagnostics.")
    rows = diagnostics.get("targets")
    if not isinstance(rows, list):
        raise ValueError("candidate diagnostics must contain target rows.")
    estimates: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("candidate diagnostics contain a malformed target row.")
        name = str(row.get("name") or "")
        raw = row.get("final_estimate")
        if (
            not name
            or not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or name in estimates
        ):
            raise ValueError("candidate diagnostics contain invalid target estimates.")
        estimates[name] = float(raw)
    expected = {spec.to_target().row_name for spec in registry.specs}
    if set(estimates) != expected:
        missing = sorted(expected - set(estimates))
        extra = sorted(set(estimates) - expected)
        raise ValueError(
            "candidate diagnostics must exactly cover the frozen local register; "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return estimates


def _incumbent_estimates(
    registry: TargetRegistry,
    incumbent_weights: pd.DataFrame,
    incumbent_metrics: pd.DataFrame,
) -> dict[str, float]:
    if len(incumbent_weights) != len(incumbent_metrics):
        raise ValueError(
            "incumbent weights and household metrics must have the same row count."
        )
    estimates: dict[str, float] = {}
    for spec in registry.specs:
        area = str(spec.metadata.get("ledger_geography_id") or "")
        if not area:
            raise ValueError(
                f"local target {spec.to_target().row_name!r} has no "
                "ledger_geography_id metadata."
            )
        if area not in incumbent_weights.columns:
            raise ValueError(f"incumbent wide weights are missing area {area!r}.")
        if spec.measure not in incumbent_metrics.columns:
            raise ValueError(
                f"incumbent household metrics are missing measure {spec.measure!r}."
            )
        weights = incumbent_weights[area].to_numpy(dtype=np.float64)
        metric = incumbent_metrics[spec.measure].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(weights).all()
            or (weights < 0.0).any()
            or not np.isfinite(metric).all()
        ):
            raise ValueError(
                f"incumbent inputs for {spec.to_target().row_name!r} are invalid."
            )
        estimates[spec.to_target().row_name] = float(np.dot(weights, metric))
    return estimates


def _relative_error(estimate: float, target: float) -> float:
    return (estimate - target) / max(abs(target), 1.0)


def score_uk_local_candidate(
    *,
    candidate_diagnostics: Mapping[str, object],
    incumbent_weights: pd.DataFrame,
    incumbent_metrics: pd.DataFrame,
    target_registry: TargetRegistry,
    expected_reference_count: int = UK_LOCAL_ACTIVE_REFERENCE_COUNT,
    target_period: int = UK_LOCAL_SCORE_TARGET_PERIOD,
) -> dict[str, Any]:
    """Score both sides on the exact frozen local target surface."""

    if target_registry.country != "uk":
        raise ValueError("UK local scoring requires a UK TargetRegistry.")
    if len(target_registry) != expected_reference_count:
        raise ValueError(
            "UK local scoring requires the frozen active reference count "
            f"{expected_reference_count}, got {len(target_registry)}."
        )
    wrong_period = [
        spec.to_target().row_name
        for spec in target_registry.specs
        if spec.period != target_period
    ]
    if wrong_period:
        raise ValueError(
            f"UK local scoring requires target period {target_period}; "
            f"mismatches={wrong_period[:10]}."
        )
    candidate = _candidate_estimates(candidate_diagnostics, target_registry)
    incumbent = _incumbent_estimates(
        target_registry,
        incumbent_weights,
        incumbent_metrics,
    )

    families: dict[str, dict[str, int]] = {}
    drift: list[dict[str, object]] = []
    candidate_losses: list[float] = []
    incumbent_losses: list[float] = []
    candidate_wins = 0
    incumbent_wins = 0
    for spec in target_registry.specs:
        name = spec.to_target().row_name
        candidate_error = _relative_error(candidate[name], spec.value)
        incumbent_error = _relative_error(incumbent[name], spec.value)
        candidate_losses.append(min(abs(candidate_error), UK_LOCAL_SCORE_LOSS_CAP))
        incumbent_losses.append(min(abs(incumbent_error), UK_LOCAL_SCORE_LOSS_CAP))
        bucket = families.setdefault(
            spec.family,
            {"candidate_target_wins": 0, "incumbent_target_wins": 0, "ties": 0},
        )
        if abs(candidate_error) < abs(incumbent_error):
            winner = "candidate"
            candidate_wins += 1
            bucket["candidate_target_wins"] += 1
        elif abs(incumbent_error) < abs(candidate_error):
            winner = "incumbent"
            incumbent_wins += 1
            bucket["incumbent_target_wins"] += 1
        else:
            winner = "tie"
            bucket["ties"] += 1
        drift.append(
            {
                "target": name,
                "family": spec.family,
                "candidate_relative_error": candidate_error,
                "incumbent_relative_error": incumbent_error,
                "winner": winner,
            }
        )
    candidate_loss = float(np.mean(candidate_losses))
    incumbent_loss = float(np.mean(incumbent_losses))
    return {
        "candidate_train_loss": candidate_loss,
        "candidate_holdout_loss": None,
        "candidate_full_loss": candidate_loss,
        "incumbent_train_loss": incumbent_loss,
        "incumbent_holdout_loss": None,
        "incumbent_full_loss": incumbent_loss,
        "candidate_target_wins": candidate_wins,
        "incumbent_target_wins": incumbent_wins,
        "holdout_basis": UK_LOCAL_SCORE_HOLDOUT_BASIS,
        "loss": {
            "objective": "relative_error_loss",
            "target_loss_cap": UK_LOCAL_SCORE_LOSS_CAP,
            "train_equals_full": True,
        },
        "register": {
            "country": target_registry.country,
            "version": target_registry.version,
            "n_specs": len(target_registry),
            "target_period": target_period,
            "surface": "active_local_reference",
        },
        "target_wins_by_family": families,
        "target_drift": drift,
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-diagnostics-json", required=True, type=Path)
    parser.add_argument("--candidate-diagnostics-sha256", required=True)
    parser.add_argument("--incumbent-weights-csv", required=True, type=Path)
    parser.add_argument("--incumbent-weights-sha256", required=True)
    parser.add_argument("--incumbent-household-metrics-csv", required=True, type=Path)
    parser.add_argument("--incumbent-household-metrics-sha256", required=True)
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    artifacts = {
        "candidate_diagnostics": _verify_artifact(
            args.candidate_diagnostics_json,
            args.candidate_diagnostics_sha256,
        ),
        "incumbent_wide_weights": _verify_artifact(
            args.incumbent_weights_csv,
            args.incumbent_weights_sha256,
        ),
        "incumbent_household_metrics": _verify_artifact(
            args.incumbent_household_metrics_csv,
            args.incumbent_household_metrics_sha256,
        ),
        "target_registry": _verify_artifact(
            args.registry_json,
            args.registry_sha256,
        ),
    }
    score = score_uk_local_candidate(
        candidate_diagnostics=_read_json(args.candidate_diagnostics_json),
        incumbent_weights=pd.read_csv(args.incumbent_weights_csv),
        incumbent_metrics=pd.read_csv(args.incumbent_household_metrics_csv),
        target_registry=TargetRegistry.from_json(args.registry_json),
    )
    score["artifacts"] = artifacts
    args.output_json.write_text(
        json.dumps(score, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
