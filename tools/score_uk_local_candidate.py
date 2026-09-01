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
#: The incumbent is scored from published weights, never re-solved, so no
#: incumbent holdout exists to place beside the candidate's rotation.
UK_LOCAL_INCUMBENT_HOLDOUT_BASIS = "none_available_incumbent_not_resolved"
_HOUSEHOLD_ID_COLUMN = "household_id"


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


def _candidate_holdout(diagnostics: Mapping[str, object]) -> dict[str, object]:
    """Read the candidate's measured rotated holdout out of its diagnostics.

    The candidate driver runs the rotation and publishes it in the same
    schema-v6 payload this scorer already reads, so a receipt that reported
    ``none_declared`` beside it would be understating what was measured.  The
    block is required: a candidate whose diagnostics carry no rotation is
    refused rather than scored on its fitted surface alone.
    """

    uk_diagnostics = diagnostics.get("uk_diagnostics")
    if not isinstance(uk_diagnostics, Mapping):
        raise ValueError("candidate diagnostics must carry a uk_diagnostics block.")
    holdout = uk_diagnostics.get("rotated_holdout")
    if not isinstance(holdout, Mapping):
        raise ValueError(
            "candidate diagnostics must carry uk_diagnostics.rotated_holdout; "
            "scoring a candidate with no measured holdout is refused."
        )
    method = str(holdout.get("method") or "")
    n_folds = holdout.get("n_folds")
    seed = holdout.get("seed")
    if (
        not method
        or isinstance(n_folds, bool)
        or not isinstance(n_folds, int)
        or n_folds < 2
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("candidate rotated holdout declares no usable basis.")
    losses: dict[str, float] = {}
    for key in ("mean_holdout_loss", "worst_holdout_loss"):
        raw = holdout.get(key)
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            raise ValueError(f"candidate rotated holdout has an invalid {key}.")
        losses[key] = float(raw)
    fold_losses = holdout.get("fold_losses")
    if not isinstance(fold_losses, list) or len(fold_losses) != n_folds:
        raise ValueError(
            "candidate rotated holdout must report one loss per declared fold."
        )
    return {
        "basis": f"{method}:n_folds={n_folds}:seed={seed}",
        "method": method,
        "n_folds": n_folds,
        "seed": seed,
        "fold_losses": [float(value) for value in fold_losses],
        **losses,
    }


def _align_on_household_id(
    incumbent_weights: pd.DataFrame,
    incumbent_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the two incumbent tables on ``household_id``.

    A row-count check passes for any permutation of the right size, so pairing
    these tables by position scores the incumbent from mismatched households
    and reports it as a win or a loss rather than as an error
    (``uk-data#468``).  The join is the check: it must be total and unique on
    both sides, and both tables are returned on one household order.
    """

    aligned: list[pd.DataFrame] = []
    for label, frame in (
        ("incumbent wide weights", incumbent_weights),
        ("incumbent household metrics", incumbent_metrics),
    ):
        if _HOUSEHOLD_ID_COLUMN not in frame.columns:
            raise ValueError(
                f"{label} must carry a {_HOUSEHOLD_ID_COLUMN!r} column; "
                "positional pairing is not a join."
            )
        keys = frame[_HOUSEHOLD_ID_COLUMN]
        if keys.isna().any():
            raise ValueError(f"{label} has missing {_HOUSEHOLD_ID_COLUMN} values.")
        indexed = frame.set_index(_HOUSEHOLD_ID_COLUMN)
        if not indexed.index.is_unique:
            duplicates = sorted(
                {str(key) for key in indexed.index[indexed.index.duplicated()]}
            )
            raise ValueError(
                f"{label} repeats {_HOUSEHOLD_ID_COLUMN!r} value(s): {duplicates[:10]}."
            )
        aligned.append(indexed)

    weights, metrics = aligned
    if set(weights.index) != set(metrics.index):
        missing = sorted({str(key) for key in set(weights.index) - set(metrics.index)})
        extra = sorted({str(key) for key in set(metrics.index) - set(weights.index)})
        raise ValueError(
            "incumbent weights and household metrics must cover the same "
            f"households; missing_from_metrics={missing[:10]}, "
            f"missing_from_weights={extra[:10]}."
        )
    order = weights.index
    return weights, metrics.reindex(order)


def _incumbent_estimates(
    registry: TargetRegistry,
    incumbent_weights: pd.DataFrame,
    incumbent_metrics: pd.DataFrame,
) -> dict[str, float]:
    incumbent_weights, incumbent_metrics = _align_on_household_id(
        incumbent_weights,
        incumbent_metrics,
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
    holdout = _candidate_holdout(candidate_diagnostics)
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
        "candidate_fitted_surface_loss": candidate_loss,
        "candidate_holdout_loss": holdout["mean_holdout_loss"],
        "incumbent_fitted_surface_loss": incumbent_loss,
        "incumbent_holdout_loss": None,
        "candidate_target_wins": candidate_wins,
        "incumbent_target_wins": incumbent_wins,
        "holdout_basis": holdout["basis"],
        "incumbent_holdout_basis": UK_LOCAL_INCUMBENT_HOLDOUT_BASIS,
        "candidate_holdout": holdout,
        "loss": {
            "objective": "relative_error_loss",
            "target_loss_cap": UK_LOCAL_SCORE_LOSS_CAP,
            # The head-to-head counters below compare both sides on the
            # surface the candidate was fitted to; only the candidate has a
            # held-out measurement, and it is reported beside them, never as
            # part of them.
            "head_to_head_surface": "candidate_fitted_surface",
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
