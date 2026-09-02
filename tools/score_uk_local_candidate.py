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

from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_TARGET_LOSS_CAP
from microcosm.calibrate import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    TargetRegistry,
    default_target_loss_scales,
    relative_error_loss,
)

UK_LOCAL_ACTIVE_REFERENCE_COUNT = 19_105
UK_LOCAL_SCORE_TARGET_PERIOD = 2025
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
) -> tuple[dict[str, float], dict[str, object]]:
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
    missing = sorted(expected - set(estimates))
    if missing:
        raise ValueError(
            "candidate diagnostics must cover every frozen local register row; "
            f"missing={missing[:10]}."
        )
    extra = sorted(set(estimates) - expected)
    return (
        {name: estimates[name] for name in expected},
        {"count": len(extra), "rows": extra},
    )


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
    # The holdout is a number recorded upstream, under whatever cap that run
    # used; the fitted-surface aggregates are computed here under the current
    # doctrine cap.  Requiring the recorded cap to match makes the agreement
    # an enforced invariant instead of a convention that quietly lapses the
    # first time microcosm#762 moves the constant — at which point the
    # candidate needs re-measuring, not re-reporting.
    declared_cap = holdout.get("target_loss_cap")
    if (
        not isinstance(declared_cap, (int, float))
        or isinstance(declared_cap, bool)
        or not math.isfinite(float(declared_cap))
    ):
        raise ValueError(
            "candidate rotated holdout must declare the target_loss_cap it was "
            "measured under."
        )
    if float(declared_cap) != float(UK_LOCAL_TARGET_LOSS_CAP):
        raise ValueError(
            "candidate rotated holdout was measured at target_loss_cap "
            f"{float(declared_cap)!r}, but this scorer reports its aggregates "
            f"at {float(UK_LOCAL_TARGET_LOSS_CAP)!r}; re-measure the candidate "
            "rather than reporting the two on different scales."
        )
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
    folds: list[float] = []
    for value in fold_losses:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "candidate rotated holdout has a non-finite or negative fold loss."
            )
        folds.append(float(value))
    # The summary is derived from these folds by `summarize_rotations`, so it
    # must still close over them.  Without this, a headline number lifted
    # from somewhere else — a fitted loss, say — passes every other check
    # with plausible folds sitting beside it, which is the substitution the
    # required-holdout refusal exists to catch.
    if not math.isclose(
        losses["mean_holdout_loss"],
        math.fsum(folds) / n_folds,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "candidate rotated holdout mean does not close over its fold losses."
        )
    if not math.isclose(
        losses["worst_holdout_loss"],
        max(folds),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate rotated holdout worst loss is not its worst fold.")
    return {
        "basis": f"{method}:n_folds={n_folds}:seed={seed}",
        "method": method,
        "target_loss_cap": float(declared_cap),
        "n_folds": n_folds,
        "seed": seed,
        "fold_losses": folds,
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


def _relative_errors(estimates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Signed, uncapped per-row misses on the canonical row scale.

    A different quantity from the aggregate objective — signed and uncapped,
    for the drift rows and the head-to-head counters — but deliberately not a
    different *scale*: the denominator is imported from
    :func:`default_target_loss_scales` rather than restated, so these rows
    cannot drift away from the aggregates printed beside them.
    """

    return (estimates - targets) / default_target_loss_scales(targets)


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
    candidate, rows_outside_register = _candidate_estimates(
        candidate_diagnostics,
        target_registry,
    )
    incumbent = _incumbent_estimates(
        target_registry,
        incumbent_weights,
        incumbent_metrics,
    )

    families: dict[str, dict[str, int]] = {}
    drift: list[dict[str, object]] = []
    targets = np.array([spec.value for spec in target_registry.specs], dtype=np.float64)
    candidate_estimates = np.array(
        [candidate[spec.to_target().row_name] for spec in target_registry.specs],
        dtype=np.float64,
    )
    incumbent_estimates = np.array(
        [incumbent[spec.to_target().row_name] for spec in target_registry.specs],
        dtype=np.float64,
    )
    candidate_errors = _relative_errors(candidate_estimates, targets)
    incumbent_errors = _relative_errors(incumbent_estimates, targets)
    candidate_wins = 0
    incumbent_wins = 0
    for index, spec in enumerate(target_registry.specs):
        name = spec.to_target().row_name
        candidate_error = float(candidate_errors[index])
        incumbent_error = float(incumbent_errors[index])
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
    # Both aggregates, and the holdout the candidate driver measured, go
    # through the one canonical objective at the one declared doctrine cap,
    # so the numbers in this receipt are on a single scale and stay there
    # when microcosm#762 adjudicates the cap.
    candidate_loss = relative_error_loss(
        candidate_estimates,
        targets,
        target_loss_cap=UK_LOCAL_TARGET_LOSS_CAP,
    )
    incumbent_loss = relative_error_loss(
        incumbent_estimates,
        targets,
        target_loss_cap=UK_LOCAL_TARGET_LOSS_CAP,
    )
    return {
        "candidate_fitted_surface_loss": candidate_loss,
        "candidate_holdout_loss": holdout["mean_holdout_loss"],
        "incumbent_fitted_surface_loss": incumbent_loss,
        "incumbent_holdout_loss": None,
        "candidate_target_wins": candidate_wins,
        "incumbent_target_wins": incumbent_wins,
        "rows_outside_register": rows_outside_register,
        "holdout_basis": holdout["basis"],
        "incumbent_holdout_basis": UK_LOCAL_INCUMBENT_HOLDOUT_BASIS,
        "candidate_holdout": holdout,
        "loss": {
            # Names the function that actually produced every loss above,
            # including the candidate's holdout, at its declared cap.
            "objective": "microcosm.calibrate.relative_error_loss",
            "target_loss_cap": UK_LOCAL_TARGET_LOSS_CAP,
            "target_loss_cap_source": "UK_LOCAL_TARGET_LOSS_CAP",
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
