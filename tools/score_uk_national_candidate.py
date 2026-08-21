"""Score a UK national candidate against the incumbent on one frozen register."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.national_build import load_uk_national_frame
from microcosm.calibrate import TargetRegistry, TargetSpec, score_targets

UK_SCORE_LOSS_CAP = 10.0


def score_uk_national_candidate(
    *,
    candidate_h5: str | Path,
    incumbent_h5: str | Path,
    target_registry: TargetRegistry,
    candidate_label: str = "populace_uk_2023",
    incumbent_label: str = "enhanced_frs_2024_25",
) -> dict[str, Any]:
    """Return the #578 rule-1 score block on a shared target registry."""

    if target_registry.country != "uk" or not target_registry.specs:
        raise ValueError("UK candidate scoring requires a non-empty UK registry.")
    candidate_frame, _candidate_provenance = load_uk_national_frame(candidate_h5)
    incumbent_frame, _incumbent_provenance = load_uk_national_frame(incumbent_h5)
    candidate = score_targets(
        candidate_frame,
        target_registry.to_target_set(),
        target_loss_cap=UK_SCORE_LOSS_CAP,
    )
    incumbent = score_targets(
        incumbent_frame,
        target_registry.to_target_set(),
        target_loss_cap=UK_SCORE_LOSS_CAP,
    )
    candidate_errors = _relative_errors(candidate)
    incumbent_errors = _relative_errors(incumbent)
    names = sorted(candidate_errors)
    missing = sorted(set(names) ^ set(incumbent_errors))
    if missing:
        raise RuntimeError(
            "candidate and incumbent scores produced different target rows: "
            f"{missing[:10]}."
        )
    wins = _target_wins(candidate_errors, incumbent_errors)
    return {
        "candidate_train_loss": float(candidate.final_loss),
        "candidate_holdout_loss": float(candidate.final_loss),
        "candidate_full_loss": float(candidate.final_loss),
        "incumbent_train_loss": float(incumbent.final_loss),
        "incumbent_holdout_loss": float(incumbent.final_loss),
        "incumbent_full_loss": float(incumbent.final_loss),
        "candidate_target_wins": wins["candidate"],
        "incumbent_target_wins": wins["incumbent"],
        "holdout_basis": "none_declared",
        "loss": {
            "objective": "relative_error_loss",
            "target_loss_cap": UK_SCORE_LOSS_CAP,
            "train_equals_full": True,
        },
        "register": {
            "country": target_registry.country,
            "version": target_registry.version,
            "n_specs": len(target_registry),
        },
        "artifacts": {
            "candidate": candidate_label,
            "incumbent": incumbent_label,
        },
        "signed_asymmetries": [
            {
                "id": "incumbent_own_registry",
                "description": (
                    "The incumbent was calibrated to its own registry; both "
                    "artifacts are rescored here on the supplied frozen register."
                ),
            },
            {
                "id": "national_direct_vs_collapsed_local",
                "description": (
                    "Candidate national weights are a direct national solve; "
                    "incumbent national weights are the published national "
                    "surface being compared on the same frozen register."
                ),
            },
        ],
        "target_wins_by_family": _target_wins_by_family(
            target_registry,
            candidate_errors,
            incumbent_errors,
        ),
    }


def _relative_errors(result) -> dict[str, float]:
    return {
        diagnostic.name: float(diagnostic.relative_error)
        for diagnostic in result.diagnostics
    }


def _target_wins(
    candidate_errors: dict[str, float],
    incumbent_errors: dict[str, float],
) -> dict[str, int]:
    candidate = 0
    incumbent = 0
    for name, candidate_error in candidate_errors.items():
        incumbent_error = incumbent_errors[name]
        if abs(candidate_error) < abs(incumbent_error):
            candidate += 1
        elif abs(incumbent_error) < abs(candidate_error):
            incumbent += 1
    return {"candidate": candidate, "incumbent": incumbent}


def _target_wins_by_family(
    registry: TargetRegistry,
    candidate_errors: dict[str, float],
    incumbent_errors: dict[str, float],
) -> dict[str, dict[str, int]]:
    by_row_name = {spec.to_target().row_name: spec.family for spec in registry.specs}
    result: dict[str, dict[str, int]] = {}
    for name, family in by_row_name.items():
        bucket = result.setdefault(
            family,
            {"candidate_target_wins": 0, "incumbent_target_wins": 0, "ties": 0},
        )
        candidate_error = abs(candidate_errors[name])
        incumbent_error = abs(incumbent_errors[name])
        if candidate_error < incumbent_error:
            bucket["candidate_target_wins"] += 1
        elif incumbent_error < candidate_error:
            bucket["incumbent_target_wins"] += 1
        else:
            bucket["ties"] += 1
    return result


def _load_registry(path: str | Path) -> TargetRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        specs_payload = payload.get("specs")
        country = str(payload.get("country", "uk"))
    else:
        specs_payload = payload
        country = "uk"
    if not isinstance(specs_payload, list):
        raise ValueError("target registry JSON must contain a specs list.")
    return TargetRegistry(
        (TargetSpec(**item) for item in specs_payload),
        country=country,
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--incumbent-h5", required=True, type=Path)
    parser.add_argument("--target-registry-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    score = score_uk_national_candidate(
        candidate_h5=args.candidate_h5,
        incumbent_h5=args.incumbent_h5,
        target_registry=_load_registry(args.target_registry_json),
    )
    _write_json(args.output_json, {"score_vs_enhanced_frs": score})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
