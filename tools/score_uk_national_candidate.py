"""Score a UK national candidate against the incumbent on one frozen register."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.national_calibration import prepare_uk_target_frame
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.calibrate import TargetRegistry, score_targets

UK_SCORE_LOSS_CAP = 10.0
UK_SCORE_HOLDOUT_BASIS = "none_declared"


def _holdout_loss(basis: str) -> float | None:
    """The holdout loss for a declared basis, or None when none exists.

    June's frozen fixture carries a genuinely different holdout value
    (0.1239 against a 0.0159 train loss) because it held rows out. This
    register declares no split, so the holdout keys report absence rather
    than the fitted loss wearing a holdout name — a copied value reads
    downstream as perfect generalization from a measurement never made.
    Declaring a basis without computing its split fails loudly here rather
    than falling back to the fitted loss.
    """

    if basis == "none_declared":
        return None
    raise NotImplementedError(
        f"holdout basis {basis!r} declares a split this scorer does not compute."
    )


def score_uk_national_candidate(
    *,
    candidate_h5: str | Path,
    incumbent_h5: str | Path,
    candidate_sha256: str,
    incumbent_sha256: str,
    target_registry: TargetRegistry,
    calibration_year: int,
    measure_resolver_factory: Callable[[Path, Any], Any] | None = None,
    candidate_label: str | None = None,
    incumbent_label: str = "enhanced_frs_2024_25",
) -> dict[str, Any]:
    """Return the #578 rule-1 score block on a shared target registry.

    Both artifacts are verified against their declared digests before a byte
    is read, and both sides are materialized through the same route the
    calibration seam uses, so the score names the artifacts it actually
    measured rather than two labels supplied on the command line.
    """

    if target_registry.country != "uk" or not target_registry.specs:
        raise ValueError("UK candidate scoring requires a non-empty UK registry.")
    candidate_path = Path(candidate_h5)
    if candidate_label is None:
        candidate_label = candidate_path.stem
    candidate_pin = _verify_artifact("candidate", candidate_h5, candidate_sha256)
    incumbent_pin = _verify_artifact("incumbent", incumbent_h5, incumbent_sha256)
    candidate_frame, candidate_resolution = _scored_frame(
        candidate_h5, target_registry, calibration_year, measure_resolver_factory
    )
    incumbent_frame, incumbent_resolution = _scored_frame(
        incumbent_h5, target_registry, calibration_year, measure_resolver_factory
    )
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
        "candidate_holdout_loss": _holdout_loss(UK_SCORE_HOLDOUT_BASIS),
        "candidate_full_loss": float(candidate.final_loss),
        "incumbent_train_loss": float(incumbent.final_loss),
        "incumbent_holdout_loss": _holdout_loss(UK_SCORE_HOLDOUT_BASIS),
        "incumbent_full_loss": float(incumbent.final_loss),
        "candidate_target_wins": wins["candidate"],
        "incumbent_target_wins": wins["incumbent"],
        "holdout_basis": UK_SCORE_HOLDOUT_BASIS,
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
            "candidate": {"label": candidate_label, **candidate_pin},
            "incumbent": {"label": incumbent_label, **incumbent_pin},
        },
        "measure_resolution": {
            "candidate": candidate_resolution,
            "incumbent": incumbent_resolution,
        },
        "target_drift": _target_drift(
            target_registry, candidate_errors, incumbent_errors
        ),
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
    """Load the frozen register through its own validating loader.

    ``TargetRegistry.from_json`` checks the format revision and re-derives the
    content hash, so a hand-edited or drifted register refuses here instead of
    silently deciding rule 1 on a surface nobody reviewed.
    """

    return TargetRegistry.from_json(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(role: str, path: str | Path, expected: str) -> dict[str, object]:
    measured = _sha256_file(path)
    if measured != expected:
        raise ValueError(
            f"{role} artifact sha mismatch: measured {measured}, pinned {expected}"
        )
    return {
        "path": str(path),
        "sha256": measured,
        "size_bytes": Path(path).stat().st_size,
    }


def _scored_frame(
    h5_path: str | Path,
    registry: TargetRegistry,
    calibration_year: int,
    factory: Callable[[Path, Any], Any] | None,
) -> tuple[Any, Any]:
    frame, _provenance = load_uk_national_frame(h5_path)
    resolver = None if factory is None else factory(Path(h5_path), frame)
    return prepare_uk_target_frame(
        frame, registry, period=calibration_year, measure_resolver=resolver
    )


def _default_measure_resolver_factory(scratch_dir: Path, year: int):
    def build(h5_path: Path, frame: Any):
        from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver

        return UKMeasureResolver(
            simulation_source=h5_path,
            scratch_dir=scratch_dir,
            year=year,
            frame=frame,
        )

    return build


def _target_drift(
    registry: TargetRegistry,
    candidate_errors: dict[str, float],
    incumbent_errors: dict[str, float],
) -> list[dict[str, object]]:
    """Per-target relative errors, the auditable half of the score block."""

    families = {spec.to_target().row_name: spec.family for spec in registry.specs}
    rows = []
    for name in sorted(candidate_errors):
        candidate = candidate_errors[name]
        incumbent = incumbent_errors[name]
        # Relative errors are signed; the win is decided on magnitude, the
        # same rule the aggregate counts use.
        rows.append(
            {
                "target": name,
                "family": families.get(name),
                "candidate_relative_error": candidate,
                "incumbent_relative_error": incumbent,
                "winner": (
                    "candidate"
                    if abs(candidate) < abs(incumbent)
                    else "incumbent"
                    if abs(incumbent) < abs(candidate)
                    else "tie"
                ),
            }
        )
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--incumbent-h5", required=True, type=Path)
    parser.add_argument("--incumbent-sha256", required=True)
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--calibration-year", type=int)
    parser.add_argument(
        "--candidate-label",
        help="Override the candidate label; by default it is the candidate H5 stem.",
    )
    parser.add_argument(
        "--incumbent-label",
        default="enhanced_frs_2024_25",
        help="Override the incumbent/reference label.",
    )
    parser.add_argument(
        "--no-measure-resolution",
        action="store_true",
        help=(
            "Score frames whose measures are already columns. Every packaged "
            "UK reference binds a prepared measure, so this is for fixtures "
            "only; a production register refuses on the first skipped target."
        ),
    )
    args = parser.parse_args(argv)
    registry = _load_registry(args.registry_json)
    calibration_year = args.calibration_year
    if calibration_year is None:
        from microcosm.build.uk_runtime.frs_release import load_uk_frs_release

        calibration_year = load_uk_frs_release().calibration_year
    factory = (
        None
        if args.no_measure_resolution
        else _default_measure_resolver_factory(
            args.output_json.parent, int(calibration_year)
        )
    )
    score = score_uk_national_candidate(
        candidate_h5=args.candidate_h5,
        incumbent_h5=args.incumbent_h5,
        candidate_sha256=args.candidate_sha256,
        incumbent_sha256=args.incumbent_sha256,
        target_registry=registry,
        calibration_year=int(calibration_year),
        measure_resolver_factory=factory,
        candidate_label=args.candidate_label,
        incumbent_label=args.incumbent_label,
    )
    _write_json(args.output_json, {"score_vs_enhanced_frs": score})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
