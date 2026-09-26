"""Score a UK national candidate against the incumbent on one frozen register.

The score core lives in ``microcosm.build.uk_runtime.candidate_score``; this
command loads the frozen register, verifies both artifacts against their
digests, scores on the surface both can materialize (pruning the rows the
incumbent cannot, loudly, unless ``--no-prune-incumbent-unresolvable``) and
writes the receipt the release-cut certifier reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.candidate_score import (  # noqa: F401 - re-exports
    UK_SCORE_HOLDOUT_BASIS,
    UK_SCORE_LOSS_CAP,
    _default_measure_resolver_factory,
    _holdout_loss,
    _load_registry,
    _relative_errors,
    _scored_frame,
    _sha256_file,
    _target_drift,
    _target_wins,
    _target_wins_by_family,
    _verify_artifact,
    evaluate_uk_candidate_against_incumbent,
    pruned_warning,
    score_uk_national_candidate,
)


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
        "--band-edge-registry-json",
        type=Path,
        help=(
            "Full compiled contract register supplying banded fan-out edges "
            "when --registry-json is a pruned scoring surface (#803: a "
            "pruned surface must never redraw its own band edges)."
        ),
    )
    parser.add_argument(
        "--no-prune-incumbent-unresolvable",
        action="store_true",
        help=(
            "Refuse the score when the incumbent cannot materialize a target "
            "instead of pruning that target from both arms and reporting it. "
            "The default prunes and warns, naming every absent measure."
        ),
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
    band_edge_registry = (
        None
        if args.band_edge_registry_json is None
        else _load_registry(args.band_edge_registry_json)
    )
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
    score = evaluate_uk_candidate_against_incumbent(
        candidate_h5=args.candidate_h5,
        incumbent_h5=args.incumbent_h5,
        candidate_sha256=args.candidate_sha256,
        incumbent_sha256=args.incumbent_sha256,
        target_registry=registry,
        calibration_year=int(calibration_year),
        measure_resolver_factory=factory,
        candidate_label=args.candidate_label,
        incumbent_label=args.incumbent_label,
        band_edge_registry=band_edge_registry,
        prune_incumbent_unresolvable_measures=not args.no_prune_incumbent_unresolvable,
    )
    warning = pruned_warning(score)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)
    # The receipt is the score block itself: the release-cut certifier reads
    # ``artifacts.candidate.sha256`` and ``evaluation.verdict`` at the top
    # level, and the assembler copies the bytes into the release directory
    # unchanged.
    _write_json(args.output_json, score)
    evaluation = score["evaluation"]
    print(
        f"evaluation {evaluation['verdict']}: candidate loss "
        f"{evaluation['rule_1']['candidate_full_loss']:.6f} vs incumbent "
        f"{evaluation['rule_1']['incumbent_full_loss']:.6f} on "
        f"{evaluation['scored_surface']['n_scored']} of "
        f"{evaluation['scored_surface']['n_surface']} targets",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
