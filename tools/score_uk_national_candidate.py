"""Compare a UK national candidate with the reference on one frozen register.

The comparison core lives in ``microcosm.build.uk_runtime.candidate_score``
(over the country-neutral ``microcosm.build.head_to_head``); this command
loads the frozen register, verifies both artifacts against their digests,
compares them on the surface both can materialize (pruning the rows the
reference cannot, loudly, unless ``--no-prune-reference-unresolvable``) and
writes the receipt the release-cut certifier reads.

The reference is the current national build a cut recorded when it fetched
it (microcosm#1095): pass ``--reference-record`` with its file. An ad hoc
comparison may name a file with ``--reference-sha256`` and
``--reference-label`` instead; its receipt carries no reference record, so
it never certifies a cut. The comparison decides nothing: it has no
threshold, like the US head-to-head.
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
    _verify_artifact,
    evaluate_uk_candidate_against_reference,
    pruned_warning,
    score_uk_national_candidate,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _reference(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """The reference's sha, label and record, from the record when given."""

    if args.reference_record is not None:
        if args.reference_sha256 is not None or args.reference_label is not None:
            parser.error(
                "--reference-record names the reference; drop --reference-sha256 "
                "and --reference-label."
            )
        from microcosm.build.uk_runtime.cut_reference import (
            read_uk_cut_reference,
            verify_uk_cut_reference_file,
        )

        record = read_uk_cut_reference(args.reference_record)
        verify_uk_cut_reference_file(record, args.reference_h5)
        return record.sha256, record.label, record.to_record()
    if args.reference_sha256 is None or args.reference_label is None:
        parser.error(
            "pass --reference-record, or both --reference-sha256 and "
            "--reference-label for an ad hoc comparison."
        )
    return args.reference_sha256, args.reference_label, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--reference-h5", required=True, type=Path)
    parser.add_argument(
        "--reference-record",
        type=Path,
        help="The cut's record of the reference build (tools/resolve_uk_reference.py).",
    )
    parser.add_argument(
        "--reference-sha256", help="Ad hoc only: the reference file's SHA-256."
    )
    parser.add_argument(
        "--reference-label", help="Ad hoc only: the reference's label in the receipt."
    )
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--calibration-year", type=int)
    parser.add_argument(
        "--candidate-label",
        help="Override the candidate label; by default it is the candidate H5 stem.",
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
        "--no-prune-reference-unresolvable",
        action="store_true",
        help=(
            "Refuse the comparison when the reference cannot materialize a "
            "target instead of pruning that target from both arms and "
            "reporting it. The default prunes and warns, naming every absent "
            "measure."
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
    reference_sha256, reference_label, reference_record = _reference(args, parser)
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
    score = evaluate_uk_candidate_against_reference(
        candidate_h5=args.candidate_h5,
        reference_h5=args.reference_h5,
        candidate_sha256=args.candidate_sha256,
        reference_sha256=reference_sha256,
        target_registry=registry,
        calibration_year=int(calibration_year),
        reference_label=reference_label,
        measure_resolver_factory=factory,
        candidate_label=args.candidate_label,
        band_edge_registry=band_edge_registry,
        prune_reference_unresolvable_measures=not args.no_prune_reference_unresolvable,
        reference_record=reference_record,
    )
    warning = pruned_warning(score)
    if warning is not None:
        print(warning, file=sys.stderr, flush=True)
    # The receipt is the score block itself: the release-cut certifier reads
    # ``artifacts.candidate.sha256`` and ``reference`` at the top level, and
    # the assembler copies the bytes into the release directory unchanged.
    _write_json(args.output_json, score)
    evaluation = score["evaluation"]
    targets = evaluation["targets"]
    print(
        f"comparison with {evaluation['reference']} (no threshold): candidate "
        f"loss {evaluation['losses']['candidate_full_loss']:.6f} vs reference "
        f"{evaluation['losses']['reference_full_loss']:.6f} on "
        f"{evaluation['scored_surface']['n_scored']} of "
        f"{evaluation['scored_surface']['n_surface']} targets; lower error on "
        f"{targets['candidate_lower_count']} for the candidate, "
        f"{targets['reference_lower_count']} for the reference, "
        f"{targets['equal_count']} equal",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
