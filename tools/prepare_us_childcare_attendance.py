#!/usr/bin/env python3
"""Prepare a local NSECE attendance candidate and aggregate validation report.

No source downloads, remote uploads or release publication occur. Download the
DS4/DS5 TSVs under the ICPSR terms first. A Frame checkpoint is optional; source
validation is useful before a population candidate is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import childcare_attendance, nsece_childcare
from microcosm.build.us_runtime.childcare_attendance_stage import (
    export_native_childcare_candidate,
    inherit_outside_domain_attendance_baseline,
    with_us_childcare_attendance_inputs,
)
from microcosm.build.us_runtime.childcare_population import (
    harmonize_asec_childcare_predictors,
)
from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CHILDCARE_FALLBACK_COLUMNS,
    NSECE_CHILDCARE_MATCH_COLUMNS,
    load_nsece_childcare,
    nsece_childcare_validation_report,
    with_us_nsece_childcare_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_assessment import (
    assess_noncalendar_bridge,
    assess_nsece_childcare,
)
from microcosm.build.us_runtime.nsece_childcare_bridge import (
    bridge_nsece_noncalendar_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_dependence import (
    fit_nsece_sibling_dependence,
)
from microcosm.frame import Frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=915)
    parser.add_argument(
        "--model-sibling-dependence",
        action="store_true",
        help="Fit measured household dependence for shared-rank schedule draws",
    )
    parser.add_argument(
        "--bridge-noncalendar",
        action="store_true",
        help="Use measured regular hours to impute missing days and irregular care",
    )
    parser.add_argument(
        "--extended-assessment",
        action="store_true",
        help="Run household cross-validation, instrument transport and masked-calendar checks",
    )
    parser.add_argument(
        "--match-columns", nargs="+", default=list(NSECE_CHILDCARE_MATCH_COLUMNS)
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-checkpoint", type=Path)
    inputs.add_argument("--asec-population-h5", type=Path)
    parser.add_argument(
        "--asec-source-cache",
        type=Path,
        help="Verified Census pppub23/24/25.csv files for omitted PTOTVAL",
    )
    parser.add_argument(
        "--population-sha256", help="Required exact hash for --asec-population-h5"
    )
    parser.add_argument(
        "--sparse-cell-fallback",
        action="store_true",
        help="Explicitly use age/parent-work then age-only donor matches",
    )
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--output-native-h5", type=Path)
    parser.add_argument(
        "--inherit-outside-domain-baseline",
        action="store_true",
        help="Explicitly retain baseline outside ages 0-12 for export; not observed nonattendance",
    )
    parser.add_argument(
        "--production-stage",
        action="store_true",
        help="Run the same complete attendance stage used by the fiscal refresh builder",
    )
    args = parser.parse_args()
    if args.production_stage:
        if not args.asec_population_h5:
            parser.error(
                "The production-stage qualification requires a native ASEC parent"
            )
        args.bridge_noncalendar = args.model_sibling_dependence = (
            args.sparse_cell_fallback
        ) = True
    input_path = args.input_checkpoint or args.asec_population_h5
    if bool(input_path) != bool(args.output_checkpoint):
        parser.error(
            "A population input and --output-checkpoint must be supplied together"
        )
    if args.asec_population_h5 and (
        not args.population_sha256
        or _sha256(args.asec_population_h5) != args.population_sha256
    ):
        parser.error("The ASEC population must match --population-sha256")
    if (
        args.sparse_cell_fallback
        and tuple(args.match_columns) != NSECE_CHILDCARE_MATCH_COLUMNS
    ):
        parser.error("The reviewed fallback requires the canonical matching fields")
    if args.output_native_h5 and (
        not args.asec_population_h5 or not args.output_checkpoint
    ):
        parser.error(
            "Native output requires an exact ASEC population parent and checkpoint output"
        )
    outputs = [
        p
        for p in (args.report, args.output_checkpoint, args.output_native_h5)
        if p is not None
    ]
    if any(p.exists() for p in outputs):
        parser.error(
            "Output paths must be new; existing artifacts will not be overwritten"
        )
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("Report and checkpoint paths must differ")
    source = load_nsece_childcare(args.household_tsv, args.calendar_tsv)
    report = nsece_childcare_validation_report(
        source, seed=args.seed, match_columns=tuple(args.match_columns)
    )
    sibling_fit = (
        fit_nsece_sibling_dependence(source.children)
        if args.model_sibling_dependence
        else {"rho": 0.0}
    )
    report["sibling_dependence"] = sibling_fit
    if args.extended_assessment:
        report["selection_and_cross_validation"] = assess_nsece_childcare(source)
        report["masked_calendar_validation"] = assess_noncalendar_bridge(source)
    if args.bridge_noncalendar:
        source = bridge_nsece_noncalendar_attendance(source, seed=args.seed)
        report["noncalendar_bridge"] = source.source_receipt["noncalendar_bridge"]
    report["candidate_frame_written"] = False
    report["environment"] = {
        package: version(package)
        for package in (
            "numpy",
            "pandas",
            "microcosm-build",
            "microcosm-frame",
            "policyengine-us",
        )
    }
    report["environment"]["python"] = platform.python_version()
    report["code_sha256"] = {
        path.name: _sha256(path)
        for path in (
            Path(__file__),
            Path(childcare_attendance.__file__),
            Path(nsece_childcare.__file__),
            *Path(childcare_attendance.__file__).parent.glob("*childcare*.py"),
            Path(childcare_attendance.__file__).parent.parent
            / "us"
            / "childcare_attendance_source.json",
            Path(childcare_attendance.__file__).parent
            / "education_assistance_source.py",
            Path(__file__).resolve().parents[1]
            / "packages/microcosm-calibrate/src/microcosm/calibrate/geography_constants.py",
            Path(__file__).resolve().parents[1] / "uv.lock",
        )
    }
    try:
        if input_path is not None:
            if args.asec_population_h5:
                from microcosm.build.frame_checkpoint import LoadedFrameCheckpoint

                frame = load_legacy_calibrated_us_h5(input_path)
                if not args.production_stage:
                    frame = harmonize_asec_childcare_predictors(
                        frame, source_cache=args.asec_source_cache
                    )
                original = LoadedFrameCheckpoint(
                    frame, {"parent_population_sha256": args.population_sha256}
                )
            else:
                original = load_frame_checkpoint(input_path)
            # The checkpoint protocol deliberately stores Frame receipts as
            # external metadata; restore them explicitly rather than dropping
            # the parent build's source/ownership evidence.
            parent = original.frame
            parent = Frame(
                {entity: parent.table(entity) for entity in parent.entities},
                parent.schema,
                {
                    entity: parent.weights_for(entity)
                    for entity in parent.weighted_entities
                },
                parent.strata,
                mass_log=parent.mass_log,
                metadata={
                    **parent.metadata,
                    **original.metadata.get("frame_metadata", {}),
                },
            )
            if args.production_stage:
                candidate = with_us_childcare_attendance_inputs(
                    parent,
                    household_tsv=args.household_tsv,
                    calendar_tsv=args.calendar_tsv,
                    asec_source_cache=args.asec_source_cache,
                    seed=args.seed,
                    inherit_outside_domain_baseline=args.inherit_outside_domain_baseline,
                )
            else:
                candidate = with_us_nsece_childcare_attendance(
                    parent,
                    source,
                    seed=args.seed,
                    match_columns=tuple(args.match_columns),
                    fallback_match_columns=NSECE_CHILDCARE_FALLBACK_COLUMNS
                    if args.sparse_cell_fallback
                    else (),
                    sibling_dependence=sibling_fit["rho"],
                )
                if args.inherit_outside_domain_baseline:
                    candidate = inherit_outside_domain_attendance_baseline(candidate)
            write_frame_checkpoint(
                args.output_checkpoint,
                candidate,
                metadata={
                    "artifact_kind": "nsece_childcare_candidate",
                    "childcare_candidate_only": True,
                    "parent_checkpoint_metadata": original.metadata,
                    "parent_checkpoint_sha256": _sha256(input_path),
                    "frame_metadata": json.loads(
                        json.dumps(candidate.metadata, default=dict)
                    ),
                },
            )
            report["candidate_frame_written"] = True
            report["candidate_checkpoint_sha256"] = _sha256(args.output_checkpoint)
            report["parent_population_sha256"] = _sha256(input_path)
            report["production_stage_executed"] = args.production_stage
            report["candidate_receipts"] = json.loads(
                json.dumps(candidate.metadata, default=dict)
            )
            if args.output_native_h5:
                export_native_childcare_candidate(
                    args.asec_population_h5, candidate, args.output_native_h5
                )
                report["native_candidate_written"] = True
                report["native_candidate_sha256"] = _sha256(args.output_native_h5)
    except Exception as exc:
        report["candidate_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Validation report: {args.report}")
    print("Candidate only; production readiness is not certified.")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
