#!/usr/bin/env python3
"""Stress-test unidentified noncalendar days/irregular hours on a fixed parent.

Outputs aggregate evidence only. This does not identify the missing schedule,
certify a dataset, or authorize publication. All output paths must be new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    assert_bound_childcare_attendance,
    attendance_recipe_identity,
)
from microcosm.build.us_runtime.childcare_population import (
    harmonize_asec_childcare_predictors,
)
from microcosm.build.us_runtime.childcare_sensitivity import (
    compare_childcare_scenarios,
    noncalendar_sensitivity_source,
    paired_noncalendar_attendance,
)
from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CHILDCARE_FALLBACK_COLUMNS,
    NSECE_CHILDCARE_MATCH_COLUMNS,
    load_nsece_childcare,
    with_us_nsece_childcare_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_bridge import (
    bridge_nsece_noncalendar_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_dependence import (
    fit_nsece_sibling_dependence,
)
from microcosm.frame import Frame


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-h5", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--asec-source-cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=915)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        help="Reuse a bound candidate from this parent/seed; retain its donor assignments",
    )
    args = parser.parse_args()
    if args.report.exists():
        parser.error("The report path must be new")
    if _sha256(args.parent_h5) != args.parent_sha256:
        parser.error("Parent population hash mismatch")
    parent = load_legacy_calibrated_us_h5(args.parent_h5)
    if args.candidate_checkpoint:
        stored = load_frame_checkpoint(args.candidate_checkpoint)
        if stored.metadata["parent_checkpoint_sha256"] != args.parent_sha256:
            parser.error("Candidate checkpoint belongs to another parent")
        frame = stored.frame
        transferred = Frame(
            {e: frame.table(e) for e in frame.entities},
            frame.schema,
            {e: frame.weights_for(e) for e in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=stored.metadata["frame_metadata"],
        )
        assert_bound_childcare_attendance(transferred)
        if transferred.metadata["childcare_attendance_stage"]["seed"] != args.seed:
            parser.error("Candidate checkpoint uses another seed")
        for entity in parent.entities:
            # Check values, missingness and IDs; the dedicated artifact verifier
            # additionally checks dtypes after normalizing string storage.
            pd.testing.assert_frame_equal(
                parent.table(entity),
                transferred.table(entity).reindex(columns=parent.table(entity).columns),
                check_dtype=False,
                check_exact=True,
            )
        for entity in parent.weighted_entities:
            np.testing.assert_array_equal(
                parent.weights_for(entity).values,
                transferred.weights_for(entity).values,
            )
    else:
        normalized = harmonize_asec_childcare_predictors(
            parent, source_cache=args.asec_source_cache
        )
    source = load_nsece_childcare(args.household_tsv, args.calendar_tsv)
    dependence = fit_nsece_sibling_dependence(source.children)
    bridged = bridge_nsece_noncalendar_attendance(source, seed=args.seed)
    arms = {
        "candidate": bridged,
        "no_modeled_irregular_hours": noncalendar_sensitivity_source(
            bridged, irregular_hours=False
        ),
        "one_fewer_day": noncalendar_sensitivity_source(bridged, day_shift=-1),
        "one_more_day": noncalendar_sensitivity_source(bridged, day_shift=1),
    }
    if not args.candidate_checkpoint:
        transferred = with_us_nsece_childcare_attendance(
            normalized,
            bridged,
            seed=args.seed,
            match_columns=NSECE_CHILDCARE_MATCH_COLUMNS,
            fallback_match_columns=NSECE_CHILDCARE_FALLBACK_COLUMNS,
            sibling_dependence=dependence["rho"],
        )
    scenarios, assignment_checks = {}, {}
    for name, donors in arms.items():
        scenarios[name], assignment_checks[name] = paired_noncalendar_attendance(
            transferred.table("person"), bridged, donors
        )
        print(f"Prepared {name}", flush=True)
    people = transferred.table("person")
    young = people.age.between(0, 12).to_numpy()
    report = {
        "parent_sha256": args.parent_sha256,
        "source": source.source_receipt,
        "recipe": attendance_recipe_identity(),
        "seed": args.seed,
        "candidate_checkpoint_sha256": _sha256(args.candidate_checkpoint)
        if args.candidate_checkpoint
        else None,
        "assignment_checks": assignment_checks,
        "coupling": "same donor identities across scenarios; no care-rank reassignment",
        "engine_version": version("policyengine-us"),
        "policy_year": args.year,
        "code_sha256": {
            str(p.name): _sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).parents[1]
                / "packages/microcosm-build/src/microcosm/build/us_runtime/childcare_sensitivity.py",
            )
        },
        "population": "fixed BuildP source ages/incomes; no aging or uprating",
        "assumptions": {
            name: donors.source_receipt.get("transport_sensitivity", {})
            for name, donors in arms.items()
        },
        "changed_children": {
            name: int(
                np.any(values[young] != scenarios["candidate"][young], axis=1).sum()
            )
            for name, values in scenarios.items()
        },
        "screens": {"national_relative_change": 0.10, "state_relative_change": 0.20},
        "states": [],
        "production_ready": False,
        "interpretation": "assumption stress tests, not confidence intervals; potential modeled benefits, not calibrated spending",
    }
    try:
        for result in compare_childcare_scenarios(parent, scenarios, year=args.year):
            report["states"].append(result)
            print(result["state"], flush=True)
        totals = {
            name: sum(row[name]["annual_modeled_benefits"] for row in report["states"])
            for name in ("baseline", *arms)
        }
        flags = []
        comparisons = []
        for name in arms:
            if name == "candidate":
                continue
            for label, original, changed, limit in (
                ("US", totals["candidate"], totals[name], 0.10),
                *(
                    (
                        r["state"],
                        r["candidate"]["annual_modeled_benefits"],
                        r[name]["annual_modeled_benefits"],
                        0.20,
                    )
                    for r in report["states"]
                ),
            ):
                relative = (changed - original) / original if original else None
                flagged = (
                    abs(relative) > limit if relative is not None else changed != 0
                )
                comparison = {
                    "scenario": name,
                    "state": label,
                    "relative_change": relative,
                    "absolute_change": changed - original,
                    "flagged": flagged,
                }
                comparisons.append(comparison)
                if flagged:
                    flags.append(comparison)
        report["summary"] = {
            "states_evaluated": len(report["states"]),
            "annual_potential_modeled_benefits": totals,
            "sensitivity_flags": flags,
            "comparisons": comparisons,
        }
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, allow_nan=False, default=dict) + "\n"
        )


if __name__ == "__main__":
    main()
