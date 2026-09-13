#!/usr/bin/env python3
"""Compare a local attendance candidate with its exact parent under each state's model.

Only finite, imputed child inputs change in this counterfactual. Unresolved
out-of-domain inputs retain their baseline engine behavior and are counted,
not certified as observed zeros. Results are potential modeled benefits, not
CCDF caseload or spending estimates. No microdata are published by this tool.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.h5_io import load_legacy_calibrated_us_h5
from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-h5", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("The report path must be new")
    with args.parent_h5.open("rb") as stream:
        parent_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if parent_hash != args.parent_sha256:
        parser.error("Parent population hash mismatch")
    parent = load_legacy_calibrated_us_h5(args.parent_h5)
    stored = load_frame_checkpoint(args.candidate_checkpoint)
    candidate = stored.frame
    for entity in parent.entities:
        pd.testing.assert_frame_equal(
            parent.table(entity), candidate.table(entity)[parent.table(entity).columns]
        )
    for entity in parent.weighted_entities:
        np.testing.assert_array_equal(
            parent.weights_for(entity).values, candidate.weights_for(entity).values
        )
    p = candidate.table("person")
    attendance = p[list(US_CHILDCARE_ATTENDANCE_COLUMNS)].to_numpy(dtype=float)
    resolved = np.isfinite(attendance).all(axis=1)
    young = p.age.between(0, 12).to_numpy()
    if not resolved[young].all():
        raise ValueError("Candidate leaves under-13 attendance unresolved")
    hh = parent.table("household").copy()
    hh["household_weight"] = parent.weights_for("household").values
    person_weights = p.person_household_id.map(
        hh.set_index("household_id").household_weight
    ).to_numpy()
    days = attendance[:, 1]
    report = {
        "parent_sha256": parent_hash,
        "candidate_checkpoint_sha256": _sha256(args.candidate_checkpoint),
        "validation_code_sha256": _sha256(Path(__file__)),
        "engine_version": version("policyengine-us"),
        "policy_year": args.year,
        "population": "fixed BuildP source ages/incomes; no aging or uprating",
        "people": len(p),
        "households": len(hh),
        "under13_children": int(young.sum()),
        "outside_source_domain_people": int((~young).sum()),
        "outside_source_domain_age13_17_with_disability": int(
            (p.age.between(13, 17) & p.is_disabled).sum()
        ),
        "outside_domain_baseline_receipt": stored.metadata.get(
            "frame_metadata", {}
        ).get("childcare_outside_domain_baseline", {}),
        "unresolved_people_retaining_baseline": int((~resolved).sum()),
        "unresolved_age13_17_with_disability": int(
            (~resolved & p.age.between(13, 17) & p.is_disabled).sum()
        ),
        "weighted_under13_attendance_share": float(
            np.average(days[young] > 0, weights=person_weights[young])
        ),
        "weighted_under13_days_per_week": float(
            np.average(days[young], weights=person_weights[young])
        ),
        "weighted_under13_hours_per_week": float(
            np.average(
                days[young] * attendance[young, 2], weights=person_weights[young]
            )
        ),
        "candidate_receipt": stored.metadata.get("frame_metadata", {}),
        "states": [],
        "production_ready": False,
        "interpretation": "attendance-only counterfactual; provider, activity, expenses and take-up inputs remain as in parent",
    }
    if "childcare_attendance_match_level" in p:
        report["matching_levels"] = (
            p.loc[young, "childcare_attendance_match_level"].value_counts().to_dict()
        )
    report["population_attendance_by_group"] = []
    for grouping in ("age", "region", "parent_work_status", "income_band"):
        for label, indices in p.loc[young].groupby(grouping).groups.items():
            positions = p.index.get_indexer(indices)
            w = person_weights[positions]
            d = attendance[positions, 1]
            report["population_attendance_by_group"].append(
                {
                    "grouping": grouping,
                    "group": str(label),
                    "children": len(indices),
                    "weighted_children": float(w.sum()),
                    "attendance_share": float(np.average(d > 0, weights=w)),
                    "days_per_week": float(np.average(d, weights=w)),
                    "hours_per_week": float(
                        np.average(d * attendance[positions, 2], weights=w)
                    ),
                }
            )
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    try:
        for state_fips, households in hh.groupby("state_fips", sort=True):
            state = US_STATE_NUMERIC_FIPS_TO_POSTAL[int(state_fips)]
            rows = p.person_household_id.isin(households.household_id)
            people = parent.table("person").loc[rows].copy()
            tables = {"person": people, "household": households}
            for entity in parent.schema.group_entities:
                if entity != "household":
                    table = parent.table(entity)
                    tables[entity] = table.loc[
                        table[f"{entity}_id"].isin(people[f"person_{entity}_id"])
                    ]
            result = {"state": state, "sample_households": len(households)}
            for name in ("baseline", "candidate"):
                sim = Microsimulation(
                    dataset=USSingleYearDataset(**tables, time_period=args.year)
                )
                if name == "candidate":
                    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
                        values = np.asarray(sim.calculate(column, args.year)).copy()
                        known = resolved[rows]
                        values[known] = p.loc[rows, column].to_numpy()[known]
                        sim.set_input(column, args.year, values)
                variable = f"{state.lower()}_child_care_subsidies"
                if variable not in sim.tax_benefit_system.variables:
                    raise ValueError(f"The pinned engine has no {variable}")
                definition = sim.tax_benefit_system.variables[variable]
                entity = definition.entity.key
                amount = np.asarray(sim.calculate(variable, args.year), dtype=float)
                weights = np.asarray(
                    sim.calculate(f"{entity}_weight", args.year), dtype=float
                )
                if not np.isfinite(amount).all():
                    raise ValueError(f"Nonfinite state benefit: {state}/{name}")
                result[name] = {
                    "variable": variable,
                    "entity": entity,
                    "positive_sample_units": int((amount > 0).sum()),
                    "weighted_positive_units": float(weights[amount > 0].sum()),
                    "annual_modeled_benefits": float(amount @ weights),
                }
                del sim
                gc.collect()
            report["states"].append(result)
            print(
                f"{state}: {result['baseline']['positive_sample_units']} -> {result['candidate']['positive_sample_units']} positive units",
                flush=True,
            )
        report["summary"] = {
            "states_evaluated": len(report["states"]),
            "all_zero_states": {
                arm: [
                    row["state"]
                    for row in report["states"]
                    if row[arm]["positive_sample_units"] == 0
                ]
                for arm in ("baseline", "candidate")
            },
            "annual_potential_modeled_benefits": {
                arm: sum(
                    row[arm]["annual_modeled_benefits"] for row in report["states"]
                )
                for arm in ("baseline", "candidate")
            },
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
