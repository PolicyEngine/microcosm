"""Validation protocol and sidecar writer for the SLD layer (populace#625).

Three surfaces, all generated into the artifact sidecar rather than left in
a session log:

1. **Achieved-vs-target** — every calibrated cell of every district, with
   initial/final estimates and relative errors (the release diagnostic).
2. **Published-profile sanity** — the B19013 median household income check:
   the weighted median of the declared money-income analog under the solved
   weights against the published district median. Medians are
   validation-only by construction (a linear reweighting operator cannot
   honestly target one), so gaps are review items, never solve inputs.
3. **Statewide coherence** — per metric and state: the sum of district
   target values, the sum under the artifact's own calibrated weights, and
   the sum under the solved district weights. Coherence with the statewide
   artifact is reported, never constrained.

The honest-boundaries statement is generated INTO the sidecar (JSON and
markdown): what is district-calibrated, what is inherited from the
state-calibrated solve, the declared vintages, the membership assignment
method mix, the money-income instrument caveats, and the small-area tail
census.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.us_runtime.sld_local_solver import SldChamberSolveResult
from populace.build.us_runtime.sld_local_targets import (
    MoneyIncomeRecipeResolution,
    SldTargetFacts,
)

__all__ = [
    "achieved_vs_target_table",
    "honest_boundaries_statement",
    "median_income_validation",
    "render_boundaries_markdown",
    "statewide_coherence_report",
    "weighted_median",
    "write_sld_sidecar",
]

#: Districts with fewer assigned households than this are flagged as thin
#: in the boundaries statement.
THIN_DISTRICT_ROWS = 150

#: Relative gap beyond which a median-income comparison is flagged for
#: review in the validation table.
MEDIAN_REVIEW_THRESHOLD = 0.15


def achieved_vs_target_table(
    chambers: list[SldChamberSolveResult],
) -> pd.DataFrame:
    """Every calibrated cell of every district, one row per target."""
    frames = [
        result.diagnostics
        for chamber in chambers
        for result in chamber.district_results
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The weight-0.5 crossing of the sorted value distribution."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("values and weights must align and be non-empty.")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with positive total.")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    midpoint = 0.5 * cumulative[-1]
    return float(values[order][np.searchsorted(cumulative, midpoint)])


def median_income_validation(
    chambers: list[SldChamberSolveResult],
    facts: SldTargetFacts,
    money_income_by_household_id: Mapping[Any, float],
) -> pd.DataFrame:
    """B19013 published medians vs solved-weight medians, per district."""
    published = facts.validation
    rows: list[dict[str, Any]] = []
    if published.empty:
        return pd.DataFrame(rows)
    by_area = {
        (str(row.area_type), str(row.area_code)): float(row.value)
        for row in published.itertuples()
    }
    for chamber in chambers:
        for result in chamber.district_results:
            key = (result.problem.area_type, result.problem.area_code)
            if key not in by_area:
                continue
            incomes = np.array(
                [
                    float(money_income_by_household_id[household])
                    for household in result.problem.household_ids
                ]
            )
            achieved = weighted_median(incomes, result.weights)
            published_value = by_area[key]
            gap = achieved - published_value
            relative_gap = gap / published_value if published_value else np.inf
            rows.append(
                {
                    "area_type": result.problem.area_type,
                    "area_code": result.problem.area_code,
                    "published_median": published_value,
                    "achieved_median": achieved,
                    "gap": gap,
                    "relative_gap": relative_gap,
                    "review_flag": bool(abs(relative_gap) > MEDIAN_REVIEW_THRESHOLD),
                }
            )
    return pd.DataFrame(rows)


def statewide_coherence_report(
    chambers: list[SldChamberSolveResult],
) -> pd.DataFrame:
    """Per (area_type, state, metric): targets vs artifact vs solved sums."""
    rows: list[dict[str, Any]] = []
    for chamber in chambers:
        accumulator: dict[tuple[str, str], dict[str, float]] = {}
        for result in chamber.district_results:
            state = result.problem.state_fips
            initial = result.problem.matrix @ result.initial_weights
            solved = result.problem.matrix @ result.weights
            for index, metric in enumerate(
                result.problem.target_frame["metric"].astype(str)
            ):
                cell = accumulator.setdefault(
                    (state, metric),
                    {"target": 0.0, "artifact": 0.0, "solved": 0.0},
                )
                cell["target"] += float(result.problem.targets[index])
                cell["artifact"] += float(initial[index])
                cell["solved"] += float(solved[index])
        for (state, metric), sums in sorted(accumulator.items()):
            rows.append(
                {
                    "area_type": chamber.area_type,
                    "state_fips": state,
                    "metric": metric,
                    "district_target_sum": sums["target"],
                    "artifact_weight_sum": sums["artifact"],
                    "solved_weight_sum": sums["solved"],
                    "solved_vs_target_ratio": (
                        sums["solved"] / sums["target"] if sums["target"] else np.nan
                    ),
                    "solved_vs_artifact_ratio": (
                        sums["solved"] / sums["artifact"]
                        if sums["artifact"]
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def honest_boundaries_statement(
    *,
    chambers: list[SldChamberSolveResult],
    facts: SldTargetFacts,
    recipe_resolution: MoneyIncomeRecipeResolution,
    membership_gate_details: Mapping[str, Any],
    doctrine_record: Mapping[str, Any],
    zero_support_districts: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """The declared-boundaries record generated into the sidecar."""
    thin = [
        {
            "area_type": str(row.area_type),
            "area_code": str(row.area_code),
            "n_households": int(row.n_households),
        }
        for chamber in chambers
        for row in chamber.district_summary.itertuples()
        if int(row.n_households) < THIN_DISTRICT_ROWS
    ]
    min_ess = {
        chamber.area_type: float(
            chamber.district_summary["effective_sample_size"].min()
        )
        for chamber in chambers
    }
    return {
        "calibrated_surface": {
            "statement": (
                "Per-district weights are calibrated to ACS 5-year "
                "demographics and household-income brackets only: population "
                "by age band (S0101), household counts and household income "
                "brackets (B19001). All tax and program detail is inherited "
                "from the state-and-national calibrated artifact solve and "
                "is NOT district-calibrated."
            ),
            "metric_families": ["age_bands", "households", "income_brackets"],
        },
        "vintages": {
            "acs_window": "2020-2024 ACS 5-year estimates (2024 dollars)",
            "district_boundaries": sorted(facts.geography_vintages),
            "membership_source": "Census 2024 SLD block-equivalency files",
            "period_alignment": (
                "district targets are 2020-2024 5-year survey aggregates "
                "applied to the 2024 build-year artifact; this alignment is "
                "declared, not adjusted"
            ),
        },
        "membership_assignment": {
            "statement": (
                "District membership is derived at the 2024 boundary "
                "vintage: exact or population-weighted within-tract lookup "
                "for rows carrying certified tract geography, seeded "
                "block-overlap draws conditional on (PUMA, congressional "
                "district, county) for ACS-spine rows."
            ),
            "method_counts": dict(membership_gate_details.get("method_counts", {})),
            "unassigned_share": membership_gate_details.get("unassigned_share"),
        },
        "income_instrument": {
            "statement": (
                "Income brackets bind on a declared ACS money-income analog "
                "built from artifact input columns; the published median "
                "household income (B19013) is validation-only — a linear "
                "reweighting operator cannot honestly target a median."
            ),
            "recipe": recipe_resolution.as_record(),
        },
        "doctrine": dict(doctrine_record),
        "small_area_tails": {
            "census_rollups": [chamber.census_rollup for chamber in chambers],
            "thin_districts": thin,
            "thin_district_row_threshold": THIN_DISTRICT_ROWS,
            "min_effective_sample_size": min_ess,
            "zero_support_districts": {
                area_type: list(codes)
                for area_type, codes in zero_support_districts.items()
            },
        },
        "consumption": {
            "statement": (
                "Three lenses: household-level results use artifact rows "
                "directly; statewide results use the artifact's calibrated "
                "weights; by-district results use the per-district sidecar "
                "weights. District weight columns are valid only within "
                "their own district's rows."
            )
        },
    }


def render_boundaries_markdown(statement: Mapping[str, Any]) -> str:
    """A human-readable rendering of the boundaries statement."""
    lines = ["# SLD layer: declared boundaries", ""]
    lines += [statement["calibrated_surface"]["statement"], ""]
    vintages = statement["vintages"]
    lines += [
        "## Vintages",
        "",
        f"- ACS window: {vintages['acs_window']}",
        f"- District boundaries: {', '.join(vintages['district_boundaries'])}",
        f"- Membership source: {vintages['membership_source']}",
        f"- Period alignment: {vintages['period_alignment']}",
        "",
        "## Membership assignment",
        "",
        statement["membership_assignment"]["statement"],
        "",
    ]
    counts = statement["membership_assignment"]["method_counts"]
    for method, count in sorted(counts.items()):
        lines.append(f"- {method}: {count:,}")
    lines += ["", "## Income instrument", ""]
    lines.append(statement["income_instrument"]["statement"])
    recipe = statement["income_instrument"]["recipe"]
    lines += ["", "Declared omissions:"]
    for name, reason in sorted(recipe["declared_omissions"].items()):
        lines.append(f"- {name}: {reason}")
    lines += ["", "## Small-area tails", ""]
    tails = statement["small_area_tails"]
    for rollup in tails["census_rollups"]:
        lines.append(
            f"- {rollup['area_type']}: {rollup['n_districts']} districts, "
            f"{rollup['past_at_final']} target rows past the loss cap at "
            f"final, {rollup['pushed_out']} pushed out"
        )
    lines.append(
        f"- Thin districts (< {tails['thin_district_row_threshold']} rows): "
        f"{len(tails['thin_districts'])}"
    )
    lines += ["", "## Consumption", "", statement["consumption"]["statement"], ""]
    return "\n".join(lines)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_sld_sidecar(
    out_dir: str | Path,
    *,
    chambers: list[SldChamberSolveResult],
    facts: SldTargetFacts,
    recipe_resolution: MoneyIncomeRecipeResolution,
    membership_gate_details: Mapping[str, Any],
    doctrine_record: Mapping[str, Any],
    zero_support_districts: Mapping[str, tuple[str, ...]],
    money_income_by_household_id: Mapping[Any, float],
) -> dict[str, str]:
    """Write the sidecar bundle; returns filename -> sha256.

    Files: ``sld_local_weights.csv.gz`` (long per-district weights),
    ``sld_local_diagnostics.json`` (district summaries, census roll-ups,
    validation tables, doctrine, provenance), ``sld_local_boundaries.json``
    and ``sld_local_boundaries.md`` (the honest-boundaries statement), and
    ``sld_achieved_vs_target.csv.gz``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shas: dict[str, str] = {}

    long_weights = pd.concat(
        [chamber.long_weights for chamber in chambers],
        ignore_index=True,
    )
    weights_bytes = gzip.compress(
        long_weights.to_csv(index=False).encode("utf-8"),
        mtime=0,
    )
    (out / "sld_local_weights.csv.gz").write_bytes(weights_bytes)
    shas["sld_local_weights.csv.gz"] = _sha256_bytes(weights_bytes)

    achieved_bytes = gzip.compress(
        achieved_vs_target_table(chambers).to_csv(index=False).encode("utf-8"),
        mtime=0,
    )
    (out / "sld_achieved_vs_target.csv.gz").write_bytes(achieved_bytes)
    shas["sld_achieved_vs_target.csv.gz"] = _sha256_bytes(achieved_bytes)

    median_table = median_income_validation(
        chambers,
        facts,
        money_income_by_household_id,
    )
    coherence = statewide_coherence_report(chambers)
    diagnostics = {
        "schema_version": 1,
        "doctrine": dict(doctrine_record),
        "target_facts": {
            "source_path": facts.source_path,
            "source_sha256": facts.source_sha256,
            "n_calibration_facts": int(len(facts.calibration)),
            "n_validation_facts": int(len(facts.validation)),
            "geography_vintages": list(facts.geography_vintages),
        },
        "chambers": [
            {
                "area_type": chamber.area_type,
                "district_summary": chamber.district_summary.to_dict(orient="records"),
                "census_rollup": chamber.census_rollup,
            }
            for chamber in chambers
        ],
        "median_income_validation": median_table.to_dict(orient="records"),
        "statewide_coherence": coherence.to_dict(orient="records"),
    }
    diagnostics_bytes = (
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (out / "sld_local_diagnostics.json").write_bytes(diagnostics_bytes)
    shas["sld_local_diagnostics.json"] = _sha256_bytes(diagnostics_bytes)

    statement = honest_boundaries_statement(
        chambers=chambers,
        facts=facts,
        recipe_resolution=recipe_resolution,
        membership_gate_details=membership_gate_details,
        doctrine_record=doctrine_record,
        zero_support_districts=zero_support_districts,
    )
    statement_bytes = (json.dumps(statement, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    (out / "sld_local_boundaries.json").write_bytes(statement_bytes)
    shas["sld_local_boundaries.json"] = _sha256_bytes(statement_bytes)
    markdown_bytes = render_boundaries_markdown(statement).encode("utf-8")
    (out / "sld_local_boundaries.md").write_bytes(markdown_bytes)
    shas["sld_local_boundaries.md"] = _sha256_bytes(markdown_bytes)
    return shas
