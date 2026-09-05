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

from microcosm.build.us_runtime.sld_local_solver import SldChamberSolveResult
from microcosm.build.us_runtime.sld_local_targets import (
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
    zero_support_facts: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Every calibrated cell of every district, one row per target.

    Zero-support districts appear too — with null estimates and
    ``district_status == "zero_support"`` — so "every district" means every
    district with facts, not every district the solve could reach.
    """
    frames = [
        result.diagnostics.assign(district_status="solved")
        for chamber in chambers
        for result in chamber.district_results
    ]
    for fact_rows in (zero_support_facts or {}).values():
        if fact_rows is None or len(fact_rows) == 0:
            continue
        frame = fact_rows[
            ["area_type", "area_code", "metric", "entity", "concept", "value"]
        ].copy()
        frame = frame.rename(columns={"value": "target"})
        frame["initial_estimate"] = np.nan
        frame["final_estimate"] = np.nan
        frame["relative_error"] = np.nan
        frame["abs_relative_error"] = np.nan
        frame["district_status"] = "zero_support"
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The weight-0.5 crossing of the sorted value distribution."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError("values and weights must be one-dimensional.")
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("values and weights must align and be non-empty.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("values and weights must be finite.")
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
            # The mapping covers household-universe rows only; rows outside
            # it (group-quarters placeholders) are not part of the B19013
            # universe and carry no weight in the comparison.
            pairs = [
                (float(money_income_by_household_id[household]), float(weight))
                for household, weight in zip(
                    result.problem.household_ids,
                    result.weights,
                    strict=True,
                )
                if household in money_income_by_household_id
            ]
            if not pairs:
                continue
            incomes = np.array([income for income, _ in pairs])
            achieved = weighted_median(
                incomes,
                np.array([weight for _, weight in pairs]),
            )
            published_value = by_area[key]
            gap = achieved - published_value
            relative_gap = gap / published_value if published_value else np.inf
            rows.append(
                {
                    "area_type": result.problem.area_type,
                    "area_code": result.problem.area_code,
                    "n_household_rows": len(pairs),
                    "n_excluded_rows": int(
                        len(result.problem.household_ids) - len(pairs)
                    ),
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
    zero_support_facts: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Per (area_type, state, metric): targets vs artifact vs solved sums.

    ``zero_support_facts`` maps area_type to the calibration-fact rows of
    districts that had no assigned households; their target mass appears in
    ``zero_support_target_sum`` so the statewide picture never silently
    shrinks to the solvable subset.
    """
    rows: list[dict[str, Any]] = []
    zero_support = zero_support_facts if zero_support_facts is not None else {}
    for chamber in chambers:
        accumulator: dict[tuple[str, str], dict[str, float]] = {}
        for result in chamber.district_results:
            state = result.problem.state_fips
            # The artifact column is the TRUE base-weight mass — never the
            # floored optimizer anchor.
            artifact = result.problem.matrix @ np.asarray(
                result.problem.base_weights, dtype=np.float64
            )
            solved = result.problem.matrix @ result.weights
            for index, metric in enumerate(
                result.problem.target_frame["metric"].astype(str)
            ):
                cell = accumulator.setdefault(
                    (state, metric),
                    {
                        "target": 0.0,
                        "artifact": 0.0,
                        "solved": 0.0,
                        "zero_support_target": 0.0,
                        "n_zero_support": 0,
                    },
                )
                cell["target"] += float(result.problem.targets[index])
                cell["artifact"] += float(artifact[index])
                cell["solved"] += float(solved[index])
        for row in zero_support.get(chamber.area_type, pd.DataFrame()).itertuples():
            cell = accumulator.setdefault(
                (str(row.state_fips), str(row.metric)),
                {
                    "target": 0.0,
                    "artifact": 0.0,
                    "solved": 0.0,
                    "zero_support_target": 0.0,
                    "n_zero_support": 0,
                },
            )
            cell["zero_support_target"] += float(row.value)
            cell["n_zero_support"] += 1
        for (state, metric), sums in sorted(accumulator.items()):
            rows.append(
                {
                    "area_type": chamber.area_type,
                    "state_fips": state,
                    "metric": metric,
                    "district_target_sum": sums["target"],
                    "zero_support_target_sum": sums["zero_support_target"],
                    "n_zero_support_targets": sums["n_zero_support"],
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
    household_universe_record: Mapping[str, Any] | None = None,
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
                "block-overlap draws for ACS-spine rows conditional on "
                "(PUMA, congressional district, county) where that cell has "
                "block support, degrading through declared coarser "
                "conditioning (PUMA+county, PUMA+CD, PUMA) where it does "
                "not; the realized method mix below is the honest record. "
                "Rows whose certified tract is absent from the ladder also "
                "degrade to the seeded path and are counted."
            ),
            "method_counts": dict(membership_gate_details.get("method_counts", {})),
            "unassigned_share": membership_gate_details.get("unassigned_share"),
            "unassigned_weight_share": membership_gate_details.get(
                "unassigned_weight_share"
            ),
            "tract_degraded_rows": membership_gate_details.get("tract_degraded_rows"),
        },
        "income_instrument": {
            "statement": (
                "Income brackets bind on a declared ACS money-income analog "
                "built from artifact input columns (members aged 15 and "
                "over); the published median household income (B19013) is "
                "validation-only — a linear reweighting operator cannot "
                "honestly target a median."
            ),
            "recipe": recipe_resolution.as_record(),
        },
        "household_universe": dict(household_universe_record or {}),
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
    unassigned = statement["membership_assignment"].get("unassigned_share")
    weight_share = statement["membership_assignment"].get("unassigned_weight_share")
    degraded = statement["membership_assignment"].get("tract_degraded_rows")
    if unassigned is not None:
        lines.append(f"- unassigned row share: {unassigned:.4%}")
    if weight_share is not None:
        lines.append(f"- unassigned weight share: {weight_share:.4%}")
    if degraded is not None:
        lines.append(f"- certified-tract rows degraded to seeded draws: {degraded:,}")
    universe = statement.get("household_universe") or {}
    if universe:
        lines += ["", "## Household universe", ""]
        for key, value in sorted(universe.items()):
            lines.append(f"- {key}: {value}")
    lines += ["", "## Income instrument", ""]
    lines.append(statement["income_instrument"]["statement"])
    recipe = statement["income_instrument"]["recipe"]
    lines += ["", "Declared omissions:"]
    for name, reason in sorted(recipe["declared_omissions"].items()):
        lines.append(f"- {name}: {reason}")
    lines += ["", "Declared exclusions:"]
    for name, reason in sorted(recipe["declared_exclusions"].items()):
        lines.append(f"- {name}: {reason}")
    if recipe.get("absent_columns"):
        lines.append(
            "- artifact columns absent from the recipe resolution: "
            + ", ".join(recipe["absent_columns"])
        )
    lines += ["", "## Doctrine", ""]
    for key, value in sorted(statement["doctrine"].items()):
        lines.append(f"- {key}: {value}")
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
    for area_type, ess in sorted(tails["min_effective_sample_size"].items()):
        lines.append(f"- minimum effective sample size ({area_type}): {ess:,.1f}")
    for area_type, codes in sorted(tails["zero_support_districts"].items()):
        if codes:
            lines.append(f"- zero-support districts ({area_type}): {', '.join(codes)}")
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
    household_universe_record: Mapping[str, Any] | None = None,
    environment_record: Mapping[str, Any] | None = None,
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

    zero_support_facts: dict[str, pd.DataFrame] = {}
    for area_type, codes in zero_support_districts.items():
        if codes and not facts.calibration.empty:
            zero_support_facts[area_type] = facts.calibration[
                (facts.calibration["area_type"] == area_type)
                & (facts.calibration["area_code"].isin(list(codes)))
            ]

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
        achieved_vs_target_table(chambers, zero_support_facts)
        .to_csv(index=False)
        .encode("utf-8"),
        mtime=0,
    )
    (out / "sld_achieved_vs_target.csv.gz").write_bytes(achieved_bytes)
    shas["sld_achieved_vs_target.csv.gz"] = _sha256_bytes(achieved_bytes)

    median_table = median_income_validation(
        chambers,
        facts,
        money_income_by_household_id,
    )
    coherence = statewide_coherence_report(chambers, zero_support_facts)
    diagnostics = {
        "schema_version": 1,
        "doctrine": dict(doctrine_record),
        "environment": dict(environment_record or {}),
        "membership_gate": dict(membership_gate_details),
        "household_universe": dict(household_universe_record or {}),
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
        household_universe_record=household_universe_record,
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
