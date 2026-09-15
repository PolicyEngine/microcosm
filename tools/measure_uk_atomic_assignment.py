#!/usr/bin/env python3
"""Measure the UK geography assignment before and after the atomic operators.

Runs the canonical full-build preparation (``full_build_cli.prepare_full_build``)
on a bound spine checkpoint and executes the one declared graph through
``uk.full.geography_gate`` for every cell of the grid
{legacy sequential ladder draw, identity-keyed single-stage draw} x {K}; the
same pool sample and expansion nodes serve both laws at a given K. Per cell it
records, at design weights and with no solve:

* analytic expected rows per constituency and local authority (the legacy
  two-stage expectation from ``expected_uk_ladder_area_support``; the
  single-stage expectation from census household shares within FRS region),
* realized support on the ladder roster (rows, Kish ESS, distinct source
  households) through ``uk_ladder_area_support_summary`` and the 50-floor
  breach tables from ``size_evaluation.area_support_tables``,
* binomial z-scores of realized against expected rows, the region mix, the UK
  distribution gate verdict and London share,
* identity stability under the keyed law: the production columns equal an
  in-process re-draw on the same table, on row-reversed rows and on the
  clone-0 subset, and the K=1 draws are a subset of the K=15 draws,
* wall time per geography node and peak RSS.

It is a diagnostic recorder only: it never gates and release builds do not
run it. Disclosure (UKDS EUL CD137 clause 8, CD171 s5.2.1): the evidence JSON
carries no unit records and reports any per-area count below
``--minimum-count`` as "<n".

Example:
    uv run python tools/measure_uk_atomic_assignment.py \
        --input-h5 spine.h5 --ladder build/uk/uk_oa_ladder_2021.npz \
        --supports-dir build/uk/supports --ledger-facts <chronicle dir> \
        --sample-fraction 0.01 --n-clones 1,15 --laws legacy,keyed \
        --graph-store /tmp/uk-step2-store --evidence-out cells.json \
        --receipt-out receipt.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime import full_build_cli as cli
from microcosm.build.uk_runtime.atomic_area_support import (
    ASSIGNMENT_COLUMNS,
    IDENTITY_COLUMN,
    SOURCES,
    SYSTEMS,
)
from microcosm.build.uk_runtime.geography_ladder import (
    UK_GEOGRAPHY_LADDER_COLUMNS,
    expected_uk_ladder_area_support,
    load_uk_oa_ladder,
    uk_region_mix,
)
from microcosm.build.uk_runtime.local_rowwise import uk_ladder_area_support_summary
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.build.uk_runtime.size_evaluation import area_support_tables
from microcosm.graph import ContentStore, compile_graph, run_graph

_REPOSITORY = Path(__file__).resolve().parents[1]
_UK = _REPOSITORY / "packages/microcosm-build/src/microcosm/build/uk"
LAWS = ("legacy", "keyed")
LEVELS = (("constituency", "constituency_code"), ("la", "local_authority_code"))
GEOGRAPHY_NODES = (
    "uk.full.identity",
    "uk.full.geography.assign",
    "uk.full.geography.derive",
    "uk.full.geography.gate",
    "uk.full.locations",
    "uk.full.geography_mapping",
    "uk.full.geography_gate",
)
SUPPORT_LABELS = dict(zip(SYSTEMS, ("ew", "scotland", "ni"), strict=True))
CLONE = ladder_clone_index_column("household")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--input-sidecar", type=Path)
    parser.add_argument("--input-spine-gates", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--ladder-sha256")
    parser.add_argument(
        "--supports-dir",
        type=Path,
        required=True,
        help="Directory holding <system>_support.npz for the three UK systems.",
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=_UK / "uk_atomic_area_supports.provenance.json",
        help="Committed support register; its sha256 pins are enforced unless --no-pin-supports.",
    )
    parser.add_argument("--no-pin-supports", action="store_true")
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=578)
    parser.add_argument("--n-clones", default="1,15", help="Comma-separated K values.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-constituency-vintage", default="2024_pcon")
    parser.add_argument("--laws", default=",".join(LAWS))
    parser.add_argument("--minimum-count", type=int, default=3)
    parser.add_argument("--graph-store", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, help="Scratch output directory for preparation."
    )
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--calibration-year", type=int)
    args = parser.parse_args(argv)
    args.n_clones = tuple(int(v) for v in str(args.n_clones).split(","))
    args.laws = tuple(str(args.laws).split(","))
    if set(args.laws) - set(LAWS) or not args.laws:
        parser.error(f"--laws must be a subset of {LAWS}.")
    if args.minimum_count < 1:
        parser.error("--minimum-count must be positive.")
    return args


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _support_arguments(args: argparse.Namespace) -> list[str]:
    pins = {}
    if not args.no_pin_supports:
        register = json.loads(args.provenance_json.read_text(encoding="utf-8"))
        pins = {system: register["supports"][system]["sha256"] for system in SYSTEMS}
    arguments = []
    for system in SYSTEMS:
        label = SUPPORT_LABELS[system]
        arguments += [
            f"--atomic-support-{label}",
            str(args.supports_dir / f"{SOURCES[system]}.npz"),
        ]
        if system in pins:
            arguments += [f"--atomic-support-sha256-{label}", pins[system]]
    return arguments


def _build_arguments(args: argparse.Namespace, *, law: str, k: int) -> list[str]:
    arguments = [
        "--input-h5",
        str(args.input_h5),
        "--ladder",
        str(args.ladder),
        "--ledger-facts",
        str(args.ledger_facts),
        "--out",
        str(args.out or args.graph_store / "out"),
        "--n-clones",
        str(k),
        "--seed",
        str(args.seed),
        "--sample-fraction",
        str(args.sample_fraction),
        "--sample-seed",
        str(args.sample_seed),
        "--expected-constituency-vintage",
        args.expected_constituency_vintage,
        "--graph-store",
        str(args.graph_store),
    ]
    for flag, value in (
        ("--input-sidecar", args.input_sidecar),
        ("--input-spine-gates", args.input_spine_gates),
        ("--input-sha256", args.input_sha256),
        ("--ladder-sha256", args.ladder_sha256),
        ("--calibration-year", args.calibration_year),
    ):
        if value is not None:
            arguments += [flag, str(value)]
    if law == "legacy":
        arguments += ["--geography-assignment", "legacy"]
    else:
        arguments += ["--geography-assignment", "atomic", *_support_arguments(args)]
    return arguments


def _through_gates(graph):
    """The ancestor-closed subgraph ending at both geography gates.

    The shared ``uk.full.geography.gate`` is consumed by ``uk.full.pool``, not
    by the UK distribution gate, so ``_through`` on the UK gate alone would
    leave it out; both verdicts and both wall times are measured here.
    """
    endpoints = [
        node.id
        for node in graph.nodes
        if node.id in {"uk.full.geography_gate", "uk.full.geography.gate"}
    ]
    needed = set()
    for endpoint in endpoints:
        needed.update(node.id for node in cli._through(graph, endpoint).nodes)
    from dataclasses import replace

    return replace(graph, nodes=tuple(n for n in graph.nodes if n.id in needed))


def run_cell(args: argparse.Namespace, *, law: str, k: int):
    """Prepare and execute the production graph through the UK geography gate."""
    prepared = cli.prepare_full_build(
        cli.parse_args(_build_arguments(args, law=law, k=k))
    )
    graph = _through_gates(prepared.full.graph)
    started = time.perf_counter()
    manifest = run_graph(
        compile_graph(graph),
        sources=prepared.sources,
        store=ContentStore(args.graph_store),
        kernels=prepared.kernels,
        resume="auto",
    )
    wall = time.perf_counter() - started
    population = manifest.population("uk.full.expand")
    household = population.table("household").copy()
    household["household_weight"] = population.weights_for("household").values
    return prepared, manifest, household, wall


# --- expectations -----------------------------------------------------------


def _region_codes(household: pd.DataFrame) -> pd.Series:
    return household["region"].astype(str).map(FRS_REGION_TO_REGION_CODE)


def expected_single_stage_area_support(household: pd.DataFrame, ladder) -> pd.DataFrame:
    """Expected rows per constituency and LA under one household-count draw.

    Each pool row draws its atomic area with probability proportional to census
    households within its FRS region, so an area's expectation is the region's
    row count times the area's share of the region's census households.
    """
    frame = pd.DataFrame(
        {
            "region_code": ladder.region_code.astype(str),
            "households": ladder.households.astype(float),
            "constituency_code": ladder.constituency_code.astype(str),
            "local_authority_code": ladder.local_authority_code.astype(str),
        }
    )
    region_rows = _region_codes(household).value_counts()
    region_mass = frame.groupby("region_code")["households"].sum()
    rows = []
    for area_type, column in LEVELS:
        mass = frame.groupby(["region_code", column])["households"].sum()
        expected = {str(code): 0.0 for code in np.unique(getattr(ladder, column))}
        for (region_code, area_code), households in mass.items():
            n_region = float(region_rows.get(region_code, 0))
            if n_region == 0 or region_mass[region_code] == 0:
                continue
            expected[str(area_code)] += n_region * households / region_mass[region_code]
        rows += [
            {"area_type": area_type, "area_code": code, "expected_rows": value}
            for code, value in expected.items()
        ]
    return pd.DataFrame(rows, columns=("area_type", "area_code", "expected_rows"))


def expected_area_support(law: str, household: pd.DataFrame, ladder) -> pd.DataFrame:
    if law == "legacy":
        return expected_uk_ladder_area_support(household, ladder, n_clones=1)
    return expected_single_stage_area_support(household, ladder)


def _area_regions(ladder) -> dict[str, dict[str, str]]:
    """The FRS region every constituency / LA lives in (modal over its OAs)."""
    frame = pd.DataFrame(
        {
            "region_code": ladder.region_code.astype(str),
            "constituency_code": ladder.constituency_code.astype(str),
            "local_authority_code": ladder.local_authority_code.astype(str),
        }
    )
    return {
        column: frame.groupby(column)["region_code"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
        for _, column in LEVELS
    }


# --- realized support -------------------------------------------------------


def realized_area_support(household: pd.DataFrame, ladder) -> dict[str, pd.DataFrame]:
    return uk_ladder_area_support_summary(
        household,
        ladder,
        weight_column="household_weight",
        source_column="source_household_id",
    )


def breach_tables(summaries: dict[str, pd.DataFrame], exclusions: dict) -> dict:
    support = pd.concat(
        [
            frame.assign(geography_level=level)[
                [
                    "geography_level",
                    "area_code",
                    "assigned_households",
                    "effective_sample_size",
                    "nonzero_source_households",
                ]
            ]
            for level, frame in summaries.items()
        ],
        ignore_index=True,
    )
    run = SimpleNamespace(
        area_support=support,
        gates={
            "gates": {
                "uk_local_area_support": {
                    "details": {"reviewed_exclusions": exclusions}
                }
            }
        },
    )
    return area_support_tables(run)


def _cell_rows(
    expected: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    household: pd.DataFrame,
    area_regions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    region_rows = _region_codes(household).value_counts()
    rows = []
    for area_type, column in LEVELS:
        exp = expected[expected["area_type"] == area_type].set_index("area_code")[
            "expected_rows"
        ]
        summary = summaries[area_type].set_index("area_code")
        for code, expected_rows in exp.items():
            observed = float(summary["assigned_households"].get(code, 0.0))
            region = area_regions[column].get(str(code))
            n_region = float(region_rows.get(region, 0.0))
            share = expected_rows / n_region if n_region > 0 else 0.0
            variance = expected_rows * (1.0 - share)
            z = (
                (observed - expected_rows) / math.sqrt(variance)
                if variance > 0
                else None
            )
            rows.append(
                {
                    "level": area_type,
                    "area_code": str(code),
                    "region_code": region,
                    "expected_rows": float(expected_rows),
                    "rows": observed,
                    "ess": float(summary["effective_sample_size"].get(code, 0.0)),
                    "sources": float(
                        summary["nonzero_source_households"].get(code, 0.0)
                    ),
                    "z": z,
                }
            )
    return pd.DataFrame(rows)


def _level_summary(rows: pd.DataFrame, level: str) -> dict:
    sub = rows[rows["level"] == level]
    z = sub.loc[sub["expected_rows"] >= 5.0, "z"].dropna()
    big = sub[sub["expected_rows"] >= 100.0]
    relative = (
        ((big["rows"] - big["expected_rows"]).abs() / big["expected_rows"])
        if len(big)
        else pd.Series(dtype=float)
    )
    return {
        "areas": int(len(sub)),
        "min_rows": float(sub["rows"].min()) if len(sub) else None,
        "min_ess": float(sub["ess"].min()) if len(sub) else None,
        "min_sources": float(sub["sources"].min()) if len(sub) else None,
        "max_abs_z": float(z.abs().max()) if len(z) else None,
        "share_abs_z_gt_3": float((z.abs() > 3).mean()) if len(z) else None,
        "areas_with_expected_ge_5": int(len(z)),
        "max_relative_error_expected_ge_100": float(relative.max())
        if len(relative)
        else None,
        "areas_with_expected_ge_100": int(len(big)),
    }


# --- identity stability -----------------------------------------------------


def _definition_and_supports(prepared, manifest, store: ContentStore):
    node = prepared.full.graph.node("uk.full.geography.assign")
    definition = json.loads(node.params["definition"])
    supports = {}
    for index, system in enumerate(SYSTEMS):
        receipt = manifest.nodes[f"uk.full.geography.support.{index}"]
        payload = store.load_bytes(receipt.opaque_artifacts["support"])
        supports[system] = geo.decode_atomic_support(payload)
    return definition, supports


def redraw(household: pd.DataFrame, definition: dict, supports: dict) -> pd.DataFrame:
    table = household[["household_id", "region", IDENTITY_COLUMN]].copy()
    assigned = pd.concat(
        [table, geo.assign_atomic(table, definition, supports)], axis=1
    )
    derived = geo.derive_geography(assigned, definition, supports)
    return pd.concat([assigned, derived], axis=1).set_index("household_id")


def identity_stability(
    household: pd.DataFrame, definition: dict, supports: dict
) -> dict:
    """Production columns equal in-process draws on the same, reversed and pruned rows."""
    columns = [*ASSIGNMENT_COLUMNS, *UK_GEOGRAPHY_LADDER_COLUMNS]
    production = household.set_index("household_id")
    same = redraw(household, definition, supports)
    reversed_rows = redraw(household.iloc[::-1], definition, supports).loc[
        production.index
    ]
    subset = household[household[CLONE] == 0]
    pruned = redraw(subset, definition, supports)

    def equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
        return all(
            left[column]
            .astype("string")
            .equals(right.loc[left.index, column].astype("string"))
            for column in columns
        )

    return {
        "production_equals_in_process": equal(production, same),
        "reversed_rows_equal": equal(production, reversed_rows),
        "clone_zero_subset_equal": equal(pruned, production),
        "keys_unique": bool(household[IDENTITY_COLUMN].is_unique),
    }


# --- driver -----------------------------------------------------------------


def _suppress(value: float, minimum: int) -> float | str:
    return f"<{minimum}" if 0 < value < minimum else value


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ladder = load_uk_oa_ladder(args.ladder)
    area_regions = _area_regions(ladder)
    exclusions = json.loads((_UK / "local_area_support_exclusions.json").read_text())[
        "exclusions"
    ]
    register = json.loads(args.provenance_json.read_text(encoding="utf-8"))
    cells = []
    receipts = []
    keyed_draws: dict[int, dict[str, str]] = {}
    model_period = args.calibration_year
    for k in args.n_clones:
        for law in args.laws:
            _log(f"cell law={law} K={k} f={args.sample_fraction}")
            prepared, manifest, household, wall = run_cell(args, law=law, k=k)
            model_period = int(prepared.full.config.calibration_year)
            expected = expected_area_support(law, household, ladder)
            summaries = realized_area_support(household, ladder)
            rows = _cell_rows(expected, summaries, household, area_regions)
            breaches = breach_tables(summaries, exclusions)
            mix = uk_region_mix(household)
            gate = manifest.nodes["uk.full.geography_gate"].receipt
            shared_gate = manifest.nodes.get("uk.full.geography.gate")
            timings = {
                node: float(manifest.nodes[node].wall_time)
                for node in GEOGRAPHY_NODES
                if node in manifest.nodes
            }
            stability = None
            if law == "keyed":
                definition, supports = _definition_and_supports(
                    prepared, manifest, ContentStore(args.graph_store)
                )
                stability = identity_stability(household, definition, supports)
                keyed_draws[k] = dict(
                    zip(
                        household[IDENTITY_COLUMN],
                        household["atomic_area_code"],
                        strict=True,
                    )
                )
            summary = {
                "law": law,
                "n_clones": k,
                "rows": int(len(household)),
                "source_households": int(household["source_household_id"].nunique()),
                "wall_seconds": wall,
                "node_wall_seconds": timings,
                "peak_rss_bytes": _peak_rss_bytes(),
                "gate_outcome": gate["outcome"],
                "shared_gate_outcome": None
                if shared_gate is None
                else shared_gate.receipt.get("outcome"),
                "london_weighted_household_share": gate["evidence"]["details"].get(
                    "london_weighted_household_share"
                ),
                "levels": {level: _level_summary(rows, level) for level, _ in LEVELS},
                "breaches": breaches["by_geography_level"],
                "region_mix": mix.to_dict("records"),
                "identity_stability": stability,
                "geography_binding": prepared.bindings["geography"],
            }
            receipts.append(summary)
            cells.append(
                {
                    **{
                        k_: v
                        for k_, v in summary.items()
                        if k_ not in {"region_mix", "geography_binding"}
                    },
                    "areas": [
                        {
                            **row,
                            "rows": _suppress(row["rows"], args.minimum_count),
                            "ess": _suppress(row["ess"], args.minimum_count),
                            "sources": _suppress(row["sources"], args.minimum_count),
                        }
                        for row in rows.to_dict("records")
                    ],
                }
            )
    if len(keyed_draws) >= 2:
        low, high = min(keyed_draws), max(keyed_draws)
        nested = set(keyed_draws[low]) <= set(keyed_draws[high]) and all(
            keyed_draws[high][key] == area for key, area in keyed_draws[low].items()
        )
        for summary in receipts:
            if summary["law"] == "keyed":
                summary["k_growth_draws_nested"] = {
                    "low": low,
                    "high": high,
                    "nested": nested,
                }
    evidence = {
        "purpose": (
            "Assignment-level before/after evidence for the UK geography-first "
            "reordering (step 2): legacy sequential ladder draw versus the "
            "identity-keyed single-stage draw, design weights, no solve; per-area "
            "counts below the minimum are suppressed and no unit records appear."
        ),
        "date": dt.date.today().isoformat(),
        "model_period": model_period,
        "pool": {
            "spine": args.input_h5.name,
            "spine_sha256": _sha256(args.input_h5),
            "sample_fraction": args.sample_fraction,
            "sample_seed": args.sample_seed,
            "seed": args.seed,
        },
        "ladder": {"path": args.ladder.name, "sha256": _sha256(args.ladder)},
        "supports": {
            system: register["supports"][system]["sha256"] for system in SYSTEMS
        },
        "laws": list(args.laws),
        "n_clones": list(args.n_clones),
        "minimum_count": args.minimum_count,
        "cells": cells,
    }
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.receipt_out is not None:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(
            json.dumps(
                {"evidence_sha256": _sha256(args.evidence_out), "cells": receipts},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    for summary in receipts:
        levels = summary["levels"]
        _log(
            f"  {summary['law']:6s} K={summary['n_clones']:<2d} rows={summary['rows']:,} "
            f"wall={summary['wall_seconds']:.0f}s gate={summary['gate_outcome']} "
            f"const(min rows/ess/src)={levels['constituency']['min_rows']}/{levels['constituency']['min_ess']:.1f}/{levels['constituency']['min_sources']} "
            f"la={levels['la']['min_rows']}/{levels['la']['min_ess']:.1f}/{levels['la']['min_sources']} "
            f"maxz={levels['constituency']['max_abs_z']}"
        )


if __name__ == "__main__":
    main()
