#!/usr/bin/env python
"""Fit and support of the CenTax parent cells on a Microcosm UK national H5.

Computes, through the production measure resolver at the calibration year, each
bound cell of ``uk/centax_childcare_parent_facts.json``: the model's weighted
parents, the records and households behind them, the Kish effective number of
households and the share of the cell's weight on its ten heaviest households,
each also over the FRS source households the Microcosm copies come from.
``parent_count`` (the parents in the aged (£50k, £200k]) is the number the facts
tool scales the CenTax shares by when this H5 is the baseline.

It also reports what the targets leave unbound (Vahid's reviews of
microcosm#1399):

- the parents at or below the aged £50k, and among them those with no
  employment, self-employment or private pension income in the year, whom
  HMRC's PAYE and Self Assessment data would not hold;
- the South excluding London and the North and Midlands, with the East of
  England in either group, since Figure 3.7 does not say where it sits;
- the parents whose main income source is not employment (Figure 3.3), a cell
  too thinly supported to bind (María's ruling of 24 November 2026).

With ``--facts`` it reports each bound cell's target and relative error, and
the level check: this H5's ``parent_count`` against the resource's, within
``--level-tolerance``. With ``--manifest`` and ``--workbook`` (the Chronicle
registration and a local copy of the workbook) it reads CenTax's own figures
for those unbound cells and the comparators kept out of the resource: Table
A.1's levels and Figure 3.12's non-working partners.

Writes aggregates only: no record-level values leave the H5. The output is a
local diagnostic: it carries support counts and concentration shares that are
not checked for publication.

    uv run --no-sync python tools/diagnose_uk_centax_parents.py \\
        --h5 microcosm_uk_2024_25.h5 --out diagnostics.json [--facts resource.json] \\
        [--manifest <chronicle manifest.yaml> --workbook <workbook.xlsx>]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_ANI_MEASURE,
    CENTAX_DIAGNOSTIC_BAND_2022,
    CENTAX_EMPLOYEE_MAIN_SOURCE,
    CENTAX_PARENT_OF_ELIGIBLE_3_4,
    CENTAX_SHARE_BASE_2022,
    CENTAX_SPLIT_BAND_2022,
    aged_edges,
    centax_ani_uprating,
    centax_cell_filters,
    centax_cell_ids,
    centax_diagnostic_cell_ids,
    centax_region_codes,
    load_uk_centax_childcare_parent_facts,
    model_ani_edges,
)
from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM

SCHEMA = "microcosm.uk.centax_parent_diagnostics.v2"
#: The nominal ANI the free hours and Tax-Free Childcare taper away above.
_CLIFF_GBP = 100_000.0
_TOP_HOUSEHOLDS = 10
#: The default level tolerance: a baseline whose parents in the share base
#: differ from the resource's by more than this needs the resource rebuilt.
DEFAULT_LEVEL_TOLERANCE = 0.02
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _builder():
    """The facts tool, for its registered-workbook reader."""

    path = _REPO_ROOT / "tools/build_uk_centax_childcare_parent_facts.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_centax_childcare_parent_facts", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _support(
    mask: np.ndarray,
    person_weight: np.ndarray,
    person_household: np.ndarray,
    person_source: np.ndarray,
) -> dict[str, Any]:
    """Weighted parents and the household support behind them.

    Microcosm copies an FRS household several times (the SPI, high-value-home
    and CGT copies), and the copies of one source household are not
    independent, so the support is also counted over source households: each
    one's contribution is the weight of all its copies' parents in the cell.
    """

    households, members = np.unique(person_household[mask], return_counts=True)
    count = float(person_weight[mask].sum())
    if not len(households):
        return {
            "model_count": 0.0,
            "person_records": 0,
            "households": 0,
            "source_households": 0,
        }
    weight_by_household = dict(
        zip(person_household[mask], person_weight[mask], strict=True)
    )
    household_weight = np.asarray(
        [weight_by_household[household] for household in households], dtype=float
    )
    contribution = household_weight * members
    heaviest = np.sort(contribution)[::-1][:_TOP_HOUSEHOLDS]
    sources, source_of_person = np.unique(person_source[mask], return_inverse=True)
    source_contribution = np.bincount(source_of_person, weights=person_weight[mask])
    heaviest_sources = np.sort(source_contribution)[::-1][:_TOP_HOUSEHOLDS]
    return {
        "model_count": count,
        "person_records": int(mask.sum()),
        "households": int(len(households)),
        "kish_households": float(
            household_weight.sum() ** 2 / (household_weight**2).sum()
        ),
        "kish_contributions": float(contribution.sum() ** 2 / (contribution**2).sum()),
        "top10_household_share": float(heaviest.sum() / contribution.sum()),
        "source_households": int(len(sources)),
        "kish_source_households": float(
            source_contribution.sum() ** 2 / (source_contribution**2).sum()
        ),
        "top10_source_share": float(heaviest_sources.sum() / source_contribution.sum()),
    }


def _mask(columns: Mapping[str, np.ndarray], filters: list[Mapping[str, Any]]):
    mask = np.ones(len(next(iter(columns.values()))), dtype=bool)
    for predicate in filters:
        values = columns[str(predicate["variable"])]
        if "in" in predicate:
            mask &= np.isin(values, predicate["in"])
        elif "equals" in predicate:
            mask &= values == predicate["equals"]
        elif predicate["operator"] == ">":
            mask &= values.astype(float) > float(predicate["value"])
        elif predicate["operator"] == "<=":
            mask &= values.astype(float) <= float(predicate["value"])
        else:
            raise ValueError(f"unexpected predicate {dict(predicate)!r}")
    return mask


def level_check(
    parent_count: float, facts_parent_count: float, tolerance: float
) -> dict[str, Any]:
    """This H5's parents in the share base against the resource's.

    The targets fix the bound bands' total at the resource's count, so a
    baseline that has moved from it pulls the fit by the drift (Vahid's
    review, item 2). Outside the tolerance, rebuild the resource on this H5.
    """

    if not facts_parent_count > 0 or not tolerance >= 0:
        raise ValueError("the level check needs a positive count and tolerance.")
    drift = parent_count / facts_parent_count - 1.0
    return {
        "parent_count": parent_count,
        "facts_parent_count": facts_parent_count,
        "drift": drift,
        "tolerance": tolerance,
        "within_tolerance": bool(abs(drift) <= tolerance),
    }


def workbook_comparators(workbook: Any) -> dict[str, Any]:
    """CenTax's figures for the unbound cells and the diagnostics-only tables."""

    builder = _builder()
    fig31 = builder.read_series(workbook, "Fig. 3.1")[
        "Number of Taxpayer in Income Bin"
    ]
    fig33 = builder.read_series(workbook, "Fig. 3.3")
    regions = builder.england_region_columns(workbook)
    lower, upper = CENTAX_DIAGNOSTIC_BAND_2022[1:]
    top = CENTAX_SHARE_BASE_2022[1]
    table = list(workbook["Tab. A.1"].iter_rows(values_only=True))
    parents = next(
        row
        for row in table
        if row[1] == "3- and 4-year-olds" and row[2] == "Number of Parents"
    )
    female = next(
        row
        for row in table
        if row[2] == "Share Female (%)" and table.index(row) < table.index(parents) + 5
    )
    fig312 = {
        row[1]: row[2:5]
        for row in workbook["Fig. 3.12"].iter_rows(values_only=True)
        if isinstance(row[1], (int, float)) and not isinstance(row[1], bool)
    }
    split_lower, split_upper = CENTAX_SPLIT_BAND_2022
    in_band = [v for label, v in fig312.items() if split_lower < label <= split_upper]
    return {
        "fig_3_1_share_of_0_to_200k_at_or_below_50k": (
            builder.band_sum(fig31, lower, upper) / builder.band_sum(fig31, 0.0, top)
        ),
        "fig_3_3_non_employee_share_within_80k_to_150k": builder.within_split_band(
            fig33, "Number of Taxpayers in Income Bin (Non-Employees)"
        ),
        "fig_3_7_shares_within_80k_to_150k": {
            group: builder.within_split_band(regions, group) for group in regions
        },
        "table_a1_3_4_year_olds": {
            "parents_whole_population": parents[3],
            "parents_80k_to_140k": parents[4],
            "share_female_whole_population_pct": female[3],
            "share_female_80k_to_140k_pct": female[4],
        },
        "fig_3_12_non_working_partner_share_80k_to_150k_unweighted_bin_mean": {
            # The figure gives shares, not counts, and its "Men & Women" column
            # repeats "Men"; the denominator is parents with an identified partner.
            "men": sum(values[1] for values in in_band) / len(in_band),
            "women": sum(values[2] for values in in_band) / len(in_band),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--facts",
        type=Path,
        default=None,
        help="The derived-facts resource to compare against (default: none).",
    )
    parser.add_argument("--label", default=None, help="Display name of the H5.")
    parser.add_argument(
        "--level-tolerance", type=float, default=DEFAULT_LEVEL_TOLERANCE
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workbook", type=Path, default=None)
    args = parser.parse_args(argv)
    if (args.manifest is None) != (args.workbook is None):
        parser.error("--manifest and --workbook go together.")

    # The resolver modules load the engine; import them only to measure.
    from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
    from microcosm.build.uk_runtime.support_floor import uk_household_source_ids
    from microcosm.build.uk_runtime.weight_year import uk_engine_year_frame

    uprating = centax_ani_uprating(args.year)
    edges = model_ani_edges(uprating)
    facts = (
        None
        if args.facts is None
        else load_uk_centax_childcare_parent_facts(args.facts)
    )
    if facts is not None and facts.parameters["model_ani_edges"] != edges:
        raise SystemExit(
            "the facts resource's edges are not this engine's at "
            f"{args.year}: {facts.parameters['model_ani_edges']} vs {edges}"
        )
    comparators = None
    if args.manifest is not None:
        workbook, registration, _licence, _manifest = (
            _builder().open_registered_workbook(args.manifest, args.workbook)
        )
        comparators = {
            "registration": registration,
            **workbook_comparators(workbook),
        }

    frame, _provenance = load_uk_national_frame(args.h5)
    frame = uk_engine_year_frame(frame, simulation_year=args.year)
    with tempfile.TemporaryDirectory() as scratch:
        resolver = UKMeasureResolver(
            simulation_source=args.h5,
            scratch_dir=Path(scratch),
            year=args.year,
            frame=frame,
        )

        def person(name: str) -> np.ndarray:
            return resolver.compute("person", name)[0]

        columns = {
            CENTAX_PARENT_OF_ELIGIBLE_3_4: person(CENTAX_PARENT_OF_ELIGIBLE_3_4),
            CENTAX_ANI_MEASURE: person(CENTAX_ANI_MEASURE).astype(float),
            CENTAX_EMPLOYEE_MAIN_SOURCE: person(CENTAX_EMPLOYEE_MAIN_SOURCE),
            "gender": person("gender").astype(str),
            "region": person("region").astype(str),
        }
        current_ani = person("adjusted_net_income").astype(float)
        work = person("employment_income_with_benefits_in_kind_annual").astype(
            float
        ) + person("self_employment_income_after_loss_relief_annual").astype(float)
        pension = person("private_pension_income_annual").astype(float)
        receipt = resolver.receipt()

    person_table = frame.table("person")
    household_table = frame.table("household")
    household_weight = dict(
        zip(
            household_table["household_id"].to_numpy(),
            np.asarray(frame.weights_for("household").values, dtype=float),
            strict=True,
        )
    )
    person_household = person_table["person_household_id"].to_numpy()
    source_of_household = dict(
        zip(
            household_table["household_id"].to_numpy(),
            uk_household_source_ids(frame),
            strict=True,
        )
    )
    person_source = np.asarray(
        [source_of_household[household] for household in person_household]
    )
    person_benunit = person_table["person_benunit_id"].to_numpy()
    person_weight = np.asarray(
        [household_weight[household] for household in person_household], dtype=float
    )

    def support(mask: np.ndarray) -> dict[str, Any]:
        return _support(mask, person_weight, person_household, person_source)

    parent = columns[CENTAX_PARENT_OF_ELIGIBLE_3_4] == 1
    ani = columns[CENTAX_ANI_MEASURE]
    base_lower, base_upper = aged_edges(*CENTAX_SHARE_BASE_2022, uprating)
    in_base = parent & (ani > base_lower) & (ani <= base_upper)
    below_base = parent & (ani > 0) & (ani <= base_lower)

    cells: dict[str, Any] = {}
    for cell in centax_cell_ids():
        entry = support(_mask(columns, centax_cell_filters(cell, edges[cell])))
        entry["model_ani_edges"] = edges[cell]
        if facts is not None:
            target = facts.value(cell)
            entry["target"] = target
            entry["relative_error"] = entry["model_count"] / target - 1.0
        cells[cell] = entry

    # The split cells the targets never bind, on the split band's edges.
    diagnostic_cells: dict[str, Any] = {}
    split_edges = aged_edges(*CENTAX_SPLIT_BAND_2022, uprating)
    for cell in centax_diagnostic_cell_ids():
        diagnostic_cells[cell] = {
            **support(_mask(columns, centax_cell_filters(cell, split_edges))),
            "model_ani_edges": split_edges,
        }

    # The region groups outside London, with the East of England in each.
    split_lower, split_upper = aged_edges(*CENTAX_SPLIT_BAND_2022, uprating)
    in_split = parent & (ani > split_lower) & (ani <= split_upper)
    region_groups: dict[str, Any] = {}
    for east_in in ("south_excl_london", "north_and_midlands"):
        codes = centax_region_codes(east_in=east_in)
        region_groups[f"east_in_{east_in}"] = {
            group: {
                "regions": list(group_codes),
                **support(
                    in_split
                    & np.isin(
                        columns["region"],
                        [UK_REGION_TIER_ENUM[code] for code in group_codes],
                    )
                ),
            }
            for group, group_codes in codes.items()
        }

    # Partners: the other adult of a two-adult benefit unit, non-working when
    # the year's pay and profit are nil, as Figure 3.12 counts them.
    partnered = np.zeros(len(parent), dtype=bool)
    partner_not_working = np.zeros(len(parent), dtype=bool)
    units: dict[Any, list[int]] = {}
    for index in np.flatnonzero(parent):
        units.setdefault(person_benunit[index], []).append(int(index))
    for members in units.values():
        if len(members) == 2:
            first, second = members
            partnered[members] = True
            partner_not_working[first] = work[second] <= 0
            partner_not_working[second] = work[first] <= 0
    partner_shares = {}
    for sex in ("MALE", "FEMALE"):
        base = in_split & partnered & (columns["gender"] == sex)
        partner_shares[sex.lower()] = {
            "partnered_weight": float(person_weight[base].sum()),
            "non_working_partner_share": float(
                person_weight[base & partner_not_working].sum()
                / max(person_weight[base].sum(), 1e-12)
            ),
            "person_records": int(base.sum()),
        }

    parent_count = float(person_weight[in_base].sum())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": args.label or args.h5.name,
        "dataset_sha256": _sha256(args.h5),
        "year": args.year,
        "ani_uprating": uprating,
        "model_ani_edges": edges,
        "parent_count_range": [base_lower, base_upper],
        "parent_count": parent_count,
        "parents": {
            "all": support(parent),
            "ani_not_positive": support(parent & (ani <= 0)),
            "share_base": support(in_base),
            "ani_0_to_50k": support(below_base),
            # Vahid's review, item 1: parents HMRC's PAYE and Self Assessment
            # data would not hold, with no earnings or private pension.
            "ani_0_to_50k_without_pay_profit_or_private_pension": support(
                below_base & (work <= 0) & (pension <= 0)
            ),
            "ani_above_top_edge": support(parent & (ani > base_upper)),
            "annual_ani_over_100k": support(parent & (ani > _CLIFF_GBP)),
            "current_ani_over_100k": support(parent & (current_ani > _CLIFF_GBP)),
        },
        "cells": cells,
        "diagnostic_cells": diagnostic_cells,
        "region_groups": region_groups,
        "partners_ani_80k_to_150k": partner_shares,
        "resolver": {
            "policyengine_uk_version": receipt.get("policyengine_uk_version"),
            "annual_income_measures": receipt.get("annual_income_measures", {}).get(
                "computed_measures"
            ),
        },
    }
    if facts is not None:
        result["facts_rows_sha256"] = facts.rows_sha256
        result["facts_baseline"] = dict(facts.baseline)
        result["level_check"] = level_check(
            parent_count, facts.parent_count, args.level_tolerance
        )
    if comparators is not None:
        result["centax_comparators"] = comparators
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"parent_count {parent_count:,.0f} in ({base_lower:,}, {base_upper:,}]")
    for cell, entry in cells.items():
        line = (
            f"{cell:38s} {entry['model_count']:>11,.0f} rec {entry['person_records']:>5} "
            f"kish {entry.get('kish_households', 0.0):7.1f} "
            f"sources {entry['source_households']:>4} "
            f"kish {entry.get('kish_source_households', 0.0):6.1f}"
        )
        if "target" in entry:
            line += (
                f"  target {entry['target']:>11,.0f} err {entry['relative_error']:+.3f}"
            )
        print(line)
    if "level_check" in result:
        check = result["level_check"]
        print(
            f"level drift {check['drift']:+.4f} (tolerance {check['tolerance']}): "
            f"{'within' if check['within_tolerance'] else 'OUTSIDE'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
