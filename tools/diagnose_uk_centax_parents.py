#!/usr/bin/env python
"""Fit and support of the CenTax parent cells on a Microcosm UK national H5.

Computes, through the production measure resolver at the calibration year, each
cell of ``uk/centax_childcare_parent_facts.json``: the model's weighted parents,
the records and households behind them, the Kish effective number of households
and the share of the cell's weight on its ten heaviest households. With
``--facts`` it also reports each cell's target and relative error, and the
CenTax figures the resource keeps as diagnostics only (the Table A.1 levels and
the Figure 3.12 non-working-partner shares). ``parent_count`` is the number the
facts tool scales the CenTax shares by when this H5 is the baseline.

Writes aggregates only: no record-level values leave the H5.

    uv run --no-sync python tools/diagnose_uk_centax_parents.py \\
        --h5 microcosm_uk_2024_25.h5 --out diagnostics.json [--facts resource.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_ANI_BANDS_2022,
    CENTAX_ANI_MEASURE,
    CENTAX_EMPLOYEE_MAIN_SOURCE,
    CENTAX_PARENT_OF_ELIGIBLE_3_4,
    CENTAX_REGION_GROUPS,
    centax_ani_uprating,
    centax_cell_filters,
    centax_cell_ids,
    load_uk_centax_childcare_parent_facts,
    model_ani_edges,
)
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.weight_year import uk_engine_year_frame

#: The nominal ANI the free hours and Tax-Free Childcare taper away above.
_CLIFF_GBP = 100_000.0
_TOP_HOUSEHOLDS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _support(
    mask: np.ndarray, person_weight: np.ndarray, person_household: np.ndarray
) -> dict[str, Any]:
    """Weighted parents and the household support behind them."""

    households, members = np.unique(person_household[mask], return_counts=True)
    count = float(person_weight[mask].sum())
    if not len(households):
        return {"model_count": 0.0, "person_records": 0, "households": 0}
    weight_by_household = dict(
        zip(person_household[mask], person_weight[mask], strict=True)
    )
    household_weight = np.asarray(
        [weight_by_household[household] for household in households], dtype=float
    )
    contribution = household_weight * members
    heaviest = np.sort(contribution)[::-1][:_TOP_HOUSEHOLDS]
    return {
        "model_count": count,
        "person_records": int(mask.sum()),
        "households": int(len(households)),
        "kish_households": float(
            household_weight.sum() ** 2 / (household_weight**2).sum()
        ),
        "kish_contributions": float(contribution.sum() ** 2 / (contribution**2).sum()),
        "top10_household_share": float(heaviest.sum() / contribution.sum()),
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
    args = parser.parse_args(argv)

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
        work = person("employment_income_annual").astype(float) + person(
            "self_employment_income_after_loss_relief_annual"
        ).astype(float)
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
    person_benunit = person_table["person_benunit_id"].to_numpy()
    person_weight = np.asarray(
        [household_weight[household] for household in person_household], dtype=float
    )
    parent = columns[CENTAX_PARENT_OF_ELIGIBLE_3_4] == 1
    ani = columns[CENTAX_ANI_MEASURE]
    top = edges[CENTAX_ANI_BANDS_2022[-1][0]][1]

    cells: dict[str, Any] = {}
    for cell in centax_cell_ids():
        entry = _support(
            _mask(columns, centax_cell_filters(cell, edges[cell])),
            person_weight,
            person_household,
        )
        entry["model_ani_edges"] = edges[cell]
        if facts is not None:
            target = facts.value(cell)
            entry["target"] = target
            entry["relative_error"] = entry["model_count"] / target - 1.0
        cells[cell] = entry

    # Partners: the other adult of a two-adult benefit unit, non-working when
    # the year's pay and profit are nil, as Figure 3.12 counts them.
    partnered = np.zeros(len(parent), dtype=bool)
    partner_not_working = np.zeros(len(parent), dtype=bool)
    order = np.flatnonzero(parent)
    units: dict[Any, list[int]] = {}
    for index in order:
        units.setdefault(person_benunit[index], []).append(int(index))
    for members in units.values():
        if len(members) == 2:
            first, second = members
            partnered[members] = True
            partner_not_working[first] = work[second] <= 0
            partner_not_working[second] = work[first] <= 0
    split_edges = edges["female_ani_80k_to_150k"]
    in_split = parent & (ani > split_edges[0]) & (ani <= split_edges[1])
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

    result: dict[str, Any] = {
        "schema": "microcosm.uk.centax_parent_diagnostics.v1",
        "dataset": args.label or args.h5.name,
        "dataset_sha256": _sha256(args.h5),
        "year": args.year,
        "ani_uprating": uprating,
        "model_ani_edges": edges,
        "parent_count": float(person_weight[parent & (ani > 0) & (ani <= top)].sum()),
        "parents": {
            "all": _support(parent, person_weight, person_household),
            "ani_not_positive": _support(
                parent & (ani <= 0), person_weight, person_household
            ),
            "ani_above_top_edge": _support(
                parent & (ani > top), person_weight, person_household
            ),
            "annual_ani_over_100k": _support(
                parent & (ani > _CLIFF_GBP), person_weight, person_household
            ),
            "current_ani_over_100k": _support(
                parent & (current_ani > _CLIFF_GBP), person_weight, person_household
            ),
        },
        "cells": cells,
        "region_groups": {
            group: list(regions) for group, regions in CENTAX_REGION_GROUPS.items()
        },
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
        result["centax_diagnostic_comparators"] = facts.parameters.get(
            "diagnostic_comparators"
        )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"parent_count {result['parent_count']:,.0f}")
    for cell, entry in cells.items():
        line = (
            f"{cell:38s} {entry['model_count']:>11,.0f} rec {entry['person_records']:>5} "
            f"kish {entry.get('kish_households', 0.0):7.1f} "
            f"top10 {entry.get('top10_household_share', 0.0):5.2f}"
        )
        if "target" in entry:
            line += (
                f"  target {entry['target']:>11,.0f} err {entry['relative_error']:+.3f}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
