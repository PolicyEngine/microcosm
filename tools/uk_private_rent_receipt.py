"""Measure a UK local build's private rents and BRMAs (microcosm#1262).

``uk.full.geography.brma`` reads each placed Great Britain household's BRMA
off its assigned area, and ``uk.full.private_rent`` redraws each private
renter's rent from that BRMA's list of rents. This tool measures one build,
and a control build when one is given, so that the PR's receipts and signed
entries rest on measured figures rather than on a gate:

- **Rent rows.** The bound ``ons.rent.private_rent`` (local authority) and
  ``ons.rent.private_rent_brma`` (Scottish BRMA) rows at the problem's design
  weights, the dense solution and the size refit: how many are within 10% of
  their target, how many are more than 25% out, and the median error. A row
  enters a summary only when ten distinct source households of positive
  weight carry it at that weighting, as the build's BRMA rule counts them;
  the others are counted, never summarised, and a group is summarised only
  over ten or more published rows. A weighting whose households have no
  source keys (``--pool`` gives the design and dense ones) is not summarised
  at all.
- **Rents by region and bedrooms.** The mean monthly rent of private renters
  with a positive rent, by nation, region and LHA category (B to E).
- **PIPR by district and bedrooms.** ONS's calendar-2025 mean rents by local
  authority (Scottish BRMA) and bedrooms against the build's: rank
  correlation and error bands across cells.
- **BRMA mix.** Each region's private renters across its BRMAs, against the
  census counts the region scale assumes (total variation distance).
- **Weights and coherence.** The effective sample size and the top 1%'s
  share of weight among private renters, and the shares of private renters
  whose reported Housing Benefit, or whose subletting income, exceeds their
  rent.

The household figures are read at the release weights of the run's local H5.
With ``--pool``, a household table of the whole clone pool written by
``tools/uk_private_rent_pool.py``, they are also read at the problem's design
weights, and the spine's rents and BRMAs give a matched before and after.

Outputs go to ``--audit-dir`` (default ``<run>/audit-1262``), which stays with
the licensed run outside the repository:

- ``private_rent_audit.json`` keeps every row and cell, including district-
  and BRMA-level model figures, for review on this machine only;
- ``private_rent_summary.json`` keeps only aggregates that pass UKDS
  disclosure control (End User Licence CD137 clause 8, CD171 section 5.2.1).
  Every model cell is carried by at least ten distinct source households, or
  reads "<10" with a complementary cell in its region withheld too; a cell
  of spine values counts its FRS-channel households only, since an
  SPI-channel row's spine rent is drawn from FRS donors. A district or BRMA
  group is summarised only over at least ten cells, the top 1%'s weight
  share only over at least ten rows and households, and an unweighted share
  only where both sides rest on at least ten households. Record counts are
  rounded (``receipt_counts.rounded_count``), no figure describes a single
  district or BRMA, and no unit-record value appears. Only this file's
  figures may be quoted in a PR or a committed receipt.

Both files, and the pool table, are refused inside any git work tree.

``region_level_at_design_weights`` is a named check. The region scale holds
each region's FRS level on the census private-renter mix of its BRMAs, while
placement follows all households, so at design weights a region's mean rent
moves by the gap between the two mixes; calibration's tenure rows restore it.

Usage::

    uv run --no-sync python tools/uk_private_rent_receipt.py \\
        --run <runs/head> [--control <runs/control>] \\
        --ledger-facts .codex-work/uk-artifact [--pool <pool.npz>] \\
        [--control-pool <control-pool.npz>]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from microcosm.build.uk_runtime.local_targets import uk_brma_area_codes
from microcosm.build.uk_runtime.private_rent_localisation import (
    CATEGORIES,
    FRS_CHANNEL,
    NORTHERN_IRELAND,
    load_private_renter_counts,
    redraw_mask,
    rent_categories,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    outside_work_trees,
    rounded_count,
)

SUMMARY_SCHEMA_VERSION = 1

#: A model cell needs this many distinct source households to be reported.
MIN_SOURCE_HOUSEHOLDS = 10
#: A district or BRMA group's statistics need this many cells, so that no
#: summary describes one district (a one-cell median ratio times PIPR's
#: published mean gives the model's district mean back).
MIN_GROUP_CELLS = 10

RENT_TARGET_IDS = ("ons.rent.private_rent", "ons.rent.private_rent_brma")

REGION_NATION = {
    "NORTH_EAST": "England",
    "NORTH_WEST": "England",
    "YORKSHIRE": "England",
    "EAST_MIDLANDS": "England",
    "WEST_MIDLANDS": "England",
    "EAST_OF_ENGLAND": "England",
    "LONDON": "England",
    "SOUTH_EAST": "England",
    "SOUTH_WEST": "England",
    "WALES": "Wales",
    "SCOTLAND": "Scotland",
    NORTHERN_IRELAND: "Northern Ireland",
}
AREA_PREFIX_NATION = {"E": "England", "W": "Wales", "S": "Scotland"}

#: PIPR's bedroom breakdown and the LHA category of a home of that size.
PIPR_BEDROOMS_CATEGORY = {
    "one bed": "B",
    "two bed": "C",
    "three bed": "D",
    "four or more bed": "E",
}
PIPR_MEASURE = "ons_rent_private_rent"
PIPR_YEAR = "2025"
PIPR_LEVELS = {"local_authority": "la", "statistical_scope": "brma"}

#: The H5 columns the household measures read.
HOUSEHOLD_COLUMNS = (
    "household_id",
    "household_weight",
    "region",
    "tenure_type",
    "rent",
    "subrent",
    "num_bedrooms",
    "brma",
    "la_code_oa",
    "source_household_key",
    "household_support_channel",
)
#: The withheld note a summary gives a weighting it cannot floor.
NO_SOURCE_KEYS = "no source keys for this weighting's households"


# ----------------------------------------------------------------------------
# Disclosure control


def floored_share(numerator: int, denominator: int) -> float | str | None:
    """An unweighted share of records, floored to two decimals.

    An exact unweighted share gives back its count; a share of one to nine
    records, or one that leaves out one to nine, reads "<10"
    (``receipt_counts``).
    """

    if denominator <= 0:
        return None
    for count in (numerator, denominator - numerator):
        if 0 < count < MIN_SOURCE_HOUSEHOLDS:
            return "<10"
    return math.floor(100 * numerator / denominator) / 100


def _carriers(values: Iterable[object]) -> int:
    """Distinct source households, a blank key counting for none.

    The receipts' one rule (``receipt_counts.distinct_carriers``);
    :func:`with_carriers` has blanked the rows of zero weight.
    """

    return distinct_carriers(values)


def with_carriers(frame: pd.DataFrame, *, frs_only: bool = False) -> pd.DataFrame:
    """The frame with the source household each row's figures rest on.

    A row of zero weight adds nothing to a figure, so it carries none, as the
    build's BRMA rule counts. An SPI-channel row's spine rent is the SPI
    housing shell's draw from FRS donors, so a figure of spine values counts
    its FRS-channel households only; elsewhere every distinct source
    household counts.
    """

    carrier = frame["source_key"].astype(str).to_numpy(dtype=object)
    carrier = np.where(frame["weight"].to_numpy(dtype=float) > 0, carrier, "")
    if frs_only:
        channel = frame["channel"].astype(str).to_numpy()
        carrier = np.where(channel == FRS_CHANNEL, carrier, "")
    return frame.assign(carrier=carrier)


def weighted_share(
    mask: np.ndarray, weight: np.ndarray, carrier: np.ndarray
) -> float | str | None:
    """A weighted share, or "<10" where fewer than ten source households carry it.

    The complement is checked too: a share just short of one would otherwise
    describe the few households outside it. A side holding rows counts even
    when none of them has a counted carrier.
    """

    total = float(weight.sum())
    if total <= 0:
        return None
    for side in (mask, ~mask):
        if side.any() and _carriers(carrier[side]) < MIN_SOURCE_HOUSEHOLDS:
            return "<10"
    return round(float(weight[mask].sum()) / total, 4)


def unweighted_share(mask: np.ndarray, carrier: np.ndarray) -> float | str | None:
    """A floored share of rows whose two sides each rest on ten households.

    Clone rows are not households: 8 households with 25 clones each are 200
    rows but 8 carriers, so either side resting on one to nine distinct
    source households reads "<10".
    """

    mask = np.asarray(mask, dtype=bool)
    carrier = np.asarray(carrier, dtype=object)
    for side in (mask, ~mask):
        if side.any() and _carriers(carrier[side]) < MIN_SOURCE_HOUSEHOLDS:
            return "<10"
    return floored_share(int(mask.sum()), int(mask.size))


def suppress_cells(
    cells: dict[str, dict[str, Any]], *, carriers_key: str = "source_households"
) -> dict[str, dict[str, Any]]:
    """Withhold cells under the threshold, and complements beside them.

    A cell carried by fewer than ten distinct source households, none
    included, reads "<10". Where a parent's figure is also reported, the
    withheld cells could be recovered together by difference, so the
    smallest reportable cells under the same parent are withheld too until
    the withheld cells together rest on ten households. The cells partition
    their parent's households, so their carriers add.
    """

    out = {name: dict(cell) for name, cell in cells.items()}
    small = [
        name
        for name, cell in out.items()
        if int(cell[carriers_key]) < MIN_SOURCE_HOUSEHOLDS
    ]
    hidden = sum(int(out[name][carriers_key]) for name in small)
    for name in small:
        out[name] = {"withheld": "<10"}
    rest = sorted(
        (int(cell[carriers_key]), name)
        for name, cell in out.items()
        if "withheld" not in cell
    )
    while small and hidden < MIN_SOURCE_HOUSEHOLDS and rest:
        carriers, name = rest.pop(0)
        out[name] = {"withheld": "complement"}
        hidden += carriers
    for cell in out.values():
        if "withheld" not in cell:
            cell[carriers_key] = rounded_count(int(cell[carriers_key]))
    return out


# ----------------------------------------------------------------------------
# Rent rows at each weighting


@dataclass(frozen=True)
class Weighting:
    name: str
    matrix: Any
    weights: np.ndarray
    names: tuple[str, ...]
    targets: np.ndarray
    metadata: tuple[Mapping[str, Any], ...]
    #: Each household's source key on the problem's axis, "" for one of zero
    #: weight, or None when the households' keys are not at hand.
    carriers: np.ndarray | None = None


def carriers_on(
    entity_ids: Sequence[object],
    weights: np.ndarray,
    keys: pd.Series | None,
) -> np.ndarray | None:
    """The source key of each household on a problem's axis.

    ``keys`` maps household id to source key. A household of zero weight
    carries nothing; a positive-weight one without a key leaves the axis
    unkeyed (None), and the weighting is then not summarised.
    """

    if keys is None:
        return None
    if not keys.index.is_unique:
        raise ValueError("Two households share an id in the source keys.")
    carriers = keys.reindex(pd.Index(list(entity_ids))).to_numpy(dtype=object)
    missing = pd.isna(carriers)
    positive = np.asarray(weights, dtype=float) > 0
    if (missing & positive).any():
        return None
    return np.where(positive, carriers, "").astype(object)


def load_weightings(run: Path, *, keys: pd.Series | None = None) -> list[Weighting]:
    """The problem's design weights, the dense solution and the size refit.

    ``keys`` maps household id to source key, for the carrier rule.
    """

    from microcosm.calibrate.artifacts import decode_problem, decode_solution

    def weighting(name: str, problem, weights: np.ndarray) -> Weighting:
        weights = np.asarray(weights, dtype=float)
        return Weighting(
            name=name,
            matrix=problem.problem.matrix,
            weights=weights,
            names=tuple(problem.problem.names),
            targets=np.asarray(problem.problem.target_vector, dtype=float),
            metadata=tuple(problem.target_metadata),
            carriers=carriers_on(problem.entity_ids, weights, keys),
        )

    problem = decode_problem((run / "uk.full.problem.problem.artifact").read_bytes())
    out = [
        weighting(
            "design",
            problem,
            np.asarray(problem.problem.initial_weights.values, dtype=float),
        )
    ]
    dense = run / "uk.full.dense.solution.artifact"
    if dense.exists():
        # A solution binds the problem its node solved, not this artifact's
        # digest; the household axis must still be the problem's.
        solution = decode_solution(dense.read_bytes(), entity_ids=problem.entity_ids)
        out.append(weighting("dense", problem, solution.weights))
    refit_problem = run / "uk.full.size_refit.problem.artifact"
    refit_solution = run / "uk.full.size_refit.solution.artifact"
    if refit_problem.exists() and refit_solution.exists():
        refit = decode_problem(refit_problem.read_bytes())
        solution = decode_solution(
            refit_solution.read_bytes(), entity_ids=refit.entity_ids
        )
        out.append(weighting("release", refit, solution.weights))
    return out


def _area_nation(code: str) -> str:
    return AREA_PREFIX_NATION.get(str(code)[:1], "Other")


def _fit_summary(relative: np.ndarray, withheld: int = 0) -> dict[str, Any]:
    """A group's fit over its published rows.

    A group of fewer than ten published rows reads withheld, its counts
    only: a median over one or two rows is a row's own error, which the
    receipt's no-single-district-or-BRMA rule forbids (the third review of
    #1305).
    """

    error = np.abs(relative)
    if error.size < MIN_GROUP_CELLS:
        return {
            "withheld": f"fewer than {MIN_GROUP_CELLS} published rows",
            "rows": int(error.size),
            **({"rows_withheld": int(withheld)} if withheld else {}),
        }
    out = {
        "rows": int(error.size),
        "share_within_10pct": round(float(np.mean(error <= 0.10)), 3)
        if error.size
        else None,
        "rows_over_25pct": int(np.sum(error > 0.25)),
        "median_abs_error": round(float(np.median(error)), 4) if error.size else None,
    }
    if withheld:
        out["rows_withheld"] = int(withheld)
    return out


def row_carriers(matrix: Any, carriers: np.ndarray, index: int) -> int:
    """Distinct source households of positive weight with an entry in a row.

    ``matrix`` is the problem's CSR matrix and ``carriers`` its axis's keys.
    """

    span = slice(matrix.indptr[index], matrix.indptr[index + 1])
    columns = matrix.indices[span][matrix.data[span] != 0]
    return _carriers(carriers[columns])


def rent_row_fit(weighting: Weighting) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit of the bound rent rows: a summary by row family and nation, and rows.

    A row rests on the distinct source households of positive weight with an
    entry in it (the build's BRMA rule, microcosm#1262). One resting on fewer
    than ten stays out of every group of the summary, which counts it; the
    audit keeps every row.
    """

    rows = [
        index
        for index, meta in enumerate(weighting.metadata)
        if meta.get("contract_target_id") in RENT_TARGET_IDS
    ]
    if not rows:
        return {}, []
    estimate = np.asarray(weighting.matrix[rows] @ weighting.weights, dtype=float)
    target = weighting.targets[rows]
    relative = estimate / target - 1.0
    matrix = sparse.csr_array(weighting.matrix)
    families: dict[str, list[int]] = {}
    withheld: dict[str, int] = {}
    audit = []
    for position, index in enumerate(rows):
        meta = weighting.metadata[index]
        family = str(meta["contract_target_id"])
        nation = _area_nation(meta.get("area_code", ""))
        carriers = (
            None
            if weighting.carriers is None
            else row_carriers(matrix, weighting.carriers, index)
        )
        groups = (f"{family} {nation}", "all")
        for group in groups:
            families.setdefault(group, [])
        if carriers is not None and carriers < MIN_SOURCE_HOUSEHOLDS:
            for group in groups:
                withheld[group] = withheld.get(group, 0) + 1
        else:
            for group in groups:
                families[group].append(position)
        audit.append(
            {
                "target_name": weighting.names[index],
                "target": float(target[position]),
                "estimate": float(estimate[position]),
                "relative_error": float(relative[position]),
                "source_households": carriers,
            }
        )
    if weighting.carriers is None:
        return {"withheld": NO_SOURCE_KEYS}, audit
    summary = {
        group: _fit_summary(
            relative[np.asarray(positions, dtype=int)], withheld.get(group, 0)
        )
        for group, positions in sorted(families.items())
    }
    return summary, audit


# ----------------------------------------------------------------------------
# Households


def release_households(h5: Path) -> pd.DataFrame:
    """The H5's households, at release weights, with reported Housing Benefit."""

    household = pd.read_hdf(h5, "household", columns=list(HOUSEHOLD_COLUMNS))
    person = pd.read_hdf(
        h5, "person", columns=["person_household_id", "housing_benefit_reported"]
    )
    benefit = person.groupby("person_household_id")["housing_benefit_reported"].sum()
    frame = pd.DataFrame(
        {
            "household_id": household["household_id"].to_numpy(),
            "weight": household["household_weight"].to_numpy(dtype=float),
            "region": household["region"].astype(str).to_numpy(),
            "tenure_type": household["tenure_type"].astype(str).to_numpy(),
            "rent": household["rent"].to_numpy(dtype=float),
            "subrent": household["subrent"].to_numpy(dtype=float),
            "num_bedrooms": household["num_bedrooms"].to_numpy(dtype=float),
            "brma": household["brma"].astype(str).to_numpy(),
            "la_code": household["la_code_oa"].astype(str).to_numpy(),
            "source_key": household["source_household_key"].astype(str).to_numpy(),
            "channel": household["household_support_channel"].astype(str).to_numpy(),
            "housing_benefit_reported": household["household_id"]
            .map(benefit)
            .fillna(0.0)
            .to_numpy(dtype=float),
        }
    )
    return frame


def private_renters(frame: pd.DataFrame, *, rent: str = "rent") -> pd.DataFrame:
    """Great Britain private renters with a positive rent, with their category."""

    mask = redraw_mask(frame["region"], frame["tenure_type"], frame[rent])
    out = frame.loc[mask].copy()
    out["category"] = rent_categories(out["num_bedrooms"].to_numpy())
    out["nation"] = out["region"].map(REGION_NATION)
    return out


def _mean_cell(rows: pd.DataFrame, *, rent: str = "rent") -> dict[str, Any]:
    weight = rows["weight"].to_numpy(dtype=float)
    total = float(weight.sum())
    return {
        "source_households": _carriers(rows["carrier"]),
        "mean_monthly_rent": round(
            float((weight * rows[rent].to_numpy(dtype=float)).sum()) / total / 12.0, 2
        )
        if total > 0
        else None,
    }


def rents_by_cell(
    renters: pd.DataFrame, *, rent: str = "rent"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mean monthly rent by nation, region and category: summary and audit."""

    audit: dict[str, Any] = {"nation": {}, "region": {}}
    summary: dict[str, Any] = {"nation": {}, "region": {}}
    for nation, rows in renters.groupby("nation", sort=True):
        cell = _mean_cell(rows, rent=rent)
        audit["nation"][nation] = cell
        summary["nation"].update(suppress_cells({nation: cell}))
    for region, rows in renters.groupby("region", sort=True):
        cells = {"all": _mean_cell(rows, rent=rent)}
        for category, part in rows.groupby("category", sort=True):
            cells[str(category)] = _mean_cell(part, rent=rent)
        audit["region"][region] = cells
        categories = suppress_cells(
            {name: cell for name, cell in cells.items() if name != "all"}
        )
        summary["region"][region] = {
            **suppress_cells({"all": cells["all"]}),
            **categories,
        }
    return summary, audit


# ----------------------------------------------------------------------------
# PIPR by district and bedrooms


def _feed_path(path: Path) -> Path:
    return path / "consumer_facts.jsonl" if path.is_dir() else path


def _la_aliases() -> dict[str, str]:
    """PIPR's later local-authority codes, mapped to the build's codes."""

    payload = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text(encoding="utf-8")
    )
    aliases = payload["levels"]["local_authority"].get("code_aliases") or {}
    return {
        str(alias): str(code)
        for code, entry in aliases.items()
        for alias in entry.get("alias_codes") or ()
    }


def rent_edition(weighting: Weighting) -> str | None:
    """The PIPR edition the bound rent rows were compiled from."""

    editions = set()
    for meta in weighting.metadata:
        if meta.get("contract_target_id") in RENT_TARGET_IDS:
            fields = [part.strip() for part in str(meta.get("source", "")).split("|")]
            if len(fields) >= 4:
                editions.add(fields[3])
    if len(editions) > 1:
        raise ValueError(f"The rent rows cite more than one PIPR edition: {editions}.")
    return next(iter(editions), None)


def pipr_bedroom_means(
    facts: Path, *, edition: str
) -> dict[tuple[str, str, str], float]:
    """PIPR's calendar-2025 mean monthly rent by (grain, area, category).

    The mean of the twelve monthly means of ``edition``, by local authority
    and by Scottish BRMA; a cell without all twelve months is left out.
    """

    aliases = _la_aliases()
    months: dict[tuple[str, str, str], dict[str, float]] = {}
    with _feed_path(facts).open(encoding="utf-8") as handle:
        for line in handle:
            if PIPR_MEASURE not in line or "bed" not in line:
                continue
            fact = json.loads(line)
            if (fact.get("source") or {}).get("vintage") != edition:
                continue
            if (fact.get("layout") or {}).get("measure_id") != PIPR_MEASURE:
                continue
            bedrooms = (fact.get("dimensions") or {}).get("bedrooms")
            if bedrooms not in PIPR_BEDROOMS_CATEGORY:
                continue
            if set(fact.get("dimensions") or {}) != {"bedrooms"}:
                continue
            geography = fact.get("geography") or {}
            grain = PIPR_LEVELS.get(str(geography.get("level")))
            period = fact.get("period") or {}
            if grain is None or period.get("type") != "month":
                continue
            month = str(period.get("value"))
            if not month.startswith(f"{PIPR_YEAR}-"):
                continue
            area = str(geography["id"])
            if grain == "la":
                area = aliases.get(area, area)
            key = (grain, area, PIPR_BEDROOMS_CATEGORY[bedrooms])
            months.setdefault(key, {})[month] = float(fact["value"])
    return {
        key: float(np.mean(list(values.values())))
        for key, values in months.items()
        if len(values) == 12
    }


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3:
        return None
    a = pd.Series(left).rank().to_numpy()
    b = pd.Series(right).rank().to_numpy()
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def pipr_comparison(
    renters: pd.DataFrame,
    pipr: Mapping[tuple[str, str, str], float],
    *,
    rent: str = "rent",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The build's mean rents by district and bedrooms against PIPR's."""

    area = {
        "la": renters["la_code"].to_numpy(),
        "brma": uk_brma_area_codes(renters["brma"].to_numpy()),
    }
    audit = []
    grouped: dict[str, dict[str, list[float]]] = {}
    withheld: dict[str, int] = {}
    for grain in ("la", "brma"):
        scope = renters.assign(area=area[grain])
        scope = scope.loc[scope["area"].astype(str) != ""]
        if grain == "la":
            scope = scope.loc[scope["nation"] != "Scotland"]
        for (code, category), rows in scope.groupby(["area", "category"], sort=True):
            published = pipr.get((grain, str(code), str(category)))
            if published is None:
                continue
            cell = _mean_cell(rows, rent=rent)
            nation = _area_nation(str(code))
            label = f"{grain} {nation}"
            audit.append(
                {
                    "grain": grain,
                    "area": str(code),
                    "category": str(category),
                    "pipr_monthly": round(published, 2),
                    **cell,
                }
            )
            if cell["source_households"] < MIN_SOURCE_HOUSEHOLDS:
                withheld[label] = withheld.get(label, 0) + 1
                continue
            bucket = grouped.setdefault(label, {"model": [], "pipr": []})
            bucket["model"].append(float(cell["mean_monthly_rent"]))
            bucket["pipr"].append(float(published))
    summary = {}
    for label in sorted(set(grouped) | set(withheld)):
        values = grouped.get(label, {"model": [], "pipr": []})
        model = np.asarray(values["model"])
        published = np.asarray(values["pipr"])
        if model.size < MIN_GROUP_CELLS:
            summary[label] = {
                "withheld": f"fewer than {MIN_GROUP_CELLS} cells",
                "cells_under_10_source_households": withheld.get(label, 0),
            }
            continue
        ratio = model / published
        summary[label] = {
            "cells": int(model.size),
            "cells_under_10_source_households": withheld.get(label, 0),
            "rank_correlation": _spearman(model, published),
            "share_within_10pct": round(float(np.mean(np.abs(ratio - 1) <= 0.10)), 3),
            "share_within_25pct": round(float(np.mean(np.abs(ratio - 1) <= 0.25)), 3),
            "median_ratio": round(float(np.median(ratio)), 3),
        }
    return summary, audit


# ----------------------------------------------------------------------------
# BRMA mix, weights and coherence


def brma_mix(
    renters: pd.DataFrame, counts: Mapping[tuple[str, str], Mapping[str, float]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Each region's renters across its BRMAs, against the census counts.

    The total variation distance between the weighted share of a region's
    private renters in each BRMA and the census share. The summary keeps a
    renter-weighted mean by nation; the audit keeps each region's shares.
    """

    audit: dict[str, Any] = {}
    by_nation: dict[str, list[tuple[float, float]]] = {}
    for region, rows in renters.groupby("region", sort=True):
        if _carriers(rows["carrier"]) < MIN_SOURCE_HOUSEHOLDS:
            continue
        census: dict[str, float] = {}
        for category in CATEGORIES:
            for brma, count in (counts.get((str(region), category)) or {}).items():
                census[brma] = census.get(brma, 0.0) + float(count)
        total = sum(census.values())
        if total <= 0:
            continue
        model = rows.groupby("brma")["weight"].sum()
        model = model / model.sum()
        names = sorted(set(census) | set(model.index.astype(str)))
        distance = 0.5 * sum(
            abs(float(model.get(name, 0.0)) - census.get(name, 0.0) / total)
            for name in names
        )
        audit[str(region)] = {
            "total_variation_distance": round(distance, 4),
            "model_share": {
                name: round(float(model.get(name, 0.0)), 4) for name in names
            },
            "census_share": {
                name: round(census.get(name, 0.0) / total, 4) for name in names
            },
        }
        nation = REGION_NATION[str(region)]
        by_nation.setdefault(nation, []).append((distance, float(rows["weight"].sum())))
    summary = {
        nation: round(sum(d * w for d, w in pairs) / sum(w for _, w in pairs), 4)
        for nation, pairs in sorted(by_nation.items())
    }
    return summary, audit


def weight_concentration(renters: pd.DataFrame) -> dict[str, Any]:
    """Effective sample size and the top 1%'s share of renters' weight."""

    raw = renters["weight"].to_numpy(dtype=float)
    order = np.argsort(-raw, kind="stable")
    weight = raw[order]
    total = float(weight.sum())
    if total <= 0:
        return {}
    if _carriers(renters["carrier"]) < MIN_SOURCE_HOUSEHOLDS:
        return {"withheld": "<10"}
    top = max(1, math.ceil(0.01 * weight.size))
    top_carriers = _carriers(renters["carrier"].to_numpy(dtype=object)[order][:top])
    # The slice must span ten rows and ten households, or it describes a few.
    share = (
        round(float(weight[:top].sum()) / total, 4)
        if top >= MIN_SOURCE_HOUSEHOLDS and top_carriers >= MIN_SOURCE_HOUSEHOLDS
        else "<10"
    )
    return {
        "renter_rows": rounded_count(int(weight.size)),
        "effective_sample_size": rounded_count(
            int(round(total**2 / float(np.square(weight).sum())))
        ),
        "top_1pct_weight_share": share,
    }


def coherence(renters: pd.DataFrame, *, rent: str = "rent") -> dict[str, Any]:
    """Shares of renters whose reported HB, or subletting income, exceeds rent."""

    weight = renters["weight"].to_numpy(dtype=float)
    carrier = renters["carrier"].to_numpy(dtype=object)
    rents = renters[rent].to_numpy(dtype=float)
    out = {}
    for name, column in (
        ("housing_benefit_reported_above_rent", "housing_benefit_reported"),
        ("subrent_above_rent", "subrent"),
    ):
        values = renters[column].to_numpy(dtype=float)
        mask = values > rents
        out[name] = {
            "weighted_share": weighted_share(mask, weight, carrier),
            "unweighted_share": unweighted_share(mask, carrier),
        }
    return out


# ----------------------------------------------------------------------------
# The clone pool at design weights

#: The pool table's columns: the household measures' columns at the problem's
#: design weights, and the spine's rent and BRMA before the rewrite.
POOL_COLUMNS = (
    "household_id",
    "weight",
    "region",
    "tenure_type",
    "rent",
    "subrent",
    "num_bedrooms",
    "brma",
    "la_code",
    "source_key",
    "channel",
    "housing_benefit_reported",
    "spine_rent",
    "spine_brma",
)


def load_pool(path: Path) -> pd.DataFrame:
    """The clone pool's household table (``tools/uk_private_rent_pool.py``)."""

    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(POOL_COLUMNS) - set(data.files))
        if missing:
            raise ValueError(f"{path}: the pool table lacks {missing}.")
        return pd.DataFrame({name: data[name] for name in POOL_COLUMNS})


# ----------------------------------------------------------------------------
# One build


def measure_households(
    frame: pd.DataFrame,
    *,
    pipr: Mapping[tuple[str, str, str], float],
    counts: Mapping[tuple[str, str], Mapping[str, float]],
    rent: str = "rent",
    frs_carriers: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The household measures' summary and audit for one weighting.

    ``frs_carriers`` counts only FRS-channel households as a figure's
    carriers, for spine values (``with_carriers``).
    """

    renters = private_renters(with_carriers(frame, frs_only=frs_carriers), rent=rent)
    cells, cells_audit = rents_by_cell(renters, rent=rent)
    districts, districts_audit = pipr_comparison(renters, pipr, rent=rent)
    mix, mix_audit = brma_mix(renters, counts)
    summary = {
        "rents_by_cell": cells,
        "pipr_by_district_and_bedrooms": districts,
        "brma_mix_total_variation_distance": mix,
        "weights": weight_concentration(renters),
        "coherence": coherence(renters, rent=rent),
    }
    audit = {
        "rents_by_cell": cells_audit,
        "pipr_by_district_and_bedrooms": districts_audit,
        "brma_mix": mix_audit,
    }
    return summary, audit


def measure_run(
    run: Path,
    *,
    facts: Path,
    counts: Mapping[tuple[str, str], Mapping[str, float]],
    h5_name: str,
    pool: Path | None = None,
    pipr_edition: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    households = release_households(run / h5_name)
    frame = None if pool is None else load_pool(pool)
    # The pool's keys cover every pool household, the H5's the released ones.
    keys = pd.Series(
        households["source_key"].to_numpy(dtype=object),
        index=pd.Index(households["household_id"].to_numpy()),
    )
    if frame is not None:
        keys = pd.Series(
            frame["source_key"].to_numpy(dtype=object),
            index=pd.Index(frame["household_id"].to_numpy()),
        )
    weightings = load_weightings(run, keys=keys)
    edition = pipr_edition or rent_edition(weightings[0])
    if edition is None:
        raise ValueError(f"{run}: the problem binds no PIPR rent rows.")
    pipr = pipr_bedroom_means(facts, edition=edition)
    summary: dict[str, Any] = {"pipr_edition": edition, "rent_rows": {}}
    audit: dict[str, Any] = {"rent_rows": {}}
    for weighting in weightings:
        fit, rows = rent_row_fit(weighting)
        summary["rent_rows"][weighting.name] = fit
        audit["rent_rows"][weighting.name] = rows
    release, release_audit = measure_households(households, pipr=pipr, counts=counts)
    summary["release_weights"] = release
    audit["release_weights"] = release_audit
    if frame is not None:
        design, design_audit = measure_households(frame, pipr=pipr, counts=counts)
        summary["design_weights"] = design
        audit["design_weights"] = design_audit
        spine = frame.assign(rent=frame["spine_rent"], brma=frame["spine_brma"])
        before, before_audit = measure_households(
            spine, pipr=pipr, counts=counts, frs_carriers=True
        )
        summary["design_weights_spine_values"] = before
        audit["design_weights_spine_values"] = before_audit
        summary["matched"] = matched_change(frame)
    return summary, audit


def matched_change(frame: pd.DataFrame) -> dict[str, Any]:
    """The pool's BRMA and rent before and after the rewrite, row for row."""

    gb = with_carriers(frame.loc[frame["region"] != NORTHERN_IRELAND])
    weight = gb["weight"].to_numpy(dtype=float)
    carrier = gb["carrier"].to_numpy(dtype=object)
    changed_brma = gb["brma"].to_numpy() != gb["spine_brma"].to_numpy()
    redrawn = redraw_mask(gb["region"], gb["tenure_type"], gb["spine_rent"])
    out: dict[str, Any] = {
        "gb_rows_whose_brma_changed": weighted_share(changed_brma, weight, carrier),
        "gb_rows_redrawn": weighted_share(redrawn, weight, carrier),
        "region_level_at_design_weights": {},
        "region_level_note": (
            "Each region's redrawn over spine mean rent of its private renters "
            "at design weights. The scale holds the FRS level on the census "
            "renter mix; placement follows all households, so a region moves by "
            "the gap between the two mixes until calibration's tenure rows "
            "restore it."
        ),
    }
    # The spine rents rest on FRS-channel households (``with_carriers``).
    renters = with_carriers(gb.loc[redrawn], frs_only=True)
    for region, rows in renters.groupby("region", sort=True):
        if _carriers(rows["carrier"]) < MIN_SOURCE_HOUSEHOLDS:
            out["region_level_at_design_weights"][region] = "<10"
            continue
        w = rows["weight"].to_numpy(dtype=float)
        after = float((w * rows["rent"].to_numpy(dtype=float)).sum())
        before = float((w * rows["spine_rent"].to_numpy(dtype=float)).sum())
        out["region_level_at_design_weights"][region] = round(after / before, 4)
    return out


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _counts() -> dict[tuple[str, str], dict[str, float]]:
    payload = (
        resources.files("microcosm.build.uk")
        .joinpath("brma_private_rented_households.json")
        .read_bytes()
    )
    return load_private_renter_counts(payload)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument(
        "--control-pool",
        type=Path,
        help="the control run's pool table, for its design and dense rent rows",
    )
    parser.add_argument("--h5-name", default="microcosm_uk_2024_25_local.h5")
    parser.add_argument(
        "--pipr-edition",
        help="PIPR edition to compare with (default: the one the rent rows cite)",
    )
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    audit_dir = outside_work_trees(
        args.audit_dir or args.run / "audit-1262", what="the private-rent audit"
    )
    counts = _counts()
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "disclosure": (
            "Aggregates only: every model cell and rent row carried by at "
            f"least {MIN_SOURCE_HOUSEHOLDS} distinct source households, rounded "
            "record counts, no single district or BRMA, no unit-record value."
        ),
        "runs": {},
    }
    audit: dict[str, Any] = {"runs": {}}
    for label, run, pool in (
        ("head", args.run, args.pool),
        ("control", args.control, args.control_pool),
    ):
        if run is None:
            continue
        summary["runs"][label], audit["runs"][label] = measure_run(
            run,
            facts=args.ledger_facts,
            counts=counts,
            h5_name=args.h5_name,
            pool=pool,
            pipr_edition=args.pipr_edition,
        )
        summary["runs"][label]["run"] = run.name
    _write(audit_dir / "private_rent_audit.json", audit)
    _write(audit_dir / "private_rent_summary.json", summary)
    print(f"wrote {audit_dir / 'private_rent_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
