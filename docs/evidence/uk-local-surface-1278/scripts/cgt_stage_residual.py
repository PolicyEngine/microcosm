"""Stage-level CGT residual at design weights, by the stage's own groups (microcosm#1278, #725).

Usage: cgt_stage_residual.py <spine.h5> <out.json>

The CGT stage rakes liable gainers onto HMRC Table 6 (taxpayers and gains by age), folded onto
seven age groups with 75 and over pooled, and onto Table 5 (taxpayers by region), folded onto four
region groups. This script reads a built spine at its design weights and compares the liable
gainers in each group (gains above the annual exempt amount) with those folded margins, through the
stage's own folding functions. A reported group rests on 30 or more distinct survey source
households (clones and copies count once), and a group that reports gains also passes two
dominance rules on its weighted gains, counted by source household: the largest holds at most half
of the group's gains, and what the group holds beyond its two largest is at least 10% of the
largest (the p% rule at p = 10). A group that fails is pooled with its neighbour (the next older
age group, or the next region group in the stage's order; the last group with the one before it)
until every reported group passes, so no group can be recovered by subtracting the others from a
total, and no group's gains reveal one household's.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_AGE_GROUP_LOWER_BOUNDS,
    UK_CGT_REGION_GROUP_LABELS,
    _fold_age_gains_margin,
    _fold_age_margin,
    _fold_region_margin,
    _person_conditioning_cells,
    uk_cgt_policy_parameters,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    load_hmrc_cgt_conditioning_facts,
)

MINIMUM_SOURCES = 30
#: Dominance rules on a group's weighted gains, by source household.
TOP_SOURCE_SHARE_MAX = 0.5
P_PERCENT = 0.10


def passes_dominance(contributions: pd.Series) -> bool:
    """The largest source holds at most half, and the p% rule holds."""

    ordered = contributions.sort_values(ascending=False).to_numpy(float)
    total = float(ordered.sum())
    if total <= 0.0:
        return False
    largest = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    remainder = total - largest - second
    return largest <= TOP_SOURCE_SHARE_MAX * total and remainder >= P_PERCENT * largest


def pooled_groups(
    group_sources: list[set], group_contributions: list[pd.Series] | None = None
) -> list[list[int]]:
    """Adjacent groups pooled until each pool passes the floor and, with gains, dominance."""

    def passes(pool: list[int]) -> bool:
        sources = set().union(*(group_sources[index] for index in pool))
        if len(sources) < MINIMUM_SOURCES:
            return False
        if group_contributions is None:
            return True
        joined = pd.concat([group_contributions[index] for index in pool])
        return passes_dominance(joined.groupby(level=0).sum())

    pools: list[list[int]] = []
    current: list[int] = []
    for index in range(len(group_sources)):
        current.append(index)
        if passes(current):
            pools.append(current)
            current = []
    if current:
        if not pools:
            raise SystemExit("The liable gainers cannot be grouped to pass the rules.")
        pools[-1].extend(current)
    # Joining the tail can break the pool it joins; join backwards until it passes.
    while not passes(pools[-1]):
        if len(pools) == 1:
            raise SystemExit("The liable gainers cannot be grouped to pass the rules.")
        tail = pools.pop()
        pools[-1].extend(tail)
    return pools


def age_labels() -> list[str]:
    bounds = UK_CGT_AGE_GROUP_LOWER_BOUNDS
    return [
        f"{low}-{bounds[index + 1] - 1}" if index + 1 < len(bounds) else f"{low}+"
        for index, low in enumerate(bounds)
    ]


def main(spine_path: str, out_path: str) -> None:
    with pd.HDFStore(spine_path, mode="r") as store:
        person = store["person"]
        household = store["household"]
    weight = (
        person["person_household_id"]
        .map(
            pd.Series(
                household["household_weight"].to_numpy(float),
                index=household["household_id"],
            )
        )
        .to_numpy(float)
    )
    source = (
        person["person_household_id"]
        .map(
            pd.Series(
                household["household_source_id"].to_numpy(),
                index=household["household_id"],
            )
        )
        .to_numpy()
    )
    _, age_group, _, region_group = _person_conditioning_cells(person, household)
    exempt = uk_cgt_policy_parameters(2024).annual_exempt_amount
    gains = person["capital_gains"].to_numpy(float)
    liable = gains > exempt
    conditioning = load_hmrc_cgt_conditioning_facts()
    targets = {
        "age_taxpayers": _fold_age_margin(conditioning),
        "age_gains": _fold_age_gains_margin(conditioning),
        "region_taxpayers": _fold_region_margin(conditioning),
    }

    def report(labels, groups, target_counts, target_gains=None):
        masks = [liable & (groups == index) for index in range(len(labels))]
        contributions = (
            None
            if target_gains is None
            else [
                pd.Series((weight * gains)[mask]).groupby(source[mask]).sum()
                for mask in masks
            ]
        )
        rows = []
        sources = [set(pd.unique(source[mask])) for mask in masks]
        for pool in pooled_groups(sources, contributions):
            mask = np.logical_or.reduce([masks[index] for index in pool])
            target_count = float(sum(target_counts[index] for index in pool))
            achieved = float(weight[mask].sum())
            row = {
                "group": " + ".join(labels[index] for index in pool),
                "target_taxpayers": round(target_count),
                "achieved_taxpayers": round(achieved),
                "taxpayers_residual": achieved / target_count - 1.0,
            }
            if target_gains is not None:
                target = float(sum(target_gains[index] for index in pool))
                achieved_gains = float((weight * gains)[mask].sum())
                row["target_gains"] = round(target)
                row["achieved_gains"] = round(achieved_gains)
                row["gains_residual"] = achieved_gains / target - 1.0
            rows.append(row)
        return rows

    out = {
        "spine": spine_path.rsplit("/", 1)[-1],
        "annual_exempt_amount": exempt,
        "weights": "design (spine household_weight)",
        "minimum_distinct_sources_per_group": MINIMUM_SOURCES,
        "gains_dominance": {
            "top_source_share_max": TOP_SOURCE_SHARE_MAX,
            "p_percent": P_PERCENT,
        },
        "age_groups": report(
            age_labels(),
            age_group,
            targets["age_taxpayers"],
            targets["age_gains"],
        ),
        "region_groups": report(
            list(UK_CGT_REGION_GROUP_LABELS),
            region_group,
            targets["region_taxpayers"],
        ),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
