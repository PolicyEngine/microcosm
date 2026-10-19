"""Engine diagnostics for UK private rents and BRMAs (microcosm#1262).

Runs policyengine-uk on a Microcosm local H5 and reports, for one year:

- Universal Credit, the UC housing element of UC recipients (all, private
  renters, social renters) and Housing Benefit;
- people in poverty after housing costs;
- the mean weekly LHA rate of LHA-eligible renters;
- UC recipients with a housing element whose rent exceeds their UC LHA rate,
  by nation, against DWP's May 2025 shares of private-rented UC households
  whose LHA does not cover their rent (CIH UK Housing Review 2026 table 111b,
  from DWP Stat-Xplore, as quoted on uk-data#521).
  The definitions differ: reported rents against DWP's verified rents, and
  benefit units against claim households.

With ``--pool`` (``tools/uk_private_rent_pool.py``), it also splits the
build's own change at its release weights into three variants:

- ``spine``: every household's spine BRMA and rent;
- ``area_brma``: the BRMA of its assigned area, with the spine's rent;
- ``built``: the build as published, with the redrawn rent too.

A control build's H5 (``--control-h5``) is measured as built.

Every figure is a weighted total, mean or share over Great Britain or a
nation. A share or mean carried by fewer than ten distinct source households,
or leaving out fewer than ten, reads "<10" (UKDS End User Licence CD137
clause 8, CD171 section 5.2.1). The output stays in the run's audit
directory; only its aggregates may be quoted.

Usage::

    uv run --no-sync python \\
        docs/evidence/uk-private-rent-1262/scripts/engine_rent_diagnostics.py \\
        --h5 <runs/head>/microcosm_uk_2024_25_local.h5 \\
        [--pool <runs/head>/audit-1262/pool.npz] [--control-h5 <control.h5>] \\
        --out <runs/head>/audit-1262/engine_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_SOURCE_HOUSEHOLDS = 10
WEEKS_IN_YEAR = 365.25 / 7

#: DWP via CIH UK Housing Review 2026 table 111b, May 2025: private-rented UC
#: households with a housing element whose LHA does not cover their rent.
#: Copied by hand from the quotation on uk-data#521; no Chronicle fact
#: carries the table, so the comparison is indicative.
DWP_RENT_ABOVE_LHA_MAY_2025 = {"England": 0.521, "Wales": 0.630, "Scotland": 0.399}
DWP_SOURCE = (
    "CIH UK Housing Review 2026 table 111b (from DWP Stat-Xplore), May 2025, "
    "copied by hand from its quotation on uk-data#521 and not a registered "
    "Chronicle fact: an indicative comparison"
)

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
    "NORTHERN_IRELAND": "Northern Ireland",
}


def _households(h5: Path) -> pd.DataFrame:
    return pd.read_hdf(
        h5,
        "household",
        columns=["household_id", "region", "source_household_key"],
    ).set_index("household_id")


def _dataset_year(h5: Path) -> int:
    return int(np.asarray(pd.read_hdf(h5, "time_period")).ravel()[0])


def _array(sim, name: str, year: int, **kwargs) -> np.ndarray:
    return np.asarray(sim.calculate(name, year, **kwargs).values)


def _share(
    mask: np.ndarray, within: np.ndarray, weight: np.ndarray, source: np.ndarray
) -> float | str | None:
    """The weighted share of ``within`` that ``mask`` covers, under control."""

    denominator = float(weight[within].sum())
    if denominator <= 0:
        return None
    for side in (within & mask, within & ~mask):
        carriers = len(pd.unique(source[side]))
        if 0 < carriers < MIN_SOURCE_HOUSEHOLDS:
            return "<10"
    return round(float(weight[within & mask].sum()) / denominator, 4)


def _mean(
    values: np.ndarray, within: np.ndarray, weight: np.ndarray, source: np.ndarray
) -> float | str | None:
    total = float(weight[within].sum())
    if total <= 0:
        return None
    if len(pd.unique(source[within])) < MIN_SOURCE_HOUSEHOLDS:
        return "<10"
    return round(float((weight * values)[within].sum()) / total, 2)


def diagnostics(sim, *, year: int, households: pd.DataFrame) -> dict[str, Any]:
    """The rent-sensitive aggregates of one simulation."""

    benunit_household = _array(sim, "household_id", year, map_to="benunit").astype(
        np.int64
    )
    lineage = households.reindex(benunit_household)
    nation = lineage["region"].map(REGION_NATION).to_numpy(dtype=object)
    source = lineage["source_household_key"].astype(str).to_numpy()
    weight = _array(sim, "benunit_weight", year).astype(float)
    uc = _array(sim, "universal_credit", year).astype(float)
    element = _array(sim, "uc_housing_costs_element", year).astype(float)
    benefit = _array(sim, "housing_benefit", year).astype(float)
    tenure = _array(sim, "benunit_tenure_type", year).astype(str)
    rent = _array(sim, "benunit_rent", year).astype(float)
    uc_cap = _array(sim, "uc_LHA_cap", year).astype(float)
    lha = _array(sim, "BRMA_LHA_rate", year).astype(float)
    eligible = _array(sim, "LHA_eligible", year).astype(bool)

    person_weight = _array(sim, "person_weight", year).astype(float)
    poverty = _array(sim, "in_poverty_ahc", year, map_to="person").astype(float)

    on_uc = uc > 0
    private = tenure == "RENT_PRIVATELY"
    social = np.isin(tenure, ["RENT_FROM_COUNCIL", "RENT_FROM_HA"])
    housed = on_uc & (element > 0)
    out: dict[str, Any] = {
        "universal_credit_bn": round(float((weight * uc).sum()) / 1e9, 3),
        "uc_housing_element_of_uc_recipients_bn": {
            "all": round(float((weight * element)[on_uc].sum()) / 1e9, 3),
            "private_renters": round(
                float((weight * element)[on_uc & private].sum()) / 1e9, 3
            ),
            "social_renters": round(
                float((weight * element)[on_uc & social].sum()) / 1e9, 3
            ),
        },
        "housing_benefit_bn": round(float((weight * benefit).sum()) / 1e9, 3),
        "people_in_ahc_poverty_m": round(
            float((person_weight * poverty).sum()) / 1e6, 3
        ),
        "mean_weekly_lha_lha_eligible_renters": {},
        "uc_recipients_rent_above_uc_lha": {},
    }
    above = rent > uc_cap
    for name in ("Great Britain", "England", "Wales", "Scotland"):
        scope = (
            nation != "Northern Ireland" if name == "Great Britain" else nation == name
        )
        out["mean_weekly_lha_lha_eligible_renters"][name] = _mean(
            lha / WEEKS_IN_YEAR, scope & eligible & (lha > 0), weight, source
        )
        within = scope & housed & private
        out["uc_recipients_rent_above_uc_lha"][name] = {
            "model": _share(above, within, weight, source),
            "dwp_may_2025": DWP_RENT_ABOVE_LHA_MAY_2025.get(name),
        }
    return out


def _simulation(h5: Path, *, year: int, overrides: Mapping[str, np.ndarray] | None):
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=str(h5))
    for name, values in (overrides or {}).items():
        sim.set_input(name, year, values)
    return sim


def _difference(after: Any, before: Any) -> Any:
    if isinstance(after, dict) and isinstance(before, dict):
        return {
            key: _difference(after[key], before.get(key))
            for key in after
            if key != "dwp_may_2025"
        }
    if isinstance(after, int | float) and isinstance(before, int | float):
        return round(float(after) - float(before), 4)
    return None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--control-h5", type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    # The report is licensed output: refused inside any git work tree before
    # the simulations, not after them.
    from microcosm.build.uk_runtime.receipt_counts import outside_work_trees

    out = outside_work_trees(args.out, what="the engine report")
    # The report measures the engine the build is pinned to (#1308's freeze),
    # so it refuses any other policyengine-uk before it reads an H5.
    from microcosm.build.uk_runtime.hmrc_uprating import (
        assert_installed_engine_is_pinned,
    )

    engine = assert_installed_engine_is_pinned()
    households = _households(args.h5)
    dataset_year = _dataset_year(args.h5)
    variants: dict[str, Mapping[str, np.ndarray] | None] = {"built": None}
    if args.pool is not None:
        with np.load(args.pool, allow_pickle=False) as pool:
            spine = pd.DataFrame(
                {"rent": pool["spine_rent"], "brma": pool["spine_brma"]},
                index=pd.Index(pool["household_id"], name="household_id"),
            ).reindex(households.index)
        if spine["rent"].isna().any():
            raise ValueError("Some H5 households are not in the pool table.")
        variants = {
            "spine": {
                "brma": spine["brma"].to_numpy(dtype=str),
                "rent": spine["rent"].to_numpy(dtype=float),
            },
            "area_brma": {"rent": spine["rent"].to_numpy(dtype=float)},
            "built": None,
        }
    report: dict[str, Any] = {
        "year": args.year,
        "dataset_year": dataset_year,
        "engine": f"policyengine-uk {engine}",
        "dwp_source": DWP_SOURCE,
        "disclosure": (
            "Weighted totals, means and shares over Great Britain or a nation; "
            f"any carried by, or leaving out, fewer than {MIN_SOURCE_HOUSEHOLDS} "
            "distinct source households reads <10."
        ),
        "variants": {},
    }
    for name, overrides in variants.items():
        sim = _simulation(args.h5, year=dataset_year, overrides=overrides)
        report["variants"][name] = diagnostics(
            sim, year=args.year, households=households
        )
        del sim
    if "spine" in report["variants"]:
        built = report["variants"]["built"]
        report["changes"] = {
            "area_brma_less_spine": _difference(
                report["variants"]["area_brma"], report["variants"]["spine"]
            ),
            "built_less_area_brma": _difference(built, report["variants"]["area_brma"]),
            "built_less_spine": _difference(built, report["variants"]["spine"]),
        }
    if args.control_h5 is not None:
        control_households = _households(args.control_h5)
        sim = _simulation(
            args.control_h5, year=_dataset_year(args.control_h5), overrides=None
        )
        report["control"] = diagnostics(
            sim, year=args.year, households=control_households
        )
        report["changes_from_control"] = _difference(
            report["variants"]["built"], report["control"]
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
