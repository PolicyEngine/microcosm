#!/usr/bin/env python
"""Compare twin UK spines on the CGT distribution HMRC publishes (microcosm#725).

Reads two spine H5 files (a control built on main and a candidate built on the
#725 branch) and their build sidecars, measures the liable-gainer population
(net gains above the 2024 annual exempt amount) by size of gain, age band,
country/region and, where the candidate carries them, asset type, and writes
the comparison tables against the published 2024-25 rows vendored in the
country package. Nothing here is a gate; it is the evidence note's input.

    .venv/bin/python experiments/build_725_cgt_distribution_receipt.py \
        --control <control.h5> --candidate <candidate.h5> --out docs/evidence/uk-cgt-725
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.cgt_asset_type import (
    CGT_ASSET_TYPE_COLUMN,
    CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
    CGT_ASSET_TYPE_RESIDENTIAL,
    load_hmrc_cgt_asset_type_facts,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_SIZE_BAND_LOWER_BOUNDS,
    load_hmrc_cgt_conditioning_facts,
)

AEA_2024 = 3_000.0
TOP_BAND = 2_000_000.0


def _frame(path: Path) -> pd.DataFrame:
    person = pd.read_hdf(path, "/person")
    household = pd.read_hdf(path, "/household")
    weights = pd.Series(household["household_weight"].to_numpy(), index=household["household_id"])
    regions = pd.Series(household["region"].astype(str).to_numpy(), index=household["household_id"])
    out = pd.DataFrame(
        {
            "gains": pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float),
            "age": pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float),
            "weight": person["person_household_id"].map(weights).to_numpy(dtype=float),
            "region": person["person_household_id"].map(regions).to_numpy(),
        }
    )
    if CGT_ASSET_TYPE_COLUMN in person.columns:
        out["asset_type"] = person[CGT_ASSET_TYPE_COLUMN].astype(str).to_numpy()
    return out


def _band(values: np.ndarray, bounds: tuple[int, ...]) -> np.ndarray:
    return np.asarray(bounds)[np.digitize(values, np.asarray(bounds)[1:])]


def _by(frame: pd.DataFrame, key: np.ndarray, labels) -> pd.DataFrame:
    liable = frame[frame["gains"] > AEA_2024]
    key = key[frame["gains"].to_numpy() > AEA_2024]
    rows = []
    for label in labels:
        mask = key == label
        rows.append(
            {
                "key": label,
                "people": float(liable["weight"][mask].sum()),
                "gains": float((liable["weight"][mask] * liable["gains"][mask]).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("key")


def _summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    conditioning = load_hmrc_cgt_conditioning_facts()
    gains = frame["gains"].to_numpy()
    tables: dict[str, pd.DataFrame] = {}
    size = _by(frame, _band(gains, HMRC_CGT_SIZE_BAND_LOWER_BOUNDS), HMRC_CGT_SIZE_BAND_LOWER_BOUNDS)
    size["published_people"] = [conditioning.size_band(b).taxpayers for b in size.index]
    size["published_gains"] = [conditioning.size_band(b).gains for b in size.index]
    tables["size_bands"] = size
    age_bounds = tuple(b.lower_bound for b in conditioning.age_bands)
    age = _by(frame, _band(frame["age"].to_numpy(), age_bounds), age_bounds)
    age["published_people"] = [b.taxpayers for b in conditioning.age_bands]
    age["published_gains"] = [b.gains for b in conditioning.age_bands]
    tables["age_bands"] = age
    region_names = [r.region for r in conditioning.regions]
    region = _by(frame, frame["region"].to_numpy(), region_names)
    share_people = conditioning.table1.individuals_share("taxpayers")
    share_gains = conditioning.table1.individuals_share("gains")
    region["published_people_individuals_basis"] = [r.taxpayers * share_people for r in conditioning.regions]
    region["published_gains_individuals_basis"] = [r.gains * share_gains for r in conditioning.regions]
    tables["regions"] = region
    # The #725 joint: age band x Table 3 gain band, people and gains.
    liable = frame[gains > AEA_2024]
    age_key = _band(liable["age"].to_numpy(), age_bounds)
    gain_key = _band(liable["gains"].to_numpy(), HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
    joint = []
    for a in age_bounds:
        for g in HMRC_CGT_GAIN_BAND_LOWER_BOUNDS:
            mask = (age_key == a) & (gain_key == g)
            joint.append({"age_lower": a, "gain_lower": g, "people": float(liable["weight"][mask].sum()), "gains": float((liable["weight"][mask] * liable["gains"][mask]).sum())})
    tables["age_by_gain_band"] = pd.DataFrame(joint)
    return tables


def _headline(frame: pd.DataFrame) -> dict[str, float]:
    liable = frame[frame["gains"] > AEA_2024]
    top = liable[liable["gains"] >= TOP_BAND]
    older = liable["age"] >= 65
    older_top = top["age"] >= 65
    return {
        "liable_taxpayers": float(liable["weight"].sum()),
        "liable_gains": float((liable["weight"] * liable["gains"]).sum()),
        "share_65_plus_of_taxpayers": float(liable["weight"][older].sum() / liable["weight"].sum()),
        "share_65_plus_of_gains": float((liable["weight"] * liable["gains"])[older].sum() / (liable["weight"] * liable["gains"]).sum()),
        "share_65_plus_of_2m_plus_gainers": float(top["weight"][older_top].sum() / top["weight"].sum()) if len(top) else float("nan"),
        "share_65_plus_of_2m_plus_gains": float((top["weight"] * top["gains"])[older_top].sum() / (top["weight"] * top["gains"]).sum()) if len(top) else float("nan"),
        "top_2m_plus_taxpayers": float(top["weight"].sum()),
        "top_2m_plus_gains": float((top["weight"] * top["gains"]).sum()),
    }


def _asset_types(frame: pd.DataFrame) -> pd.DataFrame | None:
    if "asset_type" not in frame.columns:
        return None
    facts = load_hmrc_cgt_asset_type_facts()
    liable = frame[frame["gains"] > AEA_2024]
    rows = []
    for name in (CGT_ASSET_TYPE_RESIDENTIAL, *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES):
        mask = liable["asset_type"] == name
        rows.append({"asset_type": name, "people": float(liable["weight"][mask].sum()), "gains": float((liable["weight"][mask] * liable["gains"][mask]).sum())})
    table = pd.DataFrame(rows).set_index("asset_type")
    table.loc[CGT_ASSET_TYPE_RESIDENTIAL, "published_people_individuals_basis"] = facts.residential_taxpayers_individuals_basis
    table.loc[CGT_ASSET_TYPE_RESIDENTIAL, "published_gains_individuals_basis"] = facts.residential_gains_individuals_basis
    shares = facts.non_residential_gains_shares()
    non_res = table.loc[list(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES), "gains"].sum()
    for name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES:
        table.loc[name, "achieved_gains_share"] = table.loc[name, "gains"] / non_res if non_res else float("nan")
        table.loc[name, "table7_gains_share"] = shares[name]
    return table


def _stage_evidence(sidecar: Path) -> dict[str, object]:
    if not sidecar.exists():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    evidence = payload.get("stage_evidence") or {}
    return {stage: evidence[stage] for stage in ("hmrc_cgt_gains_spine", "hmrc_cgt_asset_type_spine") if stage in evidence}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = {"control": _frame(args.control), "candidate": _frame(args.candidate)}
    receipt: dict[str, object] = {"label": args.label, "inputs": {k: str(v) for k, v in (("control", args.control), ("candidate", args.candidate))}}
    for name, frame in frames.items():
        receipt[f"{name}_headline"] = _headline(frame)
        tables = _summaries(frame)
        for table_name, table in tables.items():
            table.to_csv(args.out / f"{name}-{table_name}{args.label}.csv")
        assets = _asset_types(frame)
        if assets is not None:
            assets.to_csv(args.out / f"{name}-asset-types{args.label}.csv")
        receipt[f"{name}_stage_evidence"] = _stage_evidence(args.__dict__[name].with_suffix(".build.json"))
    (args.out / f"receipt{args.label}.json").write_text(json.dumps(receipt, indent=2, default=float) + "\n", encoding="utf-8")
    for name in frames:
        print(name, json.dumps(receipt[f"{name}_headline"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
