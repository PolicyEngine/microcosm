#!/usr/bin/env python3
"""Read the residential split's by-band identities back on calibrated weights.

microcosm#1063 review item 4: ``cgt_residential_split`` makes HMRC Table 8a's
residential count and gains identities in every gain band at design weights;
calibration binds only the two national totals, so this reports, per band,
the residential count and gains at design and at calibrated weights next to
the stage's expected values, and the row-level weight shape with the arms in
place (smallest positive weight, positive-weight median, zero-weight rows).

    uk_residential_readout.py <run dir> <spine.h5> [--out readout.json]

``run dir`` is a national build's output directory holding
``uk.full.dense.solution.artifact`` (household weights keyed by household id);
``spine.h5`` is the spine the build read, with its ``.build.json`` beside it
(the split stage's receipt supplies the bands). Licensed inputs stay where
they are; nothing is written unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def calibrated_weights(run: Path) -> pd.Series:
    solution = np.load(run / "uk.full.dense.solution.artifact", allow_pickle=False)
    metadata = json.loads(bytes(solution["metadata"]).decode())
    return pd.Series(
        np.asarray(solution["weights"], dtype=float),
        index=pd.Index(metadata["entity_ids"], name="household_id"),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("spine", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    with pd.HDFStore(a.spine, mode="r") as store:
        household = store.select(
            "/household",
            columns=[
                "household_id",
                "household_weight",
                "household_is_cgt_residential_clone",
                "cgt_residential_clone_index",
            ],
        )
        person = store.select(
            "/person",
            columns=[
                "person_id",
                "person_household_id",
                "capital_gains",
                "capital_gains_residential_property",
                "cgt_residential_probability",
            ],
        )
    build = json.loads((a.spine.parent / (a.spine.stem + ".build.json")).read_text())

    def find(o):
        if isinstance(o, dict):
            if o.get("stage") == "cgt_residential_split" and "bands" in o:
                return o
            for v in o.values():
                r = find(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r:
                    return r

    receipt = find(build)
    bands = receipt["bands"]
    design = household.set_index("household_id")["household_weight"].astype(float)
    final = calibrated_weights(a.run).reindex(design.index)
    missing = int(final.isna().sum())
    person = person.assign(
        design=person["person_household_id"].map(design).to_numpy(float),
        final=person["person_household_id"].map(final).to_numpy(float),
    )
    res = person["capital_gains_residential_property"].to_numpy(float) > 0
    gains = person["capital_gains"].to_numpy(float)
    rows = []
    for band in bands:
        lo, hi = float(band["gain_lower_bound"]), band.get("gain_upper_bound")
        inb = res & (gains >= lo) & ((gains < float(hi)) if hi is not None else True)
        rows.append(
            {
                "gain_lower_bound": lo,
                "gain_upper_bound": hi,
                "expected_count": band["expected_count"],
                "expected_gains": band["expected_gains"],
                "design_count": float(person["design"].to_numpy()[inb].sum()),
                "design_gains": float((person["design"].to_numpy() * gains)[inb].sum()),
                "calibrated_count": float(person["final"].to_numpy()[inb].sum()),
                "calibrated_gains": float(
                    (person["final"].to_numpy() * gains)[inb].sum()
                ),
                "arms": int(inb.sum()),
            }
        )
    tot = {
        k: float(sum(r[k] for r in rows))
        for k in (
            "expected_count",
            "expected_gains",
            "design_count",
            "design_gains",
            "calibrated_count",
            "calibrated_gains",
        )
    }
    arms = household["household_is_cgt_residential_clone"].astype(bool).to_numpy()
    pos = final.to_numpy(float) > 0
    out = {
        "run": str(a.run),
        "spine": str(a.spine),
        "households_without_calibrated_weight": missing,
        "bands": rows,
        "totals": tot,
        "calibrated_over_expected": {
            "count": tot["calibrated_count"] / tot["expected_count"],
            "gains": tot["calibrated_gains"] / tot["expected_gains"],
        },
        "row_level_weights": {
            "minimum_positive_calibrated": float(final.to_numpy(float)[pos].min()),
            "median_positive_calibrated": float(np.median(final.to_numpy(float)[pos])),
            "maximum_calibrated": float(final.max()),
            "zero_weight_rows": int((~pos).sum()),
            "arms": {
                "rows": int(arms.sum()),
                "minimum_design": float(design.to_numpy()[arms].min()),
                "minimum_positive_calibrated": float(
                    final.to_numpy(float)[arms & pos].min()
                )
                if (arms & pos).any()
                else None,
                "median_positive_calibrated": float(
                    np.median(final.to_numpy(float)[arms & pos])
                )
                if (arms & pos).any()
                else None,
                "zero_weight": int((arms & ~pos).sum()),
            },
        },
    }
    text = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
