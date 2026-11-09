"""Engine-side effect of the release's second contribution draw.

The release stores *desired* contribution leaves; policyengine-us caps them
before it subtracts the traditional 401(k) amount from wages. This script
runs the locked engine twice on the release H5 at the release's own weights:
once as stored, and once with the five desired leaves set back to the base
frame's values. The difference in ``traditional_401k_contributions``,
``pre_tax_contributions`` and ``irs_employment_income`` is what the second
draw moves in the engine, person by person. Nothing is recalibrated.

Usage:
    python engine_second_pass_effect.py BASE_H5 RELEASE_H5 OUT_JSON
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us.data import USSingleYearDataset

PERIOD = 2024
LEAVES = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)
VARIABLES = (
    "traditional_401k_contributions",
    "pre_tax_contributions",
    "irs_employment_income",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def simulate(release: Path, overrides: dict[str, np.ndarray] | None):
    sim = Microsimulation(
        dataset=USSingleYearDataset(file_path=str(release)),
        spm={"geography_kind": "county"},
    )
    person_id = np.asarray(sim.calculate("person_id", PERIOD))
    if overrides is not None:
        for leaf, by_id in overrides.items():
            sim.set_input(leaf, PERIOD, by_id.reindex(person_id).to_numpy())
    weight = np.asarray(sim.calculate("person_weight", PERIOD), dtype=np.float64)
    values = {
        name: np.asarray(sim.calculate(name, PERIOD), dtype=np.float64)
        for name in VARIABLES
    }
    del sim
    gc.collect()
    return person_id, weight, values


def main() -> None:
    base_path, release_path, out_path = (Path(arg) for arg in sys.argv[1:4])
    person_id, weight, stored = simulate(release_path, None)
    base = pd.read_hdf(base_path, "person", columns=["person_id", *LEAVES]).set_index(
        "person_id"
    )
    overrides = {leaf: base[leaf].astype(np.float64) for leaf in LEAVES}
    missing = int((~pd.Index(person_id).isin(base.index)).sum())
    person_id_2, weight_2, reverted = simulate(release_path, overrides)
    assert np.array_equal(person_id, person_id_2)
    assert np.array_equal(weight, weight_2)

    provenance = (
        pd.read_hdf(
            release_path,
            "person",
            columns=[
                "person_id",
                "person_support_channel",
                "person_support_clone_index",
            ],
        )
        .set_index("person_id")
        .reindex(person_id)
    )
    labels = (
        provenance["person_support_channel"].map(
            lambda v: v.decode() if isinstance(v, bytes) else str(v)
        )
        + "/clone"
        + provenance["person_support_clone_index"].astype(int).astype(str)
    ).to_numpy()

    receipt: dict[str, object] = {
        "base_h5": Path(base_path).name,
        "base_h5_sha256": sha256(base_path),
        "release_h5": Path(release_path).name,
        "release_h5_sha256": sha256(release_path),
        "policyengine_us": md.version("policyengine-us"),
        "policyengine_core": md.version("policyengine-core"),
        "period": PERIOD,
        "weights": "the release's stored weights in both runs",
        "persons": int(len(person_id)),
        "persons_missing_in_base": missing,
        "variables": {},
    }
    for name in VARIABLES:
        as_stored, with_base = stored[name], reverted[name]
        difference = as_stored - with_base
        changed = as_stored != with_base
        entry: dict[str, object] = {
            "persons_changed": int(changed.sum()),
            "persons_changed_by_more_than_1_dollar": int(
                (np.abs(difference) > 1.0).sum()
            ),
            "weighted_total_as_stored": float(np.sum(weight * as_stored)),
            "weighted_total_with_base_leaves": float(np.sum(weight * with_base)),
            "weighted_stored_minus_base_leaves": float(np.sum(weight * difference)),
            "by_channel": {},
        }
        for label in sorted(set(labels)):
            mask = labels == label
            entry["by_channel"][label] = {
                "persons_changed": int((changed & mask).sum()),
                "weighted_total_as_stored": float(np.sum((weight * as_stored)[mask])),
                "weighted_total_with_base_leaves": float(
                    np.sum((weight * with_base)[mask])
                ),
                "weighted_stored_minus_base_leaves": float(
                    np.sum((weight * difference)[mask])
                ),
            }
        receipt["variables"][name] = entry
        print(
            name, entry["persons_changed"], entry["weighted_stored_minus_base_leaves"]
        )
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
