"""Weighted size and direction of the release's second pass, by stage column.

For each column a release-time re-run changed, sum the base value and the
release value over the same rows under one weight vector, so the difference
isolates the re-run. Two weightings are reported: the release's calibrated
household weights (what ships) and the base frame's weights. Rows are split by
support channel and clone.

Usage:
    python second_pass_effect.py BASE_H5 RELEASE_H5 OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = {
    "person": (
        "traditional_401k_contributions_desired",
        "roth_401k_contributions_desired",
        "traditional_ira_contributions_desired",
        "roth_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
        "weeks_unemployed",
        "health_insurance_premiums_without_medicare_part_b",
    ),
    "spm_unit": (
        "spm_unit_pre_subsidy_childcare_expenses",
        "spm_unit_energy_subsidy",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def household_weights(path: Path) -> pd.Series:
    household = pd.read_hdf(
        path, "household", columns=["household_id", "household_weight"]
    )
    return household.set_index("household_id")["household_weight"]


def main() -> None:
    base_path, release_path, out_path = (Path(arg) for arg in sys.argv[1:4])
    weights = {
        "release": household_weights(release_path),
        "base": household_weights(base_path),
    }
    receipt: dict[str, object] = {
        "base_h5": str(base_path),
        "base_h5_sha256": sha256(base_path),
        "release_h5": str(release_path),
        "release_h5_sha256": sha256(release_path),
        "columns": {},
    }
    for entity, columns in COLUMNS.items():
        key = f"{entity}_id"
        household_key = (
            f"{entity}_household_id" if entity != "person" else "person_household_id"
        )
        provenance = [f"{entity}_support_channel", f"{entity}_support_clone_index"]
        read = [key, *provenance, *columns]
        base = pd.read_hdf(base_path, entity, columns=read).set_index(key)
        release = pd.read_hdf(release_path, entity, columns=[key, *columns]).set_index(
            key
        )
        release = release.loc[base.index]
        if entity == "person":
            household = pd.read_hdf(
                base_path, "person", columns=[key, household_key]
            ).set_index(key)[household_key]
        else:
            person = pd.read_hdf(
                base_path,
                "person",
                columns=["person_spm_unit_id", "person_household_id"],
            )
            household = person.groupby("person_spm_unit_id")[
                "person_household_id"
            ].first()
        household = household.loc[base.index].to_numpy()
        channel = (
            base[provenance[0]].map(
                lambda v: v.decode() if isinstance(v, bytes) else str(v)
            )
            + "/clone"
            + base[provenance[1]].astype(int).astype(str)
        )
        for column in columns:
            b = base[column].to_numpy(dtype=np.float64)
            r = release[column].to_numpy(dtype=np.float64)
            changed = b != r
            entry: dict[str, object] = {
                "entity": entity,
                "rows_changed": int(changed.sum()),
            }
            for label, series in weights.items():
                w = series.reindex(household).to_numpy(dtype=np.float64)
                by_channel = {}
                for name in sorted(channel.unique()):
                    mask = (channel == name).to_numpy()
                    base_total = float(np.sum(b[mask] * w[mask]))
                    release_total = float(np.sum(r[mask] * w[mask]))
                    by_channel[name] = {
                        "base_value_total": base_total,
                        "release_value_total": release_total,
                        "release_minus_base": release_total - base_total,
                        "relative_change": (
                            (release_total - base_total) / base_total
                            if base_total
                            else None
                        ),
                        "weighted_nonzero_base": float(
                            np.sum(w[mask] * (b[mask] != 0))
                        ),
                        "weighted_nonzero_release": float(
                            np.sum(w[mask] * (r[mask] != 0))
                        ),
                        "rows_changed": int((changed & mask).sum()),
                    }
                entry[f"under_{label}_weights"] = by_channel
            receipt["columns"][column] = entry
            print(column, entry["rows_changed"], flush=True)
    Path(out_path).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
