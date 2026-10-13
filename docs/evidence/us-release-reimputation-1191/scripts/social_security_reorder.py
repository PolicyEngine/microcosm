"""Does the release's Social Security repair reorder the QRF predictor?

The childcare, energy-subsidy and retirement-contribution QRFs condition on
``social_security``, the sum of four component leaves. The release multiplies
each component by its own factor. A tree splits on the order of a predictor,
so one common factor would leave every partition available to the forest
unchanged; unequal factors change the order wherever people hold different
components. This script applies the release's recorded factors to the base's
stored components and counts how the order of the sum changes.

Usage:
    python social_security_reorder.py BASE_H5 CALIBRATION_DIAGNOSTICS OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base_path, diagnostics_path, out_path = (Path(arg) for arg in sys.argv[1:4])
    components = json.loads(diagnostics_path.read_text())["build"][
        "support_value_repairs"
    ]["social_security_components"]["components"]
    columns = list(components)
    person = pd.read_hdf(
        base_path, "person", columns=["person_support_clone_index", *columns]
    )
    values = person[columns].to_numpy(dtype=np.float64)
    factors = np.array([components[column]["factor"] for column in columns])
    before = values.sum(axis=1)
    after = (values * factors).sum(axis=1)
    receipt: dict[str, object] = {
        "base_h5": Path(base_path).name,
        "base_h5_sha256": sha256(base_path),
        "diagnostics_sha256": sha256(diagnostics_path),
        "factors": dict(zip(columns, factors.tolist())),
        "groups": {},
    }
    clone = person["person_support_clone_index"].astype(int).to_numpy()
    for label, mask in (
        ("all", np.ones(len(person), dtype=bool)),
        ("asec (QRF training rows are drawn from these)", clone == 0),
        ("puf half (QRF prediction rows)", clone > 0),
    ):
        positive = mask & (before > 0)
        b, a = before[positive], after[positive]
        rank_before = pd.Series(b).rank(method="dense").to_numpy()
        rank_after = pd.Series(a).rank(method="dense").to_numpy()
        order = np.argsort(b, kind="stable")
        sorted_before, sorted_after = b[order], a[order]
        rising = sorted_before[1:] > sorted_before[:-1]
        receipt["groups"][label] = {
            "persons_with_social_security": int(positive.sum()),
            "persons_with_more_than_one_component": int(
                ((values[positive] > 0).sum(axis=1) > 1).sum()
            ),
            "persons_by_single_component": {
                column: int(
                    (
                        (values[positive][:, index] > 0)
                        & ((values[positive] > 0).sum(axis=1) == 1)
                    ).sum()
                )
                for index, column in enumerate(columns)
            },
            "persons_whose_dense_rank_changes": int((rank_before != rank_after).sum()),
            "adjacent_pairs_in_base_order": int(rising.sum()),
            "adjacent_pairs_inverted_after_repair": int(
                (rising & (sorted_after[1:] < sorted_after[:-1])).sum()
            ),
            "spearman_rank_correlation": float(
                np.corrcoef(
                    pd.Series(b).rank().to_numpy(), pd.Series(a).rank().to_numpy()
                )[0, 1]
            ),
        }
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt["groups"], indent=1))


if __name__ == "__main__":
    main()
