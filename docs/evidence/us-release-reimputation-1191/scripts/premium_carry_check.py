"""Where the base's PUF-half health-insurance premium comes from.

The release's other-health-insurance stage replaces
``health_insurance_premiums_without_medicare_part_b`` on the PUF half. This
script checks what the base stores there: for each PUF-clone person it looks
up the ASEC person with the same ``person_source_id`` and compares the stored
premium, and it compares the ASEC premium with the raw ASEC ``PHIP_VAL``.

Usage:
    python premium_carry_check.py BASE_H5 OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN = "health_insurance_premiums_without_medicare_part_b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    person = pd.read_hdf(
        base_path,
        "person",
        columns=["person_source_id", "person_support_clone_index", COLUMN, "PHIP_VAL"],
    )
    clone = person["person_support_clone_index"].astype(int)
    asec = person[clone == 0].set_index("person_source_id")
    receipt: dict[str, object] = {
        "base_h5": base_path.name,
        "base_h5_sha256": sha256(base_path),
        "column": COLUMN,
        "asec_rows": int(len(asec)),
        "asec_source_ids_unique": bool(asec.index.is_unique),
        "asec_rows_equal_to_PHIP_VAL": int(
            (asec[COLUMN].to_numpy() == asec["PHIP_VAL"].to_numpy()).sum()
        ),
        "clones": {},
    }
    for index in sorted(set(clone) - {0}):
        rows = person[clone == index]
        twin = asec[COLUMN].reindex(rows["person_source_id"].to_numpy()).to_numpy()
        receipt["clones"][str(index)] = {
            "rows": int(len(rows)),
            "rows_with_an_asec_twin": int(np.isfinite(twin).sum()),
            "rows_equal_to_asec_twin": int((rows[COLUMN].to_numpy() == twin).sum()),
        }
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
