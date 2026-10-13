"""Split each stage's base-to-release changes by cause, row by row.

For every stored row of a stage this script holds four values: the base's
(built on Linux), the release's (built on macOS after the Social Security
rescale), and the stage replayed on the unchanged base inputs on macOS and on
linux/amd64. A changed row is attributed by which of those agree:

- platform only: the macOS replay of the unchanged inputs already equals the
  release, so the rescale did not move this row;
- rescale only: the macOS replay equals the base, and the release differs;
- both: the macOS replay differs from the base and from the release.

Inputs are the exported stage tables and the per-platform outputs written by
``platform_stage_input.py`` and ``platform_other_stages.py``.

Usage:
    python release_change_decomposition.py --retirement-input INPUT.pkl \
        --retirement-macos MAC.npz --retirement-linux LINUX.npz \
        --other-dir DIR --other-outputs-dir OUT --out RECEIPT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RETIREMENT = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)
OTHER = ("weeks_unemployed", "childcare", "energy_subsidy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def decompose(
    base: np.ndarray, release: np.ndarray, macos: np.ndarray, linux: np.ndarray
) -> dict[str, object]:
    changed = base != release
    difference = np.abs(base - release)
    platform_only = changed & (macos == release)
    return {
        "rows": int(len(base)),
        "base_ne_release": int(changed.sum()),
        "base_ne_macos_replay": int((base != macos).sum()),
        "base_ne_linux_replay": int((base != linux).sum()),
        "macos_replay_ne_release": int((macos != release).sum()),
        "linux_replay_ne_macos_replay": int((linux != macos).sum()),
        "changed_platform_only": int(platform_only.sum()),
        "changed_rescale_only": int((changed & (base == macos)).sum()),
        "changed_both": int((changed & (base != macos) & (macos != release)).sum()),
        "changed_by_more_than_1e-9": int((difference > 1e-9).sum()),
        "changed_by_at_most_1e-9": int((changed & (difference <= 1e-9)).sum()),
        "max_abs_change_platform_only": (
            float(difference[platform_only].max()) if platform_only.any() else 0.0
        ),
        "zero_nonzero_flips_base_to_release": int(
            ((base == 0) != (release == 0)).sum()
        ),
        "zero_nonzero_flips_base_to_macos_replay": int(
            ((base == 0) != (macos == 0)).sum()
        ),
        "zero_nonzero_flips_macos_replay_to_release": int(
            ((macos == 0) != (release == 0)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retirement-input", type=Path, required=True)
    parser.add_argument("--retirement-macos", type=Path, required=True)
    parser.add_argument("--retirement-linux", type=Path, required=True)
    parser.add_argument("--other-dir", type=Path, required=True)
    parser.add_argument("--other-outputs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt: dict[str, object] = {"inputs": {}, "columns": {}}
    slim = pd.read_pickle(args.retirement_input)
    macos, linux = np.load(args.retirement_macos), np.load(args.retirement_linux)
    ids = slim["person_id"].to_numpy()
    assert np.array_equal(macos["person_id"], ids)
    assert np.array_equal(linux["person_id"], ids)
    for path in (args.retirement_input, args.retirement_macos, args.retirement_linux):
        receipt["inputs"][path.name] = sha256(path)
    for column in RETIREMENT:
        receipt["columns"][column] = decompose(
            slim[f"stored_base__{column}"].to_numpy(),
            slim[f"stored_release__{column}"].to_numpy(),
            macos[column],
            linux[column],
        )
    for stage in OTHER:
        payload = pd.read_pickle(args.other_dir / f"{stage}.pkl")
        macos_path = args.other_outputs_dir / f"outputs.{stage}.macos_arm64.npz"
        linux_path = args.other_outputs_dir / f"outputs.{stage}.linux_amd64.npz"
        for path in (macos_path, linux_path):
            receipt["inputs"][path.name] = sha256(path)
        macos, linux = np.load(macos_path), np.load(linux_path)
        assert np.array_equal(macos["ids"], payload["ids"])
        assert np.array_equal(linux["ids"], payload["ids"])
        receipt["inputs"][f"{stage}.pkl"] = sha256(args.other_dir / f"{stage}.pkl")
        receipt["columns"][stage] = decompose(
            payload["stored_base"],
            payload["stored_release"],
            macos["values"],
            linux["values"],
        )
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    for column, row in receipt["columns"].items():
        print(
            column,
            row["base_ne_release"],
            "platform only",
            row["changed_platform_only"],
            "rescale only",
            row["changed_rescale_only"],
            "both",
            row["changed_both"],
        )


if __name__ == "__main__":
    main()
