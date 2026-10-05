"""Concentration of the published release's own weights, for the README.

David Trimmer measured the published ACS local release's weights directly; the
sweep's ``release_repro`` re-solves them. This records the published weights'
national and Massachusetts Kish ESS and how many Massachusetts households hold
half the state's weight, on this checkpoint's households, so the README can set
the three figures side by side.

Run: ``uv run python experiments/us-acs-local-l2-basis-20260928/published_weights.py``
Writes ``results/published_weights.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gradient_scale import RELEASE_WEIGHTS  # noqa: E402
from sweep import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MASSACHUSETTS,
    kish,
    n_records_holding_half,
)


def main() -> None:
    households = pd.read_parquet(DEFAULT_CHECKPOINT / "households.parquet")
    weights = np.load(RELEASE_WEIGHTS)["weights"].astype(np.float64)
    if weights.shape != (len(households),):
        raise SystemExit(f"weights {weights.shape} vs {len(households)} households")
    ma = households["state_fips"].astype(str).to_numpy() == MASSACHUSETTS
    out = {
        "weights": str(RELEASE_WEIGHTS),
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "households": int(len(households)),
        "national_kish_ess": kish(weights),
        "massachusetts": {
            "households": int(ma.sum()),
            "kish_ess": kish(weights[ma]),
            "n_records_holding_half_weight": n_records_holding_half(weights[ma]),
        },
    }
    (HERE / "results" / "published_weights.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
