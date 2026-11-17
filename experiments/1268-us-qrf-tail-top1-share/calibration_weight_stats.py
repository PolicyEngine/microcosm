"""S2: household weight statistics before and after Route A calibration, engine-free.

Usage: calibration_weight_stats.py <populace_us_2024.h5> <populace_us_2024_calibration.npz>
The sidecar's household_weight must equal the H5's household weights row for row,
so its initial_household_weight aligns by row (route_a_top1.py checks the same).
"""

import sys

import h5py
import numpy as np

with h5py.File(sys.argv[1], "r") as f:
    h5_final = (
        f["household/table"]
        .fields(["household_weight"])[:]["household_weight"]
        .astype(np.float64)
    )
sidecar = np.load(sys.argv[2])
print("sidecar keys:", sorted(sidecar.files))
final = sidecar["household_weight"].astype(np.float64)
initial = sidecar["initial_household_weight"].astype(np.float64)
assert np.array_equal(final, h5_final), "sidecar final weights differ from the H5's"
print(f"households: {final.size:,}")
for label, w in (("initial", initial), ("final", final)):
    q = np.quantile(w, [0.5, 0.99])
    print(
        f"{label:8s} total {w.sum():,.1f}  median {q[0]:,.2f}  p99 {q[1]:,.2f}  min {w.min():,.4f}  max {w.max():,.2f}"
    )
positive = initial > 0
ratio = final[positive] / initial[positive]
print(
    f"final/initial ratio: max {ratio.max():.4f}  min {ratio.min():.6f}  households with initial weight 0: {(~positive).sum()}"
)
