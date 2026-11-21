"""Replicate calibration: Poisson bootstrap of source households, then the same solver on each replicate.

Captures noise that the fixed-weight linearization misses, including calibration's own response.
Usage: rep_boot.py <first> <last>"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/reviews/pe30-population-2026-09-27/scripts"))
import calib_resolve as cr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "rep")
first, last = int(sys.argv[1]), int(sys.argv[2])
diag, T, A, z, lw, fam, b = cr.load()
lw = lw.copy()
lw[fam == "jct"] = 0.0
w0 = z["initial_household_weight"].astype(np.float64)
H5 = cr.W / "pe_data" / "populace_us_2024.h5"
h = pd.read_hdf(H5, "household")["household_id"].to_numpy()
p = (
    pd.read_hdf(H5, "person")[
        ["person_household_id", "source_year", "source_household_id"]
    ]
    .drop_duplicates("person_household_id")
    .set_index("person_household_id")
    .reindex(h)
)
clu = pd.factorize(p.source_year.astype(str) + ":" + p.source_household_id.astype(str))[
    0
]
n = clu.max() + 1
print("households", len(h), "clusters", n, flush=True)
A = A.tocsc()
for r in range(first, last + 1):
    f = f"{OUT}/r{r:03d}.npz"
    if os.path.exists(f):
        continue
    t = time.time()
    rng = np.random.default_rng(20261101 + r)
    m = rng.poisson(1.0, n)[clu]
    keep = m > 0
    w0r = (w0 * m)[keep]
    w0r *= w0.sum() / w0r.sum()
    w, traj = cr.solve_np(A[:, keep].tocsr(), b, lw, w0r)
    full = np.zeros(len(w0))
    full[keep] = w
    np.savez_compressed(f, weights=full, loss=traj[-1])
    print(r, round(time.time() - t), round(float(traj[-1]), 4), flush=True)
