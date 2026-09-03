#!/usr/bin/env python3
"""Measure how far ``fit.qrf@1`` draws move between platforms.

Run once per platform (for example natively on arm64 and under Rosetta from
an x86_64 environment synced to the same lock)::

    uv run python tools/graph_qrf_platform_probe.py <out-dir>

Each run writes ``<out-dir>/<machine>.json`` with every drawn value for 20
seed-by-regime cases; ``--compare <out-dir>`` then reports the maximum
absolute, relative, and ulp movement and how many cells differ. The
2026-09-03 result is recorded in ``docs/graph-qrf-cross-platform.md`` and is
why the kernel declares ``Numeric.PLATFORM_BITWISE`` (amendment 16).
"""

import json
import pathlib
import platform
import sys

import numpy as np
import pandas as pd

from microcosm.fit import fit as fit_qrf


def make(regime: str, seed: int, n_donor=600, n_rec=300):
    rng = np.random.default_rng(1000 + seed)
    x1 = rng.normal(size=n_donor + n_rec)
    x2 = rng.integers(0, 5, size=n_donor + n_rec).astype(float)
    if regime == "positive":
        y = np.exp(0.5 * x1 + 0.1 * x2 + rng.normal(scale=0.3, size=n_donor + n_rec))
    elif regime == "mixed_sign":
        y = 3.0 * x1 - 0.7 * x2 + rng.normal(scale=1.0, size=n_donor + n_rec)
    elif regime == "near_ties":
        y = np.round(x1, 1) + rng.choice([0.0, 1e-9, -1e-9], size=n_donor + n_rec)
    elif regime == "zero_inflated":
        y = np.where(
            rng.random(n_donor + n_rec) < 0.6,
            0.0,
            np.abs(rng.normal(size=n_donor + n_rec)) * 1e4,
        )
    else:
        raise ValueError(regime)
    donors = pd.DataFrame({"x1": x1[:n_donor], "x2": x2[:n_donor], "y": y[:n_donor]})
    recipients = pd.DataFrame({"x1": x1[n_donor:], "x2": x2[n_donor:]})
    return donors, recipients


def main(out_dir):
    results = {}
    for regime in ("positive", "mixed_sign", "near_ties", "zero_inflated"):
        for seed in range(5):
            donors, recipients = make(regime, seed)
            model = fit_qrf(donors, ["x1", "x2"], ["y"], weights="none", seed=seed)
            drawn = model.predict(recipients)
            values = np.asarray(drawn["y"], dtype=np.float64)
            results[f"{regime}/{seed}"] = [float(v) for v in values]
    arch = platform.machine()
    with open(f"{out_dir}/{arch}.json", "w") as f:
        json.dump(
            {"arch": arch, "python": sys.version.split()[0], "results": results}, f
        )
    print("wrote", arch, len(results), "cases")


def compare(out_dir: str) -> None:
    files = sorted(pathlib.Path(out_dir).glob("*.json"))
    files = [f for f in files if f.stem not in {"summary"}]
    if len(files) != 2:
        raise SystemExit(
            f"expected two platform files under {out_dir}, found {len(files)}"
        )
    a = json.load(open(files[0]))["results"]
    b = json.load(open(files[1]))["results"]
    worst = {"abs": 0.0, "rel": 0.0, "ulps": 0, "cells": 0, "differing": 0}
    for key in a:
        x = np.asarray(a[key])
        y = np.asarray(b[key])
        diff = np.abs(x - y)
        rel = diff / np.maximum(np.abs(y), 1e-300)
        ulps = [
            abs(int(np.float64(u).view(np.int64)) - int(np.float64(v).view(np.int64)))
            for u, v in zip(x, y, strict=True)
        ]
        moved = int((diff > 0).sum())
        worst["abs"] = max(worst["abs"], float(diff.max()))
        worst["rel"] = max(worst["rel"], float(rel[diff > 0].max()) if moved else 0.0)
        worst["ulps"] = max(worst["ulps"], max(ulps))
        worst["cells"] += len(x)
        worst["differing"] += moved
        if moved:
            print(
                f"{key}: max_abs={diff.max():.3e} max_rel={worst['rel']:.3e} differing={moved}/{len(x)}"
            )
    print(f"{files[0].stem} vs {files[1].stem}:", worst)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2])
    else:
        main(sys.argv[1])
