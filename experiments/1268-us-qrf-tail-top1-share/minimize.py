"""Minimize the CI counterexample of microcosm#1268's rounding test, engine-free.

The gate and ``single_record_concentration`` sum their tail as
``np.partition(mass, -top_k)[-top_k:].sum()``. numpy fixes the summation order
of a contiguous slice (pairwise_sum in loops_utils.h.src) but leaves the order
of elements inside a partition undefined. A partition is legal when position
``kth = n - top_k`` holds the kth-smallest value, nothing before it is larger
and nothing after it smaller. This script sums the tail in every legal
arrangement, so its results hold on every platform.
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction

import numpy as np


def legal_tails(mass: np.ndarray, top_k: int):
    """Every distinct tail ``np.partition(mass, -top_k)[-top_k:]`` may return."""
    ordered = np.sort(mass)
    kth = mass.size - top_k
    first, rest = ordered[kth], ordered[kth + 1 :]
    for order in sorted(set(itertools.permutations(rest.tolist()))):
        yield np.array([first, *order])


def shares(values, top_k):
    mass = np.abs(np.asarray(values, dtype=np.float64))
    total = float(mass.sum())  # array order, fixed on every platform
    out = {}
    for tail in legal_tails(mass, top_k):
        out.setdefault(float(tail.sum()) / total, tail.tolist())
    exact = sum(map(Fraction, sorted(mass.tolist())[-top_k:])) / sum(
        map(Fraction, mass.tolist())
    )
    return total, out, float(exact)


def show(label, values, top_k):
    total, out, exact = shares(values, top_k)
    print(f"{label}: values={values} top_k={top_k}")
    print(f"  array-order total {total!r}; exact share rounds to {exact!r}")
    for share in sorted(out):
        print(f"  share {share!r:<22} e.g. tail order {out[share]}")
    return out


if __name__ == "__main__":
    ci = [2.0, 3.0, 2.0, 1.0, 1e16, 7.0, 1e15, 1.0, 0.5]
    out = show("CI input", ci, 8)
    print(
        "  introselect (macOS arm64) tail:",
        np.partition(np.array(ci), -8)[-8:].tolist(),
        "-> share",
        repr(float(np.partition(np.array(ci), -8)[-8:].sum()) / float(np.sum(ci))),
    )
    print(
        "  sorted tail (x86-simd-sort base case):",
        np.sort(ci)[-8:].tolist(),
        "-> share",
        repr(float(np.sort(ci)[-8:].sum()) / float(np.sum(ci))),
    )

    # top_k <= 2 never overshoots: search a grid that does overshoot at top_k = 3.
    alphabet = [0.5, 1.0, 2.0, 3.0, 7.0, 2.0**53, 2.0**53 + 2, 2.0**53 + 4, 1e15, 1e16]
    for top_k in (1, 2):
        worst = 0.0
        for n in range(top_k + 1, 6):
            for combo in itertools.product(alphabet, repeat=n):
                _, out, _ = shares(list(combo), top_k)
                worst = max(worst, max(out))
        print(
            f"top_k={top_k}: largest share over every grid input of 3-5 records and every legal tail: {worst!r}"
        )

    # Smallest gate-checkable input (top_k = 3, four carriers) whose every legal tail overshoots.
    found = []
    for combo in itertools.product(alphabet, repeat=4):
        _, out, _ = shares(list(combo), 3)
        if min(out) > 1.0:
            found.append(combo)
    print(
        f"4-record inputs, top_k=3, overshooting in every legal arrangement: {len(found)}"
    )
    key = lambda c: (sum(v > 1e6 for v in c), sorted(c), c)  # noqa: E731
    for combo in sorted(found, key=key)[:6]:
        print("  ", list(combo))
    show("minimized", [2.0**53 + 2, 3.0, 1.0, 0.5], 3)
    print("numpy", np.__version__, sys.platform)
