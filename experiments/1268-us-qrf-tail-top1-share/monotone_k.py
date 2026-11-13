"""Is top_k_share exactly non-decreasing in k? Search legal partition arrangements, engine-free.

measure(k) = min(fl-sum(legal tail of k) / fl-sum(array), 1), with the carriers > k
partition path; k >= carriers returns exactly 1. Reports inputs where some legal
arrangement at k gives more than some legal arrangement at k + 1 (possible
somewhere), and inputs where every arrangement at k gives more than every
arrangement at k + 1 (so every platform shows the drop).
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from minimize import legal_tails  # noqa: E402


def possible(values, k):
    mass = np.abs(np.asarray(values, dtype=np.float64))
    if mass.size <= k:
        return {1.0}
    total = float(mass.sum())
    return {min(float(t.sum()) / total, 1.0) for t in legal_tails(mass, k)}


alphabet = [0.5, 1.0, 2.0, 3.0, 2.0**53, 2.0**53 + 2, 2.0**53 + 4, 2.0**54 + 4]
somewhere, everywhere = [], []
for n in (4, 5):
    for combo in itertools.product(alphabet, repeat=n):
        for k in range(1, n):
            low, high = possible(combo, k), possible(combo, k + 1)
            if max(low) > min(high):
                somewhere.append((combo, k, max(low), min(high)))
            if min(low) > max(high):
                everywhere.append((combo, k, min(low), max(high)))
print("drops possible under some legal arrangement:", len(somewhere))
for row in sorted(
    somewhere, key=lambda r: (len(r[0]), sum(v > 1e6 for v in r[0]), r[0])
)[:5]:
    print("  ex", [*row[0]], "k", row[1], repr(row[2]), "->", repr(row[3]))
print("drops under every legal arrangement:", len(everywhere))
for row in sorted(
    everywhere, key=lambda r: (len(r[0]), sum(v > 1e6 for v in r[0]), r[0])
)[:5]:
    print("  ex", [*row[0]], "k", row[1], repr(row[2]), "->", repr(row[3]))
