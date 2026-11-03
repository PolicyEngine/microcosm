"""A/B of the thin-row binding on one spine (microcosm#1095 U3, #1333).

Usage: ab_binding.py <control-run-dir> <treatment-run-dir> <spine-h5>

Both runs calibrate the same spine; the treatment binds the rows that María's rule of
2026-10-21 brought back (thin support alone leaves no row out). Every figure is recomputed
from each run's own target-support sidecars (the compiled matrix, the targets, the design and
final weights), so a row whose published figures are withheld is still measured here.

Prints, for each run: the share of rows within 10%, the mean absolute relative error and the
weight anatomy (Kish ESS, the share of design mass whose weight was stretched 9.5 times or
more, and the max over median of the positive weights), all over the rows bound in BOTH runs;
the shared rows that cross the 10% line either way; and, per family of newly bound rows, the
fit of each row in both runs (the control's weights read through the treatment's matrix row),
its carriers' support band and how far the binding stretched their weights.

Its output describes single rows and small groups of licensed records, so it stays with the
licensed spine, outside the repository; experiments/1095-uk-thin-row-binding-receipts.md
gives what may be published.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

STRETCH = 9.5


def band(n: int) -> str:
    return "<10" if n < 10 else ("10-29" if n < 30 else "30+")


def family(name: str) -> str:
    if name.startswith("dwp.benefit_cap."):
        return "benefit-cap amount bands"
    if "uc_payment_dist" in name or "payment_distribution" in name:
        return "UC payment bands"
    if name.startswith("hmrc/"):
        return "HMRC 500k-1m cells"
    return "other"


def load(run: Path) -> dict:
    matrix = sparse.csr_matrix(sparse.load_npz(run / "target_support_matrix.npz"))
    vectors = np.load(run / "target_support_vectors.npz", allow_pickle=True)
    names = [str(n) for n in vectors["names"]]
    return {
        "matrix": matrix,
        "names": names,
        "index": {n: i for i, n in enumerate(names)},
        "target": np.asarray(vectors["target_vector"], dtype=float),
        "design": np.asarray(vectors["design_weights"], dtype=float),
        "final": np.asarray(vectors["final_weights"], dtype=float),
        "ids": np.asarray(vectors["household_ids"])
        if "household_ids" in vectors
        else None,
    }


def errors(run: dict, weights: np.ndarray) -> np.ndarray:
    estimate = run["matrix"] @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        return (estimate - run["target"]) / run["target"]


def anatomy(run: dict) -> dict:
    final, design = run["final"], run["design"]
    positive = final > 0
    ratio = np.divide(final, design, out=np.zeros_like(final), where=design > 0)
    return {
        "ess": float(final.sum() ** 2 / np.square(final).sum()),
        "stretched_mass": float(design[ratio >= STRETCH].sum() / design.sum()),
        "max_over_median": float(final[positive].max() / np.median(final[positive])),
    }


def main() -> None:
    control, treatment = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    if control["ids"] is not None and treatment["ids"] is not None:
        assert np.array_equal(control["ids"], treatment["ids"]), (
            "the runs order households differently"
        )
    with pd.HDFStore(sys.argv[3], mode="r") as store:
        household = store.select(
            "household", columns=["household_id", "source_household_id"]
        )
    source = (
        household.set_index("household_id")["source_household_id"]
        .reindex(treatment["ids"])
        .to_numpy()
        if treatment["ids"] is not None
        else None
    )
    shared = sorted(set(control["names"]) & set(treatment["names"]))
    added = sorted(set(treatment["names"]) - set(control["names"]))
    dropped = sorted(set(control["names"]) - set(treatment["names"]))
    for label, run in (("control", control), ("treatment", treatment)):
        rel = errors(run, run["final"])
        picked = np.asarray([rel[run["index"][n]] for n in shared])
        a = anatomy(run)
        print(
            f"== {label}: {len(run['names'])} rows; over the {len(shared)} shared rows "
            f"{np.mean(np.abs(picked) <= 0.10):.4f} within 10%, mean |error| {np.nanmean(np.abs(picked)):.4f}; "
            f"ESS {a['ess']:.0f}; design mass stretched {STRETCH}x+ {a['stretched_mass']:.4f}; "
            f"LOCAL max/median {a['max_over_median']:.1f}"
        )
    print(
        f"rows bound in the treatment only: {len(added)}; in the control only: {dropped}"
    )
    rel_c = errors(control, control["final"])
    rel_t = errors(treatment, treatment["final"])
    out_of, into = [], []
    for name in shared:
        c, t = rel_c[control["index"][name]], rel_t[treatment["index"][name]]
        if abs(c) <= 0.10 < abs(t):
            out_of.append(name)
        if abs(t) <= 0.10 < abs(c):
            into.append(name)
    print(f"shared rows leaving 10% under the binding: {len(out_of)} {out_of}")
    print(f"shared rows entering 10% under the binding: {len(into)} {into}")
    # Newly bound rows: the control's weights read through the treatment's matrix row.
    by_family: dict[str, list[str]] = defaultdict(list)
    for name in added:
        by_family[family(name)].append(name)
    ratio_c = np.divide(
        control["final"],
        control["design"],
        out=np.zeros_like(control["final"]),
        where=control["design"] > 0,
    )
    ratio_t = np.divide(
        treatment["final"],
        treatment["design"],
        out=np.zeros_like(treatment["final"]),
        where=treatment["design"] > 0,
    )
    for fam, names in sorted(by_family.items()):
        print(f"-- {fam}: {len(names)} rows (LOCAL ONLY below)")
        carriers_all = np.zeros(len(treatment["final"]), dtype=bool)
        for name in names:
            row = treatment["matrix"][treatment["index"][name]]
            cols = row.indices[row.data != 0]
            carriers_all[cols] = True
            target = treatment["target"][treatment["index"][name]]
            est_c = float(np.ravel(row @ control["final"])[0])
            est_t = float(np.ravel(row @ treatment["final"])[0])
            support = band(len(pd.unique(source[cols]))) if source is not None else "?"
            print(
                f"   {name}: support {support}; error unbound {est_c / target - 1:+.3f} -> bound {est_t / target - 1:+.3f}; "
                f"carriers stretched {STRETCH}x+ {np.mean(ratio_c[cols] >= STRETCH) if len(cols) else 0:.2f} -> {np.mean(ratio_t[cols] >= STRETCH) if len(cols) else 0:.2f}"
            )
        design = treatment["design"]
        mass = design[carriers_all].sum()
        print(
            f"   family carriers' design mass stretched {STRETCH}x+: control "
            f"{design[carriers_all & (ratio_c >= STRETCH)].sum() / mass if mass else 0:.3f}, treatment "
            f"{design[carriers_all & (ratio_t >= STRETCH)].sum() / mass if mass else 0:.3f}"
        )


if __name__ == "__main__":
    main()
