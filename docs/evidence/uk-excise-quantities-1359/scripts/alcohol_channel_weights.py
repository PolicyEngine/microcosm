"""How much allocating alcohol volume by spending moves it, against channel weights.

Local, licensed output (LCFS 2024-25): stays under data/ukds/acceptance.

Spending weights give every pound of alcohol spending the same volume. Channel
weights give the on-trade a published share v of national volume and the
off-trade 1 - v, each spread over its own spenders. Per pound, spending weights
over-allocate the on-trade by S_on / v and under-allocate the off-trade by
(1 - S_on) / (1 - v), where S_on is the on-trade share of alcohol spending.
This script measures S_on on the diary and the difference by equivalised
income decile, for a range of v.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

OFF = ("c21111", "c21211", "c21212", "c21213", "c21214", "c21221", "c21311")
ON = ("cb111c", "cb111d", "cb111e", "cb111f", "cb111g", "cb111h", "cb111i", "cb111j")


def main() -> None:
    tab, out = sys.argv[1], sys.argv[2]
    wanted = {*OFF, *ON, "cal_weight_annual", "p389p", "g018", "g019"}
    data = pd.read_csv(tab, sep="\t", usecols=lambda c: c.lower() in wanted, low_memory=False)
    data.columns = [c.lower() for c in data.columns]
    num = lambda c: pd.to_numeric(data[c], errors="coerce").fillna(0.0).to_numpy()  # noqa: E731
    w = num("cal_weight_annual")
    off = sum(np.maximum(num(c), 0.0) for c in OFF)
    on = sum(np.maximum(num(c), 0.0) for c in ON)
    scale = 1.0 + 0.5 * np.maximum(num("g018") - 1.0, 0.0) + 0.3 * num("g019")
    income = num("p389p") / scale
    order = np.argsort(income, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    decile = np.empty(len(w), dtype=int)
    decile[order] = np.minimum((cum * 10).astype(int), 9)
    s_on = float(w @ on / (w @ (off + on)))
    spending = (off + on) / (w @ (off + on))

    def channel(v: float) -> np.ndarray:
        return (1 - v) * off / (w @ off) + v * on / (w @ on)

    report = {
        "households": int(len(w)),
        "on_trade_share_of_alcohol_spending": s_on,
        "drinking_households_share": float(w[(off + on) > 0].sum() / w.sum()),
        "on_trade_only_share_of_drinkers": float(
            w[(on > 0) & (off == 0)].sum() / w[(off + on) > 0].sum()
        ),
        "off_trade_only_share_of_drinkers": float(
            w[(off > 0) & (on == 0)].sum() / w[(off + on) > 0].sum()
        ),
        "deciles": {},
        "by_v": {},
    }
    for d in range(10):
        mask = decile == d
        report["deciles"][d + 1] = {
            "support": "30+" if int(((off + on) > 0)[mask].sum()) >= 30 else "<30",
            "on_trade_share_of_its_alcohol_spending": float(
                w[mask] @ on[mask] / (w[mask] @ (off[mask] + on[mask]))
            ),
            "share_of_volume_spending_weights": float(w[mask] @ spending[mask]),
        }
    for v in (0.20, 0.25, 0.27, 0.30, 0.35):
        b = channel(v)
        moved = 0.5 * float(w @ np.abs(spending - b))
        report["by_v"][str(v)] = {
            "pub_pound_overallocation": s_on / v,
            "shop_pound_allocation": (1 - s_on) / (1 - v),
            "share_of_national_volume_moved": moved,
            "decile_share_channel": {
                d + 1: float(w[decile == d] @ b[decile == d]) for d in range(10)
            },
            "decile_spending_over_channel": {
                d + 1: float(
                    (w[decile == d] @ spending[decile == d])
                    / (w[decile == d] @ b[decile == d])
                )
                for d in range(10)
            },
        }
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps({k: report[k] for k in ("on_trade_share_of_alcohol_spending", "drinking_households_share", "on_trade_only_share_of_drinkers", "off_trade_only_share_of_drinkers")}, indent=1))
    for v in ("0.25", "0.27", "0.3"):
        r = report["by_v"][v]
        print(v, "pub x", round(r["pub_pound_overallocation"], 2), "shop x", round(r["shop_pound_allocation"], 2), "moved", round(r["share_of_national_volume_moved"], 3))
        print("   spending/channel by decile:", {k: round(x, 3) for k, x in r["decile_spending_over_channel"].items()})
    print("decile on-trade share of own spend:", {k: round(x["on_trade_share_of_its_alcohol_spending"], 2) for k, x in report["deciles"].items()})
    print("decile volume share (spending):", {k: round(x["share_of_volume_spending_weights"], 3) for k, x in report["deciles"].items()})
    print("decile volume share (channel 0.27):", {k: round(x, 3) for k, x in report["by_v"]["0.27"]["decile_share_channel"].items()})
    print("supports:", {k: x["support"] for k, x in report["deciles"].items()})


if __name__ == "__main__":
    main()
