"""What the fill moves: each ACS-local evaluation under the -1 fill against the same evaluation under fill 0."""

import os

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
rows = []
for name in ("acs_eval", "acsrent_eval"):
    a = pd.read_csv(f"{SP}/run/{name}_fill_minus1.csv", index_col=0)
    b = pd.read_csv(f"{SP}/run/{name}.csv", index_col=0)
    for col in (
        "acs_local",
        "se",
        "acs_only",
        "acs_only_se",
        "donor_only",
        "donor_only_se",
        "delta",
        "delta_se",
        "kish_src",
        "acs_share",
    ):
        both = pd.concat([a[col], b[col]], axis=1, keys=("m1", "z")).dropna()
        d = both.z - both.m1
        rows.append(
            dict(
                file=name,
                measure=col,
                states=len(both),
                largest_move=d.abs().max(),
                state=d.abs().idxmax(),
                states_past_1e6=int((d.abs() > 1e-6).sum()),
                states_past_01=int((d.abs() > 0.1).sum()),
            )
        )
t = pd.DataFrame(rows)
t.to_csv(f"{SP}/run/fill_moves.csv", index=False)
print(t.round(4).to_string(index=False))
for name in ("acs_solves", "acsrent_solves"):
    a = pd.read_csv(f"{SP}/run/{name}_fill_minus1.csv", index_col=0)
    b = pd.read_csv(f"{SP}/run/{name}.csv", index_col=0)
    print(name)
    print(
        pd.concat(
            [a[["mean_gap", "mad", "beyond"]], b[["mean_gap", "mad", "beyond"]]],
            axis=1,
            keys=("fill_minus1", "fill_0"),
        )
        .round(2)
        .to_string()
    )
