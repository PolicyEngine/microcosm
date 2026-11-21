"""Check the rerun: 51 rates against the deck's fill-0 run, and two states under -1 against the earlier runs. Then trace which units the fill moves."""

import json
import os
import re
import subprocess

import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
S = os.path.expanduser("~/PolicyEngine/policyengine-slides")
FIPS = eval(
    re.search(r"FIPS\s*=\s*(\{.*?\})", open(f"{SP}/acs_eval.py").read(), re.S).group(1)
)
AB = {v: k for k, v in FIPS.items()}
deck = (
    pd.DataFrame(
        json.loads(
            subprocess.run(
                [
                    "git",
                    "-C",
                    S,
                    "show",
                    "99ee5b354:slideshows/appam-2026/data/state_baseline.json",
                ],
                capture_output=True,
                text=True,
            ).stdout
        )["rows"]
    )
    .set_index("state")
    .child_spm_percent
)
earlier = pd.read_csv(f"{SP}/run/acs_eval_fill_minus1.csv", index_col=0).acs_local
h = pd.read_parquet(f"{SP}/run/acs_hh.parquet")
uh = pd.read_parquet(f"{SP}/run/acs_unit_hh.parquet").drop_duplicates(
    "person_spm_unit_id"
)


def rates(folder):
    u = pd.concat(
        [
            pd.read_parquet(f"{SP}/run/{folder}/{f}")
            for f in sorted(os.listdir(f"{SP}/run/{folder}"))
            if f.endswith(".parquet")
        ],
        ignore_index=True,
    )
    u = (
        u[u.kids > 0]
        .merge(uh, left_on="spm_unit_id", right_on="person_spm_unit_id")
        .merge(h, left_on="person_household_id", right_on="household_id")
    )
    u["W"] = u.household_weight * u.kids
    return u, (
        100
        * (u.W * u.poor).groupby(u.state_fips).sum()
        / u.W.groupby(u.state_fips).sum()
    ).rename(index=AB)


u0, r0 = rates("acs")
a = pd.DataFrame({"rerun": r0, "deck_99ee5b354": deck})
a["diff"] = a.rerun - a.deck_99ee5b354
print(
    "rates against the deck's fill-0 run: states",
    int(a.rerun.notna().sum()),
    "largest difference",
    a["diff"].abs().max(),
)
a.to_csv(f"{SP}/run/acs_rerun_vs_deck.csv")
u1, r1 = rates("check_fill-1")
print(
    "the split under -1 against the earlier -1 runs:",
    {s: (round(r1[s], 6), round(earlier[s], 6)) for s in r1.index},
)
for st in r1.index:
    fips = FIPS[st]
    m = u0[u0.state_fips == fips][
        ["spm_unit_id", "poor", "W", "household_support_channel", "household_source_id"]
    ].merge(u1[["spm_unit_id", "poor"]], on="spm_unit_id", suffixes=("_0", "_m1"))
    mv = m[m.poor_0 != m.poor_m1]
    T = m.W.sum()
    print(
        st,
        "units that change",
        len(mv),
        "by channel",
        mv.household_support_channel.astype(str).value_counts().to_dict(),
        "source households",
        mv.household_source_id.nunique(),
        "rate move",
        round(100 * (mv.W * (mv.poor_0 - mv.poor_m1)).sum() / T, 4),
        "largest source household's share of the state's child weight, percent",
        round(100 * mv.groupby("household_source_id").W.sum().max() / T, 3)
        if len(mv)
        else 0,
    )
