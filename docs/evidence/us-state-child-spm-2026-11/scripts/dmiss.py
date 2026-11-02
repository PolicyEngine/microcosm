import os

import numpy as np
import pandas as pd

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
z = np.load(
    S + "15cefac484174ae0ee11a82b164f58023ad05c84/populace_us_2024_calibration.npz",
    allow_pickle=True,
)
print(z.files)
n = pd.Series(z["target_names"].astype(str))
t = z["target_values"]
i = z["initial_estimates"]
f = z["final_estimates"]
print(n.sample(12, random_state=1).str[:90].to_string())
d = pd.DataFrame({"n": n, "t": t, "i": i, "f": f})
keys = {
    "snap_spend": r"snap.*(total|cost|benefit)|snap$",
    "snap": r"snap",
    "medicaid": r"medicaid",
    "chip": r"chip",
    "eitc": r"eitc",
    "aca": r"aca|ptc",
    "ctc": r"[^a-z]ctc",
    "wages": r"employment_income|wages",
    "agi": r"adjusted_gross_income",
    "income_tax": r"income_tax",
    "pop": r"population|age",
}
for k, p in keys.items():
    m = d.n.str.contains(p, regex=True, case=False)
    x = d[m]
    if len(x):
        print(
            k,
            len(x),
            "agg init miss",
            round(x.i.sum() / x.t.sum() - 1, 3),
            "agg final",
            round(x.f.sum() / x.t.sum() - 1, 3),
            "median rel init",
            round((x.i / x.t.replace(0, np.nan) - 1).median(), 3),
        )
