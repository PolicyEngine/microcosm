import json
import os

import numpy as np
import pandas as pd

D = os.path.expanduser("~/reviews/pe30-population-2026-09-27/")
diag = json.load(open(D + "rel/calibration_diagnostics.json"))
T = diag["targets"]
print(
    T[1000]["name"],
    T[1000]["metadata"].get("geography_scope"),
    list(T[1000]["metadata"])[:14],
)


def row(t):
    m = t["metadata"]
    return dict(
        reg=t["registry"]["family"],
        src=m.get("source_variable"),
        role=m.get("target_role"),
        geo=m.get("geography_scope")
        or ("state" if "state" in t["name"] else "national"),
        t=t["target"],
        i=t["initial_estimate"],
        f=t["final_estimate"],
        name=t["name"],
    )


d = pd.DataFrame([row(x) for x in T])
d["ri"] = d.i / d.t.replace(0, np.nan) - 1
d["rf"] = d.f / d.t.replace(0, np.nan) - 1
g = d.groupby(["reg", "role", "src"], dropna=False).agg(
    n=("t", "size"),
    t=("t", "sum"),
    i=("i", "sum"),
    f=("f", "sum"),
    med_ri=("ri", "median"),
    med_rf=("rf", "median"),
)
g["agg_ri"] = g.i / g.t - 1
g["agg_rf"] = g.f / g.t - 1
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)
print(
    g.sort_values("agg_ri")
    .round(3)[["n", "agg_ri", "agg_rf", "med_ri", "med_rf"]]
    .to_string()
)
