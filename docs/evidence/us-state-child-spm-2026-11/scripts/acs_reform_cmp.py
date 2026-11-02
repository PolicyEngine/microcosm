import pandas as pd

a = pd.read_csv("run/acs_eval.csv")
r = pd.read_csv("run/acsrent_eval.csv")
key = [c for c in a.columns if c.lower() in ("state", "st", "abbr")][0]
print(list(a.columns))
dc = [c for c in a.columns if "delta" in c.lower() and "se" not in c.lower()][0]
sc = [c for c in a.columns if "delta" in c.lower() and "se" in c.lower()][0]
m = a[[key, dc, sc]].merge(r[[key, dc, sc]], on=key, suffixes=("_pub", "_rent"))
m = m[m[dc + "_rent"] != 0].copy()
m["diff"] = m[dc + "_rent"] - m[dc + "_pub"]
print(m.round(2).to_string(index=False))
print(
    "n",
    len(m),
    "median pub",
    m[dc + "_pub"].median().round(2),
    "median rent",
    m[dc + "_rent"].median().round(2),
    "mean abs diff",
    m["diff"].abs().mean().round(2),
    "max",
    m["diff"].abs().max().round(2),
    "mean diff",
    m["diff"].mean().round(2),
)
print(
    "median se pub",
    m[sc + "_pub"].median().round(2),
    "rent",
    m[sc + "_rent"].median().round(2),
)
out = m.rename(
    columns={
        dc + "_pub": "ctc_effect_published",
        sc + "_pub": "ctc_effect_se_published",
        dc + "_rent": "ctc_effect_corrected_rent",
        sc + "_rent": "ctc_effect_se_corrected_rent",
        "diff": "corrected_minus_published",
    }
)
out.to_csv("run/acsrent_reform.csv", index=False)
