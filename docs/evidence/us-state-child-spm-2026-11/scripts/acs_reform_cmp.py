import pandas as pd

a = pd.read_csv("run/acs_eval.csv")
r = pd.read_csv("run/acsrent_eval.csv")
key = [c for c in a.columns if c.lower() in ("state", "st", "abbr")][0]
print(list(a.columns))
dc = [c for c in a.columns if "delta" in c.lower() and "se" not in c.lower()][0]
sc = [c for c in a.columns if "delta" in c.lower() and "se" in c.lower()][0]
m = a[[key, dc, sc]].merge(r[[key, dc, sc]], on=key, suffixes=("_pub", "_rent"))
m = m[m[dc + "_rent"].notna()].copy()
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
# Like for like: how far the fix moves baselines and reform effects on the same states and on all 51.
base = a[[key, "acs_local"]].merge(
    r[[key, "acs_local"]], on=key, suffixes=("_pub", "_rent")
)
base["move"] = (base.acs_local_rent - base.acs_local_pub).abs()
same = base[base[key].isin(m[key])]
print(
    "baseline move, same 15: mean",
    same.move.mean().round(3),
    "median",
    same.move.median().round(3),
    "| all 51: mean",
    base.move.mean().round(3),
    "median",
    base.move.median().round(3),
)
print(
    "reform move, 15: mean",
    m["diff"].abs().mean().round(3),
    "median",
    m["diff"].abs().median().round(3),
    "| without FL: mean",
    m[m[key] != "FL"]["diff"].abs().mean().round(3),
    "| above 1 point",
    m.loc[m["diff"].abs() > 1, key].tolist(),
)
