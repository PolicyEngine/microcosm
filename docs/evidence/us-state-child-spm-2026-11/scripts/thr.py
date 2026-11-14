import os

SP = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd

a = pd.read_parquet(os.path.join(SP, "asec/raw_asec_2022_2025.parquet"))
a["ref"] = a.SPM_POVTHRESHOLD / (a.SPM_GEOADJ * a.SPM_EQUIVSCALE)
u = a[a.A_AGE < 18]
yrcol = [c for c in a.columns if c == "income_year"][0]
g = u.groupby([yrcol, "SPM_TENMORTSTATUS"]).ref.median().unstack()
print(g.round(1).to_string())
print((g / g.shift(1) - 1).round(4).to_string())
print("mean GEOADJ by year", u.groupby(yrcol).SPM_GEOADJ.mean().round(4).to_dict())
print("spread ref within year", u.groupby(yrcol).ref.nunique().to_dict())
print(
    u.groupby(yrcol)
    .apply(
        lambda d: np.average(d.SPM_POVTHRESHOLD, weights=d.MARSUPWT),
        include_groups=False,
    )
    .round(0)
    .to_dict()
)
