import sys
import warnings

import pandas as pd
from policyengine_us import Microsimulation

warnings.filterwarnings("ignore")

sim = Microsimulation(dataset=sys.argv[1])
y = int(sys.argv[2])
df = pd.DataFrame(
    {
        "tax_unit_id": sim.calculate("tax_unit_id", y).values.astype("int64"),
        "agi_2024": sim.calculate("adjusted_gross_income", y).values.astype(float),
        "filing_status": sim.calculate("filing_status", y).values.astype(str),
        "tax_unit_weight": sim.calculate("tax_unit_weight", y).values.astype(float),
    }
)
df.to_parquet(sys.argv[3])
print(
    len(df),
    "tax units; AGI total $%.0fB" % ((df.agi_2024 * df.tax_unit_weight).sum() / 1e9),
)
