"""Run the archived generator's 2024 uprating loop on a stand-in array set.

Usage: python run_archived_uprating_loop.py <usdata_utils_uprating_42ed5d45.py> <tmp_dir>

Executes policyengine-us-data 42ed5d45 `utils/uprating.py::
create_policyengine_uprating_factors_table` unchanged (with STORAGE_FOLDER
pointed at a temp dir) against the installed policyengine-us, then runs the
loop from `datasets/puf/puf.py` lines 1384-1393 verbatim on arrays keyed by
variable name, and reports whether any array changed.
"""

import importlib.metadata as md
import sys
import types
from pathlib import Path

import numpy as np

source_path, tmp_dir = sys.argv[1], Path(sys.argv[2])
storage = types.ModuleType("policyengine_us_data.storage")
storage.STORAGE_FOLDER = tmp_dir
package = types.ModuleType("policyengine_us_data")
sys.modules["policyengine_us_data"] = package
sys.modules["policyengine_us_data.storage"] = storage
namespace = {}
exec(compile(Path(source_path).read_text(), source_path, "exec"), namespace)
uprating = namespace["create_policyengine_uprating_factors_table"]()

print("policyengine-us", md.version("policyengine-us"))
print(
    "table index name:",
    uprating.index.name,
    "| columns:",
    list(uprating.columns)[:6],
    "...",
)
wage_rows = [
    name
    for name in uprating.index
    if "employment_income" in name and "self" not in name
]
print("wage rows in the table:", wage_rows)
for name in wage_rows:
    print(
        "  ", name, "2021:", uprating.loc[name, 2021], "2024:", uprating.loc[name, 2024]
    )
print("has a 'Variable' column:", "Variable" in uprating.columns)

arrays = {
    "employment_income": np.array([100.0, 200.0]),
    "employment_income_before_lsr": np.array([100.0, 200.0]),
    "taxable_interest_income": np.array([10.0]),
}
before = {k: v.copy() for k, v in arrays.items()}
time_period = 2024
iterated, entered = [], 0
# --- verbatim from datasets/puf/puf.py lines 1384-1393 (self.time_period -> time_period)
for variable in uprating:
    iterated.append(variable)
    if variable in arrays:
        entered += 1
        current_index = uprating[uprating.Variable == variable][time_period].values[0]
        start_index = uprating[uprating.Variable == variable][2021].values[0]
        growth = current_index / start_index
        arrays[variable] = arrays[variable] * growth
# ---
print("loop iterated over:", iterated[:4], "...", iterated[-1])
print("times the body ran:", entered)
print("arrays unchanged:", all(np.array_equal(arrays[k], before[k]) for k in arrays))
