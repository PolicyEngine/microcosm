import os

SP = os.path.dirname(os.path.abspath(__file__))
import os

import pandas as pd

S = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/"
)
f = S + "725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5"
h = pd.read_hdf(f, "household")[
    [
        "household_id",
        "state_fips",
        "household_support_channel",
        "household_source_id",
        "household_weight",
        "TYPEHUGQ",
    ]
]
h.to_parquet(f"{SP}/run/acs_hh.parquet")
p = pd.read_hdf(f, "person")[
    ["person_household_id", "person_spm_unit_id"]
].drop_duplicates()
p.to_parquet(f"{SP}/run/acs_unit_hh.parquet")
print(h.shape, p.shape)
