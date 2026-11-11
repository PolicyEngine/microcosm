"""A two-household calibrated national frame carrying the full ladder
geography, shared by the terminal-node and full-build CLI tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.local_authority_input import (
    resolve_local_authority_engine_keys,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import MassChangeRecord, WeightKind


def _frame():
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_clone_index": [0, 0],
            "region": ["LONDON", "SOUTH_EAST"],
            "oa_code": ["E00000001", "E00000002"],
            "lsoa_code": ["E01000001", "E01000002"],
            "msoa_code": ["E02000001", "E02000002"],
            # April 2023 roster codes: the ladder writes the engine's
            # ``local_authority`` member name beside the code (microcosm#953).
            "local_authority_code": ["E09000001", "E07000008"],
            "ward_code": ["E05000001", "E05000002"],
            "constituency_code": ["E14000001", "E14000002"],
            "region_code": ["E12000007", "E12000008"],
            "itl3_code": ["TLI31", "TLJ31"],
            "itl2_code": ["TLI3", "TLJ3"],
            "itl1_code": ["TLI", "TLJ"],
        }
    )
    household["local_authority"] = resolve_local_authority_engine_keys(
        household["local_authority_code"]
    )
    for column in household.select_dtypes(include=["str", "object"]).columns:
        household[column] = household[column].astype("string")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_clone_index": [0, 0],
                "income": pd.Series([10.5, 20.5], dtype="float32"),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2], "benunit_clone_index": [0, 0]}),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.CALIBRATED,
        household_weights=np.array([13.0, 87.0]),
        mass_log=(MassChangeRecord("household", 100.0, 100.0, 1.0, "calibration"),),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
