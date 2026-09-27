"""Is _frame_identity WEAKER than same_replayed_frame anywhere? NaN payloads, flags, index freq."""

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import survey_population_preparation as prep
from microcosm.build.us_runtime.survey_population_replay import same_replayed_frame
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def build(nan_bits=0x7FF8000000000011, flags=True, extra=None):
    person = pd.DataFrame(
        {
            "person_id": np.arange(3, dtype=np.int64),
            "person_household_id": np.array([0, 0, 1], dtype=np.int64),
            "person_tax_unit_id": np.array([0, 0, 1], dtype=np.int64),
            "person_spm_unit_id": np.array([0, 0, 1], dtype=np.int64),
            "person_family_id": np.array([0, 0, 1], dtype=np.int64),
            "person_marital_unit_id": np.array([0, 1, 2], dtype=np.int64),
            "native_float": np.array(
                [nan_bits, 0x3FF4000000000000, 0x8000000000000000], dtype=np.uint64
            ).view(np.float64),
        }
    )
    person.flags.allows_duplicate_labels = flags
    groups = {
        e: pd.DataFrame({f"{e}_id": np.array([0, 1], dtype=np.int64)})
        for e in ("household", "tax_unit", "spm_unit", "family")
    }
    groups["marital_unit"] = pd.DataFrame(
        {"marital_unit_id": np.array([0, 1, 2], dtype=np.int64)}
    )
    return Frame(
        {"person": person, **groups},
        US_SCHEMA,
        {"household": Weights(np.array([1.5, 2.5]), WeightKind.IMPORTANCE)},
        pd.Series(["urban", "urban", "rural"], dtype=object),
    )


def report(label, a, b):
    try:
        same_replayed_frame(a, b)
        cmp = "ACCEPTS"
    except ValueError as e:
        cmp = "REFUSES " + str(e)
    try:
        eq = prep._frame_identity(a) == prep._frame_identity(b)
        ident = "same" if eq else "differs"
    except Exception as e:
        ident = "ERR " + type(e).__name__ + ":" + str(e)
    print(f"{label:34s} same_replayed_frame={cmp:52s} _frame_identity={ident}")


# 1. NaN payload bits
report(
    "nan payload 0x...11 vs 0x...12",
    build(0x7FF8000000000011),
    build(0x7FF8000000000012),
)
# 2. DataFrame.flags
report("flags allows_duplicate_labels", build(flags=True), build(flags=False))
# 3. quiet vs signalling NaN
report("quiet vs signalling NaN", build(0x7FF8000000000000), build(0x7FF0000000000001))
