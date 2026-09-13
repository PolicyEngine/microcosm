"""Pure observed-age alias; no source, population or release authority.

The caller authenticates the integer A_AGE observations. This rule changes no
observation date, top code or raw value and reads no source/weight/domain label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE = "microcosm.us.observed-age-normalization.v1"
AGE_CONVENTION = "observed_interview_age_completed_years"
MAX_ROWS = 2_000_000
MAX_EXACT_FLOAT64_INTEGER = 2**53


def rule_document() -> dict:
    return {
        "protocol": RULE,
        "input": "A_AGE",
        "output": "age",
        "dtype": "float64",
        "relation": "numeric_identity",
        "age_convention": AGE_CONVENTION,
        "existing_nonmissing": "must_equal_observed_age",
        "temporal_adjustment": False,
        "top_code_replacement": False,
    }


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_OBSERVED_AGE_" + reason)


def normalize_observed_age(
    raw_age: pd.Series, incumbent_age: pd.Series | None = None
) -> pd.Series:
    """Return a defensive common age, accepting only verified numeric identity.

    Missing common cells may acquire the already observed value; missing or
    malformed raw ages refuse. The float bound is a representation limit, not
    a scientifically permitted maximum age. Measurement keeps its own range.
    """
    _require(isinstance(raw_age, pd.Series), "RAW_SERIES")
    _require(0 < len(raw_age) <= MAX_ROWS, "ROW_BOUND")
    _require(
        pd.api.types.is_integer_dtype(raw_age.dtype)
        and not pd.api.types.is_bool_dtype(raw_age.dtype)
        and not raw_age.isna().any(),
        "RAW_INTEGER_REQUIRED",
    )
    _require(
        ((raw_age >= 0) & (raw_age <= MAX_EXACT_FLOAT64_INTEGER)).all(),
        "RAW_REPRESENTATION",
    )
    values = raw_age.to_numpy(dtype=np.float64, copy=True)
    if incumbent_age is not None:
        _require(isinstance(incumbent_age, pd.Series), "INCUMBENT_SERIES")
        _require(
            incumbent_age.index.identical(raw_age.index)
            and len(incumbent_age) == len(raw_age),
            "ORDERED_INDEX",
        )
        present = incumbent_age.notna().to_numpy(dtype=bool)
        if present.any():
            _require(
                pd.api.types.is_numeric_dtype(incumbent_age.dtype)
                and not pd.api.types.is_bool_dtype(incumbent_age.dtype)
                and not pd.api.types.is_complex_dtype(incumbent_age.dtype),
                "INCUMBENT_NUMERIC_REQUIRED",
            )
            known = incumbent_age[present]
            _require(
                ((known >= 0) & (known <= MAX_EXACT_FLOAT64_INTEGER)).all(),
                "INCUMBENT_REPRESENTATION",
            )
            _require((known == raw_age[present]).all(), "INCUMBENT_CONFLICT")
            if pd.api.types.is_integer_dtype(incumbent_age.dtype):
                _require(
                    np.array_equal(
                        known.to_numpy(dtype=np.int64),
                        raw_age[present].to_numpy(dtype=np.int64),
                    ),
                    "INCUMBENT_CONFLICT",
                )
            incumbent = incumbent_age.to_numpy(dtype=np.float64, na_value=np.nan)
            _require(
                np.isfinite(incumbent[present]).all()
                and np.array_equal(incumbent[present], values[present]),
                "INCUMBENT_CONFLICT",
            )
            # Preserve the original float bits of every known incumbent cell.
            values[present] = incumbent[present]
    return pd.Series(values, index=raw_age.index.copy(deep=True), name="age")
