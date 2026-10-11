"""Properties of the self-employed health premium attribution.

The attribution flags a person when any one business a plan can be
established under (Schedule C, a Schedule F farm, a partnership) has strictly
positive net earnings, and copies the reported premium for flagged people
outside the Medicare proxy and measured employer coverage. These tests state
the invariants for every input and check them against an independent
row-wise reference, the pre-farm Schedule C rule, and the signal gate's own
recomputation of the identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.other_health_insurance as module
from microcosm.build.us_runtime.other_health_insurance import (
    US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS,
    US_SE_HEALTH_MEDICARE_AGE_THRESHOLD,
    US_SE_HEALTH_PLAN_BUSINESS_INCOME_SOURCES,
    US_SE_HEALTH_PLAN_BUSINESS_SOURCE_COLUMNS,
    attribute_us_se_health_premiums,
)
from test_support.microcosm_build.us_other_health_insurance import (
    _REPORTED,
    _attribution_source,
)

_PREMIUMS, _FLAG = US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS
_SCHEDULE_C = (
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
)
_FARM = "farm_operations_income"
_PARTNERSHIP = "partnership_self_employment_net_earnings"
_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Net earnings draw exact zeros, losses, profits and values next to zero, so
# strict positivity and cancelling Schedule C legs are both exercised.
_money = st.one_of(
    st.just(0.0),
    st.sampled_from([-1.0, -0.01, 0.01, 1.0, 5e-324, -5e-324]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
_premium = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False),
)
_row = st.fixed_dictionaries(
    {
        "reported": _premium,
        "self_employment": _money,
        "sstb": _money,
        "farm": _money,
        "partnership": _money,
        "age": st.integers(min_value=0, max_value=90).map(float),
        "ssdi": st.one_of(st.just(0.0), st.floats(min_value=0.01, max_value=4e4)),
        "esi": st.booleans(),
    }
)
_rows = st.lists(_row, min_size=1, max_size=25)


def _source(rows: list[dict[str, float | bool]]) -> pd.DataFrame:
    return _attribution_source(
        reported=[row["reported"] for row in rows],
        self_employment=[row["self_employment"] for row in rows],
        sstb=[row["sstb"] for row in rows],
        farm=[row["farm"] for row in rows],
        partnership=[row["partnership"] for row in rows],
        age=[row["age"] for row in rows],
        ssdi=[row["ssdi"] for row in rows],
        esi=[row["esi"] for row in rows],
    )


def _reference(row: dict[str, float | bool]) -> tuple[float, bool]:
    """Form 7206's businesses, one at a time, in plain Python."""

    businesses = (
        row["self_employment"] + row["sstb"],
        row["farm"],
        row["partnership"],
    )
    self_employed = any(net > 0.0 for net in businesses)
    medicare = row["age"] >= US_SE_HEALTH_MEDICARE_AGE_THRESHOLD or row["ssdi"] > 0.0
    attributed = self_employed and not medicare and not row["esi"]
    return (row["reported"] if attributed else 0.0), self_employed


def _schedule_c_only_rule(source: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The attribution before farm and partnership businesses counted."""

    schedule_c = (source[_SCHEDULE_C[0]] + source[_SCHEDULE_C[1]]).to_numpy()
    flag = schedule_c > 0.0
    medicare = (source["age"].to_numpy() >= US_SE_HEALTH_MEDICARE_AGE_THRESHOLD) | (
        source["social_security_disability"].to_numpy() > 0.0
    )
    premiums = np.where(
        flag & ~medicare & ~source["has_esi"].to_numpy(),
        source[_REPORTED].to_numpy(),
        0.0,
    )
    return premiums, flag


def test_plan_business_sources_are_the_form_7206_businesses() -> None:
    assert dict(US_SE_HEALTH_PLAN_BUSINESS_INCOME_SOURCES) == {
        "schedule_c": _SCHEDULE_C,
        "schedule_f": (_FARM,),
        "partnership": (_PARTNERSHIP,),
    }
    assert US_SE_HEALTH_PLAN_BUSINESS_SOURCE_COLUMNS == (
        *_SCHEDULE_C,
        _FARM,
        _PARTNERSHIP,
    )


@_SETTINGS
@given(rows=_rows)
def test_attribution_matches_the_row_wise_reference(rows) -> None:
    source = _source(rows)
    original = source.copy(deep=True)

    result = attribute_us_se_health_premiums(source)

    expected = [_reference(row) for row in rows]
    assert result[_PREMIUMS].tolist() == [premium for premium, _ in expected]
    assert result[_FLAG].tolist() == [flag for _, flag in expected]
    assert result[_FLAG].dtype == bool
    pd.testing.assert_frame_equal(source, original)

    # The businesses are alternatives, so their order cannot matter.
    reversed_order = attribute_us_se_health_premiums(
        source,
        plan_business_income_source_columns=dict(
            reversed(list(US_SE_HEALTH_PLAN_BUSINESS_INCOME_SOURCES.items()))
        ),
    )
    pd.testing.assert_frame_equal(reversed_order, result)


@_SETTINGS
@given(rows=_rows)
def test_attribution_never_invents_premium_mass(rows) -> None:
    source = _source(rows)

    result = attribute_us_se_health_premiums(source)

    premiums = result[_PREMIUMS].to_numpy()
    reported = source[_REPORTED].to_numpy()
    flag = result[_FLAG].to_numpy()
    assert np.all((premiums == 0.0) | (premiums == reported))
    assert np.all((premiums >= 0.0) & (premiums <= reported))
    assert premiums.sum() <= reported.sum()
    assert not np.any((premiums > 0.0) & ~flag)


@_SETTINGS
@given(rows=_rows)
def test_widening_only_adds_people_to_the_schedule_c_rule(rows) -> None:
    source = _source(rows)
    old_premiums, old_flag = _schedule_c_only_rule(source)

    widened = attribute_us_se_health_premiums(source)
    schedule_c_only = attribute_us_se_health_premiums(
        source,
        plan_business_income_source_columns={
            "schedule_c": US_SE_HEALTH_PLAN_BUSINESS_INCOME_SOURCES["schedule_c"]
        },
    )

    # Differential: the Schedule C-only configuration is the old rule exactly.
    assert schedule_c_only[_PREMIUMS].tolist() == old_premiums.tolist()
    assert schedule_c_only[_FLAG].tolist() == old_flag.tolist()
    # Monotone: everyone the old rule attributed keeps the same premium and
    # flag; the only new people have a profitable farm or partnership.
    new_flag = widened[_FLAG].to_numpy()
    new_premiums = widened[_PREMIUMS].to_numpy()
    assert np.all(new_flag[old_flag])
    np.testing.assert_array_equal(new_premiums[old_flag], old_premiums[old_flag])
    added = new_flag & ~old_flag
    other_profit = (source[_FARM].to_numpy() > 0.0) | (
        source[_PARTNERSHIP].to_numpy() > 0.0
    )
    assert np.all(other_profit[added])
    unchanged = ~other_profit
    np.testing.assert_array_equal(new_flag[unchanged], old_flag[unchanged])
    np.testing.assert_array_equal(new_premiums[unchanged], old_premiums[unchanged])


@_SETTINGS
@given(rows=_rows, replacement=st.floats(max_value=0.0, min_value=-1e6))
def test_a_loss_never_cancels_another_business(rows, replacement) -> None:
    source = _source(rows)
    result = attribute_us_se_health_premiums(source)

    # Only whether each business is profitable matters: every non-positive
    # farm or partnership amount can be swapped for any other without
    # changing the attribution.
    moved = source.copy(deep=True)
    for column in (_FARM, _PARTNERSHIP):
        moved[column] = np.where(moved[column] > 0.0, moved[column], replacement)
    pd.testing.assert_frame_equal(
        attribute_us_se_health_premiums(moved)[[_PREMIUMS, _FLAG]],
        result[[_PREMIUMS, _FLAG]],
    )


@_SETTINGS
@given(rows=_rows)
def test_signal_gate_recomputation_agrees_with_the_attribution(rows) -> None:
    source = _source(rows)
    result = attribute_us_se_health_premiums(source)

    # Differential: the gate recomputes the identity from shipped columns
    # with its own code path; it must find no divergence on true output.
    summary = module._se_attribution_summary(
        result,
        np.ones(len(result), dtype=np.float64),
        result[_REPORTED].to_numpy(dtype=np.float64),
    )
    assert summary["identity_violations"] == 0
    assert summary["flag_violations"] == 0
    assert summary["invalid_flag_values"] == 0
    assert summary["nonfinite_sources"] == {}
    assert summary["flag_rows"] == int(result[_FLAG].sum())
    assert summary["positive_plan_business_rows"] == {
        "schedule_c": int(
            ((source[_SCHEDULE_C[0]] + source[_SCHEDULE_C[1]]) > 0.0).sum()
        ),
        "schedule_f": int((source[_FARM] > 0.0).sum()),
        "partnership": int((source[_PARTNERSHIP] > 0.0).sum()),
    }
