"""Tests split from packages/microcosm-build/tests/test_uk_student_loans.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_student_loans import *


def test_reported_cohort_boundaries_and_country_independence() -> None:
    # At 2025: these ages imply start years 2011, 2012, 2022, and 2023. The
    # last repays, and nobody repays a Plan 5 loan before April 2026, so its
    # course started before Plan 5 existed (microcosm#868).
    result = assign_student_loan_plans(
        _frame(
            ages=[32, 31, 21, 20],
            repayments=[100.0] * 4,
            regions=["WALES", "SCOTLAND", "NORTHERN_IRELAND", "WALES"],
        ),
        stocks=_stocks(plan_2=0, plan_5=0),
        year=2025,
    )

    assert result.frame.table("person")["student_loan_plan"].tolist() == [
        "PLAN_1",
        "PLAN_2",
        "PLAN_2",
        "PLAN_2",
    ]


def test_no_repayer_holds_plan_5_before_its_first_repayments() -> None:
    """SLC counts no Plan 5 borrower above the threshold in academic year
    2024/25: Plan 5 repayments start in April 2026 (microcosm#868)."""

    frame = _frame(
        ages=[20, 20],
        repayments=[100.0, 0.0],
        reports_outstanding=[False, True],
    )
    # Both courses date from 2023 at 2025 (no leaving age: started at 18).
    at_2025 = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=0, plan_5=0), year=2025
    )
    assert at_2025.frame.table("person")["student_loan_plan"].tolist() == [
        "PLAN_2",
        "PLAN_5",
    ]
    # From 2026 a repayer may hold Plan 5 (both courses then date from 2024).
    at_2026 = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=0, plan_5=0, year=2026), year=2026
    )
    assert at_2026.frame.table("person")["student_loan_plan"].tolist() == [
        "PLAN_5",
        "PLAN_5",
    ]


def test_course_start_dates_from_the_age_full_time_education_ended() -> None:
    """Three years before leaving full-time education; at 18 without an answer
    (microcosm#868)."""

    starts = course_start_year(
        year=2025,
        age=np.array([30.0, 30.0, 40.0, 30.0]),
        age_left=np.array([21, 25, 18, 0]),
    )
    # Left at 21 in 2016: started 2013. Left at 25 in 2020: started 2017, a
    # Plan 2 course the age rule (2013) would also place, unlike the 40-year-old
    # who left at 18 in 2003 and started in 2000. No answer: the age rule.
    assert starts.tolist() == [2013.0, 2017.0, 2000.0, 2013.0]


def test_an_outstanding_loan_reports_a_borrower_in_england_and_wales_only() -> None:
    reported = has_reported_student_loan(
        repayments=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0]),
        reports_outstanding=np.array([True, True, True, True, False, False]),
        in_education=np.array([False, False, False, True, False, False]),
        region_names=np.array(
            ["LONDON", "WALES", "SCOTLAND", "LONDON", "LONDON", "SCOTLAND"],
            dtype=object,
        ),
    )
    # A student is not liable; Scotland (Plan 4) and Northern Ireland take no
    # plan from the answer, and a repayer is reported anywhere, as before.
    assert reported.tolist() == [True, True, False, False, False, True]


def test_reported_borrowers_below_the_threshold_take_their_course_plan() -> None:
    result = assign_student_loan_plans(
        _frame(
            ages=[30, 30, 40, 30, 30],
            repayments=[0.0] * 5,
            regions=["LONDON", "WALES", "LONDON", "LONDON", "SCOTLAND"],
            reports_outstanding=[True, True, True, True, True],
            current_education=[
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "TERTIARY",
                "NOT_IN_EDUCATION",
            ],
            age_left=[21, 25, 18, 21, 21],
        ),
        stocks=_stocks(plan_2=0, plan_5=0),
        year=2025,
    )
    assert result.frame.table("person")["student_loan_plan"].tolist() == [
        "PLAN_2",
        "PLAN_2",
        "PLAN_1",
        "NONE",
        "NONE",
    ]
    receipt = result.plans["PLAN_2"]
    assert receipt.reported_count == 2.0 and receipt.reported_england_count == 1.0


def test_plan_2_walk_takes_nobody_in_full_time_education() -> None:
    result = assign_student_loan_plans(
        _frame(
            ages=[30, 30],
            repayments=[0.0, 0.0],
            current_education=["TERTIARY", "NOT_IN_EDUCATION"],
            age_left=[21, 21],
        ),
        stocks=_stocks(plan_2=2, plan_5=0),
        year=2025,
    )
    assert result.frame.table("person")["student_loan_plan"].tolist() == [
        "NONE",
        "PLAN_2",
    ]
    assert result.plans["PLAN_2"].eligible_rows == 1
    assert result.plans["PLAN_2"].pool_exhausted


def test_topups_apply_eligibility_gates_and_plan5_priority() -> None:
    result = assign_student_loan_plans(
        _frame(
            ages=[20, 20, 20, 31, 31, 40],
            repayments=[0.0] * 6,
            regions=["LONDON", "WALES", "LONDON", "LONDON", "LONDON", "LONDON"],
            education=["TERTIARY", "TERTIARY", "GCSE", "TERTIARY", "GCSE", "TERTIARY"],
        ),
        stocks=_stocks(plan_2=1, plan_5=1),
        year=2025,
    )
    plans = result.frame.table("person")["student_loan_plan"].tolist()

    assert plans == ["PLAN_5", "NONE", "NONE", "PLAN_2", "NONE", "NONE"]
    assert tuple(result.plans) == PLAN_PRIORITY
    assert "PLAN_4" not in plans


def test_walk_helper_takes_fits_skips_overshoots_and_bounds_the_gap() -> None:
    """The greedy walk in a hand-picked order (microcosm#1049)."""
    weights = np.array([5.0, 3.0, 2.0, 2.0, 1.0])
    order = np.arange(5)

    taken, skipped, lightest = _walk_top_up(weights, order, 4.0)
    assert taken.tolist() == [1, 4] and skipped == 3 and lightest == 2.0
    # A remainder no skipped person brings nearer stays open, inside the bound.
    taken, skipped, lightest = _walk_top_up(weights, order, 4.5)
    assert taken.tolist() == [1, 4] and skipped == 3 and lightest == 2.0
    assert abs(weights[taken].sum() - 4.5) < lightest
    # The skipped person nearest the remainder is taken when that helps.
    taken, skipped, lightest = _walk_top_up(np.array([4.0, 4.0]), np.arange(2), 3.0)
    assert taken.tolist() == [0] and skipped == 1 and lightest == 4.0
    assert abs(4.0 - 3.0) < lightest
    # Nobody fits and nobody helps: nothing is taken, the gap stays bounded.
    taken, skipped, lightest = _walk_top_up(np.array([4.0, 4.0]), np.arange(2), 1.0)
    assert taken.size == 0 and skipped == 2 and lightest == 4.0
    # A pool lighter than the shortfall is taken whole with nobody skipped.
    taken, skipped, lightest = _walk_top_up(np.array([2.0, 2.0]), np.arange(2), 10.0)
    assert taken.tolist() == [0, 1] and skipped == 0 and lightest is None


def test_top_up_walks_to_the_shortfall_within_one_skipped_weight() -> None:
    result = assign_student_loan_plans(
        _frame(
            ages=[31] * 5,
            repayments=[0.0] * 5,
            weights=[5.0, 3.0, 2.0, 2.0, 1.0],
        ),
        stocks=_stocks(plan_2=4, plan_5=0),
        year=2025,
    )
    receipt = result.plans["PLAN_2"]
    plans = result.frame.table("person")["student_loan_plan"].to_numpy()

    assert receipt.shortfall == 4.0
    assert receipt.eligible_rows == 5 and receipt.eligible_mass == 13.0
    assert not receipt.pool_exhausted
    # Either the walk skipped someone, and the gap sits inside their weight,
    # or it met the shortfall exactly without skipping.
    if receipt.rows_skipped_for_weight:
        assert abs(receipt.realization_gap) < receipt.lightest_skipped_weight
    else:
        assert receipt.realization_gap == 0.0
        assert receipt.lightest_skipped_weight is None
    assert receipt.realization_gap == receipt.topped_up_mass - receipt.shortfall
    assert receipt.topped_up_rows == int((plans == "PLAN_2").sum())
    assert receipt.final_england_count == receipt.topped_up_mass
    assert receipt.stock_attainment == receipt.final_england_count / 4.0


def test_pool_lighter_than_the_shortfall_is_taken_whole_and_receipted() -> None:
    result = assign_student_loan_plans(
        _frame(ages=[31, 31], repayments=[0.0, 0.0], weights=[2.0, 2.0]),
        stocks=_stocks(plan_2=10, plan_5=0),
        year=2025,
    )
    receipt = result.plans["PLAN_2"]

    assert result.frame.table("person")["student_loan_plan"].tolist() == ["PLAN_2"] * 2
    assert receipt.pool_exhausted
    assert receipt.rows_skipped_for_weight == 0
    assert receipt.lightest_skipped_weight is None
    assert receipt.topped_up_rows == 2 and receipt.topped_up_mass == 4.0
    assert receipt.realization_gap == -6.0
    assert receipt.stock_attainment == 0.4


def test_plan_at_or_above_its_stock_is_left_as_reported() -> None:
    result = assign_student_loan_plans(
        _frame(ages=[31, 31], repayments=[100.0, 0.0], weights=[5.0, 1.0]),
        stocks=_stocks(plan_2=3, plan_5=0),
        year=2025,
    )
    receipt = result.plans["PLAN_2"]

    assert result.frame.table("person")["student_loan_plan"].tolist() == [
        "PLAN_2",
        "NONE",
    ]
    assert receipt.shortfall == 0.0
    assert receipt.topped_up_rows == 0 and receipt.topped_up_mass == 0.0
    assert receipt.realization_gap == 0.0
    assert not receipt.pool_exhausted
    assert receipt.stock_attainment == 5.0 / 3.0


def test_walk_order_is_the_identity_keyed_uniform_order() -> None:
    """The stage walks the eligible pool in stable_identity_uniforms order.

    Frames pin the row order to ascending ids, so the keying is asserted by
    replaying the walk on the identity uniforms the declared seed and salt
    produce and matching the persons the stage topped up.
    """
    weights = np.array([5.0, 3.0, 2.0, 2.0, 1.0, 4.0, 1.5, 0.5])
    frame = _frame(ages=[31] * 8, repayments=[0.0] * 8, weights=weights)
    person = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=7, plan_5=0), year=2025
    ).frame.table("person")
    ids = frame.table("person")["person_id"].to_numpy()
    key = stable_identity_uniforms(
        ids, seed=STUDENT_LOAN_SEED, salt=PLAN_SALTS["PLAN_2"]
    )
    order = np.lexsort((ids, key))
    taken, _, _ = _walk_top_up(weights, order, 7.0)

    topped = set(ids[taken])
    assert topped == set(
        person.loc[person["student_loan_plan"] == "PLAN_2", "person_id"]
    )
    assert topped and topped != set(ids[: len(topped)])


def test_calibration_year_changes_cohort_assignment() -> None:
    frame = _frame(ages=[31], repayments=[100.0])

    at_2025 = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=0, plan_5=0), year=2025
    )
    at_2036 = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=0, plan_5=0, year=2036), year=2036
    )

    assert at_2025.frame.table("person").student_loan_plan.iloc[0] == "PLAN_2"
    assert at_2036.frame.table("person").student_loan_plan.iloc[0] == "PLAN_5"
    assert at_2036.calibration_year == 2036


def test_committed_stocks_pin_full_2025_to_2030_series() -> None:
    stocks = load_slc_liable_stocks()["plans"]

    assert stocks["plan_2"]["liable"] == {
        "2025": 8_940_000,
        "2026": 9_710_000,
        "2027": 10_360_000,
        "2028": 10_615_000,
        "2029": 10_600_000,
        "2030": 10_525_000,
    }
    assert stocks["plan_5"]["above_threshold"]["2030"] == 1_235_000


@pytest.mark.parametrize(
    "operation_index,parameter",
    [
        *[
            (0, name)
            for name in (
                "year_rule",
                "start_year_formula",
                "reported_repayment_test",
                "reported_country_gate",
                "reported_outstanding_test",
                "reported_outstanding_region_exclusions",
                "plan_1_before",
                "plan_5_from",
                "enum_domain",
                "plan_4_imputation",
            )
        ],
        *[
            (1, name)
            for name in (
                "priority",
                "resource",
                "stock_series",
                "year_rule",
                "age_min",
                "age_max",
                "cohort_start_formula",
                "cohort_start_min",
                "eligible_region_exclusions",
                "highest_education",
                "seed",
                "salt",
                "realization",
                "realization_bound",
            )
        ],
        *[
            (2, name)
            for name in (
                "priority",
                "resource",
                "stock_series",
                "year_rule",
                "age_min",
                "age_max",
                "cohort_start_formula",
                "current_education",
                "cohort_start_min",
                "cohort_start_max_exclusive",
                "eligible_region_exclusions",
                "highest_education",
                "seed",
                "salt",
                "realization",
                "realization_bound",
            )
        ],
    ],
)
def test_manifest_drift_assert_covers_every_reviewed_parameter(
    operation_index: int, parameter: str
) -> None:
    with pytest.raises((ValueError, KeyError), match="drifted|priority"):
        _assert_student_loans_stage_parameters(
            _drift(operation_index, parameter),
            stocks=load_slc_liable_stocks(),
            year=2025,
        )


def test_stock_drift_assert_rejects_2025_change() -> None:
    stocks = _stocks(plan_2=1, plan_5=10_000)
    with pytest.raises(ValueError, match="PLAN_2.*drifted"):
        _assert_student_loans_stage_parameters(_stage(), stocks=stocks, year=2025)


def test_drift_assert_rejects_extra_keys_and_operations() -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_student_loans_stage_parameters(
            _drift(2, "undeclared_extra_key"),
            stocks=load_slc_liable_stocks(),
            year=2025,
        )
    stage = _stage()
    extra = replace(stage, operations=(*stage.operations, stage.operations[-1]))
    with pytest.raises(ValueError, match="operation order drifted"):
        _assert_student_loans_stage_parameters(
            extra, stocks=load_slc_liable_stocks(), year=2025
        )


def test_top_up_receipt_records_the_taken_rows_lineage() -> None:
    """Item 4 of the #1045 review: the walk prefers light rows, and after the
    support split the lightest rows are its copies, so the receipt shows where
    the taken rows sit."""
    result = assign_student_loan_plans(
        _frame(
            ages=[31] * 5,
            repayments=[0.0] * 5,
            weights=[5.0, 3.0, 2.0, 2.0, 1.0],
            household_columns={
                "household_support_channel": ["frs", "spi", "frs", "spi", "frs"],
                "household_is_cgt_support_copy": [False, False, True, True, False],
                "household_is_capital_gains_clone": [False, True, False, False, True],
                "household_is_cgt_residential_clone": [
                    False,
                    False,
                    False,
                    True,
                    False,
                ],
            },
        ),
        stocks=_stocks(plan_2=4, plan_5=0),
        year=2025,
    )
    receipt = result.plans["PLAN_2"]
    lineage = receipt.topped_up_lineage
    plans = result.frame.table("person")["student_loan_plan"].to_numpy()
    taken = np.flatnonzero(plans == "PLAN_2")
    channels = np.asarray(["frs", "spi", "frs", "spi", "frs"])[taken]
    copies = np.asarray([False, False, True, True, False])[taken]
    clones = np.asarray([False, True, False, False, True])[taken]
    arms = np.asarray([False, False, False, True, False])[taken]
    weights = np.asarray([5.0, 3.0, 2.0, 2.0, 1.0])[taken]

    assert lineage["columns"] == [
        "household_support_channel",
        "household_is_cgt_support_copy",
        "household_is_capital_gains_clone",
        "household_is_cgt_residential_clone",
    ]
    assert lineage["rows_by_channel"] == {
        channel: int((channels == channel).sum()) for channel in sorted(set(channels))
    }
    assert lineage["mass_by_channel"] == {
        channel: float(weights[channels == channel].sum())
        for channel in sorted(set(channels))
    }
    assert lineage["support_copy_rows"] == int(copies.sum())
    assert lineage["support_copy_mass"] == float(weights[copies].sum())
    assert lineage["clone_rows"] == int(clones.sum())
    assert lineage["clone_mass"] == float(weights[clones].sum())
    assert lineage["residential_arm_rows"] == int(arms.sum())
    assert lineage["residential_arm_mass"] == float(weights[arms].sum())
    assert sum(lineage["rows_by_channel"].values()) == receipt.topped_up_rows
    assert receipt.evidence()["topped_up_lineage"] == lineage

    bare = assign_student_loan_plans(
        _frame(ages=[31] * 2, repayments=[0.0] * 2, weights=[2.0, 2.0]),
        stocks=_stocks(plan_2=10, plan_5=0),
        year=2025,
    )
    assert bare.plans["PLAN_2"].topped_up_lineage == {"columns": []}
