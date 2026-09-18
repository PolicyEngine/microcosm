"""Pure invented descriptions; no source files, qualification, models or engines."""

import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.us_runtime.current_asec_property_basis import (
    build_asec_property_basis,
)
from microcosm.build.us_runtime.property_income_constants import (
    PROPERTY_COMPONENTS,
    PROPERTY_REPORTED_TOTAL,
)


def code(frame, name, values, statuses=None):
    frame[name + "_code"] = pd.array(values, dtype="Int16")
    frame[name + "_literal_status"] = pd.array(
        statuses or ["in_printed_range"] * len(values), dtype="string"
    )


def amount(frame, name, values, statuses=None):
    if statuses is None:
        statuses = ["known_nonreceipt" if v == 0 else "known_receipt" for v in values]
    frame[name + "_published_amount"] = pd.array(values, dtype="Float64")
    frame[name + "_literal_status"] = pd.array(
        ["in_printed_range"] * len(values), dtype="string"
    )
    frame[name + "_reporting_status"] = pd.array(statuses, dtype="string")
    known = np.isin(
        statuses, ["known_receipt", "known_nonreceipt", "observed_zero_component"]
    )
    frame[name + "_amount_known"] = known
    frame[name + "_amount"] = frame[name + "_published_amount"].where(known)


def inputs(n=3):
    index = pd.Index(np.arange(n, dtype=np.int64) + 101, name="person_id")
    base = pd.DataFrame(
        {
            "native_person_id": np.arange(n, dtype=np.int64) + 901,
            "source_age": np.full(n, 40.0),
        },
        index=index,
    )
    interest, routing, dividend = (base.copy() for _ in range(3))
    amount(interest, "INT_VAL", [100.0] * n)
    amount(interest, "TRDINT_VAL", [100.0] * n)
    code(interest, "RINT_YN", [2] * n)
    for k in (1, 2):
        amount(interest, f"RINT_VAL{k}", [0.0] * n)
        code(interest, f"RINT_SC{k}", [0] * n)
    interest["TRINT_VAL1_literal"] = pd.array(["0"] * n, dtype="string")
    interest["allocation_origin"] = pd.array(
        ["publisher_allocated"] * n, dtype="string"
    )
    routing["net_property_known_amount"] = np.full(n, -110.0)
    routing["net_property_reporting_status"] = pd.array(
        ["known_receipt"] * n, dtype="string"
    )
    code(routing, "other_income_receipt", [2] * n)
    code(routing, "other_income_category", [0] * n)
    routing["other_income_reporting_status"] = pd.array(
        ["known_nonreceipt"] * n, dtype="string"
    )
    routing["other_income_routing_status"] = pd.array(
        ["niu_category"] * n, dtype="string"
    )
    amount(dividend, "DIV_VAL", [0.0] * n)
    code(dividend, "DIV_YN", [2] * n)
    code(dividend, "SUR_YN", [2] * n)
    code(dividend, "SUR_SC1", [0] * n)
    code(dividend, "SUR_SC2", [0] * n)
    dividend["survivor_property_route_clear"] = pd.array([True] * n, dtype="boolean")
    membership = pd.Series(
        np.arange(n, dtype=np.int64) // 2 + 41, index=index, name="household_id"
    )
    households = np.unique(membership)
    weights = pd.Series(
        np.arange(len(households), dtype=np.float64) * 3 + 2,
        index=pd.Index(households, name="household_id"),
    )
    return dict(
        interest=interest,
        income_routing=routing,
        dividend=dividend,
        original_household_membership=membership,
        original_household_design_weights=weights,
    )


@pytest.mark.parametrize("rental,total", [(-110, -10), (-100, 0), (-40, 60), (50, 150)])
def test_signed_anchors_preserve_positive_interest_and_losses(rental, total):
    source = inputs()
    source["income_routing"]["net_property_known_amount"] = float(rental)
    before = copy.deepcopy(source)
    result = build_asec_property_basis(**source)
    assert result.person[PROPERTY_COMPONENTS[0]].eq(100).all()
    assert result.person[PROPERTY_COMPONENTS[3]].eq(rental).all()
    assert result.person[PROPERTY_REPORTED_TOTAL].eq(total).all()
    assert result.person.joint_component_fit_eligible.all()
    assert result.person.retirement_structural_zero_slot_count.eq(0).all()
    assert (
        result.provenance["interest.allocation_origin"].eq("publisher_allocated").all()
    )
    for name in ("interest", "income_routing", "dividend"):
        assert_frame_equal(source[name], before[name])
    result.provenance.iloc[0, 0] = -9
    assert source["interest"].iloc[0, 0] == 901


def test_named_unused_slot_derivation_keeps_published_niu_unknown():
    source = inputs(1)
    i = source["interest"]
    code(i, "RINT_YN", [1])
    code(i, "RINT_SC1", [4])
    amount(i, "RINT_VAL1", [20])
    amount(i, "RINT_VAL2", [0], ["unreported_account_slot"])
    amount(i, "INT_VAL", [120])
    result = build_asec_property_basis(**source)
    row = result.person.iloc[0]
    assert row.property_retirement_interest == 20
    assert row.joint_component_fit_eligible
    assert row.retirement_structural_zero_slot_count == 1
    assert row.retirement_slot2_derived_unused_zero
    assert (
        row.retirement_interest_derivation
        == "declared_account_sum_with_unused_slot_zero"
    )
    assert pd.isna(result.provenance.iloc[0]["interest.RINT_VAL2_amount"])
    assert result.provenance.iloc[0]["interest.RINT_VAL2_published_amount"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "zero_declared",
        "missing_code",
        "missing_amount",
        "all_unused",
        "niu_receipt",
        "no_with_account",
    ],
)
def test_unresolved_retirement_never_completes_unknown_slots(case):
    source = inputs(1)
    i = source["interest"]
    code(i, "RINT_YN", [1])
    code(i, "RINT_SC1", [4])
    amount(i, "RINT_VAL1", [20])
    amount(i, "RINT_VAL2", [0], ["unreported_account_slot"])
    if case == "zero_declared":
        code(i, "RINT_SC2", [4])
        amount(i, "RINT_VAL2", [0], ["ambiguous_recipient_zero"])
    elif case == "missing_code":
        code(i, "RINT_SC2", [None], ["missing"])
    elif case == "missing_amount":
        amount(i, "RINT_VAL2", [np.nan], ["missing_amount"])
    elif case == "all_unused":
        code(i, "RINT_SC1", [0])
        amount(i, "RINT_VAL1", [0], ["unreported_account_slot"])
    elif case == "niu_receipt":
        code(i, "RINT_YN", [0])
    else:
        code(i, "RINT_YN", [2])
    row = build_asec_property_basis(**source).person.iloc[0]
    assert np.isnan(row.property_retirement_interest)
    assert row.reported_total_eligible
    assert not row.joint_component_fit_eligible
    assert row.retirement_structural_zero_slot_count == 0


def test_two_active_accounts_sum_without_a_tax_label():
    source = inputs(1)
    i = source["interest"]
    code(i, "RINT_YN", [1])
    for n, value in ((1, 20), (2, 30)):
        code(i, f"RINT_SC{n}", [n])
        amount(i, f"RINT_VAL{n}", [value])
    amount(i, "INT_VAL", [150])
    result = build_asec_property_basis(**source)
    assert result.person.property_retirement_interest.iloc[0] == 50
    assert result.person.joint_component_fit_eligible.iloc[0]
    assert (
        result.person.retirement_interest_derivation.iloc[0] == "declared_account_sum"
    )


def test_reported_aggregate_discrepancy_is_unprojected_and_excluded():
    source = inputs(1)
    amount(source["interest"], "INT_VAL", [110])
    result = build_asec_property_basis(**source)
    row = result.person.iloc[0]
    assert row.property_reported_total == 0
    assert row.property_component_sum == -10
    assert row.interest_component_discrepancy == 10
    assert row.reported_minus_component_total == 10
    assert row.reported_total_eligible and not row.joint_component_fit_eligible
    assert result.exclusions.interest_discrepancy_nonzero.iloc[0]
    assert not any("cause" in column for column in result.person)


@pytest.mark.parametrize(
    "receipt,category,status,route,clear,overlap",
    [
        (1, 20, "known_receipt", "reported_category", True, False),
        (1, 2, "ambiguous_recipient_zero", "reported_category", True, False),
        (1, 5, "known_receipt", "reported_category", False, True),
        (1, 8, "known_receipt", "reported_category", False, True),
        (1, 19, "known_receipt", "reported_category", False, False),
        (1, 0, "known_receipt", "receipt_without_category", False, False),
        (2, 5, "known_nonreceipt", "category_without_receipt", False, False),
        (0, 0, "niu", "niu_category", False, False),
        (
            None,
            None,
            "missing_receipt_literal",
            "missing_category_literal",
            False,
            False,
        ),
    ],
)
def test_other_income_routes_are_strict_and_never_added(
    receipt, category, status, route, clear, overlap
):
    source = inputs(1)
    r = source["income_routing"]
    code(
        r,
        "other_income_receipt",
        [receipt],
        ["missing" if receipt is None else "in_printed_range"],
    )
    code(
        r,
        "other_income_category",
        [category],
        ["missing" if category is None else "in_printed_range"],
    )
    r["other_income_reporting_status"] = status
    r["other_income_routing_status"] = route
    result = build_asec_property_basis(**source)
    assert bool(result.person.reported_total_eligible.iloc[0]) == clear
    assert bool(result.exclusions.other_income_possible_property.iloc[0]) == overlap
    assert result.person.property_reported_total.iloc[0] == -10


@pytest.mark.parametrize(
    "receipt,first,second,declared",
    [
        (1, 1, 0, True),
        (1, 0, 9, True),
        (1, 8, 0, False),
        (1, 8, None, False),
        (1, 10, 0, None),
        (1, 1, None, None),
        (1, 0, 0, None),
        (2, 0, 0, True),
        (2, 8, 0, None),
        (0, 0, 0, None),
    ],
)
def test_survivor_routes_keep_overlap_and_unknown_exclusions(
    receipt, first, second, declared
):
    source = inputs(1)
    d = source["dividend"]
    for name, value in (("SUR_YN", receipt), ("SUR_SC1", first), ("SUR_SC2", second)):
        code(d, name, [value], ["missing" if value is None else "in_printed_range"])
    d["survivor_property_route_clear"] = pd.array([declared], dtype="boolean")
    result = build_asec_property_basis(**source)
    full_clear = receipt == 2 and declared is True
    assert bool(result.person.reported_total_eligible.iloc[0]) is full_clear
    assert bool(result.person.joint_component_fit_eligible.iloc[0]) is full_clear
    assert bool(result.person.survivor_visible_routes_clear.iloc[0]) is (
        declared is True
    )
    assert bool(result.exclusions.survivor_additional_sources_unresolved.iloc[0]) is (
        receipt == 1
    )
    assert bool(result.exclusions.survivor_possible_property.iloc[0]) is (
        declared is False
    )


def test_visible_survivor_clearance_does_not_clear_unobserved_extra_sources():
    source = inputs(3)
    d = source["dividend"]
    code(d, "SUR_YN", [1, 1, 2])
    code(d, "SUR_SC1", [1, 8, 0])
    d["survivor_property_route_clear"] = pd.array([True, False, True], dtype="boolean")
    before = d.copy(deep=True)
    result = build_asec_property_basis(**source)
    assert result.person.survivor_visible_routes_clear.tolist() == [True, False, True]
    assert result.person.survivor_full_scope_clear.tolist() == [False, False, True]
    assert result.person.joint_component_fit_eligible.tolist() == [False, False, True]
    assert result.person.property_reported_total.eq(-10).all()
    reason = result.summary.loc["excluded:survivor_additional_sources_unresolved"]
    assert reason.person_count == 2 and reason.household_count == 1
    assert reason.design_weighted_person_mass == 4
    assert reason.union_household_design_mass == 2
    assert result.exclusions.survivor_possible_property.tolist() == [False, True, False]
    assert_frame_equal(source["dividend"], before)
    assert_frame_equal(
        result.provenance.filter(like="dividend."), before.add_prefix("dividend.")
    )


@pytest.mark.parametrize(
    "axis", ["person", "native", "age", "duplicate_native", "duplicate_person"]
)
def test_both_identity_axes_and_age_must_match_exactly(axis):
    source = inputs()
    d = source["dividend"]
    if axis == "person":
        source["dividend"] = d.iloc[::-1]
    elif axis == "native":
        d["native_person_id"] = d.native_person_id.to_numpy()[::-1]
    elif axis == "age":
        d.iloc[0, d.columns.get_loc("source_age")] = 41
    elif axis == "duplicate_native":
        d["native_person_id"] = 901
    else:
        d.index = pd.Index([101, 101, 103], name="person_id")
    with pytest.raises(ValueError):
        build_asec_property_basis(**source)


def test_joint_permutation_and_household_weight_order_preserve_result():
    source = inputs()
    original = build_asec_property_basis(**source)
    for key in (
        "interest",
        "income_routing",
        "dividend",
        "original_household_membership",
    ):
        source[key] = source[key].iloc[[2, 0, 1]]
    source["original_household_design_weights"] = source[
        "original_household_design_weights"
    ].iloc[::-1]
    permuted = build_asec_property_basis(**source)
    assert_frame_equal(original.person, permuted.person.loc[original.person.index])
    assert_frame_equal(original.summary, permuted.summary)


def test_weighted_exclusions_report_person_and_union_household_mass():
    source = inputs()
    source["interest"].loc[101, "INT_VAL_amount"] = 110
    result = build_asec_property_basis(**source)
    all_rows = result.summary.loc["all"]
    excluded = result.summary.loc["excluded_joint_component_fit"]
    assert all_rows.design_weighted_person_mass == 9
    assert all_rows.union_household_design_mass == 7
    assert excluded.person_count == 1
    assert excluded.design_weighted_person_mass == 2
    assert excluded.union_household_design_mass == 2


@pytest.mark.parametrize(
    "case",
    [
        "extra_household",
        "missing_household",
        "duplicate_household",
        "bad_person_order",
        "negative_weight",
        "infinite_weight",
    ],
)
def test_only_exact_original_household_design_mapping_is_admitted(case):
    source = inputs()
    weights = source["original_household_design_weights"]
    if case == "extra_household":
        weights.loc[999] = 4
    elif case == "missing_household":
        source["original_household_design_weights"] = weights.iloc[:1]
    elif case == "duplicate_household":
        weights.index = pd.Index([41, 41], name="household_id")
    elif case == "bad_person_order":
        source["original_household_membership"] = source[
            "original_household_membership"
        ].iloc[::-1]
    else:
        weights.iloc[0] = -1 if case == "negative_weight" else np.inf
    with pytest.raises(ValueError):
        build_asec_property_basis(**source)


def test_empty_and_zero_weight_records_are_retained():
    empty = build_asec_property_basis(**inputs(0))
    assert len(empty.person) == 0
    assert empty.summary.person_count.eq(0).all()
    source = inputs()
    source["original_household_design_weights"].iloc[:] = 0
    result = build_asec_property_basis(**source)
    assert len(result.person) == 3 and result.person.joint_component_fit_eligible.all()
    assert result.summary.design_weighted_person_mass.eq(0).all()


def test_under15_unknowns_are_excluded_without_analytic_zero():
    source = inputs(1)
    for key in ("interest", "income_routing", "dividend"):
        source[key]["source_age"] = 14.0
    i = source["interest"]
    for name in ("INT_VAL", "TRDINT_VAL", "RINT_VAL1", "RINT_VAL2"):
        amount(i, name, [0], ["outside_reporting_universe"])
    code(i, "RINT_YN", [0])
    d = source["dividend"]
    amount(d, "DIV_VAL", [0], ["outside_reporting_universe"])
    code(d, "DIV_YN", [0])
    code(d, "SUR_YN", [0])
    d["survivor_property_route_clear"] = pd.array([None], dtype="boolean")
    r = source["income_routing"]
    r["net_property_known_amount"] = np.nan
    r["net_property_reporting_status"] = "outside_reporting_universe"
    result = build_asec_property_basis(**source)
    assert result.person[list(PROPERTY_COMPONENTS)].isna().all().all()
    assert result.exclusions.under15.iloc[0]
    assert not result.person.reported_total_eligible.iloc[0]


@pytest.mark.parametrize(
    "case",
    [
        "nonfinite",
        "negative_interest",
        "false_knownness",
        "dividend_no_positive",
        "rent_receipt_zero",
        "survivor_false_clear",
        "overflow",
    ],
)
def test_invalid_descriptions_and_overflow_refuse(case):
    source = inputs(1)
    if case in ("nonfinite", "negative_interest"):
        source["interest"].loc[101, "TRDINT_VAL_amount"] = (
            np.inf if case == "nonfinite" else -1
        )
    elif case == "false_knownness":
        source["interest"]["INT_VAL_amount_known"] = False
    elif case == "dividend_no_positive":
        amount(source["dividend"], "DIV_VAL", [10])
    elif case == "rent_receipt_zero":
        source["income_routing"]["net_property_known_amount"] = 0.0
    elif case == "survivor_false_clear":
        source["dividend"]["survivor_property_route_clear"] = pd.array(
            [False], dtype="boolean"
        )
    else:
        amount(source["interest"], "INT_VAL", [1.7e308])
        source["income_routing"]["net_property_known_amount"] = 1.7e308
    with pytest.raises((ValueError, FloatingPointError)):
        build_asec_property_basis(**source)


def test_weight_summaries_are_stable_under_widely_different_weight_permutation():
    source = inputs()
    source["original_household_membership"] = pd.Series(
        [41, 42, 43], index=source["interest"].index, dtype="int64"
    )
    source["original_household_design_weights"] = pd.Series(
        [1e16, 1.0, 1.0], index=pd.Index([41, 42, 43], name="household_id")
    )
    original = build_asec_property_basis(**source)
    for key in (
        "interest",
        "income_routing",
        "dividend",
        "original_household_membership",
    ):
        source[key] = source[key].iloc[::-1]
    reordered = build_asec_property_basis(**source)
    assert_frame_equal(original.summary, reordered.summary)
    assert original.summary.loc["all", "design_weighted_person_mass"] == 1e16 + 2


def test_fully_qualified_nonreceipt_zeros_remain_eligible():
    source = inputs(1)
    for name in ("INT_VAL", "TRDINT_VAL"):
        amount(source["interest"], name, [0])
    source["income_routing"]["net_property_known_amount"] = 0.0
    source["income_routing"]["net_property_reporting_status"] = "known_nonreceipt"
    row = build_asec_property_basis(**source).person.iloc[0]
    assert row[list(PROPERTY_COMPONENTS)].eq(0).all()
    assert row.property_reported_total == 0 and row.joint_component_fit_eligible


@pytest.mark.parametrize("field", ["DIV_VAL", "RNT_VAL"])
def test_receipt_yes_zero_stays_unknown_instead_of_eligible(field):
    source = inputs(1)
    if field == "DIV_VAL":
        amount(source["dividend"], field, [0], ["ambiguous_recipient_zero"])
        code(source["dividend"], "DIV_YN", [1])
    else:
        source["income_routing"]["net_property_known_amount"] = np.nan
        source["income_routing"]["net_property_reporting_status"] = (
            "receipt_with_net_zero"
        )
    result = build_asec_property_basis(**source)
    assert np.isnan(result.person.property_reported_total.iloc[0])
    assert not result.person.reported_total_eligible.iloc[0]
    assert not result.person.joint_component_fit_eligible.iloc[0]


def test_actual_pure_projector_column_composition_with_invented_literals():
    from test_us_current_asec_income_routing import _pure_rows, _set_amount

    from microcosm.build.us_runtime import current_asec_dividend_source as dividend
    from microcosm.build.us_runtime import current_asec_income_routing_source as routing
    from microcosm.build.us_runtime import current_asec_interest_source as interest

    source = inputs(1)
    raw_interest = {name: "0" for name in interest.READ_COLUMNS}
    raw_interest.update(
        A_AGE="40",
        INT_YN="1",
        INT_VAL="120",
        TRDINT_VAL="100",
        RINT_YN="1",
        RINT_SC1="4",
        RINT_VAL1="20",
    )
    raw_dividend = {name: "0" for name in dividend.READ_COLUMNS}
    raw_dividend.update(A_AGE="40", DIV_YN="2", SUR_YN="2")
    raw_routing, ages = _pure_rows(1, [40], RNT_YN=["1"], OI_YN=["2"])
    _set_amount(raw_routing, "RNT_VAL", [-120])
    projected = {
        "interest": interest.project_interest_literals(pd.DataFrame([raw_interest])),
        "income_routing": routing.project_income_routing(raw_routing, ages),
        "dividend": dividend.project_dividend_literals(pd.DataFrame([raw_dividend])),
    }
    for name, frame in projected.items():
        frame.index = source[name].index
        frame["native_person_id"] = source[name].native_person_id
        source[name] = frame
    result = build_asec_property_basis(**source)
    row = result.person.iloc[0]
    assert row.property_reported_total == 0
    assert row[list(PROPERTY_COMPONENTS)].tolist() == [100, 20, 0, -120]
    assert row.joint_component_fit_eligible
    assert row.retirement_slot2_derived_unused_zero
    assert (
        result.provenance.iloc[0]["interest.RINT_VAL2_reporting_status"]
        == "unreported_account_slot"
    )
