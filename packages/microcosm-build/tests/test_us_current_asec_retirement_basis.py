"""Pure retirement candidate support over actual projectors and invented literals."""

import dataclasses
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_income_routing_source as routing
from microcosm.build.us_runtime import current_asec_retirement_detail_source as detail


@pytest.fixture
def owner():
    return importlib.import_module(
        "microcosm.build.us_runtime.current_asec_retirement_basis"
    )


def options(owner, **changes):
    return owner.RetirementCandidateAssumptions(
        **{
            "pension_annuity_regularity": "unresolved",
            "disability_pension_eligibility": "unresolved",
            "survivor_annuity_overlap": "unresolved",
            "withdrawal_regularity_netting": "unresolved",
            "aggregate_accounting": "exact_visible_balance_only",
            **changes,
        }
    )


def inputs(owner, changes=({},), **assumptions):
    rows = []
    for i, change in enumerate(changes):
        row = {
            **{n: "0" for n in detail.READ_COLUMNS},
            **{n: "0" for n in routing.RECEIPT_ENTRIES},
            **{n: "0" for n in routing.ACCOUNT_ENTRIES},
            **{n: "0" for n in routing.AMOUNT_FIELDS},
            "PERIDNUM": str(i + 1).zfill(22),
            "PH_SEQ": str(i // 2 + 1),
            "A_LINENO": str(i % 2 + 1),
            "A_AGE": "40",
            "OI_OFF": "0",
            "PEN_YN": "2",
            "DIS_YN": "2",
            "SUR_YN": "2",
            "ANN_YN": "2",
            "DST_YN_YNG": "2",
            "OI_YN": "2",
            **change,
        }
        rows.append(row)
    raw = pd.DataFrame(rows)
    ages = np.array([int(r["A_AGE"]) for r in rows], dtype="int64")
    values, statuses = {}, {}
    for n in routing.AMOUNT_FIELDS:
        data, labels = [], []
        for r in rows:
            token = r[n]
            value = np.nan if token == "" else float(token)
            code = (
                routing.money.CodebookStatus.MISSING_NULL
                if token == ""
                else routing.money.CodebookStatus.DECLARED_NIU
                if n == "ANN_VAL" and value == -1
                else routing.money.CodebookStatus.ZERO_NONE_OR_NIU
                if value == 0
                else routing.money.CodebookStatus.AMOUNT_NONZERO
            )
            data.append(
                0.0 if code == routing.money.CodebookStatus.DECLARED_NIU else value
            )
            labels.append(int(code))
        values[n], statuses[n] = (
            np.array(data, dtype="float64"),
            np.array(labels, dtype="u1"),
        )
    arrays = {
        "amounts": values,
        "statuses": statuses,
        "allocations": {
            n: ([0] * len(rows), ["in_printed_range"] * len(rows))
            for n in routing.ALLOCATION_ENTRIES
        },
        **{
            n: [r[n] for r in rows]
            for n in (*routing.RECEIPT_ENTRIES, *routing.ACCOUNT_ENTRIES, "OI_OFF")
        },
    }
    projected = {
        "retirement_detail": detail.project_retirement_detail_literals(raw),
        "income_routing": routing.project_income_routing(arrays, ages),
    }
    index = pd.Index(np.arange(len(rows), dtype="int64") + 2**53 + 19, name="person_id")
    for table in projected.values():
        table.index = index
        table["native_person_id"] = np.arange(len(rows), dtype="int64") + 17
    membership = pd.Series(np.arange(len(rows), dtype="int64") // 2 + 101, index=index)
    households = pd.Index(membership.unique(), dtype="int64", name="household_id")
    return {
        **projected,
        "original_household_membership": membership,
        "original_household_design_weights": pd.Series(
            np.arange(len(households)) + 2.0, index=households
        ),
        "assumptions": options(owner, **assumptions),
    }


def build(owner, changes=({},), **assumptions):
    return owner.build_asec_retirement_basis(**inputs(owner, changes, **assumptions))


def test_named_options_are_required_canonical_and_never_source_facts(owner):
    value = options(owner)
    assert json.loads(value.to_bytes())["pension_annuity_regularity"] == "unresolved"
    assert (
        dataclasses.replace(
            value, pension_annuity_regularity="assume_regular"
        ).to_bytes()
        != value.to_bytes()
    )
    assert (
        dataclasses.replace(
            value, disability_pension_eligibility="assume_qualifying"
        ).to_bytes()
        != value.to_bytes()
    )
    with pytest.raises(TypeError):
        owner.RetirementCandidateAssumptions()
    for field, bad in (
        ("pension_annuity_regularity", True),
        ("aggregate_accounting", "clip"),
        ("withdrawal_regularity_netting", "all_regular"),
    ):
        with pytest.raises(ValueError):
            options(owner, **{field: bad})


@pytest.mark.parametrize(
    "code,route",
    [(1, "candidate"), (6, "candidate"), (7, "railroad"), (8, "unresolved")],
)
def test_balanced_pension_routes_and_named_regularity(owner, code, route):
    row = {"PEN_YN": "1", "PEN_SC1": str(code), "PEN_VAL1": "100", "PNSN_VAL": "100"}
    result = build(owner, [row])
    selected = result.slots.query("family == 'pension' and slot == 1").iloc[0]
    assert selected.route == route and selected.source_known_amount == 100
    p = result.person.iloc[0]
    assert p.pension_candidate_lower == 0
    assert p.pension_candidate_upper == (0 if code == 7 else 100)
    assumed = build(
        owner, [row], pension_annuity_regularity="assume_regular"
    ).person.iloc[0]
    assert assumed.pension_candidate_lower == (100 if code in (1, 6) else 0)
    assert not p.point_identified and not p.fiscal_outputs_produced
    assert (
        result.slots.query("family == 'pension' and slot == 2")
        .iloc[0]
        .structural_zero_comparison
    )


@pytest.mark.parametrize(
    "total,status",
    [(150, "additional_scope_unresolved"), (80, "contradictory_accounting")],
)
def test_unbalanced_pension_never_clips_or_allocates_residual(owner, total, status):
    p = build(
        owner,
        [{"PEN_YN": "1", "PEN_SC1": "1", "PEN_VAL1": "100", "PNSN_VAL": str(total)}],
    ).person.iloc[0]
    assert p.pension_accounting_difference == total - 100
    assert p.pension_status == status
    assert np.isnan(p.pension_candidate_upper)
    assert p.pension_observed_candidate_subtotal == 100
    assert not p.candidate_basis_eligible


@pytest.mark.parametrize("code", [1, 5, 8, 9, 10])
def test_positive_survivor_never_claims_hidden_source_clearance(owner, code):
    result = build(
        owner,
        [
            {
                "SUR_YN": "1",
                "SUR_SC1": str(code),
                "SUR_VAL1": "100",
                "SRVS_VAL": "100",
                "ANN_YN": "1",
                "ANN_VAL": "40",
            }
        ],
    )
    p = result.person.iloc[0]
    assert p.survivor_accounting_difference == 0
    assert p.survivor_status == "additional_survivor_scope_unresolved"
    assert np.isnan(p.survivor_candidate_upper) and not p.candidate_basis_eligible
    assert bool(result.exclusions.survivor_annuity_overlap.iloc[0]) == (code == 9)
    assert not p.point_identified


@pytest.mark.parametrize(
    "code,route",
    [
        (2, "candidate"),
        (5, "candidate"),
        (6, "railroad"),
        (9, "other_compensation"),
        (10, "unresolved"),
    ],
)
def test_disability_does_not_infer_eligibility_from_health_answers(owner, code, route):
    row = {
        "DIS_YN": "1",
        "DIS_SC1": str(code),
        "DIS_VAL1": "100",
        "DSAB_VAL": "100",
        "DIS_HP": "1",
        "DIS_CS": "1",
    }
    result = build(owner, [row])
    p = result.person.iloc[0]
    assert (
        result.slots.query("family == 'disability' and slot == 1").iloc[0].route
        == route
    )
    assert p.disability_candidate_lower == 0
    assert p.disability_candidate_upper == (
        100 if route in ("candidate", "unresolved") else 0
    )
    changed = build(owner, [{**row, "DIS_HP": "2", "DIS_CS": "2"}])
    pd.testing.assert_frame_equal(result.person, changed.person, check_exact=True)
    assumed = build(
        owner, [row], disability_pension_eligibility="assume_qualifying"
    ).person.iloc[0]
    assert assumed.disability_candidate_lower == (100 if route == "candidate" else 0)


@pytest.mark.parametrize(
    "amount,receipt,known",
    [("-1", "0", False), ("0", "1", True), ("0", "2", True), ("40", "1", True)],
)
def test_annuity_niu_zero_and_regular_scenario_are_distinct(
    owner, amount, receipt, known
):
    source = inputs(owner, [{"ANN_YN": receipt, "ANN_VAL": amount}])
    p = owner.build_asec_retirement_basis(**source).person.iloc[0]
    assert np.isfinite(p.annuity_candidate_upper) == known
    if known:
        assert p.annuity_candidate_lower == 0 and p.annuity_candidate_upper == float(
            amount
        )
        source["assumptions"] = options(
            owner, pension_annuity_regularity="assume_regular"
        )
        assert owner.build_asec_retirement_basis(**source).person.iloc[
            0
        ].annuity_candidate_lower == float(amount)


@pytest.mark.parametrize("age", [57, 58])
def test_distribution_account_is_not_frequency_or_taxability(owner, age):
    young = age < 58
    suffix = "_YNG" if young else ""
    row = {
        "A_AGE": str(age),
        "DST_YN_YNG": "1" if young else "0",
        "DST_YN": "0" if young else "1",
        "DST_SC1" + suffix: "4",
        "DST_VAL1" + suffix: "100",
        "DBTN_VAL": "0" if young else "100",
    }
    p = build(owner, [row]).person.iloc[0]
    assert p.distribution_candidate_lower == 0 and p.distribution_candidate_upper == 100
    assert p.distribution_account_4_amount == 100
    assert not p.point_identified and not p.fiscal_outputs_produced


@pytest.mark.parametrize(
    "change",
    [
        {"DST_SC1_YNG": ""},
        {"DST_SC1_YNG": "9"},
        {"DST_VAL1": "1"},
        {"DST_YN": "2"},
        {"DST_SC2_YNG": "3"},
    ],
)
def test_distribution_unreadable_or_offroute_composition_stays_unknown(owner, change):
    row = {"DST_YN_YNG": "1", "DST_SC1_YNG": "4", "DST_VAL1_YNG": "100", **change}
    p = build(owner, [row]).person.iloc[0]
    assert np.isnan(p.distribution_candidate_upper)
    assert not p.candidate_basis_eligible


@pytest.mark.parametrize(
    "code,clear",
    [(2, False), (13, False), (19, False), (1, True), (8, True), (20, True)],
)
def test_other_income_not_silently_added(owner, code, clear):
    p = build(
        owner, [{"OI_YN": "1", "OI_OFF": str(code), "OI_VAL": "100"}]
    ).person.iloc[0]
    assert bool(p.other_income_scope_clear) is clear
    assert p.other_income_reported_amount == 100
    assert (
        (p.retirement_candidate_upper == 0)
        if clear
        else np.isnan(p.retirement_candidate_upper)
    )


@pytest.mark.parametrize("age", [0, 14])
def test_under15_retains_evidence_without_analytic_zero(owner, age):
    result = build(
        owner,
        [
            {
                "A_AGE": str(age),
                "PEN_YN": "0",
                "ANN_YN": "0",
                "DIS_YN": "0",
                "SUR_YN": "0",
                "DST_YN_YNG": "0",
                "OI_YN": "0",
            }
        ],
    )
    p = result.person.iloc[0]
    assert all(
        np.isnan(p[n])
        for n in p.index
        if n.endswith(("_candidate_lower", "_candidate_upper"))
    )
    assert result.exclusions.under15.iloc[0] and not p.candidate_basis_eligible


@pytest.mark.parametrize(
    "change", [{"PEN_YN": ""}, {"PEN_SC1": ""}, {"PEN_VAL1": ""}, {"PEN_VAL1": "0"}]
)
def test_source_unknowns_and_ambiguous_zeros_remain_visible(owner, change):
    row = {
        "PEN_YN": "1",
        "PEN_SC1": "1",
        "PEN_VAL1": "100",
        "PNSN_VAL": "100",
        **change,
    }
    p = build(owner, [row]).person.iloc[0]
    assert np.isnan(p.pension_candidate_upper) and not p.candidate_basis_eligible


@pytest.mark.parametrize(
    "defect",
    [
        "person_order",
        "native_id",
        "age",
        "shared_amount",
        "known_mask",
        "comparison",
        "bad_weight",
        "membership",
    ],
)
def test_incompatible_descriptions_refuse(owner, defect):
    source = inputs(owner, [{}, {}])
    d, r = source["retirement_detail"], source["income_routing"]
    if defect == "person_order":
        source["income_routing"] = r.iloc[::-1]
    elif defect == "native_id":
        r.iloc[0, r.columns.get_loc("native_person_id")] += 100
    elif defect == "age":
        r.iloc[0, r.columns.get_loc("source_age")] += 1
    elif defect == "shared_amount":
        r.iloc[0, r.columns.get_loc("pension_annuity_pension_source_total")] += 1
    elif defect == "known_mask":
        d.iloc[0, d.columns.get_loc("PEN_VAL1_amount_known")] = False
    elif defect == "comparison":
        d.iloc[0, d.columns.get_loc("pension_total_minus_visible_slots")] = 1
    elif defect == "bad_weight":
        source["original_household_design_weights"].iloc[0] = np.nan
    else:
        source["original_household_membership"] = source[
            "original_household_membership"
        ].iloc[::-1]
    with pytest.raises(ValueError):
        owner.build_asec_retirement_basis(**source)


def test_exact_permutation_detachment_and_design_mass(owner):
    source = inputs(
        owner,
        [{}, {"PEN_YN": "1", "PEN_SC1": "1", "PEN_VAL1": "100", "PNSN_VAL": "80"}, {}],
    )
    before = {
        k: v.copy(deep=True)
        for k, v in source.items()
        if isinstance(v, (pd.DataFrame, pd.Series))
    }
    result = owner.build_asec_retirement_basis(**source)
    assert result.summary.loc["all", "design_weighted_person_mass"] == 7
    assert result.summary.loc["all", "union_household_design_mass"] == 5
    assert (
        result.summary.loc["candidate_basis_eligible", "design_weighted_person_mass"]
        == 5
    )
    assert (
        result.summary.loc["candidate_basis_eligible", "union_household_design_mass"]
        == 5
    )
    for name in (
        "retirement_detail",
        "income_routing",
        "original_household_membership",
    ):
        source[name] = source[name].iloc[::-1]
    permuted = owner.build_asec_retirement_basis(**source)
    pd.testing.assert_frame_equal(
        permuted.person.loc[result.person.index], result.person, check_exact=True
    )
    pd.testing.assert_frame_equal(permuted.summary, result.summary, check_exact=True)
    result.provenance.iloc[0, 0] = "detached"
    for name, original in before.items():
        current = source[name].loc[original.index]
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(current, original, check_exact=True)
        else:
            pd.testing.assert_series_equal(current, original, check_exact=True)


def test_unknown_account_total_is_preserved_without_candidate_admission(owner):
    source = inputs(
        owner, [{"DST_YN_YNG": "1", "DST_SC1_YNG": "", "DST_VAL1_YNG": "100"}]
    )
    assert source["income_routing"].retirement_distribution_known_amount.iloc[0] == 100
    p = owner.build_asec_retirement_basis(**source).person.iloc[0]
    assert p.distribution_source_known_amount == 100
    assert np.isnan(p.distribution_candidate_upper)
    assert not p.distribution_account_composition_known


def test_allocation_and_disclosure_flags_are_provenance_not_filters(owner):
    row = {"PEN_YN": "1", "PEN_SC1": "1", "PEN_VAL1": "100", "PNSN_VAL": "100"}
    before = build(owner, [row])
    changed = build(owner, [{**row, "I_PENVAL1": "4", "TPEN_VAL1": "1"}])
    pd.testing.assert_frame_equal(before.person, changed.person, check_exact=True)
    pd.testing.assert_frame_equal(before.summary, changed.summary, check_exact=True)
    assert changed.provenance["retirement_detail.I_PENVAL1_code"].iloc[0] == 4


def test_outside_universe_unreadable_and_contradictory_evidence_stay_distinct(owner):
    unresolved = build(
        owner, [{"A_AGE": "14", "PEN_YN": "0", "PEN_VAL1": ""}]
    ).person.iloc[0]
    contradictory = build(
        owner,
        [
            {
                "A_AGE": "14",
                "PEN_YN": "1",
                "PEN_SC1": "1",
                "PEN_VAL1": "100",
                "PNSN_VAL": "100",
            }
        ],
    ).person.iloc[0]
    assert unresolved.pension_status == "unresolved_outside_reporting_universe"
    assert contradictory.pension_status == "contradictory_outside_reporting_universe"
    assert np.isnan(unresolved.pension_candidate_upper) and np.isnan(
        contradictory.pension_candidate_upper
    )


@pytest.mark.parametrize("total", [150, 50])
def test_distribution_main_aggregate_difference_blocks_candidate_interval(owner, total):
    result = build(
        owner,
        [
            {
                "A_AGE": "60",
                "DST_YN_YNG": "0",
                "DST_YN": "1",
                "DST_SC1": "4",
                "DST_VAL1": "100",
                "DBTN_VAL": str(total),
            }
        ],
    )
    p = result.person.iloc[0]
    assert p.distribution_accounting_difference == total - 100
    assert p.distribution_source_known_amount == 100
    assert p.distribution_account_4_amount == 100
    assert (
        result.provenance["retirement_detail.DBTN_VAL_published_amount"].iloc[0]
        == total
    )
    assert np.isnan(p.distribution_candidate_lower)
    assert np.isnan(p.distribution_candidate_upper)
    assert not p.candidate_interval_available and not p.candidate_basis_eligible
    assert p.distribution_status == (
        "additional_scope_unresolved" if total > 100 else "contradictory_accounting"
    )


@pytest.mark.parametrize("token", ["", "malformed"])
@pytest.mark.parametrize(
    "age,field,retained",
    [
        (40, "DST_YN", "receipt_58"),
        (40, "DST_SC1", "slot1_account"),
        (40, "DST_SC2", "slot2_account"),
        (60, "DST_YN_YNG", "receipt_young"),
        (60, "DST_SC1_YNG", "slot1_young_account"),
        (60, "DST_SC2_YNG", "slot2_young_account"),
    ],
)
def test_unreadable_offroute_distribution_literals_prevent_admission(
    owner, age, field, retained, token
):
    young = age < 58
    suffix = "_YNG" if young else ""
    result = build(
        owner,
        [
            {
                "A_AGE": str(age),
                "DST_YN_YNG": "1" if young else "0",
                "DST_YN": "0" if young else "1",
                "DST_SC1" + suffix: "4",
                "DST_VAL1" + suffix: "100",
                "DBTN_VAL": "0" if young else "100",
                field: token,
            }
        ],
    )
    p = result.person.iloc[0]
    assert (
        result.provenance[
            "income_routing.retirement_distribution_" + retained + "_literal"
        ].iloc[0]
        == token
    )
    assert not p.distribution_account_composition_known
    assert np.isnan(p.distribution_candidate_upper)
    assert not p.candidate_basis_eligible


@pytest.mark.parametrize("age,field", [(40, "DST_VAL1"), (60, "DST_VAL1_YNG")])
def test_unreadable_offroute_distribution_amount_is_not_assumed_zero(owner, age, field):
    young = age < 58
    suffix = "_YNG" if young else ""
    p = build(
        owner,
        [
            {
                "A_AGE": str(age),
                "DST_YN_YNG": "1" if young else "0",
                "DST_YN": "0" if young else "1",
                "DST_SC1" + suffix: "4",
                "DST_VAL1" + suffix: "100",
                "DBTN_VAL": "0" if young else "100",
                field: "",
            }
        ],
    ).person.iloc[0]
    assert not p.distribution_account_composition_known
    assert np.isnan(p.distribution_candidate_upper)


def test_account_subtotals_are_evidence_when_receipt_interval_is_unknown(owner):
    p = build(
        owner, [{"DST_YN_YNG": "", "DST_SC1_YNG": "4", "DST_VAL1_YNG": "100"}]
    ).person.iloc[0]
    assert p.distribution_account_composition_known
    assert p.distribution_account_4_amount == 100
    assert np.isnan(p.distribution_source_known_amount)
    assert np.isnan(p.distribution_candidate_upper)
    assert not p.candidate_interval_available


@pytest.mark.parametrize(
    "family,prefix,total,code1,code2,lower,upper,status",
    [
        ("pension", "PEN", "PNSN_VAL", 1, 8, 60, 100, "candidate_under_assumptions"),
        ("pension", "PEN", "PNSN_VAL", 1, 7, 60, 60, "candidate_under_assumptions"),
        (
            "disability",
            "DIS",
            "DSAB_VAL",
            2,
            10,
            60,
            100,
            "candidate_under_assumptions",
        ),
        ("disability", "DIS", "DSAB_VAL", 2, 1, 60, 60, "candidate_under_assumptions"),
        ("disability", "DIS", "DSAB_VAL", 6, 1, 0, 0, "route_outside_only"),
    ],
)
def test_two_slot_family_routes_preserve_candidate_and_unresolved_bounds(
    owner, family, prefix, total, code1, code2, lower, upper, status
):
    p = build(
        owner,
        [
            {
                prefix + "_YN": "1",
                prefix + "_SC1": str(code1),
                prefix + "_SC2": str(code2),
                prefix + "_VAL1": "60",
                prefix + "_VAL2": "40",
                total: "100",
            }
        ],
        pension_annuity_regularity="assume_regular",
        disability_pension_eligibility="assume_qualifying",
    ).person.iloc[0]
    assert p[family + "_candidate_lower"] == lower
    assert p[family + "_candidate_upper"] == upper
    assert p[family + "_status"] == status
    assert p[family + "_accounting_difference"] == 0
    assert (
        p[family + "_observed_candidate_subtotal"]
        + p[family + "_observed_unresolved_subtotal"]
        == upper
    )


@pytest.mark.parametrize(
    "family,total",
    [("pension", "PNSN_VAL"), ("disability", "DSAB_VAL"), ("survivor", "SRVS_VAL")],
)
def test_no_receipt_with_nonzero_total_and_zero_slots_remains_contradictory(
    owner, family, total
):
    p = build(owner, [{total: "100"}]).person.iloc[0]
    assert p[family + "_status"] == "contradictory_accounting"
    assert p[family + "_accounting_difference"] == 100
    assert p[family + "_observed_candidate_subtotal"] == 0
    assert np.isnan(p[family + "_candidate_upper"])
    assert not p.candidate_basis_eligible


def test_equal_bounds_under_explicit_scenarios_do_not_claim_identification(owner):
    row = {
        "PEN_YN": "1",
        "PEN_SC1": "1",
        "PEN_VAL1": "100",
        "PNSN_VAL": "100",
        "DIS_YN": "1",
        "DIS_SC1": "2",
        "DIS_VAL1": "40",
        "DSAB_VAL": "40",
        "ANN_YN": "1",
        "ANN_VAL": "20",
    }
    unresolved = build(owner, [row]).person.iloc[0]
    p = build(
        owner,
        [row],
        pension_annuity_regularity="assume_regular",
        disability_pension_eligibility="assume_qualifying",
    ).person.iloc[0]
    assert (
        unresolved.retirement_candidate_lower == 0
        and unresolved.retirement_candidate_upper == 160
    )
    assert not unresolved.candidate_point_under_assumptions
    assert p.retirement_candidate_lower == p.retirement_candidate_upper == 160
    assert p.candidate_point_under_assumptions and p.candidate_basis_eligible
    assert not p.point_identified and not p.fiscal_outputs_produced


def test_diagnostic_household_mass_deduplicates_only_selected_households(owner):
    row = {"PEN_YN": "1", "PEN_SC1": "1", "PEN_VAL1": "100", "PNSN_VAL": "150"}
    result = build(owner, [row, row, {}, row, {}])
    selected = result.summary.loc["diagnostic:pension_candidate_unresolved"]
    assert selected.person_count == 3 and selected.household_count == 2
    assert selected.design_weighted_person_mass == 7
    assert selected.union_household_design_mass == 5


def test_design_weights_refuse_an_unselected_household(owner):
    source = inputs(owner, [{}])
    source["original_household_design_weights"].loc[999] = 4.0
    with pytest.raises(ValueError, match="DESIGN_MEMBERSHIP"):
        owner.build_asec_retirement_basis(**source)
