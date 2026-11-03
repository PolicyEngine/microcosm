"""Unit reporting classifiers using declared inputs and fixed baseline cells."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.frame.reporting import (
    annual_unit_income,
    carry_baseline_deciles,
    classify_receipt_groups,
    decile_weighted_shares,
    validate_income_classifier,
    validate_receipt_classifier,
    weighted_deciles,
)


def _income_spec():
    return {
        "entity": "family",
        "person_entity": "person",
        "unit_id_column": "family_id",
        "membership_column": "person_family_id",
        "weight_entity": "household",
        "expected_weight_kind": "calibrated",
        "tie_key": "family_id",
        "boundary_scenario": "baseline",
        "n_bins": 10,
        "rank_method": "weighted_midpoint",
        "components": [
            {
                "entity": "person",
                "column": "market_income",
                "kind": "market",
                "tax_basis": "before_tax",
                "annualisation_factor": 1,
            },
            {
                "entity": "family",
                "column": "capital_income",
                "kind": "market",
                "tax_basis": "before_tax",
                "annualisation_factor": 1,
            },
            {
                "entity": "person",
                "column": "main_payment",
                "kind": "benefit",
                "tax_basis": "before_tax",
                "annualisation_factor": 52,
                "receipt_column": "receives_main",
            },
            {
                "entity": "person",
                "column": "pension_payment",
                "kind": "benefit",
                "tax_basis": "before_tax",
                "annualisation_factor": 52,
                "receipt_column": "receives_pension",
            },
        ],
        "excluded_columns": [
            {"entity": "family", "column": "housing_assistance"},
            {"entity": "family", "column": "tax_credit"},
            {"entity": "person", "column": "main_net_payment"},
        ],
    }


def _receipt_spec():
    return {
        "entity": "family",
        "person_entity": "person",
        "unit_id_column": "family_id",
        "membership_column": "person_family_id",
        "predicates": {
            "superannuitant": {
                "entity": "person",
                "columns": ["receives_pension", "receives_veteran"],
                "reduction": "any",
            },
            "beneficiary": {
                "entity": "person",
                "columns": ["receives_main"],
                "reduction": "any",
            },
        },
        "priority": ["superannuitant", "beneficiary"],
        "default": "other",
    }


def _tables():
    return {
        "family": pd.DataFrame(
            {
                "family_id": [100, 200, 300, 400],
                "family_source_household_id": [77, 77, 88, 99],
                "capital_income": [-25, 0, 10, 0],
                "housing_assistance": [999, 999, 999, 999],
                "tax_credit": [999, 999, 999, 999],
            }
        ),
        "person": pd.DataFrame(
            {
                "person_id": [7, 8, 9, 10, 11],
                "person_family_id": [100, 100, 200, 300, 400],
                "market_income": [500, 250, -20, 0, 1000],
                "main_payment": [100, 200, 50, 40, 10],
                "main_net_payment": [80, 160, 40, 32, 8],
                "pension_payment": [100, 200, 50, 300, 10],
                "receives_main": [False, True, True, False, False],
                "receives_pension": [True, False, False, True, False],
                "receives_veteran": [False, False, False, False, False],
            }
        ),
    }


def _deciles(incomes, weights, ids=None, bins=10):
    if ids is None:
        ids = np.arange(len(incomes))
    return weighted_deciles(
        incomes, weights, ids, bins, rank_method="weighted_midpoint"
    )


def _fraction_deciles(incomes, weights, ids, bins=10):
    """Independent oracle on the actual float64 input values, without snapping."""

    exact_weights = [Fraction(float(weight)) for weight in weights]
    total = sum(exact_weights, Fraction(0))
    preceding = Fraction(0)
    labels = np.empty(len(weights), dtype=np.int64)
    for row in sorted(range(len(weights)), key=lambda row: (incomes[row], ids[row])):
        midpoint = (preceding + exact_weights[row] / 2) / total
        labels[row] = min(int(midpoint * bins) + 1, bins)
        preceding += exact_weights[row]
    return labels


def _large_float_weighted_sample(size, seed, *, exponent_limit=400):
    """Generate large populations with ties, zero weights and dyadic float weights."""

    rng = np.random.default_rng(seed)
    incomes = rng.integers(-100, 101, size=size)
    weights = np.ldexp(
        rng.uniform(0.5, 1, size=size),
        rng.integers(-exponent_limit, exponent_limit + 1, size=size),
    )
    weights[::17] = 0
    weights[1] = 1
    ids = np.arange(size) * 17 - 35
    return incomes, weights, ids


def test_declared_income_uses_gross_assigned_amounts_and_annual_factors():
    tables = _tables()
    before = deepcopy(tables)
    np.testing.assert_array_equal(
        annual_unit_income(tables, _income_spec()), [16_325, 2580, 15_610, 1000]
    )
    for entity in tables:
        pd.testing.assert_frame_equal(tables[entity], before[entity])


def test_annual_income_person_permutation_has_stable_summation():
    tables = _tables()
    spec = _income_spec()
    spec["components"] = spec["components"][:1]
    tables["person"] = pd.DataFrame(
        {
            "person_id": [10, 20, 30, 40, 50, 60],
            "person_family_id": [100, 100, 100, 200, 300, 400],
            "market_income": [1e16, -1e16, 1, 2, 3, 4],
        }
    )
    expected = annual_unit_income(tables, spec)
    tables["person"] = tables["person"].iloc[[2, 0, 5, 4, 3, 1]]
    np.testing.assert_array_equal(annual_unit_income(tables, spec), expected)


def test_declared_unit_benefit_amounts_are_supported():
    tables = _tables()
    spec = _income_spec()
    component = dict(spec["components"][2], entity="family")
    spec["components"] = [component]
    tables["family"]["main_payment"] = [100, 200, 300, 400]
    tables["family"]["receives_main"] = [True, False, True, False]
    np.testing.assert_array_equal(
        annual_unit_income(tables, spec), [5200, 0, 15_600, 0]
    )


def test_exact_weighted_midpoint_boundaries_and_uneven_weights():
    labels, diagnostics = _deciles([0, 1, 2, 3, 4], [1, 1, 2, 2, 4])
    np.testing.assert_array_equal(labels, [1, 2, 4, 6, 9])
    np.testing.assert_allclose(
        diagnostics["weighted_shares"], [0.1, 0.1, 0, 0.2, 0, 0.2, 0, 0, 0.4, 0]
    )
    assert diagnostics["zero_weight_units"] == 0
    np.testing.assert_array_equal(_deciles([0, 1], [2, 8])[0], [2, 7])


@pytest.mark.parametrize(
    "first_weight,last_weight,expected",
    [
        (np.nextafter(2.0, 0.0), 8.0, 1),
        (2.0, np.nextafter(8.0, np.inf), 1),
        (2.0, 8.0, 2),
        (2.0, np.nextafter(8.0, 0.0), 2),
    ],
)
def test_genuine_near_boundary_midpoints_are_not_snapped(
    first_weight, last_weight, expected
):
    weights = [first_weight, last_weight]
    labels, _ = _deciles([0, 1], weights)
    assert labels[0] == expected
    np.testing.assert_array_equal(
        labels, _fraction_deciles([0, 1], weights, [0, 1])
    )


@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_reviewer_625_unit_exact_half_boundary_with_float64_accumulator(
    monkeypatch, scale
):
    # The rejected algorithm used longdouble, which has float64 precision on
    # macOS arm64. Exercise that precision on every platform if it is restored.
    monkeypatch.setattr("microcosm.frame.reporting.np.longdouble", np.float64)
    left = np.tile([78.0, 46.0, 37.0], 104)
    weights = np.r_[left, 80.0, left] * scale
    ids = np.arange(625)
    labels, _ = _deciles(ids, weights, ids)
    assert labels[312] == 6
    np.testing.assert_array_equal(labels, _fraction_deciles(ids, weights, ids))


@pytest.mark.parametrize("repetitions", [200, 500, 1000])
@pytest.mark.parametrize(
    "pattern,center", [([1.0, 3.0, 6.0], 2.0), ([78.0, 46.0, 37.0], 80.0)]
)
@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_large_symmetric_populations_have_an_exact_half_boundary(
    monkeypatch, repetitions, pattern, center, scale
):
    monkeypatch.setattr("microcosm.frame.reporting.np.longdouble", np.float64)
    left = np.tile(pattern, repetitions)
    weights = np.r_[left, center, left] * scale
    ids = np.arange(len(weights))
    labels, _ = _deciles(ids, weights, ids)
    # Identical float weights occur in each half even after non-dyadic scaling;
    # symmetry therefore makes this unit's exact midpoint precisely one half.
    assert labels[len(left)] == 6


def test_boundary_ties_split_by_stable_unit_id():
    labels, diagnostics = _deciles(
        [100, 100, 100, 100], [1, 1, 1, 1], [40, 10, 30, 20], 2
    )
    np.testing.assert_array_equal(labels, [2, 1, 2, 1])
    assert diagnostics["weighted_shares"] == [0.5, 0.5]


def test_zero_weights_use_their_sorted_boundary_position():
    labels, diagnostics = _deciles([-1, 0, 1, 2, 3], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(labels, [1, 3, 6, 8, 10])
    assert diagnostics["zero_weight_units"] == 3
    assert diagnostics["weighted_shares"] == [0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0]


def test_single_unit_has_midpoint_cell_and_reports_full_share():
    labels, diagnostics = _deciles([0], [17], [10])
    np.testing.assert_array_equal(labels, [6])
    assert diagnostics["weighted_shares"][5] == 1


def test_all_equal_income_units_fill_every_decile_by_id():
    ids = np.arange(10)[::-1]
    labels, diagnostics = _deciles(np.zeros(10), np.ones(10), ids)
    np.testing.assert_array_equal(labels, np.arange(1, 11)[::-1])
    np.testing.assert_allclose(diagnostics["weighted_shares"], np.full(10, 0.1))


def test_cloned_source_households_are_separate_reporting_units():
    tables = _tables()
    units = tables["family"]
    assert units["family_source_household_id"].tolist()[:2] == [77, 77]
    labels, _ = _deciles([0, 0, 10, 20], [2, 2, 2, 2], units["family_id"], 4)
    np.testing.assert_array_equal(labels, [1, 2, 3, 4])


@st.composite
def _weighted_samples(draw):
    rows = draw(
        st.lists(
            st.tuples(st.integers(-100, 100), st.integers(0, 100)),
            min_size=1,
            max_size=30,
        ).filter(lambda rows: any(weight > 0 for _, weight in rows))
    )
    incomes, weights = np.array(rows).T
    ids = np.arange(len(rows)) * 17 - 35
    permutation = np.array(draw(st.permutations(range(len(rows)))))
    return incomes, weights, ids, permutation


@settings(max_examples=100, deadline=None, database=None)
@given(_weighted_samples())
def test_weighted_deciles_permutation_invariance(sample):
    incomes, weights, ids, permutation = sample
    labels, diagnostics = _deciles(incomes, weights, ids)
    permuted, reordered_diagnostics = _deciles(
        incomes[permutation], weights[permutation], ids[permutation]
    )
    np.testing.assert_array_equal(permuted, labels[permutation])
    assert reordered_diagnostics == diagnostics


@settings(max_examples=100, deadline=None, database=None)
@example(size=2000, seed=0, exponent=-200)
@example(size=2000, seed=1, exponent=200)
@given(
    size=st.integers(200, 2000),
    seed=st.integers(0, 2**32 - 1),
    exponent=st.integers(-200, 200),
)
def test_weighted_deciles_power_of_two_weight_scale_invariance(size, seed, exponent):
    incomes, weights, ids = _large_float_weighted_sample(size, seed)
    labels, diagnostics = _deciles(incomes, weights, ids)
    # These exponents keep every positive weight normal and finite. Arbitrary
    # multipliers round individual floats and can change the exact population.
    scaled, scaled_diagnostics = _deciles(incomes, np.ldexp(weights, exponent), ids)
    np.testing.assert_array_equal(scaled, labels)
    np.testing.assert_allclose(
        scaled_diagnostics["weighted_shares"],
        diagnostics["weighted_shares"],
        rtol=1e-14,
        atol=1e-14,
    )


@settings(max_examples=100, deadline=None, database=None)
@example(size=2000, seed=0, bins=10)
@example(size=2000, seed=1, bins=40)
@given(
    size=st.integers(200, 2000),
    seed=st.integers(0, 2**32 - 1),
    bins=st.integers(2, 40),
)
def test_weighted_deciles_agree_with_fraction_oracle_on_large_populations(
    size, seed, bins
):
    incomes, weights, ids = _large_float_weighted_sample(
        size, seed, exponent_limit=1000
    )
    labels, _ = _deciles(incomes, weights, ids, bins)
    np.testing.assert_array_equal(
        labels, _fraction_deciles(incomes, weights, ids, bins)
    )


@pytest.mark.parametrize(
    "weights",
    [
        [np.nextafter(0.0, 1.0), np.finfo(np.float64).max, 0, 1],
        [np.finfo(np.float64).max, 1, 0, np.nextafter(0.0, 1.0)],
    ],
)
def test_weighted_deciles_agree_with_fraction_oracle_across_float64_range(weights):
    ids = np.arange(len(weights))
    labels, _ = _deciles(ids, weights, ids)
    np.testing.assert_array_equal(labels, _fraction_deciles(ids, weights, ids))


def test_variant_reweight_carries_baseline_cells_by_unit_id():
    ids = np.array([10, 20, 30, 40, 50])
    baseline, _ = _deciles([0, 1, 2, 3, 4], [1, 1, 2, 2, 4], ids)
    variant_ids = ids[::-1]
    variant_weights = np.array([1, 1, 1, 1, 96])
    variant = carry_baseline_deciles(variant_ids, ids, baseline, 10)
    np.testing.assert_array_equal(variant, baseline[::-1])
    recomputed, _ = _deciles([900, 800, 700, 600, 500], variant_weights, variant_ids)
    assert not np.array_equal(recomputed, variant)
    shares = decile_weighted_shares(variant, variant_weights, 10)
    assert shares[0] == pytest.approx(0.96)
    assert shares[8] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "unit_ids,baseline_ids,labels",
    [
        ([1, 2], [1, 3], [1, 2]),
        ([1, 2], [1, 2, 3], [1, 2, 3]),
        ([1, 1], [1, 2], [1, 2]),
        ([1, 2], [1, 1], [1, 2]),
        ([1, 2], [1, 2], [1]),
        ([1, 2], [1, 2], [0, 1]),
        ([1, 2], [1, 2], [1, 11]),
        ([1.0, 2], [1, 2], [1, 2]),
        ([1, 2], [1, 2], [True, 2]),
        ([], [], []),
    ],
)
def test_baseline_carry_refuses_population_or_label_changes(
    unit_ids, baseline_ids, labels
):
    with pytest.raises(ValueError):
        carry_baseline_deciles(unit_ids, baseline_ids, labels, 10)


def test_receipt_priority_member_reduction_and_mixed_unit_count():
    groups, diagnostics = classify_receipt_groups(_tables(), _receipt_spec())
    assert groups.index.tolist() == [100, 200, 300, 400]
    assert groups.tolist() == [
        "superannuitant",
        "beneficiary",
        "superannuitant",
        "other",
    ]
    assert diagnostics["mixed_units"] == 1
    assert diagnostics["group_counts"] == {
        "superannuitant": 2,
        "beneficiary": 1,
        "other": 1,
    }
    assert diagnostics["predicate_counts"] == {"superannuitant": 2, "beneficiary": 2}


def test_veterans_receipt_satisfies_the_same_superannuitant_predicate():
    tables = _tables()
    tables["person"]["receives_pension"] = False
    tables["person"].loc[4, "receives_veteran"] = True
    groups, diagnostics = classify_receipt_groups(tables, _receipt_spec())
    assert groups.tolist() == ["beneficiary", "beneficiary", "other", "superannuitant"]
    assert diagnostics["mixed_units"] == 0


def test_receipt_priority_is_spec_data():
    spec = _receipt_spec()
    spec["priority"] = ["beneficiary", "superannuitant"]
    groups, diagnostics = classify_receipt_groups(_tables(), spec)
    assert groups.loc[100] == "beneficiary"
    assert diagnostics["mixed_units"] == 1


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, 0, 1, "False", [], 0.5])
def test_receipt_invalid_flags_refused_even_in_preempted_group(value):
    tables = _tables()
    tables["person"]["receives_main"] = tables["person"]["receives_main"].astype(object)
    tables["person"].at[0, "receives_main"] = value
    with pytest.raises(ValueError, match="boolean flags"):
        classify_receipt_groups(tables, _receipt_spec())
    with pytest.raises(ValueError, match="boolean flags"):
        annual_unit_income(tables, _income_spec())


def test_missing_receipt_flag_is_refused():
    tables = _tables()
    del tables["person"]["receives_main"]
    with pytest.raises(ValueError, match="receives_main.*missing"):
        classify_receipt_groups(tables, _receipt_spec())
    with pytest.raises(ValueError, match="receives_main.*missing"):
        annual_unit_income(tables, _income_spec())


@pytest.mark.parametrize(
    "column", ["main_net_payment", "housing_assistance", "tax_credit"]
)
def test_excluded_net_or_outcome_income_components_refused(column):
    spec = _income_spec()
    entity = "person" if column == "main_net_payment" else "family"
    spec["components"].append(
        {
            "entity": entity,
            "column": column,
            "kind": "market",
            "tax_basis": "before_tax",
            "annualisation_factor": 1,
        }
    )
    with pytest.raises(ValueError, match="Excluded income component"):
        annual_unit_income(_tables(), spec)


def test_net_tax_basis_refused_in_a_gross_income_mix():
    spec = _income_spec()
    spec["components"][2]["tax_basis"] = "after_tax"
    with pytest.raises(ValueError, match="before_tax.*net income"):
        annual_unit_income(_tables(), spec)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "100", True])
def test_nonfinite_or_nonnumeric_income_refused(value):
    tables = _tables()
    tables["person"]["market_income"] = tables["person"]["market_income"].astype(object)
    tables["person"].at[0, "market_income"] = value
    with pytest.raises(ValueError, match="finite numeric"):
        annual_unit_income(tables, _income_spec())


def test_negative_benefit_payments_are_refused():
    tables = _tables()
    tables["person"].loc[0, "main_payment"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        annual_unit_income(tables, _income_spec())


@pytest.mark.parametrize(
    "consumer,spec",
    [(annual_unit_income, _income_spec), (classify_receipt_groups, _receipt_spec)],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_unit",
        "duplicate_person",
        "orphan",
        "missing_membership",
        "empty_unit",
        "float_id",
    ],
)
def test_invalid_unit_identity_or_memberships_refused(consumer, spec, mutation):
    tables = _tables()
    if mutation == "duplicate_unit":
        tables["family"].loc[1, "family_id"] = 100
    elif mutation == "duplicate_person":
        tables["person"].loc[1, "person_id"] = 7
    elif mutation == "orphan":
        tables["person"].loc[0, "person_family_id"] = 999
    elif mutation == "missing_membership":
        tables["person"].loc[0, "person_family_id"] = np.nan
    elif mutation == "empty_unit":
        tables["person"] = tables["person"].iloc[:-1]
    else:
        tables["family"]["family_id"] = tables["family"]["family_id"].astype(float)
    with pytest.raises(ValueError):
        consumer(tables, spec())


@pytest.mark.parametrize(
    "incomes,weights,ids,bins,method",
    [
        ([0], [0], [1], 10, "weighted_midpoint"),
        ([0, 1], [1, -1], [1, 2], 10, "weighted_midpoint"),
        ([0, 1], [1], [1, 2], 10, "weighted_midpoint"),
        ([0], [np.inf], [1], 10, "weighted_midpoint"),
        ([np.nan], [1], [1], 10, "weighted_midpoint"),
        ([0, 1], [1, 1], [1, 1], 10, "weighted_midpoint"),
        ([0, 1], [1, 1], [1, "two"], 10, "weighted_midpoint"),
        ([0], [1], [None], 10, "weighted_midpoint"),
        ([0], [1], [1], 0, "weighted_midpoint"),
        ([0], [1], [1], True, "weighted_midpoint"),
        ([0], [1], [1], 10, "unweighted"),
    ],
)
def test_invalid_decile_inputs_refused(incomes, weights, ids, bins, method):
    with pytest.raises(ValueError):
        weighted_deciles(incomes, weights, ids, bins, rank_method=method)


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf, True, "52"])
def test_invalid_annualisation_factor_refused(factor):
    spec = _income_spec()
    spec["components"][2]["annualisation_factor"] = factor
    with pytest.raises(ValueError, match="annualisation_factor"):
        validate_income_classifier(spec)


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "benefit_missing_receipt",
        "market_receipt",
        "duplicate",
        "invalid_kind",
        "empty",
        "weight_kind",
        "rank_method",
    ],
)
def test_malformed_income_spec_refused(change):
    spec = _income_spec()
    if change == "unknown":
        spec["unknown"] = True
    elif change == "benefit_missing_receipt":
        del spec["components"][2]["receipt_column"]
    elif change == "market_receipt":
        spec["components"][0]["receipt_column"] = "receives_main"
    elif change == "duplicate":
        spec["components"].append(spec["components"][0])
    elif change == "invalid_kind":
        spec["components"][0]["kind"] = "entitlement"
    elif change == "empty":
        spec["components"] = []
    elif change == "weight_kind":
        spec["expected_weight_kind"] = "unknown"
    else:
        spec["rank_method"] = "unweighted"
    with pytest.raises(ValueError):
        validate_income_classifier(spec)


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "no_predicates",
        "duplicate_priority",
        "missing_priority",
        "bad_reduction",
        "bad_entity",
        "empty_columns",
        "duplicate_columns",
        "default_is_predicate",
    ],
)
def test_malformed_receipt_spec_refused(change):
    spec = _receipt_spec()
    if change == "unknown":
        spec["unknown"] = True
    elif change == "no_predicates":
        spec["predicates"] = {}
    elif change == "duplicate_priority":
        spec["priority"] = ["superannuitant", "superannuitant"]
    elif change == "missing_priority":
        spec["priority"] = ["superannuitant"]
    elif change == "bad_reduction":
        spec["predicates"]["beneficiary"]["reduction"] = "all"
    elif change == "bad_entity":
        spec["predicates"]["beneficiary"]["entity"] = "family"
    elif change == "empty_columns":
        spec["predicates"]["beneficiary"]["columns"] = []
    elif change == "duplicate_columns":
        spec["predicates"]["beneficiary"]["columns"] = [
            "receives_main",
            "receives_main",
        ]
    else:
        spec["default"] = "superannuitant"
    with pytest.raises(ValueError):
        validate_receipt_classifier(spec)
