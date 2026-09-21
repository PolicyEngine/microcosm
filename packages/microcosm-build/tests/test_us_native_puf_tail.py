"""Invented role projections; none authenticates actual PUF persons or a run."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import native_puf_tail as tail
from microcosm.frame import WeightKind


def fixture_returns(amounts, *, ids=None, weights=None, statuses=None):
    profile = tail.native_puf_tail_profile()
    n = len(amounts)
    ids = (
        np.arange(1, n + 1, dtype=np.int64)
        if ids is None
        else np.array(ids, dtype=np.int64)
    )
    returns = pd.DataFrame(
        {name: np.zeros(n, dtype=np.float64) for name in profile.targets}
    )
    returns.insert(
        0,
        "filing_status_code",
        np.ones(n, dtype=np.int64)
        if statuses is None
        else np.array(statuses, dtype=np.int64),
    )
    returns.insert(
        0,
        "weight",
        np.ones(n, dtype=np.float64)
        if weights is None
        else np.array(weights, dtype=np.float64),
    )
    returns.insert(0, "donor_recid", ids)
    returns["employment_income_before_lsr"] = np.array(amounts, dtype=np.float64)
    people = returns.loc[:, ["donor_recid", *profile.person_outputs]].copy()
    people.insert(0, "person_id", ids * 10)
    people.insert(2, "role", pd.Series(["head"] * n, dtype="string"))
    for name in profile.boolean_outputs:
        people[name] = False
    return returns, people


def declare(returns, people, **kwargs):
    return tail.declare_puf_tail_role_projection(
        returns,
        people,
        return_known=pd.DataFrame(True, index=returns.index, columns=returns.columns),
        person_known=pd.DataFrame(True, index=people.index, columns=people.columns),
        provenance=tail.TailProjectionProvenance(),
        weight_kind=WeightKind.DESIGN,
        **kwargs,
    )


def select(returns, people, *, gains=None, eligible=None):
    return tail.select_native_puf_tail(
        declare(returns, people),
        capital_gains_mask=np.zeros(len(returns), dtype=bool)
        if gains is None
        else np.array(gains, dtype=bool),
        source_eligible=np.ones(len(returns), dtype=bool)
        if eligible is None
        else np.array(eligible, dtype=bool),
    )


def test_profile_is_exact_native_fifty_two_and_keeps_late_ss_scf_owners():
    from microcosm.build.us_runtime.full_puf_enrichment import (
        PUF55_SURVEY_SS,
        SCF_MORTGAGE_OUTPUTS,
        SURVEY_SS_COMPONENTS,
    )
    from microcosm.build.us_runtime.us_late_overlap_ownership import (
        US_LATE_OVERLAP_OWNERSHIP_TARGETS,
    )

    profile = tail.native_puf_tail_profile()
    excluded = {name for _, name in US_LATE_OVERLAP_OWNERSHIP_TARGETS}
    assert len(profile.targets) == 52
    assert profile.targets == tuple(
        name for name in PUF55_SURVEY_SS.targets if name not in excluded
    )
    assert not set(profile.targets) & (
        excluded | set(SURVEY_SS_COMPONENTS) | set(SCF_MORTGAGE_OUTPUTS)
    )
    assert set(tail.PROXY_AGI_COMPONENTS) <= set(profile.targets)


def test_inclusive_floor_and_union_arms_preserve_source_weights():
    returns, people = fixture_returns(
        [4_999_999, 5_000_000, 8_000_000], weights=[2, 3, 4]
    )
    result = select(returns, people, gains=[True, False, True])
    assert [(row.donor_recid, row.arm, row.weight) for row in result.donors] == [
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 4),
    ]
    assert result.source_admission_issued is False
    assert result.release_eligible is False
    assert json.loads(result.receipt)["agi_candidate_count"] == 2


def test_source_eligibility_and_zero_weight_are_not_high_agi_support():
    returns, people = fixture_returns([9_000_000] * 3, weights=[0, 1, 2])
    result = select(returns, people, eligible=[True, False, True])
    assert [row.donor_recid for row in result.donors] == [3]


def test_signed_proxy_uses_all_components_without_clipping_losses():
    returns, people = fixture_returns([6_000_000, 6_000_000])
    returns["rental_income"] = [-2_000_000.0, -1_000_000.0]
    people["rental_income"] = returns.rental_income.copy()
    result = select(returns, people)
    assert [(row.donor_recid, row.proxy_agi) for row in result.donors] == [
        (2, 5_000_000)
    ]


def test_nonzero_dependent_money_skips_even_both_arm_without_folding():
    returns, people = fixture_returns([6_000_000, 7_000_000])
    dependent = people.iloc[[0]].copy()
    dependent["person_id"] = 11
    dependent["role"] = pd.Series(["dependent"], dtype="string")
    dependent["employment_income_before_lsr"] = 1.0
    people.loc[0, "employment_income_before_lsr"] -= 1
    people = pd.concat([people, dependent], ignore_index=True)
    result = select(returns, people, gains=[True, False])
    assert [row.donor_recid for row in result.donors] == [2]
    skipped = json.loads(result.receipt)["skipped"]
    assert skipped["count"] == 1
    assert skipped["by_reason"]["dependent_monetary_value"]["count"] == 1


def test_boolean_dependent_is_explicitly_ignored_and_nonzero_spouse_is_required():
    returns, people = fixture_returns([6_000_000])
    spouse = people.copy()
    spouse["person_id"] = 11
    spouse["role"] = pd.Series(["spouse"], dtype="string")
    spouse["employment_income_before_lsr"] = 1_000_000.0
    people.loc[0, "employment_income_before_lsr"] = 5_000_000.0
    dependent = spouse.copy()
    dependent["person_id"] = 12
    dependent["role"] = pd.Series(["dependent"], dtype="string")
    dependent["employment_income_before_lsr"] = 0.0
    dependent[tail.native_puf_tail_profile().boolean_outputs[0]] = True
    people = pd.concat([people, spouse, dependent], ignore_index=True)
    result = select(returns, people)
    assert result.donors[0].needs_spouse is True
    assert (
        json.loads(result.receipt)["dependent_boolean_policy"] == "ignored_explicitly"
    )


@pytest.mark.parametrize("role", ["dependent", "unclassified", "ambiguous"])
def test_unrepresentable_roles_skip_without_guessing_head(role):
    returns, people = fixture_returns([6_000_000])
    people.loc[0, "role"] = role
    result = select(returns, people)
    assert result.donors == ()
    assert (
        json.loads(result.receipt)["skipped"]["by_reason"][
            "unrepresentable_person_roles"
        ]["count"]
        == 1
    )


def test_return_person_mismatch_skips_instead_of_reconciling():
    returns, people = fixture_returns([6_000_000])
    people.loc[0, "employment_income_before_lsr"] = 5_000_000
    result = select(returns, people)
    assert result.donors == ()
    assert (
        json.loads(result.receipt)["skipped"]["by_reason"][
            "person_return_monetary_mismatch"
        ]["count"]
        == 1
    )


def test_projection_is_detached_and_round_trips_physical_boolean_and_signed_zero():
    returns, people = fixture_returns([6_000_000])
    people.loc[0, "rental_income"] = -0.0
    projection = declare(returns, people)
    original = projection.sha256
    people.loc[0, "employment_income_before_lsr"] = 1
    returns.loc[0, "weight"] = 999
    restored_returns, restored_people = tail.projection_tables(projection)
    assert projection.sha256 == original
    assert restored_returns.weight.iloc[0] == 1
    assert restored_people.employment_income_before_lsr.iloc[0] == 6_000_000
    assert np.signbit(restored_people.rental_income.iloc[0])
    assert restored_people[
        tail.native_puf_tail_profile().boolean_outputs[0]
    ].dtype == np.dtype(bool)
    restored_people.loc[0, "employment_income_before_lsr"] = 2
    assert (
        tail.projection_tables(projection)[1].employment_income_before_lsr.iloc[0]
        == 6_000_000
    )


@pytest.mark.parametrize(
    "basis", ["observed_persons", "source_qualified", "modeled_allocation"]
)
def test_no_real_person_or_unimplemented_model_authority(basis):
    returns, people = fixture_returns([6_000_000])
    with pytest.raises(ValueError, match="PROVENANCE"):
        tail.declare_puf_tail_role_projection(
            returns,
            people,
            return_known=pd.DataFrame(
                True, index=returns.index, columns=returns.columns
            ),
            person_known=pd.DataFrame(True, index=people.index, columns=people.columns),
            provenance=tail.TailProjectionProvenance(basis=basis),
            weight_kind=WeightKind.DESIGN,
        )


@pytest.mark.parametrize("where", ["return", "person"])
def test_unknown_cells_refuse_without_becoming_zero(where):
    returns, people = fixture_returns([6_000_000])
    known_returns = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    known_people = pd.DataFrame(True, index=people.index, columns=people.columns)
    (known_returns if where == "return" else known_people).loc[
        0, "employment_income_before_lsr"
    ] = False
    with pytest.raises(ValueError, match="KNOWNNESS"):
        tail.declare_puf_tail_role_projection(
            returns,
            people,
            return_known=known_returns,
            person_known=known_people,
            provenance=tail.TailProjectionProvenance(),
            weight_kind=WeightKind.DESIGN,
        )


def test_integer_boolean_person_cells_refuse_instead_of_coercing():
    returns, people = fixture_returns([6_000_000])
    column = tail.native_puf_tail_profile().boolean_outputs[0]
    people[column] = people[column].astype(np.int64)
    with pytest.raises(ValueError, match="BOOLEAN_DTYPE"):
        declare(returns, people)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "nan", "duplicate_id", "float_id"]
)
def test_invalid_return_surface_refuses(change):
    returns, people = fixture_returns([6_000_000, 7_000_000])
    if change == "missing":
        returns = returns.drop(columns="rental_income")
    elif change == "extra":
        returns["prior_year_wages"] = 1.0
    elif change == "nan":
        returns.loc[0, "rental_income"] = np.nan
    elif change == "duplicate_id":
        returns.loc[1, "donor_recid"] = 1
    else:
        returns["donor_recid"] = returns.donor_recid.astype(float)
    with pytest.raises(ValueError):
        declare(returns, people)


def test_masks_cannot_smuggle_ineligible_or_zero_weight_gains():
    returns, people = fixture_returns([1_000_000], weights=[0])
    with pytest.raises(ValueError, match="CAPITAL_GAINS_ELIGIBILITY"):
        select(returns, people, gains=[True])


def test_exact_integer_identity_is_not_coerced_through_mixed_return_row():
    recid = 2**53 + 1
    returns, people = fixture_returns([6_000_000], ids=[recid])
    result = select(returns, people)
    assert result.donors[0].donor_recid == recid


@pytest.mark.parametrize("role", ["ambiguous", "dependent"])
def test_cg_only_inherits_declared_mask_without_claiming_person_fidelity(role):
    returns, people = fixture_returns([4_000_000])
    people.loc[0, "role"] = role
    result = select(returns, people, gains=[True])
    assert [(row.donor_recid, row.arm) for row in result.donors] == [(1, 1)]
    receipt = json.loads(result.receipt)
    assert receipt["agi_candidate_count"] == 0
    assert receipt["skipped"]["count"] == 0
    assert receipt["role_eligibility_scope"] == "agi_only_and_both_arms"


def test_exact_integer_person_amount_mismatch_cannot_round_to_equality():
    returns, people = fixture_returns([6_000_000])
    returns["employment_income_before_lsr"] = np.array([2**53 + 1], dtype=np.int64)
    people["employment_income_before_lsr"] = np.array([2**53], dtype=np.int64)
    assert select(returns, people).donors == ()
    people["employment_income_before_lsr"] = returns.employment_income_before_lsr.copy()
    assert len(select(returns, people).donors) == 1


@pytest.mark.parametrize(
    "kind", [WeightKind.IMPORTANCE, WeightKind.CALIBRATED, "design"]
)
def test_return_design_weight_kind_is_explicit_and_cannot_be_forged(kind):
    returns, people = fixture_returns([6_000_000])
    projection = replace(declare(returns, people), weight_kind=kind)
    with pytest.raises(ValueError, match="RETURN_WEIGHT_KIND"):
        tail.projection_tables(projection)


def test_selection_identity_binds_declared_masks_and_separates_weight_scope():
    returns, people = fixture_returns([4_000_000, 6_000_000])
    first = select(returns, people)
    gains = select(returns, people, gains=[True, False])
    eligibility = select(returns, people, eligible=[False, True])
    assert (
        first.projection_sha256
        == gains.projection_sha256
        == eligibility.projection_sha256
    )
    assert (
        len(
            {
                first.selection_input_sha256,
                gains.selection_input_sha256,
                eligibility.selection_input_sha256,
            }
        )
        == 3
    )
    assert first.donors == eligibility.donors
    assert first.weight_kind is WeightKind.IMPORTANCE
    receipt = json.loads(first.receipt)
    assert receipt["input_weight_kind"] == "design"
    assert receipt["selected_weight_kind"] == "importance"
    assert receipt["weight_scope"] == "donor_return_support_not_population_households"


def test_forged_snapshot_content_is_revalidated_at_consumption():
    returns, people = fixture_returns([6_000_000])
    projection = declare(returns, people)
    forged = replace(projection, roles=("observed_head",))
    with pytest.raises(ValueError):
        tail.select_native_puf_tail(
            forged,
            capital_gains_mask=np.array([False]),
            source_eligible=np.array([True]),
        )


def test_thinning_cap_and_cell_returns_are_exact_and_input_order_independent():
    n = 3007
    amounts = np.arange(n, dtype=np.float64) + 5_000_000
    returns, people = fixture_returns(
        amounts, weights=np.arange(n) % 5 + 1, statuses=np.arange(n) % 4 + 1
    )
    first = select(returns, people)
    second = select(
        returns.iloc[::-1].reset_index(drop=True),
        people.iloc[::-1].reset_index(drop=True),
    )
    assert first.donors == second.donors
    assert len(first.donors) == 3000
    report = json.loads(first.receipt)
    assert report["thinning"]["dropped_count"] == 7
    for cell in report["thinning"]["cells"]:
        assert cell["weight_before"] == cell["weight_after"]
        assert cell["kept_count"] >= 1
        # Independently total input rows and returned donor objects, not the
        # receipt's two self-reported weights. This fixture's amount order is
        # its exact source-id order, so rank-decile membership is known.
        status, decile = cell["filing_status_code"], cell["proxy_agi_decile"]
        original = returns.loc[
            returns.filing_status_code.eq(status)
            & ((returns.donor_recid - 1) * 10 // n).eq(decile),
            "weight",
        ].to_numpy()
        selected = np.array(
            [
                donor.weight
                for donor in first.donors
                if (donor.donor_recid - 1) % 4 + 1 == status
                and (donor.donor_recid - 1) * 10 // n == decile
            ]
        )
        assert len(selected) >= 1
        assert selected.sum() == original.sum()
    assert report["thinning"]["amount_mass_conserved"] is False
