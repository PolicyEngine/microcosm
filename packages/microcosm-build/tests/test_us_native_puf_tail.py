"""Invented role projections; none authenticates actual PUF persons or a run."""

import itertools
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


def set_component(returns, people, name, values):
    """Declare one proxy component on both grains of a one-person-per-return fixture."""
    array = np.asarray(values)
    returns[name] = array
    if name in people.columns:
        people[name] = array.copy()


def add_person(people, *, person_id, donor_recid, role, **amounts):
    """Append one all-zero technical person row, then set the named cells."""
    profile = tail.native_puf_tail_profile()
    row = people.iloc[[0]].copy()
    row["person_id"] = np.array([person_id], dtype=np.int64)
    row["donor_recid"] = np.array([donor_recid], dtype=np.int64)
    row["role"] = pd.array([role], dtype="string")
    for name in profile.person_outputs:
        row[name] = np.zeros(1, dtype=row[name].dtype)
    for name, value in amounts.items():
        row[name] = np.array([value], dtype=row[name].dtype)
    return pd.concat([people, row], ignore_index=True)


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


def receipt_without_identities(selection):
    """Drop the two content digests so fixture-order variants stay comparable."""
    report = json.loads(selection.receipt)
    for key in ("projection_sha256", "selection_input_sha256"):
        report.pop(key)
    return report


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


def test_filing_status_domain_comes_from_the_existing_puf_definitions():
    from microcosm.build.us_runtime.puf_capital_gains_tail import (
        _FILING_STATUS_BY_CODE,
    )
    from microcosm.build.us_runtime.puf_support import _FILING_STATUS_CODES

    domain = tail._filing_status_domain()
    assert domain == (1, 2, 3, 4, 5)
    assert set(domain) == {int(code) for code in _FILING_STATUS_CODES.values()}
    assert set(domain) == set(_FILING_STATUS_BY_CODE)
    assert _FILING_STATUS_BY_CODE[5] == "SURVIVING_SPOUSE"


@pytest.mark.parametrize("status", [1, 2, 3, 4, 5])
def test_every_shared_filing_status_code_is_admitted(status):
    returns, people = fixture_returns([6_000_000], statuses=[status])
    result = select(returns, people)
    assert [row.donor_recid for row in result.donors] == [1]
    assert json.loads(result.receipt)["filing_status_domain"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("status", [0, -1, 6, 99])
def test_filing_status_outside_the_shared_domain_refuses(status):
    returns, people = fixture_returns([6_000_000], statuses=[status])
    with pytest.raises(ValueError, match="FILING_STATUS_DOMAIN"):
        declare(returns, people)


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
    assert all(
        row.weight_treatment == tail.WEIGHT_TREATMENT_UNCHANGED for row in result.donors
    )
    receipt = json.loads(result.receipt)
    assert receipt["agi_candidate_count"] == 2
    assert receipt["capital_gains_candidate_count"] == 2
    assert receipt["capital_gains_selected_count"] == 2
    assert receipt["capital_gains_skipped_count"] == 0
    assert receipt["both_candidate_count"] == 1
    assert receipt["both_selected_count"] == 1
    assert receipt["both_skipped_count"] == 0
    assert receipt["weight_treatment_counts"] == {
        tail.WEIGHT_TREATMENT_UNCHANGED: 3,
        tail.WEIGHT_TREATMENT_RESCALED: 0,
    }


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


@pytest.mark.parametrize(
    "reach, selected",
    [(5_000_000, [1]), (4_999_999, [])],
)
def test_exact_integer_proxy_components_decide_the_inclusive_floor(reach, selected):
    head = 2**53 + 1
    remainder = reach - head
    returns, people = fixture_returns([0.0])
    set_component(
        returns,
        people,
        "employment_income_before_lsr",
        np.array([head], dtype=np.int64),
    )
    set_component(
        returns, people, "rental_income", np.array([remainder], dtype=np.int64)
    )
    # Accumulating the same two components in float64 lands one dollar low,
    # which is exactly the rounding the exact integer part now bypasses.
    assert float(np.float64(head) + np.float64(remainder)) == float(reach) - 1.0
    result = select(returns, people)
    assert [row.donor_recid for row in result.donors] == selected


@pytest.mark.parametrize(
    "middle, selected",
    [(4_999_999.0, []), (5_000_001.0, [1])],
)
def test_float_proxy_cancellation_is_summed_stably_at_the_floor(middle, selected):
    returns, people = fixture_returns([1e16])
    set_component(
        returns,
        people,
        "self_employment_income_before_lsr",
        np.array([middle], dtype=np.float64),
    )
    set_component(
        returns,
        people,
        "sstb_self_employment_income_before_lsr",
        np.array([-1e16], dtype=np.float64),
    )
    # Left-to-right float64 accumulation absorbs the middle term into 1e16 and
    # recovers 5,000,000.0 for both cases; the stable sum does not.
    assert (1e16 + middle) - 1e16 == 5_000_000.0
    result = select(returns, people)
    assert [row.donor_recid for row in result.donors] == selected
    if selected:
        assert result.donors[0].proxy_agi == middle


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
    receipt = json.loads(result.receipt)
    assert receipt["skipped"]["count"] == 1
    assert receipt["skipped"]["by_reason"]["dependent_monetary_value"]["count"] == 1
    assert receipt["capital_gains_candidate_count"] == 1
    assert receipt["capital_gains_selected_count"] == 0
    assert receipt["capital_gains_skipped_count"] == 1
    assert receipt["both_candidate_count"] == 1
    assert receipt["both_skipped_count"] == 1


def test_boolean_dependent_is_ignored_and_a_nonzero_spouse_payload_is_described():
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
    assert result.donors[0].spouse_payload_nonzero is True
    receipt = json.loads(result.receipt)
    assert receipt["dependent_boolean_policy"] == "ignored_explicitly"
    assert receipt["spouse_payload_flag"] == (
        "descriptive_nonzero_declared_spouse_cells_not_a_structural_requirement"
    )


@pytest.mark.parametrize("with_spouse_row", [False, True])
def test_joint_return_with_a_zero_spouse_payload_is_described_not_required(
    with_spouse_row,
):
    returns, people = fixture_returns([6_000_000], statuses=[2])
    if with_spouse_row:
        people = add_person(people, person_id=11, donor_recid=1, role="spouse")
    result = select(returns, people)
    # A declared-zero spouse payload is a fact about the fixture cells. It is
    # not a finding that a joint return needs no spouse: this contract never
    # consults filing status for structure, and the later population owner
    # decides what household the donor becomes.
    assert result.donors[0].spouse_payload_nonzero is False
    assert json.loads(result.receipt)["filing_status_domain"] == [1, 2, 3, 4, 5]


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


@pytest.mark.parametrize("change", ["two_heads", "two_spouses", "no_head"])
def test_unrepresentable_role_multiplicity_skips_without_choosing_one(change):
    returns, people = fixture_returns([6_000_000])
    if change == "two_heads":
        people = add_person(people, person_id=11, donor_recid=1, role="head")
    elif change == "two_spouses":
        people = add_person(people, person_id=11, donor_recid=1, role="spouse")
        people = add_person(people, person_id=12, donor_recid=1, role="spouse")
    else:
        people.loc[0, "role"] = "spouse"
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


def test_within_donor_person_permutation_does_not_change_the_selection():
    returns, people = fixture_returns([2.0**53])
    people = add_person(
        people,
        person_id=11,
        donor_recid=1,
        role="spouse",
        employment_income_before_lsr=1.0,
    )
    people = add_person(people, person_id=12, donor_recid=1, role="dependent")
    base = select(returns, people)
    assert base.donors[0].spouse_payload_nonzero is True
    for order in itertools.permutations(range(len(people))):
        permuted = people.iloc[list(order)].reset_index(drop=True)
        result = select(returns, permuted)
        assert result.donors == base.donors
        assert receipt_without_identities(result) == receipt_without_identities(base)


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


def test_projection_and_selection_digests_discriminate_fixture_content():
    variants = [
        fixture_returns([6_000_000]),
        fixture_returns([7_000_000]),
        fixture_returns([6_000_000], weights=[2.0]),
        fixture_returns([6_000_000], statuses=[4]),
        fixture_returns([6_000_000], ids=[7]),
    ]
    selections = [
        tail.select_native_puf_tail(
            declare(returns, people),
            capital_gains_mask=np.zeros(1, dtype=bool),
            source_eligible=np.ones(1, dtype=bool),
        )
        for returns, people in variants
    ]
    assert len({result.projection_sha256 for result in selections}) == len(variants)
    assert len({result.selection_input_sha256 for result in selections}) == len(
        variants
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


@pytest.mark.parametrize(
    "change, code",
    [
        ("orphan_person", "ORPHAN_PERSON"),
        ("zero_donor_recid", "IDENTITY_DOMAIN"),
        ("duplicate_person_id", "IDENTITY_DOMAIN"),
        ("negative_weight", "WEIGHT_DOMAIN"),
        ("integer_weight", "WEIGHT_DOMAIN"),
        ("object_role", "ROLE_TYPE"),
        ("unknown_role", "ROLE_TYPE"),
    ],
)
def test_declaration_surface_refusals_name_their_own_gate(change, code):
    returns, people = fixture_returns([6_000_000, 7_000_000])
    if change == "orphan_person":
        people.loc[0, "donor_recid"] = 99
    elif change == "zero_donor_recid":
        returns.loc[0, "donor_recid"] = 0
    elif change == "duplicate_person_id":
        people.loc[1, "person_id"] = people.person_id.iloc[0]
    elif change == "negative_weight":
        returns.loc[0, "weight"] = -1.0
    elif change == "integer_weight":
        returns["weight"] = returns.weight.astype(np.int64)
    elif change == "object_role":
        people["role"] = people.role.astype(object)
    else:
        people.loc[0, "role"] = "observed_head"
    with pytest.raises(ValueError, match=code):
        declare(returns, people)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int32", "int16", "uint8"])
@pytest.mark.parametrize("grain", ["return", "person"])
def test_money_narrower_than_int64_or_float64_refuses(dtype, grain):
    returns, people = fixture_returns([6_000_000])
    table = returns if grain == "return" else people
    table["rental_income"] = table.rental_income.astype(dtype)
    with pytest.raises(ValueError, match="AMOUNT_DTYPE"):
        declare(returns, people)


def test_int64_money_remains_admitted_at_both_grains():
    returns, people = fixture_returns([6_000_000])
    set_component(returns, people, "rental_income", np.zeros(1, dtype=np.int64))
    result = select(returns, people)
    assert [row.donor_recid for row in result.donors] == [1]


@pytest.mark.parametrize("forged", ["<f4", ">i8", "<i4", "|b1", "not-a-dtype"])
def test_forged_snapshot_dtype_refuses_instead_of_reinterpreting_bytes(forged):
    returns, people = fixture_returns([6_000_000])
    projection = declare(returns, people)
    columns = list(projection.returns.columns)
    assert columns[1].name == "weight"
    columns[1] = replace(columns[1], dtype=forged)
    forged_table = replace(projection.returns, columns=tuple(columns))
    with pytest.raises(ValueError, match="SNAPSHOT_"):
        tail.projection_tables(replace(projection, returns=forged_table))


@pytest.mark.parametrize("rows", [0, -1, tail.MAX_DECLARED_RETURNS + 1])
def test_forged_snapshot_row_count_refuses(rows):
    returns, people = fixture_returns([6_000_000])
    projection = declare(returns, people)
    forged = replace(projection, returns=replace(projection.returns, rows=rows))
    with pytest.raises(ValueError, match="SNAPSHOT_ROWS"):
        tail.projection_tables(forged)


def test_masks_cannot_smuggle_ineligible_or_zero_weight_gains():
    returns, people = fixture_returns([1_000_000], weights=[0])
    with pytest.raises(ValueError, match="CAPITAL_GAINS_ELIGIBILITY"):
        select(returns, people, gains=[True])


@pytest.mark.parametrize("which", ["capital_gains_mask", "source_eligible"])
@pytest.mark.parametrize(
    "malformed",
    [
        "wrong_length",
        "integer_dtype",
        "two_dimensional",
        "python_list",
        "masked_scalar",
    ],
)
def test_malformed_selection_masks_refuse(which, malformed):
    returns, people = fixture_returns([6_000_000])
    projection = declare(returns, people)
    values = {
        "wrong_length": np.zeros(2, dtype=bool),
        "integer_dtype": np.zeros(1, dtype=np.int8),
        "two_dimensional": np.zeros((1, 1), dtype=bool),
        "python_list": [False],
        "masked_scalar": np.bool_(False),
    }[malformed]
    masks = {
        "capital_gains_mask": np.zeros(1, dtype=bool),
        "source_eligible": np.ones(1, dtype=bool),
    }
    masks[which] = values
    with pytest.raises(ValueError, match="MASK"):
        tail.select_native_puf_tail(projection, **masks)


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
    # The capital-gains arm is never role-screened, so its spouse payload was
    # never examined. `None` records "unexamined", not "examined and zero".
    assert result.donors[0].spouse_payload_nonzero is None
    assert result.donors[0].weight_treatment == tail.WEIGHT_TREATMENT_UNCHANGED
    receipt = json.loads(result.receipt)
    assert receipt["agi_candidate_count"] == 0
    assert receipt["skipped"]["count"] == 0
    assert receipt["role_eligibility_scope"] == "agi_only_and_both_arms"
    assert receipt["spouse_payload_scope"] == "null_outside_role_eligibility_scope"
    assert receipt["capital_gains_candidate_count"] == 1
    assert receipt["capital_gains_selected_count"] == 1
    assert receipt["capital_gains_skipped_count"] == 0
    assert receipt["both_candidate_count"] == 0


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
    assert receipt["weight_treatment_counts"] == {
        tail.WEIGHT_TREATMENT_UNCHANGED: 1,
        tail.WEIGHT_TREATMENT_RESCALED: 0,
    }
    assert receipt["proxy_is_calculated_agi"] is False
    assert receipt["proxy_floor_arithmetic"] == (
        "exact_integer_part_with_correctly_rounded_float_part"
    )
    assert receipt["proxy_totals_arithmetic"] == (
        "float64_diagnostic_sums_not_exact_amount_reconciliation"
    )


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


@pytest.mark.parametrize(
    "counts",
    [
        [3000],
        [1, 1, 1],
        [2000, 1500, 10],
        [100] * 40,
        [1] * 50 + [3000],
        [7] * 50,
    ],
)
def test_cell_quotas_meet_their_own_postconditions(counts):
    values = np.asarray(counts, dtype=np.int64)
    quotas = tail._cell_quotas(values)
    budget = min(tail.MAX_AGI_ONLY_DONORS, int(values.sum()))
    assert int(quotas.sum()) == budget
    assert bool((quotas >= 1).all())
    assert bool((quotas <= values).all())


@pytest.mark.parametrize(
    "multiplier", [0.7, 1.3, 0.999, 1.0000000001, 0.9999999999, 3.0, 1.0]
)
def test_rescale_exact_only_ever_corrects_a_largest_weight(multiplier):
    # Correction placement, not a claim that this call needed a correction:
    # proportional scaling of a three-element cell is often already exact.
    # Whether or not a residual arises, only a largest weight may absorb it,
    # so every non-maximal weight must still equal its proportional share.
    weights = np.array([1e12, 3e12, 1e-6])
    base = float(weights.sum())
    total = base * multiplier
    scale = total / base
    result = tail._rescale_exact(weights.copy(), total)
    assert float(result.sum()) == total
    assert bool((result > 0).all())
    assert result[0] == weights[0] * scale
    assert result[2] == weights[2] * scale


@pytest.mark.parametrize("total", [0.0, -1.0, float("inf"), float("nan")])
def test_rescale_exact_refuses_a_cell_mass_it_cannot_represent(total):
    with pytest.raises(ValueError, match="EXACT_CELL_WEIGHT"):
        tail._rescale_exact(np.array([1.0, 3.0, 5.0]), total)


def test_rescale_exact_refuses_finite_positive_unrepresentable_mass():
    # Two positive float64 weights cannot sum to the smallest positive float.
    # This exercises the helper's defensive domain, not an admitted thinning
    # call: the latter's cell total is at least the selected weight sum.
    total = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="EXACT_CELL_WEIGHT"):
        tail._rescale_exact(np.array([1.0, 1.0]), total)


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
        assert (cell["dropped_count"] > 0) == (
            cell["weight_treatment"] == tail.WEIGHT_TREATMENT_RESCALED
        )
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
    # The amount receipts must actually move: a regression that silently
    # conserved or dropped the proxy deltas would otherwise pass.
    moved = [c for c in report["thinning"]["cells"] if c["dropped_count"] > 0]
    assert moved
    assert all(c["proxy_agi_sum_before"] != c["proxy_agi_sum_after"] for c in moved)
    assert all(
        c["weighted_proxy_agi_before"] != c["weighted_proxy_agi_after"] for c in moved
    )
    assert report["thinning"]["rescaled_cell_count"] == len(moved)
    assert report["weight_treatment_counts"][tail.WEIGHT_TREATMENT_RESCALED] == sum(
        c["kept_count"] for c in moved
    )
    assert report["weight_treatment_counts"][
        tail.WEIGHT_TREATMENT_UNCHANGED
    ] == 3000 - sum(c["kept_count"] for c in moved)


def test_dispersed_cell_weights_are_conserved_exactly_through_thinning():
    n = 3004
    amounts = np.arange(n, dtype=np.float64) + 5_000_000
    cycle = np.array([1e-6, 1e6, 1.0, 1e3, 5e-3, 2e-4, 7e5], dtype=np.float64)
    weights = np.tile(cycle, n // len(cycle) + 1)[:n]
    returns, people = fixture_returns(
        amounts, weights=weights, statuses=np.arange(n) % 5 + 1
    )
    result = select(returns, people)
    assert len(result.donors) == 3000
    assert all(row.weight > 0 for row in result.donors)
    report = json.loads(result.receipt)
    assert report["thinning"]["dropped_count"] == 4
    assert report["thinning"]["rescaled_cell_count"] >= 1
    for cell in report["thinning"]["cells"]:
        assert cell["weight_before"] == cell["weight_after"]
        assert cell["kept_count"] >= 1


def test_both_arm_role_skip_reconciles_every_arm_count_with_thinning():
    n = 3004
    amounts = np.arange(n, dtype=np.float64) + 5_000_000
    returns, people = fixture_returns(
        amounts, weights=np.arange(n) % 5 + 1, statuses=np.arange(n) % 5 + 1
    )
    # Donor 1 sits in both arms and fails the role screen; donor 2 sits in
    # both arms and passes. Neither is eligible for AGI-only thinning.
    people.loc[0, "role"] = "ambiguous"
    gains = np.zeros(n, dtype=bool)
    gains[:2] = True
    result = select(returns, people, gains=gains)
    report = json.loads(result.receipt)
    assert report["agi_candidate_count"] == n
    assert report["agi_selected_count"] == 3001
    assert report["capital_gains_candidate_count"] == 2
    assert report["capital_gains_selected_count"] == 1
    assert report["capital_gains_skipped_count"] == 1
    assert report["both_candidate_count"] == 2
    assert report["both_selected_count"] == 1
    assert report["both_skipped_count"] == 1
    assert report["skipped"]["count"] == 1
    assert report["thinning"]["count_before"] == 3002
    assert report["thinning"]["dropped_count"] == 2
    assert report["agi_candidate_count"] == (
        report["agi_selected_count"]
        + report["skipped"]["count"]
        + report["thinning"]["dropped_count"]
    )
    assert report["capital_gains_candidate_count"] == (
        report["capital_gains_selected_count"] + report["capital_gains_skipped_count"]
    )
    assert report["both_candidate_count"] == (
        report["both_selected_count"] + report["both_skipped_count"]
    )
    assert 1 not in {row.donor_recid for row in result.donors}


@pytest.mark.parametrize("amount, arm", [(4_000_000.0, 1), (6_000_000.0, 3)])
def test_capital_gains_and_both_arm_support_is_never_capped(amount, arm):
    n = 3001
    returns, people = fixture_returns([amount] * n, statuses=np.arange(n) % 5 + 1)
    result = select(returns, people, gains=np.ones(n, dtype=bool))
    assert len(result.donors) == n
    assert {row.arm for row in result.donors} == {arm}
    assert all(
        row.weight_treatment == tail.WEIGHT_TREATMENT_UNCHANGED for row in result.donors
    )
    report = json.loads(result.receipt)
    assert report["thinning"]["count_before"] == 0
    assert report["thinning"]["kept_count"] == 0
    assert report["thinning"]["dropped_count"] == 0
    assert report["thinning"]["rescaled_cell_count"] == 0
    assert report["capital_gains_candidate_count"] == n
    assert report["capital_gains_selected_count"] == n
    assert report["capital_gains_skipped_count"] == 0
    assert report["weight_treatment_counts"] == {
        tail.WEIGHT_TREATMENT_UNCHANGED: n,
        tail.WEIGHT_TREATMENT_RESCALED: 0,
    }
