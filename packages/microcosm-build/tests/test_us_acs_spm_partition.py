"""Invented ACS households; real canonical assembly, no country calculations."""

import pandas as pd
import pytest

from microcosm.build.acs_spm_partition import (
    ACS_SPM_DEVELOPMENT_POLICY,
    ACS_SPM_PARTITION_POLICY,
    AcsSpmLink,
    AcsSpmLinkAssessment,
    AcsSpmRoleDecision,
    UnsupportedAssembler,
    probe_acs_spm_assembler,
    reconstruct_acs_spm_partition,
)

# Marks exactly the cases that reach the canonical assembler. Strict, so an
# unexpected pass stays a failure, and narrowed to the dedicated exception, so
# the ValueError refusals some of these reach afterwards are never masked. The
# source-validation, group-quarters and unresolved-household cases carry no
# marker: they must keep passing on every runtime.
requires_assembler = pytest.mark.xfail(
    not probe_acs_spm_assembler().supported,
    strict=True,
    raises=UnsupportedAssembler,
    reason="Assembled membership requires the reviewed canonical assembler",
)


def roster(relationships, ages, *, marital=None):
    n = len(ages)
    if marital is None:
        marital = [5] * n
        if any(rel in {21, 23} for rel in relationships):
            marital = [1 if rel in {20, 21, 23} else 5 for rel in relationships]
    return pd.DataFrame(
        {
            "person_id": range(101, 101 + n),
            "person_household_id": [10] * n,
            "person_spm_unit_id": [700] * n,
            "person_tax_unit_id": range(900, 900 + n),
            "SPORDER": range(1, n + 1),
            "RELSHIPP": relationships,
            "AGEP": ages,
            "MAR": marital,
            "TYPEHUGQ": [1] * n,
            # Authentic raw evidence is not replaced by the old CPS recode.
            "A_EXPRRP": [13] * n,
            "PEPAR2": [1] * n,
            "SPM_ID": [700] * n,
            "family_id": [700] * n,
            "source_person_id": [f"raw-{i}" for i in range(n)],
        }
    )


def assemble(persons, **kwargs):
    old_units = pd.DataFrame(
        {
            "spm_unit_id": persons.person_spm_unit_id.unique(),
            "snap": 123.0,
            "spm_unit_net_income": 456.0,
        }
    )
    counts = kwargs.pop(
        "household_person_counts",
        persons.groupby("person_household_id").size().to_dict(),
    )
    return reconstruct_acs_spm_partition(
        persons, old_units, household_person_counts=counts, **kwargs
    )


def groups(result):
    return {
        frozenset(group.person_id)
        for _, group in result.membership.groupby("proposed_spm_unit_id")
    }


def complete(pid):
    return AcsSpmLinkAssessment(pid, "complete", "invented_reviewed_links_v1")


@requires_assembler
def test_reference_family_and_partner_keep_raw_evidence():
    persons = roster([20, 21, 25, 29, 22], [45, 44, 10, 70, 40])
    original = persons.copy(deep=True)
    result = assemble(persons)
    assert groups(result) == {frozenset(persons.person_id)}
    assert result.membership.RELSHIPP.tolist() == persons.RELSHIPP.tolist()
    assert result.provenance["canonical_diagnostics"]["method"] != "native_spm_id"
    assert result.regrouping.action.eq("membership_unchanged").all()
    pd.testing.assert_frame_equal(persons, original)


@requires_assembler
def test_adult_roommates_split_without_external_pointer_graph():
    persons = roster([20, 34, 36], [45, 35, 39])
    result = assemble(persons)
    assert groups(result) == {frozenset([101]), frozenset([102]), frozenset([103])}
    result.require_resolved()
    assert result.crosswalk.person_count.tolist() == [1, 1, 1]
    assert set(result.regrouping.field) == {"snap", "spm_unit_net_income"}
    assert result.regrouping.action.eq("requires_regrouping").all()
    assert "snap" not in result.membership
    assert "person_tax_unit_id" not in result.membership
    assert result.provenance["modeled_residual_separation_people"] == 2


@requires_assembler
def test_parent_link_precedes_unrelated_child_fallback():
    # Census WP2011-22 printed p.7: under-15 fallback excludes another member's child.
    persons = roster([20, 34, 36], [45, 35, 8])
    result = assemble(
        persons,
        links=[AcsSpmLink(103, 102, "parent", "invented_parent_rule_v1")],
        assessments=[complete(102), complete(103)],
    )
    assert groups(result) == {frozenset([101]), frozenset([102, 103])}
    assert result.links.source.eq("approved_inference").all()
    assert (
        "link_parent_child"
        in result.provenance["canonical_diagnostics"]["fallback_rules_used"]
    )


@requires_assembler
def test_partner_child_links_join_primary_component():
    result = assemble(
        roster([20, 22, 36], [45, 35, 8]),
        links=[AcsSpmLink(103, 102, "parent", "invented_parent_rule_v1")],
        assessments=[complete(103)],
    )
    assert groups(result) == {frozenset([101, 102, 103])}
    assert "source_observed" in set(result.links.source)
    assert "approved_inference" in set(result.links.source)


@requires_assembler
@pytest.mark.parametrize("age", [14, 17, 21, 22, 25])
def test_foster_age_boundary(age):
    result = assemble(roster([20, 35], [45, age]))
    assert len(groups(result)) == (1 if age < 22 else 2)


def test_unassessed_possible_parent_does_not_become_head_attachment():
    result = assemble(roster([20, 34, 36], [45, 35, 8]))
    assert result.membership.proposed_spm_unit_id.isna().all()
    assert result.provenance["unresolved_households"] == 1
    assert result.regrouping.action.eq("unresolved_partition").all()
    with pytest.raises(ValueError, match="unresolved"):
        result.require_resolved()


def test_explicit_ambiguous_assessment_is_not_overridden_by_a_link():
    result = assemble(
        roster([20, 34, 36, 36], [45, 35, 36, 8]),
        links=[AcsSpmLink(104, 102, "parent", "one_candidate")],
        assessments=[
            complete(102),
            complete(103),
            AcsSpmLinkAssessment(104, "ambiguous", "two_possible_parents"),
        ],
    )
    assert result.membership.proposed_spm_unit_id.isna().all()
    assert "ambiguous" in set(result.membership.secondary_link_status)


@requires_assembler
def test_married_roommates_split_by_assumption_without_an_accepted_spouse_link():
    persons = roster([20, 34, 34], [45, 35, 36], marital=[5, 1, 1])
    assumed = assemble(persons)
    assert len(groups(assumed)) == 3
    assert assumed.provenance["modeled_residual_separation_people"] == 2
    assert assumed.membership.MAR.tolist() == [5, 1, 1]
    assumed.require_resolved()
    result = assemble(
        persons,
        links=[AcsSpmLink(102, 103, "spouse", "invented_spouse_rule_v1")],
        assessments=[complete(102), complete(103)],
    )
    assert groups(result) == {frozenset([101]), frozenset([102, 103])}


@requires_assembler
def test_residual_under15_vs_independent_minor_role_is_separate():
    child = assemble(roster([20, 36], [45, 14]))
    assert len(groups(child)) == 1
    minor = assemble(roster([20, 36], [45, 16]))
    assert len(groups(minor)) == 2
    assert pd.isna(minor.membership.independent_minor_role.iloc[1])
    with pytest.raises(ValueError, match="role"):
        minor.require_resolved()
    resolved = assemble(
        roster([20, 36], [45, 16]),
        role_decisions=[AcsSpmRoleDecision(102, True, "reviewed_singleton_role_v1")],
    )
    resolved.require_resolved()
    assert resolved.membership.role_source.iloc[1] == "approved_inference"


@requires_assembler
def test_minor_head_and_spouse_known_but_partner_role_unresolved():
    result = assemble(roster([20, 21, 22], [17, 16, 16]))
    assert result.membership.independent_minor_role.iloc[:2].tolist() == [True, True]
    assert pd.isna(result.membership.independent_minor_role.iloc[2])
    # The role primitive does not promote a 14-year-old to a measurement adult.
    young = assemble(roster([20], [14]))
    assert young.membership.independent_minor_role.iloc[0]
    assert young.membership.AGEP.iloc[0] == 14


@pytest.mark.parametrize(
    "policy", [ACS_SPM_PARTITION_POLICY, ACS_SPM_DEVELOPMENT_POLICY]
)
def test_gq_membership_is_preserved_separately_from_exclusion(policy):
    persons = roster([37, 37], [17, 40])
    persons["TYPEHUGQ"] = 2
    persons["person_spm_unit_id"] = [700, 701]
    result = assemble(persons, policy=policy)
    assert len(groups(result)) == 2
    assert result.membership.measurement_status.eq(
        "outside_acs_household_universe"
    ).all()
    assert result.membership.independent_minor_role.isna().all()
    assert result.regrouping.action.eq("membership_unchanged").all()
    result.require_resolved()


@requires_assembler
def test_membership_stable_under_shuffle_and_household_complete_chunks():
    first = roster([20, 34], [45, 35])
    second = first.copy()
    second["person_id"] += 100
    second["person_household_id"] += 10
    second["person_spm_unit_id"] += 100
    persons = pd.concat([first, second], ignore_index=True)
    combined = assemble(persons).membership.set_index("person_id").proposed_spm_unit_id
    shuffled = (
        assemble(persons.sample(frac=1, random_state=42))
        .membership.set_index("person_id")
        .proposed_spm_unit_id
    )
    chunked = (
        pd.concat([assemble(first).membership, assemble(second).membership])
        .set_index("person_id")
        .proposed_spm_unit_id
    )
    pd.testing.assert_series_equal(combined.sort_index(), shuffled.sort_index())
    pd.testing.assert_series_equal(combined.sort_index(), chunked.sort_index())


@pytest.mark.parametrize("column", ["person_id", "SPORDER"])
def test_duplicate_pointer_keys_refused(column):
    persons = roster([20, 34], [45, 35])
    persons[column] = 1
    with pytest.raises(ValueError, match="unique"):
        assemble(persons)


@pytest.mark.parametrize("relative", [103, 101])
def test_missing_or_self_parent_refused(relative):
    with pytest.raises(ValueError, match="[Ll]ink"):
        assemble(
            roster([20, 34], [45, 35]),
            links=[AcsSpmLink(101, relative, "parent", "bad")],
        )


def test_cross_household_link_refused():
    persons = roster([20, 20], [45, 35])
    persons["person_household_id"] = [10, 20]
    persons["person_spm_unit_id"] = [700, 701]
    with pytest.raises(ValueError, match="household"):
        assemble(persons, links=[AcsSpmLink(101, 102, "parent", "bad")])


def test_direct_role_contradiction_refused():
    with pytest.raises(ValueError, match="observed"):
        assemble(
            roster([20], [16]), role_decisions=[AcsSpmRoleDecision(101, False, "bad")]
        )


def test_partial_household_is_not_accepted_as_complete():
    with pytest.raises(ValueError, match="complete households"):
        assemble(roster([20, 34], [45, 35]), household_person_counts={10: 3})


@pytest.mark.parametrize(
    "column,value",
    [("AGEP", None), ("AGEP", True), ("RELSHIPP", 99), ("person_id", 1.5)],
)
def test_invalid_or_missing_source_evidence_refused(column, value):
    persons = roster([20], [45])
    persons[column] = value
    with pytest.raises(ValueError):
        assemble(persons)


@requires_assembler
def test_raw_second_parent_alias_is_not_treated_as_an_observation():
    result = assemble(roster([20, 25], [45, 10]))
    parents = result.links.loc[result.links.kind.eq("parent")]
    assert parents[["person_id", "relative_id"]].values.tolist() == [[102, 101]]
    assert parents.source.tolist() == ["source_observed"]


@requires_assembler
def test_inventory_does_not_read_or_reallocate_old_unit_amounts():
    persons = roster([20, 34], [45, 35])
    amount = object()
    old = pd.DataFrame({"spm_unit_id": [700], "opaque_old_amount": [amount]})
    result = reconstruct_acs_spm_partition(
        persons, old, household_person_counts={10: 2}
    )
    assert result.regrouping.field.tolist() == ["opaque_old_amount"]
    assert old.opaque_old_amount.iloc[0] is amount
    assert "opaque_old_amount" not in result.membership


def test_inferred_parent_cannot_create_cycle_with_observed_child():
    with pytest.raises(ValueError, match="cycle"):
        assemble(
            roster([20, 25], [45, 25]),
            links=[AcsSpmLink(101, 102, "parent", "bad_cycle")],
        )


def test_boolean_link_id_cannot_alias_person_one():
    persons = roster([20, 34], [45, 35])
    persons["person_id"] = [1, 2]
    with pytest.raises(ValueError, match="exact"):
        assemble(persons, links=[AcsSpmLink(True, 2, "parent", "bad_id")])


@requires_assembler
def test_unobserved_adult_child_partner_is_a_declared_separation_assumption():
    # RELSHIPP observes only the reference relation, not a child's partner.
    # This proposal does not claim that a secondary partner has been ruled out.
    result = assemble(roster([20, 25, 36], [65, 30, 30]))
    assert groups(result) == {frozenset([101, 102]), frozenset([103])}
    assert (
        result.membership.secondary_link_status.iloc[2] == "modeled_residual_separation"
    )
    assert result.provenance["modeled_residual_separation_people"] == 1
    resolved = assemble(
        roster([20, 25, 36], [65, 30, 30]),
        links=[AcsSpmLink(103, 102, "partner", "reviewed_secondary_partner_v1")],
        assessments=[complete(103)],
    )
    assert groups(resolved) == {frozenset([101, 102, 103])}


@pytest.mark.parametrize("from_person,to_person", [(101, 103), (103, 101), (102, 103)])
def test_inferred_spouse_cannot_contradict_either_observed_pair_endpoint(
    from_person, to_person
):
    with pytest.raises(ValueError, match="[Cc]onflicting|cardinality"):
        assemble(
            roster([20, 21, 36], [45, 44, 43], marital=[1, 1, 1]),
            links=[
                AcsSpmLink(from_person, to_person, "spouse", "contradictory_spouse")
            ],
        )


def test_inferred_spouse_cannot_contradict_raw_never_married():
    with pytest.raises(ValueError, match="MAR"):
        assemble(
            roster([20, 34, 36], [45, 35, 36]),
            links=[AcsSpmLink(102, 103, "spouse", "bad_spouse")],
        )


def test_observed_spouse_requires_coherent_raw_marital_status():
    with pytest.raises(ValueError, match="MAR"):
        assemble(roster([20, 21], [45, 44], marital=[5, 1]))


@requires_assembler
def test_development_preset_pools_unknown_parentage_without_fabricating_links():
    result = assemble(
        roster([20, 34, 36], [45, 35, 8]), policy=ACS_SPM_DEVELOPMENT_POLICY
    )
    assert groups(result) == {frozenset([101, 103]), frozenset([102])}
    assert result.links.empty
    assert (
        result.membership.partition_assumption.iloc[2]
        == "parent_unknown_reference_pooling"
    )
    assert result.membership.parent_link_status.iloc[2] == "unknown"
    assert result.provenance["source_unresolved_relationship_households"] == 1
    assert result.provenance["policy"] == ACS_SPM_DEVELOPMENT_POLICY
    result.require_resolved()


@requires_assembler
def test_development_preset_assigns_modeled_minor_singleton_reference_roles():
    result = assemble(
        roster([20, 36, 34], [45, 16, 17]), policy=ACS_SPM_DEVELOPMENT_POLICY
    )
    assert len(groups(result)) == 3
    assert result.membership.independent_minor_role.tolist() == [True, True, True]
    assert result.membership.role_source.iloc[1:].eq("modeled_assumption").all()
    assert (
        result.membership.role_rule.iloc[1:]
        .eq("minor_singleton_unit_reference_v1")
        .all()
    )
    result.require_resolved()


@requires_assembler
@pytest.mark.parametrize("partner_role", [True, False])
def test_development_minor_partner_sensitivity_preserves_membership(partner_role):
    result = assemble(
        roster([20, 22, 25, 35], [45, 16, 16, 17]),
        policy=ACS_SPM_DEVELOPMENT_POLICY,
        minor_partner_role=partner_role,
    )
    assert len(groups(result)) == 1
    assert result.membership.independent_minor_role.tolist() == [
        True,
        partner_role,
        False,
        False,
    ]
    assert result.membership.role_source.iloc[1:].eq("modeled_assumption").all()
    assert result.provenance["minor_partner_role"] is partner_role
    result.require_resolved()


@requires_assembler
def test_development_accepted_parent_precedes_unknown_parent_pooling():
    result = assemble(
        roster([20, 34, 36], [45, 35, 8]),
        policy=ACS_SPM_DEVELOPMENT_POLICY,
        links=[AcsSpmLink(103, 102, "parent", "accepted_parent_v1")],
    )
    assert groups(result) == {frozenset([101]), frozenset([102, 103])}
    assert (
        result.membership.parent_link_status.iloc[2] == "approved_inferred_parent_link"
    )
    assert result.membership.partition_assumption.iloc[2] == "accepted_sharing_links"
    result.require_resolved()


@requires_assembler
def test_development_unique_accepted_minor_parent_gets_modeled_reference_role():
    result = assemble(
        roster([20, 34, 36], [45, 16, 0]),
        policy=ACS_SPM_DEVELOPMENT_POLICY,
        links=[AcsSpmLink(103, 102, "parent", "accepted_parent_v1")],
    )
    assert groups(result) == {frozenset([101]), frozenset([102, 103])}
    assert result.membership.independent_minor_role.tolist() == [True, True, False]
    assert (
        result.membership.role_rule.iloc[1] == "accepted_parent_component_reference_v1"
    )
    assert result.membership.role_source.iloc[1] == "modeled_assumption"
    result.require_resolved()


@requires_assembler
def test_development_does_not_promote_under15_parent():
    persons = roster([20, 34, 36], [45, 14, 0])
    kwargs = {
        "policy": ACS_SPM_DEVELOPMENT_POLICY,
        "links": [AcsSpmLink(103, 102, "parent", "accepted_parent_v1")],
    }
    result = assemble(persons, **kwargs)
    assert len(groups(result)) == 1
    assert result.membership.independent_minor_role.tolist() == [True, False, False]
    assert result.membership.parent_link_status.iloc[1] == "unknown"
    assert (
        result.membership.partition_assumption.iloc[1]
        == "accepted_sharing_and_parent_unknown_reference_pooling"
    )
    assert result.links[["person_id", "relative_id"]].values.tolist() == [[103, 102]]
    assert result.provenance["modeled_residual_child_attachment_people"] == 1
    with pytest.raises(ValueError, match="15–17"):
        assemble(
            persons,
            **kwargs,
            role_decisions=[AcsSpmRoleDecision(102, True, "invalid_promotion")],
        )


@requires_assembler
def test_development_two_possible_minor_parent_references_need_explicit_roles():
    result = assemble(
        roster([20, 34, 36, 36], [45, 16, 17, 0]),
        policy=ACS_SPM_DEVELOPMENT_POLICY,
        links=[
            AcsSpmLink(104, 102, "parent", "accepted_parent_v1"),
            AcsSpmLink(104, 103, "parent", "accepted_parent_v1"),
        ],
    )
    assert groups(result) == {frozenset([101]), frozenset([102, 103, 104])}
    assert result.membership.independent_minor_role.iloc[1:3].isna().all()
    with pytest.raises(ValueError, match="role"):
        result.require_resolved()


@requires_assembler
def test_development_adult_parent_reference_does_not_promote_minor_coparent():
    result = assemble(
        roster([20, 34, 36, 36], [45, 35, 17, 0]),
        policy=ACS_SPM_DEVELOPMENT_POLICY,
        links=[
            AcsSpmLink(104, 102, "parent", "accepted_parent_v1"),
            AcsSpmLink(104, 103, "parent", "accepted_parent_v1"),
        ],
    )
    assert not result.membership.independent_minor_role.iloc[2]
    result.require_resolved()
