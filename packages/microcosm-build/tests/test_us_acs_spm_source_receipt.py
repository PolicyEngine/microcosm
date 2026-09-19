"""Development receipt boundaries on invented, household-complete ACS rosters."""

import json
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from microcosm.build.acs_spm_partition import (
    ACS_SPM_DEVELOPMENT_POLICY,
    UnsupportedAssembler,
    probe_acs_spm_assembler,
    reconstruct_acs_spm_partition,
)
from microcosm.build.acs_spm_regroup import AcsSpmLegacyDefaults, regroup_acs_spm_units
from microcosm.build.acs_spm_source_receipt import build_acs_spm_source_receipt

requires_assembler = pytest.mark.xfail(
    not probe_acs_spm_assembler().supported,
    strict=True,
    raises=UnsupportedAssembler,
    reason="Development partition requires the reviewed canonical assembler capabilities",
)


def case(*, uncertain=False, uncertain_age=8, gq=False, amount=1200.0):
    relationships = [20, 25, 36] if not uncertain else [20, 34, 36]
    ages = [40, 8, 30] if not uncertain else [40, 30, uncertain_age]
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [10] * 3,
            "person_spm_unit_id": [10] * 3,
            "SPORDER": [1, 2, 3],
            "RELSHIPP": relationships,
            "AGEP": ages,
            "MAR": [5] * 3,
            "TYPEHUGQ": [1] * 3,
            "person_spine": ["acs_test"] * 3,
        }
    )
    units = pd.DataFrame(
        {
            "spm_unit_id": [10],
            "spm_unit_pre_subsidy_childcare_expenses": [amount],
            "receives_housing_assistance": [False],
            "takes_up_housing_assistance_if_eligible": [True],
            "spm_unit_tenure_type": ["RENTER"],
            "spm_unit_energy_subsidy": [0.0],
            "spm_unit_source_id": [70],
            "spm_unit_support_channel": ["acs_test"],
            "spm_unit_support_clone_index": [0],
            "takes_up_tanf_if_eligible": [True],
            "takes_up_snap_if_eligible": [True],
            "spm_unit_spine": ["acs_test"],
        }
    )
    households = pd.DataFrame(
        {"household_id": [10], "TYPEHUGQ": [1], "NP": [3], "TEN": [3]}
    )
    if gq:
        persons["RELSHIPP"] = 37
        persons["TYPEHUGQ"] = 2
        households["TYPEHUGQ"] = 2
        households["TEN"] = np.nan
    defaults = AcsSpmLegacyDefaults(
        "acs_test",
        "a" * 64,
        "b" * 64,
        {
            "receives_housing_assistance": False,
            "spm_unit_energy_subsidy": 0.0,
            "takes_up_tanf_if_eligible": True,
            "takes_up_snap_if_eligible": True,
        },
    )
    partitions = {
        sensitivity: reconstruct_acs_spm_partition(
            persons,
            units,
            household_person_counts={10: 3},
            policy=ACS_SPM_DEVELOPMENT_POLICY,
            minor_partner_role=sensitivity,
        )
        for sensitivity in (False, True)
    }
    labels = partitions[False].membership.proposed_spm_unit_id
    # Invented fixture allocator only. The receipt consumes and binds this registry.
    components = sorted(labels.unique())
    registry = pd.DataFrame(
        {"person_id": persons.person_id, "canonical_spm_unit_id": labels}
    )
    registry["new_spm_unit_id"] = labels.map(
        {
            label: (10 if len(components) == 1 else 100 + i)
            for i, label in enumerate(components)
        }
    )
    registry["new_spm_unit_source_id"] = labels.map(
        {
            label: (70 if len(components) == 1 else 800 + i)
            for i, label in enumerate(components)
        }
    )
    kwargs = dict(
        partitions=partitions,
        membership=registry,
        legacy_defaults=defaults,
        tenure_policy="acs_ten4_no_mortgage_v1",
        source_references={"invented_fixture": "c" * 64},
    )
    kwargs["regroup"] = regroup_acs_spm_units(
        persons,
        units,
        households,
        registry,
        legacy_defaults=defaults,
        tenure_policy=kwargs["tenure_policy"],
    )
    return (persons, units, households), kwargs


@requires_assembler
def test_json_receipt_binds_frozen_metadata_and_once_only_amounts():
    inputs, kwargs = case()
    kwargs["regroup"] = replace(
        kwargs["regroup"], metadata=MappingProxyType(kwargs["regroup"].metadata)
    )
    partition = kwargs["partitions"][False]
    kwargs["partitions"][False] = replace(
        partition,
        provenance=MappingProxyType(
            {
                **partition.provenance,
                "person_count": np.int64(partition.provenance["person_count"]),
                "canonical_diagnostics": MappingProxyType(
                    partition.provenance["canonical_diagnostics"]
                ),
            }
        ),
    )
    before = [table.copy(deep=True) for table in inputs]
    receipt = build_acs_spm_source_receipt(*inputs, **kwargs)
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    assert receipt["scope"] == "development_source_only"
    assert receipt["source_references"]["verification"] == "supplied_not_authenticated"
    assert receipt["counts"]["persons"] == 3
    assert receipt["counts"]["new_units"] == 2
    assert receipt["counts"]["childcare_ledger_rows"] == 1
    assert receipt["unit_authority_counts"]["false"]["modeled_assumption"] == 1
    for actual, expected in zip(inputs, before, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


@requires_assembler
@pytest.mark.parametrize(
    "surface", ["membership", "role", "policy", "crosswalk", "links", "ledger"]
)
def test_stale_result_evidence_is_rejected(surface):
    inputs, kwargs = case()
    result = kwargs["partitions"][False]
    if surface == "membership":
        result.membership.loc[0, "proposed_spm_unit_id"] = "stale"
    elif surface == "role":
        result.membership.loc[0, "independent_minor_role"] = False
    elif surface == "policy":
        result.provenance["policy"] = "stale"
    elif surface == "crosswalk":
        result.crosswalk.loc[0, "person_count"] = 99
    elif surface == "links":
        result.links.loc[0, "rule_id"] = "stale"
    else:
        kwargs["regroup"].childcare_ledger.loc[0, "allocated_amount"] = 2400
    with pytest.raises(ValueError, match="stale|differs"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


@requires_assembler
def test_source_uncertainty_weakens_even_the_observed_head_unit():
    inputs, kwargs = case(uncertain=True)
    receipt = build_acs_spm_source_receipt(*inputs, **kwargs)
    assert (
        kwargs["partitions"][False].membership.role_source.iloc[0]
        == "observed_relationship_rule"
    )
    assert receipt["unit_authority_counts"]["false"] == {"unresolved": 2}


def test_gq_roles_remain_null_without_assembler():
    inputs, kwargs = case(gq=True)
    receipt = build_acs_spm_source_receipt(*inputs, **kwargs)
    assert receipt["counts"]["group_quarters_units"] == 1
    assert receipt["unit_authority_counts"]["false"] == {
        "outside_acs_household_universe": 1
    }
    assert kwargs["partitions"][False].membership.independent_minor_role.isna().all()


@pytest.mark.parametrize(
    "column,value", [("PERIDNUM", "asec-id"), ("person_spine", "asec")]
)
def test_refuses_other_origin_even_if_all_membership_is_unchanged(column, value):
    inputs, kwargs = case(gq=True)
    inputs[0][column] = value
    with pytest.raises(ValueError, match="ACS|ASEC"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


def test_unit_spine_must_equal_declared_acs_spine():
    inputs, kwargs = case(gq=True)
    inputs[1]["spm_unit_spine"] = "asec"
    with pytest.raises(ValueError, match="ACS"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


@requires_assembler
def test_registry_cannot_change_component_membership():
    inputs, kwargs = case()
    kwargs["membership"].loc[0, "canonical_spm_unit_id"] = "stale"
    with pytest.raises(ValueError, match="registry"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


@requires_assembler
def test_partner_sensitivities_cannot_disagree_on_member_sets():
    inputs, kwargs = case()
    kwargs["partitions"][True].membership.loc[0, "proposed_spm_unit_id"] = "stale"
    with pytest.raises(ValueError, match="stale|membership"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


def test_nonfinite_metadata_is_not_silently_hashed_as_null():
    inputs, kwargs = case(gq=True)
    kwargs["partitions"][False].provenance["person_count"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


def test_unchanged_gq_registry_must_retain_old_ids():
    inputs, kwargs = case(gq=True)
    kwargs["membership"]["new_spm_unit_id"] = 999
    with pytest.raises(ValueError, match="Unchanged member sets"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


@requires_assembler
def test_observed_head_role_does_not_hide_an_inferred_partition():
    from microcosm.build.acs_spm_partition import AcsSpmLink, AcsSpmLinkAssessment

    inputs, kwargs = case(amount=0)
    persons, units, households = inputs
    kwargs["links"] = (AcsSpmLink(3, 1, "partner", "invented_reviewed_partner"),)
    kwargs["assessments"] = (AcsSpmLinkAssessment(3, "complete", "invented_review"),)
    kwargs["partitions"] = {
        sensitivity: reconstruct_acs_spm_partition(
            persons,
            units,
            household_person_counts={10: 3},
            policy=ACS_SPM_DEVELOPMENT_POLICY,
            minor_partner_role=sensitivity,
            links=kwargs["links"],
            assessments=kwargs["assessments"],
        )
        for sensitivity in (False, True)
    }
    labels = kwargs["partitions"][False].membership.proposed_spm_unit_id
    registry = pd.DataFrame(
        {
            "person_id": persons.person_id,
            "canonical_spm_unit_id": labels,
            "new_spm_unit_id": 10,
            "new_spm_unit_source_id": 70,
        }
    )
    kwargs["membership"] = registry
    kwargs["regroup"] = regroup_acs_spm_units(
        *inputs,
        registry,
        legacy_defaults=kwargs["legacy_defaults"],
        tenure_policy=kwargs["tenure_policy"],
    )
    receipt = build_acs_spm_source_receipt(*inputs, **kwargs)
    assert receipt["unit_authority_counts"]["false"] == {"approved_inference": 1}
    assert (
        kwargs["partitions"][False].membership.role_source.iloc[0]
        == "observed_relationship_rule"
    )
    assert "source_observed" in set(kwargs["partitions"][False].links.source)


def test_frozen_numpy_defaults_and_blank_asec_column_are_preserved():
    inputs, kwargs = case(gq=True)
    kwargs["legacy_defaults"] = replace(
        kwargs["legacy_defaults"],
        values=MappingProxyType(
            {
                key: np.float64(value)
                if key == "spm_unit_energy_subsidy"
                else np.bool_(value)
                for key, value in kwargs["legacy_defaults"].values.items()
            }
        ),
    )
    inputs[0]["PERIDNUM"] = [pd.NA, "", "  "]
    receipt = build_acs_spm_source_receipt(*inputs, **kwargs)
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt


def test_table_infinity_is_refused_but_gq_source_missingness_is_retained():
    inputs, kwargs = case(gq=True)
    inputs[1].loc[0, "spm_unit_pre_subsidy_childcare_expenses"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        build_acs_spm_source_receipt(*inputs, **kwargs)


@requires_assembler
def test_household_link_uncertainty_reaches_separate_observed_head_unit():
    inputs, kwargs = case(uncertain=True, uncertain_age=16, amount=0)
    receipt = build_acs_spm_source_receipt(*inputs, **kwargs)
    assert receipt["unit_authority_counts"]["false"] == {"unresolved": 3}
