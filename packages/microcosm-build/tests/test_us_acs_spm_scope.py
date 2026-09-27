"""Invented constructor evidence only: no issuers, source files or country runtime."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from microcosm.build.acs_spm_scope import (
    ACSAnalysisProfile,
    ACSSPMScopeError,
    classify_acs_spm_scope,
    count_acs_spm_authorities,
)


def profile(**changes):
    return ACSAnalysisProfile(
        **{
            "year": 2024,
            "source_vintage": "acs_2024_1yr",
            "admit_modeled": False,
            "admit_approved_inference": False,
            **changes,
        }
    )


def member(pid, household, unit, **changes):
    return {
        "person_id": pid,
        "person_household_id": household,
        "old_spm_unit_id": household + 1000,
        "SPORDER": 1,
        "TYPEHUGQ": 1,
        "proposed_spm_unit_id": unit,
        "independent_minor_role": True,
        "role_source": "observed_relationship_rule",
        "role_rule": "acs_relshipp_reference_head_or_spouse",
        "partition_assumption": "reference_relationship_rule",
        "secondary_link_status": "not_required",
        "measurement_status": "included_acs_household",
        **changes,
    }


def table(records, columns=None):
    columns = columns or list(records[0])
    return {
        "columns": columns,
        "dtypes": ["object"] * len(columns),
        "index": list(range(len(records))),
        "rows": [[row[column] for column in columns] for row in records],
    }


def fixture_evidence(members=None, links=(), *, partner=False):
    """JSON-shaped invented evidence, never an authenticated preparation owner."""
    if members is None:
        members = [
            member(1, 10, "u-observed"),
            member(
                2,
                10,
                "u-observed",
                SPORDER=2,
                independent_minor_role=False,
                role_source="age_not_role_sensitive",
                role_rule="canonical_role_age_scope",
            ),
            member(
                3,
                20,
                "u-modeled",
                independent_minor_role=partner,
                role_source="modeled_assumption",
                role_rule="minor_reference_partner_role_sensitivity_v1",
            ),
            member(4, 30, "u-inferred", role_source="approved_inference"),
            member(5, 40, "u-uncertain-head"),
            member(
                6,
                40,
                "u-uncertain-secondary",
                SPORDER=2,
                secondary_link_status="unassessed",
                partition_assumption="parent_unknown_residual_separation",
            ),
            member(
                7,
                50,
                "u-missing",
                independent_minor_role=None,
                role_source="age_not_role_sensitive",
            ),
            *[
                member(
                    pid,
                    hh,
                    name,
                    TYPEHUGQ=kind,
                    independent_minor_role=None,
                    role_source="outside_acs_household_universe",
                    role_rule="not_classified",
                    partition_assumption="preserved_gq_membership",
                    measurement_status="outside_acs_household_universe",
                )
                for pid, hh, name, kind in ((8, 60, "u-gq2", 2), (9, 70, "u-gq3", 3))
            ],
        ]
    units = {}
    for row in members:
        units.setdefault(row["proposed_spm_unit_id"], []).append(row)
    registry = []
    crosswalk = []
    unit_evidence = []
    uncertain = {
        row["person_household_id"]
        for row in members
        if row["secondary_link_status"] in {"unassessed", "ambiguous"}
    }
    for number, (canonical, rows) in enumerate(sorted(units.items()), 100):
        native_keys = sorted(
            (f"2024TEST{row['person_household_id']}", row["SPORDER"]) for row in rows
        )
        component = hashlib.sha256(
            json.dumps(native_keys, separators=(",", ":")).encode()
        ).hexdigest()
        for row in rows:
            registry.append(
                {
                    "serialno": f"2024TEST{row['person_household_id']}",
                    "sporder": row["SPORDER"],
                    "person_id": row["person_id"],
                    "household_id": row["person_household_id"],
                    "old_spm_unit_id": row["old_spm_unit_id"],
                    "new_spm_unit_id": number,
                    "component_sha256": component,
                    "source_record_sha256": "a" * 64,
                }
            )
        crosswalk.append(
            {
                "person_household_id": rows[0]["person_household_id"],
                "old_spm_unit_id": rows[0]["old_spm_unit_id"],
                "proposed_spm_unit_id": canonical,
                "person_count": len(rows),
            }
        )
        unit_evidence.append(
            {
                "proposed_spm_unit_id": canonical,
                "role_sources": sorted({row["role_source"] for row in rows}),
                "partition_assumptions": sorted(
                    {row["partition_assumption"] for row in rows}
                ),
                "secondary_link_statuses": sorted(
                    {row["secondary_link_status"] for row in rows}
                ),
                "household_source_uncertain": rows[0]["person_household_id"]
                in uncertain,
                "outside_acs_household_universe": rows[0]["TYPEHUGQ"] != 1,
            }
        )
    options = {
        "policy": "acs_spm_development_reconstruction_v1",
        "minor_partner_role": partner,
    }
    return {
        "protocol": "microcosm.acs-spm-native-construction-evidence.v1",
        "source_binding": "actual_construction_from_owner_captured_archives",
        "implementation": {"options": options},
        "construction_receipt": {"options": options},
        "partition_provenance": options,
        "registry": {
            "options": options,
            "id_ceiling": 10000,
            "assembler_sha256": "b" * 64,
            "entries": registry,
        },
        "partition": {
            "membership": table(members),
            "links": table(
                list(links), ["person_id", "relative_id", "kind", "source", "rule_id"]
            ),
            "crosswalk": table(crosswalk),
            "regrouping": table([], ["field", "action"]),
        },
        "native_crosswalk": table(registry),
        "unit_evidence": table(unit_evidence),
        "engine_role_delivered": False,
        "annual_universe_declared": False,
        "release_eligible": False,
    }


def set_cell(evidence, table_name, row, column, value):
    target = evidence["partition"][table_name]
    target["rows"][row][target["columns"].index(column)] = value


@pytest.mark.parametrize(
    "modeled,inference", [(False, False), (True, False), (False, True), (True, True)]
)
def test_exact_profile_decisions_preserve_labels_roles_and_whole_household_uncertainty(
    modeled, inference
):
    evidence = fixture_evidence()
    original = copy.deepcopy(evidence)
    result = classify_acs_spm_scope(
        evidence, profile(admit_modeled=modeled, admit_approved_inference=inference)
    )
    by_label = {row.canonical_id: row for row in result.units}
    assert by_label["u-observed"].status == "INCLUDED"
    assert by_label["u-modeled"].status == ("INCLUDED" if modeled else "UNRESOLVED")
    assert by_label["u-modeled"].authority == "modeled_assumption"
    assert by_label["u-inferred"].status == ("INCLUDED" if inference else "UNRESOLVED")
    assert by_label["u-inferred"].authority == "approved_inference"
    for name in ("u-uncertain-head", "u-uncertain-secondary"):
        assert by_label[name].status == "UNRESOLVED"
        assert by_label[name].authority == "unresolved"
        assert by_label[name].reason == "source_membership_or_authority_unresolved"
    assert by_label["u-missing"].reason == "source_role_missing"
    for name in ("u-gq2", "u-gq3"):
        assert by_label[name].status == "OUTSIDE"
        assert by_label[name].authority == "outside_acs_household_universe"
    assert [row.value for row in result.roles] == [
        True,
        False,
        False,
        True,
        True,
        True,
        None,
        None,
        None,
    ]
    assert count_acs_spm_authorities(result.units) == {
        "observed_relationship_rule": 2,
        "modeled_assumption": 1,
        "approved_inference": 1,
        "unresolved": 2,
        "outside_acs_household_universe": 2,
    }
    assert evidence == original
    assert result.profile.year == 2024
    assert result.minor_partner_role is False
    with pytest.raises(FrozenInstanceError):
        result.units[0].status = "INCLUDED"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("source_observed", "observed_relationship_rule"),
        ("approved_inference", "approved_inference"),
    ],
)
def test_internal_link_authority_uses_reviewed_order(source, expected):
    members = [member(1, 10, "u"), member(2, 10, "u", SPORDER=2)]
    link = {
        "person_id": 2,
        "relative_id": 1,
        "kind": "parent",
        "source": source,
        "rule_id": "invented_rule",
    }
    result = classify_acs_spm_scope(
        fixture_evidence(members, [link]), profile(admit_approved_inference=True)
    )
    assert result.units[0].authority == expected
    assert result.units[0].status == "INCLUDED"


def test_row_reordering_and_partner_sensitivity_do_not_change_identity_or_authority():
    evidence = fixture_evidence()
    before = classify_acs_spm_scope(evidence, profile())
    for value in (
        *evidence["partition"].values(),
        evidence["native_crosswalk"],
        evidence["unit_evidence"],
    ):
        value["rows"].reverse()
        value["index"].reverse()
    evidence["registry"]["entries"].reverse()
    assert classify_acs_spm_scope(evidence, profile()) == before
    after = classify_acs_spm_scope(fixture_evidence(partner=True), profile())
    assert after.units == before.units
    assert after.minor_partner_role is True
    assert after.roles[2] == replace(before.roles[2], value=True)
    assert after.roles[:2] + after.roles[3:] == before.roles[:2] + before.roles[3:]


@pytest.mark.parametrize(
    "change",
    [
        {"year": 2025},
        {"year": True},
        {"source_vintage": "acs_2025_1yr"},
        {"admit_modeled": 1},
        {"admit_approved_inference": None},
        {"name": "all_acs"},
        {"methodology": "other"},
        {"lineage": "other"},
    ],
)
def test_profile_is_explicit_and_never_carries_to_another_year(change):
    with pytest.raises(ACSSPMScopeError):
        profile(**change)


@pytest.mark.parametrize(
    "defect",
    [
        "row_width",
        "duplicate_column",
        "dtype_count",
        "index_count",
        "duplicate_person",
        "role_integer",
        "unknown_authority",
        "unknown_assumption",
        "unknown_secondary",
        "mixed_household_kind",
        "outside_observed_role",
        "native_membership",
        "native_id_type",
        "registry_coverage",
        "canonical_merge",
        "unit_evidence",
        "partition_count",
        "partner_option",
        "scope_already_declared",
        "missing_column",
    ],
)
def test_malformed_or_inconsistent_construction_evidence_refuses(defect):
    evidence = fixture_evidence()
    membership = evidence["partition"]["membership"]
    if defect == "row_width":
        membership["rows"][0].pop()
    elif defect == "duplicate_column":
        membership["columns"][1] = membership["columns"][0]
    elif defect == "dtype_count":
        membership["dtypes"].pop()
    elif defect == "index_count":
        membership["index"].pop()
    elif defect == "duplicate_person":
        set_cell(evidence, "membership", 1, "person_id", 1)
    elif defect == "role_integer":
        set_cell(evidence, "membership", 0, "independent_minor_role", 1)
    elif defect == "unknown_authority":
        set_cell(evidence, "membership", 0, "role_source", "guessed_observation")
    elif defect == "unknown_assumption":
        set_cell(evidence, "membership", 0, "partition_assumption", "default")
    elif defect == "unknown_secondary":
        set_cell(evidence, "membership", 0, "secondary_link_status", "default")
    elif defect == "mixed_household_kind":
        set_cell(evidence, "membership", 1, "TYPEHUGQ", 2)
    elif defect == "outside_observed_role":
        set_cell(evidence, "membership", 7, "independent_minor_role", False)
    elif defect == "native_membership":
        row = evidence["native_crosswalk"]
        row["rows"][0][row["columns"].index("new_spm_unit_id")] += 1
    elif defect == "native_id_type":
        row = evidence["native_crosswalk"]
        position = row["columns"].index("new_spm_unit_id")
        row["rows"][0][position] = float(row["rows"][0][position])
    elif defect == "registry_coverage":
        evidence["registry"]["entries"].pop()
    elif defect == "canonical_merge":
        for row in evidence["registry"]["entries"]:
            row["new_spm_unit_id"] = 100
        target = evidence["native_crosswalk"]
        for row in target["rows"]:
            row[target["columns"].index("new_spm_unit_id")] = 100
    elif defect == "unit_evidence":
        target = evidence["unit_evidence"]
        target["rows"][0][target["columns"].index("outside_acs_household_universe")] = (
            False
        )
    elif defect == "partition_count":
        set_cell(evidence, "crosswalk", 0, "person_count", 99)
    elif defect == "partner_option":
        evidence["implementation"] = {
            "options": {
                **evidence["implementation"]["options"],
                "minor_partner_role": True,
            }
        }
    elif defect == "scope_already_declared":
        evidence["annual_universe_declared"] = True
    elif defect == "missing_column":
        position = membership["columns"].index("old_spm_unit_id")
        for key in ("columns", "dtypes"):
            membership[key].pop(position)
        for row in membership["rows"]:
            row.pop(position)
    with pytest.raises(ACSSPMScopeError):
        classify_acs_spm_scope(evidence, profile())


@pytest.mark.parametrize(
    "defect", ["missing_person", "cross_household", "unknown_source"]
)
def test_links_cannot_reference_unknown_people_or_other_households(defect):
    members = [member(1, 10, "u1"), member(2, 20, "u2")]
    link = {
        "person_id": 1,
        "relative_id": 2,
        "kind": "parent",
        "source": "approved_inference",
        "rule_id": "invented_rule",
    }
    if defect == "missing_person":
        link["relative_id"] = 100
    elif defect == "unknown_source":
        members[1] = member(2, 10, "u1", SPORDER=2)
        link["source"] = "guess"
    code = {
        "missing_person": "LINK_PERSON",
        "cross_household": "LINK_HOUSEHOLD",
        "unknown_source": "LINK_SOURCE",
    }[defect]
    with pytest.raises(ACSSPMScopeError, match=code):
        classify_acs_spm_scope(fixture_evidence(members, [link]), profile())
