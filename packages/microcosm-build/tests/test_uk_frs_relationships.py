"""Tests for the #791 FRS relationship-grid stage and its manifest lockstep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.uk_runtime.frs_relationships import (
    CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS,
    FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES,
    FRS_RELATIONSHIPS_OUTPUT_COLUMNS,
    FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE,
    FRSRelationshipsError,
    UKFRSRelationshipsStageTransform,
    assert_frs_relationships_stage_parameters,
    derive_frs_relationships,
    frs_relationships_domains,
    frs_relationships_operation_parameters,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

REPO_ROOT = Path(__file__).resolve().parents[3]
UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
GRID = [f"r{index:02d}" for index in range(1, 15)]

SPOUSE, COHABITEE, CHILD, STEP_CHILD, FOSTER_CHILD = 1, 2, 3, 4, 5
CHILD_IN_LAW, PARENT, STEP_PARENT, FOSTER_PARENT, PARENT_IN_LAW = 6, 7, 8, 9, 10
SIBLING, GRANDCHILD, GRANDPARENT, NON_RELATIVE, CIVIL_PARTNER = 11, 15, 16, 18, 20


def _person(
    number: int,
    *,
    age: int,
    table: str = "adult",
    hrp: bool = False,
    fte: bool | None = None,
    rel: dict[int, int] | None = None,
) -> dict:
    return {
        "number": number,
        "age": age,
        "table": table,
        "hrp": hrp,
        "fte": fte,
        "rel": dict(rel or {}),
    }


def _tables(*households: list[dict]):
    """Raw (id-normalised) adult/child/househol tabs plus frame tables."""

    adult_rows, child_rows, hh_rows, person_rows = [], [], [], []
    for sernum, persons in enumerate(households, start=1):
        head = (
            next(p["number"] for p in persons if p["hrp"])
            if any(p["hrp"] for p in persons)
            else None
        )
        if head is not None:
            hh_rows.append({"household_id": sernum, "hrpnum": head})
        for p in persons:
            row = {
                "household_id": sernum,
                "person_id": sernum * 1000 + p["number"],
                "relhrp": (
                    np.nan
                    if p["hrp"] or head is None
                    else float(p["rel"].get(head, NON_RELATIVE))
                ),
                **{column: np.nan for column in GRID},
            }
            for other, code in p["rel"].items():
                row[f"r{other:02d}"] = float(code)
            if p["table"] == "adult":
                row["hrpid"] = 1.0 if p["hrp"] else 2.0
                row["educft"] = 1.0 if p["fte"] else 2.0
                adult_rows.append(row)
            else:
                row["educft"] = 1.0
                child_rows.append(row)
            person_rows.append(
                {
                    "person_id": sernum * 1000 + p["number"],
                    "person_benunit_id": sernum * 100 + 1,
                    "person_household_id": sernum,
                    "age": p["age"],
                    "is_household_head": bool(p["hrp"]),
                }
            )
    person = pd.DataFrame(person_rows).sort_values("person_id").reset_index(drop=True)
    household = pd.DataFrame({"household_id": list(range(1, len(households) + 1))})
    adult = (
        pd.DataFrame(adult_rows)
        if adult_rows
        else pd.DataFrame(
            columns=["household_id", "person_id", "relhrp", "hrpid", "educft", *GRID]
        )
    )
    child = (
        pd.DataFrame(child_rows)
        if child_rows
        else pd.DataFrame(
            columns=["household_id", "person_id", "relhrp", "educft", *GRID]
        )
    )
    return person, household, adult, child, pd.DataFrame(hh_rows)


def _derive(*households: list[dict], **kwargs):
    person, household, adult, child, raw_household = _tables(*households)
    return derive_frs_relationships(
        person,
        household,
        raw_adult=adult,
        raw_child=child,
        raw_household=raw_household,
        **kwargs,
    )


def _couple(a: int = 1, b: int = 2, code: int = SPOUSE) -> tuple[dict, dict]:
    return {b: code}, {a: code}


def _types(result) -> list[str]:
    return result.household_values["ons_household_type"].tolist()


def _roles(result) -> list[str]:
    return result.person_values["ons_family_role"].tolist()


# --- household types -------------------------------------------------------


def test_couple_plus_lodger_is_a_couple_household_with_an_individual() -> None:
    result = _derive(
        [
            _person(1, age=40, hrp=True, rel={2: SPOUSE, 3: NON_RELATIVE}),
            _person(2, age=39, rel={1: SPOUSE, 3: NON_RELATIVE}),
            _person(3, age=30, rel={1: NON_RELATIVE, 2: NON_RELATIVE}),
        ]
    )
    assert _types(result) == ["couple_no_children_households"]
    assert _roles(result) == ["COUPLE_PARTNER", "COUPLE_PARTNER", "INDIVIDUAL"]
    assert result.person_values["ons_family_index"].tolist() == [1, 1, 0]
    assert result.evidence["one_family_plus_individuals"] == 1
    assert result.person_values["relationship_to_head"].tolist() == [
        "HEAD",
        "SPOUSE",
        "OTHER_NON_RELATIVE",
    ]


def test_couple_with_an_adult_child_is_non_dependent_children_only() -> None:
    result = _derive(
        [
            _person(1, age=55, hrp=True, rel={2: COHABITEE, 3: PARENT}),
            _person(2, age=54, rel={1: COHABITEE, 3: PARENT}),
            _person(3, age=22, fte=True, rel={1: CHILD, 2: CHILD}),
        ]
    )
    assert _types(result) == ["couple_non_dependent_children_only_households"]
    assert _roles(result) == [
        "COUPLE_PARTNER",
        "COUPLE_PARTNER",
        "NON_DEPENDENT_CHILD",
    ]
    # 22 in full-time education is still non-dependent: ONS stops at 18.
    assert result.evidence["dependency_counts"]["19_plus"] == 1


def test_grandparent_couple_with_a_partnered_child_are_two_families() -> None:
    result = _derive(
        [
            _person(
                1,
                age=70,
                hrp=True,
                rel={2: SPOUSE, 3: PARENT, 4: PARENT_IN_LAW, 5: GRANDPARENT},
            ),
            _person(
                2, age=69, rel={1: SPOUSE, 3: PARENT, 4: PARENT_IN_LAW, 5: GRANDPARENT}
            ),
            _person(3, age=40, rel={1: CHILD, 2: CHILD, 4: SPOUSE, 5: PARENT}),
            _person(
                4, age=41, rel={1: CHILD_IN_LAW, 2: CHILD_IN_LAW, 3: SPOUSE, 5: PARENT}
            ),
            _person(
                5,
                age=5,
                table="child",
                rel={1: GRANDCHILD, 2: GRANDCHILD, 3: CHILD, 4: CHILD},
            ),
        ]
    )
    assert _types(result) == ["multi_family_households"]
    # The partnered child is not a child of the grandparents' family: two couples.
    assert _roles(result) == [
        "COUPLE_PARTNER",
        "COUPLE_PARTNER",
        "COUPLE_PARTNER",
        "COUPLE_PARTNER",
        "DEPENDENT_CHILD",
    ]
    assert result.person_values["ons_family_index"].tolist() == [1, 1, 2, 2, 2]


def test_couple_elderly_parent_and_grandchild_is_one_family_plus_individual() -> None:
    result = _derive(
        [
            _person(1, age=40, hrp=True, rel={2: SPOUSE, 3: CHILD, 4: PARENT}),
            _person(2, age=41, rel={1: SPOUSE, 3: CHILD_IN_LAW, 4: PARENT}),
            _person(3, age=70, rel={1: PARENT, 2: PARENT_IN_LAW, 4: GRANDPARENT}),
            _person(4, age=5, table="child", rel={1: CHILD, 2: CHILD, 3: GRANDCHILD}),
        ]
    )
    # ONS: "a married couple with one elderly parent" is a one-family household.
    assert _types(result) == ["couple_under_3_children_households"]
    assert _roles(result) == [
        "COUPLE_PARTNER",
        "COUPLE_PARTNER",
        "INDIVIDUAL",
        "DEPENDENT_CHILD",
    ]


def test_lone_parent_with_a_19_year_old_in_education_is_non_dependent_only() -> None:
    result = _derive(
        [
            _person(1, age=45, hrp=True, rel={2: PARENT}),
            _person(2, age=19, table="child", rel={1: CHILD}),
        ]
    )
    assert _types(result) == ["lone_parent_non_dependent_children_households"]
    assert _roles(result) == ["LONE_PARENT", "NON_DEPENDENT_CHILD"]


def test_sixteen_to_eighteen_dependency_follows_full_time_education() -> None:
    not_in_education = _derive(
        [
            _person(1, age=45, hrp=True, rel={2: PARENT}),
            _person(2, age=17, fte=False, rel={1: CHILD}),
        ]
    )
    in_education = _derive(
        [
            _person(1, age=45, hrp=True, rel={2: PARENT}),
            _person(2, age=17, fte=True, rel={1: CHILD}),
        ]
    )
    assert _types(not_in_education) == ["lone_parent_non_dependent_children_households"]
    assert _types(in_education) == ["lone_parent_dependent_children_households"]
    assert in_education.evidence["dependency_counts"]["16_18_fte"] == 1
    assert not_in_education.evidence["dependency_counts"]["16_18_not_fte"] == 1


def test_households_with_no_family_are_unrelated_adults() -> None:
    sharers = _derive(
        [
            _person(1, age=30, hrp=True, rel={2: NON_RELATIVE}),
            _person(2, age=31, rel={1: NON_RELATIVE}),
        ]
    )
    siblings = _derive(
        [
            _person(1, age=60, hrp=True, rel={2: SIBLING}),
            _person(2, age=58, rel={1: SIBLING}),
        ]
    )
    foster = _derive(
        [
            _person(1, age=50, hrp=True, rel={2: FOSTER_PARENT}),
            _person(2, age=10, table="child", rel={1: FOSTER_CHILD}),
        ]
    )
    for result in (sharers, siblings, foster):
        assert _types(result) == ["unrelated_adult_households"]
        assert _roles(result) == ["INDIVIDUAL", "INDIVIDUAL"]


def test_grandparent_with_a_lone_parent_daughter_is_one_lone_parent_family() -> None:
    result = _derive(
        [
            _person(1, age=70, hrp=True, rel={2: PARENT, 3: GRANDPARENT}),
            _person(2, age=35, rel={1: CHILD, 3: PARENT}),
            _person(3, age=8, table="child", rel={1: GRANDCHILD, 2: CHILD}),
        ]
    )
    assert _types(result) == ["lone_parent_dependent_children_households"]
    # The daughter has her own child, so she is not the grandparent's child.
    assert _roles(result) == ["INDIVIDUAL", "LONE_PARENT", "DEPENDENT_CHILD"]


def test_grandparent_with_an_unpartnered_childless_adult_child_is_a_lone_parent() -> (
    None
):
    result = _derive(
        [
            _person(1, age=70, hrp=True, rel={2: PARENT}),
            _person(2, age=35, rel={1: CHILD}),
        ]
    )
    assert _types(result) == ["lone_parent_non_dependent_children_households"]
    assert _roles(result) == ["LONE_PARENT", "NON_DEPENDENT_CHILD"]


def test_two_lone_parents_who_are_not_a_couple_are_multi_family() -> None:
    result = _derive(
        [
            _person(
                1, age=40, hrp=True, rel={2: NON_RELATIVE, 3: PARENT, 4: NON_RELATIVE}
            ),
            _person(2, age=38, rel={1: NON_RELATIVE, 3: NON_RELATIVE, 4: PARENT}),
            _person(
                3,
                age=6,
                table="child",
                rel={1: CHILD, 2: NON_RELATIVE, 4: NON_RELATIVE},
            ),
            _person(
                4,
                age=7,
                table="child",
                rel={1: NON_RELATIVE, 2: CHILD, 3: NON_RELATIVE},
            ),
        ]
    )
    assert _types(result) == ["multi_family_households"]
    assert _roles(result) == [
        "LONE_PARENT",
        "LONE_PARENT",
        "DEPENDENT_CHILD",
        "DEPENDENT_CHILD",
    ]
    assert result.person_values["ons_family_index"].tolist() == [1, 2, 1, 2]


def test_one_person_households_split_at_sixty_five() -> None:
    result = _derive(
        [_person(1, age=64, hrp=True)],
        [_person(1, age=65, hrp=True)],
    )
    assert _types(result) == ["lone_households_under_65", "lone_households_over_65"]
    assert _roles(result) == ["INDIVIDUAL", "INDIVIDUAL"]


def test_couples_split_on_the_count_of_dependent_children() -> None:
    def couple_with(children: int, ages: list[int]) -> list[dict]:
        rel1 = {2: SPOUSE, **{3 + i: PARENT for i in range(children)}}
        rel2 = {1: SPOUSE, **{3 + i: PARENT for i in range(children)}}
        kids = [
            _person(3 + i, age=ages[i], table="child", rel={1: CHILD, 2: CHILD})
            for i in range(children)
        ]
        return [
            _person(1, age=40, hrp=True, rel=rel1),
            _person(2, age=39, rel=rel2),
            *kids,
        ]

    result = _derive(couple_with(2, [3, 9]), couple_with(3, [3, 9, 12]))
    assert _types(result) == [
        "couple_under_3_children_households",
        "couple_3_plus_children_households",
    ]
    assert result.evidence["partition_closes"] is True
    assert sum(result.evidence["household_type_counts"].values()) == 2


def test_civil_partners_and_step_children_form_one_family() -> None:
    result = _derive(
        [
            _person(1, age=45, hrp=True, rel={2: CIVIL_PARTNER, 3: STEP_PARENT}),
            _person(2, age=44, rel={1: CIVIL_PARTNER, 3: PARENT}),
            _person(3, age=12, table="child", rel={1: STEP_CHILD, 2: CHILD}),
        ]
    )
    assert _types(result) == ["couple_under_3_children_households"]
    assert result.person_values["relationship_to_head"].tolist() == [
        "HEAD",
        "CIVIL_PARTNER",
        "STEP_CHILD",
    ]


def test_weighted_cells_and_domains_in_evidence() -> None:
    person, household, adult, child, raw_household = _tables(
        [_person(1, age=30, hrp=True)],
        [_person(1, age=80, hrp=True)],
    )
    result = derive_frs_relationships(
        person,
        household,
        raw_adult=adult,
        raw_child=child,
        raw_household=raw_household,
        household_weights=np.array([2.5, 4.0]),
    )
    weighted = result.evidence["household_type_weighted"]
    assert weighted["lone_households_under_65"] == 2.5
    assert weighted["lone_households_over_65"] == 4.0
    assert set(weighted) == set(CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS)
    assert result.evidence["stage"] == "frs_relationships"
    assert result.evidence["domain_violations"] == 0
    assert result.evidence["hrpnum_mismatches"] == 0


# --- fail-closed -----------------------------------------------------------


def test_two_household_reference_persons_refuse() -> None:
    with pytest.raises(FRSRelationshipsError, match="head_invariant_violations=1"):
        _derive(
            [
                _person(1, age=30, hrp=True, rel={2: NON_RELATIVE}),
                _person(2, age=31, hrp=True, rel={1: NON_RELATIVE}),
            ]
        )


def test_hrpnum_disagreeing_with_hrpid_refuses() -> None:
    person, household, adult, child, raw_household = _tables(
        [
            _person(1, age=30, hrp=True, rel={2: NON_RELATIVE}),
            _person(2, age=31, rel={1: NON_RELATIVE}),
        ]
    )
    raw_household["hrpnum"] = 2
    with pytest.raises(FRSRelationshipsError, match="hrpnum_mismatches=1"):
        derive_frs_relationships(
            person,
            household,
            raw_adult=adult,
            raw_child=child,
            raw_household=raw_household,
        )


def test_unmapped_relhrp_code_refuses() -> None:
    person, household, adult, child, raw_household = _tables(
        [
            _person(1, age=30, hrp=True, rel={2: NON_RELATIVE}),
            _person(2, age=31, rel={1: NON_RELATIVE}),
        ]
    )
    adult.loc[adult["person_id"] == 1002, "relhrp"] = 99
    with pytest.raises(FRSRelationshipsError, match="relhrp_unmapped_codes"):
        derive_frs_relationships(
            person,
            household,
            raw_adult=adult,
            raw_child=child,
            raw_household=raw_household,
        )


def test_a_person_with_two_partners_refuses() -> None:
    with pytest.raises(FRSRelationshipsError, match="multi_partner_persons=1"):
        _derive(
            [
                _person(1, age=30, hrp=True, rel={2: SPOUSE, 3: COHABITEE}),
                _person(2, age=31, rel={1: SPOUSE, 3: NON_RELATIVE}),
                _person(3, age=29, rel={1: COHABITEE, 2: NON_RELATIVE}),
            ]
        )


def test_non_contiguous_person_numbers_refuse() -> None:
    with pytest.raises(FRSRelationshipsError, match="person_index_gaps=1"):
        _derive(
            [
                _person(1, age=30, hrp=True, rel={3: NON_RELATIVE}),
                _person(3, age=31, rel={1: NON_RELATIVE}),
            ]
        )


def test_reciprocity_mismatches_are_counted_and_fenced() -> None:
    # The child records a parent link; the parent records a non-relative.
    household = [
        _person(1, age=45, hrp=True, rel={2: NON_RELATIVE}),
        _person(2, age=10, table="child", rel={1: CHILD}),
    ]
    with pytest.raises(FRSRelationshipsError, match="grid_reciprocity_mismatches=1"):
        _derive(household, reciprocity_mismatch_tolerance=0)
    tolerated = _derive(household, reciprocity_mismatch_tolerance=1)
    assert tolerated.evidence["grid_reciprocity_mismatches"] == 1
    # Either side's declaration establishes the link.
    assert _types(tolerated) == ["lone_parent_dependent_children_households"]


def test_a_non_numeric_age_refuses_instead_of_reading_as_zero() -> None:
    person, household, adult, child, raw_household = _tables(
        [
            _person(1, age=45, hrp=True, rel={2: PARENT}),
            _person(2, age=10, table="child", rel={1: CHILD}),
        ]
    )
    person.loc[person["person_id"] == 1002, "age"] = np.nan
    with pytest.raises(FRSRelationshipsError, match="non-numeric age"):
        derive_frs_relationships(
            person,
            household,
            raw_adult=adult,
            raw_child=child,
            raw_household=raw_household,
        )


def test_a_frame_person_missing_from_the_tabs_refuses() -> None:
    person, household, adult, child, raw_household = _tables(
        [_person(1, age=30, hrp=True)]
    )
    person = pd.concat(
        [
            person,
            pd.DataFrame(
                {
                    "person_id": [1002],
                    "person_benunit_id": [101],
                    "person_household_id": [1],
                    "age": [20],
                    "is_household_head": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(FRSRelationshipsError, match="absent from the FRS"):
        derive_frs_relationships(
            person,
            household,
            raw_adult=adult,
            raw_child=child,
            raw_household=raw_household,
        )


# --- manifest lockstep and declared domains ----------------------------------


def _committed_stage() -> SourceStageSpec:
    manifest = SourceManifest.from_mapping(
        json.loads((UK_PACKAGE / "source_stages.json").read_text(encoding="utf-8"))
    )
    return next(
        stage for stage in manifest.stages if stage.stage == "frs_relationships"
    )


def _stage_mapping() -> dict:
    payload = json.loads(
        (UK_PACKAGE / "source_stages.json").read_text(encoding="utf-8")
    )
    return next(
        stage for stage in payload["stages"] if stage["stage"] == "frs_relationships"
    )


def test_committed_manifest_matches_the_runtime_parameters() -> None:
    stage = _committed_stage()
    assert_frs_relationships_stage_parameters(stage)
    assert stage.outputs == FRS_RELATIONSHIPS_OUTPUT_COLUMNS
    declared = dict(stage.operations[1].parameters)
    assert declared["household_type_values"] == list(
        CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS
    )
    assert declared["reciprocity_mismatch_tolerance"] == (
        FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE
    )
    assert {
        item["code"]: item["label"]
        for item in declared["frs_household_grid_relationship_codes"]
    } == dict(FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda op: op.__setitem__("reciprocity_mismatch_tolerance", 99),
            "reciprocity_mismatch_tolerance",
        ),
        (
            lambda op: op["family_link_codes"].__setitem__("partner", [1, 2]),
            "family_link_codes",
        ),
        (
            lambda op: op.__setitem__(
                "household_type_values", op["household_type_values"][::-1]
            ),
            "household_type_values",
        ),
        (lambda op: op.__setitem__("unreviewed", 1), "unreviewed"),
    ],
)
def test_manifest_drift_refuses_with_the_key_named(mutate, message) -> None:
    mapping = _stage_mapping()
    mutate(mapping["operations"][1])
    with pytest.raises(ValueError, match=message):
        assert_frs_relationships_stage_parameters(SourceStageSpec.from_mapping(mapping))


def test_manifest_operation_order_is_enforced() -> None:
    mapping = _stage_mapping()
    mapping["operations"] = mapping["operations"][::-1]
    with pytest.raises(ValueError, match="read_tables followed by"):
        assert_frs_relationships_stage_parameters(SourceStageSpec.from_mapping(mapping))


def test_raw_tab_pins_match_the_spine_artifacts() -> None:
    manifest = SourceManifest.from_mapping(
        json.loads((UK_PACKAGE / "source_stages.json").read_text(encoding="utf-8"))
    )
    stages = {stage.stage: stage for stage in manifest.stages}
    spine = {
        artifact["table"]: (
            artifact["locator"],
            artifact["sha256"],
            artifact["size_bytes"],
        )
        for artifact in stages["frs_spine"].artifacts
    }
    for artifact in stages["frs_relationships"].artifacts:
        assert spine[artifact["table"]] == (
            artifact["locator"],
            artifact["sha256"],
            artifact["size_bytes"],
        )
    assert {
        artifact["table"] for artifact in stages["frs_relationships"].artifacts
    } == {
        "adult",
        "child",
        "househol",
    }
    names = [stage.stage for stage in manifest.stages]
    assert names.index("frs_relationships") == names.index("age_tail") + 1


def test_household_type_values_are_the_ten_contract_cells_in_order() -> None:
    contract = json.loads(
        (UK_PACKAGE / "uk_population_targets.json").read_text(encoding="utf-8")
    )
    rows = [
        target
        for target in contract["targets"]
        if target["family"] == "ons_household_composition"
    ]
    dimension_values = [
        target["ledger_selector"]["dimension_values"]["household_type"]
        for target in rows
    ]
    assert dimension_values == list(CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS)
    assert frs_relationships_domains()["ons_household_type"] == (
        CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS
    )
    assert len(set(CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS)) == 10


def test_operation_parameters_carry_their_source_documentation() -> None:
    parameters = frs_relationships_operation_parameters()
    assert "SN 9563" in parameters["documentation"]["frs_codes"]
    assert "Families and households" in parameters["documentation"]["ons_definitions"]
    assert "ons-families-households-2025" in parameters["documentation"]["category_ids"]
    assert parameters["head_label"] == "HEAD"
    assert parameters["source_columns"]["grid"] == GRID


def test_gates_declare_the_stage_health_and_enum_gates_and_export_columns() -> None:
    gates = json.loads((UK_PACKAGE / "gates.json").read_text(encoding="utf-8"))
    by_id = {gate["id"]: gate for gate in gates["gates"]}
    health = by_id["uk_stage_frs_relationships_composition"]
    assert (health["gate"], health["phase"]) == ("stage_health", "assembled")
    assert health["parameters"]["check"] == "household_composition"
    assert health["parameters"]["max_grid_reciprocity_mismatches"] == (
        FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE
    )
    enum = by_id["uk_ons_household_type_enum_domain"]
    assert (enum["gate"], enum["phase"]) == ("enum_domain", "assembled")
    assert enum["parameters"]["columns"] == ["ons_household_type"]
    exported = set(by_id["uk_export_surface"]["parameters"]["allowed_extra_columns"])
    assert {
        "person.relationship_to_head",
        "person.ons_family_role",
        "person.ons_family_index",
        "household.ons_household_type",
    } <= exported


# --- stage health -----------------------------------------------------------


def _health_parameters() -> dict:
    return {
        "stage": "frs_relationships",
        "check": "household_composition",
        "parameters": {
            "stage": "frs_relationships",
            "check": "household_composition",
            "max_grid_reciprocity_mismatches": (
                FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE
            ),
            "require_partition_closure": True,
        },
    }


def test_stage_health_passes_on_a_clean_receipt() -> None:
    result = _derive([_person(1, age=30, hrp=True)])
    gate = uk_stage_health_gate(evidence=result.evidence, **_health_parameters())
    assert gate.passed, gate.failures
    assert gate.details["households"] == 1


def test_stage_health_refuses_a_partition_gap_and_excess_mismatches() -> None:
    result = _derive([_person(1, age=30, hrp=True)])
    evidence = dict(result.evidence)
    evidence["households"] = 2
    gate = uk_stage_health_gate(evidence=evidence, **_health_parameters())
    assert not gate.passed
    assert any("partition covers 1 of 2" in failure for failure in gate.failures)
    evidence = dict(result.evidence)
    evidence["grid_reciprocity_mismatches"] = (
        FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE + 1
    )
    gate = uk_stage_health_gate(evidence=evidence, **_health_parameters())
    assert not gate.passed
    assert any("exceeds the reviewed tolerance" in failure for failure in gate.failures)


# --- the transform end to end ------------------------------------------------


def _write_tab(path: Path, rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, sep="\t", index=False)
    return {
        "locator": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def test_transform_writes_the_four_columns_and_receipts(tmp_path: Path) -> None:
    grid_blank = {f"R{index:02d}": "" for index in range(1, 15)}
    adult = _write_tab(
        tmp_path / "adult.tab",
        [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 1,
                "HRPID": 1,
                "EDUCFT": 2,
                "RELHRP": "",
                **grid_blank,
                "R02": PARENT,
            },
            {
                "SERNUM": 2,
                "BENUNIT": 1,
                "PERSON": 1,
                "HRPID": 1,
                "EDUCFT": 2,
                "RELHRP": "",
                **grid_blank,
                "R02": SPOUSE,
            },
            {
                "SERNUM": 2,
                "BENUNIT": 1,
                "PERSON": 2,
                "HRPID": 2,
                "EDUCFT": 2,
                "RELHRP": SPOUSE,
                **grid_blank,
                "R01": SPOUSE,
            },
        ],
    )
    child = _write_tab(
        tmp_path / "child.tab",
        [
            {
                "SERNUM": 1,
                "BENUNIT": 1,
                "PERSON": 2,
                "EDUCFT": 1,
                "RELHRP": CHILD,
                **grid_blank,
                "R01": CHILD,
            }
        ],
    )
    househol = _write_tab(
        tmp_path / "househol.tab",
        [{"SERNUM": 1, "HRPNUM": 1}, {"SERNUM": 2, "HRPNUM": 1}],
    )
    mapping = _stage_mapping()
    pins = {"adult": adult, "child": child, "househol": househol}
    for artifact in mapping["artifacts"]:
        artifact.update(pins[artifact["table"]])
    stage = SourceStageSpec.from_mapping(mapping)

    person = pd.DataFrame(
        {
            "person_id": [1001, 1002, 2001, 2002],
            "person_benunit_id": [101, 101, 201, 201],
            "person_household_id": [1, 1, 2, 2],
            "age": [40, 9, 66, 64],
            "gender": ["FEMALE", "MALE", "MALE", "FEMALE"],
            "is_household_head": [True, False, True, False],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [101, 201]})
    household = pd.DataFrame({"household_id": [1, 2], "household_weight": [3.0, 5.0]})
    frame = uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2024"
    )

    transform = UKFRSRelationshipsStageTransform(tmp_path, stage=stage)
    assert transform.output_columns() == FRS_RELATIONSHIPS_OUTPUT_COLUMNS
    returned = transform(frame)
    assert returned.table("person")["relationship_to_head"].tolist() == [
        "HEAD",
        "CHILD",
        "HEAD",
        "SPOUSE",
    ]
    assert returned.table("person")["ons_family_role"].tolist() == [
        "LONE_PARENT",
        "DEPENDENT_CHILD",
        "COUPLE_PARTNER",
        "COUPLE_PARTNER",
    ]
    assert returned.table("person")["ons_family_index"].tolist() == [1, 1, 1, 1]
    assert returned.table("household")["ons_household_type"].tolist() == [
        "lone_parent_dependent_children_households",
        "couple_no_children_households",
    ]
    assert list(returned.weights_for("household").values) == [3.0, 5.0]
    metadata = transform.checkpoint_metadata()
    assert metadata["evidence"]["stage"] == "frs_relationships"
    assert metadata["evidence"]["household_type_weighted"] == {
        **{value: 0.0 for value in CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS},
        "lone_parent_dependent_children_households": 3.0,
        "couple_no_children_households": 5.0,
    }
    assert metadata["evidence"]["full_time_education_source"] == "educft"
