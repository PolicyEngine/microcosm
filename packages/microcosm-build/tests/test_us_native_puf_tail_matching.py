"""Invented selection/matching plus the real structural EXPAND, without fitting."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import test_us_native_puf_tail as donor_fixture
import test_us_native_puf_tail_expand as host_fixture

from microcosm.build.us_runtime import native_puf_tail_matching as matching
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import ContentStore


def inputs(*, amounts=(6_000_000.0,), donor_ids=None, statuses=None, weights=None):
    returns, people = donor_fixture.fixture_returns(
        amounts, ids=donor_ids, statuses=statuses, weights=weights
    )
    host = host_fixture.invented_frame()
    frame = Frame(
        {entity: host.table(entity).copy() for entity in US_SCHEMA.entities},
        US_SCHEMA,
        {
            "household": Weights(
                np.array([8.0, 40.0, 0.0, 6.0] * 2), WeightKind.IMPORTANCE
            )
        },
        host.strata.copy(),
    )
    persons = tuple(
        matching.InventedRecipientPerson(
            person_id,
            "dependent" if person_id == 102 else "head",
            16 if person_id == 102 else 40,
            tuple(
                (
                    name,
                    amount if name == matching.RECIPIENT_PROXY_COMPONENTS[0] else 0.0,
                )
                for name in matching.RECIPIENT_PROXY_COMPONENTS
            ),
        )
        for person_id, amount in (
            (101, 40_000.0),
            (102, 0.0),
            (103, 6_000_000.0),
            (104, 200_000.0),
            (105, 3_000_000.0),
        )
    )
    return dict(
        projection=donor_fixture.declare(returns, people),
        capital_gains_mask=np.zeros(len(returns), dtype=bool),
        source_eligible=np.ones(len(returns), dtype=bool),
        donor_source_agi=tuple(
            matching.InventedDonorSourceAgi(int(i), 40_000.0)
            for i in returns.donor_recid
        ),
        donor_basis=matching.InventedIncomeBasis(2024, 2024),
        recipient_basis=matching.InventedIncomeBasis(2024, 2024),
        frame=frame,
        domains=host_fixture.domain_rows(),
        persons=persons,
        tax_units=tuple(
            matching.InventedRecipientTaxUnit(i, 1) for i in (110, 120, 130, 140)
        ),
        seed=578,
    )


def change_person(arguments, person_id, **changes):
    arguments["persons"] = tuple(
        replace(p, **changes) if p.person_id == person_id else p
        for p in arguments["persons"]
    )


def change_proxy(arguments, person_id, value):
    change_person(
        arguments,
        person_id,
        proxy_components=tuple(
            (name, value if name == matching.RECIPIENT_PROXY_COMPONENTS[0] else 0.0)
            for name in matching.RECIPIENT_PROXY_COMPONENTS
        ),
    )


def change_weight(arguments, household_id, value):
    before = arguments["frame"]
    weights = before.weights_for("household").values.copy()
    weights[
        np.flatnonzero(
            before.table("household").household_id.to_numpy() == household_id
        )[0]
    ] = value
    arguments["frame"] = Frame(
        {e: before.table(e).copy() for e in before.entities},
        before.schema,
        {"household": Weights(weights, WeightKind.IMPORTANCE)},
        before.strata.copy(),
    )


def test_source_agi_matching_is_distinct_from_selection_proxy_and_weight_units():
    arguments = inputs(weights=[12_345.0])
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.selection.donors[0].proxy_agi == 6_000_000.0
    assert [
        (p.donor_id, p.parent_household_id, p.donor_support_weight)
        for p in result.assignments
    ] == [(1, 110, 12_345.0)]
    assert result.roles == (matching.MatchedRoleCoordinates(1, 110, 110, 101, None),)
    report = json.loads(result.receipt)
    assert report["candidates"][0]["household_importance"] == 8.0
    assert report["donor_source_agi"] == [[1, 40_000.0]]
    assert (
        report["selection_sha256"]
        == hashlib.sha256(result.selection.receipt).hexdigest()
    )
    assert report["expansion_sha256"] == result.expansion.sha256
    assert result.sha256 == hashlib.sha256(result.receipt).hexdigest()
    assert report["reference_commit"] == matching.REFERENCE_COMMIT
    assert (
        not result.source_admission_issued
        and not result.matching_qualified
        and not result.release_eligible
    )
    assert not report["money_placement"] and not report["geography_assigned"]
    assert report["exclusions"] == [
        {"household_id": 130, "reason": "zero_household_importance"},
        {"household_id": 140, "reason": "group_quarters"},
    ]
    # Changing donor support magnitude alone cannot turn it into host mass.
    other = matching.match_invented_native_puf_tail(**inputs(weights=[1e100]))
    assert other.roles == result.roles
    assert [r.parent_household_id for r in other.assignments] == [110]


def test_same_status_nearest_band_and_lower_band_tie():
    arguments = inputs()
    change_proxy(arguments, 101, 7_500.0)  # band 1
    change_proxy(arguments, 103, 17_500.0)  # band 3
    arguments["donor_source_agi"] = (
        matching.InventedDonorSourceAgi(1, 12_500.0),
    )  # band 2
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.assignments[0].parent_household_id == 110
    # Exact lower bound belongs to the upper half-open band.
    arguments["donor_source_agi"] = (matching.InventedDonorSourceAgi(1, 15_000.0),)
    assert (
        matching.match_invented_native_puf_tail(**arguments)
        .assignments[0]
        .parent_household_id
        == 120
    )
    arguments["tax_units"] = tuple(
        replace(r, filing_status_code=2) if r.tax_unit_id == 120 else r
        for r in arguments["tax_units"]
    )
    assert (
        matching.match_invented_native_puf_tail(**arguments)
        .assignments[0]
        .parent_household_id
        == 110
    )


def test_spouse_demand_first_preserves_unique_capacity_for_head_only_donor():
    arguments = inputs(amounts=(6_000_000.0, 7_000_000.0), donor_ids=[1, 2])
    returns, people = donor_fixture.fixture_returns(
        [6_000_000.0, 7_000_000.0], ids=[1, 2]
    )
    people.loc[people.donor_recid == 2, "employment_income_before_lsr"] = 6_000_000.0
    people = donor_fixture.add_person(
        people,
        person_id=21,
        donor_recid=2,
        role="spouse",
        employment_income_before_lsr=1_000_000.0,
    )
    arguments["projection"] = donor_fixture.declare(returns, people)
    change_person(arguments, 102, role="spouse", age=38)
    result = matching.match_invented_native_puf_tail(**arguments)
    assert [(p.donor_id, p.parent_household_id) for p in result.assignments] == [
        (1, 120),
        (2, 110),
    ]
    assert result.roles[1].spouse_person_id == 102
    assert len({p.parent_household_id for p in result.assignments}) == 2
    assert json.loads(result.receipt)["support"][0]["spouse_demand"] == 1


@pytest.mark.parametrize("shortage", ["count", "spouse"])
def test_capacity_shortage_never_partially_assigns_or_reuses_parents(shortage):
    arguments = inputs(amounts=(6_000_000.0,) * (3 if shortage == "count" else 1))
    if shortage == "spouse":
        returns, people = donor_fixture.fixture_returns([6_000_000.0])
        people["employment_income_before_lsr"] = 5_000_000.0
        people = donor_fixture.add_person(
            people,
            person_id=11,
            donor_recid=1,
            role="spouse",
            employment_income_before_lsr=1_000_000.0,
        )
        arguments["projection"] = donor_fixture.declare(returns, people)
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.assignments == result.roles == () and result.expansion is None
    assert (
        json.loads(result.receipt)["support"][0]["status"]
        == "insufficient_unique_support"
    )


def test_status_shortage_does_not_drop_an_independent_supported_status():
    arguments = inputs(amounts=(6e6, 6e6, 6e6), statuses=[1, 1, 2])
    arguments["tax_units"] = tuple(
        replace(r, filing_status_code=2) if r.tax_unit_id == 120 else r
        for r in arguments["tax_units"]
    )
    result = matching.match_invented_native_puf_tail(**arguments)
    assert [(p.donor_id, p.parent_household_id) for p in result.assignments] == [
        (3, 120)
    ]
    assert [r["status"] for r in json.loads(result.receipt)["support"]] == [
        "insufficient_unique_support",
        "assigned",
    ]


def test_input_reordering_preserves_source_order_random_matching_and_coordinates():
    arguments = inputs(amounts=(6e6, 6e6), donor_ids=[9, 2])
    change_proxy(arguments, 103, 40_000.0)
    before = matching.match_invented_native_puf_tail(**arguments)
    frame = arguments["frame"]
    # Person rows may be permuted; group IDs and their household weights must
    # retain the canonical sorted axes required by Frame. Strata are per person.
    arguments["frame"] = Frame(
        {
            e: frame.table(e).iloc[::-1].reset_index(drop=True)
            if e == "person"
            else frame.table(e).copy()
            for e in frame.entities
        },
        frame.schema,
        {
            "household": Weights(
                frame.weights_for("household").values.copy(), WeightKind.IMPORTANCE
            )
        },
        frame.strata.iloc[::-1].reset_index(drop=True),
    )
    for name in ("domains", "persons", "tax_units", "donor_source_agi"):
        arguments[name] = arguments[name][::-1]
    returns, people = donor_fixture.fixture_returns([6e6, 6e6], ids=[2, 9])
    arguments["projection"] = donor_fixture.declare(returns, people)
    after = matching.match_invented_native_puf_tail(**arguments)
    assert before.assignments == after.assignments and before.roles == after.roles
    left, right = json.loads(before.receipt), json.loads(after.receipt)
    for name in (
        "candidates",
        "exclusions",
        "frame_input_sha256",
        "recipient_person_sha256",
        "recipient_tax_unit_sha256",
        "expansion_sha256",
    ):
        assert left[name] == right[name]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing_person", "PERSON_COVERAGE"),
        ("extra_person", "PERSON_COVERAGE"),
        ("duplicate_person", "DUPLICATE_PERSON"),
        ("unknown_role", "ROLE_KNOWN"),
        ("boolean_age", "AGE_KNOWN"),
        ("missing_proxy", "PROXY_KNOWN"),
        ("boolean_proxy", "PROXY_KNOWN"),
        ("nan_proxy", "PROXY_KNOWN"),
        ("huge_proxy", "PROXY_KNOWN"),
        ("missing_tax_unit", "TAX_UNIT_COVERAGE"),
        ("duplicate_tax_unit", "DUPLICATE_TAX_UNIT"),
        ("boolean_status", "FILING_STATUS_KNOWN"),
        ("unknown_status", "FILING_STATUS_KNOWN"),
        ("missing_domain", "DOMAIN_COVERAGE"),
        ("contradictory_domain", "DOMAIN_ORIGIN"),
    ],
)
def test_required_recipient_evidence_is_never_inferred_or_zero_filled(case, reason):
    arguments = inputs()
    if case == "missing_person":
        arguments["persons"] = arguments["persons"][1:]
    elif case == "extra_person":
        arguments["persons"] += (replace(arguments["persons"][0], person_id=999),)
    elif case == "duplicate_person":
        arguments["persons"] += (arguments["persons"][0],)
    elif case == "unknown_role":
        change_person(arguments, 101, role=None)
    elif case == "boolean_age":
        change_person(arguments, 101, age=True)
    elif case == "missing_proxy":
        change_person(arguments, 101, proxy_components=())
    elif case in ("boolean_proxy", "nan_proxy", "huge_proxy"):
        change_proxy(
            arguments,
            101,
            {"boolean_proxy": True, "nan_proxy": float("nan"), "huge_proxy": 10**1000}[
                case
            ],
        )
    elif case == "missing_tax_unit":
        arguments["tax_units"] = arguments["tax_units"][1:]
    elif case == "duplicate_tax_unit":
        arguments["tax_units"] += (arguments["tax_units"][0],)
    elif case in ("boolean_status", "unknown_status"):
        arguments["tax_units"] = (
            replace(
                arguments["tax_units"][0],
                filing_status_code=True if case == "boolean_status" else 99,
            ),
            *arguments["tax_units"][1:],
        )
    elif case == "missing_domain":
        arguments["domains"] = arguments["domains"][1:]
    else:
        arguments["domains"] = (
            replace(arguments["domains"][0], support_source_id=999),
            *arguments["domains"][1:],
        )
    with pytest.raises(ValueError, match="NATIVE_TAIL_MATCH_" + reason):
        matching.match_invented_native_puf_tail(**arguments)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "DONOR_AGI_COVERAGE"),
        ("extra", "DONOR_AGI_COVERAGE"),
        ("duplicate", "DONOR_AGI_DUPLICATE"),
        ("boolean_id", "DONOR_AGI_IDS"),
        ("boolean_value", "DONOR_AGI_KNOWN"),
        ("nan", "DONOR_AGI_KNOWN"),
        ("missing_value", "DONOR_AGI_KNOWN"),
        ("price_year", "COMPARISON_BASIS"),
        ("income_year", "COMPARISON_BASIS"),
        ("currency", "BASIS_VALUES"),
        ("period", "BASIS_VALUES"),
        ("boolean_seed", "SEED"),
    ],
)
def test_source_agi_and_comparison_basis_are_explicit(case, reason):
    arguments = inputs()
    row = arguments["donor_source_agi"][0]
    if case == "missing":
        arguments["donor_source_agi"] = ()
    elif case == "extra":
        arguments["donor_source_agi"] += (replace(row, donor_recid=999),)
    elif case == "duplicate":
        arguments["donor_source_agi"] += (row,)
    elif case == "boolean_id":
        arguments["donor_source_agi"] = (replace(row, donor_recid=True),)
    elif case in ("boolean_value", "nan", "missing_value"):
        arguments["donor_source_agi"] = (
            replace(
                row,
                source_adjusted_gross_income={
                    "boolean_value": True,
                    "nan": float("nan"),
                    "missing_value": None,
                }[case],
            ),
        )
    elif case == "boolean_seed":
        arguments["seed"] = True
    else:
        arguments["recipient_basis"] = replace(
            arguments["recipient_basis"],
            **{
                case: {
                    "price_year": 2023,
                    "income_year": 2023,
                    "currency": "GBP",
                    "period": "monthly",
                }[case]
            },
        )
    with pytest.raises(ValueError, match="NATIVE_TAIL_MATCH_" + reason):
        matching.match_invented_native_puf_tail(**arguments)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("underage", "underage_head_or_spouse"),
        ("two_heads", "incompatible_role_structure"),
        ("no_head", "incompatible_role_structure"),
        ("subnormal", "unrepresentable_equal_halves"),
        ("odd_subnormal", "unrepresentable_equal_halves"),
    ],
)
def test_known_incompatible_support_is_excluded_with_reason(case, reason):
    arguments = inputs()
    if case == "underage":
        change_person(arguments, 101, age=14)
    elif case == "two_heads":
        change_person(arguments, 102, role="head")
    elif case == "no_head":
        change_person(arguments, 101, role="dependent")
    else:
        change_weight(
            arguments, 110, np.nextafter(0.0, 1.0) * (1 if case == "subnormal" else 3)
        )
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.assignments[0].parent_household_id == 120
    assert {"household_id": 110, "reason": reason} in json.loads(result.receipt)[
        "exclusions"
    ]


def test_even_subnormal_has_representable_equal_halves():
    arguments = inputs()
    change_weight(arguments, 110, np.nextafter(0.0, 1.0) * 2)
    assert (
        matching.match_invented_native_puf_tail(**arguments)
        .assignments[0]
        .parent_household_id
        == 110
    )


@pytest.mark.parametrize("declare_extra_unit", [False, True])
def test_multiple_tax_units_require_complete_roster_then_exclude_parent(
    declare_extra_unit,
):
    arguments = inputs()
    altered = host_fixture.invented_frame("multiple_tax_units")
    arguments["frame"] = Frame(
        {entity: altered.table(entity).copy() for entity in altered.entities},
        altered.schema,
        {"household": arguments["frame"].weights_for("household")},
        altered.strata.copy(),
    )
    if not declare_extra_unit:
        with pytest.raises(ValueError, match="NATIVE_TAIL_MATCH_TAX_UNIT_COVERAGE"):
            matching.match_invented_native_puf_tail(**arguments)
        return
    arguments["tax_units"] += (matching.InventedRecipientTaxUnit(999, 1),)
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.assignments[0].parent_household_id == 120
    assert {"household_id": 110, "reason": "multiple_tax_units"} in json.loads(
        result.receipt
    )["exclusions"]


def test_no_selected_donors_returns_explicit_no_expansion():
    arguments = inputs(amounts=(100.0,))
    arguments["donor_source_agi"] = ()
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.selection.donors == result.assignments == result.roles == ()
    assert result.expansion is None
    assert json.loads(result.receipt)["support"] == []


@pytest.mark.parametrize(
    "case,reason",
    [
        ("design", "HOUSEHOLD_WEIGHT_KIND"),
        ("extra_stored", "SOLE_HOUSEHOLD_WEIGHT"),
        ("changed_channel", "MEMBERSHIP_ORIGIN"),
        ("cross_household", "GROUP_NOT_HOUSEHOLD_CLOSED"),
    ],
)
def test_live_frame_inputs_reject_wrong_weight_or_membership_contract(case, reason):
    arguments = inputs()
    before = arguments["frame"]
    if case == "design":
        arguments["frame"] = host_fixture.invented_frame()
    elif case == "extra_stored":
        arguments["frame"] = Frame(
            {e: before.table(e).copy() for e in before.entities},
            before.schema,
            {
                "household": before.weights_for("household"),
                "person": before.resolve_weights("person"),
            },
            before.strata.copy(),
        )
    elif case == "changed_channel":
        before.person.loc[
            before.person.person_id == 101, matching.support_channel_column("person")
        ] = "acs"
    else:
        altered = host_fixture.invented_frame("cross_group")
        arguments["frame"] = Frame(
            {e: altered.table(e).copy() for e in altered.entities},
            altered.schema,
            {"household": before.weights_for("household")},
            altered.strata.copy(),
        )
    with pytest.raises(ValueError, match="NATIVE_TAIL_MATCH_" + reason):
        matching.match_invented_native_puf_tail(**arguments)


def test_cg_only_refuses_whole_request_and_overlap_uses_same_tested_selector():
    arguments = inputs(amounts=(6e6, 100.0))
    arguments["capital_gains_mask"] = np.array([True, True])
    with pytest.raises(ValueError, match="CG_ONLY_UNSUPPORTED"):
        matching.match_invented_native_puf_tail(**arguments)
    arguments = inputs()
    arguments["capital_gains_mask"] = np.array([True])
    result = matching.match_invented_native_puf_tail(**arguments)
    assert result.selection.donors[0].arm == 3
    assert result.assignments[0].parent_household_id == 110


def test_selection_to_real_expand_and_required_replay_preserves_six_entity_mass(
    tmp_path,
):
    arguments = inputs(amounts=(6e6, 6e6), donor_ids=[901, 902])
    arguments["donor_source_agi"] = (
        matching.InventedDonorSourceAgi(901, 40_000.0),
        matching.InventedDonorSourceAgi(902, 6e6),
    )
    matched = matching.match_invented_native_puf_tail(**arguments)
    assert {r.parent_household_id for r in matched.assignments} == {110, 120}
    store = ContentStore(tmp_path / "store")
    cold, cold_populations, graph = host_fixture.execute(
        tmp_path, declared=matched.expansion, store=store
    )
    warm, warm_populations, _ = host_fixture.execute(
        tmp_path, declared=matched.expansion, store=store, resume="require"
    )
    assert all(not cold.nodes[n].hit and warm.nodes[n].hit for n in graph.order)
    assert cold.key == warm.key
    for run, populations in ((cold, cold_populations), (warm, warm_populations)):
        parent, expanded = populations["importance"], populations[host_fixture.EXPAND]
        receipt = run.nodes[host_fixture.EXPAND].receipt
        assert set(dict(receipt["expand"]["household"]).values()) == {110, 120}
        np.testing.assert_array_equal(
            expanded.frame.weights_for("household").values,
            [8.0, 40.0, 0.0, 6.0, 4.0, 20.0, 0.0, 6.0, 4.0, 20.0],
        )
        for entity in US_SCHEMA.entities:
            old, new = parent.frame.table(entity), expanded.frame.table(entity)
            pd.testing.assert_frame_equal(
                old, new.iloc[: len(old)].reset_index(drop=True), check_exact=True
            )
            assert (
                parent.frame.resolve_weights(entity).total
                == expanded.frame.resolve_weights(entity).total
            )
            assert len(new) == len(old) + len(receipt["expand"][entity])
        np.testing.assert_array_equal(
            expanded.design_weights["household"],
            np.concatenate(
                [
                    parent.design_weights["household"],
                    parent.design_weights["household"][[4, 5]],
                ]
            ),
        )
        assert not receipt["monetary_placement"] and not receipt["matching_qualified"]
