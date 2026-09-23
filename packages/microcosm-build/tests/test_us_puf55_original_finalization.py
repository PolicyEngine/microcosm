"""Invented whole-arm finalization arithmetic; no source owner or 55-fit claim.

The Frames below are the invented two-clone fixture used by the conservative
placement tests, with every non-fixed PUF55 output introduced on arm one. They
exercise eligibility, allocation and refusal mechanics only. Nothing here is
evidence of a genuine financial owner, fitted models or a qualified release.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_puf55_observed_recipients import _invented_values
from test_us_puf55_original_placement import _frame
from test_us_puf55_original_placement import result as placement_result

from microcosm.build.us_runtime import puf55_original_finalization as final
from microcosm.build.us_runtime import puf55_original_placement as placement
from microcosm.frame import Frame
from microcosm.graph import Node, Owned
from microcosm.graph import population as populations

values, application, codec = placement.values, placement.application, placement.codec
support = final.support
PROFILE = placement.attachment.PROFILES[0]
POLICY = final.POLICY
CLONE = final.provenance.support_clone_index_column
# Invented clone-zero membership (test_us_puf55_survey_ss_measurement._values):
# unit 10 = {1, 2}, 20 = {3, 4, 5}, 30 = {6, 7}, 40 = {8}, 50 = {9, 10}.
# Person 2 is outside the pension/IRA/net-property reporting universe, so unit
# 10 is mixed-known for those three development targets and known for farm.
UNITS = {10: [1, 2], 20: [3, 4, 5], 30: [6, 7], 40: [8], 50: [9, 10]}
NON_FIXED_PERSON = tuple(t for t in PROFILE.person_outputs if t not in final.FIXED_TARGETS)


def whole_fixture(*, dtype="float64", extra_person=None, draw=20.25):
    qualified, parent = _invented_values(rules=values.DEVELOPMENT_RULES)
    for name, column in (extra_person or {}).items():
        parent.person[name] = column
    financial = populations.Population.from_frame(parent, "invented.financial")
    tables = {e: parent.table(e).copy(deep=True) for e in parent.entities}
    outputs = []
    for entity, names in (
        ("person", NON_FIXED_PERSON),
        ("tax_unit", PROFILE.tax_unit_outputs),
    ):
        table = tables[entity]
        clone_one = table[entity + "_id"].ge(1000).to_numpy()
        for name in names:
            assert name not in table
            if name in final.BOOLEAN_OUTPUTS:
                table[name] = pd.array(
                    [True if c else pd.NA for c in clone_one], dtype="boolean"
                )
                outputs.append(Owned(entity, name, "boolean"))
            else:
                table[name] = np.where(clone_one, 777.0, np.nan).astype(dtype)
                outputs.append(Owned(entity, name, dtype))
    node = Node(
        placement.attachment.ATTACH_NODE,
        placement.attachment.SurveyPuf55AttachKernel.ref,
        population="invented.arm_one_version",
        outputs=tuple(outputs),
    )
    owners = dict(financial.owners)
    owners.update({(o.entity, o.column): node.id for o in outputs})
    arm_one = populations.Population.from_frame(
        _frame(parent, tables), node.population, owners
    )
    receiving = populations.Population.from_frame(
        _frame(parent, {e: t.copy(deep=True) for e, t in tables.items()}),
        "invented.late_version",
        owners,
    )
    inputs = placement.PlacementInputs(financial, arm_one, receiving, node)
    index = placement._selected_ids(qualified)
    conditioning = pd.DataFrame(draw, index=index, columns=PROFILE.targets)
    for name in qualified.tax_unit_values:
        mask = qualified.tax_unit_known.loc[index, name]
        conditioning.loc[mask, name] = qualified.tax_unit_values.loc[index[mask], name]
    return qualified, inputs, conditioning


def receipt(qualified, table):
    return codec.encode_json(
        {
            "protocol": application.PROTOCOL,
            "recipient_arm": 0,
            "qualification_sha256": codec.sha(qualified.receipt),
            "conditioning_table_sha256": values.recipients._table_digest(table),
        }
    )


def run(qualified, inputs, table):
    return final.finalization_result(
        qualified,
        inputs,
        table,
        receipt(qualified, table),
        profile=PROFILE,
        policy=POLICY,
    )


def person_values(columns, name):
    return columns["person", name]


def test_chain_prefixes_follow_the_actual_interleaved_fixed_positions():
    prefixes = final.chain_prefixes(PROFILE)
    financial = tuple(values.FINANCIAL_TARGETS)
    development = tuple(values.DEVELOPMENT_TARGETS)
    assert prefixes["long_term_capital_gains_on_collectibles"] == financial
    assert prefixes["alimony_income"] == financial + development[:2]
    assert prefixes["estate_income"] == financial + development[:3]
    assert prefixes["partnership_income"] == financial + development
    assert prefixes["health_savings_account_ald"] == financial + development
    # A fixed target's own prefix excludes itself.
    assert prefixes["rental_income"] == financial + development[:2]


@pytest.mark.parametrize("policy", [None, True, "v1", POLICY.encode(), POLICY + " "])
def test_policy_is_an_exact_closed_name(policy):
    with pytest.raises(ValueError, match="PUF55_ORIGINAL_FINALIZATION_POLICY"):
        final.require_policy(policy)


def test_mixed_unit_receives_only_targets_before_its_first_mixed_fixed_target():
    qualified, inputs, table = whole_fixture()
    columns, payload = run(qualified, inputs, table)
    document = codec.decode_json(payload)
    assert document["fixed_knownness"]["taxable_private_pension_income"] == {
        "all_known": 4,
        "mixed": 1,
        "all_unknown": 0,
    }
    assert document["fixed_knownness"]["farm_operations_income"]["mixed"] == 0
    collectibles = person_values(columns, "long_term_capital_gains_on_collectibles")
    alimony = person_values(columns, "alimony_income")
    # Unit 10's collectibles/non-Sch-D draws conditioned only on complete
    # financial totals; everything after its mixed pension target did not.
    assert collectibles.loc[[1, 2]].tolist() == [20.25, 0.0]
    assert alimony.loc[[1, 2]].isna().all()
    for unit in (20, 30, 40, 50):
        assert alimony.loc[UNITS[unit]].sum() == pytest.approx(20.25)
    counts = document["reason_counts"]
    assert counts["alimony_income"] == {"modeled": 4, "fixed_chain_unresolved": 1}
    assert counts["long_term_capital_gains_on_collectibles"] == {"modeled": 5}
    assert counts["health_savings_account_ald"] == {
        "modeled": 4,
        "fixed_chain_unresolved": 1,
    }
    # Fixed person values are preserved everywhere, including unit 10's known
    # adult and unknown child; no fixed target is a candidate.
    assert not {n for _, n, _ in document["candidate_outputs"]} & set(
        final.FIXED_TARGETS
    )


def test_fixed_values_clone_one_and_other_columns_are_unchanged():
    qualified, inputs, table = whole_fixture()
    before = placement._stamp(inputs)
    columns, _ = run(qualified, inputs, table)
    assert placement._stamp(inputs) == before
    for (entity, name), column in columns.items():
        incumbent = inputs.receiving.frame.table(entity).set_index(entity + "_id")[name]
        clone_one = column.index >= 1000
        pd.testing.assert_series_equal(column[clone_one], incumbent[clone_one])
        assert column.dtype == incumbent.dtype
    # Detached: changing a returned column cannot change the retained population.
    columns["person", "alimony_income"].iloc[0] = -1.0
    assert placement._stamp(inputs) == before


def test_unit_outputs_match_the_conservative_placement_where_it_writes():
    qualified, inputs, table = whole_fixture()
    table.loc[40, "partnership_income"] = -72.5
    columns, _ = run(qualified, inputs, table)
    conservative, _ = placement_result(qualified, inputs, table)
    for key, placed in conservative.items():
        written = placed.notna() & placed.index.isin(range(1000))
        pd.testing.assert_series_equal(columns[key][written], placed[written])
    # The whole arm writes a superset: collectibles reach every unit, not only
    # the complete singleton.
    assert conservative["person", "long_term_capital_gains_on_collectibles"].loc[
        range(1, 11)
    ].notna().sum() == 1
    assert columns["person", "long_term_capital_gains_on_collectibles"].loc[
        range(1, 11)
    ].notna().all()
    assert columns["person", "partnership_income"].loc[8] == -72.5


def test_person_allocation_uses_the_maintained_distribution_basis():
    qualified, inputs, table = whole_fixture()
    table.loc[20, "educator_expense"] = 120.0
    columns, payload = run(qualified, inputs, table)
    educator = person_values(columns, "educator_expense")
    # The invented employment leaf is person_id: shares 3/12, 4/12, 5/12.
    np.testing.assert_allclose(educator.loc[[3, 4, 5]], [30.0, 40.0, 50.0])
    # No declared basis: the maintained first-member rule.
    charity = person_values(columns, "charitable_cash_donations")
    assert charity.loc[[3, 4, 5]].tolist() == [20.25, 0.0, 0.0]
    allocation = codec.decode_json(payload)["allocation"]
    assert allocation["charitable_cash_donations"]["first_member_fallback_units"] == 3
    assert allocation["educator_expense"]["first_member_fallback_units"] == 0


def test_maintained_helper_parity_on_one_unit():
    qualified, inputs, table = whole_fixture()
    table.loc[30, "casualty_loss"] = 55.0
    columns, _ = run(qualified, inputs, table)
    person = inputs.receiving.frame.person.set_index("person_id", drop=False)
    work = person.loc[UNITS[30]].copy()
    work["casualty_loss"] = np.nan
    support._write_person_tax_unit_totals(
        work,
        mask=pd.Series(True, index=work.index),
        column="casualty_loss",
        totals=pd.Series([55.0], index=[30]),
        nonnegative=True,
        fallback_basis_columns=support._PERSON_OUTPUT_DISTRIBUTION_BASIS[
            "casualty_loss"
        ],
    )
    np.testing.assert_array_equal(
        person_values(columns, "casualty_loss").loc[UNITS[30]].to_numpy(),
        work.casualty_loss.to_numpy(),
    )


def test_boolean_counts_are_rounded_capped_and_ranked_by_the_basis():
    qualified, inputs, table = whole_fixture()
    name = "business_is_sstb"
    table.loc[20, name] = 1.4
    table.loc[30, name] = 7.0  # capped at the two allocation members
    table.loc[40, name] = 1.0
    table.loc[50, name] = 0.0
    table.loc[20, "partnership_income"] = 0.0
    table.loc[20, "s_corp_income"] = 0.0
    table.loc[20, "estate_income"] = 0.0
    columns, payload = run(qualified, inputs, table)
    flags = person_values(columns, name)
    assert flags.dtype == pd.BooleanDtype()
    # Ranked by maintained basis: self-employment (person_id + 1) is highest
    # for person 5, the only other nonzero basis in unit 20.
    assert flags.loc[[3, 4, 5]].tolist() == [False, False, True]
    assert flags.loc[[6, 7]].tolist() == [True, True]
    assert flags.loc[[9, 10]].tolist() == [False, False]
    assert flags.loc[8] is True or flags.loc[8] == True  # noqa: E712
    evidence = codec.decode_json(payload)["allocation"][name]
    assert evidence["count_adjusted_units"] == 2  # 1.4 and the capped 7
    assert flags.loc[[1, 2]].isna().all()  # unit 10: chain unresolved


def test_absent_declared_basis_declines_the_first_member_on_multi_member_units():
    qualified, inputs, table = whole_fixture()
    columns, payload = run(qualified, inputs, table)
    tuition = person_values(columns, "qualified_tuition_expenses")
    assert tuition.loc[UNITS[20] + UNITS[30] + UNITS[50]].isna().all()
    assert tuition.loc[8] == 20.25  # a complete singleton needs no basis
    counts = codec.decode_json(payload)["reason_counts"]["qualified_tuition_expenses"]
    assert counts == {
        "modeled": 1,
        "fixed_chain_unresolved": 1,
        "allocation_basis_absent": 3,
    }


def test_present_boolean_basis_allocates_and_a_null_member_stays_unresolved():
    student = pd.array(
        [False, False, True, False, pd.NA, True, False, False, False, True] * 2,
        dtype="boolean",
    )
    qualified, inputs, table = whole_fixture(
        extra_person={"is_full_time_college_student": student}
    )
    columns, payload = run(qualified, inputs, table)
    tuition = person_values(columns, "qualified_tuition_expenses")
    assert tuition.loc[UNITS[20]].isna().all()  # person 5's status is unknown
    assert tuition.loc[[6, 7]].tolist() == [20.25, 0.0]
    assert tuition.loc[[9, 10]].tolist() == [0.0, 20.25]
    counts = codec.decode_json(payload)["reason_counts"]["qualified_tuition_expenses"]
    assert counts["allocation_basis_unresolved"] == 1


def test_later_candidate_basis_contributes_zero_like_the_maintained_pass():
    qualified, inputs, table = whole_fixture()
    # business_is_sstb (chain 44) ranks on sstb self-employment (chain 47),
    # which is still unplaced when 44 is allocated: it is not a null basis.
    columns, payload = run(qualified, inputs, table)
    counts = codec.decode_json(payload)["reason_counts"]["business_is_sstb"]
    assert "allocation_basis_unresolved" not in counts


def test_earlier_unresolved_basis_blocks_only_multi_member_units():
    qualified, inputs, table = whole_fixture()
    # estate_income is signed; a nonfinite cannot reach here, so use the
    # nonnegative partnership-free path: make business_is_sstb unresolved by
    # domain, which blocks every sstb amount that ranks on it.
    table.loc[[20, 40], "business_is_sstb"] = -1.0
    columns, payload = run(qualified, inputs, table)
    counts = codec.decode_json(payload)["reason_counts"]
    assert counts["business_is_sstb"]["draw_domain_unresolved"] == 2
    sstb = counts["sstb_unadjusted_basis_qualified_property"]
    assert sstb["allocation_basis_unresolved"] == 1  # unit 20
    assert sstb["modeled"] == 3  # 30, 50 and the singleton 40
    assert person_values(columns, "sstb_unadjusted_basis_qualified_property").loc[
        8
    ] == pytest.approx(20.25)


def test_earnings_universe_zero_and_empty_universe():
    age = np.array([55, 55, 55, 55, 10, 55, 55, 12, 55, 55] * 2, dtype=np.int64)
    qualified, inputs, table = whole_fixture()
    for population in (inputs.financial_parent, inputs.arm_one, inputs.receiving):
        population.frame.person["age"] = age
    name = "sstb_self_employment_income_before_lsr"
    table.loc[20, name] = 90.0
    table.loc[40, name] = 15.0
    columns, payload = run(qualified, inputs, table)
    sstb = person_values(columns, name)
    assert sstb.loc[5] == 0.0  # receipted under-15 universe zero
    assert sstb.loc[[3, 4]].sum() == pytest.approx(90.0)
    assert np.isnan(sstb.loc[8])  # nonzero draw, no member aged 15+
    document = codec.decode_json(payload)
    assert document["reason_counts"][name]["allocation_universe_empty"] == 1
    assert document["allocation"][name]["universe_zero_persons"] >= 1
    # A zero draw on the under-15 singleton is the universe zero itself.
    table.loc[40, name] = 0.0
    columns, _ = run(qualified, inputs, table)
    assert person_values(columns, name).loc[8] == 0.0


def test_negative_nonnegative_draw_stays_unresolved_and_signed_draw_is_kept():
    qualified, inputs, table = whole_fixture()
    table.loc[30, "charitable_cash_donations"] = -5.0
    table.loc[30, "rental_income_would_be_qualified"] = -1.0
    table.loc[30, "farm_rent_income"] = -40.0
    columns, payload = run(qualified, inputs, table)
    assert person_values(columns, "charitable_cash_donations").loc[
        UNITS[30]
    ].isna().all()
    assert person_values(columns, "rental_income_would_be_qualified").loc[
        UNITS[30]
    ].isna().all()
    assert person_values(columns, "farm_rent_income").loc[UNITS[30]].sum() == -40.0
    counts = codec.decode_json(payload)["reason_counts"]
    assert counts["charitable_cash_donations"]["draw_domain_unresolved"] == 1


def test_numerical_policy_records_every_skipped_arm_one_step():
    qualified, inputs, table = whole_fixture()
    _, payload = run(qualified, inputs, table)
    document = codec.decode_json(payload)
    policy = document["numerical_policy"]
    for step in (
        "tail_bound_caps",
        "donor_value_snapping",
        "donor_positive_rate_pruning",
        "signed_mass_alignment",
    ):
        assert policy[step] == "not_applied"
    assert document["release_eligible"] is False
    assert document["source_admission_issued"] is False
    assert document["diagnostics_use"].startswith("comparison_only")
    diagnostics = document["descriptive_diagnostics"]["alimony_income"]
    assert diagnostics["modeled_units"] == 4
    assert diagnostics["weighted_positive_share"] == 1.0
    assert len(document["candidate_outputs"]) == 40 + 3


def test_tail_clone_in_the_receiving_population_is_refused_by_name():
    qualified, inputs, table = whole_fixture()
    frame = inputs.receiving.frame
    tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    for entity, table_ in tables.items():
        tail = table_.loc[table_[entity + "_id"].eq(1040)].copy()
        if entity == "person":
            tail = table_.loc[table_.person_tax_unit_id.eq(1040)].copy()
            tail["person_id"] += 1000
            for group in frame.schema.group_entities:
                tail["person_" + group + "_id"] += 1000
        else:
            tail[entity + "_id"] += 1000
        tail[CLONE(entity)] = 2
        tables[entity] = pd.concat([table_, tail], ignore_index=True)
    tailed = Frame(
        tables,
        frame.schema,
        {"household": type(frame.weights_for("household"))(
            np.ones(len(tables["household"])), frame.weights_for("household").kind
        )},
        pd.Series(["invented"] * len(tables["person"]), dtype="string"),
    )
    receiving = populations.Population.from_frame(
        tailed, inputs.receiving.version, inputs.receiving.owners
    )
    with pytest.raises(
        ValueError, match="PUF55_ORIGINAL_FINALIZATION_TAIL_CLONE_BEFORE"
    ):
        run(qualified, replace(inputs, receiving=receiving), table)


def test_float32_outputs_require_lossless_allocation():
    qualified, inputs, table = whole_fixture(dtype="float32", draw=0.0)
    columns, _ = run(qualified, inputs, table)  # zero allocations are exact
    assert columns["person", "educator_expense"].dtype == np.dtype("float32")
    table.loc[20, "educator_expense"] = 1.0  # shares 1/4 ... 5/12 are not
    with pytest.raises(ValueError, match="LOSSY_WRITE"):
        run(qualified, inputs, table)


@pytest.mark.parametrize(
    "change,reason",
    (
        ("known", "KNOWN_VALUE_CHANGED"),
        ("fixed_conditioning", "CONDITIONING_FIXED_VALUE"),
        ("nonnull", "PRESERVE_NULL_OWNERSHIP_CHANGED"),
    ),
)
def test_shared_preconditions_refuse(change, reason):
    qualified, inputs, table = whole_fixture()
    if change == "known":
        inputs.receiving.frame.person.loc[0, values.FINANCIAL_TARGETS[0]] += 1
    elif change == "fixed_conditioning":
        table.loc[40, values.FINANCIAL_TARGETS[0]] += 1
    else:
        inputs.receiving.frame.person.loc[0, "alimony_income"] = 0.0
    with pytest.raises(ValueError, match=reason):
        run(qualified, inputs, table)


def test_document_is_replay_invariant_across_receiving_column_order():
    qualified, inputs, table = whole_fixture()
    _, first = run(qualified, inputs, table)
    frame = inputs.receiving.frame
    tables = {e: frame.table(e).copy(deep=True) for e in frame.entities}
    tables["person"] = tables["person"][list(tables["person"].columns)[::-1]]
    replayed = replace(
        inputs,
        receiving=populations.Population.from_frame(
            _frame(frame, tables), inputs.receiving.version, inputs.receiving.owners
        ),
    )
    _, second = run(qualified, replayed, table)
    assert second == first


def test_final_input_change_during_allocation_is_refused(monkeypatch):
    qualified, inputs, table = whole_fixture()
    original = support._write_person_tax_unit_totals

    def mutate(person, **kwargs):
        original(person, **kwargs)
        inputs.receiving.frame.person.loc[0, "age"] += 1

    monkeypatch.setattr(support, "_write_person_tax_unit_totals", mutate)
    with pytest.raises(ValueError, match="FINAL_INPUT_CHANGED"):
        run(qualified, inputs, table)


@pytest.mark.parametrize("module", ("puf55_original_finalization.py",))
def test_new_module_uses_reviewed_dynamic_selectors_without_provenance_exemption(
    module,
):
    import test_us_spine_blindness as scanner

    assert module not in scanner._SOURCE_SPINE_PROVENANCE_OWNERS
    assert (
        scanner._non_owner_source_spine_accesses(
            module, (scanner._US_RUNTIME / module).read_text()
        )
        == ()
    )
