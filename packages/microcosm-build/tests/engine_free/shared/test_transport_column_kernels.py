"""Column wrappers on synthetic donor graphs and generated concept tables.

Hypothesis properties hold ids, row order, G4 differential equality, stable
take-up draws, eligibility implication, exclusivity and household rent mass.
The graph fixture uses production transport CREATE with a tiny synthetic H5;
every receipt, rule, rate and target here is a toy modelling choice.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.column_kernels import (
    CONCEPTS_ENCODE,
    CONCEPTS_ENCODE_GROUPS,
    TAKEUP_ASSIGN,
    TRANSPORT_SCENARIO_OVERRIDE,
    TRANSPORT_UNIT_ATTRIBUTES,
    assign_receipts,
    register_column_kernels,
)
from microcosm.build.transport.target_kernels import compile_target_surface
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.concept_mapping import (
    ConceptMapping,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    StateBinding,
)
from microcosm.frame.concepts import (
    CONCEPTS,
    ContentBasis,
    derive_take_up_draws,
    split_for_transport,
)
from microcosm.frame.input_closure import (
    ClosureEntry,
    InputClosure,
    Knob,
    UndeterminedPolicy,
    apply_defaults,
    resolve_judgments,
)
from microcosm.frame.unit_construction import (
    benefit_unit_attributes,
    benefit_unit_membership,
    build_benefit_units,
    units_per_household,
)
from microcosm.graph import (
    ArtifactValue,
    KernelContext,
    Node,
    NumericScope,
    Owned,
    SeedSource,
    Slice,
    StructuralDelta,
)
from test_support.microcosm_build.transport_graph import (
    canonical_text,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import (
    UNIT_RULE as DONOR_UNIT_RULE,
)
from test_support.microcosm_build.transport_population import (
    digest,
    direct_population,
    population_graph,
    population_registry,
    run_population,
    write_population_sources,
)
from test_support.microcosm_frame.concept_frames import concept_frames

PROPERTY = settings(max_examples=35, deadline=None, database=None)
GRAPH_PROPERTY = settings(
    max_examples=6,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
PERSON_PATH = "xx/person.yaml"
GROUP_PATH = "xx/group.yaml"
RESOURCE_SHA = "b" * 64
# Generated frames include unrelated co-resident parents. The toy rule makes
# their child's placement explicit instead of refusing those valid frames.
UNIT_RULE = replace(DONOR_UNIT_RULE, split_parents="first_parent_unit")


def _binding(name, entity, concept, *, group_rule=None):
    return InputBinding(
        engine_input=name,
        engine_entity=entity,
        concepts=(concept,),
        transform=Identity(),
        relation="exact",
        note="Synthetic exact input, not a statutory definition.",
        group_rule=group_rule,
        module=PERSON_PATH if entity == "Person" else GROUP_PATH,
        canonical_input=f"xx:{entity}#input.{name}",
    )


BINDINGS = (
    _binding("input_age", "Person", "fact:person.age"),
    _binding("input_income", "Person", "fact:person.employment_income"),
    _binding(
        "input_assets",
        "Family",
        "fact:person.liquid_financial_assets",
        group_rule="sum_over_members",
    ),
    _binding(
        "input_rent",
        "Family",
        "fact:household.rent",
        group_rule="allocate_to_reference_unit",
    ),
)
MAPPING = ConceptMapping(
    engine="axiom:xx",
    engine_version="synthetic",
    entity_correspondence={"person": "Person", "household": "Household"},
    input_declaration=InputDeclaration.USAGE_INFERRED,
    bindings=BINDINGS,
    unmapped={
        item.id: "Outside this synthetic mapping."
        for item in CONCEPTS
        if item.id not in {concept for binding in BINDINGS for concept in binding.reads}
    },
)
STATES = (
    StateBinding(
        engine_input="input_non_beneficiary",
        engine_entity="Family",
        columns=(
            "person.receives_alpha",
            "person.receives_beta",
            "person.receives_super",
        ),
        test="any_true",
        group_rule=GroupRule.ANY_MEMBER,
        negate=True,
        note="Synthetic receipt state.",
        module=GROUP_PATH,
        canonical_input="xx:Family#input.input_non_beneficiary",
    ),
    StateBinding(
        engine_input="input_support_channel",
        engine_entity="Family",
        columns=("household.donor_support_stratum",),
        test="equals",
        value="us:asec",
        group_rule=GroupRule.HOUSEHOLD_VALUE,
        note="Synthetic donor-channel indicator.",
        module=GROUP_PATH,
        canonical_input="xx:Family#input.input_support_channel",
    ),
)
CLOSURE = InputClosure(
    country="xx",
    content_basis=ContentBasis.TRANSPORT,
    mapping_engine=MAPPING.engine,
    mapping_engine_version=MAPPING.engine_version,
    surface_rulespec_commit="synthetic",
    surface_engine_repository="toy/engine",
    surface_engine_commit="synthetic",
    modules={PERSON_PATH: RESOURCE_SHA, GROUP_PATH: RESOURCE_SHA},
    engine_optional_evidence="Synthetic engine declares no optional inputs.",
    undetermined=UndeterminedPolicy(
        action="refuse",
        reason="Toy refusal.",
        alternatives=("not_eligible", "eligible"),
        new_default=False,
    ),
    knobs=(
        Knob(
            name="asset_test",
            kind="switch_zeroes",
            targets=("input_assets",),
            central=True,
            values=(),
            method_card_row="toy",
            note="Toy zero-assets alternative.",
        ),
        Knob(
            name="rent_allocation",
            kind="allocation",
            targets=("input_rent",),
            central="reference_unit",
            values=("reference_unit", "per_adult_share"),
            method_card_row="toy",
            note="Toy rent allocation.",
        ),
        Knob(
            name="rent_factor",
            kind="factor",
            targets=("input_rent",),
            central=1.0,
            values=(),
            method_card_row="toy",
            note="Toy scaling.",
        ),
        Knob(
            name="channel_column",
            kind="state_column",
            targets=("household.donor_support_stratum",),
            central="donor_support_stratum",
            values=("donor_support_stratum", "tenure"),
            method_card_row="toy",
            note="Toy state replacement.",
        ),
    ),
    entries=tuple(
        ClosureEntry(
            module=binding.module,
            entity=binding.engine_entity,
            input=binding.engine_input,
            closure_class="encoded",
            encoded_by="concept_binding",
        )
        for binding in BINDINGS
    )
    + tuple(
        ClosureEntry(
            module=GROUP_PATH,
            entity="Family",
            input=state.engine_input,
            closure_class="encoded",
            encoded_by="state_binding",
            state_binding=state,
            reason="Toy state.",
        )
        for state in STATES
    )
    + (
        ClosureEntry(
            module=PERSON_PATH,
            entity="Person",
            input="eligibility_code",
            closure_class="defaulted",
            value=1,
            reason="Synthetic determined eligibility.",
            citation="Toy fixture.",
        ),
        ClosureEntry(
            module=GROUP_PATH,
            entity="Family",
            input="input_default",
            closure_class="defaulted",
            value=False,
            reason="Synthetic default.",
            citation="Toy fixture.",
        ),
    ),
)


def _rule_params(rule=UNIT_RULE):
    return {
        "unit_rule": canonical_text(rule.to_dict()),
        "unit_rule_sha256": digest(rule.to_dict()),
    }


def _encode_params(*, groups=False, knobs=None):
    params = {
        "mapping": canonical_text(MAPPING.to_dict()),
        "mapping_sha256": digest(MAPPING.to_dict()),
        "closure": canonical_text(CLOSURE.to_dict()),
        "closure_sha256": digest(CLOSURE.to_dict()),
        "rulespec_paths": (GROUP_PATH if groups else PERSON_PATH,),
    }
    if groups:
        params.update(_rule_params())
        params["engine_entity"] = "Family"
    if knobs is not None:
        params.update(knobs=canonical_text(knobs), scenario_sha256=digest(knobs))
    return params


def _contract(rate=0.6):
    return {
        "seed_column": "take_up_seed",
        "programs": [
            {
                "program": "xx.alpha",
                "output": "receives_alpha",
                "judgment_column": "eligibility_code",
                "payment_column": "input_income",
                "rate": rate,
            },
            {
                "program": "xx.beta",
                "output": "receives_beta",
                "judgment_column": "eligibility_code",
                "payment_column": "input_income",
                "rate": rate,
            },
            {
                "program": "xx.super",
                "output": "receives_super",
                "judgment_column": "eligibility_code",
                "rate": rate,
            },
        ],
        "exclusion_groups": [["receives_alpha", "receives_beta", "receives_super"]],
    }


def _context(kernel, frame, params, outputs, *, artifacts=None):
    node = Node("toy", kernel.ref, population="open", params=params, outputs=outputs)
    return KernelContext(
        node=node,
        tables={
            entity: frame.table(entity).copy(deep=True)
            for entity in frame.schema.entities
        },
        weights={
            entity: frame.resolve_weights(entity) for entity in frame.schema.entities
        },
        strata=frame.strata.copy(deep=True),
        params=node.params,
        rng=np.random.default_rng(1),
        artifacts={} if artifacts is None else artifacts,
    )


def _generated_frame(tables):
    tables, _ = split_for_transport(tables)
    person = tables["person"].copy()
    household = tables["household"].sort_values("household_id").reset_index(drop=True)
    units, members = build_benefit_units(person, household, UNIT_RULE)
    person[UNIT_RULE.membership_column] = members
    person["receives_alpha"] = False
    person["receives_beta"] = False
    person["receives_super"] = False
    household["donor_support_stratum"] = pd.array(
        ["us:asec"] * len(household), dtype="string"
    )
    return Frame(
        {"person": person, "household": household, "family": units},
        NZ_SCHEMA,
        {"household": Weights(np.ones(len(household)), WeightKind.DESIGN)},
        pd.Series(["toy"] * len(person), name="stratum"),
    )


def _assert_column(result, frame, entity, column, expected):
    actual = result.columns[entity, column]
    assert actual.index.tolist() == frame.table(entity)[f"{entity}_id"].tolist()
    assert actual.tolist() == expected.tolist()


@PROPERTY
@given(tables=concept_frames(max_households=3, max_members=4))
def test_unit_attributes_equal_g4_and_keep_ids_and_order(tables):
    frame = _generated_frame(tables)
    family = frame.table("family")
    person = frame.person
    expected = benefit_unit_attributes(
        person, family, person[UNIT_RULE.membership_column], UNIT_RULE
    )
    outputs = tuple(
        Owned(
            "family",
            column,
            "string" if column == "family_type" else str(expected[column].dtype),
        )
        for column in expected
        if column != "family_id"
    ) + (Owned("household", "n_family_units", "int64"),)
    result = TRANSPORT_UNIT_ATTRIBUTES.run(
        _context(TRANSPORT_UNIT_ATTRIBUTES, frame, _rule_params(), outputs)
    )
    for output in outputs[:-1]:
        _assert_column(result, frame, "family", output.column, expected[output.column])
    _assert_column(
        result,
        frame,
        "household",
        "n_family_units",
        units_per_household(frame.table("household"), family, UNIT_RULE),
    )
    assert np.array_equal(frame.person["person_id"], tables["person"]["person_id"])


@PROPERTY
@given(tables=concept_frames(max_households=3, max_members=4))
def test_person_encoding_equals_direct_mapping_and_defaults(tables):
    frame = _generated_frame(tables)
    outputs = (
        Owned("person", "input_age", "int64"),
        Owned("person", "input_income", "float64"),
        Owned("person", "eligibility_code", "int64"),
    )
    expected = apply_defaults(
        MAPPING.encode(
            {entity: frame.table(entity) for entity in ("person", "household")}
        ).tables["person"],
        CLOSURE.defaults(PERSON_PATH, "Person"),
    )
    result = CONCEPTS_ENCODE.run(
        _context(CONCEPTS_ENCODE, frame, _encode_params(), outputs)
    )
    for output in outputs:
        _assert_column(result, frame, "person", output.column, expected[output.column])


@PROPERTY
@given(
    tables=concept_frames(max_households=3, max_members=4),
    allocation=st.sampled_from(["reference_unit", "per_adult_share"]),
    asset_test=st.booleans(),
    factor=st.floats(min_value=0.5, max_value=2, allow_nan=False),
)
def test_group_and_scenario_encoding_equal_g4_and_conserve_rent(
    tables, allocation, asset_test, factor
):
    frame = _generated_frame(tables)
    knobs = {
        "rent_allocation": allocation,
        "asset_test": asset_test,
        "rent_factor": factor,
    }
    membership = benefit_unit_membership(
        frame.person,
        frame.table("family"),
        frame.person[UNIT_RULE.membership_column],
        UNIT_RULE,
    )
    expected = MAPPING.encode_groups(
        {entity: frame.table(entity) for entity in ("person", "household")},
        {"Family": membership},
        modules=(GROUP_PATH,),
        knobs=CLOSURE.group_knobs(knobs),
        state_bindings=CLOSURE.state_bindings(),
    ).tables["family"]
    outputs = (
        Owned("family", "input_assets", "float64", rewrite=True),
        Owned("family", "input_rent", "float64", rewrite=True),
        Owned("family", "input_non_beneficiary", "bool", rewrite=True),
        Owned("family", "input_support_channel", "bool", rewrite=True),
    )
    params = _encode_params(groups=True, knobs=knobs)
    for kernel in (CONCEPTS_ENCODE_GROUPS, TRANSPORT_SCENARIO_OVERRIDE):
        result = kernel.run(_context(kernel, frame, params, outputs))
        for output in outputs:
            _assert_column(
                result, frame, "family", output.column, expected[output.column]
            )
    rents = expected.groupby(frame.table("family")[UNIT_RULE.household_column])[
        "input_rent"
    ].sum()
    household_rents = (
        frame.table("household").set_index("household_id")["rent"] * factor
    )
    np.testing.assert_allclose(
        rents.reindex(household_rents.index), household_rents, rtol=1e-12
    )


@st.composite
def receipt_rows(draw):
    count = draw(st.integers(1, 15))
    codes = draw(st.lists(st.sampled_from([-1, 1]), min_size=count, max_size=count))
    seeds = draw(
        st.lists(
            st.floats(min_value=0, max_value=np.nextafter(1.0, 0), allow_nan=False),
            min_size=count,
            max_size=count,
        )
    )
    income = draw(
        st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=count,
            max_size=count,
        )
    )
    return pd.DataFrame(
        {
            "person_id": np.arange(count, dtype=np.int64) * 7 + 1,
            "eligibility_code": codes,
            "take_up_seed": seeds,
            "input_income": income,
        }
    )


@PROPERTY
@given(
    person=receipt_rows(),
    rate=st.floats(min_value=0, max_value=1, allow_nan=False),
    data=st.data(),
)
def test_receipts_imply_eligibility_positive_payment_and_exclusivity_and_stable_draws(
    person, rate, data
):
    contract = _contract(rate)
    out, audit = assign_receipts(person, contract)
    eligible = resolve_judgments(person["eligibility_code"].to_numpy(), "refuse")
    chosen = np.zeros(len(person), dtype=bool)
    for program in contract["programs"]:
        draws = derive_take_up_draws(
            person["take_up_seed"].to_numpy(), program["program"]
        )
        assert ((draws >= 0) & (draws < 1)).all()
        flags = out[program["output"]].to_numpy()
        permitted = eligible.copy()
        if "payment_column" in program:
            permitted &= person[program["payment_column"]].to_numpy() > 0
        expected = permitted & ~chosen & (draws < rate)
        np.testing.assert_array_equal(flags, expected)
        assert not (flags & ~permitted).any()
        chosen |= flags
    assert (out.iloc[:, 1:].sum(axis=1) <= 1).all()
    permutation = data.draw(st.permutations(range(len(person))))
    permuted, permuted_audit = assign_receipts(person.iloc[list(permutation)], contract)
    pd.testing.assert_frame_equal(
        out.set_index("person_id").sort_index(),
        permuted.set_index("person_id").sort_index(),
    )
    assert audit == permuted_audit


def _column_graph(sources, *, rate=0.6, knobs=None):
    graph = population_graph(sources, through="boundary")
    frame, _ = direct_population(sources)
    all_slices = tuple(
        Slice(
            entity,
            tuple(
                column
                for column in frame.table(entity)
                if column != f"{entity}_id"
                and not (
                    entity == "person"
                    and column.startswith("person_")
                    and column.endswith("_id")
                )
            ),
        )
        for entity in frame.schema.entities
    )
    attrs = Node(
        "nz.attributes",
        TRANSPORT_UNIT_ATTRIBUTES.ref,
        population="nz.open",
        inputs=all_slices,
        params=_rule_params(),
        outputs=(
            Owned("family", "family_type", "string"),
            Owned("family", "n_dependent_children", "int64"),
            Owned("family", "is_sole_parent", "bool"),
            Owned("family", "youngest_dependent_child_age", "Int64"),
            Owned("household", "n_family_units", "int64"),
        ),
    )
    encode = Node(
        "nz.encode",
        CONCEPTS_ENCODE.ref,
        population="nz.open",
        inputs=all_slices[:2],
        params=_encode_params(),
        outputs=(
            Owned("person", "input_age", "int64"),
            Owned("person", "input_income", "float64"),
            Owned("person", "eligibility_code", "int64"),
        ),
    )
    contract = _contract(rate)
    receipt = Node(
        "nz.receipts",
        TAKEUP_ASSIGN.ref,
        population="nz.open",
        inputs=(Slice("person", ("eligibility_code", "take_up_seed", "input_income")),),
        params={
            "receipt_contract": canonical_text(contract),
            "receipt_contract_sha256": digest(contract),
        },
        outputs=tuple(
            Owned("person", row["output"], "bool") for row in contract["programs"]
        ),
    )
    group_slices = (
        Slice(
            "person",
            all_slices[0].columns
            + tuple(row["output"] for row in contract["programs"]),
        ),
        all_slices[1],
        all_slices[2],
    )
    group_outputs = (
        Owned("family", "input_assets", "float64"),
        Owned("family", "input_rent", "float64"),
        Owned("family", "input_non_beneficiary", "bool"),
        Owned("family", "input_support_channel", "bool"),
        Owned("family", "input_default", "bool"),
    )
    group = Node(
        "nz.groups",
        CONCEPTS_ENCODE_GROUPS.ref,
        population="nz.open",
        inputs=group_slices,
        params=_encode_params(groups=True),
        outputs=group_outputs,
    )
    branch = Node(
        "nz.scenario",
        "transport.boundary@1",
        structural=StructuralDelta.FILTER,
        base="nz.open",
        inputs=(Slice("person", ("age",)),),
    )
    knobs = (
        {
            "asset_test": False,
            "rent_allocation": "per_adult_share",
            "channel_column": "tenure",
        }
        if knobs is None
        else knobs
    )
    scenario = Node(
        "nz.override",
        TRANSPORT_SCENARIO_OVERRIDE.ref,
        population=branch.id,
        inputs=group_slices,
        params=_encode_params(groups=True, knobs=knobs),
        outputs=tuple(replace(output, rewrite=True) for output in group_outputs[:4]),
    )
    return replace(
        graph, nodes=graph.nodes + (attrs, encode, receipt, group, branch, scenario)
    )


@GRAPH_PROPERTY
@given(
    rate=st.floats(min_value=0, max_value=1, allow_nan=False),
    factor=st.floats(min_value=0.5, max_value=2, allow_nan=False),
)
def test_each_column_kernel_runs_in_real_synthetic_donor_graph_and_rewrites_keep_order(
    tmp_path, rate, factor
):
    sources = write_population_sources(tmp_path / "sources")
    graph = _column_graph(
        sources, rate=rate, knobs={"rent_factor": factor, "asset_test": False}
    )
    registry = population_registry()
    register_column_kernels(registry)
    run = run_population(tmp_path / "run", sources, graph=graph, registry=registry)
    before = run.manifest.population("nz.open")
    after = run.manifest.population("nz.scenario")
    for entity in before.schema.entities:
        np.testing.assert_array_equal(
            before.table(entity)[f"{entity}_id"], after.table(entity)[f"{entity}_id"]
        )
    np.testing.assert_array_equal(after.table("family")["input_assets"], 0.0)
    np.testing.assert_allclose(
        after.table("family")["input_rent"],
        before.table("family")["input_rent"] * factor,
    )
    flags = before.person[["receives_alpha", "receives_beta", "receives_super"]]
    assert (flags.sum(axis=1) <= 1).all()
    assert not (flags["receives_alpha"] & (before.person["input_income"] <= 0)).any()
    assert not before.table("family")["input_default"].any()
    assert run.manifest.known_failures == ()


def test_take_up_refuses_undetermined_even_when_rate_zero_and_keeps_existing_exclusions():
    person = pd.DataFrame(
        {
            "person_id": [7],
            "eligibility_code": [0],
            "input_income": [10.0],
            "take_up_seed": [0.5],
        }
    )
    with pytest.raises(ValueError, match="undetermined"):
        assign_receipts(person, _contract(0))
    person["eligibility_code"] = 1
    person["existing_receipt"] = True
    contract = _contract(1)
    for row in contract["programs"]:
        row["exclude_columns"] = ["existing_receipt"]
    out, _ = assign_receipts(person, contract)
    assert not out.iloc[:, 1:].to_numpy().any()
    assert person["existing_receipt"].tolist() == [True]


def test_target_driven_receipts_bind_surface_and_report_expected_and_realized_mass(
    tmp_path,
):
    sources = write_population_sources(tmp_path / "sources")
    frame, _ = direct_population(sources)
    person = frame.person.copy()
    person["eligibility_code"] = 1
    person["input_income"] = person["employment_income"]
    frame = Frame(
        {
            "person": person,
            "household": frame.table("household"),
            "family": frame.table("family"),
        },
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    document = reference_document(
        [reference_row("toy_receipts", entity="person", measure="receives_alpha")]
    )
    facts = write_facts(
        tmp_path / "receipt-facts.jsonl",
        [toy_fact("toy_receipts", 60, entity="person")],
    )
    surface, _ = compile_target_surface(
        facts, document, country="xx", references_sha256=digest(document)
    )
    artifact = ArtifactValue(
        payload=surface,
        type=TARGET_SURFACE_TYPE,
        key=RESOURCE_SHA,
        producer_key=RESOURCE_SHA,
        numerics=NumericScope(),
    )
    contract = _contract(1)
    contract["programs"] = [
        {
            "program": "xx.alpha",
            "output": "receives_alpha",
            "judgment_column": "eligibility_code",
            "target": "toy_receipts",
        }
    ]
    contract["exclusion_groups"] = []
    params = {
        "receipt_contract": canonical_text(contract),
        "receipt_contract_sha256": digest(contract),
    }
    result = TAKEUP_ASSIGN.run(
        _context(
            TAKEUP_ASSIGN,
            frame,
            params,
            (Owned("person", "receives_alpha", "bool"),),
            artifacts={"surface": artifact},
        )
    )
    audit = result.receipt["programs"]["receives_alpha"]
    assert audit["target"] == 60
    assert audit["eligible_mass"] == 120
    assert audit["rate"] == 0.5
    direct, _ = assign_receipts(
        person,
        contract,
        weights=frame.resolve_weights("person").values,
        targets={"toy_receipts": 60},
    )
    _assert_column(result, frame, "person", "receives_alpha", direct["receives_alpha"])
    assert audit["realized_mass"] == sum(
        frame.resolve_weights("person").values[direct["receives_alpha"]]
    )
    with pytest.raises(ValueError, match="exceeds eligible"):
        assign_receipts(
            person,
            contract,
            weights=frame.resolve_weights("person").values,
            targets={"toy_receipts": 121},
        )
    money_document = reference_document(
        [reference_row("toy_receipts", entity="person", measure="input_income")]
    )
    money_surface, _ = compile_target_surface(
        facts, money_document, country="xx", references_sha256=digest(money_document)
    )
    with pytest.raises(ValueError, match="person count"):
        TAKEUP_ASSIGN.run(
            _context(
                TAKEUP_ASSIGN,
                frame,
                params,
                (Owned("person", "receives_alpha", "bool"),),
                artifacts={"surface": replace(artifact, payload=money_surface)},
            )
        )
    money_fact = toy_fact("toy_receipts", 60, entity="person")
    money_fact["observed_measure"]["unit"] = "NZD"
    wrong_unit_facts = write_facts(tmp_path / "monetary-facts.jsonl", [money_fact])
    wrong_unit_surface, _ = compile_target_surface(
        wrong_unit_facts, document, country="xx", references_sha256=digest(document)
    )
    with pytest.raises(ValueError, match="person count"):
        TAKEUP_ASSIGN.run(
            _context(
                TAKEUP_ASSIGN,
                frame,
                params,
                (Owned("person", "receives_alpha", "bool"),),
                artifacts={"surface": replace(artifact, payload=wrong_unit_surface)},
            )
        )


def test_wrappers_refuse_missing_resource_ids_missing_columns_and_non_input_overrides(
    tmp_path,
):
    sources = write_population_sources(tmp_path / "sources")
    frame, _ = direct_population(sources)
    params = _encode_params()
    params.pop("mapping_sha256")
    with pytest.raises(ValueError, match="missing required"):
        CONCEPTS_ENCODE.run(
            _context(
                CONCEPTS_ENCODE, frame, params, (Owned("person", "input_age", "int64"),)
            )
        )
    with pytest.raises(ValueError, match="did not encode"):
        CONCEPTS_ENCODE.run(
            _context(
                CONCEPTS_ENCODE,
                frame,
                _encode_params(),
                (Owned("person", "unmapped_input", "float64"),),
            )
        )
    with pytest.raises(ValueError, match="sha256"):
        CONCEPTS_ENCODE.run(
            _context(
                CONCEPTS_ENCODE,
                frame,
                {**_encode_params(), "shares": "{}"},
                (Owned("person", "input_age", "int64"),),
            )
        )
    with pytest.raises(ValueError, match="non-text"):
        CONCEPTS_ENCODE.run(
            _context(
                CONCEPTS_ENCODE,
                frame,
                _encode_params(),
                (Owned("person", "input_age", "string"),),
            )
        )
    with pytest.raises(ValueError, match="non-boolean"):
        CONCEPTS_ENCODE.run(
            _context(
                CONCEPTS_ENCODE,
                frame,
                _encode_params(),
                (Owned("person", "input_age", "bool"),),
            )
        )
    huge_person = frame.person.copy()
    huge_person["age"] = np.int64(2**53 + 1)
    huge_frame = Frame(
        {
            "person": huge_person,
            "household": frame.table("household"),
            "family": frame.table("family"),
        },
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )
    with pytest.raises(ValueError, match="lose values"):
        CONCEPTS_ENCODE.run(
            _context(
                CONCEPTS_ENCODE,
                huge_frame,
                _encode_params(),
                (Owned("person", "input_age", "float64"),),
            )
        )
    complete = _context(
        CONCEPTS_ENCODE,
        frame,
        _encode_params(),
        (Owned("person", "input_age", "int64"),),
    )
    masked_person = complete.tables["person"].copy()
    masked_person["is_in_scope"] = True
    mask_context = replace(
        complete,
        tables={**complete.tables, "person": masked_person},
    )
    with pytest.raises(ValueError, match="all-row"):
        CONCEPTS_ENCODE.run(
            replace(
                mask_context,
                node=replace(
                    complete.node,
                    inputs=(
                        Slice("person", ("age", "is_in_scope"), rows="is_in_scope"),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="all-row"):
        CONCEPTS_ENCODE.run(
            replace(
                mask_context,
                node=replace(
                    complete.node,
                    inputs=(Slice("person", ("age", "is_in_scope")),),
                    outputs=(
                        Owned("person", "input_age", "int64", rows="is_in_scope"),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="declare a data-column slice"):
        CONCEPTS_ENCODE.run(
            replace(complete, tables={"person": complete.tables["person"]})
        )
    with pytest.raises(ValueError, match="rewrites"):
        TRANSPORT_SCENARIO_OVERRIDE.run(
            _context(
                TRANSPORT_SCENARIO_OVERRIDE,
                frame,
                _encode_params(groups=True, knobs={"asset_test": False}),
                (Owned("family", "input_assets", "float64"),),
            )
        )


def test_capabilities_are_deterministic_country_neutral_and_hash_code_only():
    for kernel in (
        TRANSPORT_UNIT_ATTRIBUTES,
        CONCEPTS_ENCODE,
        CONCEPTS_ENCODE_GROUPS,
        TRANSPORT_SCENARIO_OVERRIDE,
        TAKEUP_ASSIGN,
    ):
        assert "nz" not in kernel.ref
        assert kernel.capabilities.determinism.value == "deterministic"
        assert kernel.capabilities.consumes_se is False
        assert kernel.implementation_hash() == kernel.implementation_hash()
        assert kernel.capabilities.seed_source is (
            SeedSource.KEYED if kernel is TAKEUP_ASSIGN else SeedSource.NONE
        )
