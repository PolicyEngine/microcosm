"""Concept-to-input mappings: the contract every mapping meets, and the machinery.

Every committed mapping (policyengine-us, policyengine-uk, Axiom New Zealand
and Belgium) is plain data importable without any engine, so the same
properties run over all of them here: construction rules, serialization,
round trips, and a differential test of the vectorized encoder against a
row-by-row reference. Country-specific expectations live in the
``engine_free/us`` and ``engine_free/uk`` files of the same name; whether each
mapped input exists in the installed engine is checked by the engine tests.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame.concept_mapping import (
    AllocateToReferencePerson,
    ConceptMapping,
    CoresidentChildCount,
    CoverageReport,
    Fraction,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    InputRef,
    Positive,
    Predicate,
    Product,
    Recode,
    RelationshipRole,
    Role,
    Scale,
    Share,
    Sum,
    TakeUpThreshold,
    coverage_report,
    transform_from_dict,
    transform_to_dict,
)
from microcosm.frame.concepts import (
    CONCEPTS,
    AlignmentRelation,
    CanonicalConceptKind,
    concept,
    derive_take_up_draws,
)
from test_support.microcosm_frame.concept_frames import (
    assert_concepts_round_trip,
    concept_frames,
    shares,
)
from test_support.microcosm_frame.concept_mappings import (
    COVERAGE_DOCS,
    concept_mappings,
    coverage_golden,
)
from test_support.microcosm_frame.concept_reference import (
    assert_encode_matches_reference,
)

PROPERTY = settings(max_examples=100, deadline=None)
MAPPINGS = concept_mappings()
mapping_names = st.sampled_from(sorted(MAPPINGS))
_PARENTS = ("fact:person.parent_1_person_id", "fact:person.parent_2_person_id")


def _parameters(mapping: ConceptMapping, data) -> dict[str, dict[str, float]]:
    return {
        "shares": {
            name: data.draw(shares, label=name) for name in mapping.share_parameters()
        },
        "take_up_rates": {
            name: data.draw(shares, label=name) for name in mapping.take_up_programs()
        },
    }


def _assert_same_values(actual: pd.Series, expected: pd.Series) -> None:
    if actual.dtype.kind in "fiub" and expected.dtype.kind in "fiub":
        np.testing.assert_allclose(
            actual.to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-6,
            err_msg=str(actual.name),
        )
    else:
        assert actual.tolist() == expected.tolist(), actual.name


class TestEveryMapping:
    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_every_concept_is_read_or_unmapped_with_a_reason(self, name) -> None:
        mapping = MAPPINGS[name]
        for item in CONCEPTS:
            read = bool(mapping.bindings_for(item.id))
            unmapped = item.id in mapping.unmapped
            assert read != unmapped, item.id
            if unmapped:
                assert mapping.unmapped[item.id]

    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_every_binding_carries_evidence_and_a_real_relation(self, name) -> None:
        for binding in MAPPINGS[name].bindings:
            assert binding.note
            assert binding.relation is not AlignmentRelation.SOURCE_LABEL

    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_serialisation_round_trips(self, name) -> None:
        mapping = MAPPINGS[name]
        again = ConceptMapping.from_dict(json.loads(json.dumps(mapping.to_dict())))
        assert again == mapping
        assert hash(again) == hash(mapping)
        assert again.structural_inputs == mapping.structural_inputs

    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_no_binding_targets_a_structural_input(self, name) -> None:
        mapping = MAPPINGS[name]
        assert not {binding.engine_input for binding in mapping.bindings} & set(
            mapping.structural_inputs
        )

    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_allocation_reads_the_reference_person(self, name) -> None:
        for binding in MAPPINGS[name].bindings:
            places = isinstance(binding.transform, AllocateToReferencePerson) or (
                binding.group_rule is GroupRule.ALLOCATE_TO_REFERENCE_UNIT
            )
            if places:
                assert "fact:household.reference_person_id" in binding.reads

    def test_declarations_match_the_engine_kind(self) -> None:
        for name, mapping in MAPPINGS.items():
            axiom = name.startswith("axiom-")
            expected = (
                InputDeclaration.USAGE_INFERRED
                if axiom
                else InputDeclaration.ENGINE_TYPED
            )
            assert mapping.input_declaration is expected
            assert all(
                (binding.module is not None) == axiom for binding in mapping.bindings
            )


class TestExecution:
    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), data=st.data())
    def test_encode_matches_a_row_by_row_reference(self, name, tables, data) -> None:
        mapping = MAPPINGS[name]
        parameters = _parameters(mapping, data)
        compared = assert_encode_matches_reference(
            mapping, tables, parameters["shares"], parameters["take_up_rates"]
        )
        assert compared == sum(1 for b in mapping.bindings if mapping.is_executable(b))

    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), data=st.data())
    def test_round_trip_preserves_every_invertible_concept(
        self, name, tables, data
    ) -> None:
        mapping = MAPPINGS[name]
        encoded = mapping.encode(tables, **_parameters(mapping, data))
        decoded = mapping.decode(encoded.tables)
        invertible = mapping.invertible_concepts()
        assert invertible
        assert_concepts_round_trip(decoded, tables, invertible)

    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), data=st.data())
    def test_encoding_is_idempotent_through_a_round_trip(
        self, name, tables, data
    ) -> None:
        mapping = MAPPINGS[name]
        parameters = _parameters(mapping, data)
        first = mapping.encode(tables, **parameters)
        decoded = mapping.decode(first.tables)
        # The reference person is structure: an engine without a
        # reference-person input (Belgium) cannot give it back, yet
        # allocation needs it to re-encode, so it carries over.
        if "reference_person_id" not in decoded["household"].columns:
            decoded["household"]["reference_person_id"] = tables["household"][
                "reference_person_id"
            ].to_numpy()
        again = mapping.encode(decoded, **parameters)
        for entity, table in again.tables.items():
            for column in table.columns:
                _assert_same_values(table[column], first.tables[entity][column])

    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), data=st.data())
    def test_encoding_is_deterministic_and_row_aligned(
        self, name, tables, data
    ) -> None:
        mapping = MAPPINGS[name]
        parameters = _parameters(mapping, data)
        first = mapping.encode(tables, **parameters)
        second = mapping.encode(tables, **parameters)
        for entity in ("person", "household"):
            assert len(first.tables[entity]) == len(tables[entity])
            pd.testing.assert_frame_equal(first.tables[entity], second.tables[entity])

    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), data=st.data())
    def test_allocation_conserves_household_amounts(self, name, tables, data) -> None:
        mapping = MAPPINGS[name]
        person = mapping.encode(tables, **_parameters(mapping, data)).tables["person"]
        household = tables["household"].set_index("household_id")
        for binding in mapping.bindings:
            if not isinstance(binding.transform, AllocateToReferencePerson):
                continue
            item = concept(binding.concepts[0])
            sums = person.groupby("person_household_id")[binding.engine_input].sum()
            np.testing.assert_allclose(
                sums.reindex(household.index).to_numpy(),
                household[item.name].to_numpy(),
            )

    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), data=st.data())
    def test_each_share_leaf_gets_its_own_part(self, name, tables, data) -> None:
        mapping = MAPPINGS[name]
        parameters = _parameters(mapping, data)
        parameters["shares"] = dict.fromkeys(parameters["shares"], 0.9)
        person = mapping.encode(tables, **parameters).tables["person"]
        for binding in mapping.bindings:
            if isinstance(binding.transform, Share) and mapping.is_executable(binding):
                expected = tables["person"][concept(binding.concepts[0]).name] * (
                    0.1 if binding.transform.complement else 0.9
                )
                np.testing.assert_allclose(
                    person[binding.engine_input].to_numpy(),
                    expected.to_numpy(),
                    rtol=1e-12,
                    atol=1e-6,
                )

    @PROPERTY
    @given(name=mapping_names, tables=concept_frames(), low=shares, high=shares)
    def test_take_up_is_monotone_in_the_rate(self, name, tables, low, high) -> None:
        mapping = MAPPINGS[name]
        low, high = sorted((low, high))
        split = dict.fromkeys(mapping.share_parameters(), 0.5)
        at = {
            rate: mapping.encode(
                tables,
                shares=split,
                take_up_rates=dict.fromkeys(mapping.take_up_programs(), rate),
            ).tables["person"]
            for rate in (0.0, low, high, 1.0)
        }
        for binding in mapping.bindings:
            if isinstance(binding.transform, TakeUpThreshold) and mapping.is_executable(
                binding
            ):
                column = binding.engine_input
                assert not at[0.0][column].any()
                assert at[1.0][column].all()
                assert (at[low][column] <= at[high][column]).all()

    @PROPERTY
    @given(tables=concept_frames(min_households=3, max_households=4))
    def test_programs_draw_independently(self, tables) -> None:
        seeds = np.random.default_rng(0).random(len(tables["person"]))
        tables["person"]["take_up_seed"] = seeds
        bindings = [
            _binding(
                engine_input=f"takes_up_{program}",
                concepts=("fact:person.take_up_seed",),
                transform=TakeUpThreshold(program=f"us.{program}"),
            )
            for program in ("a", "b")
        ]
        out = _mapping(bindings).encode(
            tables, take_up_rates={"us.a": 0.5, "us.b": 0.5}
        )
        flags = out.tables["person"]
        for program in ("a", "b"):
            expected = derive_take_up_draws(seeds, f"us.{program}") < 0.5
            assert flags[f"takes_up_{program}"].tolist() == expected.tolist()

    def test_an_unseeded_program_is_left_to_the_engine_default(self) -> None:
        tables = {
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1], "take_up_seed": [0.3]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        binding = _binding(
            engine_input="takes_up_x",
            concepts=("fact:person.take_up_seed",),
            transform=TakeUpThreshold(program="us.x"),
        )
        encoded = _mapping([binding]).encode(tables, take_up_rates={"us.x": None})
        assert "takes_up_x" not in encoded.tables["person"].columns
        with pytest.raises(ValueError, match="us.x"):
            _mapping([binding]).encode(tables)

    def test_person_ids_beyond_two_to_the_53_resolve_exactly(self) -> None:
        big = 2**53
        person = pd.DataFrame(
            {
                "person_id": np.array([big, big + 1, big + 2], dtype=np.int64),
                "person_household_id": [1, 1, 1],
                "partner_person_id": pd.array([pd.NA, big + 2, big + 1], dtype="Int64"),
                "legal_marital_status": ["never_married", "never_married", "married"],
                "parent_1_person_id": pd.array([big + 1, pd.NA, pd.NA], dtype="Int64"),
                "parent_2_person_id": pd.array([big + 2, pd.NA, pd.NA], dtype="Int64"),
            }
        )
        household = pd.DataFrame(
            {
                "household_id": [1],
                "reference_person_id": pd.array([big + 2], dtype="Int64"),
            }
        )
        roles = {
            "spouse": Role.REFERENCE_PERSON_PARTNER,
            "unmarried": Role.UNMARRIED_PARTNER_OF_REFERENCE_PERSON,
        }
        bindings = [
            _binding(
                engine_input=name,
                concepts=RelationshipRole(role=role).concepts,
                transform=RelationshipRole(role=role),
            )
            for name, role in roles.items()
        ]
        bindings.append(
            _binding(
                engine_input="children",
                concepts=_PARENTS,
                transform=CoresidentChildCount(),
            )
        )
        out = _mapping(bindings).encode({"person": person, "household": household})
        flags = out.tables["person"]
        assert flags["spouse"].tolist() == [False, True, False]
        assert flags["unmarried"].tolist() == [False, True, False]
        assert flags["children"].tolist() == [0, 1, 1]

    def test_out_of_domain_categories_are_refused_by_name(self) -> None:
        tables = {
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1], "sex": ["other"]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        binding = _binding(
            engine_input="female",
            concepts=("fact:person.sex",),
            transform=Recode(pairs=(("female", True), ("male", False))),
        )
        with pytest.raises(ValueError, match="fact:person.sex"):
            _mapping([binding]).encode(tables)

    def test_missing_parameters_fail_closed(self) -> None:
        tables = {
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1], "interest_income": [5.0]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        interest = {"concepts": ("fact:person.interest_income",)}
        mapping = _mapping(
            [
                _binding(engine_input="a", transform=Share(parameter="p"), **interest),
                _binding(
                    engine_input="b",
                    transform=Share(parameter="p", complement=True),
                    **interest,
                ),
            ]
        )
        with pytest.raises(ValueError, match="'p'"):
            mapping.encode(tables)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            mapping.encode(tables, shares={"p": 1.5})

    def test_allocation_needs_the_reference_person(self) -> None:
        tables = {
            "person": pd.DataFrame({"person_id": [1], "person_household_id": [1]}),
            "household": pd.DataFrame({"household_id": [1], "rent": [100.0]}),
        }
        binding = _binding(
            engine_input="rent_paid",
            concepts=("fact:household.rent",),
            transform=AllocateToReferencePerson(),
        )
        with pytest.raises(ValueError, match="reference_person_id"):
            _mapping([binding]).encode(tables)

    def test_group_bindings_are_deferred_not_executed(self) -> None:
        tables = {
            "person": pd.DataFrame({"person_id": [1], "person_household_id": [1]}),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        for mapping in MAPPINGS.values():
            encoded = mapping.encode(tables)
            deferred = {binding.engine_input for binding in encoded.deferred}
            assert all(binding.group_rule is not None for binding in encoded.deferred)
            assert deferred == {
                binding.engine_input
                for binding in mapping.bindings
                if binding.group_rule is not None
            }


class TestDecodeGuards:
    def test_a_whole_number_concept_refuses_fractions(self) -> None:
        mapping = _mapping([_binding()])
        tables = {
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1], "x": [40.5]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        with pytest.raises(ValueError, match="whole numbers"):
            mapping.decode(tables)

    def test_a_household_with_two_reference_people_is_refused(self) -> None:
        mapping = _mapping(
            [
                _binding(
                    engine_input="head",
                    concepts=("fact:household.reference_person_id",),
                    transform=RelationshipRole(role=Role.REFERENCE_PERSON),
                )
            ]
        )
        tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2],
                    "person_household_id": [1, 1],
                    "head": [True, True],
                }
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        with pytest.raises(ValueError, match="more than one reference person"):
            mapping.decode(tables)

    def test_a_boolean_concept_refuses_missing_values(self) -> None:
        mapping = _mapping(
            [
                _binding(
                    engine_input="disabled",
                    concepts=("fact:person.has_disability",),
                    transform=Identity(),
                )
            ]
        )
        tables = {
            "person": pd.DataFrame(
                {"person_id": [1], "person_household_id": [1], "disabled": [np.nan]}
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        with pytest.raises(ValueError, match="true and false"):
            mapping.decode(tables)


def _binding(**change) -> InputBinding:
    base = {
        "engine_input": "x",
        "engine_entity": "person",
        "concepts": ("fact:person.age",),
        "transform": Identity(),
        "relation": AlignmentRelation.EXACT,
        "note": "evidence",
    }
    return InputBinding(**{**base, **change})


def _mapping(bindings, unmapped=None, **change) -> ConceptMapping:
    read = {concept_id for binding in bindings for concept_id in binding.reads}
    default_unmapped = {
        item.id: "not in this test" for item in CONCEPTS if item.id not in read
    }
    return ConceptMapping(
        **{
            "engine": "test",
            "engine_version": "0",
            "entity_correspondence": {"person": "person", "household": "household"},
            "input_declaration": InputDeclaration.ENGINE_TYPED,
            "bindings": tuple(bindings),
            "unmapped": default_unmapped if unmapped is None else unmapped,
            **change,
        }
    )


_AXIOM = {
    "input_declaration": InputDeclaration.USAGE_INFERRED,
    "entity_correspondence": {"person": "Person", "household": "Household"},
}


def _module_binding(module: str, **change) -> InputBinding:
    change.setdefault("engine_input", "x")
    return _binding(
        engine_entity=change.pop("engine_entity", "Person"),
        module=module,
        canonical_input=f"nz:{module}#input.{change['engine_input']}",
        **change,
    )


class TestConstructionRules:
    def test_a_minimal_mapping_constructs(self) -> None:
        mapping = _mapping([_binding()])
        assert mapping.invertible_concepts() == ("fact:person.age",)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"relation": AlignmentRelation.SOURCE_LABEL}, "source_label"),
            ({"note": ""}, "note"),
            ({"concepts": ()}, "distinct"),
            ({"concepts": ("fact:person.nope",)}, "Unknown concept"),
            ({"concepts": ("fact:person.age", "fact:person.sex")}, "exactly one"),
            ({"concepts": ("fact:person.partner_person_id",)}, "structure"),
            (
                {
                    "transform": Recode(pairs=(("female", True),)),
                    "concepts": ("fact:person.sex",),
                },
                "domain",
            ),
            ({"transform": Share(parameter="p")}, "not an amount"),
            ({"transform": Scale(factor=0.5)}, "only float concepts scale"),
            ({"transform": RelationshipRole(role=Role.REFERENCE_PERSON)}, "reads"),
            ({"transform": CoresidentChildCount()}, "both parents"),
            ({"transform": TakeUpThreshold(program="x")}, "seed"),
            (
                {
                    "transform": Predicate(clauses=(("fact:person.sex", ("other",)),)),
                    "concepts": ("fact:person.sex",),
                },
                "domain",
            ),
        ],
    )
    def test_malformed_bindings_are_refused(self, change, message) -> None:
        with pytest.raises(ValueError, match=message):
            _binding(**change)

    def test_every_concept_must_be_accounted_for(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            _mapping([_binding()], unmapped={})

    def test_a_concept_is_read_or_unmapped_not_both(self) -> None:
        with pytest.raises(ValueError, match="both bound and unmapped"):
            _mapping([_binding()], unmapped={item.id: "reason" for item in CONCEPTS})
        rent = _binding(
            engine_input="rent_paid",
            concepts=("fact:household.rent",),
            transform=AllocateToReferencePerson(),
        )
        with pytest.raises(ValueError, match="reference_person_id is both"):
            _mapping(
                [rent],
                unmapped={
                    item.id: "reason"
                    for item in CONCEPTS
                    if item.id != "fact:household.rent"
                },
            )

    def test_an_input_is_bound_once(self) -> None:
        with pytest.raises(ValueError, match="bound twice"):
            _mapping([_binding(), _binding(concepts=("fact:person.weeks_worked",))])

    def test_a_structural_input_is_never_bound(self) -> None:
        with pytest.raises(ValueError, match="structural"):
            _mapping([_binding()], structural_inputs=("x",))

    def test_shares_come_in_complementary_pairs(self) -> None:
        interest = {"concepts": ("fact:person.interest_income",)}
        lone = _binding(engine_input="a", transform=Share(parameter="p"), **interest)
        with pytest.raises(ValueError, match="complement"):
            _mapping([lone])
        twin = _binding(engine_input="b", transform=Share(parameter="p"), **interest)
        with pytest.raises(ValueError, match="complement"):
            _mapping([lone, twin])
        pair = _binding(
            engine_input="b",
            transform=Share(parameter="p", complement=True),
            **interest,
        )
        assert _mapping([lone, pair]).share_parameters() == ("p",)

    def test_a_share_pair_may_repeat_in_every_module(self) -> None:
        interest = {"concepts": ("fact:person.interest_income",)}
        bindings = [
            _module_binding(
                module,
                engine_input=name,
                transform=Share(parameter="p", complement=complement),
                **interest,
            )
            for module in ("a", "b")
            for name, complement in (("taxed", False), ("exempt", True))
        ]
        mapping = _mapping(bindings, **_AXIOM)
        assert "fact:person.interest_income" in mapping.invertible_concepts()

    def test_a_parameter_is_a_share_or_a_fraction_not_both(self) -> None:
        interest = {"concepts": ("fact:person.interest_income",)}
        pair = [
            _binding(engine_input="a", transform=Share(parameter="p"), **interest),
            _binding(
                engine_input="b",
                transform=Share(parameter="p", complement=True),
                **interest,
            ),
        ]
        clash = _binding(
            engine_input="c", transform=Fraction(parameter="p"), **interest
        )
        with pytest.raises(ValueError, match="both a share pair and a fraction"):
            _mapping([*pair, clash])

    def test_entities_must_line_up(self) -> None:
        with pytest.raises(ValueError, match="lives on"):
            _mapping([_binding(engine_entity="household")])
        with pytest.raises(ValueError, match="group rule"):
            _mapping([_binding(engine_entity="tax_unit")])
        with pytest.raises(ValueError, match="group rule"):
            _mapping([_binding(group_rule=GroupRule.REFERENCE_MEMBER)])

    def test_group_rules_fit_the_concept(self) -> None:
        rent = {"concepts": ("fact:household.rent",), "engine_entity": "spm_unit"}
        with pytest.raises(ValueError, match="double-counted"):
            _mapping([_binding(group_rule=GroupRule.HOUSEHOLD_VALUE, **rent)])
        with pytest.raises(ValueError, match="household amounts only"):
            _mapping(
                [
                    _binding(
                        engine_entity="spm_unit",
                        group_rule=GroupRule.ALLOCATE_TO_REFERENCE_UNIT,
                    )
                ]
            )
        with pytest.raises(ValueError, match="person values only"):
            _mapping([_binding(group_rule=GroupRule.SUM_OVER_MEMBERS, **rent)])

    def test_module_scoped_engines_name_modules(self) -> None:
        with pytest.raises(ValueError, match="module"):
            _mapping([_binding()], **_AXIOM)
        with pytest.raises(ValueError, match="global input"):
            _mapping([_binding(module="nz/x.yaml", canonical_input="nz:x#input.x")])

    def test_a_shared_input_name_is_computed_alike_in_every_module(self) -> None:
        same = _mapping([_module_binding("a"), _module_binding("b")], **_AXIOM)
        assert len(same.refs()) == 2
        with pytest.raises(ValueError, match="computed differently"):
            _mapping(
                [
                    _module_binding("a"),
                    _module_binding("b", concepts=("fact:person.weeks_worked",)),
                ],
                **_AXIOM,
            )

    def test_entity_correspondence_covers_both_concept_entities(self) -> None:
        with pytest.raises(ValueError, match="entity_correspondence"):
            _mapping([_binding()], entity_correspondence={"person": "person"})

    def test_a_non_injective_recode_is_not_invertible(self) -> None:
        collapse = Recode(pairs=(("female", "X"), ("male", "X")))
        assert not collapse.injective
        with pytest.raises(ValueError, match="inverse"):
            collapse.inverse()
        mapping = _mapping(
            [_binding(concepts=("fact:person.sex",), transform=collapse)]
        )
        assert mapping.invertible_concepts() == ()

    def test_replace_revalidates(self) -> None:
        mapping = _mapping([_binding()])
        with pytest.raises(ValueError, match="bound twice"):
            replace(mapping, bindings=(_binding(), _binding()))

    def test_serialised_mappings_refuse_unknown_fields(self) -> None:
        data = _mapping([_binding()]).to_dict()
        data["bindings"][0]["colour"] = "red"
        with pytest.raises(ValueError, match="unexpected"):
            ConceptMapping.from_dict(data)
        del data["bindings"][0]["colour"]
        del data["bindings"][0]["transform"]
        with pytest.raises(ValueError, match="lacks"):
            ConceptMapping.from_dict(data)


TRANSFORMS = [
    Identity(),
    Recode(pairs=(("female", "F"), ("male", "M"))),
    Share(parameter="p", complement=True),
    Fraction(parameter="f"),
    Scale(factor=1 / 52),
    Positive(),
    Sum(),
    Product(),
    AllocateToReferencePerson(),
    Predicate(clauses=(("fact:person.sex", ("female",)),)),
    RelationshipRole(role=Role.HAS_PARTNER),
    RelationshipRole(role=Role.PARENT_OF_CORESIDENT_CHILD, max_child_age=17),
    CoresidentChildCount(),
    CoresidentChildCount(max_age=17),
    TakeUpThreshold(program="nz.x"),
]


class TestTransforms:
    @pytest.mark.parametrize("transform", TRANSFORMS, ids=repr)
    def test_every_transform_serialises_and_reads_back(self, transform) -> None:
        data = json.loads(json.dumps(transform_to_dict(transform)))
        assert transform_from_dict(data) == transform

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ({"kind": "shift"}, "Unknown transform"),
            ({"kind": "identity", "factor": 2}, "unexpected"),
            (
                {"kind": "share", "parameter": "p", "complement": "false"},
                "true or false",
            ),
            ({"kind": "scale", "factor": True}, "number"),
            ({"kind": "coresident_child_count", "max_age": "17"}, "integer"),
            ({"kind": "take_up_threshold"}, "lacks"),
        ],
    )
    def test_malformed_transforms_are_refused(self, data, message) -> None:
        with pytest.raises(ValueError, match=message):
            transform_from_dict(data)

    @pytest.mark.parametrize("factor", [0.0, -1.0, float("inf")])
    def test_scale_factors_are_finite_and_positive(self, factor) -> None:
        with pytest.raises(ValueError, match="scale factor"):
            Scale(factor=factor)

    def test_only_the_parent_role_takes_a_child_age(self) -> None:
        with pytest.raises(ValueError, match="parent role"):
            RelationshipRole(role=Role.HAS_PARTNER, max_child_age=3)

    @PROPERTY
    @given(tables=concept_frames(), factor=st.floats(min_value=1e-3, max_value=1e3))
    def test_a_scaled_amount_decodes_to_float_rounding(self, tables, factor) -> None:
        binding = _binding(
            engine_input="weekly",
            concepts=("fact:person.employment_income",),
            transform=Scale(factor=factor),
        )
        mapping = _mapping([binding])
        assert mapping.invertible_concepts() == ("fact:person.employment_income",)
        decoded = mapping.decode(mapping.encode(tables).tables)
        np.testing.assert_allclose(
            decoded["person"]["employment_income"].to_numpy(),
            tables["person"]["employment_income"].to_numpy(),
            rtol=1e-12,
            atol=1e-6,
        )

    def test_a_fraction_is_encoded_but_never_decoded(self) -> None:
        interest = ("fact:person.interest_income",)
        whole = _binding(engine_input="savings", concepts=interest)
        part = _binding(
            engine_input="isa", concepts=interest, transform=Fraction(parameter="f")
        )
        mapping = _mapping([whole, part])
        tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "person_household_id": [1],
                    "interest_income": [80.0],
                }
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        }
        out = mapping.encode(tables, shares={"f": 0.25}).tables["person"]
        assert out.loc[0, "isa"] == 20.0
        decoded = mapping.decode({"person": out, "household": tables["household"]})
        assert decoded["person"].loc[0, "interest_income"] == 80.0


class TestCoverageReport:
    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_the_surface_partitions(self, name) -> None:
        mapping = MAPPINGS[name]
        extra = [
            InputRef("uncovered_input", "person"),
            InputRef("structural", "person"),
        ]
        mapping = replace(mapping, structural_inputs=("structural",))
        report = coverage_report(mapping, [*mapping.refs(), *extra])
        covered = set(report.covered_inputs)
        structural = set(report.structural_inputs)
        uncovered = set(report.uncovered_inputs)
        assert covered | structural | uncovered == {*mapping.refs(), *extra}
        assert not covered & structural and not covered & uncovered
        assert not structural & uncovered
        assert report.covered_count + len(structural) + len(uncovered) == (
            report.input_count
        )
        assert report.unknown_inputs == ()
        assert set(report.unmapped_concepts) == set(mapping.unmapped)

    def test_mapped_inputs_the_engine_lacks_are_unknown(self) -> None:
        mapping = MAPPINGS["policyengine-us"]
        surface = [ref for ref in mapping.refs() if ref.name != "age"]
        report = coverage_report(mapping, surface)
        assert report.unknown_inputs == (InputRef("age", "person"),)
        assert report.covered_count == len(surface)

    def test_a_wrong_entity_counts_as_unknown(self) -> None:
        mapping = MAPPINGS["policyengine-us"]
        surface = [
            ref._replace(entity="tax_unit") if ref.name == "age" else ref
            for ref in mapping.refs()
        ]
        report = coverage_report(mapping, surface)
        assert InputRef("age", "person") in report.unknown_inputs
        assert InputRef("age", "tax_unit") in report.uncovered_inputs

    def test_reports_serialise_and_read_back(self) -> None:
        mapping = MAPPINGS["policyengine-uk"]
        surface = [*mapping.refs(), InputRef("z", "household")]
        report = coverage_report(mapping, surface)
        again = CoverageReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert again.to_dict() == report.to_dict()
        assert again.to_markdown() == report.to_markdown()
        shuffled = coverage_report(mapping, reversed(surface))
        assert json.dumps(shuffled.to_dict(), sort_keys=True) == json.dumps(
            report.to_dict(), sort_keys=True
        )

    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_the_committed_readable_report_matches_its_golden(self, name) -> None:
        golden = json.loads(coverage_golden(name).read_text(encoding="utf-8"))
        report = CoverageReport.from_dict(golden)
        assert report.engine == MAPPINGS[name].engine
        assert (COVERAGE_DOCS / f"{name}.md").read_text(encoding="utf-8") == (
            report.to_markdown()
        ), "Regenerate with tools/refresh_concept_coverage.py"


class TestLegalAlignments:
    @pytest.mark.parametrize("name", ["axiom-nz", "axiom-be"])
    def test_axiom_bindings_read_as_chronicle_alignments_to_law(self, name) -> None:
        mapping = MAPPINGS[name]
        alignments = mapping.legal_alignments(
            authority="microcosm", legal_vintage=mapping.engine_version
        )
        assert alignments
        single = {
            (binding.concepts[0], binding.canonical_input)
            for binding in mapping.bindings
            if len(binding.concepts) == 1
        }
        assert {(a.source_concept, a.canonical_concept) for a in alignments} == single
        for alignment in alignments:
            assert alignment.canonical_kind is CanonicalConceptKind.LEGAL
            # Subnational trees keep their own code (be-bru is Brussels).
            country = alignment.jurisdiction.split("-", 1)[0]
            assert country == name.removeprefix("axiom-")
            assert alignment.evidence_notes and alignment.issues() == ()

    @pytest.mark.parametrize("name", ["policyengine-us", "policyengine-uk"])
    def test_policyengine_inputs_carry_no_legal_id(self, name) -> None:
        assert (
            MAPPINGS[name].legal_alignments(authority="microcosm", legal_vintage="x")
            == ()
        )
