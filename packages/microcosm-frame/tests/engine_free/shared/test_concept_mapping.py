"""Concept-to-input mappings: the contract every mapping meets, and the machinery.

Every committed mapping (policyengine-us, policyengine-uk, Axiom New Zealand
and Belgium) is plain data importable without any engine, so the same
properties run over all of them here: construction rules, serialization,
round trips, and a differential test of the vectorized encoder against a
row-by-row reference. Country-specific expectations live in the
``engine_free/us`` and ``engine_free/uk`` files of the same name; whether each
mapped input exists in the installed engine is checked by the engine tests.
"""

import copy
import json
import re
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.frame import concept_mapping as concept_mapping_module
from microcosm.frame import concepts as concepts_module
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
    ConceptAlignment,
    TemporalBasis,
    TransportRule,
    Unit,
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
_LIQUID_ASSETS = "fact:person.liquid_financial_assets"
_AS_MODULE = "nz/statutes/social_security/accommodation_supplement/core.yaml"

# The round-trip and idempotence properties check exactly the concepts
# invertible_concepts() returns, so the sets are pinned here: a decoder that
# quietly stops inverting a transform shrinks a set and fails a test instead
# of dropping concepts from every round trip.
INVERTIBLE = {
    "policyengine-us": (
        "fact:person.age",
        "fact:person.sex",
        "fact:household.reference_person_id",
        "fact:person.employment_income",
        "fact:person.nonfarm_self_employment_income",
        "fact:person.farm_self_employment_income",
        "fact:person.interest_income",
        "fact:person.dividend_income",
        "fact:person.rental_income",
        "fact:person.realized_capital_gains",
        "fact:person.private_pension_income",
        "fact:person.public_pension_income",
        "fact:person.usual_weekly_hours",
        "fact:person.has_disability",
        "fact:household.rent",
        "fact:household.mortgage_interest",
        "fact:household.property_tax",
    ),
    "policyengine-uk": (
        "fact:person.age",
        "fact:person.sex",
        "fact:person.employment_income",
        "fact:person.interest_income",
        "fact:person.dividend_income",
        "fact:person.rental_income",
        "fact:person.realized_capital_gains",
        "fact:person.private_pension_income",
        "fact:person.has_disability",
        "fact:household.rent",
        "fact:household.mortgage_interest",
        "fact:household.mortgage_principal",
        "fact:household.property_tax",
    ),
    "axiom-nz": (
        "fact:person.age",
        "fact:person.employment_income",
        "fact:person.has_disability",
        "fact:person.enrolled_full_time",
    ),
    "axiom-be": (
        "fact:person.age",
        "fact:person.employment_income",
        "fact:person.interest_income",
        "fact:person.dividend_income",
        "fact:person.public_pension_income",
        "fact:person.usual_weekly_hours",
        "fact:person.has_disability",
        "fact:household.rent",
        "fact:household.mortgage_interest",
    ),
}


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
    def test_the_invertible_concepts_are_pinned(self, name) -> None:
        assert MAPPINGS[name].invertible_concepts() == INVERTIBLE[name]

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

    def test_liquid_assets_bind_only_the_nz_cash_asset_test(self) -> None:
        # Every other mapping lists the concept as unmapped. The one binding
        # is a group binding, which encode defers until units exist.
        bound = {
            name: [binding.ref for binding in mapping.bindings_for(_LIQUID_ASSETS)]
            for name, mapping in MAPPINGS.items()
        }
        assert bound == {
            "axiom-be": [],
            "axiom-nz": [
                InputRef("accommodation_supplement_cash_assets", "Family", _AS_MODULE)
            ],
            "policyengine-uk": [],
            "policyengine-us": [],
        }
        nz = MAPPINGS["axiom-nz"]
        (binding,) = nz.bindings_for(_LIQUID_ASSETS)
        assert binding.concepts == (_LIQUID_ASSETS,)
        assert isinstance(binding.transform, Identity)
        assert binding.group_rule is GroupRule.SUM_OVER_MEMBERS
        assert binding.relation is AlignmentRelation.APPROXIMATE
        assert not nz.is_executable(binding)

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
    def test_liquid_assets_reach_no_executed_input(self, name, tables, data) -> None:
        # Bound only through a deferred group binding, the concept changes no
        # encoded column, and no decode recovers it.
        mapping = MAPPINGS[name]
        parameters = _parameters(mapping, data)
        with_assets = mapping.encode(tables, **parameters)
        without = {
            "person": tables["person"].drop(columns="liquid_financial_assets"),
            "household": tables["household"],
        }
        without_assets = mapping.encode(without, **parameters)
        for entity in ("person", "household"):
            pd.testing.assert_frame_equal(
                with_assets.tables[entity], without_assets.tables[entity]
            )
        if name == "axiom-nz":
            assert "accommodation_supplement_cash_assets" in {
                binding.engine_input for binding in with_assets.deferred
            }
        decoded = mapping.decode(with_assets.tables)
        assert "liquid_financial_assets" not in decoded["person"].columns

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

    def test_a_household_without_a_reference_person_is_refused(self) -> None:
        mapping = _mapping(
            [
                _binding(
                    engine_input="head",
                    concepts=("fact:household.reference_person_id",),
                    transform=RelationshipRole(role=Role.REFERENCE_PERSON),
                )
            ]
        )
        # Ids beyond 2**53 would round if the missing household turned the
        # lookup into floats, so the refusal must come before any upcast.
        tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [2**53 + 1, 2**53 + 3],
                    "person_household_id": [1, 2],
                    "head": [True, False],
                }
            ),
            "household": pd.DataFrame({"household_id": [1, 2]}),
        }
        with pytest.raises(ValueError, match="no reference person"):
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
            (
                {
                    "transform": Product(),
                    "concepts": ("fact:person.usual_weekly_hours",),
                },
                r"^Binding for 'x': a product reads two concepts\.$",
            ),
            (
                {
                    "transform": Product(),
                    "concepts": (
                        "fact:person.usual_weekly_hours",
                        "fact:person.weeks_worked",
                        "fact:person.age",
                    ),
                },
                r"^Binding for 'x': a product reads two concepts\.$",
            ),
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


#: The transforms that do arithmetic on a year's amount, and so read annual
#: flows only.
_ARITHMETIC = (Scale, Sum, Share, Fraction)
_NOT_FLOWS = tuple(
    basis for basis in TemporalBasis if basis is not TemporalBasis.ANNUAL_FLOW
)
_COMMITTED = tuple(
    (name, binding)
    for name, mapping in sorted(MAPPINGS.items())
    for binding in mapping.bindings
)
_parameter_names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_.", min_size=1)
_F6_NOTE = "microcosm#1120 review F6 probe."


def _passes_type_checks(kind: type, concept_id: str) -> bool:
    """Whether a ``kind`` binding accepts ``concept_id`` on all but its basis.

    A scale needs a float; a sum, share or fraction needs an amount. These
    are the binding checks that ran before the flow rule existed.
    """
    item = concept(concept_id)
    return item.dtype == "float" if kind is Scale else item.monetary is not None


#: Each concept that is not an annual flow, under each arithmetic transform
#: whose type checks it passes, with an annual flow to sum it with.
_REFUSED_READS = tuple(
    (kind, item.id)
    for kind in _ARITHMETIC
    for item in CONCEPTS
    if item.temporal_basis is not TemporalBasis.ANNUAL_FLOW
    and _passes_type_checks(kind, item.id)
)
_EXAMPLE_TRANSFORMS = {
    Scale: Scale(factor=1 / 52),
    Sum: Sum(),
    Share: Share(parameter="p"),
    Fraction: Fraction(parameter="f"),
}


def _arithmetic(kind: type):
    """Instances of the arithmetic transform ``kind``."""
    if kind is Scale:
        return st.floats(min_value=1e-3, max_value=1e3).map(
            lambda factor: Scale(factor=factor)
        )
    if kind is Share:
        return st.builds(Share, parameter=_parameter_names, complement=st.booleans())
    if kind is Fraction:
        return st.builds(Fraction, parameter=_parameter_names)
    return st.just(Sum())


def _refusal(kind: type, concept_id: str, engine_input: str = "x") -> str:
    """The flow rule's whole message for a ``kind`` binding on ``concept_id``."""
    name = transform_to_dict(_EXAMPLE_TRANSFORMS[kind])["kind"]
    basis = concept(concept_id).temporal_basis.value
    return (
        rf"^Binding for {re.escape(repr(engine_input))}: a {name} reads annual "
        rf"flows only, and {re.escape(concept_id)} has temporal basis {basis}\.$"
    )


@contextmanager
def _redeclared(concept_id: str, basis: TemporalBasis | None = None, **fields):
    """Run with ``concept_id`` redeclared, and nothing else.

    ``basis`` sets its temporal basis. An annual flow also takes period
    ``year``, which every annual flow must have; no other basis changes the
    period. ``fields`` change other declared fields.
    """
    if basis is not None:
        fields["temporal_basis"] = basis
        if basis is TemporalBasis.ANNUAL_FLOW:
            fields.setdefault("period", "year")
    registry = dict(concepts_module.CONCEPT_BY_ID)
    registry[concept_id] = replace(registry[concept_id], **fields)
    patched = MappingProxyType(registry)
    with pytest.MonkeyPatch.context() as patch:
        # concept() reads the first; the mapping module imported the second.
        patch.setattr(concepts_module, "CONCEPT_BY_ID", patched)
        patch.setattr(concept_mapping_module, "CONCEPT_BY_ID", patched)
        yield


def _us_reading_the_stock(*bindings) -> dict:
    """The US mapping's JSON form, edited as review finding F6 edited it.

    Each ``(engine input, concepts, transform)`` becomes a person binding,
    and the stock leaves ``unmapped``.
    """
    edited = MAPPINGS["policyengine-us"].to_dict()
    for engine_input, concepts, transform in bindings:
        edited["bindings"].append(
            {
                "engine_input": engine_input,
                "engine_entity": "person",
                "concepts": list(concepts),
                "transform": transform_to_dict(transform),
                "relation": "approximate",
                "note": _F6_NOTE,
            }
        )
    del edited["unmapped"][_LIQUID_ASSETS]
    return edited


_HOURS = "fact:person.usual_weekly_hours"
_WEEKS = "fact:person.weeks_worked"
_CONCEPT_ID = re.compile(r"fact:[a-z]+\.[a-z0-9_]+")
#: Units of the numeric quantities that are neither amounts nor pointers, so
#: their dtype and unit can be redeclared freely.
_QUANTITIES = (Unit.YEARS, Unit.HOURS_PER_WEEK, Unit.WEEKS, Unit.UNIT_INTERVAL)


def _faults(first, second) -> set[str]:
    """The ids that keep a product of ``first`` and ``second`` from validating.

    The rule as stated: a product reads one usual rate held as a float and one
    annual flow counted in weeks (an int or a float), in either order. An
    empty set means the product validates.
    """
    items = (first, second)
    faults = {
        item.id
        for item in items
        if item.temporal_basis
        not in (TemporalBasis.USUAL_RATE, TemporalBasis.ANNUAL_FLOW)
    }
    if first.temporal_basis is second.temporal_basis:
        faults |= {first.id, second.id}
    for item in items:
        if item.temporal_basis is TemporalBasis.USUAL_RATE and item.dtype != "float":
            faults.add(item.id)
        if item.temporal_basis is TemporalBasis.ANNUAL_FLOW and (
            item.unit is not Unit.WEEKS or item.dtype not in ("int", "float")
        ):
            faults.add(item.id)
    return faults


def _product(*concepts: str) -> InputBinding:
    return _binding(concepts=concepts, transform=Product())


def _uk_hours_reading(concepts: tuple[str, str]) -> dict:
    """policyengine-uk's JSON form, with ``hours_worked`` reading ``concepts``.

    Whatever the product now reads leaves ``unmapped``; whatever it no longer
    reads, and no other binding reads, joins it.
    """
    mapping = MAPPINGS["policyengine-uk"]
    edited = mapping.to_dict()
    (index,) = (
        position
        for position, binding in enumerate(mapping.bindings)
        if binding.engine_input == "hours_worked"
    )
    edited["bindings"][index]["concepts"] = list(concepts)
    for concept_id in concepts:
        edited["unmapped"].pop(concept_id, None)
    others = {
        concept_id
        for position, binding in enumerate(mapping.bindings)
        if position != index
        for concept_id in binding.reads
    }
    for concept_id in set(mapping.bindings[index].concepts) - set(concepts) - others:
        edited["unmapped"][concept_id] = "probe"
    return edited


class TestArithmeticReadsAnnualFlows:
    """A scale, sum, share or fraction reads annual flows only.

    microcosm#1120 added the first stock, ``liquid_financial_assets``. Its
    review (finding F6) bound the US donor's ``bank_account_assets`` to it,
    cleared it from ``unmapped``, and found that both ``Scale(1/52)`` and
    ``Sum(interest_income, liquid_financial_assets)`` validated. A stock has
    no weekly value and cannot be added to a year's income, so every scale,
    sum, share or fraction binding now checks the declared temporal basis of
    each concept it computes from. The household reference person that a
    binding allocated to the reference unit also reads, to place its value,
    is not an operand.

    A product has a rule of its own. Its one committed use, policyengine-uk's
    ``hours_worked``, multiplies usual weekly hours by weeks worked, yet a
    product of the stock and weeks worked validated too. A product now reads
    exactly one usual rate, held as a float, and one annual flow counted in
    weeks, in either order. Other transforms are outside both rules.
    """

    @PROPERTY
    @given(data=st.data())
    def test_arithmetic_on_anything_but_an_annual_flow_is_refused(self, data) -> None:
        kind = data.draw(st.sampled_from(_ARITHMETIC), label="kind")
        transform = data.draw(_arithmetic(kind), label="transform")
        eligible = [item for item in CONCEPTS if _passes_type_checks(kind, item.id)]
        refused = data.draw(
            st.sampled_from(
                [
                    item
                    for item in eligible
                    if item.temporal_basis is not TemporalBasis.ANNUAL_FLOW
                ]
            ),
            label="refused",
        )
        flows = [
            item.id
            for item in eligible
            if item.temporal_basis is TemporalBasis.ANNUAL_FLOW
            and item.entity == refused.entity
        ]
        concepts = [refused.id]
        if kind is Sum:
            others = data.draw(
                # Leave one flow out, to stand in for the refused concept below.
                st.lists(
                    st.sampled_from(flows),
                    min_size=1,
                    max_size=len(flows) - 1,
                    unique=True,
                ),
                label="others",
            )
            concepts = data.draw(
                st.permutations([refused.id, *others]), label="concepts"
            )
        with pytest.raises(ValueError, match="annual flows only") as error:
            _binding(concepts=tuple(concepts), transform=transform)
        assert refused.id in str(error.value)
        # Put an annual flow in its place, and the same binding validates.
        stand_in = data.draw(
            st.sampled_from([item for item in flows if item not in concepts]),
            label="stand_in",
        )
        swapped = tuple(stand_in if item == refused.id else item for item in concepts)
        assert _binding(concepts=swapped, transform=transform).concepts == swapped

    @pytest.mark.parametrize(
        ("kind", "concept_id"),
        _REFUSED_READS,
        ids=[f"{kind.__name__}-{concept_id}" for kind, concept_id in _REFUSED_READS],
    )
    def test_each_concept_that_is_not_an_annual_flow_is_refused_by_name(
        self, kind, concept_id
    ) -> None:
        # Today: a scale of the stock, usual weekly hours or the take-up seed,
        # and a sum, share or fraction of the stock.
        concepts = (concept_id,)
        if kind is Sum:
            concepts = ("fact:person.interest_income", concept_id)
        with pytest.raises(ValueError, match=_refusal(kind, concept_id)):
            _binding(concepts=concepts, transform=_EXAMPLE_TRANSFORMS[kind])
        with _redeclared(concept_id, TemporalBasis.ANNUAL_FLOW):
            _binding(concepts=concepts, transform=_EXAMPLE_TRANSFORMS[kind])

    def test_redeclaring_any_operand_of_any_committed_arithmetic_binding_refuses_it(
        self,
    ) -> None:
        # Exhaustive: every committed scale, sum, share and fraction binding,
        # every concept it computes from, and every temporal basis that is not
        # an annual flow. Changing only that concept's declared basis refuses
        # the binding, so the rule reads the declaration. Redeclaring a
        # placement pointer the binding also reads leaves it valid.
        arithmetic = [
            (name, binding)
            for name, binding in _COMMITTED
            if isinstance(binding.transform, _ARITHMETIC)
        ]
        assert arithmetic
        placed = 0
        for name, binding in arithmetic:
            assert replace(binding) == binding
            for concept_id in binding.concepts:
                for basis in _NOT_FLOWS:
                    with (
                        _redeclared(concept_id, basis),
                        pytest.raises(ValueError, match="annual flows only") as error,
                    ):
                        replace(binding)
                    assert concept_id in str(error.value), (name, binding.ref)
            for pointer in set(binding.reads) - set(binding.concepts):
                placed += 1
                for basis in _NOT_FLOWS:
                    with _redeclared(pointer, basis):
                        assert replace(binding) == binding, (name, binding.ref)
        assert placed

    def test_redeclaring_any_read_of_any_other_committed_binding_keeps_it(
        self,
    ) -> None:
        # The complement, also exhaustive: no other committed binding depends
        # on the basis of anything it reads (identities, roles, child counts,
        # take-up, recodes, predicates, positivity tests and allocations),
        # whichever basis it is redeclared under. The hours product used to be
        # one of them; it now has a rule of its own, and redeclaring either of
        # its operands refuses it (see the product tests below).
        others = [
            (name, binding)
            for name, binding in _COMMITTED
            if not isinstance(binding.transform, (*_ARITHMETIC, Product))
        ]
        kinds = {type(binding.transform) for _, binding in others}
        assert {Identity, AllocateToReferencePerson, Positive} <= kinds
        for name, binding in others:
            for concept_id in binding.reads:
                for basis in TemporalBasis:
                    with _redeclared(concept_id, basis):
                        assert replace(binding) == binding, (name, binding.ref, basis)

    @PROPERTY
    @given(name=mapping_names, data=st.data())
    def test_a_mapping_feeding_anything_but_an_annual_flow_to_arithmetic_is_refused(
        self, name, data
    ) -> None:
        # The F6 edit generalised to every mapping and its JSON form: point an
        # arithmetic input at a concept that is not an annual flow, and take
        # that concept out of ``unmapped``. The edit is applied to every
        # binding that must stay alike (a share's complement, the same input
        # in another module), and only edits that would validate if the
        # concept were an annual flow are kept, so the flow rule is the one
        # obstacle.
        mapping = MAPPINGS[name]
        index = data.draw(
            st.sampled_from(
                [
                    position
                    for position, binding in enumerate(mapping.bindings)
                    if isinstance(binding.transform, _ARITHMETIC)
                ]
            ),
            label="binding",
        )
        binding = mapping.bindings[index]
        kind = type(binding.transform)
        refused = data.draw(
            st.sampled_from(
                [concept_id for reader, concept_id in _REFUSED_READS if reader is kind]
            ),
            label="refused",
        )
        replaced = data.draw(st.sampled_from(binding.concepts), label="replaced")

        def alike(other: InputBinding) -> bool:
            if isinstance(binding.transform, Share):
                return (
                    isinstance(other.transform, Share)
                    and other.transform.parameter == binding.transform.parameter
                    and other.module == binding.module
                )
            return (other.engine_input, other.engine_entity) == (
                binding.engine_input,
                binding.engine_entity,
            )

        edited = mapping.to_dict()
        for position, other in enumerate(mapping.bindings):
            if alike(other):
                edited["bindings"][position]["concepts"] = [
                    refused if item == replaced else item for item in other.concepts
                ]
        edited["unmapped"].pop(refused, None)
        if not any(
            replaced in other.reads for other in mapping.bindings if not alike(other)
        ):
            edited["unmapped"][replaced] = "probe"
        with _redeclared(refused, TemporalBasis.ANNUAL_FLOW):
            try:
                ConceptMapping.from_dict(edited)
            except ValueError:
                assume(False)
        with pytest.raises(ValueError, match="annual flows only"):
            ConceptMapping.from_dict(edited)

    @pytest.mark.parametrize(
        "bindings",
        [
            [("bank_account_assets", (_LIQUID_ASSETS,), Scale(factor=1 / 52))],
            [
                (
                    "bank_account_assets",
                    ("fact:person.interest_income", _LIQUID_ASSETS),
                    Sum(),
                )
            ],
            [
                ("bank_account_assets", (_LIQUID_ASSETS,), Share(parameter="p")),
                (
                    "stock_assets",
                    (_LIQUID_ASSETS,),
                    Share(parameter="p", complement=True),
                ),
            ],
            [("bank_account_assets", (_LIQUID_ASSETS,), Fraction(parameter="f"))],
        ],
        ids=["scale", "sum", "share-pair", "fraction"],
    )
    def test_the_review_counterexamples_and_their_analogues_are_refused(
        self, bindings
    ) -> None:
        # The scale and the sum are F6's own; the share pair and the fraction
        # are the same edit through the other two transforms. Each validates
        # if the stock is declared an annual flow.
        engine_input, _, transform = bindings[0]
        refusal = _refusal(type(transform), _LIQUID_ASSETS, engine_input)
        with pytest.raises(ValueError, match=refusal):
            ConceptMapping.from_dict(_us_reading_the_stock(*bindings))
        with _redeclared(_LIQUID_ASSETS, TemporalBasis.ANNUAL_FLOW):
            ConceptMapping.from_dict(_us_reading_the_stock(*bindings))

    @pytest.mark.parametrize("transform", [Identity(), Positive()], ids=repr)
    def test_a_stock_still_reaches_an_engine_unchanged_or_through_a_positivity_test(
        self, transform
    ) -> None:
        mapping = ConceptMapping.from_dict(
            _us_reading_the_stock(("bank_account_assets", (_LIQUID_ASSETS,), transform))
        )
        assert [item.transform for item in mapping.bindings_for(_LIQUID_ASSETS)] == [
            transform
        ]

    @pytest.mark.parametrize("name", sorted(MAPPINGS))
    def test_every_committed_mapping_validates_and_its_arithmetic_reads_flows(
        self, name
    ) -> None:
        # MAPPINGS is built under the rule when this module loads; rebuilding
        # from JSON here states that every committed mapping still validates.
        mapping = MAPPINGS[name]
        assert ConceptMapping.from_dict(json.loads(json.dumps(mapping.to_dict()))) == (
            mapping
        )
        arithmetic = [
            binding
            for binding in mapping.bindings
            if isinstance(binding.transform, _ARITHMETIC)
        ]
        assert arithmetic
        for binding in arithmetic:
            for concept_id in binding.concepts:
                assert (
                    concept(concept_id).temporal_basis is TemporalBasis.ANNUAL_FLOW
                ), (binding.ref, concept_id)

    @PROPERTY
    @given(data=st.data())
    def test_a_product_validates_exactly_when_it_reads_a_float_rate_and_weeks(
        self, data
    ) -> None:
        # Start from the committed hours product. Each operand may be swapped
        # for any concept, and its basis, dtype and unit redeclared. In either
        # order, the binding validates exactly when the rule's text says it
        # should, and a refusal names only operands that break the rule.
        drawn = []
        for slot in (_HOURS, _WEEKS):
            concept_id = data.draw(
                st.one_of(st.just(slot), st.sampled_from([c.id for c in CONCEPTS])),
                label="concept",
            )
            basis = data.draw(
                st.one_of(st.none(), st.sampled_from(TemporalBasis)), label="basis"
            )
            fields = {}
            if concept(concept_id).unit in _QUANTITIES:
                dtype = data.draw(
                    st.one_of(st.none(), st.sampled_from(("float", "int", "str"))),
                    label="dtype",
                )
                unit = data.draw(
                    st.one_of(st.none(), st.sampled_from(_QUANTITIES)), label="unit"
                )
                if dtype is not None:
                    fields["dtype"] = dtype
                if dtype == "str":
                    fields.update(lower=None, upper=None)
                if unit is not None:
                    fields["unit"] = unit
            drawn.append((concept_id, basis, fields))
        (first, *_), (second, *_) = drawn
        assume(first != second)
        with ExitStack() as stack:
            for concept_id, basis, fields in drawn:
                stack.enter_context(_redeclared(concept_id, basis, **fields))
            faults = _faults(concept(first), concept(second))
            for order in ((first, second), (second, first)):
                if not faults:
                    assert _product(*order).concepts == order
                    continue
                with pytest.raises(
                    ValueError, match=r"^Binding for 'x': a product"
                ) as error:
                    _product(*order)
                named = set(_CONCEPT_ID.findall(str(error.value)))
                assert named and named <= faults, (order, faults, named)

    def test_of_all_committed_concepts_only_hours_and_weeks_multiply(self) -> None:
        # Exhaustive: every ordered pair of distinct committed concepts.
        valid = set()
        for first in CONCEPTS:
            for second in CONCEPTS:
                if first.id == second.id:
                    continue
                pair = (first.id, second.id)
                try:
                    _product(*pair)
                except ValueError as error:
                    named = set(_CONCEPT_ID.findall(str(error)))
                    assert str(error).startswith("Binding for 'x': a product"), pair
                    assert named and named <= _faults(first, second), (pair, named)
                else:
                    assert not _faults(first, second), pair
                    valid.add(pair)
        assert valid == {(_HOURS, _WEEKS), (_WEEKS, _HOURS)}

    @pytest.mark.parametrize(
        ("concepts", "setup", "message", "fix"),
        [
            pytest.param(
                (_LIQUID_ASSETS, _WEEKS),
                [],
                "a product reads a usual rate and an annual flow, and "
                "fact:person.liquid_financial_assets has temporal basis "
                "reference_state.",
                [(_LIQUID_ASSETS, TemporalBasis.USUAL_RATE, {})],
                id="stock-times-weeks",
            ),
            pytest.param(
                (_WEEKS, _LIQUID_ASSETS),
                [],
                "a product reads a usual rate and an annual flow, and "
                "fact:person.liquid_financial_assets has temporal basis "
                "reference_state.",
                [(_LIQUID_ASSETS, TemporalBasis.USUAL_RATE, {})],
                id="weeks-times-stock",
            ),
            pytest.param(
                ("fact:person.take_up_seed", _WEEKS),
                [],
                "a product reads a usual rate and an annual flow, and "
                "fact:person.take_up_seed has temporal basis persistent.",
                [("fact:person.take_up_seed", TemporalBasis.USUAL_RATE, {})],
                id="seed-times-weeks",
            ),
            pytest.param(
                (_HOURS, "fact:person.age"),
                [],
                "a product reads a usual rate and an annual flow, and "
                "fact:person.age has temporal basis reference_state.",
                [("fact:person.age", TemporalBasis.ANNUAL_FLOW, {"unit": Unit.WEEKS})],
                id="hours-times-age",
            ),
            pytest.param(
                ("fact:person.employment_income", _WEEKS),
                [],
                "a product reads one usual rate and one annual flow, and "
                "fact:person.employment_income and fact:person.weeks_worked both "
                "have temporal basis annual_flow.",
                [("fact:person.employment_income", TemporalBasis.USUAL_RATE, {})],
                id="income-times-weeks",
            ),
            pytest.param(
                (_HOURS, _WEEKS),
                [(_WEEKS, TemporalBasis.USUAL_RATE, {})],
                "a product reads one usual rate and one annual flow, and "
                "fact:person.usual_weekly_hours and fact:person.weeks_worked both "
                "have temporal basis usual_rate.",
                [(_WEEKS, TemporalBasis.ANNUAL_FLOW, {})],
                id="two-rates",
            ),
            pytest.param(
                (_HOURS, "fact:person.employment_income"),
                [],
                "a product multiplies a usual rate by a number of weeks, and "
                "fact:person.employment_income has unit base_currency and dtype "
                "float.",
                [
                    (
                        "fact:person.employment_income",
                        None,
                        {
                            "unit": Unit.WEEKS,
                            "monetary": None,
                            "transport": TransportRule.CARRY,
                        },
                    )
                ],
                id="hours-times-income",
            ),
            pytest.param(
                (_HOURS, _WEEKS),
                [(_HOURS, None, {"dtype": "int"})],
                "a product's usual rate is a float, and "
                "fact:person.usual_weekly_hours has dtype int.",
                [(_HOURS, None, {"dtype": "float"})],
                id="whole-number-rate",
            ),
            pytest.param(
                (_WEEKS, _HOURS),
                [(_WEEKS, None, {"unit": Unit.HOURS_PER_WEEK})],
                "a product multiplies a usual rate by a number of weeks, and "
                "fact:person.weeks_worked has unit hours_per_week and dtype int.",
                [(_WEEKS, None, {"unit": Unit.WEEKS})],
                id="weeks-in-hours",
            ),
            pytest.param(
                (_HOURS, _WEEKS),
                [(_WEEKS, None, {"dtype": "str", "lower": None, "upper": None})],
                "a product multiplies a usual rate by a number of weeks, and "
                "fact:person.weeks_worked has unit weeks and dtype str.",
                [(_WEEKS, None, {"dtype": "int"})],
                id="weeks-as-text",
            ),
            pytest.param(
                (_HOURS, _WEEKS),
                [(_HOURS, None, {"dtype": "str", "lower": None, "upper": None})],
                "a product's usual rate is a float, and "
                "fact:person.usual_weekly_hours has dtype str.",
                [(_HOURS, None, {"dtype": "float", "lower": 0.0, "upper": 168.0})],
                id="rate-as-text",
            ),
            pytest.param(
                (_HOURS, _WEEKS),
                [(_WEEKS, None, {"unit": Unit.YEARS})],
                "a product multiplies a usual rate by a number of weeks, and "
                "fact:person.weeks_worked has unit years and dtype int.",
                [(_WEEKS, None, {"unit": Unit.WEEKS})],
                id="weeks-in-years",
            ),
            pytest.param(
                (_HOURS, _WEEKS),
                [(_WEEKS, None, {"unit": Unit.UNIT_INTERVAL})],
                "a product multiplies a usual rate by a number of weeks, and "
                "fact:person.weeks_worked has unit unit_interval and dtype int.",
                [(_WEEKS, None, {"unit": Unit.WEEKS})],
                id="weeks-in-unit-interval",
            ),
        ],
    )
    def test_each_way_a_product_breaks_the_rule_is_refused_by_name(
        self, concepts, setup, message, fix
    ) -> None:
        # The whole message is matched. Each case validates once the field at
        # fault is redeclared to fit, so the rule is the only obstacle.
        with ExitStack() as stack:
            for concept_id, basis, fields in setup:
                stack.enter_context(_redeclared(concept_id, basis, **fields))
            refusal = "^" + re.escape(f"Binding for 'x': {message}") + "$"
            with pytest.raises(ValueError, match=refusal):
                _product(*concepts)
            for concept_id, basis, fields in fix:
                stack.enter_context(_redeclared(concept_id, basis, **fields))
            assert _product(*concepts).concepts == concepts

    def test_redeclaring_either_operand_of_a_committed_product_refuses_it(
        self,
    ) -> None:
        # Exhaustive: every committed product, each operand, and every basis.
        # Any basis but the operand's own refuses the binding and names the
        # operand. This is the intended change to the complement test above,
        # which used to keep the hours product valid under every basis it
        # tried (all but annual_flow).
        products = [
            (name, binding)
            for name, binding in _COMMITTED
            if isinstance(binding.transform, Product)
        ]
        assert products
        for name, binding in products:
            assert not _faults(*map(concept, binding.concepts)), (name, binding.ref)
            for concept_id in binding.concepts:
                own = concept(concept_id).temporal_basis
                for basis in TemporalBasis:
                    with _redeclared(concept_id, basis):
                        if basis is own:
                            assert replace(binding) == binding, (name, binding.ref)
                            continue
                        with pytest.raises(ValueError, match="a product") as error:
                            replace(binding)
                    assert concept_id in str(error.value), (name, binding.ref, basis)

    @pytest.mark.parametrize("slot", [0, 1], ids=["rate", "weeks"])
    def test_the_uk_hours_product_refuses_every_other_person_concept(
        self, slot
    ) -> None:
        # Exhaustive over policyengine-uk's JSON form: each operand of
        # hours_worked swapped for each other person concept, which leaves
        # ``unmapped``. The product rule refuses every edit and names the
        # newcomer.
        committed = MAPPINGS["policyengine-uk"].bindings_for(_HOURS)[0].concepts
        assert committed == (_HOURS, _WEEKS)
        swapped = 0
        for item in CONCEPTS:
            if item.entity != "person" or item.id in committed:
                continue
            concepts = list(committed)
            concepts[slot] = item.id
            with pytest.raises(ValueError, match="'hours_worked': a product") as error:
                ConceptMapping.from_dict(_uk_hours_reading(tuple(concepts)))
            assert item.id in str(error.value)
            swapped += 1
        assert swapped == sum(item.entity == "person" for item in CONCEPTS) - 2

    def test_the_stock_times_weeks_worked_is_refused_in_the_uk_mapping(self) -> None:
        # The case that prompted the rule, in a mapping's JSON form. It
        # validates if the stock is declared a usual rate.
        edited = _uk_hours_reading((_LIQUID_ASSETS, _WEEKS))
        refusal = (
            r"^Binding for 'hours_worked': a product reads a usual rate and an "
            r"annual flow, and fact:person\.liquid_financial_assets has temporal "
            r"basis reference_state\.$"
        )
        with pytest.raises(ValueError, match=refusal):
            ConceptMapping.from_dict(edited)
        with _redeclared(_LIQUID_ASSETS, TemporalBasis.USUAL_RATE):
            ConceptMapping.from_dict(edited)


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


# --- Malformed serialized input --------------------------------------------

_JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=6),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(st.text(max_size=6), inner, max_size=3)
    ),
    max_leaves=6,
)


def _paths(value, prefix=()):
    yield prefix
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _paths(item, (*prefix, index))


@st.composite
def _corrupted(draw, record):
    """``record`` with one value replaced, one field deleted or one added."""
    data = copy.deepcopy(record)
    path = draw(st.sampled_from(list(_paths(data))))
    action = draw(st.sampled_from(("replace", "delete", "add")))
    if not path or action == "replace":
        if not path:
            return draw(_JSON)
        parent = data
        for step in path[:-1]:
            parent = parent[step]
        parent[path[-1]] = draw(_JSON)
        return data
    parent = data
    for step in path[:-1]:
        parent = parent[step]
    if action == "delete" and isinstance(parent, dict):
        del parent[path[-1]]
    elif isinstance(parent, dict):
        parent[draw(st.text(min_size=1, max_size=6))] = draw(_JSON)
    else:
        parent[path[-1]] = draw(_JSON)
    return data


def _alignment_record() -> dict:
    return ConceptAlignment(
        canonical_concept="nz:statutes/income_tax#input.x",
        source_concept="fact:person.age",
        relation="exact",
        authority="axiom",
        evidence_notes="Engine label.",
        legal_vintage="2026-09-01",
    ).to_dict()


_READERS = {
    **{
        f"mapping:{name}": (ConceptMapping.from_dict, MAPPINGS[name].to_dict())
        for name in sorted(MAPPINGS)
    },
    **{
        f"report:{name}": (
            CoverageReport.from_dict,
            json.loads(coverage_golden(name).read_text(encoding="utf-8")),
        )
        for name in sorted(MAPPINGS)
    },
    "alignment": (ConceptAlignment.from_dict, _alignment_record()),
    **{
        f"transform:{kind}": (transform_from_dict, transform_to_dict(transform))
        for kind, transform in (
            ("recode", Recode(pairs=(("female", "F"), ("male", "M")))),
            ("predicate", Predicate(clauses=(("fact:person.sex", ("female",)),))),
            ("scale", Scale(factor=1 / 52)),
            ("role", RelationshipRole(role=Role.PARENT_OF_CORESIDENT_CHILD)),
        )
    },
}


class TestMalformedInput:
    """Every reader refuses malformed input with ValueError, and nothing else."""

    @pytest.mark.parametrize("reader", sorted(_READERS))
    @settings(max_examples=60, deadline=None)
    @given(data=st.data())
    def test_corrupted_records_raise_only_value_errors(self, reader, data) -> None:
        read, record = _READERS[reader]
        corrupted = data.draw(_corrupted(json.loads(json.dumps(record))))
        try:
            read(corrupted)
        except ValueError:
            pass

    @pytest.mark.parametrize(
        ("reader", "change", "message"),
        [
            (
                "mapping:policyengine-uk",
                lambda d: d.pop("bindings"),
                r"lacks \['bindings'\]",
            ),
            (
                "mapping:policyengine-uk",
                lambda d: d.update(bindngs=d.pop("bindings")),
                r"unexpected fields \['bindngs'\]",
            ),
            (
                "mapping:policyengine-uk",
                lambda d: d["bindings"][0].update(concepts=5),
                "'concepts' must list concept ids",
            ),
            (
                "mapping:policyengine-uk",
                lambda d: d["bindings"][0].update(transform="identity"),
                "transform must be an object",
            ),
            (
                "mapping:policyengine-uk",
                lambda d: d["bindings"][0].update(engine_input=7),
                "'engine_input' must be text",
            ),
            (
                "mapping:policyengine-uk",
                lambda d: d.update(unmapped=[]),
                "'unmapped' must map",
            ),
            ("report:policyengine-uk", lambda d: d.pop("covered_inputs"), "lacks"),
            (
                "report:policyengine-uk",
                lambda d: d["covered_inputs"][0].update(nmae="x"),
                r"input reference has unexpected fields \['nmae'\]",
            ),
            (
                "report:policyengine-uk",
                lambda d: d.update(input_count="9"),
                "'input_count'",
            ),
            (
                "report:policyengine-uk",
                lambda d: d.update(input_count=-3),
                "'input_count' must be a non-negative integer",
            ),
            (
                "report:policyengine-uk",
                lambda d: d.update(unmapped_concepts={"fact:person.age": 5}),
                "'unmapped_concepts' must map text to text",
            ),
            (
                "alignment",
                lambda d: d.update(canonical_concept=7),
                "'canonical_concept' must be text",
            ),
        ],
    )
    def test_malformed_fields_are_named(self, reader, change, message) -> None:
        read, record = _READERS[reader]
        data = json.loads(json.dumps(record))
        change(data)
        with pytest.raises(ValueError, match=message):
            read(data)

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            ("identity", "must be an object"),
            ({"kind": "predicate", "clauses": 5}, "'clauses' must list pairs"),
            (
                {"kind": "predicate", "clauses": [[1, ["x"]]]},
                "pair concept ids with value lists",
            ),
            ({"kind": "recode", "pairs": [["a"]]}, "'pairs' must list pairs"),
            ({"kind": "scale", "factor": 10**400}, "Malformed transform"),
        ],
    )
    def test_malformed_transform_shapes_are_refused(self, data, message) -> None:
        with pytest.raises(ValueError, match=message):
            transform_from_dict(data)
