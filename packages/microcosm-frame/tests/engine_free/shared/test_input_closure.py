"""The input closure: every root input of a bound module closed exactly once.

The closure's checks run against a toy mapping and a toy engine surface, so
each way a closure can disagree with them is shown to be reported; the
country closures are checked against their real surfaces where they live.
"""

import json
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame.adapters.axiom_input_surface import (
    AxiomInputSurface,
    AxiomModuleSurface,
)
from microcosm.frame.concept_mapping import (
    Allocation,
    ConceptMapping,
    GroupRule,
    Identity,
    InputBinding,
    InputDeclaration,
    InputRef,
    StateBinding,
    StateTest,
)
from microcosm.frame.concepts import CONCEPTS, AlignmentRelation
from microcosm.frame.input_closure import (
    INPUT_CLOSURE_FORMAT,
    ClosureClass,
    ClosureEntry,
    EncodedBy,
    InputClosure,
    Knob,
    KnobKind,
    UndeterminedAction,
    UndeterminedPolicy,
    apply_defaults,
    resolve_judgments,
)

MODULE = "zz/m.yaml"
SHA = "0" * 64
SURFACE = AxiomInputSurface(
    country="zz",
    rulespec_commit="r1",
    engine_commit="e1",
    engine_surface="dense_root_inputs",
    modules=MappingProxyType(
        {
            MODULE: AxiomModuleSurface(
                path=MODULE,
                sha256=SHA,
                status="compiled",
                inputs=MappingProxyType(
                    {"Family": ("age", "area_1", "bridge", "flag")}
                ),
                canonical_inputs=MappingProxyType({}),
            )
        }
    ),
)
BOUND = InputBinding(
    engine_input="age",
    engine_entity="Family",
    concepts=("fact:person.age",),
    transform=Identity(),
    relation=AlignmentRelation.APPROXIMATE,
    note="evidence",
    group_rule=GroupRule.REFERENCE_MEMBER,
    module=MODULE,
    canonical_input="zz:m#input.age",
)
MAPPING = ConceptMapping(
    engine="axiom:zz",
    engine_version="r1",
    entity_correspondence={"person": "Person", "household": "Household"},
    input_declaration=InputDeclaration.USAGE_INFERRED,
    bindings=(BOUND,),
    unmapped={item.id: "x" for item in CONCEPTS if item.id != "fact:person.age"},
)
AREA = StateBinding(
    engine_input="area_1",
    engine_entity="Family",
    columns=("household.area",),
    test=StateTest.EQUALS,
    value=1,
    group_rule=GroupRule.HOUSEHOLD_VALUE,
    note="evidence",
    module=MODULE,
)
ENTRIES = (
    ClosureEntry(
        module=MODULE,
        entity="Family",
        input="age",
        closure_class=ClosureClass.ENCODED,
        encoded_by=EncodedBy.CONCEPT_BINDING,
    ),
    ClosureEntry(
        module=MODULE,
        entity="Family",
        input="area_1",
        closure_class=ClosureClass.ENCODED,
        encoded_by=EncodedBy.STATE_BINDING,
        state_binding=AREA,
        reason="model state",
    ),
    ClosureEntry(
        module=MODULE,
        entity="Family",
        input="bridge",
        closure_class=ClosureClass.ENCODED,
        encoded_by=EncodedBy.GRAPH_NODE,
        node="zz.bridge",
        reason="engine-computed elsewhere",
    ),
    ClosureEntry(
        module=MODULE,
        entity="Family",
        input="flag",
        closure_class=ClosureClass.DEFAULTED,
        value=False,
        reason="not observed",
        citation="Act s 1",
        new_default=True,
    ),
)
KNOBS = (
    Knob(
        name="area_column",
        kind=KnobKind.STATE_COLUMN,
        targets=("household.area",),
        central="area",
        values=("area", "area_alt"),
        method_card_row="MC9",
        note="which area",
    ),
    Knob(
        name="age_factor",
        kind=KnobKind.FACTOR,
        targets=("age",),
        central=1.0,
        values=(),
        method_card_row="MC0",
        note="scale",
    ),
)


def _closure(**change) -> InputClosure:
    fields = {
        "country": "zz",
        "content_basis": "transport",
        "mapping_engine": "axiom:zz",
        "mapping_engine_version": "r1",
        "surface_rulespec_commit": "r1",
        "surface_engine_repository": "TheAxiomFoundation/axiom-rules-engine",
        "surface_engine_commit": "e1",
        "modules": {MODULE: SHA},
        "engine_optional_evidence": "the engine has no optional inputs",
        "undetermined": UndeterminedPolicy(
            action=UndeterminedAction.REFUSE,
            reason="none can arise",
            alternatives=(UndeterminedAction.NOT_ELIGIBLE,),
            new_default=True,
        ),
        "knobs": KNOBS,
        "entries": ENTRIES,
        **change,
    }
    return InputClosure(**fields)


class TestCheck:
    def test_a_complete_closure_holds(self) -> None:
        closure = _closure()
        assert closure.check(MAPPING, SURFACE) == ()
        assert closure.defaults(MODULE, "Family") == {"flag": False}
        assert closure.state_bindings() == (AREA,)
        assert closure.state_bindings(["other.yaml"]) == ()
        assert closure.graph_node_inputs() == {
            InputRef("bridge", "Family", MODULE): "zz.bridge"
        }
        assert [entry.input for entry in closure.new_defaults()] == ["flag"]

    def test_an_unclosed_input_is_reported(self) -> None:
        closure = _closure(entries=ENTRIES[:-1])
        assert closure.check(MAPPING, SURFACE) == (
            f"flag (Family, {MODULE}) is not closed.",
        )

    def test_a_closed_name_the_surface_lacks_is_reported(self) -> None:
        stray = ClosureEntry(
            module=MODULE,
            entity="Person",
            input="flag",
            closure_class=ClosureClass.DEFAULTED,
            value=0,
            reason="r",
            citation="c",
        )
        (problem,) = _closure(entries=(*ENTRIES, stray)).check(MAPPING, SURFACE)
        assert "closed but not a root input" in problem

    def test_concept_encoding_must_be_the_mappings_binding(self) -> None:
        defaulted_age = ClosureEntry(
            module=MODULE,
            entity="Family",
            input="age",
            closure_class=ClosureClass.DEFAULTED,
            value=30,
            reason="r",
            citation="c",
        )
        (problem,) = _closure(
            entries=(defaulted_age, *ENTRIES[1:]), knobs=KNOBS[:1]
        ).check(MAPPING, SURFACE)
        assert "bound by the mapping but closed as defaulted" in problem
        unbound = ConceptMapping(
            **{
                **{
                    name: getattr(MAPPING, name)
                    for name in (
                        "engine",
                        "engine_version",
                        "entity_correspondence",
                        "input_declaration",
                    )
                },
                "bindings": (),
                "unmapped": {item.id: "x" for item in CONCEPTS},
            }
        )
        (problem,) = _closure().check(unbound, SURFACE)
        assert "concept-encoded but not bound" in problem
        awaiting = (
            ClosureEntry(
                module=MODULE,
                entity="Family",
                input="age",
                closure_class=ClosureClass.ENCODED,
                encoded_by=EncodedBy.CONCEPT_BINDING,
                awaiting="a later change",
            ),
            *ENTRIES[1:],
        )
        assert _closure(entries=awaiting).check(unbound, SURFACE) == ()
        (problem,) = _closure(entries=awaiting).check(MAPPING, SURFACE)
        assert "no longer awaits a later change" in problem

    def test_a_bound_input_must_be_closed_even_off_the_surface(self) -> None:
        ghost = InputBinding(
            engine_input="ghost",
            engine_entity="Family",
            concepts=("fact:person.age",),
            transform=Identity(),
            relation=AlignmentRelation.APPROXIMATE,
            note="evidence",
            group_rule=GroupRule.REFERENCE_MEMBER,
            module=MODULE,
            canonical_input="zz:m#input.ghost",
        )
        mapping = ConceptMapping(
            **{
                name: getattr(MAPPING, name)
                for name in (
                    "engine",
                    "engine_version",
                    "entity_correspondence",
                    "input_declaration",
                    "unmapped",
                )
            },
            bindings=(BOUND, ghost),
        )
        (problem,) = _closure().check(mapping, SURFACE)
        assert (
            problem
            == f"ghost (Family, {MODULE}) is bound by the mapping but not closed."
        )

    def test_a_binding_transport_drops_cannot_encode_a_transport_run(self) -> None:
        receipt = InputBinding(
            engine_input="age",
            engine_entity="Family",
            concepts=("fact:person.public_pension_income",),
            transform=Identity(),
            relation=AlignmentRelation.APPROXIMATE,
            note="evidence",
            group_rule=GroupRule.SUM_OVER_MEMBERS,
            module=MODULE,
            canonical_input="zz:m#input.age",
        )
        mapping = ConceptMapping(
            **{
                name: getattr(MAPPING, name)
                for name in (
                    "engine",
                    "engine_version",
                    "entity_correspondence",
                    "input_declaration",
                )
            },
            bindings=(receipt,),
            unmapped={
                item.id: "x"
                for item in CONCEPTS
                if item.id != "fact:person.public_pension_income"
            },
        )
        (problem,) = _closure().check(mapping, SURFACE)
        assert "reads a concept transport frames drop" in problem
        defaulted_age = ClosureEntry(
            module=MODULE,
            entity="Family",
            input="age",
            closure_class=ClosureClass.DEFAULTED,
            value=0,
            reason="the donor's receipt is dropped",
            citation="Act s 2",
        )
        closure = _closure(entries=(defaulted_age, *ENTRIES[1:]), knobs=KNOBS[:1])
        assert closure.check(mapping, SURFACE) == ()
        own = _closure(content_basis="own_data")
        assert own.check(mapping, SURFACE) == ()

    def test_pins_are_checked(self) -> None:
        problems = _closure(
            mapping_engine_version="r0",
            surface_rulespec_commit="r0",
            surface_engine_commit="e0",
            modules={MODULE: "1" * 64},
        ).check(MAPPING, SURFACE)
        assert len(problems) == 4
        (problem,) = _closure(modules={MODULE: SHA, "zz/absent.yaml": SHA}).check(
            MAPPING, SURFACE
        )
        assert "not in the input surface" in problem


class TestConstruction:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"closure_class": "encoded"}, "says what encodes it"),
            (
                {
                    "closure_class": "encoded",
                    "encoded_by": "graph_node",
                    "value": None,
                    "citation": None,
                },
                "names its node",
            ),
            ({"value": None}, "boolean or a number"),
            ({"value": "x"}, "boolean or a number"),
            ({"value": float("nan")}, "finite"),
            ({"citation": None}, "cites"),
            ({"reason": ""}, "needs a reason"),
            ({"input": ""}, "names its module"),
        ],
    )
    def test_malformed_entries_are_refused(self, change, message) -> None:
        fields = {
            "module": MODULE,
            "entity": "Family",
            "input": "flag",
            "closure_class": ClosureClass.DEFAULTED,
            "value": False,
            "reason": "r",
            "citation": "c",
            **change,
        }
        with pytest.raises(ValueError, match=message):
            ClosureEntry(**fields)

    def test_only_defaults_carry_values(self) -> None:
        with pytest.raises(ValueError, match="only a default"):
            ClosureEntry(
                module=MODULE,
                entity="Family",
                input="age",
                closure_class=ClosureClass.ENCODED,
                encoded_by=EncodedBy.CONCEPT_BINDING,
                value=1,
            )

    def test_a_state_entry_carries_its_own_binding(self) -> None:
        with pytest.raises(ValueError, match="feeds another input"):
            ClosureEntry(
                module=MODULE,
                entity="Family",
                input="flag",
                closure_class=ClosureClass.ENCODED,
                encoded_by=EncodedBy.STATE_BINDING,
                state_binding=AREA,
                reason="r",
            )
        with pytest.raises(ValueError, match="carries its binding"):
            ClosureEntry(
                module=MODULE,
                entity="Family",
                input="area_1",
                closure_class=ClosureClass.ENCODED,
                encoded_by=EncodedBy.STATE_BINDING,
                reason="r",
            )

    def test_closures_refuse_duplicates_and_stray_knobs(self) -> None:
        with pytest.raises(ValueError, match="closed twice"):
            _closure(entries=(*ENTRIES, ENTRIES[-1]))
        with pytest.raises(ValueError, match="does not list"):
            _closure(modules={})
        with pytest.raises(ValueError, match="no encoded entry"):
            _closure(
                knobs=(
                    Knob(
                        name="off",
                        kind=KnobKind.SWITCH_ZEROES,
                        targets=("flag",),
                        central=True,
                        values=(),
                        method_card_row="MC0",
                        note="n",
                    ),
                )
            )
        with pytest.raises(ValueError, match="unique"):
            _closure(knobs=(KNOBS[0], KNOBS[0]))

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"central": "elsewhere"}, "takes"),
            ({"values": ()}, "lists the values"),
            ({"targets": ("household.a", "household.b")}, "one state column"),
            ({"kind": "factor", "values": ("x",)}, "no value list"),
            ({"kind": "factor", "values": (), "central": -1.0}, "non-negative"),
            ({"kind": "switch_zeroes", "values": (), "central": 1}, "true or false"),
            (
                {"kind": "allocation", "values": ("evenly",), "central": "evenly"},
                "evenly",
            ),
            ({"note": ""}, "needs a name"),
        ],
    )
    def test_malformed_knobs_are_refused(self, change, message) -> None:
        fields = {**KNOBS[0].to_dict(), **change}
        fields["kind"] = KnobKind(fields["kind"])
        with pytest.raises(ValueError, match=message):
            Knob(**fields)


class TestKnobs:
    def test_central_values_change_nothing(self) -> None:
        knobs = _closure().group_knobs()
        assert knobs.targets() == frozenset()
        assert dict(knobs.state_columns) == {}
        assert dict(knobs.factors) == {}
        assert knobs.zeroed == frozenset()

    def test_scenario_values_translate(self) -> None:
        allocation = Knob(
            name="rent_allocation",
            kind=KnobKind.ALLOCATION,
            targets=("age",),
            central="reference_unit",
            values=("reference_unit", "per_adult_share"),
            method_card_row="MC8",
            note="n",
        )
        switch = Knob(
            name="test_on",
            kind=KnobKind.SWITCH_ZEROES,
            targets=("bridge",),
            central=True,
            values=(),
            method_card_row="MC10",
            note="n",
        )
        closure = _closure(knobs=(*KNOBS, allocation, switch))
        knobs = closure.group_knobs(
            {
                "area_column": "area_alt",
                "age_factor": 2,
                "rent_allocation": "per_adult_share",
                "test_on": False,
            }
        )
        assert dict(knobs.state_columns) == {"household.area": "household.area_alt"}
        assert dict(knobs.factors) == {"age": 2.0}
        assert dict(knobs.allocation) == {"age": Allocation.PER_ADULT_SHARE}
        assert knobs.zeroed == frozenset({"bridge"})
        with pytest.raises(ValueError, match="Unknown knobs"):
            closure.group_knobs({"nope": 1})
        with pytest.raises(ValueError, match="takes"):
            closure.group_knobs({"area_column": "area_x"})


class TestSerialization:
    def test_closures_round_trip_through_json(self) -> None:
        closure = _closure()
        payload = json.loads(json.dumps(closure.to_dict()))
        assert payload["format"] == INPUT_CLOSURE_FORMAT
        again = InputClosure.from_dict(payload)
        assert again.to_dict() == closure.to_dict()
        assert again.check(MAPPING, SURFACE) == ()

    def test_review_fields_beside_the_closure_are_allowed(self) -> None:
        payload = {**_closure().to_dict(), "description": "notes", "status": "draft"}
        payload["input_surface"]["fixture"] = "path/to/surface.json"
        assert InputClosure.from_dict(payload).to_dict() == _closure().to_dict()

    def test_the_surface_engine_is_a_record_naming_its_repository(self) -> None:
        # The engine commit is not a rulespec commit, so it sits in a record
        # that names the engine's repository.
        payload = _closure().to_dict()
        assert payload["input_surface"] == {
            "rulespec_commit": "r1",
            "engine": {
                "repository": "TheAxiomFoundation/axiom-rules-engine",
                "commit": "e1",
            },
        }
        again = InputClosure.from_dict(payload)
        assert again.surface_engine_repository == (
            "TheAxiomFoundation/axiom-rules-engine"
        )
        assert again.surface_engine_commit == "e1"

    @pytest.mark.parametrize(
        "change",
        [
            {"format": "x"},
            {"input_surface": {"rulespec_commit": "r1", "engine_commit": "e1"}},
            {"input_surface": {"rulespec_commit": "r1", "engine": "e1"}},
            {"input_surface": {"rulespec_commit": "r1", "engine": {"commit": "e1"}}},
            {
                "input_surface": {
                    "rulespec_commit": "r1",
                    "engine": {"repository": 7, "commit": "e1"},
                }
            },
            {"rulespec_paths": {"m": 1}},
            {"knobs": {}},
            {"entries": [{"module": "m"}]},
            {"undetermined_judgments": {"action": "guess"}},
            {"mapping": {"engine": "x"}},
            {"engine_optional_evidence": 3},
            {"content_basis": "survey"},
            {"entries": [{**ENTRIES[1].to_dict(), "state_binding": {"module": "x"}}]},
        ],
    )
    def test_malformed_closures_raise_value_error(self, change) -> None:
        payload = {**_closure().to_dict(), **change}
        with pytest.raises(ValueError):
            InputClosure.from_dict(payload)

    @settings(max_examples=200, deadline=None)
    @given(
        data=st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=5),
            lambda inner: (
                st.lists(inner, max_size=3)
                | st.dictionaries(st.text(max_size=6), inner, max_size=4)
            ),
            max_leaves=10,
        )
    )
    def test_arbitrary_payloads_only_raise_value_error(self, data) -> None:
        for reader in (InputClosure.from_dict, ClosureEntry.from_dict, Knob.from_dict):
            with pytest.raises(ValueError):
                reader(data)


class TestJudgments:
    @settings(max_examples=200, deadline=None)
    @given(
        codes=st.lists(st.sampled_from([-1, 0, 1]), min_size=1, max_size=40),
        action=st.sampled_from(
            [UndeterminedAction.NOT_ELIGIBLE, UndeterminedAction.ELIGIBLE]
        ),
    )
    def test_determined_codes_map_exactly(self, codes, action) -> None:
        values = np.asarray(codes, dtype=np.int64)
        eligible = resolve_judgments(values, action)
        assert eligible.dtype == bool
        assert eligible[values == 1].all()
        assert not eligible[values == -1].any()
        assert (eligible[values == 0] == (action is UndeterminedAction.ELIGIBLE)).all()
        lower = resolve_judgments(values, UndeterminedAction.NOT_ELIGIBLE)
        upper = resolve_judgments(values, UndeterminedAction.ELIGIBLE)
        assert (lower <= upper).all()

    def test_no_codes_resolve_to_no_entitlement(self) -> None:
        for action in UndeterminedAction:
            assert resolve_judgments(np.asarray([]), action).tolist() == []

    def test_refuse_fails_on_an_undetermined_code_only(self) -> None:
        codes = np.asarray([1, -1, 1], dtype=np.int8)
        assert resolve_judgments(codes, "refuse").tolist() == [True, False, True]
        with pytest.raises(ValueError, match="1 judgment"):
            resolve_judgments(np.asarray([1, 0]), UndeterminedAction.REFUSE)

    @pytest.mark.parametrize("codes", [[2], [0.5], [True]])
    def test_other_codes_are_refused(self, codes) -> None:
        with pytest.raises(ValueError, match="-1, 0 or 1"):
            resolve_judgments(np.asarray(codes), UndeterminedAction.NOT_ELIGIBLE)


class TestDefaults:
    def test_defaults_become_constant_columns(self) -> None:
        table = pd.DataFrame({"family_id": [1, 2, 3]})
        out = apply_defaults(table, {"flag": False, "count": 0, "amount": 2.5})
        assert out["flag"].dtype == bool and not out["flag"].any()
        assert out["count"].dtype.kind == "i" and (out["count"] == 0).all()
        assert out["amount"].dtype == np.float64
        assert list(table.columns) == ["family_id"]
        with pytest.raises(ValueError, match="already encoded"):
            apply_defaults(out, {"flag": True})
