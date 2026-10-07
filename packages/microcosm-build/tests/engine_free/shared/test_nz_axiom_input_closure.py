"""The New Zealand input closure, against the committed engine surface.

``build/nz/axiom_input_closure.json`` decides how every root input of the four
Axiom modules the NZ v0 graph binds is fed: the main-benefit entitlement and
rates modules, New Zealand Superannuation and the Accommodation Supplement.
These tests hold it to the engine-generated input surface and the ``axiom:nz``
concept mapping, show that encoding plus the closure's defaults supplies every
input a run needs, and replay the Axiom oracle's golden-08 household: the
Accommodation Supplement inputs that unit construction and group encoding
produce from a concept frame equal the inputs the reproduction harness sent
(engine-free), and the real engine turns them into the oracle's weekly amount
(engine-present; skipped without the engine and a rulespec-nz checkout).
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame.adapters.axiom import _batch_from_table, axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import load_axiom_input_surface
from microcosm.frame.concept_mapping import Allocation, GroupRule, Identity
from microcosm.frame.concepts import CONCEPTS, ContentBasis, split_for_transport
from microcosm.frame.input_closure import (
    ClosureClass,
    EncodedBy,
    InputClosure,
    UndeterminedAction,
    apply_defaults,
)
from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    DependentChildRule,
    benefit_unit_membership,
    build_benefit_units,
)
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.paths import paths_for

_BUILD = paths_for("microcosm-build")
_FRAME = paths_for("microcosm-frame")
CLOSURE_PATH = _BUILD.package / "src/microcosm/build/nz/axiom_input_closure.json"
SURFACE_PATH = _FRAME.tests / "fixtures/axiom_input_surfaces/nz.json"
GOLDEN_PATH = (
    _BUILD.tests / "fixtures/nz/axiom_oracle_golden_08_accommodation_supplement.json"
)

ENTITLEMENT = "nz/statutes/social_security/main_benefits/entitlement.yaml"
RATES = "nz/statutes/social_security/main_benefits/rates.yaml"
NZS = "nz/statutes/new_zealand_superannuation/core.yaml"
AS = "nz/statutes/social_security/accommodation_supplement/core.yaml"
PERSON_MODULES = (ENTITLEMENT, RATES, NZS)

CLOSURE = InputClosure.from_dict(json.loads(CLOSURE_PATH.read_text(encoding="utf-8")))
SURFACE = load_axiom_input_surface(SURFACE_PATH)
MAPPING = axiom_concept_mapping("nz")
#: Concept-encoded inputs whose binding lands in a later change. None does now:
#: the cash-asset binding arrived with the liquid-asset concept (work package
#: G3a). An entry that awaits a binding the mapping already has is reported by
#: the closure check until its awaiting note is removed.
AWAITING = {entry.ref for entry in CLOSURE.entries if entry.awaiting is not None}
#: Stand-in unit rule for these tests. The NZ rule is spec data (method card
#: MC7, build/nz/benefit_unit_rule.json, which G2 added); these properties
#: hold for any rule.
RULE = BenefitUnitRule(
    entity="family",
    dependent_child=DependentChildRule(max_age=17, financial_independence=None),
    unparented_child="reference_person_unit",
    split_parents="first_parent_unit",
    provenance={"max_age": "test stand-in, not the NZ rule"},
)
WEEK = 1 / 52


def _payload() -> dict:
    return json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))


class TestClosure:
    def test_the_closure_partitions_every_bound_module(self) -> None:
        assert CLOSURE.content_basis is ContentBasis.TRANSPORT
        assert CLOSURE.check(MAPPING, SURFACE) == ()

    def test_the_cash_assets_are_the_bound_liquid_asset_input(self) -> None:
        # G3a's binding has landed, so no input awaits a later change: the
        # cash-asset input is concept-encoded like any other.
        assert AWAITING == set()
        (entry,) = (
            entry
            for entry in CLOSURE.entries_for(AS)
            if entry.input == "accommodation_supplement_cash_assets"
        )
        assert entry.closure_class is ClosureClass.ENCODED
        assert entry.encoded_by is EncodedBy.CONCEPT_BINDING
        (binding,) = (
            binding for binding in MAPPING.bindings if binding.ref == entry.ref
        )
        assert binding.concepts == ("fact:person.liquid_financial_assets",)
        assert isinstance(binding.transform, Identity)
        assert binding.group_rule is GroupRule.SUM_OVER_MEMBERS

    def test_the_receipt_layer_requirements_are_declared(self) -> None:
        requirements = _payload()["receipt_requirements"]["requirements"]
        assert len(requirements) == 3
        text = " ".join(requirements)
        for name in (
            "main_benefit_receiving_another_main_benefit",
            "main_benefit_receiving_new_zealand_superannuation",
            "jobseeker_income_less_than_zero_rate_cutout",
        ):
            assert name in text
            (entry,) = (
                entry
                for entry in CLOSURE.entries_for(ENTITLEMENT)
                if entry.input == name
            )
            assert entry.closure_class is ClosureClass.DEFAULTED
            assert "receipt_requirements" in entry.reason

    def test_the_four_modules_are_closed_at_their_surface_sizes(self) -> None:
        sizes = {path: len(SURFACE.modules[path].refs()) for path in CLOSURE.modules}
        assert sizes == {ENTITLEMENT: 60, RATES: 28, NZS: 13, AS: 39}
        for path, size in sizes.items():
            assert len(CLOSURE.entries_for(path)) == size

    def test_every_default_is_cited_and_listed_for_review(self) -> None:
        for entry in CLOSURE.entries:
            if entry.closure_class is ClosureClass.DEFAULTED:
                assert entry.new_default, entry.input
                assert entry.citation and entry.citation.endswith(")"), entry.input
                assert len(entry.reason) > 40, entry.input  # a sentence, not a tag
        assert len(CLOSURE.new_defaults()) == sum(
            entry.closure_class is ClosureClass.DEFAULTED for entry in CLOSURE.entries
        )

    def test_no_input_is_claimed_engine_optional(self) -> None:
        # The axiom-rules-engine commit in input_surface.engine (04315d94) has
        # no optional inputs; the closure says why and classes none as
        # engine-optional.
        assert not any(
            entry.closure_class is ClosureClass.ENGINE_OPTIONAL
            for entry in CLOSURE.entries
        )
        assert "input_surface.engine" in CLOSURE.engine_optional_evidence
        assert CLOSURE.surface_engine_commit.startswith("04315d94")
        assert CLOSURE.surface_engine_repository == (
            "TheAxiomFoundation/axiom-rules-engine"
        )
        # The record is the committed surface's own engine record.
        engine = json.loads(SURFACE_PATH.read_text(encoding="utf-8"))["engine"]
        assert _payload()["input_surface"]["engine"] == {
            "repository": engine["repository"],
            "commit": engine["commit"],
        }

    def test_the_engine_record_holds_the_surface_engine_commit(self) -> None:
        # A record that names axiom-rules-engine is foreign to the rulespec
        # commit scan (test_nz_spec_package.py), so the check pins its commit
        # to the surface: any other commit there, a rulespec one included, is
        # reported.
        payload = _payload()
        pin = payload["rulespec_pin"]["commit"]
        payload["input_surface"]["engine"]["commit"] = pin
        assert InputClosure.from_dict(payload).check(MAPPING, SURFACE) == (
            f"Surface engine {SURFACE.engine_commit} is not {pin}.",
        )

    def test_undetermined_judgments_are_refused(self) -> None:
        assert CLOSURE.undetermined.action is UndeterminedAction.REFUSE
        assert CLOSURE.undetermined.new_default

    def test_state_bindings_read_only_receipt_flags_and_the_area(self) -> None:
        receipt = set(_payload()["receipt_flags"]["columns"])
        for binding in CLOSURE.state_bindings():
            assert binding.module == AS
            assert set(binding.columns) <= receipt | {"household.as_area"}
        (non_beneficiary,) = (
            binding
            for binding in CLOSURE.state_bindings()
            if binding.engine_input == "accommodation_supplement_non_beneficiary"
        )
        assert set(non_beneficiary.columns) == receipt
        assert non_beneficiary.negate

    def test_the_bridge_values_come_from_their_nodes(self) -> None:
        assert {
            ref.name: node for ref, node in CLOSURE.graph_node_inputs().items()
        } == {
            "accommodation_supplement_base_rate_weekly_amount": "nz.as.bridge.base_rate",
            "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount": (
                "nz.as.bridge.cutout"
            ),
        }

    def test_central_knobs_are_the_central_encoding(self) -> None:
        assert CLOSURE.group_knobs().targets() == frozenset()
        assert dict(CLOSURE.group_knobs().state_columns) == {}
        knobs = CLOSURE.group_knobs(
            {
                "rent_allocation": "per_adult_share",
                "asset_test": False,
                "area_column": "as_area_alt",
                "rent_factor": 1.1,
            }
        )
        rent = "accommodation_supplement_weekly_rent_paid"
        assert dict(knobs.allocation) == {rent: Allocation.PER_ADULT_SHARE}
        assert knobs.zeroed == frozenset({"accommodation_supplement_cash_assets"})
        assert dict(knobs.factors) == {rent: 1.1}
        assert dict(knobs.state_columns) == {
            "household.as_area": "household.as_area_alt"
        }
        rows = {knob.name: knob.method_card_row for knob in CLOSURE.knobs}
        assert rows == {
            "rent_allocation": "MC8",
            "asset_test": "MC10",
            "area_column": "MC9",
            "rent_factor": "MC8",
        }

    def test_the_closure_resource_is_stable_json(self) -> None:
        text = CLOSURE_PATH.read_text(encoding="utf-8")
        assert text == json.dumps(json.loads(text), indent=1, ensure_ascii=False) + "\n"
        assert InputClosure.from_dict(_payload()).to_dict() == CLOSURE.to_dict()


def _with_state(tables, data) -> dict[str, pd.DataFrame]:
    person = tables["person"].copy()
    household = tables["household"].copy()
    receipt = _payload()["receipt_flags"]["columns"]
    for column in receipt:
        person[column.partition(".")[2]] = np.asarray(
            data.draw(
                st.lists(st.booleans(), min_size=len(person), max_size=len(person))
            ),
            dtype=bool,
        )
    for column in ("as_area", "as_area_alt"):
        household[column] = np.asarray(
            data.draw(
                st.lists(
                    st.integers(1, 4), min_size=len(household), max_size=len(household)
                )
            ),
            dtype=np.int64,
        )
    return {"person": person, "household": household}


def _family_inputs(tables, knobs=None) -> pd.DataFrame:
    person, household = tables["person"], tables["household"]
    family, membership = build_benefit_units(person, household, RULE)
    group = benefit_unit_membership(person, family, membership, RULE)
    encoded = MAPPING.encode_groups(
        tables,
        {"Family": group},
        modules=[AS],
        state_bindings=CLOSURE.state_bindings([AS]),
        knobs=CLOSURE.group_knobs(knobs),
    )
    return apply_defaults(encoded.tables["family"], CLOSURE.defaults(AS, "Family"))


class TestExecutableClosure:
    @settings(max_examples=60, deadline=None)
    @given(
        tables=concept_frames(max_households=4, max_members=6, min_id=-(10**9)),
        data=st.data(),
    )
    def test_encoding_and_defaults_supply_every_supplement_input(
        self, tables, data
    ) -> None:
        tables = _with_state(split_for_transport(tables)[0], data)
        scenarios = [
            {},
            {"rent_allocation": "per_adult_share"},
            {"area_column": "as_area_alt"},
            {"rent_factor": 1.25},
        ]
        if not AWAITING:
            scenarios.append({"asset_test": False})
        expected = {name for name in SURFACE.modules[AS].inputs["Family"]}
        from_nodes = {ref.name for ref in CLOSURE.graph_node_inputs()}
        awaited = {ref.name for ref in AWAITING}
        for knobs in scenarios:
            family = _family_inputs(tables, knobs)
            assert (
                set(family.columns) - {"family_id"} == expected - from_nodes - awaited
            )
            for name in expected - from_nodes - awaited:
                assert family[name].dtype.kind in "bif", name

    @settings(max_examples=40, deadline=None)
    @given(
        tables=concept_frames(max_households=4, max_members=6, min_id=-(10**9)),
        data=st.data(),
    )
    def test_cash_assets_are_each_familys_summed_liquid_assets(
        self, tables, data
    ) -> None:
        tables = _with_state(split_for_transport(tables)[0], data)
        person = tables["person"]
        _, membership = build_benefit_units(person, tables["household"], RULE)
        summed = person["liquid_financial_assets"].groupby(membership).sum()
        cash = "accommodation_supplement_cash_assets"
        family = _family_inputs(tables).set_index("family_id")
        assert np.allclose(
            family[cash].to_numpy(),
            summed.reindex(family.index).to_numpy(),
            rtol=1e-12,
            atol=1e-6,
        )
        # Scenario S1 (method card MC10) switches the asset test off.
        off = _family_inputs(tables, {"asset_test": False})
        assert (off[cash] == 0).all()

    @settings(max_examples=40, deadline=None)
    @given(tables=concept_frames(max_households=4, max_members=6, min_id=-(10**9)))
    def test_encoding_and_defaults_supply_every_person_input(self, tables) -> None:
        # The v0 frame is transported: donor program receipts are dropped, so
        # a binding that reads one never runs and the closure feeds its input.
        transported, dropped = split_for_transport(tables)
        assert "public_pension_income" in dropped["person"]
        encoded = MAPPING.encode(
            transported,
            shares={name: 0.5 for name in MAPPING.share_parameters()},
            take_up_rates={name: None for name in MAPPING.take_up_programs()},
        ).tables["person"]
        for module in PERSON_MODULES:
            names = set(SURFACE.modules[module].inputs["Person"])
            concept_encoded = {
                entry.input
                for entry in CLOSURE.entries_for(module)
                if entry.encoded_by is EncodedBy.CONCEPT_BINDING
            }
            table = apply_defaults(
                encoded.loc[:, sorted(concept_encoded)],
                CLOSURE.defaults(module, "Person"),
            )
            assert set(table.columns) == names, module
            assert all(table[name].dtype.kind in "bif" for name in names), module


# --- The Axiom oracle's golden-08 household ---------------------------------

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _golden_frame() -> dict[str, pd.DataFrame]:
    """Golden-08 as a concept frame: a sole parent and children aged 0, 1, 10.

    The scenario gives no parent age (any adult age gives the same units), a
    private rent of 600 a week in area 1, no income of any kind, and a parent
    the receipt layer has made a Sole Parent Support recipient.
    """

    ages = [30, 0, 1, 10]
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 5, dtype=np.int64),
            "person_household_id": np.ones(4, dtype=np.int64),
            "age": np.asarray(ages, dtype=np.int64),
            "partner_person_id": pd.array([None] * 4, dtype="Int64"),
            "parent_1_person_id": pd.array([None, 1, 1, 1], dtype="Int64"),
            "parent_2_person_id": pd.array([None] * 4, dtype="Int64"),
        }
    )
    for item in CONCEPTS:
        if item.entity == "person" and item.monetary is not None:
            person[item.name] = 0.0
    for column in _payload()["receipt_flags"]["columns"]:
        person[column.partition(".")[2]] = False
    person.loc[0, "receives_sole_parent_support"] = True
    household = pd.DataFrame(
        {
            "household_id": np.ones(1, dtype=np.int64),
            "reference_person_id": pd.array([1], dtype="Int64"),
            "tenure": ["rented_private"],
            "rent": [600.0 * 52],
            "mortgage_interest": [0.0],
            "mortgage_principal": [0.0],
            "property_tax": [0.0],
            "as_area": np.asarray([1], dtype=np.int64),
        }
    )
    return {"person": person, "household": household}


def _golden_inputs() -> pd.DataFrame:
    family = _family_inputs(_golden_frame())
    for ref, _ in CLOSURE.graph_node_inputs().items():
        # The bridge nodes' outputs: golden-08's engine-computed values.
        family[ref.name] = float(GOLDEN["inputs"][ref.name]["value"])
    return family


class TestGoldenEight:
    def test_group_encoding_reproduces_the_harness_inputs(self) -> None:
        family = _golden_inputs()
        assert len(family) == 1
        for name, value in GOLDEN["inputs"].items():
            actual = family[name].iloc[0]
            if value["kind"] == "bool":
                assert family[name].dtype == bool, name
                assert bool(actual) is value["value"], name
            else:
                assert np.isclose(
                    float(actual), float(value["value"]), rtol=0, atol=1e-9
                ), (
                    name,
                    actual,
                    value["value"],
                )
        # Inputs the payment does not read are still supplied (the closure
        # feeds every root input); the harness sent only the 28 it reads.
        extra = set(family.columns) - {"family_id"} - set(GOLDEN["inputs"])
        assert extra == {
            ref.name
            for ref in SURFACE.modules[AS].refs()
            if ref.name not in GOLDEN["inputs"]
        } - {ref.name for ref in AWAITING}


_ENGINE_INSTALLED = importlib.util.find_spec("axiom_rules_engine") is not None
if _ENGINE_INSTALLED:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram

    _DENSE_AVAILABLE = NativeCompiledDenseProgram is not None
else:
    _DENSE_AVAILABLE = False
RULESPEC_NZ = os.environ.get("POPULACE_RULESPEC_NZ")


@pytest.mark.skipif(
    not _DENSE_AVAILABLE or not RULESPEC_NZ,
    reason=(
        "needs axiom_rules_engine with its dense extension and a rulespec-nz "
        "checkout at POPULACE_RULESPEC_NZ"
    ),
)
def test_the_engine_turns_encoded_inputs_into_the_oracle_amount() -> None:
    """Golden-08 through group encoding and the real AS module, within $0.01."""

    import axiom_rules_engine as engine

    root = Path(RULESPEC_NZ)
    program = engine.CompiledDenseProgram.from_file(
        root / AS, rulespec_roots=(root,), entity="Family"
    )
    assert not program.relations
    family = _golden_inputs()
    period = GOLDEN["period"]
    outputs = program.execute(
        period_kind=period["period_kind"],
        start=period["start"],
        end=period["end"],
        inputs=_batch_from_table(family, program.root_inputs),
        outputs=list(GOLDEN["outputs"]),
    )["outputs"]
    for name, expected in GOLDEN["outputs"].items():
        (value,) = np.asarray(outputs[name], dtype=object).ravel()
        assert abs(float(value) - float(expected["value"])) <= 0.01, (name, value)
