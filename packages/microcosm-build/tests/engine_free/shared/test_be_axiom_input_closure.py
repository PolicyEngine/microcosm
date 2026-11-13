"""Belgium's input closure, against the committed engine surface.

``build/be/axiom_rules_bindings.json`` binds the twelve rulespec-be
family-benefit modules whose programs run on the engine entity ``Child``, and
``build/be/axiom_input_closure.json`` decides how each of their 277 ``Child``
root inputs is fed: a ``person_role`` binding of the ``axiom:be`` concept
mapping encodes 35, and the other 242 take a default with a reason and a
citation. No frame entity maps to ``Child``, so each binding declares a
binding-local view on it (``docs/transport-declared-units.md``).

Belgium has no composed transport graph yet, so no closure node runs. These
tests call
:func:`~microcosm.build.transport.evidence_contract.closure_partition`, the
closure kernel's own partition code, on the committed surface, the mapping,
the bindings and the closure. They take ``mapped_entities`` from the
bindings' ``entity_names`` as the kernel does, so ``Child`` counts as live
only through each binding's view. ``frame_columns`` stays ``None``: the
frame-slice checks need a producer's slice, which the graph change brings.
"""

import copy
import json
from collections import Counter

import pytest

from microcosm.build.transport.evidence_contract import (
    binding_view_entities,
    closure_partition,
    partition_counts,
)
from microcosm.frame.adapters.axiom import axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import load_axiom_input_surface
from microcosm.frame.concepts import ContentBasis
from microcosm.frame.input_closure import ClosureClass, EncodedBy, InputClosure
from test_support.paths import paths_for

_BUILD = paths_for("microcosm-build")
_FRAME = paths_for("microcosm-frame")
_BE = _BUILD.package / "src/microcosm/build/be"
CLOSURE_PATH = _BE / "axiom_input_closure.json"
BINDINGS_PATH = _BE / "axiom_rules_bindings.json"
GATES_PATH = _BE / "gates.json"
SURFACE_PATH = _FRAME.tests / "fixtures/axiom_input_surfaces/be.json"

LOGICAL_ENTITY = "Child"
MODULES = 12
ROOTS = 277
ENCODED = 35
DEFAULTED = 242
#: Roots, encoded and defaulted per module, in rulespec_path order.
PER_MODULE = {
    "be-bru/statutes/family_benefits/basic_amounts.yaml": (3, 2, 1),
    "be-bru/statutes/family_benefits/eligibility.yaml": (19, 4, 15),
    "be-bru/statutes/family_benefits/selected_amount.yaml": (8, 3, 5),
    "be-bru/statutes/family_benefits/social_supplements.yaml": (5, 1, 4),
    "be-dg/statutes/family_benefits/amounts.yaml": (12, 0, 12),
    "be-dg/statutes/family_benefits/eligibility.yaml": (31, 2, 29),
    "be-vlg/statutes/family_benefits/amounts.yaml": (7, 1, 6),
    "be-vlg/statutes/family_benefits/eligibility.yaml": (23, 2, 21),
    "be-wal/statutes/family_benefits/amounts.yaml": (11, 2, 9),
    "be-wal/statutes/family_benefits/eligibility.yaml": (33, 4, 29),
    "be/statutes/family_benefits/lgaf_amounts.yaml": (18, 2, 16),
    "be/statutes/family_benefits/regional_routing.yaml": (107, 12, 95),
}

SURFACE = load_axiom_input_surface(SURFACE_PATH)
MAPPING = axiom_concept_mapping("be")


def _closure_payload() -> dict:
    return json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))


def _bindings_payload() -> dict:
    return json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))


def _partition(closure_payload: dict, bindings_payload: dict) -> dict:
    """The closure kernel's partition, fed from the surface and the bindings.

    ``roots`` and ``module_sha256`` come from the committed surface, and
    ``module_imports`` from each binding's ``imports_declared``; the kernel
    reads the same three from its prepared engines. ``mapped_entities`` is
    the bindings' ``entity_names`` values, as the kernel passes them.
    """

    rows = bindings_payload["bindings"]
    modules = {row["id"]: SURFACE.modules[row["rulespec_path"]] for row in rows}
    return closure_partition(
        InputClosure.from_dict(closure_payload),
        MAPPING,
        rows,
        {binding: dict(module.inputs) for binding, module in modules.items()},
        module_sha256={binding: module.sha256 for binding, module in modules.items()},
        module_imports={row["id"]: row["imports_declared"] for row in rows},
        mapped_entities=bindings_payload["entity_names"].values(),
    )


@pytest.fixture(scope="module")
def partition() -> dict:
    return _partition(_closure_payload(), _bindings_payload())


class TestPartition:
    def test_the_bindings_cover_the_twelve_child_modules(self) -> None:
        rows = _bindings_payload()["bindings"]
        assert len(rows) == MODULES
        assert len({row["id"] for row in rows}) == MODULES
        assert {row["rulespec_path"] for row in rows} == set(PER_MODULE)
        # The surface holds no other module with a root on Child.
        assert {
            path
            for path, module in SURFACE.modules.items()
            if LOGICAL_ENTITY in module.inputs
        } == set(PER_MODULE)

    def test_every_root_is_encoded_or_defaulted(self, partition: dict) -> None:
        rows = partition["modules"]
        assert len(rows) == MODULES
        totals: Counter[str] = Counter()
        for row in rows:
            counts = partition_counts(row)
            assert sum(counts.values()) == len(row["roots"])
            assert (
                len(row["roots"]),
                counts["encoded"],
                counts["defaulted"],
            ) == PER_MODULE[row["rulespec_path"]]
            assert row["roots"] == sorted(
                f"{LOGICAL_ENTITY}.{name}"
                for name in SURFACE.modules[row["rulespec_path"]].inputs[LOGICAL_ENTITY]
            )
            assert row["encoded"][EncodedBy.CONCEPT_BINDING.value] == sorted(
                name for names in row["encoded"].values() for name in names
            )
            assert row["sha256"] == row["reviewed_sha256"]
            for name in ("gap", "unexpected", "unmapped_entity"):
                assert row[name] == [], (row["binding"], name)
            for name in ("engine_optional", "knob_targeted", "relation", "unreached"):
                assert row[name] == [], (row["binding"], name)
            totals.update(counts)
            totals["roots"] += len(row["roots"])
        assert totals["roots"] == ROOTS
        assert totals["encoded"] == ENCODED
        assert totals["defaulted"] == DEFAULTED

    def test_the_closure_check_is_empty(self, partition: dict) -> None:
        assert partition["check"] == []

    def test_child_is_live_only_through_each_view(self) -> None:
        # Without the views, Child sits outside mapped_entities, and the
        # partition reports every root unmapped instead of checking it.
        bindings = _bindings_payload()
        assert LOGICAL_ENTITY not in bindings["entity_names"].values()
        for row in bindings["bindings"]:
            del row["views"]
        rows = _partition(_closure_payload(), bindings)["modules"]
        assert sum(len(row["unmapped_entity"]) for row in rows) == ROOTS
        assert all(row["roots"] == [] for row in rows)

    def test_dropping_any_default_leaves_a_gap_that_names_it(self) -> None:
        payload = _closure_payload()
        bindings = _bindings_payload()
        by_path = {row["rulespec_path"]: row["id"] for row in bindings["bindings"]}
        defaults = [
            index
            for index, entry in enumerate(payload["entries"])
            if entry["class"] == ClosureClass.DEFAULTED.value
        ]
        assert len(defaults) == DEFAULTED
        for index in defaults:
            mutant = copy.deepcopy(payload)
            dropped = mutant["entries"].pop(index)
            label = f"{dropped['entity']}.{dropped['input']}"
            result = _partition(mutant, bindings)
            gaps = {
                row["binding"]: row["gap"] for row in result["modules"] if row["gap"]
            }
            assert gaps == {by_path[dropped["rulespec_path"]]: [label]}
            assert any(
                dropped["input"] in problem and "not closed" in problem
                for problem in result["check"]
            )

    def test_an_entry_for_no_root_is_unexpected(self) -> None:
        payload = _closure_payload()
        extra = copy.deepcopy(
            next(
                entry
                for entry in payload["entries"]
                if entry["class"] == ClosureClass.DEFAULTED.value
            )
        )
        extra["input"] = "belgium_family_benefits_no_such_root"
        payload["entries"].append(extra)
        result = _partition(payload, _bindings_payload())
        unexpected = [
            row["unexpected"] for row in result["modules"] if row["unexpected"]
        ]
        assert unexpected == [[f"{LOGICAL_ENTITY}.{extra['input']}"]]

    def test_defaulting_an_encoded_root_fails_the_check(self) -> None:
        payload = _closure_payload()
        entry = next(
            entry
            for entry in payload["entries"]
            if entry["class"] == ClosureClass.ENCODED.value
        )
        del entry["encoded_by"]
        entry.update(
            {
                "class": ClosureClass.DEFAULTED.value,
                "value": 0,
                "reason": "A mutant.",
                "citation": "None.",
            }
        )
        result = _partition(payload, _bindings_payload())
        assert any(
            entry["input"] in problem and "bound by the mapping" in problem
            for problem in result["check"]
        )


class TestClosure:
    def test_it_closes_child_roots_only_and_each_once(self) -> None:
        closure = InputClosure.from_dict(_closure_payload())
        assert closure.country == "be"
        assert closure.content_basis is ContentBasis.TRANSPORT
        assert {entry.entity for entry in closure.entries} == {LOGICAL_ENTITY}
        assert len(closure.entries) == ROOTS
        assert set(closure.modules) == set(PER_MODULE)
        assert closure.modules == {
            path: SURFACE.modules[path].sha256 for path in PER_MODULE
        }
        assert closure.knobs == ()

    def test_it_was_reviewed_with_the_committed_mapping_and_surface(self) -> None:
        closure = InputClosure.from_dict(_closure_payload())
        assert closure.mapping_engine == MAPPING.engine
        assert closure.mapping_engine_version == MAPPING.engine_version
        assert closure.surface_rulespec_commit == SURFACE.rulespec_commit
        assert closure.surface_engine_commit == SURFACE.engine_commit

    def test_every_default_is_a_cited_proposal(self) -> None:
        payload = _closure_payload()
        closure = InputClosure.from_dict(payload)
        defaults = [
            entry
            for entry in closure.entries
            if entry.closure_class is ClosureClass.DEFAULTED
        ]
        assert len(defaults) == DEFAULTED
        assert closure.new_defaults() == tuple(defaults)
        assert all(entry.reason and entry.citation for entry in defaults)
        assert "proposal" in payload["status"]
        assert "new_default" in payload["status"]

    def test_the_mapping_encodes_the_rest_by_person_role(self) -> None:
        closure = InputClosure.from_dict(_closure_payload())
        encoded = {
            entry.ref
            for entry in closure.entries
            if entry.closure_class is ClosureClass.ENCODED
        }
        assert all(
            entry.encoded_by is EncodedBy.CONCEPT_BINDING
            for entry in closure.entries
            if entry.closure_class is ClosureClass.ENCODED
        )
        bound = {
            binding.ref
            for binding in MAPPING.bindings
            if binding.ref.entity == LOGICAL_ENTITY
        }
        assert encoded == bound
        assert len(encoded) == ENCODED

    def test_a_name_keeps_one_default_across_modules(self) -> None:
        # The routing module reads the four regional schemes' roots through
        # its imports, so one name must not take two values.
        closure = InputClosure.from_dict(_closure_payload())
        values: dict[str, set] = {}
        for entry in closure.entries:
            if entry.closure_class is ClosureClass.DEFAULTED:
                values.setdefault(entry.input, set()).add(
                    (type(entry.value).__name__, entry.value)
                )
        assert {name: found for name, found in values.items() if len(found) > 1} == {}

    def test_engine_line_citations_are_pinned_at_the_surface_engine(self) -> None:
        payload = _closure_payload()
        engine = payload["input_surface"]["engine"]
        assert engine["commit"] == SURFACE.engine_commit
        pins = payload["engine_source_pins"]["pins"]
        assert all(
            pin["repository"] == engine["repository"]
            and pin["commit"] == engine["commit"]
            for pin in pins
        )
        evidence = payload["engine_optional_evidence"]
        assert {pin["path"] for pin in pins} == {
            path for path in (pin["path"] for pin in pins) if f"{path}:" in evidence
        }


class TestBindings:
    def test_each_binding_pins_its_module_as_the_surface_does(self) -> None:
        payload = _bindings_payload()
        assert payload["country"] == "be"
        assert payload["entity_names"] == MAPPING.entity_correspondence
        for row in payload["bindings"]:
            module = SURFACE.modules[row["rulespec_path"]]
            assert module.status == "compiled"
            assert row["sha256"] == module.sha256
            assert row["root_input_count"] == sum(
                len(names) for names in module.inputs.values()
            )
            assert row["entity"] == "person"
            assert row["engine_entity"] == payload["entity_names"]["person"]
            assert row["period"] in payload["periods"]
            assert row["relations_declared"] is False
            assert isinstance(row["imports_declared"], bool)
            assert "relations" not in row

    def test_each_binding_has_one_child_view_over_its_variables(self) -> None:
        payload = _bindings_payload()
        structural = {"person_id", "household_id"}
        for row in payload["bindings"]:
            assert binding_view_entities(row) == {LOGICAL_ENTITY}
            (view,) = row["views"]
            assert set(view) == {"logical_entity", "selector", "outside"}
            assert set(view["selector"]) == {"column"}
            selector = view["selector"]["column"]
            variables = row["variables"]
            assert variables and len(variables) == len(set(variables))
            assert set(view["outside"]) == set(variables)
            assert selector not in variables
            assert selector not in structural
            # A view input reads the person column of the same name, so a
            # selector must not shadow a root input either.
            module = SURFACE.modules[row["rulespec_path"]]
            assert selector not in module.inputs[LOGICAL_ENTITY]

    def test_the_engine_pins_await_the_surface_refresh(self) -> None:
        engine = _bindings_payload()["engine"]
        assert engine["commit"] is None
        assert engine["wheel_sha256"] is None
        assert engine["status"]

    def test_the_rulespec_pin_holds_the_surface_bytes(self) -> None:
        # The surface was generated at an ancestor of the pin; the bindings
        # record that every bound module has the same bytes at both.
        payload = _bindings_payload()
        assert payload["rulespec"]["commit"] != SURFACE.rulespec_commit
        assert SURFACE.rulespec_commit in payload["rulespec"]["verification"]
        closure = _closure_payload()
        assert closure["rulespec_pin"]["commit"] == payload["rulespec"]["commit"]


class TestGate:
    def test_release_input_coverage_never_lets_an_engine_default_close(self) -> None:
        gates = json.loads(GATES_PATH.read_text(encoding="utf-8"))["gates"]
        (gate,) = (gate for gate in gates if gate["gate"] == "release_input_coverage")
        assert gate["id"] == "release_input_coverage"
        assert gate["criticality"] == "release_blocking"
        assert gate["parameters"] == {"allow_engine_optional": False}
        assert gate["evidence_absent_blocks"] is True
