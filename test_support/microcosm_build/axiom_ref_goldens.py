"""Engine-free ``axiom_engine_ref`` documents for NZ-shaped and relation rows.

No NZ CI test builds an Axiom reference for NZ's rows: NZ's engine-free graph
injects Python toy engines. These helpers build native references through
``build_transport_registry`` over a synthetic tree that holds NZ's six
binding paths, with NZ's rows, ``unreached_roots`` included, and over the
relation fixtures. The golden ``axiom_engine_refs.json`` holds the documents
generated at the base of S2b PR 1 (main ``4a36edcf``); a reference that moves
fails the byte comparison.

Preparation compiles nothing here: the module checks that would compile a
program (``assert_no_relations`` and ``assert_declared_relations``) and the
output metadata reads are replaced, because a reference needs the tree, not
the engine.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

import microcosm.build.transport.registry as registry_module
from microcosm.build.transport.registry import build_transport_registry
from microcosm.frame import VariableMetadata
from microcosm.frame.adapters.axiom import (
    AxiomEngine,
    AxiomRelation,
    axiom_engine_ref,
)
from microcosm.frame.schema import EntitySchema
from test_support.microcosm_build.transport_population import UNIT_RULE
from test_support.paths import paths_for

NZ_BINDINGS = (
    paths_for("microcosm-build").package
    / "src/microcosm/build/nz/axiom_rules_bindings.json"
)
GOLDEN = paths_for("microcosm-build").tests / "golden/axiom_engine_refs.json"
PINS = {
    "engine": {"commit": "a" * 40, "wheel_sha256": "b" * 64},
    "rulespec": {"commit": "c" * 40},
}
RELATION_ROOT = paths_for("microcosm-frame").tests / "fixtures/rulespec-zz"
RELATION_MODULE = RELATION_ROOT / "zz/policies/tests/axiom_toy_relations.yaml"


def _module_bytes(path: str) -> bytes:
    return f"# synthetic module for the reference golden: {path}\nrules: []\n".encode()


def nz_document(root: Path) -> dict:
    """NZ's six binding rows over a synthetic tree written below ``root``."""
    source = json.loads(NZ_BINDINGS.read_text())
    rows = []
    for row in source["bindings"]:
        payload = _module_bytes(row["rulespec_path"])
        module = root / row["rulespec_path"]
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        row = copy.deepcopy(row)
        row["sha256"] = digest
        if "unreached_roots" in row:
            row["unreached_roots"]["sha256"] = digest
        rows.append(row)
    return {
        **PINS,
        "entity_names": source["entity_names"],
        "periods": source["periods"],
        "bindings": rows,
    }


def relation_row(root: Path) -> dict:
    """The registry relation fixture: one family module with a role selector."""
    payload = b'{"entity": "family"}\n'
    (root / "housing.yaml").write_bytes(payload)
    return {
        "id": "housing",
        "rulespec_path": "housing.yaml",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "entity": "family",
        "engine_entity": "Family",
        "period": "2026-27",
        "variables": ["toy_housing"],
        "relations": [
            {
                "predicate": "family_member",
                "key": "family_member:0:1",
                "current_slot": 0,
                "related_slot": 1,
                "arguments": ["Family", "Person"],
                "root_entity": "family",
                "related_entity": "person",
                "membership_column": "person_family_id",
                "related_inputs": {"member_income": "income"},
                "cardinality": "many",
                "period": "2026-27",
                "role_column": "person_family_role",
                "roles": ["head", "partner"],
            }
        ],
    }


def _adapter_relation_refs() -> dict[str, str]:
    """The adapter-level relation fixture's references (frame relation tests)."""
    relation = AxiomRelation(
        predicate="toy_member",
        key="toy_member:0:1",
        current_slot=0,
        related_slot=1,
        arguments=("TaxUnit", "Person"),
        root_entity="tax_unit",
        related_entity="person",
        membership_column="person_tax_unit_id",
        related_inputs={"toy_income": "income"},
        cardinality="many",
        period="2025",
    )
    pins = dict(
        engine_commit="a" * 40,
        wheel_sha256="b" * 64,
        rulespec_root=RELATION_ROOT,
        rulespec_commit="c" * 40,
    )
    refs = {}
    for name, relations in (
        ("absent", None),
        ("empty", ()),
        ("declared", (relation,)),
    ):
        adapter = AxiomEngine(
            RELATION_MODULE,
            EntitySchema(group_entities=("household", "tax_unit")),
            rulespec_roots=(RELATION_ROOT,),
            entity_names={
                "person": "Person",
                "household": "Household",
                "tax_unit": "TaxUnit",
            },
            nesting={"tax_unit": "household"},
            relations=relations,
        )
        refs[f"adapter_relations_{name}"] = axiom_engine_ref(adapter, **pins)
    return refs


def engine_refs(root: Path) -> dict[str, str]:
    """Every golden reference, built engine-free below ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    nz_root = root / "nz-tree"
    document = nz_document(nz_root)
    relation_root = root / "relation-tree"
    relation_root.mkdir()
    relation_document = {
        **copy.deepcopy(document),
        "bindings": [relation_row(relation_root)],
    }
    entity_by_variable: Mapping[str, str] = {
        variable: row["entity"]
        for row in (*document["bindings"], *relation_document["bindings"])
        for variable in row["variables"]
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry_module, "assert_no_relations", lambda *args: None)
        patch.setattr(
            AxiomEngine, "assert_declared_relations", lambda engine, entity: None
        )
        patch.setattr(
            AxiomEngine,
            "variable_metadata",
            lambda engine, name: VariableMetadata(
                name, entity_by_variable[name], "float", "year"
            ),
        )
        patch.setattr(AxiomEngine, "graph_dtype", lambda engine, name: "float64")
        nz = build_transport_registry(document, nz_root, unit_rule=UNIT_RULE)
        relation = build_transport_registry(
            relation_document, relation_root, unit_rule=UNIT_RULE
        )
    return {
        **{f"nz_{binding}": ref for binding, ref in nz.engine_refs.items()},
        "registry_relation_housing": relation.engine_refs["housing"],
        **_adapter_relation_refs(),
    }
